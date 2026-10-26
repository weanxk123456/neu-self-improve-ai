import numpy as np, torch, torch.nn as nn, gymnasium as gym
import matplotlib.pyplot as plt
from torch.distributions import Categorical, Normal

torch.set_num_threads(1)

class OneHot(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.n = env.observation_space.n
        self.observation_space = gym.spaces.Box(0, 1, (self.n,), np.float32)
    def observation(self, o):
        v = np.zeros(self.n, np.float32); v[o] = 1.0
        return v

def make_env(name, max_steps):
    env = gym.make(name, max_episode_steps=max_steps)
    if isinstance(env.observation_space, gym.spaces.Discrete):
        env = OneHot(env)
    return env

CONFIGS = {
    "CartPole-v1":     dict(episodes=500,  hidden=(64, 64),   lr_pi=1e-3, lr_v=1e-3, gamma=0.99, reward_scale=0.01, max_steps=500),
    "Acrobot-v1":      dict(episodes=500,  hidden=(64, 64),   lr_pi=1e-3, lr_v=5e-3, gamma=0.99, reward_scale=0.1,  max_steps=500),
    "LunarLander-v3":  dict(episodes=1000, hidden=(128, 128), lr_pi=5e-4, lr_v=1e-3, gamma=0.99, reward_scale=0.01, max_steps=1000),
    "Pendulum-v1":     dict(episodes=1500, hidden=(64, 64),   lr_pi=1e-3, lr_v=1e-3, gamma=0.99, reward_scale=0.1,  max_steps=200),
    "CliffWalking-v1": dict(episodes=200,  hidden=(32,),      lr_pi=3e-3, lr_v=3e-3, gamma=0.99, reward_scale=0.05, max_steps=200),
}
N_SEEDS = 5

def mlp(inp, hidden, out):
    layers, d = [], inp
    for h in hidden:
        layers += [nn.Linear(d, h), nn.Tanh()]; d = h
    layers.append(nn.Linear(d, out))
    return nn.Sequential(*layers)

class Policy(nn.Module):
    def __init__(self, env, hidden):
        super().__init__()
        obs_dim = env.observation_space.shape[0]
        self.discrete = isinstance(env.action_space, gym.spaces.Discrete)
        if self.discrete:
            self.net = mlp(obs_dim, hidden, env.action_space.n)
        else:
            self.net = mlp(obs_dim, hidden, env.action_space.shape[0])
            self.log_std = nn.Parameter(torch.full((env.action_space.shape[0],), -0.5))
            self.low = torch.as_tensor(env.action_space.low)
            self.high = torch.as_tensor(env.action_space.high)
    def dist(self, s):
        out = self.net(s)
        if self.discrete:
            return Categorical(logits=out)
        return Normal(out, self.log_std.exp())
    def log_prob(self, d, a):
        lp = d.log_prob(a)
        return lp if self.discrete else lp.sum(-1)
    def to_env(self, a):
        if self.discrete:
            return int(a.item())
        return torch.clamp(a, self.low, self.high).numpy()

def make_value(env, hidden):
    return mlp(env.observation_space.shape[0], hidden, 1)

def rollout(env, pi):
    obs, _ = env.reset()
    S, A, R, done = [], [], [], False
    while not done:
        s = torch.as_tensor(obs, dtype=torch.float32)
        with torch.no_grad():
            a = pi.dist(s).sample()
        obs, r, term, trunc, _ = env.step(pi.to_env(a))
        S.append(s); A.append(a); R.append(r)
        done = term or trunc
    return torch.stack(S), torch.stack(A), R

def returns_to_go(R, gamma, scale):
    G, out = 0.0, []
    for r in reversed(R):
        G = r * scale + gamma * G
        out.append(G)
    return torch.tensor(out[::-1], dtype=torch.float32)

def train_episodic(env, cfg, use_baseline, seed):
    pi = Policy(env, cfg["hidden"])
    opt_pi = torch.optim.Adam(pi.parameters(), lr=cfg["lr_pi"])
    if use_baseline:
        V = make_value(env, cfg["hidden"])
        opt_v = torch.optim.Adam(V.parameters(), lr=cfg["lr_v"])
    scores = []
    for ep in range(cfg["episodes"]):
        S, A, R = rollout(env, pi)
        scores.append(sum(R))
        G = returns_to_go(R, cfg["gamma"], cfg["reward_scale"])
        disc = cfg["gamma"] ** torch.arange(len(R), dtype=torch.float32)
        if use_baseline:
            v = V(S).squeeze(-1)
            delta = (G - v).detach()
            loss_v = ((G - v) ** 2).mean()
            opt_v.zero_grad(); loss_v.backward(); opt_v.step()
        else:
            delta = G
        logp = pi.log_prob(pi.dist(S), A)
        loss_pi = -(disc * delta * logp).mean()
        opt_pi.zero_grad(); loss_pi.backward(); opt_pi.step()
    return scores

def train_actor_critic(env, cfg, seed):
    pi = Policy(env, cfg["hidden"])
    V = make_value(env, cfg["hidden"])
    opt_pi = torch.optim.Adam(pi.parameters(), lr=cfg["lr_pi"])
    opt_v = torch.optim.Adam(V.parameters(), lr=cfg["lr_v"])
    gamma, scale = cfg["gamma"], cfg["reward_scale"]
    scores = []
    for ep in range(cfg["episodes"]):
        obs, _ = env.reset()
        I, total, done = 1.0, 0.0, False
        while not done:
            s = torch.as_tensor(obs, dtype=torch.float32)
            d = pi.dist(s)
            a = d.sample()
            logp = pi.log_prob(d, a)
            obs, r, term, trunc, _ = env.step(pi.to_env(a))
            total += r
            s2 = torch.as_tensor(obs, dtype=torch.float32)
            with torch.no_grad():
                v_next = torch.zeros(1) if term else V(s2)
            target = r * scale + gamma * v_next
            v = V(s)
            delta = (target - v).detach()
            loss_v = ((target - v) ** 2).mean()
            loss_pi = -(I * delta * logp).sum()
            opt_v.zero_grad(); loss_v.backward(); opt_v.step()
            opt_pi.zero_grad(); loss_pi.backward(); opt_pi.step()
            I *= gamma
            done = term or trunc
        scores.append(total)
    return scores

ALGOS = {
    "REINFORCE":               lambda env, cfg, seed: train_episodic(env, cfg, False, seed),
    "REINFORCE with baseline": lambda env, cfg, seed: train_episodic(env, cfg, True, seed),
    "Actor-Critic":            lambda env, cfg, seed: train_actor_critic(env, cfg, seed),
}


def run_one(env_name, algo_name, seed):
    cfg = CONFIGS[env_name]
    torch.manual_seed(seed); np.random.seed(seed)
    env = make_env(env_name, cfg["max_steps"])
    env.reset(seed=seed); env.action_space.seed(seed)
    return ALGOS[algo_name](env, cfg, seed)

def run_env(env_name, n_seeds=N_SEEDS):
    results = {}
    for algo in ALGOS:
        runs = [run_one(env_name, algo, s) for s in range(n_seeds)]
        results[algo] = np.array(runs)
        print(f"{env_name} | {algo:25s} | Average return over the last 50 rounds: {results[algo][:, -50:].mean():.1f}")
    return results

def smooth(x, k=10):
    return np.convolve(x, np.ones(k) / k, mode="valid")

def plot_env(env_name, results):
    colors = {"REINFORCE": "tab:red", "REINFORCE with baseline": "tab:green", "Actor-Critic": "tab:blue"}
    plt.figure(figsize=(7, 4.5))
    for algo, arr in results.items():
        m, sd = arr.mean(0), arr.std(0)
        ms, sds = smooth(m), smooth(sd)
        x = np.arange(1, len(ms) + 1)
        plt.plot(x, ms, label=algo, color=colors[algo])
        plt.fill_between(x, ms - sds, ms + sds, color=colors[algo], alpha=0.15)
    plt.xlabel("Episode"); plt.ylabel(f"Total reward on episode (G0)\naveraged over {arr.shape[0]} runs")
    plt.title(env_name); plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
    plt.savefig(f"{env_name}.png", dpi=150); plt.show()


ALL = {}

for name in CONFIGS:
    ALL[name] = run_env(name)
    plot_env(name, ALL[name])
