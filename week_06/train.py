"""Sequential online Flow-GRPO: one question group/update, planner LoRA only."""
import json
import random
from pathlib import Path
from core import Agent, advantages, load_data, normalize, write_json
from judge import Judge
from tools import Tools


def train(args):
    import torch
    from model import LocalModel
    if args.group < 2 or args.temperature <= 0:
        raise ValueError('Training requires --group >= 2 and --temperature > 0')
    if args.steps < 1 or args.lr <= 0 or args.beta < 0 or not 0 < args.clip < 1:
        raise ValueError('Invalid training hyperparameters')
    rows = load_data(args.data)
    # Check the full supplied held-out set, not an evaluation limit.
    heldout = load_data(args.heldout)
    forbidden = {normalize(x['question']) for x in heldout}
    if any(normalize(x['question']) in forbidden for x in rows):
        raise ValueError('Training/evaluation question overlap detected')
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    write_json(out / 'config.json', vars(args))
    model = LocalModel(args.model, args.context, args.max_new, args.adapter,
                       train=True, four_bit=args.four_bit, rank=args.rank, alpha=args.alpha)
    tools = Tools(args.search, args.cache + '/tools')
    agent = Agent(model, tools, args.turns)
    judge = Judge(args.judge, args.judge_model, args.cache + '/judge')
    params = [p for p in model.model.parameters() if p.requires_grad]
    if not params or any('lora_' not in name for name, p in model.model.named_parameters() if p.requires_grad):
        raise RuntimeError('Expected only LoRA parameters to be trainable')
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0)
    rng = random.Random(args.seed)
    order = list(range(len(rows)))
    rng.shuffle(order)
    updates = 0
    for step in range(args.steps):
        if step and step % len(order) == 0:
            rng.shuffle(order)
        row = rows[order[step % len(order)]]
        trajectories, rewards = [], []
        for rollout in range(args.group):
            print(f'step={step+1} rollout={rollout+1}/{args.group}', flush=True)
            trajectory = agent.run(row['question'], planner_adapter=True, temperature=args.temperature)
            reward = judge.score(row['question'], trajectory.answer, row['answers'])
            trajectories.append(trajectory)
            rewards.append(reward)
        adv = advantages(rewards)
        write_json(out / 'trajectories' / f'{step+1:05d}.json', {
            'id': row['id'], 'answers': row['answers'], 'rewards': rewards,
            'advantages': adv, 'trajectories': [t.to_dict() for t in trajectories]})
        # Freeze sampled-policy log probabilities BEFORE any optimizer step.
        # Reference = initial frozen base with adapters disabled, no extra model copy.
        replay = []
        varied = max(rewards) != min(rewards)
        if varied:
            with torch.no_grad():
                for trajectory, advantage in zip(trajectories, adv):
                    for d in trajectory.decisions:
                        old = model.log_probs(d, True, args.temperature).detach().cpu()
                        ref = model.log_probs(d, False, args.temperature).detach().cpu()
                        # Equal weighting per trajectory, then per turn, then per token.
                        weight = 1.0 / (len(trajectories) * len(trajectory.decisions))
                        replay.append((d, advantage, old, ref, weight))
        optimizer.zero_grad(set_to_none=True)
        total_loss, total_kl = 0.0, 0.0
        # One on-policy update/group. Separate backward calls bound peak activation memory.
        for d, advantage, old, ref, weight in replay:
            new = model.log_probs(d, True, args.temperature)
            old, ref = old.to(new.device), ref.to(new.device)
            ratio = torch.exp(new - old)
            unclipped = ratio * advantage
            clipped = ratio.clamp(1 - args.clip, 1 + args.clip) * advantage
            # GRPO's sampled nonnegative KL estimator against the frozen base policy.
            delta = ref - new
            kl = torch.exp(delta) - delta - 1
            loss = weight * (-torch.minimum(unclipped, clipped) + args.beta * kl).mean()
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite loss; lower LR/length, inspect saved trajectory')
            loss.backward()
            total_loss += float(loss.detach())
            total_kl += weight * float(kl.detach().mean())
        grad_norm = 0.0
        if replay:
            grad_norm = float(torch.nn.utils.clip_grad_norm_(params, 1.0, error_if_nonfinite=True))
            optimizer.step()
            updates += 1
        record = {'step': step+1, 'question_id': row['id'], 'mean_reward': sum(rewards)/len(rewards),
                  'rewards': rewards, 'loss': total_loss, 'kl': total_kl, 'grad_norm': grad_norm,
                  'updated': bool(replay), 'optimizer_updates': updates,
                  'mean_turns': sum(len(t.decisions) for t in trajectories)/len(trajectories)}
        with (out / 'train.jsonl').open('a', encoding='utf-8') as f:
            f.write(json.dumps(record) + '\n')
        print(json.dumps(record), flush=True)
        if (step+1) % args.save_every == 0:
            model.save(out / f'adapter_step_{step+1}')
    model.save(out / 'adapter_final')
    write_json(out / 'status.json', {'steps': args.steps, 'optimizer_updates': updates,
        'warning': 'No policy updates: all groups had identical rewards' if not updates else None})
    if not updates:
        print('WARNING: no learning signal. Increase group size/data diversity; inspect trajectories. '
              'The saved adapter is unchanged, not evidence of successful training.', flush=True)
