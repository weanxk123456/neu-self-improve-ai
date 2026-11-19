"""Results table (vs. paper) and training curves for PPO vs. GRPO.

If several runs were appended to the same log file (step counter restarts),
the log is split into separate runs. The table reports the final Bamboogle EM
of every complete run, and mean ± SD when there is more than one.
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

PAPER = {  # Search-R1 (arXiv v4) Table 2 / Table 3, Qwen2.5-3B-Instruct, Bamboogle EM
    "Direct Inference": 0.024,
    "Search-R1-Instruct (PPO)": 0.264,
    "Search-R1-Instruct (GRPO)": 0.232,
}
RUNS = {"PPO": Path("runs/ppo"), "GRPO": Path("runs/grpo")}
COLORS = {"PPO": "tab:purple", "GRPO": "tab:green"}
OUT = Path("results"); OUT.mkdir(exist_ok=True)


def read_jsonl(p):
    return [json.loads(l) for l in open(p) if l.strip()] if p.exists() else []


def split_runs(rows):
    """Start a new run whenever the step counter does not increase."""
    runs, cur = [], []
    for r in rows:
        if cur and r["step"] <= cur[-1]["step"]:
            runs.append(cur); cur = []
        cur.append(r)
    if cur:
        runs.append(cur)
    return runs


def smooth(x, k=10):
    x = np.asarray(x, float)
    return np.convolve(x, np.ones(k) / k, mode="valid") if len(x) >= k else x


ours, detail = {}, []
direct = OUT / "direct_inference.json"
if direct.exists():
    ours["Direct Inference"] = json.loads(direct.read_text())["em"]

curves = {}
for name, d in RUNS.items():
    tr_runs, ev_runs = split_runs(read_jsonl(d / "train_log.jsonl")), split_runs(read_jsonl(d / "eval_log.jsonl"))
    curves[name] = (tr_runs, ev_runs)
    if not ev_runs:
        continue
    final_step = max(e[-1]["step"] for e in ev_runs)
    complete = [e for e in ev_runs if e[-1]["step"] == final_step]
    finals = [e[-1]["bamboogle_em"] for e in complete]
    key = f"Search-R1-Instruct ({name})"
    ours[key] = (np.mean(finals), np.std(finals), len(finals), final_step)
    if len(ev_runs) > 1:
        detail.append(f"{name}: {len(ev_runs)} runs found in the log; final EMs of complete runs "
                      f"(step {final_step}): {', '.join(f'{x:.3f}' for x in finals)}")

lines = ["| Method (Qwen2.5-3B-Instruct) | Paper EM | Ours EM (LoRA) |", "|---|---|---|"]
for k, v in PAPER.items():
    o = ours.get(k)
    if o is None:
        s = "-"
    elif isinstance(o, float):
        s = f"{o:.3f}"
    else:
        m, sd, n, st = o
        s = f"{m:.3f}" + (f" ± {sd:.3f} (n={n})" if n > 1 else "") + f" @ step {st}"
    lines.append(f"| {k} | {v:.3f} | {s} |")
table = "\n".join(lines + [""] + detail)
(OUT / "summary.md").write_text(table + "\n")
print(table)

fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
for name, (tr_runs, ev_runs) in curves.items():
    c = COLORS[name]
    for i, tr in enumerate(tr_runs):
        lab = name if len(tr_runs) == 1 else f"{name} run {i + 1}"
        alpha = 1.0 if i == len(tr_runs) - 1 else 0.35
        steps = np.array([x["step"] for x in tr])
        for ax, key in ((axes[0], "reward"), (axes[1], "avg_search")):
            y = smooth([x[key] for x in tr])
            ax.plot(steps[len(steps) - len(y):], y, color=c, alpha=alpha, label=lab)
    for i, ev in enumerate(ev_runs):
        lab = name if len(ev_runs) == 1 else f"{name} run {i + 1}"
        alpha = 1.0 if i == len(ev_runs) - 1 else 0.35
        axes[2].plot([x["step"] for x in ev], [x["bamboogle_em"] for x in ev], "o-", color=c, alpha=alpha, label=lab)
for name, v in PAPER.items():
    if "PPO" in name or "GRPO" in name:
        algo = "PPO" if "PPO" in name else "GRPO"
        axes[2].axhline(v, ls="--", color=COLORS[algo], alpha=0.5, label=f"paper {algo}")
axes[2].axhline(PAPER["Direct Inference"], ls=":", color="gray", label="paper Direct Inference")
for ax, t in zip(axes, ["Train reward (EM, smoothed)", "Valid searches / rollout", "Bamboogle EM (greedy)"]):
    ax.set_title(t); ax.set_xlabel("RL step"); ax.grid(alpha=0.3); ax.legend(fontsize=8)
plt.tight_layout(); plt.savefig(OUT / "training_curves.png", dpi=150)
print("saved", OUT / "training_curves.png")
