# Experiment results

Mode: **llm**. No policy training.
Demo results validate plumbing only; they are not LLM generation evidence.

| Level | Structural score | Rule success | Rule collision | Random success |
|---|---:|---:|---:|---:|
| 0 | 0.439 | 100% | 0% | 20% |

20 held-out seeds per policy/level; rates have substantial sampling uncertainty.
Structural score is hand-defined and does not establish monotonically increasing difficulty.
Rule controller reads simulator state. Acceptance only witnesses some successful design-seed rollouts.
Neither smoke checks nor these rollouts prove all instances feasible.