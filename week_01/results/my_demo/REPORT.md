# Experiment results

Mode: **demo**. No policy training.
Demo results validate plumbing only; they are not LLM generation evidence.

| Level | Structural score | Rule success | Rule collision | Random success |
|---|---:|---:|---:|---:|
| 0 | 0.214 | 100% | 0% | 15% |
| 1 | 0.245 | 100% | 0% | 30% |
| 2 | 0.275 | 100% | 0% | 30% |
| 3 | 0.329 | 100% | 0% | 30% |
| 4 | 0.360 | 100% | 0% | 30% |

20 held-out seeds per policy/level; rates have substantial sampling uncertainty.
Structural score is hand-defined and does not establish monotonically increasing difficulty.
Rule controller reads simulator state. Acceptance only witnesses some successful design-seed rollouts.
Neither smoke checks nor these rollouts prove all instances feasible.