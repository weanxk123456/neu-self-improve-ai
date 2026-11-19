| Method (Qwen2.5-3B-Instruct) | Paper EM | Ours EM (LoRA) |
|---|---|---|
| Direct Inference | 0.024 | 0.024 |
| Search-R1-Instruct (PPO) | 0.264 | 0.144 @ step 50 |
| Search-R1-Instruct (GRPO) | 0.232 | 0.300 ± 0.004 (n=2) @ step 50 |

PPO: 2 runs found in the log; final EMs of complete runs (step 50): 0.144
GRPO: 3 runs found in the log; final EMs of complete runs (step 50): 0.304, 0.296
