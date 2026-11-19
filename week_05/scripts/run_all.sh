#!/usr/bin/env bash
# Full pipeline (retriever must already be running on :8000).
set -euo pipefail
cd "$(dirname "$0")/.."

python eval.py --mode direct --out results/direct_inference.json      # Table 2, Direct Inference row
python train.py --algo grpo --out runs/grpo                             # Search-R1-Instruct (GRPO) + LoRA
python train.py --algo ppo  --out runs/ppo                              # Search-R1-Instruct (PPO)  + LoRA
python plot_results.py
