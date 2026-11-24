#!/usr/bin/env bash
# Run from an allocated GPU node. Defaults need no paid API keys.
set -euo pipefail
cd "$(dirname "$0")"
OUT="${OUT:-runs/full_$(date +%Y%m%d_%H%M%S)}"
SEARCH="${SEARCH:-wiki}"
JUDGE="${JUDGE:-em}"
STEPS="${STEPS:-100}"
if [[ ! -f data/bamboogle.jsonl || ! -f data/train.jsonl ]]; then
  echo 'Prepare both datasets first: python prepare.py --train-limit 2000' >&2
  exit 1
fi
python run.py evaluate --mode direct --data data/bamboogle.jsonl --judge "$JUDGE" --out "$OUT/direct"
python run.py evaluate --mode agentflow --data data/bamboogle.jsonl --search "$SEARCH" --judge "$JUDGE" --out "$OUT/agentflow"
python run.py train --data data/train.jsonl --heldout data/bamboogle.jsonl --search "$SEARCH" --judge "$JUDGE" --steps "$STEPS" --out "$OUT/train"
python run.py evaluate --mode flow-grpo --adapter "$OUT/train/adapter_final" --data data/bamboogle.jsonl --search "$SEARCH" --judge "$JUDGE" --out "$OUT/flow_grpo"
python summarize.py "$OUT/direct" "$OUT/agentflow" "$OUT/flow_grpo"
