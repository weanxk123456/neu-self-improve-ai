#!/usr/bin/env bash
# E5 dense retriever over the 2018 Wikipedia dump (the paper's setting), served by
# Search-R1's own retrieval server. Run it in a SEPARATE conda env and keep it running.
# Check the Search-R1 README for the current download/launch flags if anything changed.
#
# Needs ~65 GB disk; the flat FAISS index needs a lot of RAM (or use --faiss_gpu).
set -euo pipefail
SAVE=${SAVE:-$HOME/searchr1_index}

conda create -n retriever python=3.10 -y
conda activate retriever
conda install -c pytorch -c nvidia faiss-gpu -y   # or faiss-cpu
pip install torch transformers datasets uvicorn fastapi pyserini

git clone https://github.com/PeterGriffinJin/Search-R1.git
cd Search-R1
python scripts/download.py --save_path "$SAVE"
cat "$SAVE"/part_* > "$SAVE"/e5_Flat.index
gzip -d "$SAVE"/wiki-18.jsonl.gz

python search_r1/search/retrieval_server.py \
  --index_path "$SAVE"/e5_Flat.index \
  --corpus_path "$SAVE"/wiki-18.jsonl \
  --topk 3 --retriever_name e5 --retriever_model intfloat/e5-base-v2 --faiss_gpu
# Server listens on http://127.0.0.1:8000/retrieve
# Quick test:
#   curl -s -X POST http://127.0.0.1:8000/retrieve -H 'Content-Type: application/json' \
#        -d '{"queries":["Who founded CNN?"],"topk":3,"return_scores":true}'
