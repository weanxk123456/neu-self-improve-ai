"""Evaluate on Bamboogle.

Direct Inference baseline (no retrieval):
    python eval.py --mode direct --out results/direct_inference.json
Search-R1 inference with a trained LoRA adapter:
    python eval.py --mode search --adapter runs/ppo/adapter_step200 --out results/search_r1_ppo.json
Sanity check with an official full-finetuned Search-R1 checkpoint (any HF model id):
    python eval.py --mode search --model <official-checkpoint-id> --out results/official_check.json
"""
import argparse
import json
from pathlib import Path

from transformers import AutoTokenizer

from searchr1.data import load_bamboogle
from searchr1.evaluate import evaluate
from searchr1.retriever import make_retriever
from searchr1.rollout import RolloutConfig


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["direct", "search"], required=True)
    p.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    p.add_argument("--adapter", default=None, help="path to a LoRA adapter saved by train.py")
    p.add_argument("--backend", choices=["hf", "vllm"], default="vllm")
    p.add_argument("--bamboogle", default="data/BambooglePrerelease.csv")
    p.add_argument("--retriever", choices=["http", "bm25"], default="http")
    p.add_argument("--retriever_url", default="http://127.0.0.1:8000/retrieve")
    p.add_argument("--topk", type=int, default=3)
    p.add_argument("--max_turns", type=int, default=4)
    p.add_argument("--out", required=True)
    a = p.parse_args()

    tok = AutoTokenizer.from_pretrained(a.model)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    if a.backend == "vllm":
        from searchr1.generation import VLLMGenerator

        gen = VLLMGenerator(a.model, gpu_memory_utilization=0.8, enable_lora=a.adapter is not None)
        if a.adapter:
            gen.set_adapter(a.adapter, 1)
    else:
        import torch
        from transformers import AutoModelForCausalLM

        from searchr1.generation import HFGenerator

        model = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=torch.bfloat16).cuda()
        if a.adapter:
            from peft import PeftModel

            model = PeftModel.from_pretrained(model, a.adapter).merge_and_unload()
        gen = HFGenerator(model, tok)

    retriever = make_retriever(a.retriever, a.retriever_url, a.topk) if a.mode == "search" else None
    em, recs = evaluate(load_bamboogle(a.bamboogle), gen, tok, a.mode, retriever,
                        RolloutConfig(max_turns=a.max_turns))
    print(f"Bamboogle EM ({a.mode}, adapter={a.adapter}): {em:.3f}  (n={len(recs)})")
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps({"em": em, "mode": a.mode, "model": a.model,
                                       "adapter": a.adapter, "records": recs}, indent=1))


if __name__ == "__main__":
    main()
