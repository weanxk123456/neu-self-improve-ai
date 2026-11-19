"""Search-R1 training with LoRA: PPO or GRPO on NQ+HotpotQA, evaluated on Bamboogle.

Example:
    python train.py --algo grpo --out runs/grpo
    python train.py --algo ppo  --out runs/ppo
"""
import argparse
import json
import math
import random
import shutil
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer, get_constant_schedule_with_warmup

from searchr1 import algos as A
from searchr1.data import infinite_batches, load_bamboogle, load_train
from searchr1.evaluate import evaluate
from searchr1.generation import HFGenerator, VLLMGenerator
from searchr1.models import Critic, load_policy
from searchr1.prompts import build_prompt_ids
from searchr1.retriever import make_retriever
from searchr1.reward import em_check
from searchr1.rollout import RolloutConfig, run_rollouts


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--algo", choices=["ppo", "grpo"], required=True)
    p.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    p.add_argument("--out", required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--task", choices=["search", "countdown"], default="search")
    p.add_argument("--countdown_train_size", type=int, default=327680)
    p.add_argument("--countdown_test_size", type=int, default=1024)
    p.add_argument("--countdown_template", choices=["base", "qwen-instruct"], default="qwen-instruct")
    # data
    p.add_argument("--bamboogle", default="data/BambooglePrerelease.csv")
    p.add_argument("--train_file", default=None, help="optional JSONL instead of FlashRAG NQ+HotpotQA")
    # retrieval
    p.add_argument("--retriever", choices=["http", "bm25"], default="http")
    p.add_argument("--retriever_url", default="http://127.0.0.1:8000/retrieve")
    p.add_argument("--topk", type=int, default=3)
    # rollout
    p.add_argument("--backend", choices=["hf", "vllm"], default="vllm")
    p.add_argument("--vllm_mem", type=float, default=0.3)
    p.add_argument("--gen_batch_size", type=int, default=64, help="HF backend only")
    p.add_argument("--max_turns", type=int, default=4)
    p.add_argument("--max_tokens_per_turn", type=int, default=500)
    p.add_argument("--max_obs_tokens", type=int, default=500)
    p.add_argument("--max_response_tokens", type=int, default=3000)
    p.add_argument("--temperature", type=float, default=1.0)
    # RL
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--prompts_per_step", type=int, default=None, help="default: 64 (ppo) / 32 (grpo)")
    p.add_argument("--group_size", type=int, default=5, help="GRPO responses per prompt")
    p.add_argument("--ppo_epochs", type=int, default=1)
    p.add_argument("--mini_batches", type=int, default=2, help="optimizer steps per epoch")
    p.add_argument("--micro_bs", type=int, default=2)
    p.add_argument("--clip", type=float, default=0.2)
    p.add_argument("--value_clip", type=float, default=0.5)
    p.add_argument("--kl_coef", type=float, default=0.001)
    p.add_argument("--gamma", type=float, default=1.0)
    p.add_argument("--lam", type=float, default=1.0)
    p.add_argument("--lr_actor", type=float, default=1e-5)
    p.add_argument("--lr_critic", type=float, default=1e-4)
    p.add_argument("--warmup_steps", type=int, default=10)
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    # LoRA
    p.add_argument("--lora_r", type=int, default=32)
    p.add_argument("--lora_alpha", type=int, default=64)
    p.add_argument("--lora_dropout", type=float, default=0.0)
    # eval / io
    p.add_argument("--eval_every", type=int, default=25)
    p.add_argument("--no_mask", action="store_true", help="ablation: train on retrieved tokens too")
    return p.parse_args()


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


def append_jsonl(path, row):
    with open(path, "a") as f:
        f.write(json.dumps(row) + "\n")


def main():
    args = parse_args()
    set_seed(args.seed)
    out = Path(args.out)
    if out.exists() and any(out.iterdir()):
        raise ValueError("Output directory is not empty; choose a new --out to preserve results")
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(vars(args), indent=2))
    dev = "cuda"
    n_resp = 1 if args.algo == "ppo" else args.group_size
    n_prompts = args.prompts_per_step or (64 if args.algo == "ppo" else 32)

    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    # vLLM must be created before the training models (it reserves a fraction of GPU memory)
    vllm = VLLMGenerator(args.model, args.vllm_mem, max_lora_rank=max(args.lora_r, 16), seed=args.seed) \
        if args.backend == "vllm" else None

    policy = load_policy(args.model, args.lora_r, args.lora_alpha, args.lora_dropout, device=dev)
    opt_a = torch.optim.AdamW([p for p in policy.parameters() if p.requires_grad], lr=args.lr_actor, weight_decay=0.0)
    sch_a = get_constant_schedule_with_warmup(opt_a, args.warmup_steps)
    critic = opt_c = sch_c = None
    if args.algo == "ppo":
        critic = Critic(args.model, args.lora_r, args.lora_alpha, args.lora_dropout, device=dev)
        opt_c = torch.optim.AdamW([p for p in critic.parameters() if p.requires_grad], lr=args.lr_critic, weight_decay=0.0)
        sch_c = get_constant_schedule_with_warmup(opt_c, args.warmup_steps)

    generator = vllm if vllm else HFGenerator(policy, tok, args.gen_batch_size)
    version = [0]

    def sync_adapter():
        if vllm is None:
            return
        version[0] += 1
        path = out / "adapter_sync" / f"v{version[0]}"
        policy.save_pretrained(path)
        vllm.set_adapter(str(path), version[0])
        old = out / "adapter_sync" / f"v{version[0] - 2}"
        if old.exists():
            shutil.rmtree(old)

    is_countdown = args.task == "countdown"
    if is_countdown:
        from tinyzero_lora.task import load_data, prompt_ids as countdown_prompt, rollouts as countdown_rollouts, reward as countdown_reward
    retriever = None if is_countdown else make_retriever(args.retriever, args.retriever_url, args.topk)
    rcfg = RolloutConfig(args.max_turns, args.max_tokens_per_turn, args.max_obs_tokens,
                         args.max_response_tokens, args.temperature)
    if is_countdown:
        train_rows, test = load_data(args.countdown_train_size, args.countdown_test_size)
    else:
        train_rows, test = load_train(seed=args.seed, train_file=args.train_file), load_bamboogle(args.bamboogle)
    if len(train_rows) < n_prompts or not test:
        raise ValueError("Need at least prompts_per_step training rows and a nonempty test set")
    batches = infinite_batches(train_rows, n_prompts, args.seed)

    def run_eval(step):
        sync_adapter()
        if is_countdown:
            recs = []
            for start in range(0, len(test), args.gen_batch_size):
                chunk = test[start:start + args.gen_batch_size]
                ts = countdown_rollouts([countdown_prompt(tok, ex, args.countdown_template) for ex in chunk],
                                        generator, tok, args.max_response_tokens, 0.0)
                recs.extend({"id": ex["id"], "response": t.response_text,
                             "reward": countdown_reward(t, ex["golden_answers"]),
                             "ground_truth": ex["golden_answers"]} for t, ex in zip(ts, chunk))
            em = sum(r["reward"] == 1.0 for r in recs) / len(recs)
        else:
            em, recs = evaluate(test, generator, tok, "search", retriever, rcfg)
        append_jsonl(out / "eval_log.jsonl", {"step": step, ("countdown_accuracy" if is_countdown else "bamboogle_em"): em,
                                               "avg_search": float(np.mean([r.get("n_search", 0) for r in recs]))})
        (out / f"{'countdown' if is_countdown else 'bamboogle'}_step{step}.json").write_text(json.dumps(recs, indent=1))
        print(f"[eval] step {step}: {args.task} accuracy/EM = {em:.3f}", flush=True)
        if step > 0:
            policy.save_pretrained(out / f"adapter_step{step}")
            tok.save_pretrained(out / f"adapter_step{step}")
            if critic is not None:
                critic.backbone.save_pretrained(out / f"critic_adapter_step{step}")
                torch.save(critic.value_head.state_dict(), out / f"critic_value_head_step{step}.pt")

    run_eval(0)
    for step in range(1, args.steps + 1):
        t0 = time.time()
        exs = next(batches)
        prompt_ids, golds, groups = [], [], []
        for gi, ex in enumerate(exs):
            ids = countdown_prompt(tok, ex, args.countdown_template) if is_countdown else build_prompt_ids(tok, ex["question"], "search")
            for _ in range(n_resp):
                prompt_ids.append(ids); golds.append(ex["golden_answers"]); groups.append(gi)

        # ---------------------------------------------------------------- rollout
        sync_adapter()
        if is_countdown:
            trajs = countdown_rollouts(prompt_ids, generator, tok, args.max_response_tokens, args.temperature)
        else:
            trajs = run_rollouts(prompt_ids, generator, retriever, tok, rcfg)
        score_fn = countdown_reward if is_countdown else lambda t, g: float(em_check(t.answer, g))
        scores = torch.tensor([score_fn(t, g) for t, g in zip(trajs, golds)], device=dev)
        batch = A.collate(trajs, tok.pad_token_id, dev)
        mask = batch["resp_mask"] if not args.no_mask else (batch["resp_attn"] > 0).float()
        batch["mask"] = mask
        t_roll = time.time() - t0

        # ---------------------------------------------------------------- old log-probs, ref, values
        policy.eval()
        old_logp = A.batched(A.response_logprobs, policy, batch, args.micro_bs)
        with policy.disable_adapter():
            ref_logp = A.batched(A.response_logprobs, policy, batch, args.micro_bs)
        kl_k1 = A.kl_estimate(old_logp, ref_logp, "k1") * mask
        if args.algo == "ppo":
            critic.eval()
            old_values = A.batched(A.critic_values, critic, batch, args.micro_bs) * mask
            tok_rewards = A.outcome_to_token_rewards(scores, mask) - args.kl_coef * kl_k1
            adv, returns = A.gae_advantages(tok_rewards, old_values, mask, args.gamma, args.lam)
            batch.update(old_values=old_values, returns=returns)
        else:
            adv = A.grpo_advantages(scores, torch.tensor(groups, device=dev), mask)
        batch.update(old_logp=old_logp, ref_logp=ref_logp, adv=adv)

        # ---------------------------------------------------------------- updates
        policy.train()
        if critic is not None:
            critic.train()
        B = len(trajs)
        mini = math.ceil(B / args.mini_batches)
        stats = {"pg_loss": [], "clipfrac": [], "kl_loss": [], "vf_loss": []}
        for _ in range(args.ppo_epochs):
            perm = torch.randperm(B, device=dev)
            for s in range(0, B, mini):
                mb = A.index_batch(batch, perm[s : s + mini])
                n = mb["input_ids"].shape[0]
                opt_a.zero_grad(set_to_none=True)
                for sl in A.micro_slices(n, args.micro_bs):
                    w = (sl.stop - sl.start) / n
                    logp = A.response_logprobs(policy, mb, sl)
                    m = mb["mask"][sl]
                    pg, cf = A.ppo_policy_loss(logp, mb["old_logp"][sl], mb["adv"][sl], m, args.clip)
                    loss = pg
                    if args.algo == "grpo":
                        kl = A.seq_mean_token_mean(A.kl_estimate(logp, mb["ref_logp"][sl], "low_var_kl"), m)
                        loss = loss + args.kl_coef * kl
                        stats["kl_loss"].append(kl.item())
                    (loss * w).backward()
                    stats["pg_loss"].append(pg.item()); stats["clipfrac"].append(cf.item())
                torch.nn.utils.clip_grad_norm_([p for p in policy.parameters() if p.requires_grad], args.max_grad_norm)
                opt_a.step()

                if critic is not None:
                    opt_c.zero_grad(set_to_none=True)
                    for sl in A.micro_slices(n, args.micro_bs):
                        w = (sl.stop - sl.start) / n
                        v = A.critic_values(critic, mb, sl)
                        vl = A.value_loss(v, mb["old_values"][sl], mb["returns"][sl], mb["mask"][sl], args.value_clip)
                        (vl * w).backward()
                        stats["vf_loss"].append(vl.item())
                    torch.nn.utils.clip_grad_norm_([p for p in critic.parameters() if p.requires_grad], args.max_grad_norm)
                    opt_c.step()
        sch_a.step()
        if sch_c is not None:
            sch_c.step()

        row = {
            "step": step,
            "reward": scores.mean().item(),
            "avg_search": float(np.mean([t.n_search for t in trajs])),
            "avg_invalid": float(np.mean([t.n_invalid for t in trajs])),
            "answered": float(np.mean([t.answer is not None for t in trajs])),
            "resp_tokens": float(np.mean([len(t.resp_ids) for t in trajs])),
            "policy_tokens": float(np.mean([sum(t.resp_mask) for t in trajs])),
            "kl_k1": A.masked_mean(kl_k1, mask).item(),
            **{k: float(np.mean(v)) for k, v in stats.items() if v},
            "t_rollout": t_roll, "t_step": time.time() - t0,
        }
        append_jsonl(out / "train_log.jsonl", row)
        print(" ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}" for k, v in row.items()), flush=True)

        if step % args.eval_every == 0 or step == args.steps:
            run_eval(step)


if __name__ == "__main__":
    main()
