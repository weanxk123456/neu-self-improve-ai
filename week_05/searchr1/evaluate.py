"""Exact-match evaluation for Direct Inference and Search-R1 style inference."""
from .prompts import build_prompt_ids
from .reward import em_check
from .rollout import RolloutConfig, run_rollouts


def evaluate(examples, generator, tokenizer, mode, retriever=None, rollout_cfg: RolloutConfig = None):
    prompts = [build_prompt_ids(tokenizer, ex["question"], mode) for ex in examples]
    records = []
    if mode == "direct":
        texts = generator.generate(prompts, max_new_tokens=64, temperature=0.0, stop=None)
        for ex, text in zip(examples, texts):
            pred = text.strip()
            records.append({**ex, "prediction": pred, "output": text,
                            "em": em_check(pred, ex["golden_answers"])})
    else:
        cfg = RolloutConfig(**{**(rollout_cfg or RolloutConfig()).__dict__, "temperature": 0.0})
        trajs = run_rollouts(prompts, generator, retriever, tokenizer, cfg)
        for ex, t in zip(examples, trajs):
            records.append({**ex, "prediction": t.answer, "output": t.response_text,
                            "n_search": t.n_search, "em": em_check(t.answer, ex["golden_answers"])})
    em = sum(r["em"] for r in records) / len(records)
    return em, records
