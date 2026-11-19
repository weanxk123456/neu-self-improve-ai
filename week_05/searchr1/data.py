"""Datasets: Bamboogle (test) and NQ + HotpotQA (train), as in Search-R1."""
import json
import random


def load_bamboogle(path: str):
    import pandas as pd

    df = pd.read_csv(path)
    rows = []
    for i, r in df.iterrows():
        if not isinstance(r["Question"], str) or not r["Question"].strip():
            continue
        golds = [str(r["Answer"]).strip()]
        if "Answer 2" in df.columns and isinstance(r["Answer 2"], str) and r["Answer 2"].strip():
            golds.append(r["Answer 2"].strip())
        rows.append({"id": f"bamboogle_{i}", "question": r["Question"].strip(), "golden_answers": golds})
    return rows


def load_train(seed: int = 0, max_examples: int | None = None, train_file: str | None = None):
    """NQ train + HotpotQA train (merged and shuffled), from FlashRAG's HF release.

    If `train_file` is given it must be a JSONL with fields question / golden_answers.
    """
    rows = []
    if train_file:
        with open(train_file) as f:
            for line in f:
                ex = json.loads(line)
                rows.append({"question": ex["question"], "golden_answers": list(ex["golden_answers"])})
    else:
        from datasets import load_dataset

        for name in ("nq", "hotpotqa"):
            ds = load_dataset("RUC-NLPIR/FlashRAG_datasets", name, split="train")
            for ex in ds:
                rows.append({"question": ex["question"], "golden_answers": list(ex["golden_answers"])})
    rng = random.Random(seed)
    rng.shuffle(rows)
    if max_examples:
        rows = rows[:max_examples]
    return rows


def infinite_batches(rows, batch_size: int, seed: int = 0):
    rng = random.Random(seed)
    order = list(range(len(rows)))
    while True:
        rng.shuffle(order)
        for i in range(0, len(order) - batch_size + 1, batch_size):
            yield [rows[j] for j in order[i : i + batch_size]]
