"""Prompt templates.

SEARCH_TEMPLATE is the Search-R1 training/inference template (paper, Table 1).
DIRECT_SYSTEM is the FlashRAG-style "naive generation" prompt used for the
Direct Inference baseline (no retrieval, answer only).
"""

SEARCH_TEMPLATE = (
    "Answer the given question. You must conduct reasoning inside <think> and </think> "
    "first every time you get new information. After reasoning, if you find you lack some "
    "knowledge, you can call a search engine by <search> query </search>, and it will return "
    "the top searched results between <information> and </information>. You can search as "
    "many times as you want. If you find no further external knowledge needed, you can "
    "directly provide the answer inside <answer> and </answer> without detailed "
    "illustrations. For example, <answer> xxx </answer>. Question: {question}\n"
)

DIRECT_SYSTEM = (
    "Answer the question based on your own knowledge. "
    "Only give me the answer and do not output any other words."
)


def clean_question(q: str) -> str:
    q = q.strip()
    if not q.endswith("?"):
        q += "?"
    return q


def build_messages(question: str, mode: str):
    question = clean_question(question)
    if mode == "search":
        return [{"role": "user", "content": SEARCH_TEMPLATE.format(question=question)}]
    if mode == "direct":
        return [
            {"role": "system", "content": DIRECT_SYSTEM},
            {"role": "user", "content": f"Question: {question}"},
        ]
    raise ValueError(f"unknown mode {mode}")


def build_prompt_ids(tokenizer, question: str, mode: str) -> list[int]:
    """Apply the chat template (instruct model) and tokenize."""
    text = tokenizer.apply_chat_template(
        build_messages(question, mode), tokenize=False, add_generation_prompt=True
    )
    return tokenizer(text, add_special_tokens=False)["input_ids"]
