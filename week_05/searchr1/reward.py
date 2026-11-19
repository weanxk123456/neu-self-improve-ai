"""Exact-match reward, following Search-R1's qa_em.py.

compute_score() has the same signature style as TinyZero's countdown
compute_score(solution_str, ground_truth), so it can be dropped into the
veRL/TinyZero reward manager as well.
"""
import re
import string

_ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)
_PUNCT = set(string.punctuation)


def normalize_answer(s: str) -> str:
    s = s.lower()
    s = "".join(ch for ch in s if ch not in _PUNCT)
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def em_check(prediction: str, golden_answers) -> int:
    if prediction is None:
        return 0
    if isinstance(golden_answers, str):
        golden_answers = [golden_answers]
    p = normalize_answer(prediction)
    return int(any(p == normalize_answer(g) for g in golden_answers))


def extract_answer(text: str):
    """Return the content of the LAST <answer>...</answer> block, or None."""
    matches = _ANSWER_RE.findall(text)
    return matches[-1].strip() if matches else None


def compute_score(solution_str: str, ground_truth, format_score: float = 0.0, score: float = 1.0) -> float:
    """Outcome reward: 1 if the extracted answer exactly matches a gold answer, else 0.

    `solution_str` must contain only the model's response (not the prompt,
    whose template itself contains an <answer> example).
    """
    golds = ground_truth["target"] if isinstance(ground_truth, dict) else ground_truth
    answer = extract_answer(solution_str)
    if answer is None:
        return 0.0
    return score if em_check(answer, golds) else format_score
