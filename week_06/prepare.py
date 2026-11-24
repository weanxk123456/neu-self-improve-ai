"""Download official Bamboogle and an independent NQ training subset to JSONL."""
import argparse
import json
from pathlib import Path


def convert(row, index):
    question = row.get('question', row.get('Question'))
    answers = row.get('golden_answers', row.get('answers', row.get('answer', row.get('Answer'))))
    if isinstance(answers, dict):
        answers = answers.get('text')
    if isinstance(answers, str):
        answers = [answers]
    if not question or not answers:
        raise ValueError(f'Unexpected dataset schema: {list(row)}; use local JSONL instead')
    return {'id': str(row.get('id', index)), 'question': question, 'answers': answers}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--train-limit', type=int, default=2000)
    p.add_argument('--out', default='data')
    p.add_argument('--test-source', help='Optional local raw JSONL with question/answers or golden_answers')
    p.add_argument('--train-source', help='Optional local raw JSONL training source')
    a = p.parse_args()
    if a.train_limit < 1:
        raise ValueError('--train-limit must be positive')
    from datasets import load_dataset
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    # Never overwrite user data or pretend toy questions are the benchmark.
    for name in ['bamboogle.jsonl', 'train.jsonl']:
        if (out / name).exists():
            raise FileExistsError(out / name)
    test_source = a.test_source or 'https://huggingface.co/datasets/RUC-NLPIR/FlashRAG_datasets/resolve/main/bamboogle/test.jsonl'
    test = load_dataset('json', data_files=test_source, split='train')
    test_rows = [convert(row, i) for i, row in enumerate(test)]
    forbidden = {' '.join(x['question'].lower().split()) for x in test_rows}
    train_source = a.train_source or 'https://huggingface.co/datasets/RUC-NLPIR/FlashRAG_datasets/resolve/main/nq/train.jsonl'
    nq = load_dataset('json', data_files=train_source, split='train', streaming=True)
    training, seen = [], set()
    for i, row in enumerate(nq):
        item = convert(row, i)
        key = ' '.join(item['question'].lower().split())
        if key in forbidden or key in seen:
            continue
        seen.add(key)
        training.append(item)
        if len(training) >= a.train_limit:
            break
    for name, rows in [('bamboogle.jsonl', test_rows), ('train.jsonl', training)]:
        with (out / name).open('w', encoding='utf-8') as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + '\n')
        print(f'{out / name}: {len(rows)} rows')
    print('NQ-only subset is a reduced training setting; not the paper mixed training corpus.')


if __name__ == '__main__':
    main()
