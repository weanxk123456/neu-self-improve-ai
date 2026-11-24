"""CLI for direct/AgentFlow/LoRA evaluation, demonstrations and training."""
import argparse
import json
import random
from pathlib import Path
from core import Agent, DIRECT, exact_match, load_data, write_json
from judge import Judge
from tools import Tools


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command', choices=['evaluate', 'demo', 'train'])
    p.add_argument('--model', default='Qwen/Qwen2.5-7B-Instruct')
    p.add_argument('--mode', choices=['direct', 'agentflow', 'flow-grpo'], default='agentflow')
    p.add_argument('--adapter', help='Adapter directory; required for flow-grpo evaluation')
    p.add_argument('--data', default='data/bamboogle.jsonl')
    p.add_argument('--heldout', default='data/bamboogle.jsonl', help='Full held-out set for overlap checks')
    p.add_argument('--question', default='Who was president of the United States when Citibank was founded?')
    p.add_argument('--out', default='runs/experiment', help='New output directory; must not already exist')
    p.add_argument('--search', choices=['wiki', 'serper'], default='wiki')
    p.add_argument('--judge', choices=['em', 'llm'], default='em')
    p.add_argument('--judge-model', default='gpt-4o')
    p.add_argument('--cache', default='.cache')
    p.add_argument('--context', type=int, default=2048, help='Prompt + completion token budget')
    p.add_argument('--max-new', type=int, default=256)
    p.add_argument('--turns', type=int, default=3)
    p.add_argument('--four-bit', action='store_true', help='Optional QLoRA/quantized inference')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--limit', type=int)
    p.add_argument('--steps', type=int, default=100)
    p.add_argument('--group', type=int, default=4)
    p.add_argument('--temperature', type=float, default=0.5, help='Planner rollout sampling temperature')
    p.add_argument('--lr', type=float, default=1e-5, help='Small-run LoRA starting point, not paper setting')
    p.add_argument('--beta', type=float, default=0.001)
    p.add_argument('--clip', type=float, default=0.2)
    p.add_argument('--rank', type=int, default=16)
    p.add_argument('--alpha', type=int, default=32)
    p.add_argument('--save-every', type=int, default=20)
    return p


def evaluate(args, model, agent, judge):
    rows = load_data(args.data, args.limit)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    write_json(out / 'config.json', vars(args))
    scores, ems = [], []
    with (out / 'predictions.jsonl').open('w', encoding='utf-8') as f:
        for i, row in enumerate(rows):
            if args.mode == 'direct':
                prediction = model.generate(DIRECT, row['question'], adapter=False)[0]
                trace = None
            else:
                trace = agent.run(row['question'], planner_adapter=args.mode == 'flow-grpo')
                prediction = trace.answer
            score = judge.score(row['question'], prediction, row['answers'])
            em = exact_match(prediction, row['answers'])
            record = {**row, 'prediction': prediction, 'score': score, 'em': em,
                      'trajectory': trace.to_dict() if trace else None}
            f.write(json.dumps(record, ensure_ascii=False) + '\n')
            f.flush()
            scores.append(score)
            ems.append(em)
            print(f'[{i+1}/{len(rows)}] score={score} prediction={prediction}', flush=True)
    write_json(out / 'metrics.json', {'mode': args.mode, 'n': len(rows), 'judge': args.judge,
        'accuracy_percent': 100*sum(scores)/len(scores), 'em_percent': 100*sum(ems)/len(ems),
        'search_backend': args.search, 'model': args.model, 'adapter': args.adapter,
        'note': 'Minimal reimplementation; see README for deviations from paper.'})


def main():
    args = parser().parse_args()
    if args.turns < 1 or args.save_every < 1 or (args.limit is not None and args.limit < 1):
        raise ValueError('turns/save-every/limit must be positive')
    if args.command != 'train':
        if args.mode == 'flow-grpo' and not args.adapter:
            raise ValueError('--mode flow-grpo requires --adapter')
        if args.mode != 'flow-grpo' and args.adapter:
            raise ValueError('Only flow-grpo evaluation/demo accepts --adapter')
    import torch
    from transformers import set_seed
    random.seed(args.seed)
    set_seed(args.seed)
    if args.command == 'train':
        from train import train
        train(args)
        return
    from model import LocalModel
    model = LocalModel(args.model, args.context, args.max_new, args.adapter, four_bit=args.four_bit)
    tools = Tools(args.search, args.cache + '/tools') if args.mode != 'direct' else None
    agent = Agent(model, tools, args.turns)
    if args.command == 'demo':
        if args.mode == 'direct':
            print(model.generate(DIRECT, args.question)[0])
        else:
            trajectory = agent.run(args.question, args.mode == 'flow-grpo', verbose=True)
            print('Final answer:', trajectory.answer)
        return
    evaluate(args, model, agent, Judge(args.judge, args.judge_model, args.cache + '/judge'))


if __name__ == '__main__':
    main()
