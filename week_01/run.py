import argparse
from dataclasses import asdict
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import urllib.request
import numpy as np
from environment import Scenario, RampMergeEnv, rule_action

ROOT = Path(__file__).parent

def write_json(path, obj):
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')

def validate(s, seeds):
    for seed in seeds:
        e = RampMergeEnv(asdict(s))
        try:
            obs,_ = e.reset(seed=seed)
            if not np.isfinite(obs).all():
                raise ValueError('non-finite initial observation')
            # Conservative clearance check; all initial vehicles are axis aligned.
            for i,a in enumerate(e.road.vehicles):
                for b in e.road.vehicles[i+1:]+e.road.objects:
                    if abs(a.position[0]-b.position[0]) < (a.LENGTH+b.LENGTH)/2+1 and abs(a.position[1]-b.position[1]) < (a.WIDTH+b.WIDTH)/2:
                        raise ValueError('initial overlap / insufficient clearance')
            for _ in range(3):
                obs,r,t,tr,_ = e.step(1)
                if not np.isfinite(obs).all() or not np.isfinite(r):
                    raise ValueError('non-finite rollout')
                if t or tr:
                    break
        finally:
            e.close()


def capture_frame(env):
    frame = env.render()
    # HighwayEnv disables ALL drawing for SDL dummy, including offscreen surfaces.
    # Re-enable surface drawing only; no display window is created in rgb_array mode.
    if env.viewer is not None and not env.viewer.enabled:
        env.viewer.enabled = True
        frame = env.render()
    if frame is None or np.ptp(frame) == 0:
        raise RuntimeError('renderer produced an empty frame')
    return frame.copy()


def episode(s, seed, policy='rule', gif=None):
    env = RampMergeEnv(asdict(s), render_mode='rgb_array' if gif else None)
    frames, trace = [], []
    total = 0.0
    rng = np.random.default_rng(seed+100000)
    try:
        env.reset(seed=seed)
        if gif:
            frames.append(capture_frame(env))
        for step in range(60):
            action = rule_action(env) if policy == 'rule' else int(rng.integers(5))
            obs, reward, term, trunc, info = env.step(action)
            if not np.isfinite(obs).all() or not np.isfinite(reward):
                raise ValueError('invalid numeric rollout')
            total += reward
            trace.append({'t':env.time, 'action':action,
                          'x':float(env.vehicle.position[0]), 'y':float(env.vehicle.position[1]),
                          'speed':float(env.vehicle.speed), 'reward':reward})
            if gif:
                frames.append(capture_frame(env))
            if term or trunc:
                break
        result = {'seed':seed, 'policy':policy, 'success':bool(info['success']),
                  'crashed':bool(info['crashed']), 'offroad':not info['on_road'],
                  'seconds':env.time, 'return':round(total,6), 'trace':trace}
        if gif:
            import imageio.v2 as imageio
            imageio.mimsave(gif, frames, duration=500, loop=0)
        return result
    finally:
        env.close()


def evaluate(s, seeds):
    episodes = [episode(s, seed, policy) for policy in ['rule','random'] for seed in seeds]
    summary = {}
    for policy in ['rule','random']:
        rows = [r for r in episodes if r['policy']==policy]
        summary[policy] = {key:float(np.mean([r[key] for r in rows]))
                           for key in ['success','crashed','offroad','seconds','return']}
        summary[policy]['n'] = len(rows)
    return summary, episodes


def demo_proposal(memory, attempt):
    """Offline feedback-driven fixture; explicitly NOT an LLM designer."""
    accepted = [m for m in memory if m['accepted']]
    if not accepted:
        s = Scenario()
    else:
        prev = accepted[-1]
        s = Scenario.from_dict(prev['scenario'])
        # Feedback selects mutation order. Retries explore different dimensions.
        order = ['vehicles','spacing','speed_spread','merge_length'] if prev['evaluation']['rule']['success'] >= .5 else ['speed_spread','vehicles','spacing','merge_length']
        changed = False
        for offset in range(4):
            key = order[(len(accepted)-1+attempt+offset)%4]
            if key=='vehicles' and s.vehicles<=16:
                s.vehicles += 2; changed=True
            elif key=='spacing' and s.spacing>=23:
                s.spacing -= 5; changed=True
            elif key=='speed_spread' and s.speed_spread<=5.5:
                s.speed_spread += 1.5; changed=True
            elif key=='merge_length' and s.merge_length>=160:
                s.merge_length -= 20; changed=True
            if changed:
                break
        if not changed:
            raise ValueError('demo design space exhausted')
        s.rationale = f'Offline fixture mutation: {key}; previous rule success={prev["evaluation"]["rule"]["success"]:.2f}'
    s.name = f'demo_{len(accepted):02d}_{attempt}'
    return {'scenario':asdict(s), 'hypothesis':s.rationale}


def llm_proposal(memory, model, out, round_index, attempt):
    key = os.environ.get('LLM_API_KEY')
    if not key or not model:
        raise RuntimeError('LLM mode requires LLM_API_KEY and --model; no silent demo fallback')
    base = os.environ.get('LLM_BASE_URL','https://api.openai.com/v1').rstrip('/')
    context = {'memory':memory[-12:], 'round':round_index,
               'note':'Only design seeds 0..7 are used for feedback; held-out 100..119 are unseen.'}
    messages = [{'role':'system','content':(ROOT/'prompts/designer.txt').read_text()},
                {'role':'user','content':json.dumps(context)}]
    write_json(out/f'prompt_{round_index:02d}_{attempt}.json', messages)
    body = {'model':model, 'messages':messages, 'response_format':{'type':'json_object'}}
    request = urllib.request.Request(base+'/chat/completions',
              data=json.dumps(body).encode(), headers={'Authorization':'Bearer '+key,
                                                      'Content-Type':'application/json'})
    # Never persist headers/credentials. HTTP failures propagate to the bounded retry loop.
    with urllib.request.urlopen(request, timeout=90) as response:
        payload = json.load(response)
    write_json(out/f'response_{round_index:02d}_{attempt}.json', payload)
    if payload['choices'][0].get('finish_reason') != 'stop':
        raise ValueError('incomplete or refused model response')
    obj = json.loads(payload['choices'][0]['message']['content'])
    if set(obj) != {'scenario','hypothesis'} or not isinstance(obj['hypothesis'],str):
        raise ValueError('expected scenario and hypothesis fields')
    return obj


def run(args):
    out = Path(args.out)
    out.mkdir(parents=True,exist_ok=True)
    if (out/'memory.json').exists():
        raise RuntimeError('output already contains a run; choose a new --out to preserve results')
    if args.rounds<1 or args.rounds>12:
        raise ValueError('rounds must be in 1..12')
    memory, accepted, fingerprints = [], [], set()
    write_json(out/'metadata.json', {'mode':args.mode,'model':args.model,'python':platform.python_version(),
               'highway_env':importlib.metadata.version('highway-env'),
               'design_seeds':list(range(8)), 'heldout_seeds':list(range(100,120)),
               'disclaimer':'SPADE-inspired constrained designer loop, no training or hint-regret. Demo is not LLM.'})
    for level in range(args.rounds):
        for attempt in range(3):
            record = {'round':level,'attempt':attempt,'accepted':False,'mode':args.mode}
            try:
                proposal = demo_proposal(memory,attempt) if args.mode=='demo' else llm_proposal(memory,args.model,out,level,attempt)
                s = Scenario.from_dict(proposal['scenario'])
                record.update(scenario=asdict(s),hypothesis=proposal['hypothesis'],complexity=s.complexity())
                numeric = {k:v for k,v in asdict(s).items() if k not in ['name','rationale']}
                digest = hashlib.sha256(json.dumps(numeric,sort_keys=True).encode()).hexdigest()
                if digest in fingerprints:
                    raise ValueError('duplicate scenario')
                if accepted and s.complexity() < accepted[-1]['complexity']+.015:
                    raise ValueError('structural complexity increment below 0.015')
                validate(s,[0,1,2])
                evaluation, episodes = evaluate(s,range(8))
                record['evaluation'] = evaluation
                write_json(out/f'design_{level:02d}_{attempt}.json', episodes)
                if evaluation['rule']['success']==0:
                    raise ValueError('no successful rule rollout on design seeds; solvability unknown, request revision')
                record['accepted']=True
                fingerprints.add(digest)
                accepted.append(record)
            except Exception as exc:
                record['error']=f'{type(exc).__name__}: {exc}'
            memory.append(record)
            write_json(out/'memory.json',memory)
            print(json.dumps({k:v for k,v in record.items() if k not in ['scenario','hypothesis']},ensure_ascii=False),flush=True)
            if record['accepted']:
                break
        else:
            print('Stopped after three rejected proposals; no artificial success.',flush=True)
            break
    # Held-out results do NOT feed back into the designer or acceptance decisions.
    results=[]
    for i,record in enumerate(accepted):
        s=Scenario.from_dict(record['scenario'])
        summary, episodes=evaluate(s,range(100,120))
        write_json(out/f'heldout_{i:02d}.json',episodes)
        results.append({'scenario':asdict(s),'complexity':s.complexity(),'evaluation':summary})
        if args.render:
            episode(s,100,gif=out/f'level_{i:02d}.gif')
    write_json(out/'summary.json',results)
    lines=['# Experiment results','',f'Mode: **{args.mode}**. No policy training.',
           'Demo results validate plumbing only; they are not LLM generation evidence.','',
           '| Level | Structural score | Rule success | Rule collision | Random success |',
           '|---|---:|---:|---:|---:|']
    for i,r in enumerate(results):
        q=r['evaluation']
        lines.append(f'| {i} | {r["complexity"]:.3f} | {q["rule"]["success"]:.0%} | {q["rule"]["crashed"]:.0%} | {q["random"]["success"]:.0%} |')
    lines += ['', '20 held-out seeds per policy/level; rates have substantial sampling uncertainty.',
              'Structural score is hand-defined and does not establish monotonically increasing difficulty.',
              'Rule controller reads simulator state. Acceptance only witnesses some successful design-seed rollouts.',
              'Neither smoke checks nor these rollouts prove all instances feasible.']
    (out/'REPORT.md').write_text('\n'.join(lines),encoding='utf-8')
    if not accepted:
        raise SystemExit('No environments accepted; see memory.json')

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--mode',choices=['demo','llm'],default='demo')
    p.add_argument('--model',default=None)
    p.add_argument('--rounds',type=int,default=5)
    p.add_argument('--out',default='results/my_run')
    p.add_argument('--render',action='store_true')
    run(p.parse_args())
