"""Small model-based ramp-merging MDP, adapted from week_01's domain."""
from pathlib import Path
import json
import argparse
from sqlalchemy import (MetaData, Table, Column, Integer, String, Boolean, Float,
                        ForeignKey, CheckConstraint, create_engine, select)

metadata = MetaData()
states = Table('states', metadata,
    Column('id', Integer, primary_key=True), Column('name', String, nullable=False, unique=True),
    Column('terminal', Boolean, nullable=False))
actions = Table('actions', metadata,
    Column('id', Integer, primary_key=True), Column('name', String, nullable=False, unique=True))
transitions = Table('transitions', metadata,
    Column('state_id', ForeignKey('states.id'), primary_key=True),
    Column('action_id', ForeignKey('actions.id'), primary_key=True),
    Column('next_state_id', ForeignKey('states.id'), primary_key=True),
    Column('probability', Float, nullable=False), Column('reward', Float, nullable=False),
    CheckConstraint('probability >= 0 AND probability <= 1'))
policy_actions = Table('policy_actions', metadata,
    Column('iteration', Integer, primary_key=True),
    Column('state_id', ForeignKey('states.id'), primary_key=True),
    Column('action_id', ForeignKey('actions.id'), primary_key=True),
    Column('probability', Float, nullable=False),
    CheckConstraint('probability >= 0 AND probability <= 1'))
state_values = Table('state_values', metadata,
    Column('iteration', Integer, primary_key=True),
    Column('state_id', ForeignKey('states.id'), primary_key=True),
    Column('value', Float, nullable=False))

# 0=safe gap, 1=unsafe gap, 2=success, 3=failure; 0=wait, 1=merge.
# Teaching assumptions, NOT probabilities estimated from HighwayEnv.
MODEL = {
    (0, 0): [(0, .70, -.01), (1, .25, -.01), (3, .05, -1.)],
    (1, 0): [(0, .30, -.01), (1, .65, -.01), (3, .05, -1.)],
    (0, 1): [(2, .98, 1.), (3, .02, -1.)],
    (1, 1): [(2, .20, 1.), (3, .80, -1.)],
}

def build_database(path):
    """Rebuild only this experiment's tables; reruns are deterministic."""
    engine = create_engine('sqlite:///' + Path(path).resolve().as_posix())
    with engine.connect() as conn:
        conn.exec_driver_sql('PRAGMA foreign_keys=ON')
    metadata.drop_all(engine)
    metadata.create_all(engine)
    with engine.begin() as conn:
        conn.execute(states.insert(), [dict(id=i, name=n, terminal=i >= 2)
            for i, n in enumerate(['safe_gap', 'unsafe_gap', 'success', 'failure'])])
        conn.execute(actions.insert(), [dict(id=0, name='wait'), dict(id=1, name='merge')])
        conn.execute(transitions.insert(), [dict(state_id=s, action_id=a,
            next_state_id=ns, probability=p, reward=r)
            for (s, a), outcomes in MODEL.items() for ns, p, r in outcomes])
    return engine

def load_model(engine):
    with engine.connect() as conn:
        state_rows = conn.execute(select(states)).mappings().all()
        rows = conn.execute(select(transitions)).mappings().all()
    terminal = {row['id']: row['terminal'] for row in state_rows}
    model = {}
    for row in rows:
        key = row['state_id'], row['action_id']
        model.setdefault(key, []).append((row['next_state_id'], row['probability'], row['reward']))
    for key, outcomes in model.items():
        if abs(sum(p for _, p, _ in outcomes) - 1.) > 1e-12:
            raise ValueError(f'Transitions do not sum to one: {key}')
    return model, terminal

def action_values(model, values, gamma):
    return {(s, a): sum(p * (r + gamma * values[ns]) for ns, p, r in outcomes)
            for (s, a), outcomes in model.items()}

def evaluate(model, terminal, policy, gamma=.95, tolerance=1e-10):
    if not 0 <= gamma < 1:
        raise ValueError('This implementation requires 0 <= gamma < 1')
    for s, done in terminal.items():
        if not done:
            probabilities = [policy.get((s, a), 0.) for ss, a in model if ss == s]
            if any(p < 0 or p > 1 for p in probabilities) or abs(sum(probabilities)-1) > 1e-12:
                raise ValueError(f'Invalid policy for state {s}')
    values = dict.fromkeys(terminal, 0.)
    deltas = []
    for _ in range(100000):
        q = action_values(model, values, gamma)
        updated = {s: 0. if done else sum(policy.get((s, a), 0.) * q[s, a]
            for ss, a in model if ss == s) for s, done in terminal.items()}
        delta = max(abs(updated[s] - values[s]) for s in values)
        deltas.append(delta)
        values = updated
        if delta < tolerance:
            return values, deltas
    raise RuntimeError('Policy evaluation did not converge')

def improve(model, terminal, values, gamma=.95):
    q = action_values(model, values, gamma)
    policy = {}
    for s, done in terminal.items():
        if done:
            continue
        choices = sorted(a for ss, a in model if ss == s)
        # Deterministic tie breaking avoids cycling among equivalent actions.
        best = max(choices, key=lambda a: q[s, a])
        policy.update({(s, a): float(a == best) for a in choices})
    return policy, q

def run(output_dir, gamma=.95):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    engine = build_database(output_dir / 'merge_mdp.sqlite')
    model, terminal = load_model(engine)
    policy = {(s, a): .5 for s, a in model}
    history = []
    for iteration in range(100):
        with engine.begin() as conn:
            conn.execute(policy_actions.insert(), [dict(iteration=iteration, state_id=s,
                action_id=a, probability=p) for (s, a), p in policy.items()])
            rows = conn.execute(select(policy_actions).where(
                policy_actions.c.iteration == iteration)).mappings().all()
        policy = {(r['state_id'], r['action_id']): r['probability'] for r in rows}
        values, deltas = evaluate(model, terminal, policy, gamma)
        new_policy, q = improve(model, terminal, values, gamma)
        with engine.begin() as conn:
            conn.execute(state_values.insert(), [dict(iteration=iteration, state_id=s, value=v)
                                                for s, v in values.items()])
        history.append(dict(iteration=iteration, values=values,
            policy={f'{s},{a}': p for (s, a), p in policy.items()},
            q_values={f'{s},{a}': v for (s, a), v in q.items()},
            evaluation_sweeps=len(deltas), deltas=deltas))
        if new_policy == policy:
            break
        policy = new_policy
    else:
        raise RuntimeError('Policy iteration did not converge')
    residual = max(abs(values[s] - max(q[ss, a] for ss, a in model if ss == s))
                   for s, done in terminal.items() if not done)
    report = dict(gamma=gamma, initial_distribution={'safe_gap': .5, 'unsafe_gap': .5},
        initial_return=sum(history[0]['values'][s] for s in [0, 1]) / 2,
        final_return=(values[0] + values[1]) / 2,
        bellman_optimality_residual=residual, history=history)
    (output_dir / 'results.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    engine.dispose()
    return report

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=Path(__file__).parent / 'results')
    args = parser.parse_args()
    result = run(args.out)
    print(f"Initial expected return: {result['initial_return']:.6f}")
    print(f"Final expected return:   {result['final_return']:.6f}")
    print('Final policy:', result['history'][-1]['policy'])
    print(f"Bellman optimality residual: {result['bellman_optimality_residual']:.3e}")
