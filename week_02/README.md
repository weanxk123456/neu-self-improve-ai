# Week 02：Minimum MDP and Policy Iteration

This project continues the work on autonomous highway merging from week_01, adopting the reward structure found in its `environment.py`: +1 for success, -1 for failure, and -0.01 for non-terminal steps.

A standalone implementation of a finite MDP has been added this week

## Run

```bash
python -m pip install -r requirements.txt
python mdp.py
python -m unittest discover -s . -v
```

```bash
jupyter notebook assignment.ipynb
```

## Models and Experiments

Four states: safe gap, unsafe gap, success, and failure. The first two states allow for waiting or attempting to merge; the latter two are terminal states with zero value.


| Current State | Action | Next state and probability          |
| ------------- | ------ | ----------------------------------- |
| Safe          | wait   | Safe 0.70，Unsafe 0.25，fail 0.05 |
| Unsafe        | wait   | Safe 0.30，Unsafe 0.65，fail 0.05   |
| Safe          | merge  | success 0.98，fail 0.02             |
| Unsafe        | merge  | success 0.20，fail 0.80             |

Entering a success state yields a reward of +1, entering a failure state yields -1, and other transitions yield -0.01; only one reward is counted per transition. All probabilities are manually set for pedagogical purposes and were not estimated from the previous week's data. The probability of failure while waiting represents a fixed risk of the lane ending; vehicle position, speed, and remaining distance are not explicitly tracked.

SQLAlchemy is used to define and create five tables—`states`, `actions`, `transitions`, `policy_actions`, and `state_values`—with `schema.sql` containing the corresponding SQLite DDL. The algorithm reads the transition model and per-round policy from the database and writes value and policy history back to it.

The initial policy selects between two actions with equal probability in each non-terminal state. A discount factor of 0.95 is used; the policy is evaluated via Bellman expectation updates, with evaluation terminating when the maximum update magnitude falls below 1e-10. Action values ​​are then calculated to perform greedy policy improvement, a process repeated until the policy stabilizes. When multiple actions are available, the selection is fixed based on action ID to avoid meaningless switching.

## Results

Comparison based on an identical initial distribution of 0.5 for both safe and unsafe states:

- Expected discounted return of the initial policy: **0.130634**.
- Expected discounted return of the final policy: **0.759869**.
- Final policy: **Merge when safe; wait when unsafe**.
- Value of the safe state: **0.960000**; value of the unsafe state: **0.559739**.
- Bellman optimality residual: approximately **5.54e-11**.
