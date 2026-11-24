"""Combine actual metrics only. No invented replication scores."""
import argparse
import json
from pathlib import Path
p = argparse.ArgumentParser()
p.add_argument('directories', nargs='+')
a = p.parse_args()
print('| Mode | N | Judge | Accuracy (%) | EM (%) |')
print('|---|---:|---|---:|---:|')
for directory in a.directories:
    m = json.loads((Path(directory) / 'metrics.json').read_text())
    print(f"| {m['mode']} | {m['n']} | {m['judge']} | {m['accuracy_percent']:.2f} | {m['em_percent']:.2f} |")
