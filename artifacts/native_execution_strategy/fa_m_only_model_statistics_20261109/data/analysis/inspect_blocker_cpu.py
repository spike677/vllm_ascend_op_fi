"""Read stored exact resource identity, and bounded ATK queue process metadata."""
import json
from pathlib import Path

ROOT=Path('/home/fault28-m-only-model-20261109')


for attempt in ('formal_b1_a1','formal_b1_a3'):
    path=ROOT/'outputs'/attempt/'resource_checks.jsonl'
    if not path.exists():continue
    last=json.loads(path.read_text().splitlines()[-1])
    for row in last.get('unapproved_device_capable_containers',[]):
        print(json.dumps({'attempt':attempt,'blocker':row}))
