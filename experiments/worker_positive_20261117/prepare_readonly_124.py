"""Targeted old/new discovery comparison; never claims wheel-positive acceptance."""
import json
from pathlib import Path

root = Path('/home/worker-positive-20261117')
controller = Path('/home/overnight-17-20261116/run_cases_124.py').read_text()
controller = controller.replace("ROOT=Path('/home/overnight-17-20261116')", "ROOT=Path('/home/worker-positive-20261117')")
controller = controller.replace("str(ROOT/'source-a1/fault39/tools')", "'/home/overnight-17-20261116/source-a1/fault39/tools'")
controller = controller.replace("LABEL='overnight-17-20261116'", "LABEL='worker-positive-20261117'")
(root / 'run_cases_124.py').write_text(controller)
old = Path('/home/overnight-17-20261116/source-a2/experiments/overnight_20261116/worker_cases_weightnz_a2.json')
# Case files live at the controller root on 124, not necessarily inside source.
if not old.exists():
    old = Path('/home/overnight-17-20261116/worker_cases_weightnz_a2.json')
cases = json.loads(old.read_text())
for row in cases:
    row['id'] = 'observe-weightnz-nz2-current-call'
    row['env'].update(VLLM_ASCEND_ENABLE_NZ='2',
                      DEMONCAT_F20_DIAGNOSTICS_PATH='/evidence/native20-current-call.jsonl')
    for mount in row['mounts']:
        if mount[1] == '/native':
            mount[0] = str(root / 'build-a2/build20')
        elif mount[1].endswith('/libopapi_nn.so'):
            mount[0] = str(root / 'build-a2/build20/libopapi_nn.so')
(root / 'readonly-weightnz-a1.json').write_text(json.dumps(cases, indent=2) + '\n')
