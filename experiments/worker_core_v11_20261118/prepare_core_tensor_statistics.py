"""Prepare one bounded, disposable Worker18 statistics case; never launch it."""
import argparse
import ast
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import re

from experiments.c2c3_closure_20261117.prepare_tensor_worker_core import core_cases

COLLECTOR_SOURCE = '/home/batch-scale-dynamic-msprobe-20261023/python_packages'
QUEUE_SCRIPT = Path(__file__).resolve().parents[1]/'c2c3_closure_20261117/run_queue_124.py'


def queue_collector_mount():
    """Verify the queue's literal RO dependency without importing or running it."""
    raw = QUEUE_SCRIPT.read_bytes()
    tree = ast.parse(raw)
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'run_case']
    mounts = []
    if len(functions) == 1:
        stores = [node for node in ast.walk(functions[0])
                  if isinstance(node, ast.Name) and node.id == 'mounts' and isinstance(node.ctx, ast.Store)]
        assignments = [node for node in functions[0].body if isinstance(node, ast.Assign)
                       and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name)
                       and node.targets[0].id == 'mounts' and isinstance(node.value, ast.List)]
        if len(stores) == len(assignments) == 1:
            for item in assignments[0].value.elts:
                if (isinstance(item, (ast.Tuple, ast.List)) and len(item.elts) == 3
                        and isinstance(item.elts[1], ast.Constant) and item.elts[1].value == '/msprobe_packages'):
                    try:
                        mounts.append(tuple(ast.literal_eval(item)))
                    except (ValueError, TypeError):
                        raise ValueError('QUEUE_FIXED_READ_ONLY_COLLECTOR_MOUNT_REQUIRED') from None
    if mounts != [(COLLECTOR_SOURCE, '/msprobe_packages', 'ro')]:
        raise ValueError('QUEUE_FIXED_READ_ONLY_COLLECTOR_MOUNT_REQUIRED')
    return dict(source=COLLECTOR_SOURCE, destination='/msprobe_packages', mode='ro',
                supplied_by='run_queue_124.run_case fixed mount; not a case override',
                queue_script_sha256=hashlib.sha256(raw).hexdigest())


def cases(source, manifest_sha256, *, collector=COLLECTOR_SOURCE,
          card=5, image='quay.io/ascend/vllm-ascend:v0.18.0'):
    if not re.fullmatch('[0-9a-f]{64}', manifest_sha256):
        raise ValueError('EXACT_SOURCE_MANIFEST_SHA256_REQUIRED')
    if collector != COLLECTOR_SOURCE:
        raise ValueError('QUEUE_COLLECTOR_SOURCE_OVERRIDE_FORBIDDEN')
    collector_dependency = queue_collector_mount()
    row = deepcopy(core_cases(source, card, image)[0])
    row['id'] = 'public-worker18-statistics-three-windows'
    row['env']['PYTHONPATH'] = '/msprobe_packages:'+row['env']['PYTHONPATH']
    row['command'] = row['command'].replace(
        '/source/experiments/c2c3_closure_20261117/natural_tensor_worker_core_probe.py',
        '/source/experiments/worker_core_v11_20261118/core_tensor_statistics_probe.py '
        '--source-manifest /source/source_manifest.json --source-manifest-sha256 '+manifest_sha256)
    row['review'].update(experiment_id=row['id'],
        worker='One disposable fresh standard TP1 eager Worker; actual public enqueue; same Worker B/repeat/F',
        acceptance='RAW_STATISTICS_REVIEW_REQUIRED_NO_E5_GRANTED_NO_HISTORICAL_SCOPE_TRANSFER',
        capture_scope='root entry through logits processor return; no sampler; native root module absent',
        actual_step_mapping={'step0': 'baseline', 'step1': 'baseline_repeat', 'step2': 'c2_fault'},
        required_queue_collector_mount=collector_dependency,
        collector_control='Pinned recursive hooks; same-model cache rearm only after exact owned teardown',
        global_api_wrappers_restored=False, disposable_worker_required=True)
    row['review']['raw_evidence'] += ['/evidence/probe/raw_model_capture/collector_sidecar.json',
                                    '/evidence/probe/raw_model_capture/statistics']
    return [row]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True)
    parser.add_argument('--source-manifest-sha256', required=True)
    parser.add_argument('--collector', choices=(COLLECTOR_SOURCE,), default=COLLECTOR_SOURCE)
    parser.add_argument('--card', type=int, choices=(2, 5), default=5)
    parser.add_argument('--image', default='quay.io/ascend/vllm-ascend:v0.18.0')
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args(argv)
    with args.out.open('x', encoding='utf-8') as stream:
        json.dump(cases(args.source, args.source_manifest_sha256, collector=args.collector,
                        card=args.card, image=args.image), stream, indent=2)


if __name__ == '__main__':
    main()
