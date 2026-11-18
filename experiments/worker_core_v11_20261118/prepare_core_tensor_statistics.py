"""Prepare one bounded, disposable Worker18 statistics case; never launch it."""
import argparse
from copy import deepcopy
import json
from pathlib import Path, PurePosixPath
import re

from experiments.c2c3_closure_20261117.prepare_tensor_worker_core import core_cases


def cases(source, manifest_sha256, *, collector='/home/batch-scale-dynamic-msprobe-20261023/python_packages',
          card=5, image='quay.io/ascend/vllm-ascend:v0.18.0'):
    if not re.fullmatch('[0-9a-f]{64}', manifest_sha256):
        raise ValueError('EXACT_SOURCE_MANIFEST_SHA256_REQUIRED')
    if (not collector.startswith('/') or '..' in PurePosixPath(collector).parts
            or any(c in collector for c in ('\n', '\r', '\0', ','))):
        raise ValueError('ABSOLUTE_REVIEWED_COLLECTOR_MOUNT_REQUIRED')
    row = deepcopy(core_cases(source, card, image)[0])
    row['id'] = 'public-worker18-statistics-three-windows'
    row['mounts'].append([collector, '/msprobe_packages', 'ro'])
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
        collector_control='Pinned recursive hooks; same-model cache rearm only after exact owned teardown',
        global_api_wrappers_restored=False, disposable_worker_required=True)
    row['review']['raw_evidence'] += ['/evidence/probe/raw_model_capture/collector_sidecar.json',
                                    '/evidence/probe/raw_model_capture/statistics']
    return [row]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True)
    parser.add_argument('--source-manifest-sha256', required=True)
    parser.add_argument('--collector', default='/home/batch-scale-dynamic-msprobe-20261023/python_packages')
    parser.add_argument('--card', type=int, choices=(2, 5), default=5)
    parser.add_argument('--image', default='quay.io/ascend/vllm-ascend:v0.18.0')
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args(argv)
    with args.out.open('x', encoding='utf-8') as stream:
        json.dump(cases(args.source, args.source_manifest_sha256, collector=args.collector,
                        card=args.card, image=args.image), stream, indent=2)


if __name__ == '__main__':
    main()
