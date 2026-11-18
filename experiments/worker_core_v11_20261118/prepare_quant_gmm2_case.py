"""Prepare one zero-F WeightNz observation case; never launch a container."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from fault_injection.quant_gmm2_native import validate_extension
from experiments.worker_core_v11_20261118.prepare_quant_container_case import adapt as container_case


def adapt(case, *, trace, trace_sha, quant_sha, support, support_sha, native_source_commit):
    if not str(support).startswith('/') or any(c in str(support) for c in ('\n', '\r', '\0', ',')):
        raise ValueError('ABSOLUTE_READONLY_PROBE_SUPPORT_REQUIRED')
    case = container_case(case, trace, trace_sha, quant_sha)
    name = 'quant-weightnz-current-slots-one-decode'
    case['id'] = name
    case['mounts'].append([str(support), '/probe_quant_containers.py', 'ro'])
    case['env'].update(VLLM_ENABLE_V1_MULTIPROCESSING='0', TASK_QUEUE_ENABLE='0')
    case['command'] += ' --support-sha256 ' + support_sha
    review = case['review']
    review.update(experiment_id=name,
        provider='quant read-only current WeightNz GWS/execute and container lifetimes; zero F; no row authority',
        worker='standard DemoncatWorkerExtension; actual same-process InprocClient; TP1 B8 first decode; complete 135 census',
        native_compile_source_commit=native_source_commit, support_source_sha256=support_sha,
        recovery_plan='Before importing/constructing LLM, attach by RTLD_NOLOAD and resolve the five exact stock functions '
            'already loaded through explicit startup dependencies. Refuse before LLM on missing/foreign owner. Hold the '
            'same Python object in the installed binding module; actual Worker RPC verifies PID, /proc start, thread, '
            'Worker/runner/model and observed logical device. Different-process workers are outside this scope. Observe '
            'one actual B8 decode and all 135 Native/public quant projections. Every real WeightNz prepare/execute runs '
            'exactly once with unchanged arguments and return; observation errors preserve originals and refuse evidence. '
            'Record all 23 actual typed slots, current creator registrations, SDK count getters, group-list descriptor, '
            'actual device/executor/workspace/stream and 26 independent GWS/execute pairs. No opaque container layout, '
            'floating payload copies, reference helpers, F or routed row binding. Keep first error and immutable raw '
            'snapshot even if stop/removal fails. Never execute or destroy a pending model executor to clear counters. '
            'Remove owned hooks; stop both journals; release GMM journal before creator journal only after all observed '
            'lifetimes retire. Failure retains owner/evidence for retry. No claim that execute return proves device '
            'drain; no observer device allocations. Require same-Worker healthy tokens and zero owned resources. '
            '10/135 historical row proofs are not enlarged or inherited as a new qualification; full denominator stays.')
    review['resource_budget'].update(gmm2_calls=128, gmm2_events=256, gmm2_pending_pairs=32,
        gmm2_record_bytes=1024*1024, expected_natural_gmm2_pairs=26)
    return case


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('installed', 'quant', 'trace', 'source-commit', 'image-id'):
        p.add_argument('--' + name, required=True)
    p.add_argument('--model', default='/home/haoran/models/DeepSeek-V2-Lite-W8A8')
    p.add_argument('--card', type=int, choices=(2, 5), default=5)
    p.add_argument('--probe', default=str(Path(__file__).with_name('probe_quant_gmm2.py')))
    p.add_argument('--support', default=str(Path(__file__).with_name('probe_quant_containers.py')))
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    trace = Path(a.trace).resolve(strict=True)
    trace_sha = hashlib.sha256((trace / 'build.json').read_bytes()).hexdigest()
    _, build, _ = validate_extension(trace, trace_sha)
    if build['image_id'] != a.image_id:
        raise RuntimeError('EXACT_SDK_AND_RUNTIME_IMAGE_REQUIRED')
    support = Path(a.support).resolve(strict=True)
    support_sha = hashlib.sha256(support.read_bytes()).hexdigest()
    quant_sha = hashlib.sha256((Path(a.quant) / 'build.json').read_bytes()).hexdigest()
    with tempfile.TemporaryDirectory(prefix='quant-gmm2-case-') as directory:
        base = Path(directory) / 'base.json'
        subprocess.run([sys.executable, str(ROOT / 'experiments/c2c3_closure_20261117/providers_quant_prepare_inventory_case.py'),
            '--installed', a.installed, '--quant', a.quant, '--probe', a.probe, '--source-commit', a.source_commit,
            '--image-id', a.image_id, '--model', a.model, '--card', str(a.card), '--group-abi', '--out', str(base)], check=True)
        rows = json.loads(base.read_bytes())
    if len(rows) != 1:
        raise RuntimeError('ONE_CENSUS_BASE_CASE_REQUIRED')
    value = adapt(rows[0], trace=trace, trace_sha=trace_sha, quant_sha=quant_sha, support=support,
                  support_sha=support_sha, native_source_commit=build['source_commit'])
    with a.out.open('x') as stream:
        json.dump([value], stream, indent=2)


if __name__ == '__main__':
    main()
