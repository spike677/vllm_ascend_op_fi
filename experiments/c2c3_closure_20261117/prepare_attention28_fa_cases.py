"""Prepare at most six root-queue cases; never starts containers or NPU work."""
import argparse
import hashlib
import json
from pathlib import Path
import shlex


def require(ok, reason):
    if not ok:raise RuntimeError(reason)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('source', 'build-root', 'observer', 'out'):p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--batch', type=int, choices=(1, 8), action='append')
    p.add_argument('--card', type=int, choices=(2, 5), default=5)
    p.add_argument('--selection-target', choices=('query_tile', 'key_group', 'joint'), default='joint')
    p.add_argument('--include-statistics', action='store_true')
    p.add_argument('--statistics-backend', choices=('original', 'cpu-fp64-v1'), default='original')
    p.add_argument('--cpu-preflight', type=Path)
    a = p.parse_args();batches = a.batch or [8]
    require(len(batches) == len(set(batches)) and 1 <= len(batches) <= 2, 'UNIQUE_BOUNDED_BATCHES_REQUIRED')
    completion = a.build_root / 'evidence/completion.json';built = json.loads(completion.read_bytes())
    controller = json.loads((a.build_root.parents[1] / 'controller.json').read_bytes())
    require(built.get('schema') == 'demoncat.fa.fresh_build.v1'
            and built.get('status') == 'CPU_BUILD_ONLY_REQUIRES_NATIVE_QUALIFICATION'
            and built.get('compiled_keys') == [22619644976] and built.get('historical_certificate_inherited') is False
            and controller.get('status') == 'BUILD_COMMAND_COMPLETED' and controller.get('actual_npu_calls') == 0,
            'MATCHED_NEW_FA_CPU_BUILD_REQUIRED')
    observer = a.observer / 'attention-preflight/result.json';observed = json.loads(observer.read_bytes())
    require(observed.get('observer_compiled') is True and observed.get('actual_npu_calls') == 0
            and observed['observer_source']['sha256'] == sha(a.source / 'fault_injection/native_adapters/attention28/launch_observer.cpp'),
            'EXACT_CURRENT_OBSERVER_SOURCE_BUILD_REQUIRED')
    if a.include_statistics and a.statistics_backend == 'cpu-fp64-v1':
        require(a.cpu_preflight is not None, 'MATCHED_CPU_STATISTICS_PREFLIGHT_REQUIRED')
        preflight = json.loads(a.cpu_preflight.read_bytes())
        require(preflight.get('status') == 'CPU_COLLECTOR_AND_WRITER_VERIFIED'
                and preflight.get('actual_npu_tensor_operations') == 0
                and preflight['backend']['adapter_sha256'] == sha(a.source / 'experiments/c2c3_closure_20261117/attention_cpu_statistics.py')
                and preflight['backend']['original_method_restored'] is True, 'MATCHED_CPU_STATISTICS_PREFLIGHT_REQUIRED')
    cases = []
    for batch in batches:
        for mode in (('sequence', 'baseline', 'fault') if a.include_statistics else ('sequence',)):
            statistics = mode != 'sequence'
            label = f'fa-fresh-b{batch}-{a.selection_target}-{mode}'
            argv = ['python3', '-B', '/driver/experiments/c2c3_closure_20261117/attention28_fa_launch_probe.py',
                '--build-root', '/build/run', '--completion-sha256', sha(completion),
                '--observer-receipt', '/build/attention-preflight/result.json', '--observer-receipt-sha256', sha(observer),
                '--mode', mode, '--batch', str(batch), '--selection-target', a.selection_target,
                '--out', '/evidence/run', '--experimental-qualification']
            if statistics:argv += ['--statistics', '--statistics-backend', a.statistics_backend]
            cases.append(dict(id=label, image=controller['image_id'], timeout=900,
                mounts=[[str(a.source), '/driver', 'ro'], [str(a.observer), '/build', 'ro'],
                        [str(a.build_root), '/build/run', 'ro']],
                env=dict(PYTHONPATH='/driver:/driver/src:/msprobe_packages', OMP_NUM_THREADS='4',
                         ACLNN_CACHE_LIMIT='0', TASK_QUEUE_ENABLE='0', PYTHONDONTWRITEBYTECODE='1'),
                command='source /usr/local/Ascend/cann-9.1.0/set_env.sh\nexec ' + shlex.join(argv),
                review=dict(experiment_id=label, provider='28.fa.legal_tiling', model='new fixed synthetic standalone FA; no Worker claim',
                    cann='9.1.0', soc='Ascend910B4/A2', physical_cards=[a.card], ranks=1,
                    worker='standalone synchronous torch_npu.npu_fusion_attention_v3', build_fingerprint=sha(completion),
                    expected_hits=4 if mode == 'sequence' else 3,
                    resource_budget=dict(timeout_seconds=900, memory_gib=32, cpus=8),
                    recovery_plan='Exact owned typed-plan cleanup each call; real unscoped OFF in sequence; synchronous completion.',
                    qualification='Actual registered ELF/function/FA-ABI/POD only; no old certificate or public C2/C3',
                    statistics_backend=a.statistics_backend if statistics else None,
                    cpu_statistics_preflight_sha256=sha(a.cpu_preflight) if statistics and a.cpu_preflight else None,
                    independent_e5='Separate fresh raw B/F dump/stack/construct/config; fixed rtol1e-5 atol1e-8; preserve null/NaN/mismatch.')))
    require(len(cases) <= 6, 'BOUNDED_ROOT_QUEUE_REQUIRED')
    with a.out.open('x', encoding='utf-8') as stream:
        json.dump(cases, stream, indent=2, allow_nan=False);stream.write('\n')
    print(json.dumps(dict(status='QUEUE_MANIFEST_ONLY_NOT_EXECUTED', cases=len(cases), out=str(a.out))))


if __name__ == '__main__':raise SystemExit(main())
