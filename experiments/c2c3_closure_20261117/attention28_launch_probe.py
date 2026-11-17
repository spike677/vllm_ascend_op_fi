"""Run the existing fresh FIA experiment with independent final-args capture.

Root resource controller owns devices/locks. This entry point starts no container
and grants no public Worker/C2/C3 certificate. Raw statistics remain untouched.
"""
import argparse
import functools
import importlib.util
import json
import os
from pathlib import Path
import sys
import traceback

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO/'experiments/engineering_20261117'))
from fault_injection.native_adapters.attention28.launch import LaunchObserver, verify_launch, require, sha

spec = importlib.util.spec_from_file_location('attention28_fresh_implementation',
        REPO/'experiments/engineering_20261117/fresh_fia_qualification.py')
fresh = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fresh)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--completion', type=Path, required=True)
    p.add_argument('--completion-sha256', required=True)
    p.add_argument('--vendor', type=Path, required=True)
    p.add_argument('--public-schema', type=Path, required=True)
    p.add_argument('--schema-sha256', required=True)
    p.add_argument('--observer-receipt', type=Path, required=True)
    p.add_argument('--observer-receipt-sha256', required=True)
    p.add_argument('--phase', choices=('off', 'baseline', 'fault', 'off_recovery'), required=True)
    p.add_argument('--batch', choices=(1, 8), type=int, default=8)
    p.add_argument('--out', type=Path, default=Path('/evidence/run'))
    p.add_argument('--statistics', action='store_true')
    p.add_argument('--experimental-qualification', action='store_true')
    p.add_argument('--child', action='store_true', help=argparse.SUPPRESS)
    a = p.parse_args()
    verified = fresh.verify_build(a.completion, a.completion_sha256, a.vendor)
    require(a.experimental_qualification and sha(a.public_schema) == a.schema_sha256,
            'EXPLICIT_EXPERIMENT_AND_NEW_SCHEMA_REQUIRED')
    require(sha(a.observer_receipt) == a.observer_receipt_sha256, 'OBSERVER_BUILD_RECEIPT_HASH_DIFFERS')
    receipt = json.loads(a.observer_receipt.read_bytes())
    require(receipt.get('observer_compiled') is True and receipt.get('actual_npu_calls') == 0,
            'EXACT_IMAGE_OBSERVER_BUILD_REQUIRED')
    require(receipt.get('headers') and all(sha(x['path']) == x['sha256'] for x in receipt['headers']),
            'RUNTIME_SDK_DIFFERS_FROM_OBSERVER_BUILD')
    library = receipt['observer_library']
    require(sha(library['path']) == library['sha256'], 'NEW_OBSERVER_LIBRARY_HASH_DIFFERS')
    observer_dir = a.out.parent/'runtime'
    if not a.child:
        require(not os.environ.get('LD_PRELOAD') and not os.environ.get('ASCEND_CUSTOM_OPP_PATH')
                and not any(name.startswith(('DCAT_ATTN_', 'FAULT28_')) for name in os.environ),
                'CLEAN_FRESH_PROCESS_REQUIRED')
        require(not any(name in sys.modules for name in ('torch', 'torch_npu')), 'PARENT_MUST_NOT_IMPORT_FRAMEWORK')
        require(not observer_dir.exists() and not a.out.exists(), 'FRESH_EVIDENCE_REQUIRED')
        observer_dir.mkdir(parents=True)
        env = dict(os.environ, ACLNN_CACHE_LIMIT='0', TASK_QUEUE_ENABLE='0', PYTHONDONTWRITEBYTECODE='1',
                   ASCEND_CUSTOM_OPP_PATH=verified['vendor_root'], LD_PRELOAD=library['path'],
                   DCAT_ATTN_OBSERVER_DIR=str(observer_dir))
        env['LD_LIBRARY_PATH'] = ':'.join([str(Path(verified['vendor_root'])/'op_api/lib'), env.get('LD_LIBRARY_PATH', '')])
        os.execve(sys.executable, [sys.executable, str(Path(__file__).resolve()), *sys.argv[1:], '--child'], env)
    import torch
    import torch_npu
    observer = LaunchObserver(library['path'], library['sha256'])
    schema = json.loads(a.public_schema.read_bytes())
    original = torch_npu.npu_fused_infer_attention_score
    launches, failures = [], []
    @functools.wraps(original)
    def observed(*args, **kwargs):
        call_id = os.environ.get('FAULT28_FIA_GRAPH_TOKEN')
        require(call_id, 'ACTUAL_FRESH_HOST_SCOPE_REQUIRED')
        with observer.scope(call_id):
            result = original(*args, **kwargs)
            torch.npu.synchronize()
        rows = [json.loads(line) for line in Path(os.environ['FAULT28_FIA_GRAPH_LOG']).read_text().splitlines()]
        hosts = [row for row in rows if row.get('token') == call_id]
        require(len(hosts) == 1, 'ONE_ACTUAL_HOST_CALLBACK_REQUIRED')
        try:
            launches.append(verify_launch(observer_dir, call_id, kernel_sha256=verified['kernel_binary_sha256'],
                                          schema=schema, host=hosts[0]))
        except BaseException as error:
            failures.append(dict(call_id=call_id, error=repr(error), traceback=traceback.format_exc()))
            # Preserve the ordinary B/F/OFF result even when new independent
            # evidence is unavailable. Qualification still fails below.
        return result
    torch_npu.npu_fused_infer_attention_score = observed
    try:
        code = fresh.run_child(a, verified)
    finally:
        require(torch_npu.npu_fused_infer_attention_score is observed, 'OBSERVER_API_OWNERSHIP_CHANGED')
        torch_npu.npu_fused_infer_attention_score = original
    expected = 4 if a.phase == 'off_recovery' else 2
    qualified = code == 0 and len(launches) == expected and not failures
    report = dict(schema='demoncat.attention28.new_launch_probe.v1',
        status='FINAL_ARGUMENTS_AND_LAUNCHES_VERIFIED' if qualified else 'INDEPENDENT_LAUNCH_QUALIFICATION_INCOMPLETE',
        phase=a.phase, variant=verified['variant'], completion_sha256=a.completion_sha256,
        observer_receipt_sha256=a.observer_receipt_sha256, launches=launches, failures=failures,
        runtime_result_sha256=sha(a.out/'result.json'), native_launch_verified=qualified,
        public_worker_qualified=False, c2=False, c3=False, statistics='SEPARATE_RAW_COMPARATOR_REQUIRED')
    (a.out/'independent_launch.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    print(json.dumps({key: report[key] for key in ('status', 'phase', 'variant', 'native_launch_verified')}))
    return 0 if qualified else 3


if __name__ == '__main__':
    raise SystemExit(main())
