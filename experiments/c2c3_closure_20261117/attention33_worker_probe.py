"""Bounded real Qwen TND/page16 FAI B/F/OFF experiment for the root NPU queue.

This script starts no containers. It runs an ordinary eager, serial TP1 Worker
with a natural 1024-token prefill and first decode. No runner method, API
argument, page-table value, KV payload or output is rewritten.
"""
import argparse
import hashlib
import inspect
import json
import os
from pathlib import Path
import sys
import traceback
from attention33_contract import PHASES, RUNNER, execution_contract, verify_bridge_header

OWNER = '_demoncat_tnd33_actual_fai_probe'
PROMPT = 'A clear explanation uses definitions, a concrete example, and a careful check of the result. '


def require(value, message):
    if not value:
        raise RuntimeError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def worker_object(worker):
    from vllm.v1.worker.worker_base import WorkerWrapperBase, WorkerBase
    worker = worker.worker if isinstance(worker, WorkerWrapperBase) else worker
    require(isinstance(worker, WorkerBase), 'ACTUAL_STANDARD_WORKER_REQUIRED')
    return worker


def setup(worker, native_args, evidence_dir, statistics, boundary_args=None):
    worker = worker_object(worker)
    from demoncat.runtime.vllm_ascend.worker_extension import paged33_tnd_probe
    require(Path(inspect.getfile(paged33_tnd_probe)).resolve().is_relative_to('/installed'),
            'INSTALLED_WHEEL_REQUIRED')
    require(not hasattr(worker, OWNER), 'TND33_EXISTING_SESSION_OWNER')
    for owner in (worker, worker.model_runner):
        require('execute_model' not in vars(owner), 'CORE_METHOD_INSTANCE_OVERRIDE')
    session = paged33_tnd_probe.TNDWorkerProbeSession(worker, native_args=native_args,
        evidence_dir=evidence_dir, statistics=statistics, boundary_args=boundary_args)
    setattr(worker, OWNER, session)
    try:
        session.install()
    except BaseException:
        # Keep the object owned for explicit cleanup and failure evidence.
        raise
    return dict(candidate_count=len(session.candidates), source=session.source,
        module_path=inspect.getfile(paged33_tnd_probe), module_sha256=sha(inspect.getfile(paged33_tnd_probe)),
        stock_read_only=native_args is None, core_methods_replaced=False,
        first_boundary_diagnostic=boundary_args is not None)


def expect(worker, request_id, tokens):
    session = getattr(worker_object(worker), OWNER)
    session.expect({request_id: dict(logical_id=0, prompt=list(tokens))})
    return dict(actual_internal_request_id=request_id)


def finish(worker, collect):
    worker = worker_object(worker)
    session = getattr(worker, OWNER, None)
    if session is None:
        return dict(session_present=False, closed=True)
    result = dict(session_present=True, closed=False)
    try:
        if collect:
            result['capture'] = session.collect()
    except BaseException as error:
        result.update(error=repr(error), traceback=traceback.format_exc(), audit=session.report())
    finally:
        try:
            result['cleanup'] = session.close()
            result['closed'] = True
            delattr(worker, OWNER)
        except BaseException as error:
            result.update(cleanup_error=repr(error), cleanup_traceback=traceback.format_exc())
    return result


def rpc(llm, method, *args):
    rows = llm.collective_rpc(method, args=args, timeout=600)
    require(len(rows) == 1, 'TP1_REPLY_REQUIRED')
    return rows[0]


def response(row):
    return dict(request_id=row.request_id, prompt_token_ids=list(row.prompt_token_ids),
                outputs=[dict(tokens=list(x.token_ids), text=x.text, finish_reason=x.finish_reason) for x in row.outputs])


def run(args, bundle):
    # Imports occur only in the exec'd child, after private vendor/preload setup.
    from demoncat.runtime.vllm_ascend.worker_extension.serving_context import require_host_driven_core, read_enqueued_identity
    from demoncat.runtime.vllm_ascend.worker_extension.paged33_tnd_probe import payload_identity
    report = dict(schema='demoncat.paged33.actual_fai_worker_run.v1', status='STARTED',
        scope='STANDARD_QWEN_TP1_NATURAL_TND_PAGE16_PREFILL_AND_FIRST_DECODE',
        phase=args.phase, phases=[], statistics=args.statistics,
        public_worker_qualified=False, c2=False, c3=False,
        statistical_first_difference='SEPARATE_INDEPENDENT_RAW_COMPARATOR_REQUIRED',
        completion_sha256=args.completion_sha256 if bundle else None,
        schema_sha256=args.schema_sha256 if bundle else None,
        observer_receipt_sha256=args.observer_receipt_sha256 if bundle or args.first_boundary_diagnostic else None)
    if args.first_boundary_diagnostic:
        report.update(schema='demoncat.paged33.first_boundary_worker_run.v1',
            diagnostic_only=True, bridge_compatible=False, fault_authorized=False,
            first_boundary_readbacks='IDENTICAL_CAPTURE_CODE_IN_STOCK_AND_PRIVATE_NO_STATISTICS')
    target = args.out/'result.json'
    def save():
        target.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    save()
    llm, installed, seen = None, False, []
    try:
        report['execution_contract'] = execution_contract(args.model, '/installed', __file__, args.memory, os.environ)
        if args.phase == 'fault':
            verify_bridge_header(args.bridge, args.bridge_sha256, report['execution_contract'],
                args.completion_sha256, Path(__file__).with_name('attention33_bridge.py'))
            report['baseline_bridge_sha256'] = args.bridge_sha256
        save()
        coexistence = None
        if bundle is not None:
            from demoncat.runtime.vllm_ascend.worker_extension.paged33_loading import bootstrap
            coexistence = bootstrap(bundle)
            report['library_bootstrap'] = coexistence.report();save()
        from vllm import LLM, SamplingParams
        if coexistence is not None:
            from vllm.platforms import current_platform
            report['library_coexistence'] = coexistence.prepare_platform(current_platform);save()
        llm = LLM(model=args.model, gpu_memory_utilization=args.memory, **RUNNER)
        if coexistence is not None:
            report['library_coexistence_after_standard_worker_initialization'] = coexistence.verify();save()
        require_host_driven_core(llm)
        tokens = llm.get_tokenizer().encode(PROMPT*128, add_special_tokens=True)[:1024]
        require(len(tokens) == 1024, 'BOUNDED_LEGITIMATE_1024_TOKEN_PREFILL_REQUIRED')
        prompt = dict(prompt_token_ids=tokens)
        params = SamplingParams(temperature=0, max_tokens=2, ignore_eos=True)
        report['prompt'] = dict(token_count=len(tokens), token_sha256=hashlib.sha256(
            json.dumps(tokens, separators=(',', ':')).encode()).hexdigest(), natural_text=PROMPT)
        before, = llm.generate([prompt], params, use_tqdm=False)
        report['warm_healthy'] = response(before);save()
        for phase in PHASES:
            require(not llm.llm_engine.has_unfinished_requests(), 'UNFINISHED_OR_UNRELATED_REQUEST')
            require(llm.reset_prefix_cache(reset_running_requests=False) is True, 'PREFIX_RESET_NOT_CONFIRMED')
            native = None
            if bundle is not None and phase != 'healthy_after':
                arm = args.phase if phase == 'measured' else 'off'
                native = dict(completion=str(args.completion), completion_sha256=args.completion_sha256,
                    vendor=str(args.vendor), schema=str(args.public_schema), schema_sha256=args.schema_sha256,
                    observer_receipt=str(args.observer_receipt), observer_receipt_sha256=args.observer_receipt_sha256,
                    arm=arm, experimental_qualification=True, max_calls=28)
            row = dict(phase=phase);report['phases'].append(row);save()
            installed = True  # setup may install a partial owned session
            boundary = (dict(path=str(args.observer_receipt), expected_sha256=args.observer_receipt_sha256,
                            capture_source_sha256=sha(Path(__file__).resolve().parents[2]/
                                'src/demoncat/runtime/vllm_ascend/worker_extension/paged33_boundary_probe.py'))
                        if args.first_boundary_diagnostic else None)
            row['setup'] = rpc(llm, setup, native, str(args.out/phase), args.statistics and phase == 'measured', boundary)
            require(row['setup']['candidate_count'] == 28, 'COMPLETE_QWEN_28_LAYER_CENSUS_REQUIRED')
            ids = llm.enqueue([prompt], params, use_tqdm=False)
            mapping = read_enqueued_identity(llm, ids, seen_internal_request_ids=seen)
            seen.extend(ids)
            row['mapping'] = mapping
            row['expected'] = rpc(llm, expect, ids[0], tokens);save()
            outputs = llm.wait_for_completion(use_tqdm=False)
            require(len(outputs) == 1, 'ONE_COMPLETED_REQUEST_REQUIRED')
            row['response'] = response(outputs[0])
            require(row['response']['request_id'] == mapping['external_request_id'], 'ACTUAL_REQUEST_MAPPING_DIFFERS')
            row['finish'] = rpc(llm, finish, True)
            installed = not row['finish']['closed'];save()
            require(not installed and 'capture' in row['finish'] and 'error' not in row['finish'],
                    'NATURAL_WORKER_CAPTURE_OR_CLEANUP_INCOMPLETE')
        values = {row['phase']: row for row in report['phases']}
        baseline = payload_identity(values['baseline_before']['finish']['capture'])
        exact = {phase: payload_identity(values[phase]['finish']['capture']) == baseline for phase in PHASES}
        report['exact_complete_payload_identity_vs_baseline'] = exact
        require(all(exact[phase] for phase in ('baseline_repeat', 'off_after', 'healthy_after')),
                'COMPLETE_BB_OFF_OR_HEALTHY_RECOVERY_DIFFERS')
        if args.phase in ('stock', 'baseline', 'off'):
            require(exact['measured'], 'BASELINE_MEASURED_COMPLETE_PAYLOAD_DIFFERS')
        for phase in ('baseline_before', 'baseline_repeat', 'off_after', 'healthy_after'):
            require(values[phase]['response']['outputs'] == report['warm_healthy']['outputs'],
                    'HEALTHY_GENERATION_DIFFERS')
        require(not any(name.startswith('DCAT33_TND_') for name in os.environ), 'NATIVE_SCOPE_REMAINS_AFTER_OFF')
        clean, = llm.generate([prompt], params, use_tqdm=False)
        report['healthy_after_owned_observers_removed'] = response(clean)
        require(report['healthy_after_owned_observers_removed']['outputs'] == report['warm_healthy']['outputs'],
                'POST_CLEAN_GENERATION_DIFFERS')
        report['status'] = ('NATURAL_STOCK_WORKER_COMPLETE' if bundle is None else
                            'NATIVE_WORKER_ATTEMPT_AND_OFF_COMPLETE_REQUIRES_EVIDENCE_REVIEW')
        if args.first_boundary_diagnostic:
            report['status'] = 'FIRST_BOUNDARY_DIAGNOSTIC_COMPLETE_REQUIRES_INDEPENDENT_REVIEW'
    except BaseException as error:
        report.update(status='FAILED_WITH_EVIDENCE', error=repr(error), traceback=traceback.format_exc())
        if llm is not None and seen:
            try:
                llm.llm_engine.abort_request(seen, internal=True)
                report['owned_request_abort'] = dict(ids=seen, unfinished=llm.llm_engine.has_unfinished_requests())
            except BaseException as abort_error:
                report['abort_error'] = repr(abort_error)
    finally:
        if llm is not None and installed:
            try:
                report['failure_cleanup'] = rpc(llm, finish, False)
            except BaseException as cleanup_error:
                report['cleanup_error'] = repr(cleanup_error)
        save()
    print(json.dumps({key: report[key] for key in ('status', 'phase', 'c2', 'c3')}))
    return 0 if report['status'] != 'FAILED_WITH_EVIDENCE' else 2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=('stock', 'baseline', 'fault', 'off'), required=True)
    parser.add_argument('--model', default='/model')
    parser.add_argument('--memory', type=float, default=.35)
    parser.add_argument('--completion', type=Path)
    parser.add_argument('--completion-sha256')
    parser.add_argument('--vendor', type=Path)
    parser.add_argument('--public-schema', type=Path)
    parser.add_argument('--schema-sha256')
    parser.add_argument('--observer-receipt', type=Path)
    parser.add_argument('--observer-receipt-sha256')
    parser.add_argument('--bridge', type=Path)
    parser.add_argument('--bridge-sha256')
    parser.add_argument('--out', type=Path, default=Path('/evidence/run'))
    parser.add_argument('--statistics', action='store_true')
    parser.add_argument('--experimental-qualification', action='store_true')
    parser.add_argument('--first-boundary-diagnostic', action='store_true')
    parser.add_argument('--child', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args()
    require(args.experimental_qualification and str(args.out).startswith('/evidence/'),
            'EXPLICIT_PRIVATE_EXPERIMENT_REQUIRED')
    require(not args.first_boundary_diagnostic or (args.phase in ('stock', 'baseline') and not args.statistics
            and args.bridge is None), 'FIRST_BOUNDARY_DIAGNOSTIC_NEVER_AUTHORIZES_FAULT_OR_STATISTICS')
    bundle, library = None, None
    if args.phase != 'stock':
        from demoncat.runtime.vllm_ascend.worker_extension.paged33_tnd import verify_bundle
        bundle = verify_bundle(args.completion, args.completion_sha256, args.vendor)
        require(sha(args.public_schema) == args.schema_sha256 == bundle['completion']['fai_public_schema_sha256'],
                'EXACT_NEW_FAI_SCHEMA_REQUIRED')
    if bundle is not None or args.first_boundary_diagnostic:
        require(sha(args.observer_receipt) == args.observer_receipt_sha256, 'EXACT_OBSERVER_RECEIPT_REQUIRED')
        receipt = json.loads(args.observer_receipt.read_bytes())
        require(receipt.get('observer_compiled') and receipt.get('actual_npu_calls') == 0
                and receipt.get('observer_memory_selftest', {}).get('exit_code') == 0,
                'CPU_OBSERVER_MEMORY_SELFTEST_REQUIRED')
        library = receipt['observer_library']
        require(sha(library['path']) == library['sha256'], 'OBSERVER_LIBRARY_CHANGED')
    if not args.child:
        require(not os.environ.get('LD_PRELOAD') and not os.environ.get('ASCEND_CUSTOM_OPP_PATH')
                and not any(x.startswith(('DCAT33_TND_', 'DCAT_ATTN_')) for x in os.environ), 'CLEAN_NATIVE_ENV_REQUIRED')
        require(not any(x in sys.modules for x in ('torch', 'torch_npu')), 'PARENT_MUST_NOT_IMPORT_FRAMEWORK')
        require(not args.out.exists(), 'FRESH_EVIDENCE_DIRECTORY_REQUIRED')
        args.out.mkdir(parents=True)
        env = dict(os.environ, ACLNN_CACHE_LIMIT='0', TASK_QUEUE_ENABLE='0', PYTHONDONTWRITEBYTECODE='1',
                   VLLM_ENABLE_V1_MULTIPROCESSING='0')
        if library:
            runtime = args.out/'runtime';runtime.mkdir()
            env.update(LD_PRELOAD=library['path'], DCAT_ATTN_OBSERVER_DIR=str(runtime))
        if bundle:
            env['ASCEND_CUSTOM_OPP_PATH'] = bundle['vendor']
        os.execve(sys.executable, [sys.executable, str(Path(__file__).resolve()), *sys.argv[1:], '--child'], env)
    return run(args, bundle)


if __name__ == '__main__':
    raise SystemExit(main())
