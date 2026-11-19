"""Blue-zone four-request Host acceptance; not inner-main-project integration."""
import argparse
import hashlib
import importlib
import json
from pathlib import Path
import signal
import time
import traceback

from demoncat.runtime.vllm_ascend.worker_extension.config import FaultConfig
from demoncat.runtime.vllm_ascend.worker_extension.host import HostAdapter, WorkerFaultClient
from demoncat.runtime.vllm_ascend.worker_extension.serving_context import read_enqueued_identity
from experiments.c2c3_postwrite_20261117.natural41_probe import PROMPT, actual_worker, output
from experiments.c2c3_postwrite_20261117.natural41_coverage_probe import actual_vllm_weights

ROOT = Path(__file__).resolve().parents[2]
SUCCESS = 'BLUE_ZONE_PUBLIC_OUTPUT39_SINGLE_TASK_ACCEPTED_NOT_MAIN_PROJECT'
CPU_SUCCESS = 'CPU_OUTPUT39_TASK_DRIVER_CONTRACT_ONLY'
PHASES = ('normal', 'fault', 'off', 'health')
IMAGE = 'sha256:35f9e569dbe3dc8db90cb361886d7889489efd8fa065860c8de319e13ab025d0'
MODULES = tuple(('fault_injection.'+name, 'fault_injection/'+name.replace('.', '/')+'.py')
    for name in ('deployments.output39_execution', 'deployments.postwrite', 'deployments.postwrite_boundaries',
                 'deployments.postwrite_npu_consumers', '_scoped_dispatch', 'tensor_callable', 'module_faults', 'core')) + tuple(
    ('demoncat.runtime.vllm_ascend.worker_extension.'+name,
     'src/demoncat/runtime/vllm_ascend/worker_extension/'+name.replace('.', '/')+'.py')
    for name in ('providers.output39_task', 'providers.fault39_output', 'providers._scope', 'output39_task',
                 'serving39_output', 'serving39_output_discovery', 'serving', 'serving41', 'serving_context', 'call_binding',
                 'gemm_lineage', 'host', 'config', 'runtime', 'identity', 'registry'))
HELPERS = ('experiments/worker_core_v11_20261118/output39_single_task_probe.py',
    'experiments/worker_core_v11_20261118/prepare_output39_single_task.py',
    'experiments/c2c3_postwrite_20261117/natural41_probe.py',
    'experiments/c2c3_postwrite_20261117/natural41_coverage_probe.py',
    'experiments/c2c3_postwrite_20261117/model_identity.py')
SOURCES = tuple(relative for _, relative in MODULES) + HELPERS


def require(value, reason):
    if not value: raise RuntimeError(reason)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b''): digest.update(chunk)
    return digest.hexdigest()


def write(path, value, *, fresh=False):
    raw = (json.dumps(value, sort_keys=True, indent=2, allow_nan=False)+'\n').encode()
    require(len(raw) <= 16*1024*1024, 'BOUNDED_SINGLE_TASK_METADATA_EXCEEDED')
    if fresh:
        with path.open('xb') as stream: stream.write(raw)
    else:
        temporary = path.with_suffix('.tmp')
        temporary.write_bytes(raw)
        temporary.replace(path)


def source_inputs(root, expected_manifest):
    root = Path(root).resolve(strict=True)
    manifest = root/'source_manifest.json'
    require(sha(manifest) == expected_manifest, 'FROZEN_SOURCE_MANIFEST_CHANGED')
    data = json.loads(manifest.read_text(encoding='utf-8'))
    entries = {row['path']: row for row in data['files']}
    require(len(entries) == len(data['files']), 'DUPLICATE_FROZEN_SOURCE_ENTRY')
    pins = {}
    for relative in SOURCES:
        path = (root/relative).resolve(strict=True)
        row = entries[relative]
        require(path.is_relative_to(root) and path.stat().st_size == row['bytes'] and sha(path) == row['sha256'],
                'FROZEN_TASK_SOURCE_CHANGED:'+relative)
        pins[relative] = row['sha256']
    return dict(manifest_sha256=expected_manifest, commit=data['base_commit'], pins=pins)


def setup(worker, identity, library, library_sha256, prompt, pins):
    """Trusted deployment only; no hooks, census, scope arming or model call."""
    from vllm.v1.worker.worker_base import WorkerBase
    from fault_injection.deployments.postwrite import NativePostwriteLibrary
    from demoncat.runtime.vllm_ascend.worker_extension.serving39_output import install_fault39_output_serving_binding
    worker = actual_worker(worker)
    require(isinstance(worker, WorkerBase) and worker.demoncat_worker_identity().get('worker') == identity,
            'ACTUAL_INITIALIZED_STANDARD_WORKER_CHANGED')
    loaded = {}
    for name, relative in MODULES:
        path = Path(importlib.import_module(name).__file__).resolve(strict=True)
        require(sha(path) == pins[relative], 'ACTUAL_LOADED_TASK_SOURCE_CHANGED:'+relative)
        loaded[relative] = dict(path=str(path), sha256=pins[relative])
    transport = NativePostwriteLibrary(library, library_sha256)
    bound = install_fault39_output_serving_binding(worker, transport=transport,
        prompt_token_ids=prompt, runtime_only=True)
    return dict(worker=identity, binding=bound, loaded_sources=loaded,
                no_model_forward=True, full_census=False, native_sha256=transport.sha256)


def remove_binding(worker, identity):
    from demoncat.runtime.vllm_ascend.worker_extension.serving39_output import remove_fault39_output_serving_binding
    worker = actual_worker(worker)
    require(worker.demoncat_worker_identity().get('worker') == identity, 'CLEANUP_WORKER_CHANGED')
    return remove_fault39_output_serving_binding(worker)


def verify_normal(row, *, cpu_contract=False):
    report = row['report']
    require(report['target']['operator'] == ('aten.mm.default' if cpu_contract else 'vllm.unquantized_gemm.default'),
            'EXPECTED_CURRENT_CUSTOM_GEMM_NOT_SELECTED')
    require(1 <= report['inspected_projections'] <= 16 and report['consumers']
        and report['consumers'][0]['original_returned'] is True
        and report['consumers'][0]['selected_elements_read'] == [0]
        and report['effect']['changed_elements'] == 0 and report['recovered_storage_exact'] is True,
        'CURRENT_NORMAL_CONSUMER_LOSSLESS_NOT_PROVEN')
    native = report.get('native_lease_receipt') or {}
    require(native.get('closed') is True and native.get('first_failure') is None
        and len(native.get('applications', [])) == 1 and native['applications'][0]['mode'] == 0
        and native['applications'][0]['return_code'] == 0, 'NORMAL_NATIVE_ORIGINAL_RECEIPT_REQUIRED')


def checks(rows):
    require([row['phase'] for row in rows] == list(PHASES), 'FOUR_ACTUAL_PHASES_REQUIRED')
    normal, fault, off, health = rows
    require(len({row['actual_internal_request_id'] for row in rows}) == 4
        and len({row['input_sha256'] for row in rows}) == 1, 'MATCHED_FRESH_INPUT_REQUESTS_REQUIRED')
    effect = fault['report']['effect']
    require(effect['changed_elements'] == effect['selected_elements'] == effect['finite_to_positive_inf'] == 1
        and effect['changed_only_selected'] is True and effect['finite_to_nan'] == effect['finite_to_negative_inf'] == 0,
        'ACTUAL_TYPED_SINGLE_TARGET_EFFECT_REQUIRED')
    for row in (normal, fault, off):
        require(row['report']['recovered_storage_exact'] is True and row['report']['consumers'],
                'ACTUAL_CONSUMER_AND_STORAGE_RESTORE_REQUIRED')
    native = fault['report'].get('native_lease_receipt') or {}
    require(native.get('closed') is True and native.get('first_failure') is None
        and len(native.get('applications', [])) == 1 and native['applications'][0]['mode'] == 1
        and native['applications'][0]['changed_elements'] == 1 and native['applications'][0]['return_code'] == 0
        and native.get('restore', {}).get('return_code') == 0 and native.get('release', {}).get('return_code') == 0,
        'FAULT_NATIVE_ORIGINAL_RECEIPT_REQUIRED')
    require(normal['hidden_logits_sha256'] == off['hidden_logits_sha256'] == health['hidden_logits_sha256'],
            'MATCHED_OFF_HEALTH_NUMERICAL_RECOVERY_REQUIRED')
    return dict(typed_effect=effect, actual_consumer=fault['report']['consumers'][0],
        normal_fault_logits_differ=normal['model_output_sha256'] != fault['model_output_sha256'],
        exact_off_hidden_logits=True, exact_health_hidden_logits=True,
        native_span_restored=True, generated_text_change_required=False)


def drive(llm, client, *, prompt, params, out, report, cpu_contract=False, stop_fault_budget=lambda: None):
    """Four real enqueues, with at most two recovery enqueues; never retry F."""
    pending, seen, last = [], [], None
    report.setdefault('requests', [])
    def save(): write(out/'result.json', report)
    def action(name, *args, expected='ACTION_COMPLETED'):
        nonlocal last
        last = getattr(client, name)(*args)
        require(last['status'] == expected, 'PUBLIC_TASK_ACTION_FAILED:'+name+':'+last['status'])
        return last.get('response', {}).get('provider_state') or {}
    def request(phase, *, recovery=False):
        nonlocal pending
        require(len(seen) < 6 and not llm.llm_engine.has_unfinished_requests(), 'BOUNDED_IDLE_REAL_QUEUE_REQUIRED')
        require(llm.reset_prefix_cache(reset_running_requests=False) is True, 'FRESH_PREFILL_CACHE_RESET_UNCONFIRMED')
        pending = list(llm.enqueue([prompt], params, use_tqdm=False))
        mapping = read_enqueued_identity(llm, pending, seen_internal_request_ids=seen)
        require(len(pending) == 1, 'ONE_ACTUAL_ENQUEUE_REQUIRED')
        seen.extend(pending)
        row = dict(phase=phase, mapping=mapping, completed=False)
        report.setdefault('recovery_requests' if recovery else 'requests', []).append(row)
        save()  # Preserve the actual mapping even if advance/wait/collect fails.
        state = action('advance', pending[0])
        require(state.get('next_phase') == phase and state.get('evidence', {}).get('runtime_only') is True,
                'ACTUAL_ORDINARY_TASK_PHASE_REQUIRED')
        result = llm.wait_for_completion(use_tqdm=False)
        require(len(result) == 1 and result[0].request_id == mapping['external_request_id'], 'ACTUAL_REQUEST_MAPPING_CHANGED')
        row['output'] = output(result[0])
        state = action('query', expected='QUERY_OK')
        actual = state['evidence']['actual_requests'][-1]
        require(actual['actual_internal_request_id'] == pending[0] and actual['phase'] == phase
            and state['failure'] is None and state['leases'] == 0, 'ACTUAL_TASK_REQUEST_INCOMPLETE')
        require(bool(state['native'].get('simulation')) == cpu_contract, 'ACTUAL_CPU_NPU_SCOPE_CHANGED')
        row.update(completed=True, observation=actual)
        pending = []
        save()
        if phase == 'normal': verify_normal(actual, cpu_contract=cpu_contract)
        return state
    try:
        try:
            for name in ('prepare', 'register', 'check', 'enable'): action(name)
            for phase in PHASES: request(phase)
            report['checks'] = checks([row['observation'] for row in report['requests']])
            outputs = [row['output'] for row in report['requests']]
            report['checks'].update(fault_tokens_differ=outputs[0]['tokens'] != outputs[1]['tokens'],
                fault_text_differs=outputs[0]['text'] != outputs[1]['text'],
                off_tokens_match=outputs[0]['tokens'] == outputs[2]['tokens'],
                health_tokens_match=outputs[0]['tokens'] == outputs[3]['tokens'])
            state = action('finish')
            require(state['evidence']['matched_off_and_health_confirmed'] and not any(
                state[key] for key in ('owned_hooks', 'leases', 'inflight', 'replay_pending')),
                'ORDINARY_TASK_EFFECT_RECOVERY_OR_RESOURCES_INCOMPLETE')
            report['resource_cleanup'] = dict(owned_hooks=0, native_leases=0, inflight=0,
                                              matched_numerical_recovery_separate=True)
            action('disable'); action('unregister'); action('detach', expected='DETACHED')
            report['status'] = CPU_SUCCESS if cpu_contract else SUCCESS
        except BaseException as error:
            report.setdefault('error', repr(error))
            report.setdefault('traceback', traceback.format_exc())
            report['status'] = 'REFUSED_OR_FAILED'
            stop_fault_budget()
            if pending:
                try:
                    llm.llm_engine.abort_request(pending, internal=True)
                    report['owned_request_abort'] = list(pending)
                    pending = []
                except BaseException as secondary:
                    report['abort_error'] = repr(secondary)
        if report['status'] == 'REFUSED_OR_FAILED' and last and last.get('ticket'):
            try:
                state = action('recover')
                if state.get('replay_pending') and state.get('next_phase') == 'off':
                    request('off', recovery=True); request('health', recovery=True)
                action('unregister'); action('detach', expected='DETACHED')
            except BaseException as error:
                report['recovery_error'] = repr(error)
    finally:
        try:
            final = client.snapshot()
            report['ownership_retained'], report['worker'] = final['ownership_retained'], final['worker']
            write(out/'host_receipt.json', final, fresh=True)
            report['host_receipt'] = dict(path='host_receipt.json', sha256=sha(out/'host_receipt.json'))
        except BaseException as error:
            report.setdefault('error', repr(error))
            report['receipt_error'] = repr(error)
            report['status'] = 'REFUSED_OR_FAILED'
        save()
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('model', 'library', 'sha256', 'build-fingerprint', 'source-manifest-sha256'):
        parser.add_argument('--'+name, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--memory', type=float, default=.35)
    parser.add_argument('--deadline-seconds', type=int, default=480)
    args = parser.parse_args(argv)
    require(1 <= args.deadline_seconds <= 1500 and .1 <= args.memory <= .8, 'BOUNDED_ACCEPTANCE_REQUEST_REQUIRED')
    require(args.out.is_absolute() and args.out.resolve().is_relative_to(Path('/evidence')), 'PRIVATE_ACCEPTANCE_OUTPUT_REQUIRED')
    args.out.mkdir(parents=True, exist_ok=False)
    report = dict(schema='demoncat.output39.single_task_probe.v1', status='STARTED',
        mechanism='39.output_proxy', experiment_only=True, main_project_integration='NOT_VERIFIED',
        C1=False, C2=False, C3=False, E5=False, qualification_granted=False,
        image_expected=IMAGE, native_sha256=args.sha256, build_fingerprint=args.build_fingerprint,
        build_fingerprint_authority='PREVIOUSLY_REVIEWED_BUILD_SELECTED_BY_ROOT',
        nominal_requests=4, max_total_requests_with_recovery=6, requests=[])
    started = time.monotonic()
    def deadline(*_): raise TimeoutError('OUTPUT39_SINGLE_TASK_SOFT_DEADLINE')
    def recovery_budget(): signal.alarm(120)
    prior = signal.signal(signal.SIGALRM, deadline)
    signal.alarm(args.deadline_seconds)
    try:
        report['source'] = source_inputs(ROOT, args.source_manifest_sha256)
        require(sha(args.library) == args.sha256, 'ACTUAL_NATIVE_LIBRARY_BYTES_CHANGED')
        from vllm import LLM, SamplingParams
        from demoncat.runtime.vllm_ascend.worker_extension.serving_context import require_host_driven_core
        report['weights'] = actual_vllm_weights(args.model)
        llm = LLM(model=args.model, dtype='bfloat16', load_format='safetensors', trust_remote_code=False,
            tensor_parallel_size=1, enforce_eager=True, max_model_len=128, max_num_seqs=1,
            max_num_batched_tokens=128, num_gpu_blocks_override=64, enable_prefix_caching=False,
            enable_chunked_prefill=False, async_scheduling=False, gpu_memory_utilization=args.memory,
            seed=42, disable_log_stats=True,
            worker_extension_cls='demoncat.runtime.vllm_ascend.controller.DemoncatWorkerExtension')
        require_host_driven_core(llm)
        require(actual_vllm_weights(args.model) == report['weights'], 'LOADED_MODEL_ARTIFACTS_CHANGED')
        tokens = llm.get_tokenizer().encode(PROMPT, add_special_tokens=True)
        require(2 <= len(tokens) <= 128, 'BOUNDED_ACTUAL_PROMPT_REQUIRED')
        report['prompt_token_ids'] = tokens
        def binding(identity, config):
            require(config == FaultConfig(39, 'output_proxy'), 'FIXED_MECHANISM_REQUIRED')
            replies = llm.collective_rpc(setup, args=(identity, args.library, args.sha256,
                tokens, report['source']['pins']), timeout=120)
            require(len(replies) == 1 and replies[0]['worker'] == identity, 'TP1_ACTUAL_SETUP_ROUTE_REQUIRED')
            report['binding'] = replies[0]
            return {}
        client = WorkerFaultClient(HostAdapter(llm, prepare_binding=binding, timeout=120), FaultConfig(39, 'output_proxy'))
        drive(llm, client, prompt=dict(prompt_token_ids=tokens),
            params=SamplingParams(temperature=0, max_tokens=1, ignore_eos=True),
            out=args.out, report=report, stop_fault_budget=recovery_budget)
        if report.get('ownership_retained') is False and report.get('binding'):
            report['binding_removal'] = llm.collective_rpc(remove_binding, args=(report['worker'],), timeout=120)
            require(len(report['binding_removal']) == 1 and report['binding_removal'][0]['status'] == 'SERVING_DEPLOYMENT_REMOVED',
                    'OWNED_DEPLOYMENT_REMOVAL_UNCONFIRMED')
    except BaseException as error:
        report.setdefault('error', repr(error))
        report.setdefault('traceback', traceback.format_exc())
        report['status'] = 'REFUSED_OR_FAILED'
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, prior)
        report['elapsed_seconds'] = time.monotonic()-started
        write(args.out/'result.json', report)
    print(json.dumps({key: report[key] for key in ('status', 'error') if key in report}))
    return 0 if report['status'] == SUCCESS else 2


if __name__ == '__main__':
    raise SystemExit(main())
