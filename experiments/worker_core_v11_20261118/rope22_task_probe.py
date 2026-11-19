"""Blue-zone four-request ROPE22 acceptance; never an ordinary runtime dependency."""
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
from experiments.worker_core_v11_20261118.output39_single_task_probe import require, sha, write, IMAGE
from experiments.c2c3_postwrite_20261117.natural41_probe import PROMPT, actual_worker, output
from experiments.c2c3_postwrite_20261117.model_identity import vllm_safetensors_identity as actual_vllm_weights

ROOT = Path(__file__).resolve().parents[2]
SUCCESS = 'BLUE_ZONE_PUBLIC_ROPE22_SINGLE_TASK_ACCEPTED_NOT_MAIN_PROJECT'
CPU_SUCCESS = 'CPU_ROPE22_TASK_DRIVER_CONTRACT_ONLY'
PHASES = ('normal', 'fault', 'off', 'health')
MODULES = tuple(('fault_injection.'+name, 'fault_injection/'+name+'.py') for name in (
    'prerequisites', 'rope_census', 'rope_fused22', 'rope_triton_trace', 'rope_sites', 'rmsnorm_return', 'actions', 'effect_summary', '_scoped_dispatch', 'tensor_callable',
    'tensor_observer', 'module_faults', 'core', 'catalog', 'delivery_scope')) + tuple(
    ('demoncat.runtime.vllm_ascend.worker_extension.'+name,
     'src/demoncat/runtime/vllm_ascend/worker_extension/'+name.replace('.', '/')+'.py')
    for name in ('deployment_acceptance','ordinary_evidence','rope22_task', 'ordinary_admission', 'prerequisites', 'providers.rms18_task', 'providers.tensor_serving', 'providers._scope',
        'providers.tensor_modules', 'rms18_request', 'serving_tensor_binding', 'serving',
        'serving_context', 'call_binding', 'host', 'config', 'runtime', 'identity', 'registry'))
SOURCES = tuple(relative for _, relative in MODULES) + (
    'experiments/worker_core_v11_20261118/rope22_task_probe.py',
    'experiments/worker_core_v11_20261118/prepare_rope22_single_task.py',
    'experiments/worker_core_v11_20261118/output39_single_task_probe.py',
    'experiments/c2c3_closure_20261117/queue_contract.py',
    'experiments/c2c3_postwrite_20261117/natural41_probe.py',
    'experiments/c2c3_postwrite_20261117/natural41_coverage_probe.py',
    'experiments/c2c3_postwrite_20261117/model_identity.py')


def source_inputs(root, expected_manifest):
    root = Path(root).resolve(strict=True)
    manifest = root/'source_manifest.json'
    require(sha(manifest) == expected_manifest, 'FROZEN_SOURCE_MANIFEST_CHANGED')
    data = json.loads(manifest.read_text(encoding='utf-8'))
    entries = {row['path']: row for row in data['files']}
    require(len(entries) == len(data['files']), 'DUPLICATE_FROZEN_SOURCE_ENTRY')
    pins = {}
    for relative in SOURCES:
        path = (root/relative).resolve(strict=True); row = entries[relative]
        require(path.is_relative_to(root) and path.stat().st_size == row['bytes'] and sha(path) == row['sha256'],
                'FROZEN_TASK_SOURCE_CHANGED:'+relative)
        pins[relative] = row['sha256']
    return dict(manifest_sha256=expected_manifest, commit=data['base_commit'], pins=pins)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False, separators=(',', ':')).encode()).hexdigest()


def setup(worker, identity, prompt, weights, pins, manifest_sha256):
    from vllm.v1.worker.worker_base import WorkerBase
    from demoncat.runtime.vllm_ascend.worker_extension.serving_tensor_binding import install_rope22_task_binding
    import torch
    import torch_npu
    import vllm
    worker = actual_worker(worker)
    require(isinstance(worker, WorkerBase) and worker.demoncat_worker_identity().get('worker') == identity,
            'ACTUAL_INITIALIZED_STANDARD_WORKER_CHANGED')
    loaded = {}
    for name, relative in MODULES:
        path = Path(importlib.import_module(name).__file__).resolve(strict=True)
        require(sha(path) == pins[relative], 'ACTUAL_LOADED_TASK_SOURCE_CHANGED:'+relative)
        loaded[relative] = dict(path=str(path), sha256=pins[relative])
    versions = dict(torch=torch.__version__, torch_npu=torch_npu.__version__, vllm=vllm.__version__)
    bound = install_rope22_task_binding(worker, prompt_token_ids=prompt, weight_manifest=weights,
        workload_sha256=digest(prompt), environment_sha256=digest(dict(versions=versions, sources=pins)))
    from demoncat.runtime.vllm_ascend.worker_extension.ordinary_admission import ordinary_route_report, require_ordinary_route
    from demoncat.runtime.vllm_ascend.worker_extension.deployment_acceptance import install_blue_acceptance_permit, remove_blue_acceptance_permit
    from demoncat.runtime.vllm_ascend.worker_extension.serving_tensor_binding import remove_tensor_serving_binding
    try:
        default=ordinary_route_report('22.wrong_row',worker,worker.model_runner.model)
        if not default['deployment_acceptance']['accepted']:
            install_blue_acceptance_permit(worker,mechanisms=['22.wrong_row'],attempt='ordinary22-two-scopes',
                source_manifest_sha256=manifest_sha256)
        blue=require_ordinary_route('22.wrong_row',worker,worker.model_runner.model)
    except BaseException:
        remove_tensor_serving_binding(worker,'22.wrong_row')
        if hasattr(worker,'_demoncat_blue_acceptance_permit'):remove_blue_acceptance_permit(worker)
        raise
    return dict(worker=identity, binding=bound, loaded_sources=loaded, versions=versions,
                default_acceptance=default, blue_acceptance=blue,
                no_model_forward=True, full_census=False, new_native_build=False)


def remove_binding(worker, identity):
    from demoncat.runtime.vllm_ascend.worker_extension.serving_tensor_binding import remove_tensor_serving_binding
    worker = actual_worker(worker)
    require(worker.demoncat_worker_identity().get('worker') == identity, 'CLEANUP_WORKER_CHANGED')
    result=remove_tensor_serving_binding(worker, '22.wrong_row')
    from demoncat.runtime.vllm_ascend.worker_extension.deployment_acceptance import remove_blue_acceptance_permit
    if hasattr(worker,'_demoncat_blue_acceptance_permit'):remove_blue_acceptance_permit(worker)
    return result


def verify_normal(row, *, cpu_contract=False):
    report = row['mechanism']
    require(report['execution_devices'] == (['cpu'] if cpu_contract else ['npu']), 'ACTUAL_CPU_NPU_SCOPE_CHANGED')
    events = report['events']
    require(len(events) == 1 and events[0]['applied'] is False and events[0]['output_changed'] is False
        and events[0]['original_execution_verified'] and events[0]['return_readback_exact']
        and len(events[0]['consumers']) == 1 and events[0]['consumers'][0]['returned'],
        'ONE_NORMAL_ORIGINAL_RETURN_AND_REAL_CONSUMER_REQUIRED')


def checks(rows):
    require([row['phase'] for row in rows] == list(PHASES), 'FOUR_ACTUAL_PHASES_REQUIRED')
    normal, fault, off, health = rows
    require(len({row['request_id'] for row in rows}) == 4 and all(row['input'] == normal['input'] for row in rows),
            'MATCHED_FRESH_INPUT_REQUESTS_REQUIRED')
    first = normal['mechanism']['events'][0]
    events = fault['mechanism']['events']
    require(len(events) == 1 and events[0]['applied'] is True and events[0]['same_complete_inputs_as_healthy'] is True
        and events[0]['return_readback_exact'] is True and events[0]['output_changed'] is True
        and events[0]['actual'] == first['actual'] and len(events[0]['consumers']) == 1,
        'ONE_ACTUAL_FUSED_WRONG_ROW_AND_CONSUMER_REQUIRED')
    require(off['mechanism'] is None and health['mechanism'] is None
        and normal['output'] == off['output'] == health['output'], 'EXACT_OFF_HEALTH_NO_ROPE_OBSERVER_REQUIRED')
    reader=events[0]['independent_final_reader']
    require(reader['independent_position_transform']=='last_from_first'
        and events[0]['non_target_output_unchanged'], 'ACTUAL_FINAL_WRONG_ROW_AND_NON_TARGET_REQUIRED')
    effect=dict(comparison='MATCHED_FULL_QK_INPUTS_AND_IMMUTABLE_CACHE',
        complete_qk_changed=events[0]['changed_from_healthy'],non_target_unchanged=True,
        normal_qk=first['actual_qk_after'],fault_qk=events[0]['actual_qk_after'],
        correct_source_row=events[0]['correct_source_row'],replacement_source_row=events[0]['replacement_source_row'])
    return dict(typed_effect=effect, actual_consumer=events[0]['consumers'][0],
        selected_path=first['path'], normal_fault_logits_differ=normal['output']['logits'] != fault['output']['logits'],
        normal_fault_hidden_differ=normal['output']['hidden'] != fault['output']['hidden'],
        normal_fault_observed_returns=[1, 1], off_health_rope_observers=[0, 0],
        exact_off_hidden_logits=True, exact_health_hidden_logits=True, generated_text_change_required=False)


def drive(llm, client, *, prompt, params, out, report, cpu_contract=False, stop_fault_budget=lambda: None):
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
        report.setdefault('recovery_requests' if recovery else 'requests', []).append(row); save()
        state = action('advance', pending[0])
        require(state['serving']['next_phase'] == phase and state['ordinary_task'] is True,
                'ACTUAL_ORDINARY_TASK_PHASE_REQUIRED')
        result = llm.wait_for_completion(use_tqdm=False)
        require(len(result) == 1 and result[0].request_id == mapping['external_request_id'], 'ACTUAL_REQUEST_MAPPING_CHANGED')
        row['output'] = output(result[0])
        state = action('query', expected='QUERY_OK')
        actual = state['serving']['actual_requests'][-1]
        require(actual['request_id'] == pending[0] and actual['phase'] == phase
            and state['serving']['failure'] is None and state['leases'] == state['inflight'] == 0,
            'ACTUAL_TASK_REQUEST_INCOMPLETE')
        require(state['simulation'] is cpu_contract, 'ACTUAL_CPU_NPU_SCOPE_CHANGED')
        row.update(completed=True, observation=actual); pending = []; save()
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
            require(state['serving']['recovered'] and not any(state[k] for k in ('owned_hooks', 'leases', 'inflight', 'replay_pending')),
                    'ORDINARY_TASK_RECOVERY_OR_RESOURCES_INCOMPLETE')
            report['resource_cleanup'] = dict(owned_hooks=0, leases=0, inflight=0, numerical_recovery_separate=True)
            action('disable'); action('unregister'); action('detach', expected='DETACHED')
            report['status'] = CPU_SUCCESS if cpu_contract else SUCCESS
        except BaseException as error:
            report.setdefault('error', repr(error)); report.setdefault('traceback', traceback.format_exc())
            report['status'] = 'REFUSED_OR_FAILED'; stop_fault_budget()
            if pending:
                try:
                    llm.llm_engine.abort_request(pending, internal=True)
                    report['owned_request_abort'] = list(pending); pending = []
                except BaseException as secondary: report['abort_error'] = repr(secondary)
        if report['status'] == 'REFUSED_OR_FAILED' and last and last.get('ticket'):
            try:
                state = action('recover')
                if state.get('replay_pending') and state['serving']['next_phase'] == 'off':
                    request('off', recovery=True); request('health', recovery=True)
                action('unregister'); action('detach', expected='DETACHED')
            except BaseException as error: report['recovery_error'] = repr(error)
    finally:
        try:
            final = client.snapshot()
            report['ownership_retained'], report['worker'] = final['ownership_retained'], final['worker']
            write(out/'host_receipt.json', final, fresh=True)
            report['host_receipt'] = dict(path='host_receipt.json', sha256=sha(out/'host_receipt.json'))
        except BaseException as error:
            report.setdefault('error', repr(error)); report['receipt_error'] = repr(error)
            report['status'] = 'REFUSED_OR_FAILED'
        save()
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True)
    parser.add_argument('--source-manifest-sha256', required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--memory', type=float, default=.65)
    parser.add_argument('--deadline-seconds', type=int, default=480)
    args = parser.parse_args(argv)
    require(1 <= args.deadline_seconds <= 1500 and .1 <= args.memory <= .8, 'BOUNDED_ACCEPTANCE_REQUEST_REQUIRED')
    require(args.out.is_absolute() and args.out.resolve().is_relative_to(Path('/evidence')), 'PRIVATE_ACCEPTANCE_OUTPUT_REQUIRED')
    args.out.mkdir(parents=True, exist_ok=False)
    report = dict(schema='demoncat.rope22.single_task_probe.v1', status='STARTED', mechanism='22.wrong_row',
        experiment_only=True, main_project_integration='NOT_VERIFIED', C1=False, C2=False, C3=False, E5=False,
        qualification_granted=False, image_expected=IMAGE, new_native_build=False,
        nominal_requests=8, max_total_requests_with_recovery=12, requests=[])
    started = time.monotonic()
    def deadline(*_): raise TimeoutError('ROPE22_SINGLE_TASK_SOFT_DEADLINE')
    prior = signal.signal(signal.SIGALRM, deadline); signal.alarm(args.deadline_seconds)
    try:
        report['source'] = source_inputs(ROOT, args.source_manifest_sha256)
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
            require(config == FaultConfig(22, 'wrong_row'), 'FIXED_MECHANISM_REQUIRED')
            replies = llm.collective_rpc(setup, args=(identity, tokens, report['weights'], report['source']['pins'],report['source']['manifest_sha256']), timeout=120)
            require(len(replies) == 1 and replies[0]['worker'] == identity, 'TP1_ACTUAL_SETUP_ROUTE_REQUIRED')
            report['binding'] = replies[0]
            return {}
        report['tasks'] = []
        for index in range(2):
            task_out=args.out/('task'+str(index));task_out.mkdir()
            task=dict(schema='demoncat.rope22.single_task_probe.v1',status='STARTED',requests=[])
            report['tasks'].append(task)
            client = WorkerFaultClient(HostAdapter(llm, prepare_binding=binding, timeout=120), FaultConfig(22, 'wrong_row'))
            drive(llm, client, prompt=dict(prompt_token_ids=tokens), params=SamplingParams(temperature=0, max_tokens=1, ignore_eos=True),
                out=task_out, report=task, stop_fault_budget=lambda: signal.alarm(120))
            if task.get('ownership_retained') is False and report.get('binding'):
                task['binding_removal'] = llm.collective_rpc(remove_binding, args=(task['worker'],), timeout=120)
                require(len(task['binding_removal']) == 1 and task['binding_removal'][0]['status'] == 'SERVING_DEPLOYMENT_REMOVED',
                        'OWNED_DEPLOYMENT_REMOVAL_UNCONFIRMED')
            write(task_out/'result.json',task)
            require(task['status']==SUCCESS,'INDEPENDENT_ORDINARY_TASK_FAILED:'+str(index)+':'+str(task.get('error')))
        report.update(status=SUCCESS,independent_task_scopes=2,actual_requests=8)
    except BaseException as error:
        report.setdefault('error', repr(error)); report.setdefault('traceback', traceback.format_exc())
        report['status'] = 'REFUSED_OR_FAILED'
    finally:
        signal.alarm(0); signal.signal(signal.SIGALRM, prior)
        report['elapsed_seconds'] = time.monotonic()-started
        write(args.out/'result.json', report)
    print(json.dumps({key: report[key] for key in ('status', 'error') if key in report}))
    return 0 if report['status'] == SUCCESS else 2


if __name__ == '__main__': raise SystemExit(main())
