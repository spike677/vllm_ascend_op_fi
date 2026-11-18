"""Root-queued public Fault8/18 probe; every forward is a real enqueue.

No historical scope/certificate authorizes this fresh Worker. Raw receipts stay
under /evidence. Numerical E5 capture/review is a separate operation.
"""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
import traceback

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT/'src'), str(ROOT/'fault39/src')]
from demoncat.runtime.vllm_ascend.worker_extension.config import FaultConfig
from demoncat.runtime.vllm_ascend.worker_extension.host import HostAdapter, WorkerFaultClient
from demoncat.runtime.vllm_ascend.worker_extension.providers.tensor_serving import REQUESTS
from demoncat.runtime.vllm_ascend.worker_extension.serving_tensor_binding import MECHANISMS
from demoncat.runtime.vllm_ascend.worker_extension.serving_context import read_enqueued_identity

PHASES = tuple(name for name, _ in REQUESTS)
PROMPT = 'Explain what a checksum is in one sentence.'
SUCCESS = 'PUBLIC_WORKER_TENSOR_GROUP_PROTOCOL_COMPLETE'


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def sha_file(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for data in iter(lambda: stream.read(8*1024*1024), b''):
            value.update(data)
    return value.hexdigest()


def actual(worker):
    from vllm.v1.worker.worker_base import WorkerWrapperBase
    return worker.worker if isinstance(worker, WorkerWrapperBase) else worker


def setup(worker, expected, mechanism, prompt, weights, workload):
    import torch
    import torch_npu
    from demoncat.runtime.vllm_ascend.worker_extension.serving_tensor_binding import install_tensor_serving_binding
    worker = actual(worker)
    identity = worker.demoncat_worker_identity()
    if (identity.get('status') != 'WORKER_IDENTIFIED' or identity.get('quarantined') is not False
            or identity.get('worker') != expected):
        raise RuntimeError('ACTUAL_DEPLOYMENT_WORKER_IDENTITY_CHANGED')
    environment = dict(torch=torch.__version__, torch_npu=torch_npu.__version__,
                       device=torch.npu.get_device_name(expected['logical_device']))
    bound = install_tensor_serving_binding(worker, mechanism, prompt_token_ids=prompt,
        weight_manifest=weights, workload_sha256=workload, environment_sha256=digest(environment),
        first_live_validation=True)
    return dict(binding=bound, environment=environment, worker=identity['worker'])


def remove_binding(worker, expected, mechanism):
    from demoncat.runtime.vllm_ascend.worker_extension.serving_tensor_binding import remove_tensor_serving_binding
    worker = actual(worker)
    identity = worker.demoncat_worker_identity()
    if identity.get('worker') != expected or identity.get('quarantined') is not False:
        raise RuntimeError('ACTUAL_CLEANUP_WORKER_IDENTITY_CHANGED')
    return remove_tensor_serving_binding(worker, mechanism)


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def drive(llm, client, *, prompt, params, out, report):
    if client.config.mechanism not in MECHANISMS:
        raise ValueError('EXPLICIT_NATURAL_TENSOR_MECHANISM_REQUIRED')
    receipts = out/'worker_core_rpc'
    receipts.mkdir(exist_ok=False)
    core = report.setdefault('worker_core', dict(schema='demoncat.tensor.public_probe.v1',
        config=client.config.as_dict(), receipts=[], qualification_granted=False))
    seen, pending = [], []
    recorded = 0
    report.setdefault('requests', [])
    def save():
        write_json(out/'result.json', report)
    def action(name, *args, expected='ACTION_COMPLETED'):
        nonlocal recorded
        snapshot = getattr(client, name)(*args)
        for item in snapshot['history'][recorded:]:
            path = receipts/(str(len(core['receipts'])).zfill(3)+'-'+item['action']+'.json')
            write_json(path, item)
            core['receipts'].append(dict(action=item['action'], path=str(path.relative_to(out)), sha256=sha_file(path)))
        recorded = len(snapshot['history'])
        core['last_scope'] = {key: deepcopy(value) for key, value in snapshot.items()
                              if key not in ('history', 'response', 'capability_enumeration')}
        state = snapshot['response'].get('provider_state') or {}
        coverage, serving = state.get('coverage', {}), state.get('serving', {})
        if serving.get('provenance') is not None:
            report.setdefault('setup', {})['provenance'] = deepcopy(serving['provenance'])
        if 'discovery' in coverage:
            report['qualification'] = deepcopy(coverage['discovery'])
        for key in ('C2', 'C3'):
            if key in coverage:
                report[key.lower()] = deepcopy(coverage[key])
        if 'cleanup' in coverage:
            report.setdefault('cleanup', deepcopy(coverage['cleanup']))
            core['final_cleanup'] = deepcopy(coverage['cleanup'])
        save()
        if expected is not None and snapshot['status'] != expected:
            raise RuntimeError('PUBLIC_CORE_ACTION_FAILED:'+name+':'+snapshot['status'])
        return snapshot, state
    def request(phase, *, recovery=False):
        nonlocal pending
        if llm.llm_engine.has_unfinished_requests():
            raise RuntimeError('UNRELATED_PENDING_REQUEST')
        if llm.reset_prefix_cache(reset_running_requests=False) is not True:
            raise RuntimeError('FRESH_PREFILL_CACHE_RESET_UNCONFIRMED')
        pending = list(llm.enqueue([prompt], params, use_tqdm=False))
        mapping = read_enqueued_identity(llm, pending, seen_internal_request_ids=seen)
        if len(pending) != 1:
            raise RuntimeError('EXACTLY_ONE_PUBLIC_ENQUEUE_REQUIRED')
        seen.extend(pending)
        _, state = action('advance', pending[0])
        if state['serving']['next_phase'] != phase:
            raise RuntimeError('ACTUAL_PUBLIC_PROTOCOL_PHASE_CHANGED')
        if phase in ('c2_fault', 'c3_fault_0'):
            group = state['serving']['group']
            if group is None:
                raise RuntimeError('ACTUAL_PUBLIC_GROUP_NOT_OPENED')
            report['c2_open' if phase == 'c2_fault' else 'c3_open'] = deepcopy(group)
        results = llm.wait_for_completion(use_tqdm=False)
        if len(results) != 1 or results[0].request_id != mapping['external_request_id']:
            raise RuntimeError('ACTUAL_DRIVER_WORKER_REQUEST_MAPPING_CHANGED')
        snapshot, before = action('query', expected='QUERY_OK')
        if snapshot['response'].get('recovery_required') is not False:
            raise RuntimeError('PUBLIC_CORE_REQUEST_REQUIRES_RECOVERY')
        if before['serving']['next_phase'] != phase or before['serving']['pending_request'] != [pending[0], phase]:
            raise RuntimeError('READ_ONLY_QUERY_ADVANCED_OR_LOST_REQUEST')
        is_health = phase in ('cleanup_healthy', 'recovery_health')
        row = before['coverage']['same_worker_healthy'] if is_health else before['serving']['last_request']
        if row is None or row['request_id'] != pending[0] or (not is_health and row['phase'] != phase):
            raise RuntimeError('ACTUAL_PUBLIC_PHASE_ROW_MISSING')
        action('collect_replay')
        item = dict(phase=phase, mapping=mapping, worker=deepcopy(row), output_tokens=list(results[0].outputs[0].token_ids))
        if is_health:
            if row['cleanup_before_request'] is not True:
                raise RuntimeError('ACTUAL_CLEANUP_HEALTH_REQUEST_UNPROVEN')
            report['health_request'], report['healthy_after'] = item, item['output_tokens']
            if report['healthy_after'] != report['healthy_before']:
                raise RuntimeError('SAME_WORKER_GENERATION_NOT_RESTORED')
        else:
            report.setdefault('recovery_requests' if recovery else 'requests', []).append(item)
        pending = []
        save()
    try:
        try:
            for name in ('prepare', 'register', 'check', 'enable'):
                action(name)
            for phase in PHASES:
                request(phase)
            _, state = action('finish')
            if not state['serving']['complete'] or not all(report[key]['contract_satisfied'] for key in ('c2', 'c3')):
                raise RuntimeError('PUBLIC_CORE_TENSOR_PROTOCOL_NOT_COMPLETE')
            for name in ('disable', 'unregister'):
                action(name)
            action('detach', expected='DETACHED')
            report.update(status=SUCCESS, statistics_status='SEPARATE_RAW_BF_CAPTURE_REQUIRED')
        except BaseException as error:
            report.update(status='REFUSED_OR_FAILED', error=repr(error), traceback=traceback.format_exc())
            if pending:
                try:
                    llm.llm_engine.abort_request(pending, internal=True)
                except BaseException as aborted:
                    report['abort_error'] = repr(aborted)
        # Outside the failed Python exception context: at most one new OFF
        # and one new health request. No F retry or reconstructed C2 object.
        if report['status'] == 'REFUSED_OR_FAILED':
            try:
                if client.snapshot()['ticket'] is not None:
                    action('query', expected=None)
                    recovered, state = action('recover', expected=None)
                    recovery_ready = recovered['status'] == 'ACTION_COMPLETED'
                    for _ in range(2):
                        phase = state.get('serving', {}).get('next_phase')
                        if not recovery_ready or phase not in ('recovery_off', 'recovery_health'):
                            break
                        request(phase, recovery=True)
                        observed, state = action('query', expected='QUERY_OK')
                        # A successful query is read-only, but confirms the
                        # previous explicit collect's next recovery phase.
                        recovery_ready = observed['status'] == 'QUERY_OK' and observed['response']['recovery_required'] is False
                    action('finish', expected=None)
                    action('disable', expected=None)
                    action('unregister', expected=None)
                if client.snapshot()['ownership_retained']:
                    action('detach', expected=None)
            except BaseException as error:
                report['cleanup_error'] = repr(error)
    finally:
        final = client.snapshot()
        write_json(out/'host_receipt.json', final)
        core.update(final_host_receipt=dict(path='host_receipt.json', sha256=sha_file(out/'host_receipt.json')),
                    ownership_retained=final['ownership_retained'])
        save()
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mechanism', choices=MECHANISMS, required=True)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--memory', type=float, required=True)
    args = parser.parse_args(argv)
    if not .1 <= args.memory <= .8 or not args.out.is_absolute() or not args.out.resolve().is_relative_to(Path('/evidence')):
        parser.error('Bounded memory and private /evidence output are required')
    args.out.mkdir(parents=True, exist_ok=False)
    report = dict(status='STARTED', mechanism=args.mechanism, standard_vllm_worker=True,
        public_entrypoint='WorkerFaultClient+FaultConfig', public_qualification_granted=False,
        historical_scope_inherited=False, requests=[], probe_sha256=sha_file(__file__))
    client = None
    try:
        from vllm import LLM, SamplingParams
        from demoncat.runtime.vllm_ascend.worker_extension.serving_context import require_host_driven_core
        weights = {path.name: dict(sha256=sha_file(path), bytes=path.stat().st_size)
            for path in sorted({*args.model.glob('*.safetensors'), *args.model.glob('pytorch_model*.bin')})}
        if not weights:
            raise RuntimeError('ACTUAL_PRETRAINED_WEIGHTS_REQUIRED')
        model_config = dict(config_sha256=sha_file(args.model/'config.json'), routing_modified=False)
        if args.mechanism == '8.restore_offset':
            if model_config['config_sha256'] != '229913ea2a346ccdb571bb3cb23414ca3a0ee1a9455fe88a15a5788bc837cb75' or len(weights) != 4:
                raise RuntimeError('ACTUAL_REVIEWED_DEEPSEEK_W8A8_REQUIRED')
            model_config['quantization_sha256'] = sha_file(args.model/'quant_model_description.json')
        report['model_identity'] = dict(weights=weights, **model_config)
        llm = LLM(model=str(args.model), dtype='bfloat16', tensor_parallel_size=1, enforce_eager=True,
            max_model_len=128, max_num_seqs=1, max_num_batched_tokens=128, num_gpu_blocks_override=64,
            enable_prefix_caching=False, enable_chunked_prefill=False, async_scheduling=False,
            gpu_memory_utilization=args.memory, seed=42, disable_log_stats=True,
            worker_extension_cls='demoncat.runtime.vllm_ascend.controller.DemoncatWorkerExtension')
        require_host_driven_core(llm)
        tokens = llm.get_tokenizer().encode(PROMPT, add_special_tokens=True)
        prompt, params = dict(prompt_token_ids=tokens), SamplingParams(temperature=0, max_tokens=1, ignore_eos=True)
        healthy, = llm.generate([prompt], params, use_tqdm=False)
        report['healthy_before'] = list(healthy.outputs[0].token_ids)
        workload = digest(dict(weights=weights, tokens=tokens, model_config=model_config))
        def deploy(expected, config):
            rows = llm.collective_rpc(setup, args=(expected, config.mechanism, tokens, weights, workload), timeout=300)
            if len(rows) != 1 or rows[0]['worker'] != expected:
                raise RuntimeError('TP1_ACTUAL_DEPLOYMENT_ROUTE_REQUIRED')
            report['setup'] = rows[0]
            return {}
        fault, variant = args.mechanism.split('.', 1)
        client = WorkerFaultClient(HostAdapter(llm, prepare_binding=deploy, timeout=300), FaultConfig(int(fault), variant))
        drive(llm, client, prompt=prompt, params=params, out=args.out, report=report)
        if not client.snapshot()['ownership_retained'] and report.get('setup'):
            rows = llm.collective_rpc(remove_binding, args=(client.snapshot()['worker'], args.mechanism), timeout=300)
            if len(rows) != 1 or rows[0].get('status') != 'SERVING_DEPLOYMENT_REMOVED':
                raise RuntimeError('ACTUAL_DEPLOYMENT_REMOVAL_UNCONFIRMED')
            report['binding_removal'] = rows[0]
    except BaseException as error:
        report.update(status='REFUSED_OR_FAILED', error=repr(error), traceback=traceback.format_exc())
    finally:
        if client is not None:
            report.setdefault('worker_core', {})['ownership_retained'] = client.snapshot()['ownership_retained']
        write_json(args.out/'result.json', report)
    print(json.dumps({key: report[key] for key in ('status', 'error') if key in report}))
    return 0 if report['status'] == SUCCESS else 2


if __name__ == '__main__':
    raise SystemExit(main())
