"""Root-scheduled public KV41 probe; existing serving enqueue is the only forward.

Complete request observations, original RPC replies and live-proof audit receipts
stay in /evidence. No JSON object is reconstructed into a live C2 capability.
"""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys
import traceback

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT/'src'), str(ROOT/'fault39/src')]
from experiments.c2c3_postwrite_20261117.natural41_probe import PROMPT, actual_worker, output
from experiments.c2c3_postwrite_20261117.natural41_coverage_probe import actual_vllm_weights, SOURCES as GROUP_SOURCES
from experiments.c2c3_postwrite_20261117.model_identity import file_sha, digest, model_source_identity
from demoncat.runtime.vllm_ascend.worker_extension.config import FaultConfig
from demoncat.runtime.vllm_ascend.worker_extension.host import HostAdapter, WorkerFaultClient
from demoncat.runtime.vllm_ascend.worker_extension.providers.fault41 import _REQUESTS
from demoncat.runtime.vllm_ascend.worker_extension.serving_context import read_enqueued_identity


PHASES = tuple(label for label, phase in _REQUESTS)
SUCCESS = 'STANDARD_WORKER_KV41_CORE_CONTRACT_COMPLETE'
SOURCES = tuple(dict.fromkeys((*GROUP_SOURCES,
    'experiments/c2c3_postwrite_20261117/natural41_worker_core_probe.py',
    'experiments/c2c3_postwrite_20261117/prepare41_worker_core.py',
    *('src/demoncat/runtime/vllm_ascend/worker_extension/' + name for name in (
        'config.py', 'host.py', 'runtime.py', 'extension.py', 'identity.py', 'registry.py',
        'serving41.py', 'providers/__init__.py', 'providers/_scope.py', 'providers/fault41.py')))))


def setup(worker, expected_identity, library, sha256, build_fingerprint, prompt, weights, sources, max_seconds):
    """Trusted deployment only, outside the public fault RPC and its lock."""
    import torch
    import torch_npu
    from fault_injection.coverage_session import CoverageScope
    from fault_injection.deployments.postwrite import NativePostwriteLibrary
    from demoncat.runtime.vllm_ascend.worker_extension.serving41 import install_fault41_serving_binding
    worker = actual_worker(worker)
    identity = worker.demoncat_worker_identity()
    if identity.get('worker') != expected_identity or identity.get('quarantined') is not False:
        raise RuntimeError('ACTUAL_DEPLOYMENT_WORKER_IDENTITY_CHANGED')
    environment = dict(torch=torch.__version__, torch_npu=torch_npu.__version__, worker=expected_identity,
        device=torch.npu.get_device_name(expected_identity['logical_device']),
        native_sha256=sha256, build_fingerprint=build_fingerprint)
    model_source = model_source_identity(worker.model_runner.model)
    scope = CoverageScope('41.tail_column', digest(dict(prompt=prompt, max_tokens=2, requests=2)),
        digest(environment), digest(sources), 'natural_worker', weights['weights_sha256'],
        model_source['source_sha256'])
    transport = NativePostwriteLibrary(library, sha256)
    bound = install_fault41_serving_binding(worker, transport=transport, prompt_token_ids=prompt,
        scope=scope, max_seconds=max_seconds, first_live_validation=True)
    return dict(binding=bound, environment=environment, model_source=model_source, worker=expected_identity)


def remove_binding(worker, expected_identity):
    from demoncat.runtime.vllm_ascend.worker_extension.serving41 import remove_fault41_serving_binding
    worker = actual_worker(worker)
    identity = worker.demoncat_worker_identity()
    if identity.get('worker') != expected_identity or identity.get('quarantined') is not False:
        raise RuntimeError('ACTUAL_CLEANUP_WORKER_IDENTITY_CHANGED')
    return remove_fault41_serving_binding(worker)


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def drive(llm, client, *, prompt, params, out, report):
    receipts = out/'worker_core_rpc'
    receipts.mkdir(exist_ok=False)
    core = report.setdefault('worker_core', dict(schema='demoncat.kv41.public_probe.v1',
        config=client.config.as_dict(), receipts=[], qualification_granted=False))
    report.setdefault('requests', [])
    seen, pending, recorded_history = [], [], 0

    def save():
        write_json(out/'result.json', report)

    def action(name, *args, expected='ACTION_COMPLETED'):
        nonlocal recorded_history
        snapshot = getattr(client, name)(*args)
        for item in snapshot['history'][recorded_history:]:
            path = receipts/(str(len(core['receipts'])).zfill(3)+'-'+item['action']+'.json')
            write_json(path, item)
            core['receipts'].append(dict(action=item['action'], path=str(path.relative_to(out)), sha256=file_sha(path)))
        recorded_history = len(snapshot['history'])
        core['last_scope'] = {key: deepcopy(value) for key, value in snapshot.items()
                              if key not in ('history', 'response', 'capability_enumeration')}
        state = snapshot['response'].get('provider_state') or {}
        coverage = state.get('coverage', {})
        for key in ('C1', 'C2', 'C3', 'deployment', 'failed_attempt_recovery', 'failed_requests', 'failed_c1_off'):
            if key in coverage:
                report[key.lower()] = deepcopy(coverage[key])
        if state.get('serving'):
            core['serving'] = deepcopy(state['serving'])
        save()
        if expected is not None and snapshot['status'] != expected:
            raise RuntimeError('PUBLIC_CORE_ACTION_FAILED:'+name+':'+snapshot['status'])
        return snapshot, state

    def enqueue():
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
        return mapping

    def complete(mapping):
        values = llm.wait_for_completion(use_tqdm=False)
        if len(values) != 1 or values[0].request_id != mapping['external_request_id']:
            raise RuntimeError('ACTUAL_DRIVER_WORKER_REQUEST_MAPPING_CHANGED')
        return output(values[0])

    def healthy(value):
        return all(value[key] == report['healthy_before'][key] for key in ('tokens', 'text'))

    def request(phase, *, recovery=False):
        nonlocal pending
        mapping = enqueue()
        _, state = action('advance', pending[0])
        if list(state['serving']['pending_request'])[:2] != [pending[0], phase]:
            raise RuntimeError('ACTUAL_PUBLIC_PHASE_NOT_SELECTED')
        row = dict(phase=phase, mapping=mapping)
        report.setdefault('recovery_requests' if recovery else 'requests', []).append(row)
        save()
        row['output'] = complete(mapping)
        snapshot, state = action('query', expected='QUERY_OK')
        if snapshot['response'].get('recovery_required') is not False:
            raise RuntimeError('PUBLIC_CORE_REQUEST_REQUIRES_RECOVERY')
        if phase.endswith('health'):
            actual = state['group']['health']
            if (actual is None or actual['request']['request_ids'] != pending or not healthy(row['output'])):
                raise RuntimeError('ACTUAL_UNINSTALLED_HEALTH_UNPROVEN')
        else:
            actual = state['provider']['pending']
            if (actual is None or actual['request_ids'] != pending
                    or 'model_output_sha256' not in actual or 'independent_inventory' not in actual):
                raise RuntimeError('ACTUAL_PUBLIC_REQUEST_NOT_RETURNED')
        row['worker'] = deepcopy(actual)  # Complete writer/cache/consumer inventory, never only totals.
        pending = []
        save()

    def ordinary_health():
        nonlocal pending
        core['post_unregister_health_started'] = True
        mapping = enqueue()
        value = complete(mapping)
        identity = client.host.route(client.config)
        row = dict(mapping=mapping, output=value, worker=identity, after_public_unregister=True)
        report.setdefault('post_unregister_health_attempts', []).append(row)
        if identity != client.snapshot()['worker'] or not healthy(value):
            raise RuntimeError('POST_UNREGISTER_SAME_WORKER_GENERATION_MISMATCH')
        report['post_unregister_health'] = row
        pending = []
        save()

    def recover_failed_attempt():
        try:
            if client.snapshot()['ticket'] is not None:
                action('query', expected=None)
                recovered, state = action('recover', expected=None)
                if recovered['status'] == 'ACTION_COMPLETED':
                    phase = state.get('serving', {}).get('next_phase')
                    if state.get('replay_pending') and phase in ('recovery_c1_off', 'recovery_group_off', 'recovery_health'):
                        request(phase, recovery=True)
                        if phase == 'recovery_group_off':
                            request('recovery_health', recovery=True)
                        action('finish', expected=None)
                action('disable', expected=None)
                action('unregister', expected=None)
            if client.snapshot()['ownership_retained']:
                action('detach', expected=None)
            if not client.snapshot()['ownership_retained'] and not core.get('post_unregister_health_started'):
                ordinary_health()
        except BaseException as error:
            report['cleanup_error'] = repr(error)

    try:
        try:
            for name in ('prepare', 'register', 'check', 'enable'):
                action(name)
            for phase in PHASES:
                request(phase)
            _, state = action('finish')
            if (not state['serving']['complete'] or not report['c2']['contract_satisfied']
                    or not report['c3']['contract_satisfied'] or state['leases'] or state['inflight']
                    or state['owned_hooks'] or state['owned_api_state']['owned_links_present']
                    or state['coverage_held_leases']):
                raise RuntimeError('PUBLIC_CORE_KV41_PROTOCOL_NOT_COMPLETE')
            core['final_cleanup'] = deepcopy(state)
            for name in ('disable', 'unregister'):
                action(name)
            action('detach', expected='DETACHED')
            ordinary_health()
            report.update(status=SUCCESS, statistics_status='SEPARATE_RAW_BF_CAPTURE_REQUIRED')
        except BaseException as error:
            report.update(status='REFUSED_OR_FAILED', error=repr(error), traceback=traceback.format_exc())
            if pending:
                try:
                    llm.llm_engine.abort_request(pending, internal=True)
                    pending = []
                except BaseException as aborted:
                    report['abort_error'] = repr(aborted)
        # New OFF/health forwards run outside the failed Python exception context.
        # At most one recovery OFF plus one health; no fault or Worker restart.
        if report['status'] == 'REFUSED_OR_FAILED':
            recover_failed_attempt()
    finally:
        final = client.snapshot()
        write_json(out/'host_receipt.json', final)
        core.update(final_host_receipt=dict(path='host_receipt.json', sha256=file_sha(out/'host_receipt.json')),
                    ownership_retained=final['ownership_retained'])
        save()
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('model', 'library', 'sha256', 'build-fingerprint'):
        parser.add_argument('--'+name, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--memory', type=float, default=.35)
    parser.add_argument('--max-seconds', type=int, default=600)
    args = parser.parse_args(argv)
    if not .1 <= args.memory <= .8 or not 0 < args.max_seconds <= 1500:
        parser.error('Finite Worker memory and session time bounds required')
    if any(len(value) != 64 or any(c not in '0123456789abcdef' for c in value)
           for value in (args.sha256, args.build_fingerprint)):
        parser.error('Actual reviewed Native SHA and build fingerprint required')
    if not args.out.is_absolute() or not args.out.resolve().is_relative_to(Path('/evidence')):
        parser.error('Full private observations must stay under /evidence')
    args.out.mkdir(parents=True, exist_ok=False)
    report = dict(schema='demoncat.kv41.worker_core_probe.v1', status='STARTED', mechanism='41.tail_column',
        standard_vllm_worker=True, public_qualification_granted=False, native_out_of_bounds_claim=False,
        build_fingerprint=args.build_fingerprint, native_sha256=args.sha256, probe_sha256=file_sha(__file__),
        public_entrypoint='WorkerFaultClient+FaultConfig(41,tail_column)', requests=[])
    client = None
    try:
        from vllm import LLM, SamplingParams
        from demoncat.runtime.vllm_ascend.worker_extension.serving_context import require_host_driven_core
        report['weights'] = actual_vllm_weights(args.model)
        report['source_sha256'] = {name: file_sha(ROOT/name) for name in SOURCES}
        llm = LLM(model=args.model, dtype='bfloat16', load_format='safetensors', trust_remote_code=False,
            tensor_parallel_size=1, enforce_eager=True, max_model_len=128, max_num_seqs=1,
            max_num_batched_tokens=128, num_gpu_blocks_override=64, enable_prefix_caching=False,
            enable_chunked_prefill=False, async_scheduling=False, gpu_memory_utilization=args.memory,
            seed=42, disable_log_stats=True,
            worker_extension_cls='demoncat.runtime.vllm_ascend.controller.DemoncatWorkerExtension')
        require_host_driven_core(llm)
        if actual_vllm_weights(args.model) != report['weights']:
            raise RuntimeError('MODEL_ARTIFACTS_CHANGED_DURING_LOAD')
        tokens = llm.get_tokenizer().encode(PROMPT, add_special_tokens=True)
        prompt = dict(prompt_token_ids=tokens)
        params = SamplingParams(temperature=0, max_tokens=2, ignore_eos=True)
        before, = llm.generate([prompt], params, use_tqdm=False)
        report['healthy_before'] = output(before)

        def binding(identity, config):
            rows = llm.collective_rpc(setup, args=(identity, args.library, args.sha256, args.build_fingerprint,
                tokens, report['weights'], report['source_sha256'], args.max_seconds), timeout=300)
            if len(rows) != 1 or rows[0]['worker'] != identity:
                raise RuntimeError('TP1_ACTUAL_DEPLOYMENT_ROUTE_REQUIRED')
            report['setup'] = rows[0]
            return {}

        client = WorkerFaultClient(HostAdapter(llm, prepare_binding=binding, timeout=300), FaultConfig(41, 'tail_column'))
        drive(llm, client, prompt=prompt, params=params, out=args.out, report=report)
        if not client.snapshot()['ownership_retained'] and report.get('setup'):
            rows = llm.collective_rpc(remove_binding, args=(client.snapshot()['worker'],), timeout=300)
            if len(rows) != 1 or rows[0].get('status') != 'SERVING_DEPLOYMENT_REMOVED':
                raise RuntimeError('ACTUAL_DEPLOYMENT_REMOVAL_UNCONFIRMED')
            report['binding_removal'] = rows[0]
    except BaseException as error:
        report.update(status='REFUSED_OR_FAILED', error=repr(error), traceback=traceback.format_exc())
    finally:
        write_json(args.out/'result.json', report)
    print(json.dumps({key: report[key] for key in ('status', 'error') if key in report}))
    return 0 if report['status'] == SUCCESS else 2


if __name__ == '__main__':
    raise SystemExit(main())
