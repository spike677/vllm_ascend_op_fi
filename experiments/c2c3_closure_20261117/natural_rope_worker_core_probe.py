"""Root-scheduled public Worker Core22 probe; all inference is real enqueue.

Full responses/observations stay in the supplied /evidence directory on the
serving host. This code never reconstructs a C2 capability from JSON.
"""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys
import traceback

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT/'src'), str(ROOT/'fault39/src')]
from experiments.c2c3_closure_20261117.natural_rms_worker_probe import actual, digest
from experiments.c2c3_closure_20261117.natural_tensor_census import PROMPT, sha_file
from demoncat.runtime.vllm_ascend.worker_extension.config import FaultConfig
from demoncat.runtime.vllm_ascend.worker_extension.host import HostAdapter, WorkerFaultClient
from demoncat.runtime.vllm_ascend.worker_extension.serving_context import read_enqueued_identity


PHASES = ('native_baseline', 'baseline', 'baseline_repeat', 'lossless',
          'c2_fault', 'c2_off', 'c3_fault_0', 'c3_fault_1', 'c3_off', 'cleanup_healthy')
SUCCESS = 'STANDARD_WORKER_ROPE_GROUP_CONTRACT_COMPLETE'


def setup(worker, expected_identity, prompt, weights, workload):
    """Inert trusted deployment call, outside the fault RPC lock."""
    import torch
    import torch_npu
    from demoncat.runtime.vllm_ascend.worker_extension.serving22 import install_fault22_serving_binding
    worker = actual(worker)
    identity = worker.demoncat_worker_identity()
    if (identity.get('status') != 'WORKER_IDENTIFIED' or identity.get('quarantined') is not False
            or identity.get('worker') != expected_identity):
        raise RuntimeError('ACTUAL_DEPLOYMENT_WORKER_IDENTITY_CHANGED')
    environment = dict(torch=torch.__version__, torch_npu=torch_npu.__version__,
        device=torch.npu.get_device_name(expected_identity['logical_device']))
    bound = install_fault22_serving_binding(worker, prompt_token_ids=prompt, weight_manifest=weights,
        workload_sha256=workload, environment_sha256=digest(environment), first_live_validation=True)
    return dict(binding=bound, environment=environment, worker=identity['worker'])


def remove_binding(worker, expected_identity):
    from demoncat.runtime.vllm_ascend.worker_extension.serving22 import remove_fault22_serving_binding
    worker = actual(worker)
    identity = worker.demoncat_worker_identity()
    if identity.get('worker') != expected_identity or identity.get('quarantined') is not False:
        raise RuntimeError('ACTUAL_CLEANUP_WORKER_IDENTITY_CHANGED')
    return remove_fault22_serving_binding(worker)


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def drive(llm, client, *, prompt, params, out, report):
    """Consume an existing serving object and public client, never run a model RPC."""
    receipts = out/'worker_core_rpc'
    receipts.mkdir(exist_ok=False)
    core = report.setdefault('worker_core', dict(schema='demoncat.rope22.public_probe.v1',
        config=client.config.as_dict(), receipts=[], qualification_granted=False))
    report.setdefault('requests', [])
    seen, pending = [], []
    recorded_history = 0

    def save():
        write_json(out/'result.json', report)

    def action(name, *args, expected='ACTION_COMPLETED'):
        nonlocal recorded_history
        snapshot = getattr(client, name)(*args)
        # Preserve every original Worker response once. Repeated whole Host
        # history snapshots would multiply the full raw phase observations.
        delta = snapshot['history'][recorded_history:]
        for item in delta:
            path = receipts/(str(len(core['receipts'])).zfill(3)+'-'+item['action']+'.json')
            write_json(path, item)
            core['receipts'].append(dict(action=item['action'], path=str(path.relative_to(out)),
                                         sha256=sha_file(path)))
        recorded_history = len(snapshot['history'])
        core['last_scope'] = {key: deepcopy(value) for key, value in snapshot.items()
                              if key not in ('history', 'response', 'capability_enumeration')}
        state = snapshot['response'].get('provider_state') or {}
        serving, coverage = state.get('serving', {}), state.get('coverage', {})
        if serving.get('provenance') is not None:
            report.setdefault('setup', {})['provenance'] = deepcopy(serving['provenance'])
        if 'discovery' in coverage:
            report['qualification'] = deepcopy(coverage['discovery'])
        for key in ('C2', 'C3'):
            if key in coverage:
                report[key.lower()] = deepcopy(coverage[key])  # Full original report, including live proof ledger.
        if 'cleanup' in coverage:
            # The actual group cleanup preceding request 10 retains the legacy
            # nine-row envelope; final public cleanup is recorded separately.
            if 'cleanup' not in report:
                report['cleanup'] = deepcopy(coverage['cleanup'])
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
        if phase in ('c2_fault', 'c3_fault_0'):
            group = state['serving']['group']
            if group is None:
                raise RuntimeError('ACTUAL_PUBLIC_GROUP_NOT_OPENED')
            report['c2_open' if phase == 'c2_fault' else 'c3_open'] = deepcopy(group)
        results = llm.wait_for_completion(use_tqdm=False)
        if len(results) != 1 or results[0].request_id != mapping['external_request_id']:
            raise RuntimeError('ACTUAL_DRIVER_WORKER_REQUEST_MAPPING_CHANGED')
        snapshot, state = action('query', expected='QUERY_OK')
        if snapshot['response'].get('recovery_required') is not False:
            raise RuntimeError('PUBLIC_CORE_REQUEST_REQUIRES_RECOVERY')
        if phase == 'cleanup_healthy':
            row = state['coverage']['same_worker_healthy']
            if row['request_id'] != pending[0] or row['cleanup_before_request'] is not True:
                raise RuntimeError('ACTUAL_CLEANUP_HEALTH_REQUEST_UNPROVEN')
        else:
            row = state['serving']['last_request']
            if row is None or row['phase'] != phase or row['request_id'] != pending[0]:
                raise RuntimeError('ACTUAL_PUBLIC_PHASE_ROW_MISSING')
        item = dict(phase=phase, mapping=mapping, worker=deepcopy(row),
                    output_tokens=list(results[0].outputs[0].token_ids))
        if phase == 'cleanup_healthy':
            report['health_request'] = item
            report['healthy_after'] = item['output_tokens']
            if report['healthy_after'] != report['healthy_before']:
                raise RuntimeError('SAME_WORKER_GENERATION_NOT_RESTORED')
        else:
            report.setdefault('recovery_requests' if recovery else 'requests', []).append(item)
        pending = []
        save()

    def recover_failed_attempt():
        try:
            if client.snapshot()['ticket'] is not None:
                action('query', expected=None)
                recovered, state = action('recover', expected=None)
                if (recovered['status'] == 'ACTION_COMPLETED' and state.get('replay_pending') is True
                        and state.get('serving', {}).get('next_phase') == 'recovery_off'):
                    request('recovery_off', recovery=True)
                    action('finish', expected=None)
                action('disable', expected=None)
                action('unregister', expected=None)
            if client.snapshot()['ownership_retained']:
                action('detach', expected=None)
        except BaseException as cleanup_error:
            report['cleanup_error'] = repr(cleanup_error)

    try:
        try:
            for name in ('prepare', 'register', 'check', 'enable'):
                action(name)
            for phase in PHASES:
                request(phase)
            _, state = action('finish')
            if (not state['serving']['complete'] or not report['c2']['contract_satisfied']
                    or not report['c3']['contract_satisfied']):
                raise RuntimeError('PUBLIC_CORE_ROPE_PROTOCOL_NOT_COMPLETE')
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
        # Leave the failed Python exception context before the new real OFF.
        # Existing observers require a healthy return context. Only one OFF is
        # allowed; fault requests are never retried and run status stays failed.
        if report['status'] == 'REFUSED_OR_FAILED':
            recover_failed_attempt()
    finally:
        final = client.snapshot()
        write_json(out/'host_receipt.json', final)
        core.update(final_host_receipt=dict(path='host_receipt.json', sha256=sha_file(out/'host_receipt.json')),
                    ownership_retained=final['ownership_retained'])
        save()
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--memory', type=float, default=.35)
    args = parser.parse_args(argv)
    if not .1 <= args.memory <= .8:
        parser.error('Bounded Worker memory .1..8 required')
    if not args.out.is_absolute() or not args.out.resolve().is_relative_to(Path('/evidence')):
        parser.error('Full private observations must stay under /evidence')
    args.out.mkdir(parents=True, exist_ok=False)
    report = dict(status='STARTED', mechanism='22.wrong_row.fused_positions', standard_vllm_worker=True,
        public_qualification_granted=False, raw_device_launch_payload_observed=False, requests=[],
        public_entrypoint='WorkerFaultClient+FaultConfig(22,wrong_row)', probe_sha256=sha_file(__file__))
    client = None
    try:
        from vllm import LLM, SamplingParams
        from demoncat.runtime.vllm_ascend.worker_extension.serving_context import require_host_driven_core
        weights = {path.name: dict(sha256=sha_file(path), bytes=path.stat().st_size)
                   for path in sorted(args.model.glob('*.safetensors'))}
        if not weights:
            raise RuntimeError('ACTUAL_PRETRAINED_WEIGHTS_REQUIRED')
        config_sha = sha_file(args.model/'config.json')
        report['model_identity'] = dict(weights=weights, config_sha256=config_sha, route_configuration_modified=False)
        llm = LLM(model=str(args.model), dtype='bfloat16', tensor_parallel_size=1, enforce_eager=True,
            max_model_len=128, max_num_seqs=1, max_num_batched_tokens=128, num_gpu_blocks_override=64,
            enable_prefix_caching=False, enable_chunked_prefill=False, async_scheduling=False,
            gpu_memory_utilization=args.memory, seed=42, disable_log_stats=True, trust_remote_code=False,
            worker_extension_cls='demoncat.runtime.vllm_ascend.controller.DemoncatWorkerExtension')
        require_host_driven_core(llm)
        tokens = llm.get_tokenizer().encode(PROMPT, add_special_tokens=True)
        prompt = dict(prompt_token_ids=tokens)
        params = SamplingParams(temperature=0, max_tokens=1, ignore_eos=True)
        healthy, = llm.generate([prompt], params, use_tqdm=False)
        report['healthy_before'] = list(healthy.outputs[0].token_ids)
        workload = digest(dict(weights=weights, config_sha256=config_sha, tokens=tokens))

        def binding(identity, config):
            rows = llm.collective_rpc(setup, args=(identity, tokens, weights, workload), timeout=300)
            if len(rows) != 1 or rows[0]['worker'] != identity:
                raise RuntimeError('TP1_ACTUAL_DEPLOYMENT_ROUTE_REQUIRED')
            report['setup'] = rows[0]
            return {}

        client = WorkerFaultClient(HostAdapter(llm, prepare_binding=binding, timeout=300),
                                   FaultConfig(22, 'wrong_row'))
        drive(llm, client, prompt=prompt, params=params, out=args.out, report=report)
        if not client.snapshot()['ownership_retained'] and report.get('setup'):
            rows = llm.collective_rpc(remove_binding, args=(client.snapshot()['worker'],), timeout=300)
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
