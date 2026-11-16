"""Use the installed Worker Extension through vLLM's real collective RPC."""

import argparse
import hashlib
from importlib.metadata import version
import inspect
import json
import os
from pathlib import Path
import tempfile


EXTENSION = 'demoncat.runtime.vllm_ascend.controller.DemoncatWorkerExtension'
TENANT = 'blue-worker-extension-validation'
PROMPT = 'The capital of France is'
MECHANISMS = {'fault20': '20.buffer_substitute', 'fault39': '39.matmul_range'}
REPLAY_REFUSALS = {
    'fault20': 'FAULT20_TRUSTED_EXCLUSIVE_REPLAY_REQUIRED',
    'fault39': 'FAULT39_TRUSTED_SCHEDULER_REPLAY_REQUIRED',
}


def write_private(path, value):
    fd, filename = tempfile.mkstemp(dir=path.parent, prefix=path.name + '-')
    with os.fdopen(fd, 'w') as output:
        json.dump(value, output, allow_nan=False, indent=2)
        output.write('\n')
    Path(filename).replace(path)


def token_record(llm):
    from vllm import SamplingParams

    result, = llm.generate([PROMPT], SamplingParams(temperature=0, max_tokens=2),
                           use_tqdm=False)
    candidate, = result.outputs
    if not candidate.token_ids:
        raise RuntimeError('NO_MODEL_TOKENS')
    return {'prompt_tokens_sha256': hashlib.sha256(
        json.dumps(result.prompt_token_ids).encode()).hexdigest(),
        'output_token_ids': list(candidate.token_ids), 'finish_reason': candidate.finish_reason}


def identity_set(llm, tp):
    replies = llm.collective_rpc('demoncat_worker_identity', timeout=60)
    if any(row['status'] != 'WORKER_IDENTIFIED' for row in replies):
        raise RuntimeError(f'WORKER_CONTEXT_NOT_READY: {replies}')
    records = [row['worker'] for row in replies]
    if len(records) != tp or {row['rank'] for row in records} != set(range(tp)):
        raise RuntimeError('WORKER_RANKS_UNPROVEN')
    if len({row['pid'] for row in records}) != tp:
        raise RuntimeError('WORKER_PROCESSES_UNPROVEN')
    if {row['logical_device'] for row in records} != set(range(tp)):
        raise RuntimeError('LOGICAL_DEVICE_ASSIGNMENT_UNPROVEN')
    if any(not row['process_start_identity'] or not row['model_instance_id'] for row in records):
        raise RuntimeError('WORKER_START_OR_MODEL_IDENTITY_UNAVAILABLE')
    return sorted(records, key=lambda row: row['rank'])


def rpc_one(llm, method, *args):
    result, = llm.collective_rpc(method, timeout=90, args=args)
    if not isinstance(result, dict):
        raise RuntimeError(f'INVALID_RPC_RESPONSE: {method}')
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--case', choices=('identity', 'restart', *MECHANISMS), required=True)
    parser.add_argument('--tp', type=int, choices=(1, 2), required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--previous', type=Path)
    args = parser.parse_args()
    if args.case != 'identity' and args.tp != 1:
        parser.error('Collective mutation probe requires TP1')

    from demoncat.runtime.vllm_ascend.controller import DemoncatWorkerExtension
    from vllm import LLM

    record = {'schema': 'demoncat.worker_extension.validation.v1', 'case': args.case,
              'status': 'STARTING', 'tp': args.tp, 'driver_pid': os.getpid(),
              'extension': EXTENSION, 'extension_file': inspect.getfile(DemoncatWorkerExtension),
              'packages': {name: version(name) for name in (
                  'demoncat-worker-extension', 'vllm', 'vllm-ascend', 'torch', 'torch-npu')}}
    write_private(args.out, record)
    try:
        llm = LLM(model='/model', dtype='bfloat16', tensor_parallel_size=args.tp,
                  enforce_eager=True, max_model_len=128, max_num_seqs=1,
                  enable_prefix_caching=False, gpu_memory_utilization=0.35,
                  worker_extension_cls=EXTENSION, disable_log_stats=True)
        records = identity_set(llm, args.tp)
        record['workers'] = records
        record['capabilities'] = llm.collective_rpc('demoncat_capabilities', timeout=60)
        baseline = token_record(llm)
        record['baseline'] = baseline
        write_private(args.out, record)
        if args.tp == 1:
            check_attachment(llm, args, records[0], record)
        after = token_record(llm)
        record['after'] = after
        if after != baseline or identity_set(llm, args.tp) != records:
            raise RuntimeError('SAME_WORKER_BASELINE_NOT_RESTORED')
        record['status'] = 'PASS'
    except Exception as exc:
        record['status'] = 'FAIL'
        record['error'] = {'type': type(exc).__name__, 'message': str(exc)}
        raise
    finally:
        write_private(args.out, record)


def check_attachment(llm, args, worker, record):
    attached = rpc_one(llm, 'demoncat_attach', {'tenant_id': TENANT, 'worker': worker})
    if attached['status'] != 'ATTACHED':
        raise RuntimeError(f'ATTACH_FAILED: {attached}')
    ticket = attached['ticket']
    record['attachment_ticket'] = ticket  # Private receipt for the actual restart probe.
    if args.case == 'restart':
        if args.previous is None:
            raise RuntimeError('PREVIOUS_WORKER_RECEIPT_REQUIRED')
        previous = json.loads(args.previous.read_text())
        old = previous['attachment_ticket']
        if old['worker']['process_start_identity'] == worker['process_start_identity']:
            raise RuntimeError('WORKER_PROCESS_DID_NOT_CHANGE')
        stale = rpc_one(llm, 'demoncat_detach', old)
        if stale.get('status') != 'REJECTED' or stale.get('reason') != 'WORKER_IDENTITY_MISMATCH':
            raise RuntimeError('STALE_TENANT_WAS_ACCEPTED')
        record['previous_attachment_rejection'] = stale
        if 'owner_ticket' in previous:
            stale_owner = rpc_one(llm, 'demoncat_fault_query', previous['owner_ticket'])
            if (stale_owner.get('status') != 'REJECTED'
                    or stale_owner.get('reason') != 'WORKER_IDENTITY_MISMATCH'):
                raise RuntimeError('STALE_OWNER_WAS_ACCEPTED')
            record['previous_owner_rejection'] = stale_owner
        if previous['baseline'] != record['baseline']:
            raise RuntimeError('RESTART_WORKLOAD_OUTPUT_CHANGED')

    def prepare(mechanism):
        payload = dict(ticket, mechanism=mechanism, implementation_version='1',
                       owner='validation-' + mechanism, settings={})
        return rpc_one(llm, 'demoncat_fault_rpc', 'prepare', payload)

    if args.case == 'identity':
        absent = {mechanism: prepare(mechanism) for mechanism in MECHANISMS.values()}
        if any(row['status'] != 'NATIVE_PRELOAD_NOT_ATTACHED' for row in absent.values()):
            raise RuntimeError(f'EXPECTED_NATIVE_PRELOAD_REFUSAL: {absent}')
        record['no_preload'] = absent
    if args.case in MECHANISMS:
        prepared = prepare(MECHANISMS[args.case])
        record['provider_prepare'] = prepared
        if prepared['status'] != 'ACTION_COMPLETED':
            raise RuntimeError(f'NATIVE_PREPARE_FAILED: {prepared}')
        owner_ticket = prepared['ticket']
        record['owner_ticket'] = owner_ticket
        state = rpc_one(llm, 'demoncat_fault_query', owner_ticket)
        record['native_before'] = state
        if state['status'] != 'QUERY_OK':
            raise RuntimeError(f'LIVE_NATIVE_STATE_UNAVAILABLE: {state}')
        native = state['native']
        if (native['native_loaded'] is not True or native['enabled'] is not False
                or native['registered'] is not False or native['native_registered'] is not False
                or native['leases'] != 0 or native['inflight'] != 0):
            raise RuntimeError(f'PREPARED_NATIVE_NOT_IDLE: {native}')
        if token_record(llm) != record['baseline']:
            raise RuntimeError('DISARMED_PROVIDER_CHANGED_REQUEST')
        # This serving probe deliberately has no trusted replay callback. Verify
        # that the bridge refuses to arm in an ordinary Worker installation.
        refusal = rpc_one(llm, 'demoncat_fault_rpc', 'register', owner_ticket)
        record['unauthorized_register'] = refusal
        if (refusal['status'] != 'PROVIDER_ACTION_FAILED'
                or refusal.get('provider_code') != REPLAY_REFUSALS[args.case]):
            raise RuntimeError(f'UNQUALIFIED_REGISTRATION_NOT_REFUSED: {refusal}')

    detached = rpc_one(llm, 'demoncat_detach', ticket)
    record['detach'] = detached
    if detached['status'] != 'DETACHED':
        raise RuntimeError(f'DETACH_UNVERIFIED: {detached}')


if __name__ == '__main__':
    main()
