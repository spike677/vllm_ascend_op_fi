"""Temporary serving acceptance: Worker quarantine or read-only Fault20 discovery.

The parent controller owns card gates, resource locks and this isolated process.
Uses vLLM's standard LLM/Worker lifecycle and existing collective RPC, never a
replacement Worker class or patched execute_model. These cases do not inject.
"""
import argparse
from collections import Counter
import hashlib
import inspect
import json
from pathlib import Path
import threading

from experiments.worker_extension_20261116.probe import (
    EXTENSION, MECHANISMS, PROMPT, REPLAY_REFUSALS, identity_set, rpc_one, write_private,
)


def memory_utilization(value):
    value = float(value)
    if not 0.1 <= value <= 0.85:
        raise argparse.ArgumentTypeError('memory utilization must be between 0.1 and 0.85')
    return value


def request_record(llm):
    from vllm import SamplingParams
    row, = llm.generate([PROMPT], SamplingParams(temperature=0, max_tokens=1), use_tqdm=False)
    output, = row.outputs
    return dict(prompt_sha256=hashlib.sha256(json.dumps(row.prompt_token_ids).encode()).hexdigest(),
                tokens=list(output.token_ids), finish_reason=output.finish_reason)


def observe_start(worker, operator):
    """Read-only runtime observer inside the actual already initialized Worker."""
    from fault_injection.deployments.matmul import CtypesMatmulRuntime
    from fault_injection.native import NativeSession
    from demoncat.runtime.vllm_ascend.worker_extension.providers.fault20 import Fault20Provider
    if getattr(worker, '_demoncat_overnight_observer', None) is not None:
        raise RuntimeError('OBSERVATION_ALREADY_OWNED')
    context = worker.demoncat_worker_identity()
    if context['status'] != 'WORKER_IDENTIFIED' or context.get('quarantined'):
        raise RuntimeError('WORKER_NOT_AVAILABLE_FOR_OBSERVATION')
    model = worker.model_runner.model
    facts = Fault20Provider(worker, model, context['worker'])._worker_facts()
    evidence = Path('/evidence') / ('observation-rank' + str(facts['rank']))
    runtime = CtypesMatmulRuntime('20.buffer_substitute', facts, evidence, operator=operator)
    session = NativeSession(model, '20.buffer_substitute', arm='observe', runtime=runtime)
    session.__enter__()
    worker._demoncat_overnight_observer = (session, runtime, evidence)
    return dict(rank=facts['rank'], status='READONLY_OBSERVER_INSTALLED', profile=runtime.build_profile,
                install_thread=threading.get_ident(), install_native_thread=threading.get_native_id())


def observe_finish(worker):
    session, runtime, evidence = worker._demoncat_overnight_observer
    # Keep the ownership reference if close fails; the parent must isolate the
    # temporary process instead of pretending all native callbacks are gone.
    session.close()
    report = session.report()
    snapshots = runtime.snapshot()
    target = evidence.parent / (evidence.name + '.json')
    write_private(target, dict(report=report, native=snapshots))
    events = [e for e in report['events'] if e.get('native_facts')]
    signatures = Counter(e['native_facts']['native_signature'] for e in events)
    unique_modules = sorted({e['path'] for e in events})
    idle = all(not s['native_registered'] and not s['inflight'] and not s['leases'] for s in snapshots.values())
    if not idle or report['cleanup_pending']:
        raise RuntimeError('NATIVE_OBSERVATION_DID_NOT_DRAIN')
    worker._demoncat_overnight_observer = None
    return dict(status='NATIVE_OPERATOR_OBSERVED' if events else 'NO_SUPPORTED_NATIVE_OPERATOR_OBSERVED',
                rank=worker.rank, operator=runtime.operator, matching_modules=len(unique_modules),
                actual_calls=len(events), signatures=dict(signatures), model_forwards=len(report['forwards']),
                finish_thread=threading.get_ident(), finish_native_thread=threading.get_native_id(),
                same_thread_observed=bool(events and report['forwards'] and all(
                    row['status'] == 'returned' and not row['errors'] for row in report['forwards'])),
                thread_proof='NativeSession root and ctypes callback require install owner thread',
                native_idle=True, fault_calls=0, qualification='DISCOVERY_ONLY_NOT_LOSSLESS_OR_INJECTION',
                report=str(target), report_sha256=hashlib.sha256(target.read_bytes()).hexdigest())


def quarantine_case(llm, case, record):
    kind = case.removeprefix('quarantine-')
    mechanism = MECHANISMS[kind]
    identity = record['workers'][0]
    attached = rpc_one(llm, 'demoncat_attach', dict(tenant_id='overnight-quarantine', worker=identity))
    if attached['status'] != 'ATTACHED':
        raise RuntimeError('ATTACH_FAILED')
    tenant = attached['ticket']
    prepared = rpc_one(llm, 'demoncat_fault_rpc', 'prepare',
                       dict(tenant, mechanism=mechanism, implementation_version='1', owner='overnight'))
    record['prepared'] = prepared
    if prepared['status'] != 'ACTION_COMPLETED':
        raise RuntimeError('MATCHED_NATIVE_PRELOAD_REQUIRED')
    ticket = prepared['ticket']
    try:
        rejected = rpc_one(llm, 'demoncat_fault_rpc', 'register', ticket)
        record['register'] = rejected
        if rejected.get('provider_code') != REPLAY_REFUSALS[kind] or not rejected.get('recovery_required'):
            raise RuntimeError('EXPECTED_TRUSTED_REPLAY_REFUSAL_NOT_OBSERVED')
        other = MECHANISMS['fault39' if kind == 'fault20' else 'fault20']
        blocked = rpc_one(llm, 'demoncat_fault_rpc', 'prepare',
                          dict(tenant, mechanism=other, implementation_version='1', owner='other'))
        record['other_provider'] = blocked
        if blocked.get('reason') != 'WORKER_RECOVERY_REQUIRED':
            raise RuntimeError('UNRESOLVED_WORKER_ADMITTED_ANOTHER_PROVIDER')
        state = rpc_one(llm, 'demoncat_worker_identity')
        if state.get('quarantined') is not True:
            raise RuntimeError('WORKER_QUARANTINE_NOT_REPORTED')
    finally:
        recovered = rpc_one(llm, 'demoncat_fault_rpc', 'unregister', ticket)
        record['recovery'] = recovered
        if recovered['status'] != 'ACTION_COMPLETED':
            raise RuntimeError('WORKER_RECOVERY_NOT_CONFIRMED')
        detached = rpc_one(llm, 'demoncat_detach', tenant)
        record['detach'] = detached
        if detached['status'] != 'DETACHED' or rpc_one(llm, 'demoncat_worker_identity').get('quarantined'):
            raise RuntimeError('WORKER_REMAINS_QUARANTINED')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case', choices=('quarantine-fault20', 'quarantine-fault39',
                        'observe-fault20-weightnz', 'observe-fault20-qmm'), required=True)
    parser.add_argument('--tp', type=int, choices=(1, 2), default=1)
    parser.add_argument('--memory-utilization', type=memory_utilization, default=0.35)
    parser.add_argument('--model', default='/model')
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.case.startswith('quarantine-') and args.tp != 1:
        parser.error('Quarantine mutation acceptance is TP1 only')
    from demoncat.runtime.vllm_ascend.worker_extension import runtime
    from vllm import LLM
    record = dict(schema='demoncat.overnight.worker_probe.v1', status='STARTING', case=args.case,
                  actual_fault_injection=False, tp=args.tp, memory_utilization=args.memory_utilization,
                  runtime_file=inspect.getfile(runtime),
                  runtime_sha256=hashlib.sha256(Path(inspect.getfile(runtime)).read_bytes()).hexdigest())
    write_private(args.out, record)
    try:
        llm = LLM(model=args.model, dtype='bfloat16', tensor_parallel_size=args.tp,
                  enforce_eager=True, max_model_len=128, max_num_seqs=1,
                  enable_prefix_caching=False, gpu_memory_utilization=args.memory_utilization,
                  worker_extension_cls=EXTENSION, disable_log_stats=True)
        record['workers'] = identity_set(llm, args.tp)
        record['baseline'] = request_record(llm)
        if args.case.startswith('quarantine-'):
            quarantine_case(llm, args.case, record)
        else:
            operator = 'aclnnMatmulWeightNz' if args.case.endswith('weightnz') else 'aclnnQuantMatmulWeightNz'
            record['observer_start'] = llm.collective_rpc(observe_start, args=(operator,), timeout=90)
            try:
                record['observed_requests'] = [request_record(llm) for _ in range(2)]
            finally:
                record['discovery'] = llm.collective_rpc(observe_finish, timeout=90)
        record['after'] = request_record(llm)
        if record['baseline'] != record['after'] or identity_set(llm, args.tp) != record['workers']:
            raise RuntimeError('NORMAL_MODEL_OR_WORKER_IDENTITY_CHANGED')
        if any(row != record['baseline'] for row in record.get('observed_requests', [])):
            raise RuntimeError('READONLY_OBSERVER_CHANGED_MODEL_OUTPUT')
        record['status'] = 'PASS_QUARANTINE_RECOVERY' if args.case.startswith('quarantine-') else 'DISCOVERY_COMPLETED'
    except BaseException as exc:
        record['status'] = 'FAILED_WITH_EVIDENCE'
        record['error'] = dict(type=type(exc).__name__, message=str(exc))
        raise
    finally:
        write_private(args.out, record)


if __name__ == '__main__':
    main()
