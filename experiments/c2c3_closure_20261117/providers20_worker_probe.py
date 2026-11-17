"""Root-queued natural DeepSeek QMM C2, then optional same-Worker C3; no C1 replay."""
import argparse
import hashlib
import inspect
import json
from pathlib import Path
import traceback

EXTENSION = 'demoncat.runtime.vllm_ascend.controller.DemoncatWorkerExtension'
PROMPT = 'Hello.'


def rpc(llm, method, *args):
    result = llm.collective_rpc(method, args=args, timeout=300)
    if len(result) != 1:
        raise RuntimeError('ONE_ACTUAL_WORKER_REPLY_REQUIRED')
    return result[0]


def actual_worker(worker):
    from vllm.v1.worker.worker_base import WorkerBase, WorkerWrapperBase
    if isinstance(worker, WorkerWrapperBase):
        worker = worker.worker
    if not isinstance(worker, WorkerBase):
        raise RuntimeError('ACTUAL_STANDARD_WORKER_REQUIRED')
    return worker


def setup(worker, tokens, statistics, trigger):
    worker = actual_worker(worker)
    from demoncat.runtime.vllm_ascend.worker_extension.serving20_coverage import install_fault20_coverage_binding
    import demoncat.runtime.vllm_ascend.worker_extension.quant_group as adapter
    import fault_injection.coverage_session as coverage
    modules = []
    for module in (install_fault20_coverage_binding, adapter, coverage):
        path = Path(inspect.getfile(module)).resolve()
        if not path.is_relative_to(Path('/installed')):
            raise RuntimeError('ACTUAL_ADAPTER_NOT_FROM_INSTALLED_WHEEL')
        modules.append(dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    if any('execute_model' in vars(obj) or '_prepare_inputs' in vars(obj)
           for obj in (worker, worker.model_runner)):
        raise RuntimeError('CORE_METHOD_INSTANCE_OVERRIDE')
    return dict(binding=install_fault20_coverage_binding(worker, prompt_token_ids=tokens,
        evidence_dir='/evidence/' + ('c2' if trigger == 'once' else 'c3'),
        collect_statistics=statistics, trigger=trigger), imported_sources=modules)


def remove(worker):
    from demoncat.runtime.vllm_ascend.worker_extension.serving20 import remove_fault20_serving_binding
    return remove_fault20_serving_binding(actual_worker(worker))


def response(row):
    return dict(external_request_id=row.request_id, prompt_token_ids=row.prompt_token_ids,
        outputs=[dict(tokens=list(x.token_ids), text=x.text, finish_reason=x.finish_reason) for x in row.outputs])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='/model')
    parser.add_argument('--statistics', action='store_true')
    parser.add_argument('--c3', action='store_true')
    parser.add_argument('--metadata-preflight', action='store_true', help='Only two natural discovery requests; never fault or C2/C3')
    parser.add_argument('--memory', type=float, default=.55)
    parser.add_argument('--out', type=Path, default=Path('/evidence/result.json'))
    args = parser.parse_args()
    if args.metadata_preflight and (args.c3 or args.statistics):
        parser.error('Metadata preflight has no fault/statistics/C3 phase')
    from vllm import LLM, SamplingParams
    from demoncat.runtime.vllm_ascend.worker_extension.quant_group import PHASES
    from demoncat.runtime.vllm_ascend.worker_extension.serving_context import require_host_driven_core, read_enqueued_identity
    report = dict(status='STARTING', scope='NATURAL_TP1_FIXED_SHORT_FRESH_PREFILL_ALL_NATIVE_ELIGIBLE_QMM',
                  mechanisms=['20.buffer_substitute'], c2=False, c3=False, attempts=[], statistics=args.statistics,
                  e5='UNVERIFIED_RAW_BF_COMPARISON_REQUIRED', metadata_preflight_only=args.metadata_preflight)
    def save():
        args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    llm = None
    ticket = tenant = None
    seen = []
    save()
    try:
        llm = LLM(model=args.model, dtype='bfloat16', tensor_parallel_size=1, enforce_eager=True,
            max_model_len=128, max_num_seqs=1, max_num_batched_tokens=128,
            enable_prefix_caching=False, enable_chunked_prefill=False, async_scheduling=False,
            gpu_memory_utilization=args.memory, worker_extension_cls=EXTENSION, disable_log_stats=True)
        require_host_driven_core(llm)
        tokens = llm.get_tokenizer().encode(PROMPT, add_special_tokens=True)
        if not 1 <= len(tokens) <= 4:
            raise RuntimeError('FIXED_NATURAL_PROMPT_EXCEEDS_UNCHANGED_NATIVE_X1_BYTE_SCOPE')
        report['actual_prompt_token_ids'] = tokens
        prompt = {'prompt_token_ids': tokens}
        sampling = SamplingParams(temperature=0, max_tokens=1, ignore_eos=True)
        baseline, = llm.generate([prompt], sampling, use_tqdm=False)
        report['healthy_before'] = response(baseline)
        identity = rpc(llm, 'demoncat_worker_identity')
        report['identity'] = identity
        attached = rpc(llm, 'demoncat_attach', dict(tenant_id='providers-c2c3-20', worker=identity['worker']))
        if attached['status'] != 'ATTACHED':
            raise RuntimeError(str(attached))
        tenant = attached['ticket']
        for trigger in (('once', 'every_match') if args.c3 else ('once',)):
            attempt = dict(trigger=trigger, phases=[], setup=rpc(llm, setup, tokens, args.statistics, trigger))
            report['attempts'].append(attempt)
            prepared = rpc(llm, 'demoncat_fault_rpc', 'prepare', dict(tenant, mechanism='20.buffer_substitute',
                implementation_version='1', owner='providers-c2c3-20', settings={'operator': 'aclnnQuantMatmulWeightNz'}))
            attempt['prepare'] = prepared
            if prepared['status'] != 'ACTION_COMPLETED':
                raise RuntimeError(str(prepared))
            ticket = prepared['ticket']
            for action in ('register', 'check', 'execute'):
                payload = ticket
                reply = rpc(llm, 'demoncat_fault_rpc', action, payload)
                attempt[action] = reply
                save()
                if reply['status'] != 'ACTION_COMPLETED':
                    raise RuntimeError(str(reply))
            phases = list(PHASES[:2] if args.metadata_preflight else PHASES)
            if trigger == 'every_match':
                phases.insert(phases.index('off_observed'), 'fault_repeat')
            for phase in phases:
                if llm.llm_engine.has_unfinished_requests():
                    raise RuntimeError('UNRELATED_OR_UNFINISHED_REQUEST')
                if llm.reset_prefix_cache(reset_running_requests=False) is not True:
                    raise RuntimeError('CACHE_RESET_NOT_CONFIRMED')
                ids = llm.enqueue([prompt], sampling, use_tqdm=False)
                mapping = read_enqueued_identity(llm, ids, seen_internal_request_ids=seen)
                seen.append(ids[0])
                row = dict(phase=phase, mapping=mapping)
                attempt['phases'].append(row)
                row['advance'] = rpc(llm, 'demoncat_fault_rpc', 'advance', dict(ticket, settings={'request_id': ids[0]}))
                save()
                if row['advance']['status'] != 'ACTION_COMPLETED':
                    raise RuntimeError(str(row['advance']))
                outputs = llm.wait_for_completion(use_tqdm=False)
                if len(outputs) != 1:
                    raise RuntimeError('ONE_COMPLETE_REAL_REQUEST_REQUIRED')
                row['response'] = response(outputs[0])
                if row['response']['external_request_id'] != mapping['external_request_id']:
                    raise RuntimeError('ACTUAL_DRIVER_REQUEST_MAPPING_CHANGED')
                row['query'] = rpc(llm, 'demoncat_fault_query', ticket)
                save()
                if row['query'].get('quarantined'):
                    raise RuntimeError('GROUP_ATTEMPT_FAILED')
            if args.metadata_preflight:
                qualification = row['query']['provider_state']['qualification']
                if qualification.get('targets', 0) <= 0 or not qualification.get('metadata_array_activity'):
                    raise RuntimeError('METADATA_DISCOVERY_PREFLIGHT_INCOMPLETE')
                attempt['metadata_activity'] = qualification['metadata_array_activity']
                attempt['unregister'] = rpc(llm, 'demoncat_fault_rpc', 'unregister', ticket)
                if attempt['unregister']['status'] != 'ACTION_COMPLETED':
                    raise RuntimeError('PREFLIGHT_OWNER_NOT_RELEASED')
                ticket = None
                attempt['unbind'] = rpc(llm, remove)
                healthy, = llm.generate([prompt], sampling, use_tqdm=False)
                attempt['healthy_after'] = response(healthy)
                if attempt['healthy_after']['outputs'] != report['healthy_before']['outputs']:
                    raise RuntimeError('PREFLIGHT_HEALTHY_CONTROL_CHANGED')
                break
            attempt['finish'] = rpc(llm, 'demoncat_fault_rpc', 'finish', ticket)
            if attempt['finish']['status'] != 'ACTION_COMPLETED':
                raise RuntimeError('GROUP_FINISH_NOT_CONFIRMED')
            qualification = attempt['finish']['native']['qualification']
            if (not qualification['recovery_verified'] or not qualification['coverage']['contract_satisfied']
                    or not qualification['coverage']['real_worker_evidence']):
                raise RuntimeError('GROUP_COVERAGE_OR_SAME_PROCESS_RECOVERY_UNPROVEN')
            # Contract receipts remain subject to the integrator's actual-evidence review.
            report['c2' if trigger == 'once' else 'c3'] = True
            attempt['unregister'] = rpc(llm, 'demoncat_fault_rpc', 'unregister', ticket)
            if attempt['unregister']['status'] != 'ACTION_COMPLETED':
                raise RuntimeError('NATIVE_OWNER_NOT_RELEASED')
            ticket = None
            attempt['unbind'] = rpc(llm, remove)
            healthy, = llm.generate([prompt], sampling, use_tqdm=False)
            attempt['healthy_after'] = response(healthy)
            if attempt['healthy_after']['outputs'] != report['healthy_before']['outputs']:
                raise RuntimeError('SAME_PROCESS_GENERATION_NOT_RESTORED')
            save()
        report['detach'] = rpc(llm, 'demoncat_detach', tenant)
        tenant = None
        report['status'] = ('METADATA_PREFLIGHT_EVIDENCE_READY_FOR_REVIEW' if args.metadata_preflight else
                            'C2_C3_NATIVE_EVIDENCE_READY_FOR_REVIEW' if args.c3 else 'C2_NATIVE_EVIDENCE_READY_FOR_REVIEW')
    except BaseException as error:
        report.update(status='FAILED_WITH_EVIDENCE', error=repr(error), traceback=traceback.format_exc())
        if llm is not None and seen:
            try:
                llm.llm_engine.abort_request(seen, internal=True)
                report['owned_request_abort'] = dict(ids=seen, unfinished=llm.llm_engine.has_unfinished_requests())
            except BaseException as abort_error:
                report['owned_request_abort_error'] = repr(abort_error)
        if llm is not None and ticket is not None:
            try:
                report['failure_query'] = rpc(llm, 'demoncat_fault_query', ticket)
                report['unregister_after_failure'] = rpc(llm, 'demoncat_fault_rpc', 'unregister', ticket)
            except BaseException as cleanup_error:
                report['cleanup_error'] = repr(cleanup_error)
        # An incomplete target OFF leaves the Worker quarantined. The root
        # queue disposes only its owned temporary container and keeps failure.
    finally:
        save()
    return 0 if report['status'].endswith('EVIDENCE_READY_FOR_REVIEW') else 2


if __name__ == '__main__':
    raise SystemExit(main())
