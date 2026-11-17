"""Root-queued v2 natural inventory and bounded per-target candidate B/L/F/OFF."""
import argparse
import hashlib
import inspect
import json
from pathlib import Path
import traceback

PROMPT = 'Explain in one sentence why the sky appears blue during the day.'


def actual(worker):
    from vllm.v1.worker.worker_base import WorkerBase, WorkerWrapperBase
    if isinstance(worker, WorkerWrapperBase):
        worker = worker.worker
    if not isinstance(worker, WorkerBase):
        raise RuntimeError('ACTUAL_STANDARD_WORKER_REQUIRED')
    return worker


def setup(worker, tokens, mechanism, manifest_sha256, stage, max_targets, statistics):
    from demoncat.runtime.vllm_ascend.worker_extension import serving26_group
    from fault_injection.deployments import writer26_group, writer26_experiment
    sources = []
    for module in (serving26_group, writer26_group, writer26_experiment):
        path = Path(inspect.getfile(module)).resolve()
        if not path.is_relative_to(Path('/installed')):
            raise RuntimeError('FRESH_INSTALLED_V2_WRITER_SOURCE_REQUIRED')
        sources.append(dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    worker = actual(worker)
    result = serving26_group.install_writer26_group_experiment(worker, mechanism=mechanism,
        prompt_token_ids=tokens, evidence_dir='/evidence/writer-v2', manifest_sha256=manifest_sha256,
        stage=stage, max_targets=max_targets, statistics=statistics)
    return dict(binding=result, sources=sources)


def advance(worker, request_id):
    session = actual(worker)._demoncat_writer26_group_experiment
    session.expect(request_id)
    return session.report()


def query(worker):
    return actual(worker)._demoncat_writer26_group_experiment.report()


def remove(worker):
    from demoncat.runtime.vllm_ascend.worker_extension.serving26_group import remove_writer26_group_experiment
    return remove_writer26_group_experiment(actual(worker))


def rpc(llm, function, *args):
    rows = llm.collective_rpc(function, args=args, timeout=300)
    if len(rows) != 1:
        raise RuntimeError('ONE_ACTUAL_WORKER_REQUIRED')
    return rows[0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mechanism', choices=('26.residual_stomp', '26.fia_destination'), required=True)
    parser.add_argument('--manifest-sha256', required=True)
    parser.add_argument('--stage', choices=('inventory', 'lossless', 'effects'), default='inventory')
    parser.add_argument('--max-targets', type=int, default=8)
    parser.add_argument('--statistics', action='store_true')
    parser.add_argument('--memory', type=float, default=.55)
    args = parser.parse_args()
    from vllm import LLM, SamplingParams
    from demoncat.runtime.vllm_ascend.worker_extension.serving_context import require_host_driven_core, read_enqueued_identity
    report = dict(status='STARTING', scope='SINGLE_FRESH_PREFILL_CANDIDATE_WINDOWS_NOT_ALL_ELIGIBLE_C2',
        mechanism=args.mechanism, stage=args.stage, native_qualified_targets=0, c1=False, c2=False, c3=False,
        e5='UNVERIFIED_INDEPENDENT_RAW_B_F_COMPARISON_REQUIRED', phases=[], num_gpu_blocks_override=64)
    def save():
        Path('/evidence/result.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    llm, installed, seen = None, False, []
    try:
        save()
        llm = LLM(model='/model', dtype='bfloat16', tensor_parallel_size=1, enforce_eager=True,
            max_model_len=128, max_num_seqs=1, max_num_batched_tokens=128, num_gpu_blocks_override=64,
            enable_prefix_caching=False, enable_chunked_prefill=False, async_scheduling=False,
            gpu_memory_utilization=args.memory, seed=42,
            worker_extension_cls='demoncat.runtime.vllm_ascend.controller.DemoncatWorkerExtension', disable_log_stats=True)
        require_host_driven_core(llm)
        tokens = llm.get_tokenizer().encode(PROMPT, add_special_tokens=True)
        prompt = {'prompt_token_ids': tokens}
        sampling = SamplingParams(temperature=0, max_tokens=1, ignore_eos=True)
        before, = llm.generate([prompt], sampling, use_tqdm=False)
        report['healthy_before_tokens'] = list(before.outputs[0].token_ids)
        report['setup'] = rpc(llm, setup, tokens, args.mechanism, args.manifest_sha256, args.stage, args.max_targets, args.statistics)
        report['identity'] = rpc(llm, 'demoncat_worker_identity')
        report['actual_prompt_token_ids'] = tokens
        installed = True
        for index in range(5 * args.max_targets + 4):
            state = rpc(llm, query)
            if state['phase'] == 'complete':
                break
            if state['failure']:
                raise RuntimeError('ACTUAL_NATIVE_WRITER_FAILURE:' + str(state['failure']))
            if llm.llm_engine.has_unfinished_requests() or llm.reset_prefix_cache(reset_running_requests=False) is not True:
                raise RuntimeError('IDLE_FRESH_PREFILL_REQUIRED')
            ids = llm.enqueue([prompt], sampling, use_tqdm=False)
            mapping = read_enqueued_identity(llm, ids, seen_internal_request_ids=seen)
            seen += ids
            row = dict(index=index, phase=state['phase'], mapping=mapping, advance=rpc(llm, advance, ids[0]))
            report['phases'].append(row)
            save()
            outputs = llm.wait_for_completion(use_tqdm=False)
            if len(outputs) != 1 or outputs[0].request_id != mapping['external_request_id']:
                raise RuntimeError('ACTUAL_REQUEST_MAPPING_CHANGED')
            row['result'] = rpc(llm, query)
            row['generated_tokens'] = list(outputs[0].outputs[0].token_ids)
            save()
        state = rpc(llm, query)
        if state['phase'] != 'complete' or not state['recovered']:
            raise RuntimeError('BOUNDED_CANDIDATE_PLAN_OR_SAME_PROCESS_HEALTH_INCOMPLETE')
        report['remove'] = rpc(llm, remove)
        installed = False
        after, = llm.generate([prompt], sampling, use_tqdm=False)
        report['healthy_after_tokens'] = list(after.outputs[0].token_ids)
        if report['healthy_before_tokens'] != report['healthy_after_tokens']:
            raise RuntimeError('POST_RELEASE_STANDARD_WORKER_GENERATION_CHANGED')
        report['status'] = 'NATIVE_CANDIDATE_EXPERIMENT_RECORDED_NOT_C2' if not state['blockers'] else 'NATURAL_INVENTORY_RECORDED_DEPENDENT_PHASE_BLOCKED'
    except BaseException as error:
        report.update(status='FAILED_WITH_EVIDENCE', error=repr(error), traceback=traceback.format_exc())
    finally:
        if installed and llm is not None:
            try:
                report['cleanup'] = rpc(llm, remove)
            except BaseException as error:
                report['cleanup_error'] = repr(error)
                report['quarantine_required'] = True
        save()
    return 0 if report['status'] in ('NATIVE_CANDIDATE_EXPERIMENT_RECORDED_NOT_C2',
                                     'NATURAL_INVENTORY_RECORDED_DEPENDENT_PHASE_BLOCKED') else 2


if __name__ == '__main__':
    raise SystemExit(main())
