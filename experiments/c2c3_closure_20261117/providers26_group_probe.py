"""Root-queued native-v2 candidates or conditional same-Worker C2 then C3."""
import argparse
import hashlib
import inspect
import json
from pathlib import Path
import traceback

PROMPT = 'Explain in one sentence why the sky appears blue during the day.'
SINGLE_SITE_STAGE = 'c1_single_effects'


def request_limit(stage, max_targets):
    return 8 if stage == SINGLE_SITE_STAGE else 9 if stage == 'coverage' else 5 * max_targets + 4


def actual(worker):
    from vllm.v1.worker.worker_base import WorkerBase, WorkerWrapperBase
    if isinstance(worker, WorkerWrapperBase):
        worker = worker.worker
    if not isinstance(worker, WorkerBase):
        raise RuntimeError('ACTUAL_STANDARD_WORKER_REQUIRED')
    return worker


def setup(worker, tokens, mechanism, manifest_sha256, stage, max_targets, statistics,
          evidence_dir='/evidence/writer-v2', trigger='once'):
    from demoncat.runtime.vllm_ascend.worker_extension import serving26_group
    from fault_injection.deployments import writer26_group, writer26_experiment, writer26_public
    sources = []
    modules = [serving26_group, writer26_group, writer26_experiment, writer26_public]
    if stage == 'coverage':
        from fault_injection.deployments import writer26_coverage
        modules.append(writer26_coverage)
    for module in modules:
        path = Path(inspect.getfile(module)).resolve()
        if not path.is_relative_to(Path('/installed')):
            raise RuntimeError('FRESH_INSTALLED_V2_WRITER_SOURCE_REQUIRED')
        sources.append(dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    worker = actual(worker)
    result = serving26_group.install_writer26_group_experiment(worker, mechanism=mechanism,
        prompt_token_ids=tokens, evidence_dir=evidence_dir, manifest_sha256=manifest_sha256,
        stage=stage, max_targets=max_targets, statistics=statistics, trigger=trigger)
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
    parser.add_argument('--stage', choices=('inventory', 'lossless', 'effects', 'coverage', SINGLE_SITE_STAGE), default='inventory')
    parser.add_argument('--max-targets', type=int, default=8)
    parser.add_argument('--statistics', action='store_true')
    parser.add_argument('--memory', type=float, default=.55)
    args = parser.parse_args()
    if not 1 <= args.max_targets <= 128 or not 0 < args.memory < 1:
        parser.error('Explicit bounded target and memory budgets required')
    if args.stage == SINGLE_SITE_STAGE and args.mechanism != '26.residual_stomp':
        parser.error('c1_single_effects is an explicit QMM-only single-site experiment')
    report = dict(status='STARTING', scope=('ALL_ELIGIBLE_NATIVE_WRITERS_FIXED_SINGLE_REQUEST_FRESH_PREFILL_NOT_DECODE'
        if args.stage == 'coverage' else 'ONE_ACTUAL_QMM_SITE_FIXED_FRESH_PREFILL_NOT_ALL_ELIGIBLE_NOT_C2_C3'
        if args.stage == SINGLE_SITE_STAGE else 'SINGLE_FRESH_PREFILL_CANDIDATE_WINDOWS_NOT_ALL_ELIGIBLE_C2'),
        mechanism=args.mechanism, stage=args.stage, native_qualified_targets=0, c1=False, c2=False, c3=False,
        e5='UNVERIFIED_INDEPENDENT_RAW_B_F_COMPARISON_REQUIRED', phases=[], sessions={}, num_gpu_blocks_override=64)
    def save():
        Path('/evidence/result.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    llm, installed, seen = None, False, []
    try:
        save()
        from fault_injection.deployments.writer26_loader import StartupDependencies
        startup = StartupDependencies(bundle='/native', manifest_sha256=args.manifest_sha256)
        report['metadata_startup'] = startup.evidence
        save()
        from vllm import LLM, SamplingParams
        from demoncat.runtime.vllm_ascend.worker_extension.serving_context import require_host_driven_core, read_enqueued_identity
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
        report['identity'] = rpc(llm, 'demoncat_worker_identity')
        report['actual_prompt_token_ids'] = tokens
        sessions = [('c2', 'once'), ('c3', 'every_match')] if args.stage == 'coverage' else [('candidate', 'once')]
        for name, trigger in sessions:
            if name == 'c3' and not report['c2']:
                break
            folder = '/evidence/' + (name if args.stage == 'coverage' else 'writer-v2')
            session = dict(trigger=trigger, phases=[])
            report['sessions'][name] = session
            session['setup'] = rpc(llm, setup, tokens, args.mechanism, args.manifest_sha256, args.stage,
                args.max_targets, args.statistics, folder, trigger)
            installed = True
            # The coverage plan is 8 C2 or 9 C3 complete fresh forwards. No
            # target subset or additional fault request is chosen by this loop.
            for index in range(request_limit(args.stage, args.max_targets)):
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
                row = dict(session=name, index=index, phase=state['phase'], mapping=mapping, advance=rpc(llm, advance, ids[0]))
                report['phases'].append(row)
                session['phases'].append(row)
                save()
                outputs = llm.wait_for_completion(use_tqdm=False)
                if len(outputs) != 1 or outputs[0].request_id != mapping['external_request_id']:
                    raise RuntimeError('ACTUAL_REQUEST_MAPPING_CHANGED')
                row['result'] = rpc(llm, query)
                row['generated_tokens'] = list(outputs[0].outputs[0].token_ids)
                save()
            state = rpc(llm, query)
            if state['phase'] != 'complete' or not state['recovered']:
                raise RuntimeError('BOUNDED_WRITER_PLAN_OR_SAME_PROCESS_HEALTH_INCOMPLETE')
            session['remove'] = rpc(llm, remove)
            installed = False
            final = session['remove']['report']
            if args.stage == 'coverage':
                report[name] = final[name] is True
                if report[name]:
                    report['native_qualified_targets'] = final['coverage']['planned_targets']
            elif args.stage == SINGLE_SITE_STAGE:
                report['single_site_c1'] = final['single_site_c1']
            save()
        after, = llm.generate([prompt], sampling, use_tqdm=False)
        report['healthy_after_tokens'] = list(after.outputs[0].token_ids)
        if report['healthy_before_tokens'] != report['healthy_after_tokens']:
            raise RuntimeError('POST_RELEASE_STANDARD_WORKER_GENERATION_CHANGED')
        if args.stage == 'coverage':
            report['status'] = 'C2_C3_EVIDENCE_READY_FOR_REVIEW' if report['c2'] and report['c3'] else 'COMPLETE_NATIVE_COVERAGE_BLOCKED'
        elif args.stage == SINGLE_SITE_STAGE:
            report['status'] = ('SINGLE_SITE_C1_EFFECTS_EVIDENCE_READY_FOR_REVIEW_NOT_C2'
                if report['single_site_c1']['evidence_complete'] else 'NATURAL_INVENTORY_RECORDED_DEPENDENT_PHASE_BLOCKED')
        else:
            report['status'] = 'NATIVE_CANDIDATE_EXPERIMENT_RECORDED_NOT_C2' if not state['blockers'] else 'NATURAL_INVENTORY_RECORDED_DEPENDENT_PHASE_BLOCKED'
    except BaseException as error:
        report.update(status='FAILED_WITH_EVIDENCE', error=repr(error), traceback=traceback.format_exc())
        if isinstance(getattr(error, 'evidence', None), dict):
            report['metadata_startup_failure'] = error.evidence
    finally:
        if installed and llm is not None:
            try:
                report['cleanup'] = rpc(llm, remove)
                coverage = report['cleanup']['report'].get('coverage')
                if coverage is not None and not coverage['recovery']['complete']:
                    report['quarantine_required'] = True
            except BaseException as error:
                report['cleanup_error'] = repr(error)
                report['quarantine_required'] = True
        save()
    return 0 if report['status'] in ('NATIVE_CANDIDATE_EXPERIMENT_RECORDED_NOT_C2',
                                     'NATURAL_INVENTORY_RECORDED_DEPENDENT_PHASE_BLOCKED', 'C2_C3_EVIDENCE_READY_FOR_REVIEW',
                                     'SINGLE_SITE_C1_EFFECTS_EVIDENCE_READY_FOR_REVIEW_NOT_C2') else 2


if __name__ == '__main__':
    raise SystemExit(main())
