"""Read-only natural Worker decode audit; no Fault33 injection or qualification."""
import argparse
import inspect
import json
from pathlib import Path
import traceback


PROMPT = 'Explain in one sentence why the sky appears blue during the day.'
OWNER_ATTRIBUTE = '_demoncat_paged33_readonly_audit'


def rpc(llm, method, *args):
    rows = llm.collective_rpc(method, args=args, timeout=300)
    if len(rows) != 1:raise RuntimeError('TP1_REQUIRED')
    return rows[0]


def worker_object(worker):
    from vllm.v1.worker.worker_base import WorkerWrapperBase
    return worker.worker if isinstance(worker, WorkerWrapperBase) else worker


def setup(worker):
    worker = worker_object(worker)
    from demoncat.runtime.vllm_ascend.worker_extension import paged33_audit
    if not Path(inspect.getfile(paged33_audit)).resolve().is_relative_to('/installed'):
        raise RuntimeError('INSTALLED_WHEEL_REQUIRED')
    if hasattr(worker, OWNER_ATTRIBUTE):raise RuntimeError('AUDIT_OWNER_EXISTS')
    observer = paged33_audit.Paged33Audit(worker, max_seqs=1)
    observer.install()
    setattr(worker, OWNER_ATTRIBUTE, observer)
    return observer.report()


def expect(worker, request_id, prompt):
    observer = getattr(worker_object(worker), OWNER_ATTRIBUTE)
    observer.expect({request_id: dict(logical_id=0, prompt=list(prompt))})
    return dict(expected_request_id=request_id)


def finish(worker):
    worker = worker_object(worker)
    observer = getattr(worker, OWNER_ATTRIBUTE)
    observer.close()
    result = observer.report()
    delattr(worker, OWNER_ATTRIBUTE)
    return result


def output(row):
    return dict(request_id=row.request_id, tokens=list(row.outputs[0].token_ids), text=row.outputs[0].text)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--memory', type=float, default=.35)
    args = parser.parse_args()
    report = dict(status='STARTED', diagnostic_only=True, injection_enabled=False, public_worker_qualified=False)
    llm = None
    installed = False
    def save():Path('/evidence/result.json').write_text(json.dumps(report, indent=2) + '\n')
    try:
        from vllm import LLM, SamplingParams
        from demoncat.runtime.vllm_ascend.worker_extension.serving_context import require_host_driven_core, read_enqueued_identity
        llm = LLM(model='/model', dtype='float16', tensor_parallel_size=1, enforce_eager=True,
                  max_model_len=128, max_num_seqs=1, max_num_batched_tokens=128,
                  num_gpu_blocks_override=64, enable_prefix_caching=False, enable_chunked_prefill=False,
                  async_scheduling=False, gpu_memory_utilization=args.memory, seed=42,
                  worker_extension_cls='demoncat.runtime.vllm_ascend.controller.DemoncatWorkerExtension',
                  disable_log_stats=True)
        require_host_driven_core(llm)
        tokens = llm.get_tokenizer().encode(PROMPT, add_special_tokens=True)
        prompt = dict(prompt_token_ids=tokens)
        params = SamplingParams(temperature=0, max_tokens=2, ignore_eos=True)
        before, = llm.generate([prompt], params, use_tqdm=False)
        report['healthy_before'] = output(before)
        report['setup'] = rpc(llm, setup)
        installed = True
        if llm.reset_prefix_cache(reset_running_requests=False) is not True:
            raise RuntimeError('RESET_NOT_CONFIRMED')
        ids = llm.enqueue([prompt], params, use_tqdm=False)
        report['mapping'] = read_enqueued_identity(llm, ids)
        report['expected'] = rpc(llm, expect, ids[0], tokens)
        observed, = llm.wait_for_completion(use_tqdm=False)
        report['observed_generation'] = output(observed)
        report['audit'] = rpc(llm, finish)
        installed = False
        after, = llm.generate([prompt], params, use_tqdm=False)
        report['healthy_after'] = output(after)
        if any(report['healthy_before'][key] != report['healthy_after'][key]
               or report['healthy_before'][key] != report['observed_generation'][key]
               for key in ('tokens', 'text')):
            raise RuntimeError('READ_ONLY_AUDIT_GENERATION_CHANGED')
        rows = report['audit']['rows']
        if not rows or any(not row.get('consumer_returned') for row in rows):
            raise RuntimeError('NO_COMPLETE_ACTUAL_DECODE_API')
        report['status'] = 'NATURAL_API_OBSERVED_INJECTION_REFUSED'
        report['all_observed_metadata_verified'] = all(row['metadata_binding'] == 'VERIFIED' for row in rows)
    except BaseException as error:
        report.update(status='DIAGNOSTIC_FAILED_WITH_EVIDENCE', error=repr(error), traceback=traceback.format_exc())
    finally:
        if installed:
            try:report['cleanup'] = rpc(llm, finish)
            except BaseException as error:report['cleanup_error'] = repr(error)
        save()
    return 0 if report['status'] == 'NATURAL_API_OBSERVED_INJECTION_REFUSED' else 2


if __name__ == '__main__':raise SystemExit(main())
