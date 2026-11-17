"""Root-queued read-only census of a natural 2048-token Qwen prefill.

No native bundle or policy is loaded. This preparation probe establishes which
actual APIs/layouts the ordinary Worker uses; it cannot qualify either FIA28
mechanism. Only root may queue this script on an NPU after the run window opens.
"""
import argparse
import hashlib
import inspect
import json
import os
from pathlib import Path
import traceback

from attention33_worker_probe import worker_object, rpc, response

OWNER = '_demoncat_fia28_readonly_prefill_audit'
PROMPT = 'A clear explanation uses definitions, a concrete example, and a careful check of the result. '


def require(ok, reason):
    if not ok:raise RuntimeError(reason)


def setup(worker):
    worker = worker_object(worker)
    from demoncat.runtime.vllm_ascend.worker_extension import attention28_prefill
    require(Path(inspect.getfile(attention28_prefill)).resolve().is_relative_to('/installed'), 'INSTALLED_WHEEL_REQUIRED')
    require(not hasattr(worker, OWNER), 'FIA28_EXISTING_AUDIT_OWNER')
    for obj in (worker, worker.model_runner):
        require('execute_model' not in vars(obj), 'FIA28_CORE_METHOD_INSTANCE_OVERRIDE')
    session = attention28_prefill.FiaPrefillAudit(worker, max_calls=224)
    setattr(worker, OWNER, session);session.install()
    return dict(candidate_count=len(session.candidates), source=session.sources, injection_enabled=False)


def expect(worker, request_id, tokens):
    getattr(worker_object(worker), OWNER).expect(request_id, tokens)
    return dict(actual_internal_request_id=request_id)


def finish(worker):
    worker = worker_object(worker);session = getattr(worker, OWNER, None)
    if session is None:return dict(closed=True, session_present=False)
    result = dict(closed=False, session_present=True, audit=session.report())
    try:
        session.close();delattr(worker, OWNER);result['closed'] = True
    except BaseException as error:
        result.update(cleanup_error=repr(error), traceback=traceback.format_exc())
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', default='/model')
    p.add_argument('--out', type=Path, default=Path('/evidence/run'))
    p.add_argument('--memory', type=float, default=.35)
    a = p.parse_args()
    require(a.out.resolve().is_relative_to('/evidence') and not a.out.exists(), 'NEW_PRIVATE_EVIDENCE_DIRECTORY_REQUIRED')
    require(not os.environ.get('LD_PRELOAD') and not os.environ.get('ASCEND_CUSTOM_OPP_PATH')
            and not any(name.startswith(('FAULT28_', 'NES_FA_', 'DCAT33_TND_')) for name in os.environ),
            'STOCK_PROCESS_WITHOUT_NATIVE_POLICY_REQUIRED')
    a.out.mkdir(parents=True)
    report = dict(schema='demoncat.attention28.prefill_worker_probe.v1', status='STARTED', read_only=True,
        public_worker_qualified=False, c2=False, c3=False, native_policy_enabled=False,
        complete_tensor_comparison_performed=False)
    def save():(a.out / 'result.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    save();llm = None;installed = False;ids = []
    try:
        from vllm import LLM, SamplingParams
        from demoncat.runtime.vllm_ascend.worker_extension.serving_context import require_host_driven_core, read_enqueued_identity
        llm = LLM(model=a.model, dtype='float16', tensor_parallel_size=1, enforce_eager=True,
            max_model_len=2304, max_num_seqs=1, max_num_batched_tokens=2304, block_size=16,
            num_gpu_blocks_override=256, enable_prefix_caching=False, enable_chunked_prefill=False,
            async_scheduling=False, gpu_memory_utilization=a.memory, seed=42,
            worker_extension_cls='demoncat.runtime.vllm_ascend.controller.DemoncatWorkerExtension', disable_log_stats=True)
        require_host_driven_core(llm)
        tokens = llm.get_tokenizer().encode(PROMPT * 256, add_special_tokens=True)[:2048]
        require(len(tokens) == 2048, 'LEGITIMATE_2048_TOKEN_PREFILL_REQUIRED')
        prompt = dict(prompt_token_ids=tokens)
        sampling = SamplingParams(temperature=0, max_tokens=1, ignore_eos=True)
        report['prompt'] = dict(token_count=len(tokens), text_rule=PROMPT,
            token_sha256=hashlib.sha256(json.dumps(tokens, separators=(',', ':')).encode()).hexdigest())
        before, = llm.generate([prompt], sampling, use_tqdm=False)
        report['before'] = response(before);save()
        require(llm.reset_prefix_cache(reset_running_requests=False) is True, 'PREFIX_RESET_NOT_CONFIRMED')
        installed = True;report['setup'] = rpc(llm, setup)
        ids = llm.enqueue([prompt], sampling, use_tqdm=False)
        report['actual_request_mapping'] = read_enqueued_identity(llm, ids)
        require(len(ids) == 1, 'ONE_REAL_REQUEST_REQUIRED')
        report['expected'] = rpc(llm, expect, ids[0], tokens);save()
        actual, = llm.wait_for_completion(use_tqdm=False)
        report['observed'] = response(actual)
        require(actual.request_id == report['actual_request_mapping']['external_request_id'], 'ACTUAL_REQUEST_MAPPING_DIFFERS')
        report['finish'] = rpc(llm, finish);installed = not report['finish']['closed'];save()
        require(not installed, 'OWNED_AUDIT_CLEANUP_INCOMPLETE')
        after, = llm.generate([prompt], sampling, use_tqdm=False)
        report['after'] = response(after)
        report['generated_token_identity_unchanged'] = (report['before']['outputs'] == report['observed']['outputs'] == report['after']['outputs'])
        require(report['generated_token_identity_unchanged'], 'READ_ONLY_OBSERVER_CHANGED_GENERATION')
        audit = report['finish']['audit']
        require(audit['forward_count'] == 1 and audit['rows'] and not audit['errors'] and not audit['overflow_calls'],
                'NATURAL_PREFILL_OBSERVATION_INCOMPLETE')
        report['status'] = 'NATURAL_PREFILL_API_OBSERVED_NO_NATIVE_QUALIFICATION'
    except BaseException as error:
        report.update(status='FAILED_WITH_EVIDENCE', error=repr(error), traceback=traceback.format_exc())
        if llm is not None and ids:
            try:llm.llm_engine.abort_request(ids, internal=True)
            except BaseException as abort_error:report['abort_error'] = repr(abort_error)
    finally:
        if llm is not None and installed:
            try:report['failure_cleanup'] = rpc(llm, finish)
            except BaseException as error:report['cleanup_error'] = repr(error)
        save()
    print(json.dumps({k: report[k] for k in ('status', 'c2', 'c3')}))
    return 2 if report['status'] == 'FAILED_WITH_EVIDENCE' else 0


if __name__ == '__main__':raise SystemExit(main())
