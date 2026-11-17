"""One fresh standard Worker, one complete raw statistics/mix request window.

Run the same command in three fresh owned containers: baseline, an independent
baseline repeat, and fault. All capture the fifth managed request after the
same warmup and four actual qualification requests. No raw field is filtered.
"""
import argparse
import json
from pathlib import Path
import sys
import traceback

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT/'src')]
from experiments.c2c3_closure_20261117.natural_rms_worker_probe import (
    actual, setup, advance, collect, qualify, cleanup, digest)
from experiments.c2c3_closure_20261117.natural_tensor_census import PROMPT, sha_file


def configure(worker, folder, arm):
    from experiments.c2c3_closure_20261117.tensor_statistics_capture import TensorStatisticsCapture
    session = actual(worker)._natural_rms_group
    capture = TensorStatisticsCapture(folder, session.provider.synchronize)
    return session.configure_statistics(capture, arm)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--arm', choices=('baseline', 'fault'), required=True)
    parser.add_argument('--memory', type=float, default=.35)
    args = parser.parse_args()
    if not .1 <= args.memory <= .8:
        parser.error('Bounded Worker memory fraction .1..8 required')
    args.out.mkdir(parents=True, exist_ok=False)
    report = dict(status='STARTED', mechanism='18.api_half', arm=args.arm,
        standard_vllm_worker=True, public_qualification_granted=False, coverage_granted=False,
        requests=[], sampling=dict(temperature=0, max_tokens=1, ignore_eos=True),
        capture_scope='Complete actual scheduled request, including logits and sampler; owned diagnostics paused',
        captured_managed_request_ordinal=5, captured_driver_request_ordinal=6,
        raw_statistics_rewritten=False, null_equal=False, rtol=1e-5, atol=1e-8)
    llm = None; installed = False; completed = False; seen = []
    def save():
        (args.out/'result.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    try:
        from vllm import LLM, SamplingParams
        from demoncat.runtime.vllm_ascend.worker_extension.serving_context import (
            require_host_driven_core, read_enqueued_identity)
        weights = {path.name: dict(sha256=sha_file(path), bytes=path.stat().st_size)
            for path in sorted({*args.model.glob('*.safetensors'), *args.model.glob('pytorch_model*.bin')})}
        if not weights:
            raise RuntimeError('ACTUAL_PRETRAINED_WEIGHTS_REQUIRED')
        llm = LLM(model=str(args.model), dtype='bfloat16', tensor_parallel_size=1, enforce_eager=True,
            max_model_len=128, max_num_seqs=1, max_num_batched_tokens=128, num_gpu_blocks_override=64,
            enable_prefix_caching=False, enable_chunked_prefill=False, async_scheduling=False,
            gpu_memory_utilization=args.memory, seed=42, disable_log_stats=True,
            worker_extension_cls='demoncat.runtime.vllm_ascend.controller.DemoncatWorkerExtension')
        require_host_driven_core(llm)
        def rpc(function, *values):
            rows = llm.collective_rpc(function, args=values, timeout=300)
            if len(rows) != 1:
                raise RuntimeError('TP1_SERVING_SCOPE_REQUIRED')
            return rows[0]
        tokens = llm.get_tokenizer().encode(PROMPT, add_special_tokens=True)
        prompt = dict(prompt_token_ids=tokens)
        params = SamplingParams(temperature=0, max_tokens=1, ignore_eos=True)
        healthy, = llm.generate([prompt], params, use_tqdm=False)
        report['healthy_before'] = list(healthy.outputs[0].token_ids)
        report['setup'] = rpc(setup, tokens, weights, digest(dict(weights=weights, tokens=tokens)))
        installed = True; save()
        def request(phase, arm=None):
            if llm.llm_engine.has_unfinished_requests():
                raise RuntimeError('UNRELATED_PENDING_REQUEST')
            if llm.reset_prefix_cache(reset_running_requests=False) is not True:
                raise RuntimeError('FRESH_PREFILL_CACHE_RESET_UNCONFIRMED')
            ids = llm.enqueue([prompt], params, use_tqdm=False)
            mapping = read_enqueued_identity(llm, ids, seen_internal_request_ids=seen)
            seen.extend(ids)
            rpc(advance, ids[0], phase, arm)
            responses = llm.wait_for_completion(use_tqdm=False)
            if len(responses) != 1 or responses[0].request_id != mapping['external_request_id']:
                raise RuntimeError('ACTUAL_DRIVER_WORKER_REQUEST_MAPPING_CHANGED')
            row = rpc(collect)
            output_tokens = list(responses[0].outputs[0].token_ids)
            report['requests'].append(dict(phase=phase, managed_ordinal=len(report['requests'])+1,
                mapping=mapping, worker=row, output_tokens=output_tokens,
                tokens_equal_healthy=output_tokens == report['healthy_before']))
            save()
        for phase, arm in (('native_baseline', None), ('baseline', 'baseline'),
                           ('baseline_repeat', 'baseline'), ('lossless', 'baseline')):
            request(phase, arm)
        report['qualification'] = rpc(qualify); save()
        report['capture_setup'] = rpc(configure, str(args.out/'statistics'), args.arm)
        # Both arms and the independent repeat execute this exact call site.
        request('statistics_request', args.arm)
        request('statistics_off')
        report['statistics'] = report['requests'][4]['worker']['statistics']
        raw = args.out/'statistics/observation/step0/rank0'
        report['raw_statistics'] = {name: dict(path=str(raw/name), sha256=sha_file(raw/name),
            bytes=(raw/name).stat().st_size) for name in ('dump.json', 'stack.json', 'construct.json')}
        report['independent_comparison'] = 'REQUIRED_ACROSS_FRESH_BASELINE_REPEAT_AND_FAULT_WORKERS'
        completed = True
        report['status'] = 'STANDARD_WORKER_RMS_RAW_CAPTURE_COMPLETE'
    except BaseException as exc:
        report.update(status='REFUSED_OR_FAILED', error=repr(exc), traceback=traceback.format_exc())
        if llm is not None and seen:
            try:
                llm.llm_engine.abort_request(seen, internal=True)
            except BaseException as error:
                report['abort_error'] = repr(error)
    finally:
        if installed:
            try:
                report['cleanup'] = rpc(cleanup, not completed)
                if completed:
                    healthy, = llm.generate([prompt], params, use_tqdm=False)
                    report['healthy_after'] = list(healthy.outputs[0].token_ids)
                    if report['healthy_after'] != report['healthy_before']:
                        raise RuntimeError('SAME_WORKER_GENERATION_NOT_RESTORED')
            except BaseException as error:
                report.update(status='REFUSED_OR_FAILED', cleanup_error=repr(error))
        report['probe_sha256'] = sha_file(__file__)
        save()
    print(json.dumps({key: report[key] for key in ('status', 'arm', 'error') if key in report}))
    return 0 if report['status'] == 'STANDARD_WORKER_RMS_RAW_CAPTURE_COMPLETE' else 2


if __name__ == '__main__':
    raise SystemExit(main())
