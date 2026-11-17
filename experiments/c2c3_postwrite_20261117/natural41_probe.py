"""Root-queued natural Fault41 B/F control; never starts an auxiliary Worker."""
import argparse
import hashlib
import inspect
import json
from pathlib import Path
import traceback


PROMPT = 'Explain in one sentence why the sky appears blue during the day.'
OWNER = '_demoncat_natural_postwrite41'
ROOT = Path(__file__).resolve().parents[2]


def rpc(llm, function, *args):
    rows = llm.collective_rpc(function, args=args, timeout=300)
    if len(rows) != 1:raise RuntimeError('TP1_REQUIRED')
    return rows[0]


def actual_worker(worker):
    from vllm.v1.worker.worker_base import WorkerWrapperBase
    return worker.worker if isinstance(worker, WorkerWrapperBase) else worker


def setup(worker, library, sha256):
    import torch
    import torch_npu
    from fault_injection.deployments.postwrite import NativePostwriteLibrary
    from demoncat.runtime.vllm_ascend.worker_extension.postwrite41 import Postwrite41WorkerProvider
    worker = actual_worker(worker)
    if hasattr(worker, OWNER):raise RuntimeError('POSTWRITE41_OWNER_EXISTS')
    provider = Postwrite41WorkerProvider(worker, NativePostwriteLibrary(library, sha256), tail_elements=2)
    provider.install()
    setattr(worker, OWNER, provider)
    return dict(provider=provider.report(), actual_device=torch.npu.get_device_name(0),
                torch=torch.__version__, torch_npu=torch_npu.__version__,
                adapter_path=inspect.getfile(Postwrite41WorkerProvider), library_sha256=sha256)


def expect(worker, phase, request_id, prompt, capture_path):
    provider = getattr(actual_worker(worker), OWNER)
    collector = None
    if capture_path is not None:
        from experiments.provider_adaptation_20261116.capture import capture
        collector = capture(Path(capture_path))
    provider.expect(phase, request_id, prompt, collector=collector)
    return dict(phase=phase, actual_request_id=request_id)


def finish_request(worker):
    return getattr(actual_worker(worker), OWNER).finish_request()


def finish(worker):
    worker = actual_worker(worker)
    provider = getattr(worker, OWNER)
    provider.close()
    result = provider.report()
    delattr(worker, OWNER)
    return result


def record_cleanup_failure(report, error):
    # Keep the primary failure and traceback intact, including close failures.
    report['cleanup_status'] = 'FAILED'
    report['cleanup_error'] = repr(error)
    if 'error' not in report:
        report.update(status='RECOVERY_FAILED', error=repr(error))


def output(row):
    return dict(request_id=row.request_id, tokens=list(row.outputs[0].token_ids), text=row.outputs[0].text)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='/model')
    parser.add_argument('--library', required=True)
    parser.add_argument('--sha256', required=True)
    parser.add_argument('--build-fingerprint', required=True)
    parser.add_argument('--out', type=Path, default=Path('/evidence'))
    parser.add_argument('--arm', choices=('baseline', 'fault'), required=True)
    parser.add_argument('--statistics', action='store_true')
    parser.add_argument('--memory', type=float, default=.35)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    report = dict(schema='demoncat.postwrite41.natural_probe.v1', status='STARTED',
        mechanism='41.tail_column', arm=args.arm, build_fingerprint=args.build_fingerprint,
        native_out_of_bounds_claim=False, scope='natural BF16 first pure decode in standard vLLM Worker',
        public_worker_qualified=False, all_eligible_qualified=False, every_match_qualified=False,
        phases=[])
    llm = None
    installed = False
    seen = []
    def save():
        (args.out / 'result.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    try:
        from vllm import LLM, SamplingParams
        from demoncat.runtime.vllm_ascend.worker_extension.serving_context import require_host_driven_core, read_enqueued_identity
        llm = LLM(model=args.model, dtype='bfloat16', tensor_parallel_size=1, enforce_eager=True,
            max_model_len=128, max_num_seqs=1, max_num_batched_tokens=128, num_gpu_blocks_override=64,
            enable_prefix_caching=False, enable_chunked_prefill=False, async_scheduling=False,
            gpu_memory_utilization=args.memory, seed=42,
            worker_extension_cls='demoncat.runtime.vllm_ascend.controller.DemoncatWorkerExtension', disable_log_stats=True)
        require_host_driven_core(llm)
        tokens = llm.get_tokenizer().encode(PROMPT, add_special_tokens=True)
        prompt = dict(prompt_token_ids=tokens)
        params = SamplingParams(temperature=0, max_tokens=2, ignore_eos=True)
        healthy, = llm.generate([prompt], params, use_tqdm=False)
        report['healthy_before'] = output(healthy)
        report['setup'] = rpc(llm, setup, args.library, args.sha256)
        installed = True
        for phase in ('observe', 'observe', 'lossless', args.arm, 'off'):
            if llm.llm_engine.has_unfinished_requests():raise RuntimeError('UNRELATED_PENDING_REQUEST')
            if llm.reset_prefix_cache(reset_running_requests=False) is not True:raise RuntimeError('RESET_NOT_CONFIRMED')
            ids = llm.enqueue([prompt], params, use_tqdm=False)
            mapping = read_enqueued_identity(llm, ids, seen_internal_request_ids=seen)
            seen += ids
            captured = args.statistics and len(report['phases']) == 3
            row = dict(phase=phase, mapping=mapping, captured=captured)
            report['phases'].append(row)
            row['expected'] = rpc(llm, expect, phase, ids[0], tokens,
                                   str(args.out / 'capture') if captured else None)
            save()
            response, = llm.wait_for_completion(use_tqdm=False)
            row['output'] = output(response)
            row['provider'] = rpc(llm, finish_request)
            if row['output']['request_id'] != mapping['external_request_id']:
                raise RuntimeError('DRIVER_REQUEST_MAPPING_CHANGED')
            save()
        controlled = report['phases'][3]['provider']
        if controlled['new_fault_hits'] != int(args.arm == 'fault'):
            raise RuntimeError('ACTUAL_C1_HIT_COUNT_MISMATCH')
        if controlled['exact_healthy_output'] != (args.arm == 'baseline'):
            raise RuntimeError('ACTUAL_ROOT_OUTPUT_EFFECT_NOT_OBSERVED')
        report['provider'] = rpc(llm, finish)
        installed = False
        healthy, = llm.generate([prompt], params, use_tqdm=False)
        report['healthy_after'] = output(healthy)
        if any(report['healthy_before'][key] != report['healthy_after'][key] for key in ('tokens', 'text')):
            raise RuntimeError('SAME_WORKER_POST_CLEAN_GENERATION_CHANGED')
        report.update(status='NATURAL_WORKER_C1_BOUNDED_RECOVERED', capture_scope='ACTUAL_FIRST_DECODE_MODEL_CALL',
                      same_worker_health=True, native_live_leases=report['provider']['native_live_leases'])
    except BaseException as error:
        report.update(status='FAILED_WITH_EVIDENCE', error=repr(error), traceback=traceback.format_exc())
        if llm is not None and seen:
            try:llm.llm_engine.abort_request(seen, internal=True)
            except BaseException as error:report['abort_error'] = repr(error)
    finally:
        if installed:
            try:report['cleanup'] = rpc(llm, finish)
            except BaseException as error:record_cleanup_failure(report, error)
        sources = ('fault_injection/deployments/postwrite.py', 'fault_injection/deployments/postwrite_boundaries.py',
                   'fault_injection/deployments/postwrite_kv.py',
                   'fault_injection/native_adapters/postwrite/postwrite.cpp',
                   'fault_injection/native_adapters/postwrite/postwrite.h',
                   'src/demoncat/runtime/vllm_ascend/worker_extension/postwrite41.py',
                   'src/demoncat/runtime/vllm_ascend/worker_extension/postwrite_collector.py',
                   'src/demoncat/runtime/vllm_ascend/worker_extension/paged33_audit.py',
                   'experiments/c2c3_postwrite_20261117/natural41_probe.py')
        report['source_sha256'] = {path: hashlib.sha256((ROOT / path).read_bytes()).hexdigest() for path in sources}
        save()
    return 0 if report['status'] == 'NATURAL_WORKER_C1_BOUNDED_RECOVERED' else 2


if __name__ == '__main__':raise SystemExit(main())
