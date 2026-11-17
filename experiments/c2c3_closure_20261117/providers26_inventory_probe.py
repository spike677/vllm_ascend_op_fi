"""Root-queued natural native writer census; no fault is armed or injected."""
import argparse
import hashlib
import inspect
import json
from pathlib import Path
import traceback

PROMPT = 'Explain in one sentence why the sky appears blue during the day.'
LIBRARIES = dict(trace='libfault26_readonly_trace_abi0.so', qmm='libfault26_qmm_output_underflow_abi0.so',
                 fia='libfault26_fia_writer_proxy_abi0.so')


def actual(worker):
    from vllm.v1.worker.worker_base import WorkerBase, WorkerWrapperBase
    if isinstance(worker, WorkerWrapperBase):
        worker = worker.worker
    if not isinstance(worker, WorkerBase):
        raise RuntimeError('ACTUAL_STANDARD_WORKER_REQUIRED')
    return worker


def setup(worker, tokens, mechanism, manifest_sha256):
    from demoncat.runtime.vllm_ascend.worker_extension import serving26_inventory
    from fault_injection.deployments import writer26_inventory
    sources = []
    for module in (serving26_inventory, writer26_inventory):
        path = Path(inspect.getfile(module)).resolve()
        if not path.is_relative_to(Path('/installed')):
            raise RuntimeError('FRESH_INSTALLED_WRITER_INVENTORY_REQUIRED')
        sources.append(dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    source = Path('/native/build.json')
    if hashlib.sha256(source.read_bytes()).hexdigest() != manifest_sha256:
        raise RuntimeError('REVIEWED_NATIVE_BUNDLE_MANIFEST_CHANGED')
    manifest = json.loads(source.read_text())
    if manifest.get('schema') != 'demoncat.writer26.deployment.v1' or manifest.get('actual_npu_calls') != 0:
        raise RuntimeError('FRESH_WRITER26_BUILD_REQUIRED')
    libraries = {kind: dict(path='/native/' + name, sha256=manifest['files'][name]) for kind, name in LIBRARIES.items()}
    for name, sha in manifest['files'].items():
        path = (source.parent / name).resolve(strict=True)
        if not path.is_relative_to(source.parent) or hashlib.sha256(path.read_bytes()).hexdigest() != sha:
            raise RuntimeError('ACTUAL_NATIVE_BUNDLE_FILE_CHANGED:' + name)
    worker = actual(worker)
    return dict(binding=serving26_inventory.install_writer26_inventory(worker, mechanism=mechanism,
        prompt_token_ids=tokens, evidence_dir='/evidence/writer-census', libraries=libraries), sources=sources,
        model_class=type(worker.model_runner.model).__module__ + '.' + type(worker.model_runner.model).__name__,
        reviewed_native_manifest_sha256=manifest_sha256)


def advance(worker, request_id):
    session = actual(worker)._demoncat_writer26_inventory
    session.expect(request_id)
    return session.report()


def query(worker):
    return actual(worker)._demoncat_writer26_inventory.report()


def remove(worker):
    from demoncat.runtime.vllm_ascend.worker_extension.serving26_inventory import remove_writer26_inventory
    return remove_writer26_inventory(actual(worker))


def rpc(llm, function, *args):
    rows = llm.collective_rpc(function, args=args, timeout=300)
    if len(rows) != 1:
        raise RuntimeError('ONE_ACTUAL_WORKER_REQUIRED')
    return rows[0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mechanism', choices=('26.residual_stomp', '26.fia_destination'), required=True)
    parser.add_argument('--manifest-sha256', required=True)
    parser.add_argument('--memory', type=float, default=.55)
    args = parser.parse_args()
    from vllm import LLM, SamplingParams
    from demoncat.runtime.vllm_ascend.worker_extension.serving_context import require_host_driven_core, read_enqueued_identity
    report = dict(status='STARTING', scope='TWO_COMPLETE_FRESH_PREFILLS_NOT_DECODE', mechanism=args.mechanism,
        native_qualified_targets=0, c1=False, c2=False, c3=False, phases=[], num_gpu_blocks_override=64)
    def save():
        Path('/evidence/result.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    llm, installed, seen = None, False, []
    try:
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
        report['healthy_before'] = list(before.outputs[0].token_ids)
        report['setup'] = rpc(llm, setup, tokens, args.mechanism, args.manifest_sha256)
        report['identity'] = rpc(llm, 'demoncat_worker_identity')
        report['actual_prompt_token_ids'] = tokens
        installed = True
        for index in range(2):
            if llm.llm_engine.has_unfinished_requests() or llm.reset_prefix_cache(reset_running_requests=False) is not True:
                raise RuntimeError('IDLE_FRESH_PREFILL_REQUIRED')
            ids = llm.enqueue([prompt], sampling, use_tqdm=False)
            mapping = read_enqueued_identity(llm, ids, seen_internal_request_ids=seen)
            seen += ids
            row = dict(index=index, mapping=mapping, advance=rpc(llm, advance, ids[0]))
            report['phases'].append(row)
            save()
            outputs = llm.wait_for_completion(use_tqdm=False)
            if len(outputs) != 1 or outputs[0].request_id != mapping['external_request_id']:
                raise RuntimeError('ACTUAL_REQUEST_MAPPING_CHANGED')
            row['inventory'] = rpc(llm, query)
            row['generated_tokens'] = list(outputs[0].outputs[0].token_ids)
            save()
        report['remove'] = rpc(llm, remove)
        installed = False
        after, = llm.generate([prompt], sampling, use_tqdm=False)
        report['healthy_after'] = list(after.outputs[0].token_ids)
        final = report['remove']['report']
        if (not final['healthy_repeated'] or final['native_injections'] != 0 or final['native_prepare_claims'] != 0
                or not all(final['actual_native_writers']) or report['healthy_before'] != report['healthy_after']):
            raise RuntimeError('READONLY_WHOLE_WORKER_INVENTORY_HEALTH_NOT_CONFIRMED')
        report['status'] = 'NATIVE_WRITER_INVENTORY_OBSERVED_NOT_QUALIFIED'
    except BaseException as error:
        report.update(status='FAILED_WITH_EVIDENCE', error=repr(error), traceback=traceback.format_exc())
    finally:
        if installed and llm is not None:
            try:
                report['cleanup'] = rpc(llm, remove)
            except BaseException as error:
                report['cleanup_error'] = repr(error)
        save()
    return 0 if report['status'] == 'NATIVE_WRITER_INVENTORY_OBSERVED_NOT_QUALIFIED' else 2


if __name__ == '__main__':
    raise SystemExit(main())
