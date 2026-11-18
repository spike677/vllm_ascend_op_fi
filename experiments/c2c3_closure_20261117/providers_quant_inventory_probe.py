"""Root-queued all-projection native census in two natural B8 first decodes."""
import argparse
import hashlib
import inspect
import json
from pathlib import Path
import traceback

PROMPTS = ['Explain why the sky looks blue.', 'Name the capital city of France.',
    'Complete this sentence: Water freezes at', 'Write a Python function to add two integers.',
    'What is the result of seven multiplied by nine?', 'Describe a forest in spring.',
    'Translate good morning into Spanish.', 'Give one reason to read books.']


def actual(worker):
    from vllm.v1.worker.worker_base import WorkerBase, WorkerWrapperBase
    if isinstance(worker, WorkerWrapperBase):
        worker = worker.worker
    if not isinstance(worker, WorkerBase):
        raise RuntimeError('ACTUAL_STANDARD_WORKER_REQUIRED')
    return worker


def setup(worker, prompts, build_sha, group_abi=False):
    from demoncat.runtime.vllm_ascend.worker_extension import serving_shared_scale_inventory, shared_scale_inventory, quant_row_provenance
    from fault_injection.deployments import dynamic_quant_inventory
    modules = (serving_shared_scale_inventory, shared_scale_inventory, dynamic_quant_inventory, quant_row_provenance)
    install = serving_shared_scale_inventory.install_shared_scale_inventory
    if group_abi:
        from demoncat.runtime.vllm_ascend.worker_extension import serving_shared_scale_group_inventory, shared_scale_group_inventory
        from fault_injection.deployments import quant_group_inventory
        from fault_injection import quant_group_native, quant_native_isolation
        from demoncat.runtime.vllm_ascend.worker_extension import quant_source_rows, quant_mla_metadata_rows
        modules += (serving_shared_scale_group_inventory, shared_scale_group_inventory, quant_group_inventory,
                    quant_group_native, quant_native_isolation, quant_source_rows, quant_mla_metadata_rows)
        install = serving_shared_scale_group_inventory.install_shared_scale_group_inventory
    sources = []
    for module in modules:
        path = Path(inspect.getfile(module)).resolve()
        if not path.is_relative_to('/installed'):
            raise RuntimeError('FRESH_INSTALLED_NATIVE_QUANT_OBSERVER_REQUIRED')
        sources.append(dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    worker = actual(worker)
    return dict(binding=install(worker, prompts=prompts,
        root='/quant', build_sha256=build_sha, evidence_dir='/evidence/shared-scale'), sources=sources,
        model_class=type(worker.model_runner.model).__module__ + '.' + type(worker.model_runner.model).__name__)


def advance(worker, request_ids=None, capture=False, collect=False):
    session = actual(worker)._demoncat_shared_scale_inventory
    if collect:
        if collect is not True or request_ids is not None or capture is not False:
            raise RuntimeError('EXPLICIT_INVENTORY_COLLECTION_REQUIRED')
        session.collect()
    elif capture:
        session.begin_decode_capture()
    else:
        session.expect(request_ids)
    return session.report()


def query(worker):
    session = actual(worker)._demoncat_shared_scale_inventory
    return session.report()


def remove(worker):
    from demoncat.runtime.vllm_ascend.worker_extension.serving_shared_scale_inventory import remove_shared_scale_inventory
    return remove_shared_scale_inventory(actual(worker))


def rpc(llm, method, *args):
    rows = llm.collective_rpc(method, args=args, timeout=300)
    if len(rows) != 1:
        raise RuntimeError('ONE_ACTUAL_WORKER_REQUIRED')
    return rows[0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--build-sha256', required=True)
    parser.add_argument('--group-abi', action='store_true')
    args = parser.parse_args()
    report = dict(status='STARTING', scope='TWO_ACTUAL_B8_FIRST_DECODES_ALL_NATIVE_V2_CALLS', phases=[],
        c1=False, c2=False, c3=False, native_qualified_targets=0, num_gpu_blocks_override=64, group_abi=args.group_abi)
    def save():
        Path('/evidence/result.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    llm, installed, seen = None, False, []
    try:
        from vllm import LLM, SamplingParams
        from demoncat.runtime.vllm_ascend.worker_extension.serving_context import require_host_driven_core, actual_driver_request_map
        llm = LLM(model='/model', dtype='bfloat16', tensor_parallel_size=1, enforce_eager=True,
            max_model_len=128, max_num_seqs=8, max_num_batched_tokens=256, num_gpu_blocks_override=64,
            enable_prefix_caching=False, enable_chunked_prefill=False, async_scheduling=False,
            gpu_memory_utilization=.85, seed=42, trust_remote_code=False, disable_log_stats=True,
            worker_extension_cls='demoncat.runtime.vllm_ascend.controller.DemoncatWorkerExtension')
        require_host_driven_core(llm)
        tokens = [llm.get_tokenizer().encode(prompt, add_special_tokens=True) for prompt in PROMPTS]
        prompts = [dict(prompt_token_ids=row) for row in tokens]
        params = SamplingParams(temperature=0, max_tokens=2, ignore_eos=True)
        report['healthy_before'] = [list(row.outputs[0].token_ids) for row in llm.generate(prompts, params, use_tqdm=False)]
        report['setup'] = rpc(llm, setup, tokens, args.build_sha256, args.group_abi)
        report['identity'] = rpc(llm, 'demoncat_worker_identity')
        installed = True
        for index in range(2):
            if llm.llm_engine.has_unfinished_requests() or llm.reset_prefix_cache(reset_running_requests=False) is not True:
                raise RuntimeError('IDLE_FRESH_BATCH_REQUIRED')
            ids = llm.enqueue(prompts, params, use_tqdm=False)
            mapping = actual_driver_request_map(llm.llm_engine.output_processor)
            if len(ids) != 8 or set(ids) != {row['internal_request_id'] for row in mapping} or any(key in seen for key in ids):
                raise RuntimeError('ACTUAL_ENQUEUED_REQUEST_MAPPING_CHANGED')
            seen += ids
            row = dict(index=index, mapping=mapping, enqueued=ids, advance=rpc(llm, advance, ids))
            report['phases'].append(row)
            save()
            partial = llm.llm_engine.step()
            if any(value.finished for value in partial) or not llm.llm_engine.has_unfinished_requests():
                raise RuntimeError('ACTUAL_COMPLETE_PREFILL_STEP_REQUIRED')
            row['prefill_step'] = dict(public_engine_step=True, any_finished=False)
            row['capture_advance'] = rpc(llm, advance, None, True)
            outputs = llm.wait_for_completion(use_tqdm=False)
            if len(outputs) != 8 or {value.request_id for value in outputs} != {value['external_request_id'] for value in mapping}:
                raise RuntimeError('ACTUAL_COMPLETED_BATCH_MAPPING_CHANGED')
            row['output_tokens'] = [list(value.outputs[0].token_ids) for value in outputs]
            rpc(llm, advance, None, False, True)
            row['inventory'] = rpc(llm, query)
            save()
        report['remove'] = rpc(llm, remove)
        installed = False
        report['healthy_after'] = [list(row.outputs[0].token_ids) for row in llm.generate(prompts, params, use_tqdm=False)]
        final = report['remove']
        if (not final['recovered'] or not final['complete_healthy_replays'] or not final['native_public_binding_complete']
                or not final['actual_native_calls'] or report['healthy_before'] != report['healthy_after']):
            raise RuntimeError('COMPLETE_NATIVE_PUBLIC_CENSUS_AND_HEALTH_NOT_CONFIRMED')
        report['status'] = 'ALL_NATIVE_V2_CALLS_OBSERVED_ROW_GATES_NOT_QUALIFIED'
    except BaseException as error:
        report.update(status='FAILED_WITH_EVIDENCE', error=repr(error), traceback=traceback.format_exc())
    finally:
        if installed and llm is not None:
            try:
                report['cleanup'] = rpc(llm, remove)
            except BaseException as error:
                report['cleanup_error'] = repr(error)
        save()
    return 0 if report['status'] == 'ALL_NATIVE_V2_CALLS_OBSERVED_ROW_GATES_NOT_QUALIFIED' else 2


if __name__ == '__main__':
    raise SystemExit(main())
