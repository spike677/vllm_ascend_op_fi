"""Root-only one B8 decode, zero F, complete projection census plus four APIs."""
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
    worker = worker.worker if isinstance(worker, WorkerWrapperBase) else worker
    if not isinstance(worker, WorkerBase):
        raise RuntimeError('ACTUAL_STANDARD_WORKER_REQUIRED')
    return worker


def rpc(llm, function, *args):
    rows = llm.collective_rpc(function, args=args, timeout=300)
    if len(rows) != 1:
        raise RuntimeError('ONE_REAL_TP1_WORKER_REQUIRED')
    return rows[0]


def setup(worker, prompts, quant_sha, trace_sha):
    from fault_injection import quant_container_native, metadata_arrays
    from demoncat.runtime.vllm_ascend.worker_extension import serving_shared_scale_group_inventory as serving
    from demoncat.runtime.vllm_ascend.worker_extension import quant_source_rows, quant_container_scopes
    from demoncat.runtime.vllm_ascend.worker_extension import shared_scale_group_inventory, quant_token_routes, quant_route_frames
    from demoncat.runtime.vllm_ascend.worker_extension import quant_row_provenance, quant_mla_metadata_rows, quant_moe_rows
    from demoncat.runtime.vllm_ascend.worker_extension import serving_shared_scale_inventory, shared_scale_inventory, quant_inventory
    from fault_injection import quant_group_native, quant_native_isolation
    from fault_injection.deployments import quant_group_inventory
    modules = (quant_container_native, metadata_arrays, serving, quant_source_rows, quant_container_scopes,
        shared_scale_group_inventory, quant_token_routes, quant_route_frames, quant_row_provenance,
        quant_mla_metadata_rows, quant_moe_rows, serving_shared_scale_inventory, shared_scale_inventory,
        quant_inventory, quant_group_native, quant_native_isolation, quant_group_inventory)
    loaded = []
    for module in modules:
        path = Path(inspect.getfile(module)).resolve()
        if not path.is_relative_to('/installed'):
            raise RuntimeError('FRESH_INSTALLED_DIAGNOSTIC_SOURCE_REQUIRED')
        loaded.append(dict(module=module.__name__, path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    worker = actual(worker)
    if hasattr(worker, '_demoncat_container_trace'):
        raise RuntimeError('EXCLUSIVE_CONTAINER_DIAGNOSTIC_REQUIRED')
    trace = quant_container_native.NativeContainers('/container-trace', trace_sha)
    worker._demoncat_container_trace = trace
    try:
        result = serving.install_shared_scale_group_inventory(worker, prompts=prompts, root='/quant',
            build_sha256=quant_sha, evidence_dir='/evidence/shared-scale', native_containers=trace)
        session = worker._demoncat_shared_scale_inventory
        if session.source_rows is None or session.source_rows.container_trace is None:
            raise RuntimeError('CURRENT_SOURCE_CONTAINER_SCOPES_UNAVAILABLE:' + str(session.source_row_refusal))
        return dict(inventory=result, loaded_sources=loaded, trace_initial=trace.query(),
                    zero_fault=True, integer_snapshot_helpers=False)
    except BaseException:
        # Keep an installed inventory reachable for the explicit cleanup RPC.
        raise


def advance(worker, request_ids=None, capture=False, collect=False):
    worker = actual(worker); session = worker._demoncat_shared_scale_inventory
    if collect:
        worker._demoncat_container_trace.stop()
        session.collect()
    elif capture:
        session.begin_decode_capture()
        worker._demoncat_container_trace.begin()
    else:
        session.expect(request_ids)
    return session.report()


def collect(worker):
    from fault_injection.quant_container_native import assess_observation
    worker = actual(worker); trace = worker._demoncat_container_trace
    snapshot = trace.snapshot()
    session = worker._demoncat_shared_scale_inventory
    snapshot['source_scopes'] = session.source_rows.container_trace.report()
    snapshot['assessment'] = assess_observation(snapshot)
    if snapshot['source_scopes']['errors'] or snapshot['source_scopes']['active']:
        snapshot['assessment']['status'] = 'OBSERVABILITY_REFUSED'
        snapshot['assessment']['reasons'].append('SOURCE_SCOPE_OBSERVER_NOT_COMPLETE')
    folder = Path('/evidence/container-trace'); folder.mkdir(exist_ok=True)
    raw = (json.dumps(snapshot, sort_keys=True, indent=2, allow_nan=False) + '\n').encode()
    path = folder / 'observation.json'
    with path.open('xb') as stream:
        stream.write(raw)
    return dict(path=str(path), bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest(),
                assessment=snapshot['assessment'], state=snapshot['state'], source_scopes=snapshot['source_scopes'])


def cleanup(worker):
    from demoncat.runtime.vllm_ascend.worker_extension.serving_shared_scale_inventory import remove_shared_scale_inventory
    worker = actual(worker); result = {}; errors = []
    if hasattr(worker, '_demoncat_shared_scale_inventory'):
        try:
            result['inventory'] = remove_shared_scale_inventory(worker)
        except BaseException as error:
            errors.append(repr(error))
    trace = getattr(worker, '_demoncat_container_trace', None)
    if trace is not None:
        try:
            if trace.started:
                trace.stop()
                snapshot = trace.snapshot()
                folder = Path('/evidence/container-trace'); folder.mkdir(exist_ok=True)
                raw = (json.dumps(snapshot, sort_keys=True, indent=2) + '\n').encode()
                trace.cleanup_attempt = getattr(trace, 'cleanup_attempt', 0) + 1
                path = folder / ('cleanup-observation-%03d.json' % trace.cleanup_attempt)
                with path.open('xb') as stream:
                    stream.write(raw)
                result['original'] = dict(path=str(path), sha256=hashlib.sha256(raw).hexdigest(), bytes=len(raw))
                result['released'] = trace.release()
            else:
                result['never_started'] = trace.query()
            del worker._demoncat_container_trace
        except BaseException as error:
            errors.append(repr(error)); result['retained_native'] = trace.query()
    result['errors'] = errors
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--quant-build-sha256', required=True); p.add_argument('--trace-build-sha256', required=True)
    a = p.parse_args()
    result = dict(status='STARTING', scope='ONE_ACTUAL_B8_FIRST_DECODE_FOUR_CONTAINER_DIAGNOSTIC',
        phases=[], native_faults=0, c1=False, c2=False, c3=False, request_rows_proven=False)
    def save():
        Path('/evidence/result.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    llm = None; attached = False
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
        result['healthy_before'] = [list(row.outputs[0].token_ids) for row in llm.generate(prompts, params, use_tqdm=False)]
        attached = True
        result['setup'] = rpc(llm, setup, tokens, a.quant_build_sha256, a.trace_build_sha256)
        result['identity'] = rpc(llm, 'demoncat_worker_identity'); save()
        if llm.llm_engine.has_unfinished_requests() or llm.reset_prefix_cache(reset_running_requests=False) is not True:
            raise RuntimeError('IDLE_FRESH_BATCH_REQUIRED')
        ids = llm.enqueue(prompts, params, use_tqdm=False)
        mapping = actual_driver_request_map(llm.llm_engine.output_processor)
        if len(ids) != 8 or set(ids) != {row['internal_request_id'] for row in mapping}:
            raise RuntimeError('ACTUAL_EIGHT_REQUEST_MAPPING_REQUIRED')
        row = dict(mapping=mapping, request_ids=ids, expect=rpc(llm, advance, ids)); result['phases'].append(row)
        partial = llm.llm_engine.step()
        if any(value.finished for value in partial) or not llm.llm_engine.has_unfinished_requests():
            raise RuntimeError('ACTUAL_PREFILL_THEN_ONE_DECODE_REQUIRED')
        row['capture'] = rpc(llm, advance, None, True); save()
        outputs = llm.wait_for_completion(use_tqdm=False)
        if len(outputs) != 8 or {value.request_id for value in outputs} != {value['external_request_id'] for value in mapping}:
            raise RuntimeError('ACTUAL_EIGHT_COMPLETED_REQUESTS_REQUIRED')
        row['output_tokens'] = [list(value.outputs[0].token_ids) for value in outputs]
        row['census'] = rpc(llm, advance, None, False, True)
        result['container_observation'] = rpc(llm, collect); save()
        result['cleanup'] = rpc(llm, cleanup); attached = bool(result['cleanup']['errors']); save()
        if attached:
            raise RuntimeError('OWNED_OBSERVER_CLEANUP_PENDING')
        result['healthy_after'] = [list(value.outputs[0].token_ids) for value in llm.generate(prompts, params, use_tqdm=False)]
        census = row['census']
        if (census['actual_generations'] != 1 or census['structural_candidates'] != 135
                or census['actual_projection_calls'] != 135 or census['actual_native_calls'] != 135
                or census['actual_public_calls'] != 270 or not census['row_operator_observation_complete']
                or census['actual_native_prepare_calls'] != 135 or census['actual_native_execute_calls'] != 135
                or not census['native_public_binding_complete'] or census['native_faults'] != 0
                or result['healthy_before'] != result['healthy_after']):
            raise RuntimeError('FULL_135_CENSUS_AND_SAME_WORKER_HEALTH_REQUIRED')
        result['status'] = ('ZERO_F_CONTAINER_ENTRY_OBSERVABILITY_EVIDENCE' if
            result['container_observation']['assessment']['status'] == 'FOUR_CONTAINER_OBSERVABILITY_COMPLETE'
            else 'ZERO_F_OBSERVABILITY_REFUSED_WITH_COMPLETE_CENSUS')
    except BaseException as error:
        result.update(status='FAILED_WITH_EVIDENCE', error=repr(error), traceback=traceback.format_exc())
        try:
            from fault_injection.quant_container_native import NativeContainers
            result['failure_noload_state'] = NativeContainers('/container-trace', a.trace_build_sha256).query()
        except BaseException as diagnostic_error:
            result['failure_noload_error'] = repr(diagnostic_error)
    finally:
        if attached and llm is not None:
            try:
                result['cleanup_retry'] = rpc(llm, cleanup)
            except BaseException as error:
                result['cleanup_error'] = repr(error)
        save()
    return 0 if result['status'] == 'ZERO_F_CONTAINER_ENTRY_OBSERVABILITY_EVIDENCE' else 2


if __name__ == '__main__':
    raise SystemExit(main())
