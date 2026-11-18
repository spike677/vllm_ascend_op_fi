"""One same-process B8 decode; current typed WeightNz slots, no fault or row grant."""
import argparse
import hashlib
import inspect
import json
from pathlib import Path
import time
import traceback

if __package__:
    from . import probe_quant_containers as common
else:
    # The preparer mounts this exact source file read-only next to /probe.py.
    import probe_quant_containers as common


def immutable(folder, name, value):
    folder.mkdir(parents=True, exist_ok=True)
    raw = (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + '\n').encode()
    path = folder / name
    with path.open('xb') as stream:
        stream.write(raw)
    return dict(path=str(path), bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())


def support_source(expected):
    path = Path(inspect.getfile(common)).resolve(strict=True)
    if str(path) != '/probe_quant_containers.py' or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
        raise RuntimeError('EXACT_READONLY_PROBE_SUPPORT_REQUIRED')
    return dict(module=common.__name__, path=str(path), sha256=expected)


def bind_worker(worker):
    from fault_injection import quant_gmm2_native as native
    return native.startup_trace().bind_worker(common.actual(worker))


def setup(worker, prompts, quant_sha, trace_sha, support_sha):
    from fault_injection import quant_gmm2_native as native
    from demoncat.runtime.vllm_ascend.worker_extension import identity as identity_module
    worker = common.actual(worker)
    trace = native.startup_trace()  # A new Worker process cannot reconstruct this held object.
    trace.verify_worker(worker)
    extra = [support_source(support_sha)]
    for module in (native, identity_module):
        path = Path(inspect.getfile(module)).resolve(strict=True)
        if not path.is_relative_to('/installed'):
            raise RuntimeError('FRESH_INSTALLED_GMM2_BINDING_REQUIRED')
        extra.append(dict(module=module.__name__, path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    result = common.setup(worker, prompts, quant_sha, trace_sha, trace=trace)
    result['loaded_sources'].extend(extra)
    result.update(gmm2_initial=trace.query_gmm2(), pre_llm_startup=native.clone(trace.startup),
                  worker_binding=native.clone(trace.worker_identity), typed_formals=list(native.FORMALS))
    return result


def advance(worker, request_ids=None, capture=False, collect=False):
    worker = common.actual(worker)
    worker._demoncat_container_trace.verify_worker(worker)
    return common.advance(worker, request_ids, capture, collect)


def query(worker):
    worker = common.actual(worker); trace = worker._demoncat_container_trace
    trace.verify_worker(worker)
    return trace.query_all()


def collect(worker):
    from fault_injection.quant_gmm2_native import assess_observation
    worker = common.actual(worker); trace = worker._demoncat_container_trace
    trace.verify_worker(worker)
    snapshot = trace.snapshot()
    snapshot['source_scopes'] = worker._demoncat_shared_scale_inventory.source_rows.container_trace.report()
    snapshot['assessment'] = assess_observation(snapshot)
    if snapshot['source_scopes']['errors'] or snapshot['source_scopes']['active']:
        snapshot['assessment']['status'] = 'OBSERVABILITY_REFUSED'
        snapshot['assessment']['reasons'].append('SOURCE_SCOPE_OBSERVER_NOT_COMPLETE')
    saved = immutable(Path('/evidence/container-trace'), 'observation.json', snapshot)
    return dict(**saved, assessment=snapshot['assessment'], state=snapshot['state'],
                gmm2_state=snapshot['gmm2']['state'], source_scopes=snapshot['source_scopes'])


def cleanup(worker):
    from fault_injection import quant_gmm2_native as native
    from demoncat.runtime.vllm_ascend.worker_extension.serving_shared_scale_inventory import remove_shared_scale_inventory
    worker = common.actual(worker); result = {}; errors = []
    trace = getattr(worker, '_demoncat_container_trace', None)
    if trace is None:
        try:
            trace = native.startup_trace()
        except RuntimeError:
            if hasattr(worker, '_demoncat_shared_scale_inventory'):
                return dict(errors=['STARTUP_HANDLE_MISSING_WITH_OWNED_INVENTORY'], retained_inventory=True)
            return dict(errors=[], no_held_startup_handle=True)
    try:
        if trace.worker_identity is not None:
            trace.verify_worker(worker)
    except BaseException as error:
        trace.failure('cleanup_identity', error)
        return dict(errors=[repr(error)], retained_native=trace.query_all())
    if hasattr(worker, '_demoncat_shared_scale_inventory'):
        try:
            result['inventory'] = remove_shared_scale_inventory(worker)
        except BaseException as error:
            trace.failure('cleanup_inventory', error); errors.append(repr(error))
    if trace.started:
        try:
            trace.stop()
        except BaseException as error:
            trace.failure('cleanup_stop', error); errors.append(repr(error))
    # Save a fresh immutable attempt even when stop/removal failed. Never erase
    # the first failed request or consume model-owned prepared executors here.
    try:
        trace.cleanup_attempt = getattr(trace, 'cleanup_attempt', 0) + 1
        snapshot = trace.snapshot()
        result['original'] = immutable(Path('/evidence/container-trace'),
            'cleanup-observation-%03d.json' % trace.cleanup_attempt, snapshot)
    except BaseException as error:
        trace.failure('cleanup_snapshot', error); errors.append(repr(error))
    if not errors:
        try:
            result['released'] = trace.release() if trace.started else trace.query_all()
            native.discard_startup(trace)
            if getattr(worker, '_demoncat_container_trace', None) is trace:
                del worker._demoncat_container_trace
        except BaseException as error:
            trace.failure('cleanup_release', error); errors.append(repr(error))
    if errors:
        result['retained_native'] = trace.query_all()
    result.update(errors=errors, first_failure=trace.failures[0] if trace.failures else None,
                  failures=native.clone(trace.failures))
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('quant-build-sha256', 'trace-build-sha256', 'support-sha256'):
        p.add_argument('--' + name, required=True)
    a = p.parse_args()
    result = dict(status='STARTING', scope='ONE_SAME_PROCESS_B8_FIRST_DECODE_CURRENT_WEIGHTNZ_DIAGNOSTIC',
        phases=[], native_faults=0, c1=False, c2=False, c3=False, request_rows_proven=False,
        requested_builds=dict(quant_sha256=a.quant_build_sha256, trace_sha256=a.trace_build_sha256,
                              support_sha256=a.support_sha256))
    began = time.monotonic_ns()
    def save():
        result['elapsed_ns'] = time.monotonic_ns() - began
        Path('/evidence/result.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    llm = None; attached = False
    try:
        from fault_injection import quant_gmm2_native as native
        result['support_source'] = support_source(a.support_sha256)
        # No vLLM import or constructor until the exact original functions have
        # been resolved in this actual PID. No diagnostic Worker mixin exists.
        result['pre_llm_startup'] = native.prepare_startup('/container-trace', a.trace_build_sha256)
        result['startup_original'] = immutable(Path('/evidence/container-trace'),
            'startup-before-llm.json', result['pre_llm_startup'])
        save()
        from vllm import LLM, SamplingParams
        from demoncat.runtime.vllm_ascend.worker_extension.serving_context import require_host_driven_core, actual_driver_request_map
        llm = LLM(model='/model', dtype='bfloat16', tensor_parallel_size=1, enforce_eager=True,
            max_model_len=128, max_num_seqs=8, max_num_batched_tokens=256, num_gpu_blocks_override=64,
            enable_prefix_caching=False, enable_chunked_prefill=False, async_scheduling=False,
            gpu_memory_utilization=.85, seed=42, trust_remote_code=False, disable_log_stats=True,
            worker_extension_cls='demoncat.runtime.vllm_ascend.controller.DemoncatWorkerExtension')
        require_host_driven_core(llm)
        result['worker_before_requests'] = common.rpc(llm, bind_worker)
        tokens = [llm.get_tokenizer().encode(prompt, add_special_tokens=True) for prompt in common.PROMPTS]
        prompts = [dict(prompt_token_ids=row) for row in tokens]
        params = SamplingParams(temperature=0, max_tokens=2, ignore_eos=True)
        result['healthy_before'] = [list(row.outputs[0].token_ids) for row in llm.generate(prompts, params, use_tqdm=False)]
        attached = True
        result['setup'] = common.rpc(llm, setup, tokens, a.quant_build_sha256, a.trace_build_sha256, a.support_sha256)
        result['identity'] = common.rpc(llm, 'demoncat_worker_identity'); save()
        if llm.llm_engine.has_unfinished_requests() or llm.reset_prefix_cache(reset_running_requests=False) is not True:
            raise RuntimeError('IDLE_FRESH_BATCH_REQUIRED')
        ids = llm.enqueue(prompts, params, use_tqdm=False)
        mapping = actual_driver_request_map(llm.llm_engine.output_processor)
        if len(ids) != 8 or set(ids) != {row['internal_request_id'] for row in mapping}:
            raise RuntimeError('ACTUAL_EIGHT_REQUEST_MAPPING_REQUIRED')
        row = dict(mapping=mapping, request_ids=ids, expect=common.rpc(llm, advance, ids)); result['phases'].append(row)
        partial = llm.llm_engine.step()
        if any(value.finished for value in partial) or not llm.llm_engine.has_unfinished_requests():
            raise RuntimeError('ACTUAL_PREFILL_THEN_ONE_DECODE_REQUIRED')
        row['capture'] = common.rpc(llm, advance, None, True); save()
        outputs = llm.wait_for_completion(use_tqdm=False)
        if len(outputs) != 8 or {value.request_id for value in outputs} != {value['external_request_id'] for value in mapping}:
            raise RuntimeError('ACTUAL_EIGHT_COMPLETED_REQUESTS_REQUIRED')
        row['output_tokens'] = [list(value.outputs[0].token_ids) for value in outputs]
        row['census'] = common.rpc(llm, advance, None, False, True)
        result['gmm2_observation'] = common.rpc(llm, collect); save()
        result['cleanup'] = common.rpc(llm, cleanup); attached = bool(result['cleanup']['errors']); save()
        if attached:
            raise RuntimeError('OWNED_OBSERVER_CLEANUP_PENDING')
        result['healthy_after'] = [list(value.outputs[0].token_ids) for value in llm.generate(prompts, params, use_tqdm=False)]
        census = row['census']
        if (census['actual_generations'] != 1 or census['structural_candidates'] != 135
                or census['actual_projection_calls'] != 135 or census['actual_native_calls'] != 135
                or census['actual_public_calls'] != 270 or not census['row_operator_observation_complete']
                or census['actual_native_prepare_calls'] != 135 or census['actual_native_execute_calls'] != 135
                or not census['native_public_binding_complete'] or census['native_faults'] != 0
                or census['request_row_proven_calls'] != 10
                or result['healthy_before'] != result['healthy_after']):
            raise RuntimeError('FULL_135_CENSUS_UNCHANGED_TEN_ROW_PROOFS_AND_SAME_WORKER_HEALTH_REQUIRED')
        result['status'] = ('ZERO_F_CURRENT_WEIGHTNZ_ENTRY_EVIDENCE' if
            result['gmm2_observation']['assessment']['status'] == 'CURRENT_WEIGHTNZ_SLOTS_AND_EXECUTION_OBSERVED'
            else 'ZERO_F_OBSERVABILITY_REFUSED_WITH_COMPLETE_CENSUS')
    except BaseException as error:
        result.update(status='FAILED_WITH_EVIDENCE', error=repr(error), traceback=traceback.format_exc())
        try:
            from fault_injection import quant_gmm2_native as native
            trace = native.startup_trace(); trace.failure('probe', error)
            result['failure_noload_state'] = trace.query_all()
            if not attached:
                result['unused_startup_original'] = immutable(Path('/evidence/container-trace'),
                    'unused-startup-failure.json', trace.snapshot())
                native.discard_startup(trace)
        except BaseException as diagnostic_error:
            result['failure_noload_error'] = repr(diagnostic_error)
    finally:
        if attached and llm is not None:
            try:
                result['cleanup_retry'] = common.rpc(llm, cleanup)
            except BaseException as error:
                result['cleanup_error'] = repr(error)
        save()
    return 0 if result['status'] == 'ZERO_F_CURRENT_WEIGHTNZ_ENTRY_EVIDENCE' else 2


if __name__ == '__main__':
    raise SystemExit(main())
