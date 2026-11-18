"""Root-only queued natural single-site Quant through public Worker Core V1."""
import argparse
import hashlib
import inspect
import json
import os
from pathlib import Path
import traceback


PROMPTS = ['Explain why the sky looks blue.', 'Name the capital city of France.',
    'Complete this sentence: Water freezes at', 'Write a Python function to add two integers.',
    'What is the result of seven multiplied by nine?', 'Describe a forest in spring.',
    'Translate good morning into Spanish.', 'Give one reason to read books.']
EVIDENCE = Path('/evidence')


def diagnose_initialization_failure(report, build_sha256):
    """Read the failing process's existing DSO; never construct a binding."""
    try:
        from fault_injection.quant_group_diagnostics import capture_loaded_quant_state, diagnostic_summary
        diagnostic = capture_loaded_quant_state('/quant', approved_build_sha256=build_sha256)
        path = EVIDENCE / 'quant-native-init-diagnostic.json'
        path.write_text(json.dumps(diagnostic, indent=2, allow_nan=False) + '\n')
        report['initialization_native_diagnostic'] = dict(path=str(path),
            sha256=hashlib.sha256(path.read_bytes()).hexdigest(), **diagnostic_summary(diagnostic))
    except BaseException as error:
        report['initialization_native_diagnostic'] = dict(status='DIAGNOSTIC_FAILED_PRIMARY_ERROR_PRESERVED',
            error=repr(error), traceback=traceback.format_exc())


def actual(worker):
    from vllm.v1.worker.worker_base import WorkerBase, WorkerWrapperBase
    if isinstance(worker, WorkerWrapperBase):
        worker = worker.worker
    if not isinstance(worker, WorkerBase):
        raise RuntimeError('ACTUAL_STANDARD_WORKER_REQUIRED')
    return worker


def setup(worker, tokens, manifest_sha256):
    from demoncat.runtime.vllm_ascend.worker_extension import (
        serving_quant_group_c1, shared_scale_group_inventory, serving_shared_scale_group_inventory,
        quant_row_provenance, quant_source_rows, quant_mla_metadata_rows, quant_moe_rows, quant_token_routes, quant_route_frames, serving_quant, host, config)
    from demoncat.runtime.vllm_ascend.worker_extension.providers import quant
    from fault_injection import quant_group_native, quant_native_isolation
    from fault_injection.deployments import quant_group_c1, quant_group_inventory
    modules = (serving_quant_group_c1, shared_scale_group_inventory, serving_shared_scale_group_inventory,
        quant_row_provenance, quant_source_rows, quant_mla_metadata_rows, quant_moe_rows, quant_token_routes, quant_route_frames, serving_quant, host, config, quant,
        quant_group_native, quant_native_isolation, quant_group_c1, quant_group_inventory)
    sources = []
    for module in modules:
        path = Path(inspect.getfile(module)).resolve()
        if not path.is_relative_to('/installed'):
            raise RuntimeError('FRESH_INSTALLED_QUANT_C1_AND_PUBLIC_CORE_REQUIRED')
        sources.append(dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    worker = actual(worker)
    return dict(binding=serving_quant_group_c1.install_quant_group_c1_binding(worker, prompts=tokens,
        root='/quant', build_sha256=manifest_sha256, evidence_dir='/evidence/quant-c1'), sources=sources,
        worker=worker.demoncat_worker_identity())


def remove(worker):
    from demoncat.runtime.vllm_ascend.worker_extension.serving_quant import remove_quant_binding
    return remove_quant_binding(actual(worker))


def rpc(llm, method, *args):
    values = llm.collective_rpc(method, timeout=300, args=args)
    if len(values) != 1:
        raise RuntimeError('ONE_ACTUAL_TP1_WORKER_REQUIRED')
    return values[0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--build-sha256', required=True)
    parser.add_argument('--group-abi', action='store_true', required=True)
    args = parser.parse_args()
    report = dict(status='STARTING', c1=False, c2=False, c3=False, e5='NOT_ASSESSED', phases=[],
        scope='ONE_AUTOMATICALLY_PROVEN_NATURAL_B8_SITE_ONCE_NOT_ALL_ELIGIBLE', num_gpu_blocks_override=64)
    client, llm = None, None
    bound = False
    stage = 'IMPORT_RUNTIME'
    def save():
        (EVIDENCE / 'result.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
        if client is not None:
            (EVIDENCE / 'host-core-receipts.json').write_text(json.dumps(client.snapshot(), indent=2, allow_nan=False) + '\n')
    def checked(value, status='ACTION_COMPLETED'):
        if value['status'] != status or value.get('response', {}).get('recovery_required'):
            raise RuntimeError('PUBLIC_WORKER_CORE_' + value['status'] + ':' + str(value.get('response'))[:2048])
        return value['response'].get('provider_state') or {}
    save()
    try:
        from vllm import LLM, SamplingParams
        from demoncat.runtime.vllm_ascend.worker_extension.config import FaultConfig
        from demoncat.runtime.vllm_ascend.worker_extension.host import HostAdapter, WorkerFaultClient
        from demoncat.runtime.vllm_ascend.worker_extension.serving_context import require_host_driven_core, actual_driver_request_map
        stage = 'LLM_CONSTRUCTION'
        llm = LLM(model='/model', dtype='bfloat16', tensor_parallel_size=1, enforce_eager=True,
            max_model_len=128, max_num_seqs=8, max_num_batched_tokens=256, num_gpu_blocks_override=64,
            enable_prefix_caching=False, enable_chunked_prefill=False, async_scheduling=False,
            gpu_memory_utilization=.85, seed=42, trust_remote_code=False, disable_log_stats=True,
            worker_extension_cls='demoncat.runtime.vllm_ascend.controller.DemoncatWorkerExtension')
        stage = 'HEALTHY_BEFORE'
        require_host_driven_core(llm)
        tokens = [llm.get_tokenizer().encode(prompt, add_special_tokens=True) for prompt in PROMPTS]
        prompts = [dict(prompt_token_ids=row) for row in tokens]
        params = SamplingParams(temperature=0, max_tokens=2, ignore_eos=True)
        report['healthy_before'] = [list(row.outputs[0].token_ids) for row in llm.generate(prompts, params, use_tqdm=False)]
        stage = 'PUBLIC_CORE_C1'
        def prepare_binding(identity, config):
            nonlocal bound
            report['deployment'] = rpc(llm, setup, tokens, args.build_sha256)
            bound = True
            if report['deployment']['worker']['worker'] != identity:
                raise RuntimeError('PUBLIC_CORE_AND_DEPLOYMENT_WORKER_DIFFER')
            return {'prepare': {}, 'register': {}}
        client = WorkerFaultClient(HostAdapter(llm, prepare_binding=prepare_binding, timeout=300),
                                   FaultConfig('quant', 'batch_shared_scale'))
        for action in ('prepare', 'register', 'check', 'execute'):
            checked(getattr(client, action)())
        seen = set()
        for phase in ('discovery_1', 'discovery_2', 'reference', 'lossless', 'qualification_off', 'fault', 'off'):
            if llm.llm_engine.has_unfinished_requests() or llm.reset_prefix_cache(reset_running_requests=False) is not True:
                raise RuntimeError('IDLE_FRESH_B8_REQUESTS_REQUIRED')
            ids = llm.enqueue(prompts, params, use_tqdm=False)
            mapping = actual_driver_request_map(llm.llm_engine.output_processor)
            if (len(ids) != 8 or len(set(ids)) != 8 or set(ids).intersection(seen)
                    or set(ids) != {row['internal_request_id'] for row in mapping}):
                raise RuntimeError('ACTUAL_ENQUEUED_B8_MAPPING_CHANGED')
            seen.update(ids)
            state = checked(client.advance_batch(ids))
            if state['serving']['next_phase'] != phase:
                raise RuntimeError('PROVIDER_PHASE_AND_ACTUAL_HOST_REQUEST_DIFFER')
            row = dict(phase=phase, mapping=mapping, enqueued=ids)
            report['phases'].append(row)
            save()
            partial = llm.llm_engine.step()
            if any(value.finished for value in partial) or not llm.llm_engine.has_unfinished_requests():
                raise RuntimeError('COMPLETE_ACTUAL_PREFILL_STEP_REQUIRED')
            checked(client.begin_decode_capture())
            outputs = llm.wait_for_completion(use_tqdm=False)
            if len(outputs) != 8 or {value.request_id for value in outputs} != {value['external_request_id'] for value in mapping}:
                raise RuntimeError('ACTUAL_COMPLETED_B8_MAPPING_CHANGED')
            row['output_tokens'] = [list(value.outputs[0].token_ids) for value in outputs]
            checked(client.collect_replay())
            state = checked(client.query(), 'QUERY_OK')
            serving = state['serving']
            row['receipt'] = {name: serving[name] for name in ('status', 'next_phase', 'native_faults', 'functional_effect_proven',
                'recovered', 'native_owner_released', 'native_qualified_targets')}
            save()
        final = checked(client.finish())
        report['serving_final'] = final['serving']
        if not report['serving_final']['c1'] or final['native_registered'] or final['leases'] or final['inflight']:
            raise RuntimeError('C1_SAME_WORKER_OFF_AND_ZERO_RESOURCES_NOT_CONFIRMED')
        checked(client.unregister())
        if client.detach()['status'] != 'DETACHED':
            raise RuntimeError('PUBLIC_CORE_OWNER_NOT_DETACHED')
        report['remove'] = rpc(llm, remove)
        bound = False
        report['healthy_after'] = [list(row.outputs[0].token_ids) for row in llm.generate(prompts, params, use_tqdm=False)]
        report['worker_after'] = rpc(llm, 'demoncat_worker_identity')
        if (report['healthy_before'] != report['healthy_after']
                or report['worker_after']['worker'] != report['deployment']['worker']['worker']
                or report['worker_after']['quarantined']):
            raise RuntimeError('SAME_WORKER_FINAL_HEALTH_NOT_CONFIRMED')
        report.update(status='SINGLE_SITE_C1_FUNCTION_AND_RECOVERY_READY_FOR_REVIEW', c1=True)
    except BaseException as error:
        report.update(status='FAILED_WITH_EVIDENCE', error=repr(error), traceback=traceback.format_exc(),
            failure_stage=stage, failure_process_id=os.getpid(),
            failure_context=dict(llm_constructed=llm is not None, client_constructed=client is not None, binding_installed=bound))
        if stage == 'LLM_CONSTRUCTION':
            save()  # Preserve the first exception before the optional read-only diagnostic.
            diagnose_initialization_failure(report, args.build_sha256)
    finally:
        if client is not None and client.snapshot()['ownership_retained']:
            report['cleanup'] = []
            for action in ('disable', 'unregister', 'detach'):
                try:
                    reply = getattr(client, action)()
                    report['cleanup'].append(dict(action=action, status=reply['status'], response=reply['response']))
                    if reply['status'] not in ('ACTION_COMPLETED', 'DETACHED'):
                        break  # Retain an uncertain owner; no forced late restore or process restart.
                except BaseException as error:
                    report['cleanup'].append(dict(action=action, error=repr(error)))
                    break
        if bound and client is not None and not client.snapshot()['ownership_retained']:
            try:
                report['remove_after_failure'] = rpc(llm, remove)
            except BaseException as error:
                report['remove_error'] = repr(error)
        save()
    return 0 if report['c1'] else 2


if __name__ == '__main__':
    raise SystemExit(main())
