"""Root-queued blue-zone acceptance only, NOT main-project integration.

Six real B8 batches use the public Host/Runtime and normal vLLM scheduler.
This script installs no alternative runner and imports no runtime at module load.
"""
import argparse
import hashlib
import importlib
import inspect
import json
import os
from pathlib import Path
import time
import traceback


EVIDENCE = Path('/evidence')
SOURCE = Path('/source')
PHASES = ('normal', 'reference', 'lossless', 'calibration_off', 'fault', 'off')
PROMPTS = ('Explain why the sky looks blue.', 'Name the capital city of France.',
    'Complete this sentence: Water freezes at', 'Write a Python function to add two integers.',
    'What is the result of seven multiplied by nine?', 'Describe a forest in spring.',
    'Translate good morning into Spanish.', 'Give one reason to read books.')
PREFIX = 'demoncat.runtime.vllm_ascend.worker_extension.'
MODULES = tuple(PREFIX + name for name in ('serving_quant_projection', 'serving_quant', 'quant_row_provenance',
    'call_binding', 'batch_context', 'serving', 'serving_context', 'serving20_qmm', 'providers.quant',
    'providers._scope', 'host', 'host_receipt_ledger', 'runtime', 'config', 'identity', 'registry')) + (
    'fault_injection.quant_group_native', 'fault_injection.quant_native_isolation',
    'fault_injection.deployments.quant_group_c1', 'fault_injection.deployments.quant_group_inventory',
    'fault_injection.effect_summary', 'fault_injection.quantization')


def sha_file(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def model_identity(root):
    """Exact one-file ModelSlim output; no rename or guessed shard selection."""
    root = Path(root).resolve(strict=True)
    weight = 'quant_model_weight_w8a8_dynamic.safetensors'
    desc = 'quant_model_description_w8a8_dynamic.json'
    files = sorted(root.iterdir())
    if (len(files) > 64 or any(path.is_symlink() or not path.is_file() for path in files)
            or [path.name for path in files if path.suffix == '.safetensors'] != [weight]
            or any(path.name.endswith('.safetensors.index.json') for path in files)
            or not (root / desc).is_file() or not (root / 'config.json').is_file()):
        raise ValueError('EXACT_SINGLE_FILE_MODELSLIM_DYNAMIC_ARTIFACT_REQUIRED')
    config = json.loads((root / 'config.json').read_bytes())
    description = json.loads((root / desc).read_bytes())
    quant = config.get('quantization_config', {})
    if (config.get('model_type') not in ('qwen2', 'qwen3') or config.get('quantize') != 'w8a8_dynamic'
            or config.get('torch_dtype') != 'bfloat16'
            or quant.get('w_bit') != 8 or quant.get('a_bit') != 8 or quant.get('is_dynamic') is not True
            or description.get('model_quant_type') != 'W8A8_DYNAMIC'):
        raise ValueError('ACTUAL_ORDINARY_QWEN_DYNAMIC_W8A8_CONFIG_REQUIRED')
    records = {path.name: dict(bytes=path.stat().st_size, sha256=sha_file(path)) for path in files}
    manifest = dict(files=records, selected_safetensors=[weight], quant_description=desc,
        config=dict(model_type=config['model_type'], quantize=config['quantize'], quantization_config=quant,
                    torch_dtype=config.get('torch_dtype')),
        loader_selection="vllm.LLM(load_format='safetensors', trust_remote_code=False)", qualification_granted=False)
    return dict(manifest=manifest, sha256=digest(manifest))


def source_path(root, name):
    relative = Path(*name.split('.')).with_suffix('.py')
    return Path(root) / ('src' if name.startswith('demoncat.') else '') / relative


def installed_source_pins(source=SOURCE, installed=Path('/installed')):
    records = []
    for name in MODULES:
        target = Path(installed) / Path(*name.split('.')).with_suffix('.py')
        original = source_path(source, name)
        if (not target.is_file() or target.is_symlink() or not original.is_file()
                or sha_file(target) != sha_file(original)):
            raise ValueError('FRESH_INSTALLED_SOURCE_REQUIRED:' + name)
        records.append(dict(module=name, path=str(target), sha256=sha_file(target)))
    return records


def actual(worker):
    from vllm.v1.worker.worker_base import WorkerBase, WorkerWrapperBase
    if isinstance(worker, WorkerWrapperBase):
        worker = worker.worker
    if not isinstance(worker, WorkerBase):
        raise RuntimeError('ACTUAL_STANDARD_WORKER_REQUIRED')
    return worker


def setup(worker, expected_identity, build_sha256, pins):
    # Verify what this actual Worker loaded, not just files in a package folder.
    for pin in pins:
        module = importlib.import_module(pin['module'])
        path = Path(inspect.getfile(module)).resolve(strict=True)
        if str(path) != pin['path'] or sha_file(path) != pin['sha256']:
            raise RuntimeError('ACTUAL_LOADED_WORKER_SOURCE_CHANGED:' + pin['module'])
    from demoncat.runtime.vllm_ascend.worker_extension.serving_quant_projection import install_quant_projection_binding
    worker = actual(worker)
    identity = worker.demoncat_worker_identity()
    if identity.get('worker') != expected_identity or identity.get('quarantined') is not False:
        raise RuntimeError('PUBLIC_CORE_AND_ACTUAL_BINDING_WORKER_DIFFER')
    binding = install_quant_projection_binding(worker, root='/quant', build_sha256=build_sha256)
    return dict(binding=binding, worker=identity, loaded_sources=pins)


def remove(worker):
    from demoncat.runtime.vllm_ascend.worker_extension.serving_quant import remove_quant_binding
    return remove_quant_binding(actual(worker))


def rpc(llm, method, *args):
    values = llm.collective_rpc(method, timeout=120, args=args)
    if not isinstance(values, list) or len(values) != 1:
        raise RuntimeError('ONE_ACTUAL_TP1_WORKER_REQUIRED')
    return values[0]


def checked(value, status='ACTION_COMPLETED'):
    response = value.get('response') or {}
    if value.get('status') != status or response.get('recovery_required'):
        raise RuntimeError('PUBLIC_WORKER_CORE_' + str(value.get('status')) + ':' + str(response)[:1024])
    return response.get('provider_state') or {}


def write_original(path, value):
    raw = (json.dumps(value, indent=2, allow_nan=False) + '\n').encode()
    if len(raw) > 8 * 1024 * 1024:
        raise RuntimeError('BOUNDED_ACCEPTANCE_ORIGINAL_EXCEEDED')
    with Path(path).open('xb') as stream:
        stream.write(raw)
    return dict(path=str(path), bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())


def one_batch(llm, client, phase, prompts, params, *, seen, record, save, map_requests):
    if llm.llm_engine.has_unfinished_requests() or llm.reset_prefix_cache(reset_running_requests=False) is not True:
        raise RuntimeError('IDLE_FRESH_B8_REQUESTS_REQUIRED')
    ids = llm.enqueue(prompts, params, use_tqdm=False)
    record.update(phase=phase, mapping=None, request_ids=ids, completed=False)
    save()
    mapping = map_requests(llm.llm_engine.output_processor)
    record['mapping'] = mapping
    save()  # Even rejected mappings belong to this actual attempted enqueue.
    if (not isinstance(mapping, list) or len(mapping) != 8
            or any(not isinstance(row, dict) or any(not isinstance(row.get(key), str) or not row[key]
                for key in ('internal_request_id', 'external_request_id')) for row in mapping)
            or len({row['external_request_id'] for row in mapping}) != 8
            or len(ids) != 8 or len(set(ids)) != 8 or set(ids).intersection(seen)
            or set(ids) != {row['internal_request_id'] for row in mapping}):
        raise RuntimeError('ACTUAL_ENQUEUED_B8_MAPPING_CHANGED')
    seen.update(ids)
    state = checked(client.advance_batch(ids))
    if state['serving']['next_phase'] != phase:
        raise RuntimeError('PROVIDER_AND_ACTUAL_REQUEST_PHASE_DIFFER')
    partial = llm.llm_engine.step()  # Actual normal scheduler prefill.
    if any(value.finished for value in partial) or not llm.llm_engine.has_unfinished_requests():
        raise RuntimeError('ONE_COMPLETE_PREFILL_STEP_REQUIRED')
    checked(client.begin_decode_capture())
    outputs = llm.wait_for_completion(use_tqdm=False)  # Actual first pure decode.
    if (len(outputs) != 8 or len({value.request_id for value in outputs}) != 8
            or {value.request_id for value in outputs} != {row['external_request_id'] for row in mapping}):
        raise RuntimeError('ACTUAL_COMPLETED_B8_MAPPING_CHANGED')
    record['output_tokens'] = {value.request_id: list(value.outputs[0].token_ids) for value in outputs}
    checked(client.collect_replay())
    state = checked(client.query(), 'QUERY_OK')
    record.update(completed=True, serving=state['serving'], native=state['native'])
    return state


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('build-sha256', 'source-commit', 'source-manifest-sha256', 'model-sha256'):
        parser.add_argument('--' + name, required=True)
    args = parser.parse_args(argv)
    started = time.monotonic()
    work_deadline, final_deadline = started + 660, started + 840  # Queue hard limit900s includes bootstrap/cleanup.
    report = dict(status='STARTING', scope='BLUE_ZONE_ACCEPTANCE_NOT_MAIN_PROJECT_WIRING', phases=[],
        main_project_integration_verified=False, qualification_granted=False, c1=False, c2=False, c3=False,
        e5='NOT_ASSESSED_NO_STATISTICS_RUN', source_commit=args.source_commit, native_build_sha256=args.build_sha256,
        planned_calibration_and_fault_batches=6, planned_health_batches=2, cleanup=[])
    client = llm = None
    bound, phase, seen = False, 'SOURCE_AND_MODEL_IDENTITY', set()
    prompts = params = map_requests = None

    def save():
        report['elapsed_seconds'] = time.monotonic() - started
        for name, value in [('result.json', report)] + ([] if client is None else [('host-core-receipts.json', client.snapshot())]):
            temp = EVIDENCE / (name + '.tmp')
            temp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
            temp.replace(EVIDENCE / name)

    def retain_secondary(error):
        report.setdefault('secondary_errors', []).append(repr(error))

    def retained_owner():
        if client is None:
            return False
        try:
            return client.snapshot()['ownership_retained']
        except BaseException as error:
            retain_secondary(error)
            report['ownership_readback'] = 'UNKNOWN_NO_FORCED_CLEANUP'
            return None

    def require_time(deadline, reserve):
        if time.monotonic() + reserve >= deadline:
            raise RuntimeError('BOUNDED_ACCEPTANCE_TIME_RESERVE_REQUIRED')

    try:
        save()
        manifest = SOURCE / 'source_manifest.json'
        if sha_file(manifest) != args.source_manifest_sha256 or json.loads(manifest.read_bytes()).get('base_commit') != args.source_commit:
            raise RuntimeError('EXACT_FROZEN_SOURCE_REQUIRED')
        report['installed_sources'] = installed_source_pins()
        report['model_before'] = model_identity('/model')
        if report['model_before']['sha256'] != args.model_sha256:
            raise RuntimeError('ACTUAL_MODEL_BYTES_CHANGED_BEFORE_LLM')
        from vllm import LLM, SamplingParams
        from demoncat.runtime.vllm_ascend.worker_extension.host import HostAdapter, WorkerFaultClient
        from demoncat.runtime.vllm_ascend.worker_extension.config import FaultConfig
        from demoncat.runtime.vllm_ascend.worker_extension.serving_context import require_host_driven_core, actual_driver_request_map
        map_requests = actual_driver_request_map
        phase = 'LLM_CONSTRUCTION'
        llm = LLM(model='/model', load_format='safetensors', dtype='bfloat16', tensor_parallel_size=1,
            enforce_eager=True, max_model_len=128, max_num_seqs=8, max_num_batched_tokens=256,
            num_gpu_blocks_override=64, enable_prefix_caching=False, enable_chunked_prefill=False,
            async_scheduling=False, gpu_memory_utilization=.85, seed=42, trust_remote_code=False,
            disable_log_stats=True, worker_extension_cls='demoncat.runtime.vllm_ascend.controller.DemoncatWorkerExtension')
        require_host_driven_core(llm)
        tokens = [llm.get_tokenizer().encode(text, add_special_tokens=True) for text in PROMPTS]
        report['prompt_token_ids'] = tokens
        prompts = [dict(prompt_token_ids=value) for value in tokens]
        params = SamplingParams(temperature=0, max_tokens=2, ignore_eos=True)
        phase = 'HEALTHY_BEFORE'
        require_time(work_deadline, 180)
        report['healthy_before'] = [list(value.outputs[0].token_ids) for value in llm.generate(prompts, params, use_tqdm=False)]
        def prepare(identity, config):
            nonlocal bound
            value = rpc(llm, setup, identity, args.build_sha256, report['installed_sources'])
            bound = True
            report['deployment'] = value
            return {'prepare': {}, 'register': {}}
        client = WorkerFaultClient(HostAdapter(llm, prepare_binding=prepare, timeout=120), FaultConfig('quant', 'batch_shared_scale'))
        for action in ('prepare', 'register', 'check', 'enable'):
            checked(getattr(client, action)())
        for phase in PHASES:
            require_time(work_deadline, 180 if phase == 'fault' else 90)
            record = {}
            report['phases'].append(record)
            state = one_batch(llm, client, phase, prompts, params, seen=seen, record=record, save=save, map_requests=map_requests)
            reference = write_original(EVIDENCE / ('phase-' + phase + '.json'), record)
            record.clear()
            record.update(phase=phase, completed=True, original=reference,
                          next_phase=state['serving']['next_phase'])
            save()
        final = checked(client.finish())
        report['serving_final'] = final['serving']
        if (final['serving']['status'] != 'FUNCTIONAL_EFFECT_AND_OFF_OBSERVED'
                or not final['serving']['functional_effect_observed'] or not final['serving']['recovery_proven']
                or final['native_registered'] or final['leases'] or final['inflight']):
            raise RuntimeError('ACTUAL_DOUBLE_EFFECT_AND_SAME_WORKER_OFF_REQUIRED')
        checked(client.unregister())
        if client.detach()['status'] != 'DETACHED':
            raise RuntimeError('PUBLIC_OWNER_NOT_DETACHED')
        report['remove'] = rpc(llm, remove)
        bound = False
        phase = 'HEALTHY_AFTER'
        require_time(final_deadline, 60)
        report['healthy_after'] = [list(value.outputs[0].token_ids) for value in llm.generate(prompts, params, use_tqdm=False)]
        report['worker_after'] = rpc(llm, 'demoncat_worker_identity')
        report['model_after'] = model_identity('/model')
        if (report['healthy_after'] != report['healthy_before']
                or report['worker_after']['worker'] != report['deployment']['worker']['worker']
                or report['worker_after'].get('quarantined') is not False
                or report['model_after'] != report['model_before']):
            raise RuntimeError('SAME_WORKER_FINAL_HEALTH_OR_MODEL_IDENTITY_CHANGED')
        report.update(status='BLUE_ZONE_ORDINARY_QUANT_FUNCTION_AND_OFF_EVIDENCE_READY_FOR_REVIEW',
                      functional_effect_observed=True, same_worker_off_observed=True)
    except BaseException as error:
        report.update(status='FAILED_WITH_EVIDENCE', error=repr(error), traceback=traceback.format_exc(),
                      failure_phase=phase, failure_pid=os.getpid())
        if phase == 'LLM_CONSTRUCTION':
            try:
                from experiments.c2c3_closure_20261117.providers_quant_c1_probe import diagnose_initialization_failure
                diagnose_initialization_failure(report, args.build_sha256)
            except BaseException as secondary:
                retain_secondary(secondary)
    finally:
        if retained_owner() is True:
            try:
                if report.get('error'):
                    try:
                        report['failure_readback'] = write_original(EVIDENCE / 'failure-readback.json', client.query())
                    except BaseException as secondary:
                        retain_secondary(secondary)
                # Recover first. Never uninstall observers before the actual
                # OFF batch required by a consumed F, and never repeat F.
                state = checked(client.recover())
                report['cleanup'].append(dict(action='recover', state=state))
                if state['serving'].get('recovery_only') and not state['serving']['recovered']:
                    require_time(final_deadline, 60)
                    record = {}
                    report['cleanup_off'] = record
                    one_batch(llm, client, 'off', prompts, params, seen=seen, record=record, save=save, map_requests=map_requests)
                    checked(client.finish())
                checked(client.unregister())
                reply = client.detach()
                report['cleanup'].append(dict(action='detach', status=reply['status']))
                if reply['status'] != 'DETACHED':
                    raise RuntimeError('FAILED_SCOPE_OWNER_NOT_DETACHED')
            except BaseException as error:
                report['cleanup'].append(dict(status='OWNERSHIP_RETAINED_OR_UNVERIFIED', error=repr(error)))
        if bound and retained_owner() is False:
            try:
                report['remove_after_failure'] = rpc(llm, remove)
                bound = False
            except BaseException as error:
                retain_secondary(error)
        try:
            save()
        except BaseException as error:
            retain_secondary(error)
            print(json.dumps(dict(status='FINAL_EVIDENCE_WRITE_FAILED', primary=report.get('error'), secondary=repr(error))), flush=True)
    return 0 if report['status'] == 'BLUE_ZONE_ORDINARY_QUANT_FUNCTION_AND_OFF_EVIDENCE_READY_FOR_REVIEW' and not report.get('secondary_errors') else 2


if __name__ == '__main__':
    raise SystemExit(main())
