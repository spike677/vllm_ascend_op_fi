"""All-layer stock Qwen3 SDPA integration; original calls and returns are preserved."""
import argparse
from dataclasses import asdict, replace
import functools
import importlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import re
import sys
import threading
import time
import traceback
import uuid

from model_helpers import (append, digest, inputs_unchanged, require, sha, statistics_config,
                           statistics_receipt, tensor_snapshot, token_sha, verify_fingerprint, write, physical_device_binding)
from runtime_binding import verify_runtime, verify_imports
import native_evidence
import resource_meter
from model_native_stream import NativeLogCursor


def emit(event, **fields):
    row = {'event': event, 'pid': os.getpid(), 'thread_id': threading.get_native_id(),
           'epoch_ns': time.time_ns(), 'monotonic_ns': time.monotonic_ns(), **fields}
    print(json.dumps(row, ensure_ascii=False, allow_nan=False), flush=True)
    return row


class ModelObserver:
    """Module hooks establish an unconditional ledger, independently of policy matches."""
    def __init__(self, torch, torch_npu, case, output, model, execution_config):
        self.torch, self.npu, self.case, self.output = torch, torch_npu, case, output
        self.phase, self.active_module = None, None
        self.modules, self.calls, self.handles = [], [], []
        self.adapters, self.sessions, self.off_sessions, self.native_calls = {}, {}, {}, {}
        self.original_calls = 0
        self.functional = self.original = self.wrapper = None
        self.launch_reader = NativeLogCursor(output.parent / (case['case_id'] + '.log'))
        from native_execution_strategy.__main__ import registry_adapter
        from native_execution_strategy.policies import Policy
        from native_execution_strategy.session import StrategySession
        from model_context import bind_sdpa
        self.bind_sdpa = bind_sdpa
        package = Path('/audit/runtime/native_execution_strategy')
        for index, layer in enumerate(model.model.layers):
            resolved = registry_adapter(execution_config, package / 'data/catalog.json', package, False)
            self.adapters[index] = resolved.adapter
            self.sessions[index] = StrategySession(resolved.adapter, Policy(execution_config.policy))
            self.off_sessions[index] = StrategySession(resolved.adapter, Policy(replace(execution_config.policy, enabled=False)))
            layer_name = f'model.layers.{index}.self_attn'
            self.modules.append({'layer_index': index, 'layer_name': layer_name, 'class': type(layer.self_attn).__name__})
            self.handles.append(layer.self_attn.register_forward_pre_hook(self._enter(index, layer_name), with_kwargs=True))
            self.handles.append(layer.self_attn.register_forward_hook(self._leave(index, layer_name), with_kwargs=True, always_call=True))
        require(len(self.modules) == 28, 'Unexpected real attention module inventory')
        write(output / 'module_inventory.json', {'schema': 'nes.model_modules.v1', 'layers': self.modules,
             'config_sha256': case['model_config_sha256'], 'resolved_adapter': resolved.as_dict()})

    def _enter(self, index, name):
        def observed(module, args, kwargs):
            require(self.phase is not None and self.active_module is None, 'Nested or unlabelled real attention')
            hidden = kwargs.get('hidden_states', args[0] if args else None)
            require(hidden is not None, 'Missing actual model hidden states')
            event = {'event': 'module_enter', 'step': self.phase['step'], 'layer_index': index, 'layer_name': name,
                'module_call_id': f"step{self.phase['step']}.layer{index}.visit{self.phase['module_visits'].get(index, 0)}",
                'hidden_shape': list(hidden.shape), 'request_ids': self.phase['request_ids']}
            append(self.output / 'module_events.jsonl', event)
            self.phase['module_visits'][index] = self.phase['module_visits'].get(index, 0) + 1
            self.active_module = event
        return observed

    def _leave(self, index, name):
        def observed(module, args, kwargs, returned):
            require(self.active_module is not None and self.active_module['layer_index'] == index, 'Module hook identity mismatch')
            first = returned[0] if isinstance(returned, (tuple, list)) else returned
            append(self.output / 'module_events.jsonl', {**self.active_module, 'event': 'module_exit',
                'output_shape': list(first.shape) if first is not None else None})
            self.active_module = None
        return observed

    def install(self, functional):
        self.functional, self.original = functional, functional.scaled_dot_product_attention
        @functools.wraps(self.original)
        def observed(*args, **kwargs):
            return self.observe(args, kwargs)
        self.wrapper = observed
        functional.scaled_dot_product_attention = observed

    def restore(self):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
        if self.functional is not None:
            require(self.functional.scaled_dot_product_attention is self.wrapper, 'External SDPA wrapper change')
            self.functional.scaled_dot_product_attention = self.original
        self.launch_reader.close()

    def begin(self, phase, rows, mapping_file):
        require(self.active_module is None, 'Unfinished attention module')
        self.phase = {**phase, 'request_ids': [row['request_id'] for row in rows], 'rows': rows,
            'request_mapping_file': mapping_file.relative_to(self.output).as_posix(),
            'request_mapping_sha256': sha(mapping_file), 'module_visits': {}, 'calls': []}

    def end(self):
        require(self.active_module is None and len(self.phase['calls']) == 28
                and self.phase['module_visits'] == {index: 1 for index in range(28)}, 'Incomplete all-layer phase')
        require([self.calls[index]['layer_index'] for index in self.phase['calls']] == list(range(28)), 'Actual layer ordering changed')
        return [self.calls[index]['call_id'] for index in self.phase['calls']]

    def observe(self, args, kwargs):
        from model_context import probe_template, actual_native_call, prefill_context
        p, module = self.phase, self.active_module
        require(p is not None and module is not None, 'SDPA outside observed model attention: coverage unknown')
        values = self.bind_sdpa(args, kwargs)
        index, layer = len(self.calls), module['layer_index']
        token = f"model_s{p['step']}_l{layer}_{uuid.uuid4().hex}"
        folder = self.output / 'native_calls' / f'call_{index:04d}'
        folder.mkdir(parents=True, exist_ok=False)
        profile = folder / 'native_profile'
        call = {'case_id': self.case['case_id'], 'call_id': token,
            'api': 'torch.nn.functional.scaled_dot_product_attention', 'pid': os.getpid(),
            'thread_id': threading.get_native_id(), 'phase': 'operator_execution', 'step': p['step'],
            'mode': p['mode'], 'batch_size': self.case['batch_size'], 'scope_token': None,
            'layer_index': layer, 'model_stage': p['stage'], 'run_index': p['run_index']}
        observed = {'schema': 'nes.model_sdpa_call.v1', 'call_id': token, 'call_index': index,
            'step': p['step'], 'run_index': p['run_index'], 'mode': p['mode'], 'stage': p['stage'],
            'layer_index': layer, 'layer_name': module['layer_name'], 'module_call_id': module['module_call_id'],
            'request_mapping_file': p['request_mapping_file'], 'request_mapping_sha256': p['request_mapping_sha256'],
            'request_ids': p['request_ids'], 'status': 'STARTING', 'eligibility': 'UNKNOWN'}
        metadata = {name: {'shape': list(values[name].shape), 'dtype': str(values[name].dtype)} for name in ('query', 'key', 'value')}
        metadata.update({key: values[key] for key in ('dropout_p', 'is_causal', 'scale', 'enable_gqa')})
        mask = values['attn_mask']
        metadata['attn_mask'] = None if mask is None else {'shape': list(mask.shape), 'dtype': str(mask.dtype)}
        observed['sdpa_metadata'] = metadata
        before = {name: tensor_snapshot(self.torch, self.npu, values[name]) for name in ('query', 'key', 'value')}
        if mask is not None:
            before['attn_mask'] = tensor_snapshot(self.torch, self.npu, mask)
        observed['inputs_before'] = before
        write(folder / 'call.json', observed)
        captured = {}
        def invoke_original():
            plan = os.environ.get('NES_FA_PLAN_JSON')
            if plan is not None:
                captured['plan'] = json.loads(plan)
                call['scope_token'] = captured['plan']['scope']['token']
                captured['host_path'] = Path(os.environ['NES_FA_LOG_PATH'])
            self.npu.npu.synchronize()
            entered = emit('native_api_enter', **call)
            call.update(enter_epoch_ns=entered['epoch_ns'], enter_monotonic_ns=entered['monotonic_ns'])
            self.original_calls += 1
            returned = self.original(*args, **kwargs)
            self.npu.npu.synchronize()
            exited = emit('native_api_exit', status='returned', **call)
            call.update(status='returned', exit_epoch_ns=exited['epoch_ns'], exit_monotonic_ns=exited['monotonic_ns'])
            captured['value'] = returned
            return returned
        def observer(value, plan, evidence_dir):
            require(value is captured['value'] and plan == captured['plan'], 'Original return or native scope replaced')
            record = native_evidence.verified_record(self.output.parent / (self.case['case_id'] + '.log'), call,
                profile, captured['host_path'], Path(self.case['approved_pod_schema_file']), self.case['approved_pod_schema_sha256'],
                launch_reader=self.launch_reader)
            actual, provenance = actual_native_call(record, p['rows'], self.case['physical_device'])
            captured.update(record=record, actual_call=actual, provenance=provenance)
            write(profile / 'verified_record.json', record)
            return record
        try:
            adapter = self.adapters[layer]
            if p['stage'] == 'decode':
                require(all(not os.environ.get(key) for key in ('NES_FA_PLAN_JSON', 'NES_FA_LOG_PATH')), 'Unsupported decode inherited injection scope')
                require(values['query'].shape[2] == 1 and values['key'].shape[2] == 2049, 'Unexpected actual decode geometry')
                value = invoke_original()
                observed.update(eligibility='UNSUPPORTED_SHAPE_OBSERVED', unsupported_reason='query_length1/key_length2049 outside certified2048 scope',
                                session=None, expected_injection=False)
            elif p['mode'] == 'calibration':
                template = probe_template(adapter, self.case['batch_size'])
                with adapter.native_scope(None, token, folder, context=template) as plan:
                    value = invoke_original()
                record = observer(value, plan, folder)
                actual_context = adapter.inspect(captured['actual_call'])
                require(prefill_context(adapter, values, p['rows'], captured['actual_call']).as_dict() == actual_context.as_dict(),
                        'Calibration Python SDPA differs from actual native descriptor/source contract')
                observation = adapter.parse_native_evidence(actual_context, record)
                self.native_calls[layer] = captured['actual_call']
                observed.update(eligibility='ACTUAL_NATIVE_VALIDATED', native_descriptor_provenance=captured['provenance'],
                    calibration_template_is_intent_only=True, session=None,
                    actual_native_context=actual_context.as_dict(), actual_strategy=observation.strategy.as_dict())
            else:
                require(layer in self.native_calls, 'No real native calibration for this exact layer')
                context = prefill_context(adapter, values, p['rows'], self.native_calls[layer])
                session = self.off_sessions[layer] if p['mode'] == 'off' else self.sessions[layer]
                session_mode = 'native' if p['mode'] == 'native' else 'controlled'
                # One identical Session callsite for native, both faults and off.
                result = session.execute(context, invoke_original, observer, token, folder, mode=session_mode)
                require(result.observation_error is None and result.observation is not None,
                        'Native observation missing or invalid: ' + str(result.observation_error))
                value = result.value
                require(adapter.inspect(captured['actual_call']).as_dict() == context.as_dict(), 'Actual native descriptors differ from bound SDPA mapping')
                observed.update(eligibility='ACTUAL_NATIVE_VALIDATED', native_descriptor_provenance=captured['provenance'],
                    actual_native_context=context.as_dict(), actual_strategy=result.observation.strategy.as_dict(),
                    session={'assessment': asdict(result.assessment), 'decision': None if result.decision is None else result.decision.as_dict(),
                        'original_return_identity_preserved': value is captured['value'], 'observation_error': result.observation_error})
            require(value is captured['value'] and list(value.shape) == list(values['query'].shape), 'Original SDPA result identity/shape changed')
            after = {name: tensor_snapshot(self.torch, self.npu, values[name]) for name in before}
            observed.update(inputs_after=after, allowed_input_format_normalizations=inputs_unchanged(before, after),
                output=tensor_snapshot(self.torch, self.npu, value), native_call=call, original_api_calls=1,
                original_return_identity_preserved=True, status='CAPTURE_COMPLETED')
            if 'record' in captured:
                observed.update(native_record_file=(profile / 'verified_record.json').relative_to(self.output).as_posix(),
                                native_record_sha256=sha(profile / 'verified_record.json'))
            write(folder / 'call.json', observed)
            row = {'file': (folder / 'call.json').relative_to(self.output).as_posix(), 'sha256': sha(folder / 'call.json'),
                   'call_id': token, 'step': p['step'], 'layer_index': layer, 'stage': p['stage']}
            self.calls.append(row)
            p['calls'].append(index)
            return value
        except BaseException:
            observed.update(status='CAPTURE_FAILED', traceback=traceback.format_exc(), native_call=call)
            write(folder / 'call.json', observed)
            raise


def request_rows(requests, stage):
    return [{'request_id': row['request_id'], 'batch_slot': slot,
        'prompt_token_ids_sha256': row['prompt_token_ids_sha256'], 'prompt_length': 2048,
        'query_length': 2048 if stage == 'prefill' else 1, 'kv_length': 2048 if stage == 'prefill' else 2049,
        'query_positions': list(range(2048)) if stage == 'prefill' else [2048]}
        for slot, row in enumerate(requests)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case-file', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    case = json.loads(args.case_file.read_bytes())
    require(case['schema'] == 'nes.model_coverage.case.v1' and case['batch_size'] in (1, 8), 'Wrong model case')
    require(case['mode_sequence'] in (['calibration'], ['calibration', 'native', 'controlled', 'controlled', 'off']), 'Unexpected registered mode plan')
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    result = {'schema': 'nes.model_coverage.capture.v1', 'status': 'STARTING', 'case_id': case['case_id'],
        'case_sha256': sha(args.case_file), 'runtime_manifest_sha256': case['runtime_manifest_sha256'],
        'checkpoint_manifest_sha256': case['model_manifest_sha256'], 'tokenizer_manifest_sha256': case['tokenizer_manifest_sha256'],
        'source_fingerprints_sha256': case['source_fingerprints_sha256'], 'mode_sequence': case['mode_sequence'], 'calls': [],
        'scope': 'Actual stock HF complete model prefill and one teacher-forced decode; no attention substitution.',
        'model_coverage_accepted': False, 'original_api_calls': 0}
    observer, debugger, active = None, None, False
    phases = []
    write(output / 'result.json', result)
    try:
        runtime = verify_runtime(case)
        require(os.environ.get('ACLNN_CACHE_LIMIT') == '0' and not os.environ.get('LD_PRELOAD'), 'Cache/preload scope invalid')
        from vendor_exec import validate_vendor
        vendor = validate_vendor(case)
        receipt_path = output.with_name(output.name + '.vendor_launch.json')
        require(os.environ.get('NES_VENDOR_LAUNCH_RECEIPT') == str(receipt_path)
                and sha(receipt_path) == os.environ.get('NES_VENDOR_LAUNCH_RECEIPT_SHA256'), 'Fresh process vendor launch missing')
        launch = json.loads(receipt_path.read_bytes())
        require(launch['probe_sha256'] == sha(__file__) and launch['case_sha256'] == result['case_sha256'], 'Vendor launch belongs to other source/case')
        require(all(launch['child_environment'].get(key) == os.environ.get(key) for key in
                    ('ASCEND_CUSTOM_OPP_PATH', 'LD_LIBRARY_PATH', 'LD_PRELOAD', 'ACLNN_CACHE_LIMIT', 'PYTHONPATH')),
                'Actual loader environment differs from fresh-process receipt')
        result['loader'] = vendor
        for key in ('model_manifest', 'tokenizer_manifest', 'source_fingerprints'):
            require(sha(case[key + '_file']) == case[key + '_sha256'], 'Frozen model source manifest changed: ' + key)
        verify_fingerprint(case['model_manifest_file'], 'nes.model_file_fingerprint.v1')
        verify_fingerprint(case['tokenizer_manifest_file'], 'nes.tokenizer_file_fingerprint.v1')
        require(sha(case['input_manifest_file']) == case['input_manifest_sha256'], 'Request inputs changed')
        inputs = json.loads(Path(case['input_manifest_file']).read_bytes())
        by_id = {row['request_id']: row for row in inputs['requests']}
        requests = [by_id[key] for key in case['request_ids']]
        require(len(requests) == case['batch_size'] and len(set(case['request_ids'])) == len(requests), 'Actual request mapping invalid')
        for row in requests:
            require(len(row['prompt_token_ids']) == 2048 and token_sha(row['prompt_token_ids']) == row['prompt_token_ids_sha256'], 'Request token digest mismatch')
        from native_execution_strategy.config import from_dict
        config = from_dict(case['execution_config'])
        require(config.adapter.version == 'validated' and config.selection_target == 'joint', 'Registered model strategy scope differs')
        from collector_stage import select_collector, verify_loaded
        collector = select_collector(case)
        import torch
        import torch_npu
        from torch.nn import functional
        from torch.nn.attention import sdpa_kernel, SDPBackend
        from transformers import AutoModelForCausalLM
        sources = json.loads(Path(case['source_fingerprints_file']).read_bytes())
        result['versions'] = {name: importlib.metadata.version(name) for name in ('torch', 'torch-npu', 'transformers')}
        require(result['versions'] == sources['versions'], 'Installed package versions changed')
        result['actual_source_modules'] = {}
        for name, row in sources['modules'].items():
            actual = Path(importlib.import_module(name).__file__).resolve()
            require(str(actual) == row['path'] and sha(actual) == row['sha256'], 'Installed model source differs: ' + name)
            result['actual_source_modules'][name] = {'path': str(actual), 'sha256': sha(actual)}
        torch.manual_seed(42)
        torch_npu.npu.set_device(0)
        result['physical_device_binding'] = physical_device_binding(torch_npu, output.parent)
        require(result['physical_device_binding']['device'] == case['physical_device'], 'Actual device differs from registered model context')
        checkpoint_config = Path('/models/Qwen3-0.6B/config.json')
        require(sha(checkpoint_config) == case['model_config_sha256'], 'Checkpoint config changed')
        (output / 'model_config.json').write_bytes(checkpoint_config.read_bytes())
        checkpoint = json.loads(checkpoint_config.read_bytes())
        require(not checkpoint.get('quantization_config') and (checkpoint.get('dtype') or checkpoint.get('torch_dtype')) in ('bfloat16', 'torch.bfloat16'), 'Expected original unquantized BF16 checkpoint')
        model = AutoModelForCausalLM.from_pretrained('/models/Qwen3-0.6B', dtype=torch.float16,
            attn_implementation='sdpa', local_files_only=True, trust_remote_code=False).eval().to('npu:0')
        require(model.config.num_hidden_layers == 28 and model.config.num_attention_heads == 16
                and model.config.num_key_value_heads == 8 and model.config.head_dim == 128
                and model.config._attn_implementation == 'sdpa', 'Unexpected actual Qwen3 architecture/path')
        require(all(parameter.dtype == torch.float16 for parameter in model.parameters()), 'Actual model parameters are not FP16')
        observer = ModelObserver(torch, torch_npu, case, output, model, config)
        result['runtime_imports'] = verify_imports(runtime)
        from msprobe.pytorch import PrecisionDebugger
        processor = importlib.import_module('msprobe.core.dump.data_dump.data_processor.pytorch_processor')
        collector['actual_loaded'] = verify_loaded(collector, processor)
        write(output / 'collector_provenance.json', collector)
        result['statistics_collector'] = collector
        count = len(case['mode_sequence']) * 2
        write(output / 'config.json', statistics_config(output, count))
        debugger = PrecisionDebugger(config_path=str(output / 'config.json'))
        ids = torch.tensor([row['prompt_token_ids'] for row in requests], dtype=torch.long, device='npu:0')
        next_ids = torch.tensor([[row['teacher_forced_next_token']] for row in requests], dtype=torch.long, device='npu:0')
        positions = {'prefill': torch.arange(2048, dtype=torch.long, device='npu:0').unsqueeze(0),
                     'decode': torch.tensor([[2048]], dtype=torch.long, device='npu:0')}
        require(ids.cpu().tolist() == [row['prompt_token_ids'] for row in requests], 'Actual prompt tensor differs')
        with torch.inference_mode(), sdpa_kernel(SDPBackend.FLASH_ATTENTION):
            for run_index, mode in enumerate(case['mode_sequence']):
                cache = None
                for stage_index, stage in enumerate(('prefill', 'decode')):
                    step = run_index * 2 + stage_index
                    phase = {'step': step, 'run_index': run_index, 'mode': mode, 'stage': stage}
                    actual_ids = ids if stage == 'prefill' else next_ids
                    expected_ids = [row['prompt_token_ids'] if stage == 'prefill' else [row['teacher_forced_next_token']] for row in requests]
                    require(actual_ids.cpu().tolist() == expected_ids, 'Actual request IDs changed')
                    before_cache = 0 if cache is None else cache.get_seq_length()
                    require(before_cache == (0 if stage == 'prefill' else 2048), 'Cache from a different run/length')
                    mapping = output / 'phases' / f'step{step}' / 'request_mapping.json'
                    rows = request_rows(requests, stage)
                    write(mapping, {**phase, 'schema': 'nes.model_request_mapping.v1', 'request_rows': rows,
                        'actual_input_ids_sha256': token_sha([value for row in actual_ids.cpu().tolist() for value in row]),
                        'actual_position_ids_sha256': token_sha(positions[stage].cpu().reshape(-1).tolist()),
                        'cache_source_run': run_index, 'cache_length_before': before_cache})
                    observer.begin(phase, rows, mapping)
                    meter = resource_meter.begin_window(torch_npu)
                    debugger.start(model=model, rank_id=0)
                    active = True
                    if observer.wrapper is None:
                        observer.install(functional)
                    require(functional.scaled_dot_product_attention is observer.wrapper, 'msprobe replaced the observed natural SDPA')
                    entered = emit('model_phase_enter', case_id=case['case_id'], batch_size=case['batch_size'], **phase)
                    try:
                        replies = model(input_ids=actual_ids, position_ids=positions[stage], attention_mask=None,
                            past_key_values=cache, use_cache=True, logits_to_keep=1, return_dict=True)
                        torch_npu.npu.synchronize()
                    finally:
                        debugger.stop()
                        debugger.step()
                        active = False
                    exited = emit('model_phase_exit', status='returned', case_id=case['case_id'], batch_size=case['batch_size'], **phase)
                    resource_call = {'status': 'returned', 'enter_monotonic_ns': entered['monotonic_ns'], 'exit_monotonic_ns': exited['monotonic_ns']}
                    measured = resource_meter.finish_window(torch_npu, meter, resource_call)
                    require(actual_ids.cpu().tolist() == expected_ids, 'Model changed the actual request token tensor')
                    require(positions[stage].cpu().reshape(-1).tolist() == (list(range(2048)) if stage == 'prefill' else [2048]),
                            'Model changed the actual query positions')
                    cache = replies.past_key_values
                    require(cache is not None and cache.get_seq_length() == (2048 if stage == 'prefill' else 2049), 'Actual model cache length invalid')
                    call_ids = observer.end()
                    require(list(replies.logits.shape) == [case['batch_size'], 1, model.config.vocab_size], 'All-request logits mapping differs')
                    private = output / 'private_output' / f'step{step}_logits.bin'
                    logits = tensor_snapshot(torch, torch_npu, replies.logits, private_file=private)
                    phase.update(request_mapping_file=mapping.relative_to(output).as_posix(), request_mapping_sha256=sha(mapping),
                        layer_count=28, sdpa_call_ids=call_ids, logits=logits, private_logits_file=private.relative_to(output).as_posix(),
                        cache_length_before=before_cache, cache_length_after=cache.get_seq_length(), resource_measurement=measured,
                        model_enter=entered, model_exit=exited)
                    phases.append(phase)
                    write(output / 'phase_manifest.json', {'schema': 'nes.model_phases.v1', 'phases': phases})
                    result.update(calls=observer.calls, original_api_calls=observer.original_calls,
                                  completed_phases=len(phases), native_log_reader=observer.launch_reader.receipt())
                    write(output / 'result.json', result)
                del cache, replies
        observer.restore()
        result['native_log_reader'] = observer.launch_reader.receipt()
        result['original_api_calls'] = observer.original_calls
        result['calls'] = observer.calls
        require(result['original_api_calls'] == count * 28 and len(observer.calls) == count * 28, 'Original SDPA call count differs')
        observer = None
        receipt = statistics_receipt(output, collector, count)
        write(output / 'receipt.json', receipt)
        result.update(status='CAPTURE_COMPLETED', receipt_sha256=sha(output / 'receipt.json'),
            phase_manifest_file='phase_manifest.json', phase_manifest_sha256=sha(output / 'phase_manifest.json'),
            module_inventory_sha256=sha(output / 'module_inventory.json'), module_events_sha256=sha(output / 'module_events.jsonl'),
            checkpoint_unchanged_after=verify_fingerprint(case['model_manifest_file'], 'nes.model_file_fingerprint.v1')['files'],
            tokenizer_unchanged_after=verify_fingerprint(case['tokenizer_manifest_file'], 'nes.tokenizer_file_fingerprint.v1')['files'])
        write(output / 'result.json', result)
        print(json.dumps({'status': result['status'], 'case_id': case['case_id'], 'original_api_calls': result['original_api_calls']}), flush=True)
        return 0
    except BaseException:
        result.update(status='CAPTURE_FAILED', traceback=traceback.format_exc(), calls=observer.calls if observer else result['calls'])
        if observer is not None:
            result['native_log_reader'] = observer.launch_reader.receipt()
        write(output / 'result.json', result)
        print(result['traceback'], file=sys.stderr, flush=True)
        return 1
    finally:
        if active and debugger is not None:
            debugger.stop()
        if observer is not None:
            observer.restore()


if __name__ == '__main__':
    raise SystemExit(main())
