"""Real FA discovery or native -> controlled -> controlled -> off replay.

Each step invokes the original public API once. Policies select only adapter
candidates; this harness never implements a tiling choice or a batch trigger.
"""
import argparse
from dataclasses import asdict, replace
import importlib
import importlib.metadata
import inspect
import json
import os
from pathlib import Path
import re
import sys
import threading
import time
import traceback
import uuid

from replay_support import (ORIGINAL_CALL_SHA, REFERENCE_QUERIES, SUPPORT_SHA, capture_outputs,
    input_integrity, load_frozen_inputs, reference_queries, require, sha, snapshot,
    statistics_config, statistics_receipt, write)
from runtime_binding import verify_runtime, verify_imports
import native_evidence
import resource_meter


def validate_case(case):
    require(case['schema'] == 'nes.harness.case.v1' and case['phase'] in ('discovery', 'controlled'), 'Unexpected case schema/phase')
    require(case['batch_size'] in (1, 2, 4, 8) and case['request_count'] == case['batch_size']
            and case['token_count'] == case['batch_size'] * 2048 and case['seq_lens'] == [2048] * case['batch_size'],
            'Explicit request/token/sequence context mismatch')
    modes = ['native'] if case['phase'] == 'discovery' else ['native', 'controlled', 'controlled', 'off']
    require(case['mode_sequence'] == modes and case['native_calls_per_process'] == len(modes), 'Fixed mode sequence differs')
    require(type(case['repeat_index']) is int and case['repeat_index'] in (0, 1), 'Two independent repeats required')
    require(case['capture_msprobe'] is True and case['statistics_variant'] == 'task_local_fp32_high_v1'
            and case['capture_private_target_output'] is True and case['reference_query_indices'] == REFERENCE_QUERIES,
            'Required statistics and private full-output evidence missing')
    require(case['native_log_required'] is True, 'Actual native launch evidence is required')
    if case['phase'] == 'controlled':
        require(case['batch_size'] in (1, 8) and case['operator_variant'] == 'private_vendor'
                and case['selection_target'] in ('query_tile', 'key_group', 'joint')
                and type(case['allow_experimental']) is bool, 'Controlled candidate scope differs')
    else:
        require(case['operator_variant'] == 'stock' and case['observation_only'] is True,
                'Discovery must remain original-library observation only')


def validate_loader(case, output, case_sha):
    require(os.environ.get('ACLNN_CACHE_LIMIT') == '0' and not os.environ.get('LD_PRELOAD'), 'Unapproved cache/preload state')
    require(not any(os.environ.get(k) for k in ('NES_FA_PLAN_JSON', 'NES_FA_LOG_PATH')), 'Inherited native scope forbidden')
    if case['operator_variant'] == 'stock':
        require(not os.environ.get('ASCEND_CUSTOM_OPP_PATH'), 'Stock discovery cannot load custom vendor')
        return {'operator_variant': 'stock', 'native_bridge_loaded': False}
    from vendor_exec import validate_vendor
    verified = validate_vendor(case)
    path = output.with_name(output.name + '.vendor_launch.json')
    require(os.environ.get('NES_VENDOR_LAUNCH_RECEIPT') == str(path)
            and path.is_file() and sha(path) == os.environ.get('NES_VENDOR_LAUNCH_RECEIPT_SHA256'),
            'Fresh-process vendor receipt missing')
    receipt = json.loads(path.read_bytes())
    require(receipt['schema'] == 'nes.vendor_launch.v1' and receipt['probe_sha256'] == sha(__file__)
            and receipt['case_sha256'] == case_sha and receipt['case_id'] == case['case_id']
            and receipt['verified_vendor']['manifest_sha256'] == case['approved_vendor_manifest_sha256'], 'Vendor launch binding differs')
    require(all(receipt['child_environment'].get(key) == os.environ.get(key) for key in
                ('ASCEND_CUSTOM_OPP_PATH', 'LD_LIBRARY_PATH', 'LD_PRELOAD', 'ACLNN_CACHE_LIMIT', 'PYTHONPATH')),
            'Actual loader environment differs from receipt')
    return {'operator_variant': 'private_vendor', 'manifest_sha256': verified['manifest_sha256'],
            'policy_source_manifest_sha256': verified['policy_source_manifest_sha256'],
            'artifact_count': len(verified['artifacts']), 'vendor_root': verified['vendor_root'],
            'launcher_receipt': str(path), 'launcher_receipt_sha256': sha(path), 'vendor_inventory_verified': True}


def actual_call_metadata(case, inputs, mask, effective_scale, device):
    def metadata(tensor):
        return {'shape': list(tensor.shape), 'dtype': str(tensor.dtype).removeprefix('torch.')}
    return {'operator': 'FlashAttentionScore', 'layout': 'BNSD',
        'request_count': case['request_count'], 'token_count': case['token_count'], 'seq_lens': case['seq_lens'],
        'inputs': {**{key: metadata(value) for key, value in inputs.items()}, 'atten_mask': metadata(mask)},
        'attributes': {'scale': effective_scale, 'keep_prob': 1, 'pre_tokens': 2147483647, 'next_tokens': 0,
                       'heads': 16, 'inner_precise': 0, 'sparse_mode': 2, 'optional_inputs_absent': True},
        'device': device}


def physical_device_binding(torch_npu, evidence_root):
    preflight_path = evidence_root / 'before.json'
    container_path = evidence_root / 'container_identity.json'
    preflight = json.loads(preflight_path.read_bytes())
    container = json.loads(container_path.read_bytes())
    require(preflight['selected_card'] == 5 and preflight['eligible'] is True, 'Physical card5 preflight missing')
    found = re.search(r'^\|\s*5\s+(\S+)\s*\|', preflight['npu_smi'], re.M)
    require(found is not None and found[1] == '910B4', 'Actual physical model is not the supported 910B4')
    devices = [row['PathOnHost'] for row in container['devices'] if re.fullmatch(r'/dev/davinci\d+', row['PathOnHost'])]
    require(devices == ['/dev/davinci5'], 'Logical NPU mapping is not the selected physical device')
    name = str(torch_npu.npu.get_device_name(0))
    require(name, 'Public device name unavailable')
    return {'device': 'Ascend' + found[1], 'public_api_device_name': name,
            'logical_device': 0, 'physical_card': 5, 'npu_smi_device_name': found[1],
            'preflight_sha256': sha(preflight_path), 'container_identity_sha256': sha(container_path),
            'source': 'Observed public API name plus controller npu-smi and single physical-device mapping'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case-file', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    case = json.loads(args.case_file.read_bytes())
    validate_case(case)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    result = {'schema': 'nes.harness.replay.v1', 'status': 'STARTING', 'case_id': case['case_id'],
        'phase': case['phase'], 'batch_size': case['batch_size'], 'repeat_index': case['repeat_index'],
        'source_case_sha256': sha(args.case_file), 'replay_source_sha256': sha(__file__),
        'runtime_manifest_sha256': case['runtime_manifest_sha256'], 'original_api_calls': 0, 'steps': [],
        'execution_kind': 'original_operator_execution', 'model_forward': False,
        'origin': 'Captured Qwen3 layer1 QKV provenance; this process does not execute a model layer.',
        'assessment': {'strategy_switched': None, 'numerics_observed': None, 'model_impact': None},
        'mechanism_accepted': False, 'controlled_validation_certificate_granted': False, 'start_epoch': time.time()}
    debugger, active = None, False
    save = lambda: write(output / 'result.json', result)
    save()
    try:
        runtime_manifest = verify_runtime(case)
        result['loader'] = validate_loader(case, output, result['source_case_sha256'])
        from native_execution_strategy.config import from_dict
        from native_execution_strategy.models import InvocationContext
        from native_execution_strategy.policies import Policy
        from native_execution_strategy.session import StrategySession
        cfg = from_dict(case['execution_config'])
        policy = Policy(cfg.policy)
        off_policy = Policy(replace(cfg.policy, enabled=False))
        adapter = None
        if case['phase'] == 'controlled':
            from native_execution_strategy.adapters.flash_attention import FlashAttentionAdapter
            from native_execution_strategy.registry import AdapterRegistry, AdapterRegistration
            require(sha(case['adapter_certificate_file']) == case['adapter_certificate_sha256'], 'Adapter certificate changed')
            require(case['adapter_evidence_root'] == '/audit/runtime/native_execution_strategy',
                    'Certificate evidence root must be the frozen runtime package')
            adapter = FlashAttentionAdapter(Path(case['adapter_certificate_file']), evidence_root=Path(case['adapter_evidence_root']),
                                           selection_target=cfg.selection_target, allow_experimental=case.get('allow_experimental', False),
                                           on_unsupported=cfg.adapter.on_unsupported)
            require(adapter.fingerprints.source_sha256 == result['loader']['policy_source_manifest_sha256']
                    and adapter.fingerprints.build_sha256 == result['loader']['manifest_sha256'],
                    'Certificate belongs to a different deployed native source/build')
            registry = AdapterRegistry()
            registry.register(AdapterRegistration(adapter.name, adapter.version, adapter.fingerprints,
                adapter.certificate['validation_state'], adapter.loaded.evidence, route='S1S2'), adapter)
            result['adapter_resolution'] = registry.resolve(cfg.adapter).as_dict()
            session = StrategySession(adapter, policy)
            off_session = StrategySession(adapter, off_policy)
        result['runtime_imports'] = verify_imports(runtime_manifest)
        from collector_stage import select_collector, verify_loaded
        collector = select_collector(case)
        import torch
        import torch_npu
        require(sys.byteorder == 'little', 'Little-endian captured input required')
        result['versions'] = {key: importlib.metadata.version(key) for key in ('torch', 'torch-npu')}
        manifest, cpu = load_frozen_inputs(torch, case)
        reference, effective_scale = reference_queries(torch, cpu, manifest['native_kwargs']['scale'])
        reference_contract = {'query_indices': REFERENCE_QUERIES, 'batch_slot': 0, 'heads': 16, 'head_dim': 128,
            'kv_group_rule': 'query_head//2', 'effective_scale_float32': effective_scale, 'compute_device': 'CPU',
            'dtype': 'float64', 'causal_keys': 'keys 0 through query_index inclusive', 'not_a_native_output_proxy': True}
        result['fp64_reference_contract'] = reference_contract
        torch_npu.npu.set_device(0)
        result['physical_device_binding'] = physical_device_binding(torch_npu, output.parent)
        inputs = {}
        for name, value in cpu.items():
            tensor = torch.empty_strided(value.shape, value.stride(), dtype=torch.float16, device='npu:0')
            tensor.copy_(value)
            inputs[name] = tensor
        mask = torch.triu(torch.ones((2048, 2048), dtype=torch.bool), diagonal=1).to('npu:0')
        kwargs = dict(manifest['native_kwargs'])
        kwargs.update(pse=None, padding_mask=None, atten_mask=mask, prefix=None, actual_seq_qlen=None, actual_seq_kvlen=None, sink=None)
        result['native_kwargs'] = {key: value for key, value in kwargs.items() if key != 'atten_mask'}
        actual_call = actual_call_metadata(case, inputs, mask, effective_scale, result['physical_device_binding']['device'])
        if adapter is not None:
            context = adapter.inspect(actual_call)
        else:
            context = InvocationContext('FlashAttentionScore', request_count=case['request_count'],
                token_count=case['token_count'], seq_lens=tuple(case['seq_lens']), dtype='float16', layout='BNSD',
                matrix_dims={'query_heads': inputs['query'].shape[1], 'kv_heads': inputs['key'].shape[1],
                             'query_tokens': inputs['query'].shape[2], 'kv_tokens': inputs['key'].shape[2],
                             'head_dim': inputs['query'].shape[3]}, attributes=actual_call['attributes'])
        require(context.request_count == case['request_count'], 'Adapter must retain explicit request count')
        result['invocation_context'] = context.as_dict()
        class ReplayModule(torch.nn.Module):
            def forward(self, query, key, value):
                return torch_npu.npu_fusion_attention_v3(query, key, value, **kwargs)
        module = ReplayModule().eval()
        holder = torch.nn.Module()
        holder.add_module('replay', module)
        from msprobe.pytorch import PrecisionDebugger
        processor = importlib.import_module('msprobe.core.dump.data_dump.data_processor.pytorch_processor')
        collector['actual_loaded'] = verify_loaded(collector, processor)
        support = Path(inspect.getsourcefile(PrecisionDebugger)).parents[1] / 'api_dump/support_wrap_ops.yaml'
        require(sha(support) == SUPPORT_SHA and '- npu_fusion_attention_v3' in support.read_text(), 'Native API collection support changed')
        write(output / 'collector_provenance.json', collector)
        write(output / 'config.json', statistics_config(output, len(case['mode_sequence'])))
        result['statistics_collector'] = collector
        debugger = PrecisionDebugger(config_path=str(output / 'config.json'))
        baseline_observation = None
        for step, mode in enumerate(case['mode_sequence']):
            step_output = output / 'steps' / f'step{step}'
            step_output.mkdir(parents=True, exist_ok=False)
            before = {name: snapshot(torch, torch_npu, tensor) for name, tensor in inputs.items()}
            mask_before = snapshot(torch, torch_npu, mask, batch_sliced=False)
            for name, value in before.items():
                expected = manifest['inputs'][name]
                require(value['full_sha256'] == expected['expected_full_sha256'][str(case['batch_size'])]
                        and value['per_batch_sha256'] == [expected['sha256']] * case['batch_size'], 'Actual input differs from frozen captured values')
            invocation_id = 'nes_' + uuid.uuid4().hex
            call = {'case_id': case['case_id'], 'call_id': invocation_id, 'api': 'torch_npu.npu_fusion_attention_v3',
                'pid': os.getpid(), 'thread_id': threading.get_native_id(), 'phase': 'operator_execution',
                'step': step, 'mode': mode, 'batch_size': case['batch_size'], 'scope_token': None}
            captured = {}
            metadata_path = step_output / 'step_metadata.json'
            profile_output = step_output / 'native_profile'
            log_path = output.parent / (case['case_id'] + '.log')

            def invoke_original():
                nonlocal active
                with torch.no_grad():
                    if adapter:
                        captured['plan'] = json.loads(os.environ['NES_FA_PLAN_JSON'])
                        call['scope_token'] = captured['plan']['scope']['token']
                        captured['host_path'] = Path(os.environ['NES_FA_LOG_PATH'])
                    else:
                        captured['plan'], captured['host_path'] = None, None
                    torch_npu.npu.synchronize()
                    captured['resource_start'] = resource_meter.begin_window(torch_npu, device=0)
                    debugger.start(model=holder, rank_id=0)
                    active = True
                    call.update(enter_epoch_ns=time.time_ns(), enter_monotonic_ns=time.monotonic_ns())
                    print(json.dumps({'event': 'native_api_enter', **call}), flush=True)
                    try:
                        result['original_api_calls'] += 1
                        native_value = module(inputs['query'], inputs['key'], inputs['value'])
                        captured['original_return'] = native_value
                        torch_npu.npu.synchronize()
                        call.update(status='returned', exit_epoch_ns=time.time_ns(), exit_monotonic_ns=time.monotonic_ns())
                    finally:
                        print(json.dumps({'event': 'native_api_exit', **call}), flush=True)
                        if active:
                            debugger.stop()
                            active = False
                            debugger.step()
                return native_value

            def freeze_step_metadata():
                require(not any(os.environ.get(key) for key in ('NES_FA_PLAN_JSON', 'NES_FA_LOG_PATH')), 'Native scope did not restore')
                captured['resource_measurement'] = resource_meter.finish_window(torch_npu, captured['resource_start'], call)
                write(metadata_path, {'schema': 'nes.harness.step_metadata.v1', 'case_id': case['case_id'],
                    'step': step, 'mode': mode, 'native_call': call,
                    'resource_measurement': captured['resource_measurement'],
                    'physical_device_binding': result['physical_device_binding'],
                    'resource_meter_source_sha256': sha(resource_meter.__file__)})

            def observer(value, plan, evidence_dir):
                require(value is captured['original_return'], 'Session observer did not receive the original returned object')
                require(plan == captured['plan'] and Path(evidence_dir) == step_output, 'Session observation scope differs')
                freeze_step_metadata()
                evidence = native_evidence.verified_record(log_path, call, profile_output, captured['host_path'],
                    Path(case['approved_pod_schema_file']), case['approved_pod_schema_sha256'], metadata_path)
                write(profile_output / 'verified_record.json', evidence)
                return evidence

            decision, session_execution = None, None
            if adapter:
                selected_session = off_session if mode == 'off' else session
                session_result = selected_session.execute(context, invoke_original, observer, invocation_id, step_output,
                    mode='native' if mode == 'native' else 'controlled')
                native = session_result.value
                session_execution = {'schema': 'nes.harness.session_execution.v1',
                    'entry_point': 'StrategySession.execute', 'mode': 'native' if mode == 'native' else 'controlled',
                    'policy_enabled': mode != 'off', 'original_return_identity': native is captured.get('original_return'),
                    'observation_error': session_result.observation_error,
                    'passthrough_reason': session_result.passthrough_reason,
                    'observation_present': session_result.observation is not None,
                    'decision': None if session_result.decision is None else session_result.decision.as_dict(),
                    'assessment': session_result.assessment.as_dict()}
                write(step_output / 'session_execution.json', session_execution)
                require(session_execution['original_return_identity'], 'Session changed the original returned object')
                require(session_result.observation_error is None and session_result.observation is not None,
                        'Required native observation failed: ' + str(session_result.observation_error or session_result.passthrough_reason))
                decision = session_result.decision
                require((decision is None) if mode == 'native' else
                        (decision is not None and decision.requires_execution_change is (mode == 'controlled')),
                        'Session policy did not produce expected native/eligible/off intent')
            else:
                native = invoke_original()
                freeze_step_metadata()
            resource_measurement = captured['resource_measurement']
            record = {'step': step, 'mode': mode, 'native_call': call,
                'policy_decision': None if decision is None else decision.as_dict(), 'typed_plan_intent': captured['plan'],
                'session_execution': session_execution,
                'selection_current_observation_step': 0 if baseline_observation is not None else None,
                'inputs_before': before, 'mask_before': mask_before,
                'resources': resource_measurement['resources'], 'resource_measurement': resource_measurement,
                'step_metadata_file': metadata_path.relative_to(output).as_posix(), 'step_metadata_sha256': sha(metadata_path),
                'assessment': {'strategy_switched': None, 'numerics_observed': None, 'model_impact': None}}
            if adapter:
                observed = session_result.observation
                record['native_observation'] = asdict(observed)
                record['assessment'].update(session_result.assessment.as_dict())
                if step == 0:
                    baseline_observation = observed
            else:
                launch = native_evidence.actual_launch(log_path, call, profile_output / 'native_window.log')
                require(sha(case['approved_pod_schema_file']) == case['approved_pod_schema_sha256'], 'POD schema changed')
                schema = json.loads(Path(case['approved_pod_schema_file']).read_bytes())
                try:
                    pod = native_evidence.decode_pod(bytes.fromhex(launch['raw_hex']), schema)
                    discovery = {'status': 'NATIVE_POD_OBSERVED_ONLY', 'native_launch': launch, 'pod': pod}
                except ValueError as error:
                    discovery = {'status': 'NATIVE_PROFILE_UNSUPPORTED_SCHEMA_OBSERVATION_ONLY', 'native_launch': launch, 'error': str(error)}
                discovery.update(controlled_support_granted=False, host_callback_available=False,
                    resources=resource_measurement['resources'],
                    resource_measurement={'step_metadata_file': str(metadata_path), 'step_metadata_sha256': sha(metadata_path),
                        'accounting_status': resource_measurement['accounting_status'],
                        'device_seconds_semantics': resource_measurement['device_seconds_semantics'],
                        'allocator_window_semantics': resource_measurement['allocator_window_semantics']})
                write(profile_output / 'stock_discovery.json', discovery)
                record['native_observation_only'] = discovery
            after = {name: snapshot(torch, torch_npu, tensor) for name, tensor in inputs.items()}
            mask_after = snapshot(torch, torch_npu, mask, batch_sliced=False)
            record.update(inputs_after=after, mask_after=mask_after,
                          input_format_normalizations=input_integrity(before, after))
            input_integrity({'mask': mask_before}, {'mask': mask_after})
            record.update(capture_outputs(torch, torch_npu, native, case['batch_size'], step_output, reference, reference_contract))
            if result['steps']:
                record['assessment']['numerics_observed'] = record['output']['target_sha256'] != result['steps'][0]['output']['target_sha256']
            result['steps'].append(record)
            write(step_output / 'result.json', record)
            save()
            del native
        require(result['original_api_calls'] == case['native_calls_per_process'], 'Original call count differs')
        if case['phase'] == 'controlled':
            result['every_match_checks'] = {'eligible_calls': 2, 'observed_switches': [result['steps'][i]['assessment']['strategy_switched'] for i in (1, 2)],
                'same_policy_instance': True, 'both_selected_from_step0_native_observation': True}
            result['off_recovery'] = {'native_strategy_restored': result['steps'][3]['assessment']['strategy_switched'] is False,
                'target_output_restored': result['steps'][3]['assessment']['numerics_observed'] is False,
                'auxiliary_returns_not_silently_conflated_with_primary_output': True}
        receipt = statistics_receipt(output, collector, len(case['mode_sequence']))
        write(output / 'receipt.json', receipt)
        result['receipt_sha256'] = sha(output / 'receipt.json')
        result['runtime_imports'] = verify_imports(runtime_manifest)
        result.update(status='CAPTURE_COMPLETED', input_unchanged=True, host_scope_restored=True, end_epoch=time.time())
        save()
        print(json.dumps({'status': result['status'], 'case_id': case['case_id'], 'original_api_calls': result['original_api_calls'],
                          'mechanism_accepted': False, 'model_impact': None}), flush=True)
        return 0
    except BaseException:
        result.update(status='CAPTURE_FAILED', traceback=traceback.format_exc(), end_epoch=time.time())
        save()
        print(result['traceback'], file=sys.stderr)
        return 1
    finally:
        if active:
            debugger.stop()
            debugger.step()


if __name__ == '__main__':
    raise SystemExit(main())
