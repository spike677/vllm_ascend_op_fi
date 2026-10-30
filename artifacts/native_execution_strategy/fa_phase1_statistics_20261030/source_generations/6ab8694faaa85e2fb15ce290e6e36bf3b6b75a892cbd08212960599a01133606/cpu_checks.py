"""CPU-only source, resource-boundary, native-log framing and receipt checks."""
import argparse
import ast
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parent
sys.dont_write_bytecode = True
sys.path.insert(0, str(ROOT))
import controller
import native_evidence as ne
import replay_support as rs
import replay
import resource_meter


def reject(function):
    try:
        function()
    except (ValueError, RuntimeError):
        return
    raise AssertionError('Expected rejection')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists()
    checks = []
    def check(name, function):
        function()
        checks.append({'name': name, 'passed': True})
    temp_parent = ROOT / 'outputs'
    temp_parent.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='nes_harness_cpu_', dir=temp_parent) as temporary:
        temp = Path(temporary).resolve()
        assert temp.is_relative_to(temp_parent.resolve())  # Bound automatic temporary cleanup to this workspace.
        def syntax():
            for p in ROOT.glob('*.py'):
                ast.parse(p.read_text(encoding='utf-8'), filename=str(p))
        check('all_harness_python_syntax_without_framework_import', syntax)
        def shared_session_entry():
            source = (ROOT / 'replay.py').read_text(encoding='utf-8')
            tree = ast.parse(source)
            calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)]
            execute = [node for node in calls if isinstance(node.func, ast.Attribute) and node.func.attr == 'execute']
            assert len(execute) == 1 and ast.unparse(execute[0].func) == 'selected_session.execute'
            assert not any(isinstance(node.func, ast.Attribute) and node.func.attr in
                           ('evaluate', 'native_scope', 'parse_native_evidence', 'run_native', 'run_controlled') for node in calls)
            invoke = next(node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == 'invoke_original')
            native = [node for node in ast.walk(invoke) if isinstance(node, ast.Call)
                      and isinstance(node.func, ast.Name) and node.func.id == 'module']
            assert len(native) == 1
            observer = next(node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == 'observer')
            assert 'freeze_step_metadata()' in ast.unparse(observer)
            assert 'native_evidence.verified_record' in ast.unparse(observer)
            assert "selected_session = off_session if mode == 'off' else session" in source
            assert 'session_result.observation_error is None and session_result.observation is not None' in source
            assert "native is captured.get('original_return')" in source
        check('harness_uses_one_session_execute_and_one_original_call_with_strict_observation', shared_session_entry)
        container = {'HostConfig': {'Privileged': False, 'Devices': [
            {'PathOnHost': '/dev/davinci2'}, {'PathOnHost': '/dev/davinci_manager'}]}, 'Mounts': []}
        def selected_card():
            assert not controller.device_capability(container, 5)
            assert controller.device_capability(container, 2)
            other = copy.deepcopy(container)
            other['HostConfig']['Devices'][0]['PathOnHost'] = '/dev/davinci5'
            assert controller.device_capability(other, 5)
        check('explicit_card2_does_not_grant_card5_but_card5_does', selected_card)
        def privileged_and_root():
            for kind in ('privileged', 'root_bind'):
                other = copy.deepcopy(container)
                if kind == 'privileged': other['HostConfig']['Privileged'] = True
                else: other['Mounts'] = [{'Source': '/'}]
                assert controller.device_capability(other, 5)
        check('privileged_and_root_bind_remain_selected_card_capable', privileged_and_root)
        check('cleanup_rejects_foreign_container', lambda: reject(lambda: controller.verify_owned(
            {'Id': 'b' * 64, 'Image': controller.IMAGE, 'Name': '/owned',
             'Config': {'Labels': {'demoncat.controller_token': 'token', 'demoncat.task': controller.TASK}}},
            'token', 'owned', 'a' * 64)))
        def runtime_inventory():
            task = temp / 'runtime_task'; task.mkdir()
            for name in ('controller.py', 'env.sh'): (task / name).write_text('# fixture\n')
            (task / 'memory_api_source_audit.json').write_text('{}\n')
            package = task / 'runtime/native_execution_strategy'; package.mkdir(parents=True)
            p = package / '__init__.py'; p.write_text('# fixture\n')
            payload = p.read_bytes()
            manifest = {'schema': 'nes.runtime_snapshot.v1',
                'files': {'native_execution_strategy/__init__.py': {'sha256': hashlib.sha256(payload).hexdigest(), 'bytes': len(payload)}},
                'native_components_sha256': {'fixture_header': 'a' * 64}}
            (task / 'runtime_manifest.json').write_text(json.dumps(manifest))
            prior = controller.ROOT; controller.ROOT = task
            try:
                found = controller.source_manifest()
                assert 'runtime/native_execution_strategy/__init__.py' in found and 'runtime_manifest.json' in found
                extra = package / 'extra.py'; extra.write_text('# extra')
                reject(controller.source_manifest); extra.unlink()
                p.write_text('# changed')
                reject(controller.source_manifest)
            finally: controller.ROOT = prior
        check('runtime_snapshot_rejects_extra_and_changed_files', runtime_inventory)
        before = {'query': {'actual_format': 0, 'full_sha256': 'a', 'object_id': 1, 'version': 0}}
        check('native_zero_to_two_descriptor_only_normalization', lambda: rs.input_integrity(
            before, {'query': {**before['query'], 'actual_format': 2}}))
        check('hash_change_rejected', lambda: reject(lambda: rs.input_integrity(
            before, {'query': {**before['query'], 'full_sha256': 'b'}})))
        check('unreviewed_descriptor_transition_rejected', lambda: reject(lambda: rs.input_integrity(
            {'query': {**before['query'], 'actual_format': 2}}, before)))
        def device_binding():
            folder = temp / 'device_binding'; folder.mkdir()
            before = {'selected_card': 5, 'eligible': True, 'npu_smi': '| 5 910B4 | OK |\n'}
            identity = {'devices': [{'PathOnHost': '/dev/davinci5'}, {'PathOnHost': '/dev/davinci_manager'}]}
            (folder / 'before.json').write_text(json.dumps(before))
            (folder / 'container_identity.json').write_text(json.dumps(identity))
            public = SimpleNamespace(npu=SimpleNamespace(get_device_name=lambda index: 'PublicAPI910BAlias'))
            binding = replay.physical_device_binding(public, folder)
            assert binding['device'] == 'Ascend910B4' and binding['public_api_device_name'] == 'PublicAPI910BAlias'
            identity['devices'][0]['PathOnHost'] = '/dev/davinci2'
            (folder / 'container_identity.json').write_text(json.dumps(identity))
            reject(lambda: replay.physical_device_binding(public, folder))
        check('physical_device_binds_actual_smi_and_exclusive_container_mapping_preserving_api_alias', device_binding)
        call = {'call_id': 'fixture_call', 'case_id': 'fixture', 'api': 'torch_npu.npu_fusion_attention_v3',
                'pid': 123, 'thread_id': 456, 'scope_token': 'fixture_token', 'phase': 'operator_execution',
                'batch_size': 1, 'step': 0}
        lines = [json.dumps({'event': 'native_api_enter', **call}) + '\n',
            'OP(123,fixture) [NNOP][Launch][456] OpName:[aclnnFlashAttentionScore_1_FlashAttentionScore] Tiling Key:22619644976, len:16, numBlocks:20, Workspace Num:1.\n',
            'OP(123,fixture) [SplitDataAndPrint][456] OpName:[aclnnFlashAttentionScore_1_FlashAttentionScore] PrintTilingData[0]: Aligned TilingData Size: 16, data(uint32_t): 1 2 3 4\n',
            'OP(123,fixture) [CommonOpExecutorRun][456] OpName:[aclnnFlashAttentionScore_1] Workspace addr: 0x100, size:4096, end\n',
            json.dumps({'event': 'native_api_exit', **call, 'status': 'returned'}) + '\n']
        def framed():
            p = temp / 'framed.log'; payload = ''.join(lines).encode(); p.write_bytes(payload + b'later unrelated bytes\n')
            output = temp / 'window.log'; result = ne.actual_launch(p, call, output)
            assert output.read_bytes() == payload and result['raw_hex'] == '01000000020000000300000004000000'
            assert result['source_prefix_sha256'] == hashlib.sha256(payload).hexdigest()
            assert result['framing_verified_against_public_context'] is False
        check('exact_native_window_and_uint32_framing_no_host_claim', framed)
        def invalid_log(name, changed):
            p = temp / (name + '.log'); p.write_text(''.join(changed))
            reject(lambda: ne.actual_launch(p, call, temp / (name + '.window')))
        check('wrong_pid_rejected', lambda: invalid_log('wrong_pid', [line.replace('OP(123,', 'OP(124,') for line in lines]))
        def worker_tid():
            p = temp / 'worker.log'; p.write_text(''.join(line.replace('[456]', '[457]') for line in lines))
            result = ne.actual_launch(p, call, temp / 'worker.window')
            assert result['tid'] == 457 and call['thread_id'] == 456
        check('distinct_cann_worker_tid_allowed_when_native_records_agree', worker_tid)
        check('raw_launch_tid_disagreement_rejected', lambda: invalid_log('wrong_tid',
            [lines[0], lines[1], lines[2].replace('[456]', '[457]'), *lines[3:]]))
        check('missing_exit_rejected', lambda: invalid_log('missing_exit', lines[:-1]))
        check('duplicate_raw_chunk_rejected', lambda: invalid_log('duplicate', [*lines[:3], lines[2], *lines[3:]]))
        raw_prefix = 'OP(123,fixture) [SplitDataAndPrint][456] OpName:[aclnnFlashAttentionScore_1_FlashAttentionScore] '
        def chunks_text(parts):
            raw_lines = [raw_prefix + f'PrintTilingData[{i}]: ' +
                         ('Aligned TilingData Size: 16, data(uint32_t): ' if i == 0 else '') + part + '\n'
                         for i, part in enumerate(parts)]
            return [*lines[:2], *raw_lines, *lines[3:]]
        def split_at_every_character():
            payload = '1 1024 32 4294967295 '
            expected = '010000000004000020000000ffffffff'
            for cut in range(1, len(payload)):
                p = temp / f'character_cut_{cut}.log'
                p.write_text(''.join(chunks_text([payload[:cut], payload[cut:]])))
                value = ne.actual_launch(p, call, temp / f'character_cut_{cut}.window')
                assert value['raw_hex'] == expected
                assert value['raw_chunk_payload_lengths'] == [cut, len(payload) - cut]
                assert value['raw_text_sha256'] == hashlib.sha256(payload.encode('ascii')).hexdigest()
                assert value['raw_reconstruction'] == 'exact_character_concat_before_uint32_split'
        check('all_character_cuts_preserve_midnumber_and_leading_trailing_word_boundaries', split_at_every_character)
        def adjacent_spaces():
            p = temp / 'adjacent_spaces.log'
            p.write_text(''.join(chunks_text(['1  ', '102', '4 ', ' 32 4294967295 '])))
            value = ne.actual_launch(p, call, temp / 'adjacent_spaces.window')
            assert value['raw_hex'] == '010000000004000020000000ffffffff'
        check('multi_chunk_midnumber_and_multiple_payload_spaces_preserved', adjacent_spaces)
        check('missing_chunk_index_rejected', lambda: invalid_log('missing_chunk_index',
            [line.replace('PrintTilingData[1]', 'PrintTilingData[2]') for line in chunks_text(['1 102', '4 32 4294967295 '])]))
        check('missing_fixed_payload_delimiter_rejected', lambda: invalid_log('missing_payload_delimiter',
            [line.replace('data(uint32_t): ', 'data(uint32_t):') for line in lines]))
        check('invalid_raw_character_rejected', lambda: invalid_log('raw_character', chunks_text(['1 102x', '4 32 4294967295 '])))
        check('raw_uint32_overflow_rejected', lambda: invalid_log('raw_overflow', chunks_text(['1 1024 32 4294967296 '])))
        check('truncated_uint32_wordlist_rejected', lambda: invalid_log('raw_short', chunks_text(['1 1024 32 '])))
        check('extra_malformed_selected_chunk_rejected', lambda: invalid_log('raw_extra_bad',
            [*lines[:3], raw_prefix + 'PrintTilingData[1]: not_a_uint32\n', *lines[3:]]))
        def receipts():
            out = temp / 'statistics'; out.mkdir()
            rs.write(out / 'config.json', rs.statistics_config(out, 4))
            rs.write(out / 'collector_provenance.json', {'modified_official_collector': True})
            api, module = 'NPU.npu_fusion_attention_v3.0.forward', 'Module.replay.ReplayModule.forward.0'
            for step in range(4):
                target = out / f'observation/step{step}/rank0'
                rs.write(target / 'dump.json', {'task': 'statistics', 'level': 'mix', 'dump_data_dir': None, 'data': {api: {}, module: {}}})
                rs.write(target / 'construct.json', {api: module, module: None})
                rs.write(target / 'stack.json', {'0': [[api, module], ['File /audit/replay.py, line 1, in forward, original_call']]})
            value = rs.statistics_receipt(out, {'modified_official_collector': True}, 4)
            assert len(value['native_files']) == 12 and value['steps'] == [0, 1, 2, 3]
            (out / 'observation/step3/rank0/construct.json').write_text(json.dumps({api: None}))
            reject(lambda: rs.statistics_receipt(out, {}, 4))
        check('four_steps_have_original_api_module_and_receipt_bindings', receipts)
        completed_call = {'call_id': 'resource_fixture', 'status': 'returned',
                          'enter_monotonic_ns': 1_000_000_000, 'exit_monotonic_ns': 2_250_000_000}
        def metering():
            events = []
            api = SimpleNamespace(npu=SimpleNamespace(reset_peak_memory_stats=lambda device: events.append(('reset', device)),
                memory_stats=lambda device: {'allocated_bytes.all.peak': 256}, max_memory_allocated=lambda device: 300))
            measured = resource_meter.finish_window(api, resource_meter.begin_window(api), completed_call)
            assert events == [('reset', 0)] and measured['resources'] == {'device_seconds': 1.25, 'peak_memory_bytes': 300}
            assert measured['accounting_complete'] is True and measured['whole_card_hbm_measured'] is False
            path = temp / 'step_metadata.json'
            rs.write(path, {'schema': 'nes.harness.step_metadata.v1', 'native_call': completed_call,
                           'resource_measurement': measured, 'resource_meter_source_sha256': ne.sha(resource_meter.__file__)})
            resources, ref, evidence = ne.bound_resources(path, completed_call)
            assert resources == measured['resources'] and ref['step_metadata_sha256'] == ne.sha(path) and len(evidence) == 1
            reject(lambda: ne.bound_resources(path, dict(completed_call, call_id='another_call')))
        check('measured_window_allocator_peak_and_exact_step_metadata_sha_binding', metering)
        def unavailable_api():
            api = SimpleNamespace(npu=SimpleNamespace())
            measured = resource_meter.finish_window(api, resource_meter.begin_window(api), completed_call)
            assert measured['resources'] == {'device_seconds': 1.25, 'peak_memory_bytes': None}
            assert measured['accounting_status'] == 'RESOURCE_ACCOUNTING_UNKNOWN'
        check('missing_public_memory_api_keeps_peak_unknown', unavailable_api)
        def absent_counter():
            getters = []
            api = SimpleNamespace(npu=SimpleNamespace(reset_peak_memory_stats=lambda device: None,
                memory_stats=lambda device: {}, max_memory_allocated=lambda device: getters.append(device) or 0))
            measured = resource_meter.finish_window(api, resource_meter.begin_window(api), completed_call)
            assert measured['resources']['peak_memory_bytes'] is None and getters == []
        check('missing_peak_counter_not_misreported_as_public_getter_default_zero', absent_counter)
        def reset_failure():
            def failed(device): raise RuntimeError('fixture reset failed')
            api = SimpleNamespace(npu=SimpleNamespace(reset_peak_memory_stats=failed,
                memory_stats=lambda device: {'allocated_bytes.all.peak': 512}, max_memory_allocated=lambda device: 512))
            measured = resource_meter.finish_window(api, resource_meter.begin_window(api), completed_call)
            assert measured['resources']['peak_memory_bytes'] is None and measured['peak_reset_completed'] is False
        check('failed_reset_does_not_publish_process_lifetime_peak_as_window_peak', reset_failure)
        def incomplete_call():
            api = SimpleNamespace(npu=SimpleNamespace())
            measured = resource_meter.finish_window(api, resource_meter.begin_window(api), {'status': 'failed'})
            assert measured['resources'] == {'device_seconds': None, 'peak_memory_bytes': None}
        check('incomplete_synchronized_call_retains_unknown_duration', incomplete_call)
    report = {'schema': 'nes.harness.cpu_review.v1', 'status': 'PASS_CPU_PREPARATION_CHECKS',
              'checks': checks, 'npu_calls': 0, 'containers_started': 0,
              'source_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(ROOT.glob('*.py'))},
              'not_verified': ['Actual new vendor execution', 'Adapter real record integration', 'Native strategy or numerical acceptance']}
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(report, stream, indent=2); stream.write('\n')
    print(json.dumps({'status': report['status'], 'checks': len(checks), 'output': str(args.output),
                      'sha256': hashlib.sha256(args.output.read_bytes()).hexdigest()}))


if __name__ == '__main__':
    main()
