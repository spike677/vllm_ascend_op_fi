"""CPU-only resource isolation, immutable-dependency and native-framing checks.

No framework imports, subprocess execution, device queries or container actions.
Synthetic fixtures do not establish model or NPU execution correctness.
"""
import argparse
import ast
import copy
import hashlib
import json
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parent
sys.dont_write_bytecode = True
sys.path.insert(0, str(ROOT))
import controller
import native_evidence as ne
import runtime_binding
import vendor_exec


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

    def forbidden(*args, **kwargs):
        raise AssertionError('CPU checks must not execute subprocesses or inspect devices')

    controller.run = forbidden
    controller.subprocess.run = forbidden
    controller.subprocess.Popen = forbidden
    temp_parent = ROOT / 'analysis'
    temp_parent.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='cpu_', dir=temp_parent) as directory:
        temp = Path(directory).resolve()
        assert temp.is_relative_to(temp_parent.resolve())

        def syntax():
            for path in ROOT.glob('*.py'):
                ast.parse(path.read_text(encoding='utf-8'), filename=str(path))
            assert 'torch' not in sys.modules and 'torch_npu' not in sys.modules
        check('new_top_python_syntax_without_framework_import', syntax)

        def bounds():
            source = (ROOT / 'controller.py').read_text()
            tree = ast.parse(source)
            arguments = {}
            for node in ast.walk(tree):
                if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                        and node.func.attr == 'add_argument' and node.args
                        and isinstance(node.args[0], ast.Constant)):
                    arguments[node.args[0].value] = {keyword.arg: ast.literal_eval(keyword.value)
                        for keyword in node.keywords if keyword.arg in ('default', 'choices')}
            assert arguments['--card']['choices'] == (5,)
            assert arguments['--probe']['choices'] == ('model_probe.py',)
            assert arguments['--case-seconds']['default'] == 900
            assert arguments['--max-seconds']['default'] == 1800
            assert arguments['--max-log-mib']['default'] == 2048
            assert "('/home/gyl/models/Qwen3-0.6B', '/models/Qwen3-0.6B', True)" in source
            assert "'--network', 'none', '--memory', '16g', '--cpus', '8'" in source
            assert 'ASCEND_GLOBAL_LOG_LEVEL=0' in source
            assert 'external_container_mutations\': False' in source
            assert controller.ROOT.as_posix() == '/home/fault28-model-coverage-20261030'
            assert controller.ALLOW_IDLE_IDS == set()
            assert set(controller.REVIEWED_IDLE_CONTAINERS) == {
                'a8eeda3a5db34389fbcc83fdd8cabc99d687ac1db5aa4011a0586f2a9f0ddc4c',
                'db21bd452c30c64dfd61152b6cf0a98bebfcea468a0f1c8a18452d2cb0e99de1'}
        check('card5_only_bounded_resources_readonly_model_and_exact_idle_ids', bounds)

        def routing():
            case = {'case_id': 'fixture', 'file': 'setup/fixture.json',
                    'case': {'operator_variant': 'private_vendor'}}
            actual = controller.build_probe_command('model_probe.py', case, 'native-logs')
            assert actual == ['python', '/audit/vendor_exec.py', '--case-file', '/audit/setup/fixture.json',
                              '--output', '/evidence/fixture']
            assert str(vendor_exec.PROBE) == '/audit/model_probe.py'
            reject(lambda: controller.build_probe_command('replay.py', case, 'native-logs'))
            reject(lambda: controller.build_probe_command('model_probe.py',
                dict(case, case={'operator_variant': 'unknown'}), 'native-logs'))
        check('private_route_has_only_validated_launcher_then_model_probe', routing)

        container = {'HostConfig': {'Privileged': False, 'Devices': [
            {'PathOnHost': '/dev/davinci2'}, {'PathOnHost': '/dev/davinci_manager'}]}, 'Mounts': []}
        def card_capability():
            assert controller.device_capability(container, 2)
            assert not controller.device_capability(container, 5)
            item = copy.deepcopy(container)
            item['HostConfig']['Devices'][0]['PathOnHost'] = '/dev/davinci5'
            assert controller.device_capability(item, 5)
            for kind in ('privileged', 'root_bind'):
                item = copy.deepcopy(container)
                if kind == 'privileged': item['HostConfig']['Privileged'] = True
                else: item['Mounts'] = [{'Source': '/'}]
                assert controller.device_capability(item, 5)
        check('explicit_mapping_and_broad_device_capability_remain_fail_closed', card_capability)
        check('foreign_cleanup_identity_rejected', lambda: reject(lambda: controller.verify_owned(
            {'Id': 'b' * 64, 'Image': controller.IMAGE, 'Name': '/owned',
             'Config': {'Labels': {'demoncat.controller_token': 'token', 'demoncat.task': controller.TASK}}},
            'token', 'owned', 'a' * 64)))

        def process_identity():
            owned = 'a' * 64
            item = {'host_pid': 123, 'starttime_ticks': 456, 'pid_namespace': 'pid:[78]',
                    'nspid_vector': [123, 1], 'cgroup': '0::/docker/' + owned,
                    'process_state': 'S', 'process_flags': 0}
            proven, teardown = {}, {}
            assert controller.classify_process_identity(item, owned, proven, teardown, 0)[0]
            foreign = dict(item, cgroup='0::/', process_state='S')
            assert not controller.classify_process_identity(foreign, owned, proven, teardown, 1)[0]
            zombie = dict(foreign, process_state='Z')
            assert controller.classify_process_identity(zombie, owned, proven, teardown, 2)[0]
            assert not controller.classify_process_identity(zombie, owned, proven, teardown, 63)[0]
            reused = dict(zombie, starttime_ticks=457)
            assert not controller.classify_process_identity(reused, owned, proven, teardown, 3)[0]
            missing = controller.missing_process_identity(123, owned, True, proven, {}, 4)
            assert missing['owned'] is False and missing['retry_pending'] is True
        check('live_foreign_pid_reuse_and_unbounded_teardown_rejected', process_identity)

        def source_inventory():
            task = temp / 'source'; task.mkdir()
            for name in ('controller.py', 'env.sh', 'model_probe.py'):
                (task / name).write_text('# synthetic fixture\n')
            metadata = ('memory_api_source_audit.json', 'inputs/manifest.json',
                        'model_manifest.json', 'tokenizer_manifest.json', 'source_fingerprints.json')
            for name in metadata:
                path = task / name; path.parent.mkdir(exist_ok=True)
                path.write_text('{}\n')
            package = task / 'runtime/native_execution_strategy'; package.mkdir(parents=True)
            entries = {}
            for index in range(23):
                path = package / ('item' + str(index) + '.py'); path.write_text('# fixture\n')
                entries[path.relative_to(task / 'runtime').as_posix()] = {
                    'sha256': ne.sha(path), 'bytes': path.stat().st_size}
            manifest = {'schema': 'nes.runtime_snapshot.v1', 'files': entries,
                        'native_components_sha256': {'fixture': 'a' * 64}}
            mp = task / 'runtime_manifest.json'; mp.write_text(json.dumps(manifest))
            prior, pinned = controller.ROOT, controller.EXPECTED_RUNTIME_MANIFEST_SHA256
            controller.ROOT, controller.EXPECTED_RUNTIME_MANIFEST_SHA256 = task, ne.sha(mp)
            try:
                found = controller.source_manifest()
                assert all(name in found for name in metadata) and len(entries) == 23
                (package / 'extra.py').write_text('# extra')
                reject(controller.source_manifest); (package / 'extra.py').unlink()
                p = package / 'item0.py'; original = p.read_bytes(); p.write_text('# modified')
                reject(controller.source_manifest); p.write_bytes(original)
                p = task / 'inputs/manifest.json'; p.unlink()
                reject(controller.source_manifest); p.write_text('{}\n')
                controller.EXPECTED_RUNTIME_MANIFEST_SHA256 = '0' * 64
                reject(controller.source_manifest)
            finally:
                controller.ROOT, controller.EXPECTED_RUNTIME_MANIFEST_SHA256 = prior, pinned
        check('exact23_runtime_and_all_model_input_source_metadata_frozen', source_inventory)

        def case_validation():
            task = temp / 'cases'; (task / 'setup').mkdir(parents=True)
            p = task / 'setup/case.json'
            payload = {'schema': 'nes.model_coverage.case.v1', 'case_id': 'fixture',
                       'phase': 'model_coverage', 'operator_variant': 'private_vendor'}
            p.write_text(json.dumps(payload))
            prior = controller.ROOT; controller.ROOT = task
            try:
                assert controller.load_cases(['setup/case.json'])[0]['case_id'] == 'fixture'
                reject(lambda: controller.load_cases(['setup/case.json', 'setup/case.json']))
                reject(lambda: controller.load_cases(['../case.json']))
                p.write_text(json.dumps(dict(payload, schema='old.replay')))
                reject(lambda: controller.load_cases(['setup/case.json']))
            finally: controller.ROOT = prior
        check('only_unique_task_local_full_model_cases_accepted', case_validation)

        def vendor_boundary():
            bad = {'operator_variant': 'private_vendor', 'schema': 'nes.model_coverage.case.v1',
                   'approved_vendor_manifest_sha256': '0' * 64}
            reject(lambda: vendor_exec.validate_vendor(bad))
            for path in ('../x', '/x', 'a/../x', 'a\\x', 'a//x'):
                reject(lambda path=path: vendor_exec.relative_path(path))
            env, receipt = vendor_exec.loader_environment({'LD_LIBRARY_PATH': ':one::two:'},
                {'vendor_root': '/audit/vendor', 'op_api_library_directory': '/audit/vendor/op_api/lib'})
            assert env['LD_LIBRARY_PATH'] == '/audit/vendor/op_api/lib:one:two'
            assert receipt['removed_empty_count'] == 3
            assert runtime_binding.EXPECTED_RUNTIME_MANIFEST_SHA256 == controller.EXPECTED_RUNTIME_MANIFEST_SHA256
        check('vendor_pin_path_traversal_and_empty_loader_components_rejected', vendor_boundary)

        call = {'call_id': 'fixture_call', 'case_id': 'fixture', 'api': 'torch.nn.functional.scaled_dot_product_attention',
                'pid': 123, 'thread_id': 456, 'scope_token': 'fixture_token', 'phase': 'model_coverage',
                'batch_size': 1, 'step': 0}
        prefix = 'OP(123,fixture) '
        op = 'aclnnFlashAttentionScore_1_FlashAttentionScore'
        def log_lines(parts):
            return [json.dumps({'event': 'native_api_enter', **call}) + '\n',
                prefix + '[NNOP][Launch][456] OpName:[' + op + '] Tiling Key: 123, len: 16, numBlocks: 20, Workspace Num: 1.\n',
                *[prefix + '[SplitDataAndPrint][456] OpName:[' + op + '] PrintTilingData[' + str(i) + ']: ' +
                  ('Aligned TilingData Size: 16, data(uint32_t): ' if i == 0 else '') + part + '\n'
                  for i, part in enumerate(parts)],
                prefix + '[CommonOpExecutorRun][456] OpName:[aclnnFlashAttentionScore_1] Workspace addr: 0x1, size: 4096,\n',
                json.dumps({'event': 'native_api_exit', **call, 'status': 'returned'}) + '\n']
        def every_boundary():
            payload = '1 1024 32 4294967295 '
            for cut in range(1, len(payload)):
                p = temp / ('cut' + str(cut) + '.log')
                p.write_text(''.join(log_lines([payload[:cut], payload[cut:]])))
                record = ne.actual_launch(p, call, temp / ('cut' + str(cut) + '.window'))
                assert record['raw_hex'] == '010000000004000020000000ffffffff'
                assert record['raw_text_sha256'] == hashlib.sha256(payload.encode()).hexdigest()
                assert record['framing_verified_against_public_context'] is False
        check('native_uint32_character_concat_all_split_boundaries_preserved', every_boundary)
        def invalid_native():
            variants = [log_lines(['1 102x', '4 32 4294967295 ']), log_lines(['1 1024 32 ']),
                        log_lines(['1 1024 32 4294967296 ']), log_lines(['1 1024 32 9 '])[:-1]]
            lines = log_lines(['1 102', '4 32 9 '])
            variants.append([line.replace('PrintTilingData[1]', 'PrintTilingData[2]') for line in lines])
            variants.append([line.replace('OP(123,', 'OP(124,') for line in lines])
            variants.append([line.replace('data(uint32_t): ', 'data(uint32_t):') for line in lines])
            for i, lines in enumerate(variants):
                p = temp / ('bad' + str(i) + '.log'); p.write_text(''.join(lines))
                output = temp / ('bad' + str(i) + '.window')
                reject(lambda: ne.actual_launch(p, call, output))
                assert not output.exists()
        check('native_missing_corrupt_wrong_pid_or_malformed_chunks_rejected_without_output', invalid_native)
    report = {'schema': 'nes.model_coverage.cpu_resource_review.v1', 'status': 'PASS_CPU_PREPARATION_CHECKS',
              'checks': checks, 'npu_calls': 0, 'containers_started': 0,
              'source_sha256': {p.name: ne.sha(p) for p in sorted(ROOT.glob('*.py'))},
              'not_verified': ['Model execution', 'Native eligibility of cached decode',
                  'Actual native raw launch availability', 'Full-model coverage or numerical acceptance']}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(report, stream, indent=2); stream.write('\n')
    print(json.dumps({'status': report['status'], 'checks': len(checks), 'sha256': ne.sha(args.output)}))


if __name__ == '__main__':
    main()
