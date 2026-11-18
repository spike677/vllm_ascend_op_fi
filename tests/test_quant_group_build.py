"""CPU orchestration contracts. Every subprocess call is mocked; no Docker."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch


SCRIPT = Path(__file__).resolve().parents[1] / 'experiments/c2c3_closure_20261117/providers_quant_build_group.py'
spec = importlib.util.spec_from_file_location('quant_group_build_test_module', SCRIPT)
builder = importlib.util.module_from_spec(spec)
spec.loader.exec_module(builder)
IMAGE_ID = 'sha256:' + 'b' * 64


class QuantGroupBuildTests(unittest.TestCase):
    def test_real_loader_contract_requires_every_case_and_retains_mock_boundary(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'loader-contract.jsonl'
            rows = [dict(case=name, status='PASS_REAL_ELF_LOADER_CPU_MOCK', actual_npu_calls=0,
                mocked_sdk_functions=True, native_prepare_calls=int(name in ('no_cache', 'with_cache', 'foreign_cache')),
                native_execute_calls=int(name in ('no_cache', 'with_cache', 'foreign_cache')),
                foreign_cache_calls=0) for name in builder.LOADER_CASES]
            path.write_text('\n'.join(json.dumps(row) for row in rows))
            self.assertEqual(builder.loader_contract(path)['cases'], rows)
            path.write_text('\n'.join(json.dumps(row) for row in rows[:-1]))
            with self.assertRaisesRegex(RuntimeError, 'COMPLETE_REAL_ELF'):
                builder.loader_contract(path)
            rows[-1]['foreign_cache_calls'] = 1
            path.write_text('\n'.join(json.dumps(row) for row in rows))
            with self.assertRaisesRegex(RuntimeError, 'REAL_ELF_LOADER_CPU_CONTRACT_FAILED'):
                builder.loader_contract(path)

    def test_loader_driver_loads_production_dso_and_separate_host_only_standins(self):
        self.assertIn('test_batch_scale_group_loader.cpp', builder.COMMANDS)
        self.assertIn('-I/source/tests/native_quant_group_mock_sdk', builder.COMMANDS)
        self.assertIn('/build/test_group_loader /build/runtime/vendor/op_api/lib/libcust_opapi.so', builder.COMMANDS)
        for name in builder.LOADER_CASES:
            self.assertIn(name, builder.COMMANDS)
        self.assertNotIn('-Bsymbolic', builder.COMMANDS)

    def test_vendor_headers_are_system_headers_and_our_warnings_stay_strict(self):
        flags = next(line for line in builder.COMMANDS.splitlines() if line.startswith('flags=('))
        for option in ('-Wall', '-Wextra', '-Werror', '-pedantic'):
            self.assertIn(option, flags)
        for path in ('include', 'include/aclnn', 'pkg_inc'):
            self.assertIn('-isystem "$sdk/' + path + '"', flags)
        self.assertNotIn('-I"$sdk/', flags)
        self.assertNotIn('-Wno-', flags)

    def test_tag_is_resolved_and_matched_to_approved_immutable_id(self):
        value = dict(Id=IMAGE_ID, Architecture='arm64', Os='linux')
        with patch.object(builder.subprocess, 'run', return_value=SimpleNamespace(stdout=json.dumps([value]))) as run:
            self.assertEqual(builder.inspect_image(IMAGE_ID), value)
            self.assertEqual(run.call_args.args[0][:3], ['docker', 'image', 'inspect'])
            with self.assertRaisesRegex(RuntimeError, 'ACTUAL_IMAGE'):
                builder.inspect_image('sha256:' + 'c' * 64)

    def test_wrong_platform_refused(self):
        value = dict(Id=IMAGE_ID, Architecture='amd64', Os='linux')
        with patch.object(builder.subprocess, 'run', return_value=SimpleNamespace(stdout=json.dumps([value]))):
            with self.assertRaisesRegex(RuntimeError, 'PLATFORM'):
                builder.inspect_image(IMAGE_ID)

    def test_foreign_same_name_container_is_never_removed(self):
        value = dict(Name='/build-one', Image=IMAGE_ID, Config=dict(Labels={'demoncat.quant.group-build-owner': 'foreign'}))
        with patch.object(builder.subprocess, 'run', return_value=SimpleNamespace(returncode=0, stdout=json.dumps([value]))) as run:
            result = builder.cleanup_owned('build-one', 'mine', IMAGE_ID)
            self.assertFalse(result['confirmed'])
            self.assertEqual(run.call_count, 1)

    def test_only_exact_owned_container_can_be_removed(self):
        value = dict(Name='/build-one', Image=IMAGE_ID, Config=dict(Labels={'demoncat.quant.group-build-owner': 'mine'}))
        replies = [SimpleNamespace(returncode=0, stdout=json.dumps([value])), SimpleNamespace(returncode=0, stdout='', stderr='')]
        with patch.object(builder.subprocess, 'run', side_effect=replies) as run:
            self.assertTrue(builder.cleanup_owned('build-one', 'mine', IMAGE_ID)['confirmed'])
            self.assertEqual(run.call_args.args[0], ['docker', 'container', 'rm', '-f', 'build-one'])

    def test_inspection_error_is_not_absence(self):
        with patch.object(builder.subprocess, 'run', return_value=SimpleNamespace(returncode=1, stderr='daemon unavailable')):
            self.assertFalse(builder.cleanup_owned('build-one', 'mine', IMAGE_ID)['confirmed'])

    def test_timeout_keeps_receipt_and_cleans_only_owned_attempt(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source, out = root / 'source', root / 'build'
            for relative in builder.SOURCES:
                path = source / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text('// CPU builder fixture\n')
            argv = ['build', '--source', str(source), '--out', str(out), '--source-commit', 'a' * 40, '--image-id', IMAGE_ID]
            with patch.object(sys, 'argv', argv), patch.object(builder.platform, 'machine', return_value='aarch64'), \
                    patch.object(builder, 'inspect_image', return_value=dict(Id=IMAGE_ID)), \
                    patch.object(builder.subprocess, 'run', side_effect=subprocess.TimeoutExpired('docker', 300)) as run, \
                    patch.object(builder, 'cleanup_owned', return_value=dict(confirmed=True, status='OWNED_REMOVAL')) as cleanup:
                with self.assertRaisesRegex(RuntimeError, 'RECEIPT_PRESERVED'):
                    builder.main()
            receipt = json.loads((out / 'build-receipt.json').read_bytes())
            command = run.call_args.args[0]
            self.assertTrue(receipt['timed_out'])
            self.assertEqual(receipt['actual_image_id'], IMAGE_ID)
            self.assertEqual(receipt['actual_npu_calls'], 0)
            self.assertEqual(receipt['device_mounts'], [])
            self.assertIn('ASCEND_VISIBLE_DEVICES=void', command)
            self.assertEqual(command[command.index('--runtime') + 1], 'runc')
            self.assertEqual(command[command.index('--network') + 1], 'none')
            self.assertIn(str(source.resolve()) + ':/source:ro', command)
            self.assertIn(IMAGE_ID, command)
            self.assertNotIn(builder.IMAGE, command)
            cleanup.assert_called_once_with(receipt['container_name'], receipt['container_owner_nonce'], IMAGE_ID)
            self.assertFalse((out / 'build.json').exists())


if __name__ == '__main__':
    unittest.main()
