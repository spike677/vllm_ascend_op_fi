"""No Docker is run by these command/receipt contracts."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from experiments.worker_core_v11_20261118 import build_quant_container_trace as build


class ContainerBuildTests(unittest.TestCase):
    def test_sdk_includes_are_system_but_own_warnings_stay_errors(self):
        flags = build.flags(Path('/build'))
        self.assertIn('-Werror', flags); self.assertIn('-Wall', flags); self.assertIn('-pedantic', flags)
        self.assertEqual(flags.count('-isystem'), 2)
        self.assertNotIn('-Wno-pedantic', flags)

    def test_exact_four_functions_and_no_operator_framework(self):
        self.assertEqual(build.FUNCTIONS, ('aclCreateTensorList', 'aclDestroyTensorList',
                                          'aclCreateIntArray', 'aclDestroyIntArray'))
        self.assertEqual(len(build.CASES), 19)
        self.assertEqual(len(set(build.CASES)), 19)
        self.assertTrue(all(len(value) == 64 for value in build.PINS.values()))

    def run_failure(self, exception=None, cleanup=True):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory) / 'fresh'
            argv = ['builder', '--source', str(build.ROOT), '--out', str(out), '--source-commit', 'a' * 40]
            result = subprocess.CompletedProcess([], 7)
            with patch.object(sys, 'argv', argv), patch.object(build, 'inspect_image', return_value={'Id': build.IMAGE}), \
                    patch.object(build.subprocess, 'run', side_effect=exception, return_value=result) as execute, \
                    patch.object(build, 'cleanup_owned', return_value={'confirmed': cleanup}) as remove:
                with self.assertRaisesRegex(RuntimeError, 'CPU_BUILD_FAILED_RECEIPT_PRESERVED'):
                    build.main()
            receipt = json.loads((out / 'build-receipt.json').read_bytes())
            self.assertEqual(receipt['status'], 'FAILED_CPU_BUILD')
            self.assertEqual(receipt['actual_npu_calls'], 0)
            command = execute.call_args.args[0]
            self.assertIn(build.IMAGE, command)
            self.assertEqual(command[command.index('--runtime') + 1], 'runc')
            self.assertEqual(command[command.index('--network') + 1], 'none')
            self.assertIn('ASCEND_VISIBLE_DEVICES=void', command)
            self.assertFalse(any('/dev/' in part for part in command))
            self.assertEqual(remove.call_args.args,
                             (receipt['container_name'], receipt['container_owner_nonce'], build.IMAGE))
            self.assertFalse((out / 'build.json').exists())
            return receipt

    def test_failed_compile_retains_receipt_and_cleans_only_owned_name(self):
        self.assertEqual(self.run_failure()['returncode'], 7)

    def test_timeout_always_records_then_cleans(self):
        value = self.run_failure(subprocess.TimeoutExpired('owned-docker', 300))
        self.assertTrue(value['timed_out'])

    def test_failure_to_clean_cannot_publish_manifest(self):
        self.assertFalse(self.run_failure(cleanup=False)['cleanup']['confirmed'])

    def test_foreign_image_rejected_before_any_container(self):
        with tempfile.TemporaryDirectory() as directory:
            argv = ['builder', '--source', str(build.ROOT), '--out', directory,
                    '--source-commit', 'a' * 40, '--image-id', 'sha256:' + 'b' * 64]
            with patch.object(sys, 'argv', argv), patch.object(build.subprocess, 'run') as execute:
                with self.assertRaisesRegex(RuntimeError, 'FIXED_IMAGE_AND_ACTUAL_SOURCE_COMMIT_REQUIRED'):
                    build.main()
                execute.assert_not_called()


if __name__ == '__main__':
    unittest.main()
