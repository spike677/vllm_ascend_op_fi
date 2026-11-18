"""Command and receipt contracts only; these tests never launch Docker."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

from experiments.worker_core_v11_20261118 import build_quant_gmm2_trace as build


class Gmm2BuildTests(unittest.TestCase):
    def test_exact_pair_and_independent_full_source_fingerprint(self):
        self.assertEqual(build.PAIR, ('aclnnGroupedMatmulWeightNzGetWorkspaceSize', 'aclnnGroupedMatmulWeightNz'))
        self.assertEqual(len(build.CASES), 28)
        self.assertEqual(len(set(build.CASES)), 28)
        self.assertEqual(len(set(build.SOURCES)), len(build.SOURCES))
        for name in ('quant_gmm2_trace.h', 'quant_gmm2_trace.inc', 'test_quant_gmm2_trace.cpp'):
            self.assertIn('fault_injection/native_adapters/' + name, build.SOURCES)
        for name in build.SOURCES:
            self.assertTrue((build.ROOT / name).is_file(), name)
        self.assertEqual(build.PINS['device_runtime'],
                         'bdfa66949fc798ff21c0124a025b4227e119678cc819febed9fe1c39f2a87306')
        self.assertEqual(len(build.HEADER_SHA), 64)
        self.assertEqual(len(build.RUNTIME_HEADER_SHA), 64)

    def test_failed_build_keeps_owned_timeout_and_new_stage_receipts(self):
        for failure in (None, subprocess.TimeoutExpired('owned', 300)):
            with self.subTest(timeout=failure is not None), tempfile.TemporaryDirectory() as directory:
                out = Path(directory) / 'fresh'
                argv = ['build', '--source', str(build.ROOT), '--out', str(out), '--source-commit', 'a' * 40]
                with patch.object(sys, 'argv', argv), \
                        patch.object(build.base, 'inspect_image', return_value={'Id': build.IMAGE}), \
                        patch.object(build.base, 'driver_identity', return_value={'files': {}}), \
                        patch.object(build.subprocess, 'run', side_effect=failure,
                                     return_value=subprocess.CompletedProcess([], 7)) as execute, \
                        patch.object(build.base, 'cleanup_owned', return_value={'confirmed': True}) as cleanup:
                    with self.assertRaisesRegex(RuntimeError, 'CPU_BUILD_FAILED_RECEIPT_PRESERVED'):
                        build.main()
                receipt = json.loads((out / 'build-receipt.json').read_text())
                command = execute.call_args.args[0]
                self.assertIn(build.IMAGE, command)
                self.assertEqual(command[command.index('--runtime') + 1], 'runc')
                self.assertEqual(command[command.index('--network') + 1], 'none')
                self.assertIn('ASCEND_VISIBLE_DEVICES=void', command)
                self.assertFalse(any('/dev/' in item for item in command))
                self.assertIn('/source/experiments/worker_core_v11_20261118/build_quant_gmm2_trace.py', command)
                self.assertEqual(receipt['timed_out'], failure is not None)
                self.assertEqual(receipt['sources'].keys(), dict.fromkeys(build.SOURCES).keys())
                self.assertEqual(cleanup.call_args.args,
                                 (receipt['container_name'], receipt['container_owner_nonce'], build.IMAGE))
                self.assertFalse((out / 'build.json').exists())

    def fixture(self, root):
        source, out = root / 'source', root / 'out'
        source.mkdir(); out.mkdir()
        (source / 'fixture.cpp').write_text('CPU fixture')
        (out / 'gmm2_cpu_contract').write_bytes(b'CPU_EXECUTABLE_FIXTURE')
        return source, out

    def test_each_successful_contract_has_an_independent_log_hash(self):
        def execute(command, **kwargs):
            kwargs['stdout'].write(('PASS_CPU_SDK_GMM2_TYPED_FUNCTION_DOUBLE_NOT_NPU ' + command[-1] + '\n').encode())
            return subprocess.CompletedProcess(command, 0)
        with tempfile.TemporaryDirectory() as directory:
            source, out = self.fixture(Path(directory))
            with patch.object(build, 'SOURCES', ('fixture.cpp',)), patch.object(build.base, 'run', side_effect=execute):
                build.cpu_pair_contracts(source, out)
            value = json.loads((out / 'gmm2-cpu-contract.json').read_text())
            self.assertEqual(value['status'], 'PASS_ALL_GMM2_SDK_TYPED_CPU_CONTRACTS_NOT_NPU')
            self.assertEqual(len(value['cases']), 28)
            for row in value['cases']:
                self.assertEqual(row['status'], 'PASS')
                self.assertEqual(row['log_sha256'], build.base.sha(out / row['log']))
            self.assertEqual(value['combined_log_sha256'], build.base.sha(out / 'gmm2-cpu-contract.log'))

    def test_failed_contract_never_overwrites_earlier_stage_or_claims_pass(self):
        def execute(command, **kwargs):
            if command[-1] == build.CASES[1]:
                kwargs['stdout'].write(b'CPU negative failed\n')
                raise subprocess.CalledProcessError(8, command)
            kwargs['stdout'].write(('PASS_CPU_SDK_GMM2_TYPED_FUNCTION_DOUBLE_NOT_NPU ' + command[-1] + '\n').encode())
            return subprocess.CompletedProcess(command, 0)
        with tempfile.TemporaryDirectory() as directory:
            source, out = self.fixture(Path(directory))
            (out / 'cpu-contract.json').write_text('{"original_19":"preserved"}')
            old = (out / 'cpu-contract.json').read_bytes()
            with patch.object(build, 'SOURCES', ('fixture.cpp',)), patch.object(build.base, 'run', side_effect=execute):
                with self.assertRaises(subprocess.CalledProcessError):
                    build.cpu_pair_contracts(source, out)
            value = json.loads((out / 'gmm2-cpu-contract.json').read_text())
            self.assertEqual(value['status'], 'INCOMPLETE')
            self.assertEqual([r['status'] for r in value['cases']], ['PASS', 'FAILED'])
            self.assertEqual(value['cases'][-1]['returncode'], 8)
            self.assertEqual((out / 'cpu-contract.json').read_bytes(), old)

    def test_marker_resolves_only_and_persists_failure(self):
        for result in (0, 1):
            with self.subTest(result=result), tempfile.TemporaryDirectory() as directory:
                out = Path(directory); library = out / 'libcust_opapi.so'
                loaded = Mock(); loaded.dcat_qct_gmm2_marker_v1.return_value = 1
                loaded.dcat_qct_gmm2_resolve_v1.return_value = result
                with patch.object(build.base, 'marker_load') as base_load, \
                        patch.object(build.os, 'RTLD_NOW', 2, create=True), \
                        patch.object(build.os, 'RTLD_LOCAL', 0, create=True), \
                        patch.object(build.os, 'RTLD_NOLOAD', 4, create=True), \
                        patch.object(build.C, 'CDLL', return_value=loaded) as attach:
                    if result:
                        with self.assertRaisesRegex(RuntimeError, 'EXACT_PRELOADED_GMM2_ORIGINALS_REQUIRED'):
                            build.marker_preflight(out, library, 'a' * 64)
                    else:
                        build.marker_preflight(out, library, 'a' * 64)
                base_load.assert_called_once_with(out, library, 'a' * 64)
                self.assertTrue(attach.call_args.kwargs['mode'] & 4)
                self.assertEqual([call[0] for call in loaded.method_calls],
                                 ['dcat_qct_gmm2_marker_v1', 'dcat_qct_gmm2_resolve_v1'])
                receipt = json.loads((out / 'gmm2-startup.json').read_text())
                self.assertEqual(receipt['operator_calls'], 0)
                self.assertEqual(receipt['device_query_calls'], 0)
                self.assertEqual(receipt['install_calls'], 0)
                self.assertEqual(receipt['status'].startswith('FAILED'), bool(result))


if __name__ == '__main__':
    unittest.main()
