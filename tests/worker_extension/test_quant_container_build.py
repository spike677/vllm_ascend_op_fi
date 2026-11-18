"""No Docker is run by these command/receipt contracts."""
import ctypes
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from experiments.worker_core_v11_20261118 import build_quant_container_trace as build


class ContainerBuildTests(unittest.TestCase):
    def driver_fixture(self):
        return dict(schema='demoncat.quant.container_driver_dependencies.v1', root=str(build.DRIVER),
                    files={}, device_mounts=[], explicit_driver_api_calls=0)

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
        self.assertEqual(set(build.DRIVER_PINS), {'driver/libascend_hal.so', 'common/libc_sec.so', 'common/libmmpa.so'})
        self.assertEqual(build.DRIVER_PINS['common/libmmpa.so'],
                         '2094e03c1ad683c43fbfbe7e93f705d119bc605aaf96db78f0adbc7c69e1f6f7')

    def test_shared_header_warning_scope_ends_before_own_code(self):
        source = (build.ROOT / 'fault_injection/native_adapters/quant_container_trace.cpp').read_text()
        start = source.index('#pragma GCC diagnostic push')
        end = source.index('#pragma GCC diagnostic pop', start)
        self.assertEqual(source[start:end].count('#include'), 1)
        self.assertIn('"../../fault39/native/runtime_identity.h"', source[start:end])
        self.assertEqual(source.count('#pragma GCC diagnostic ignored'), 1)
        self.assertIn('"-Wmisleading-indentation"', source[start:end])
        self.assertLess(end, source.index('#include "../../fault39/native/metadata_array_owner.h"'))
        self.assertIn('dcat_metadata::capture_exception();\n    throw;', source)
        self.assertFalse(any(flag.startswith('-Wno') for flag in build.flags(Path('/build'))))

    def test_original_elf_is_not_rewritten_or_masked(self):
        text = Path(build.__file__).read_text()
        self.assertNotIn('--set-soname', text)
        self.assertIn("'--add-needed', str(paths['custom'].resolve()), library", text)
        self.assertIn('ORIGINAL_IMAGE_BYTES_CHANGED_DURING_BUILD', text)
        self.assertIn('PASS_REAL_ELF_ABSOLUTE_NEEDED_SAME_SONAME', text)

    def run_failure(self, exception=None, cleanup=True):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory) / 'fresh'
            argv = ['builder', '--source', str(build.ROOT), '--out', str(out), '--source-commit', 'a' * 40]
            result = subprocess.CompletedProcess([], 7)
            with patch.object(sys, 'argv', argv), patch.object(build, 'inspect_image', return_value={'Id': build.IMAGE}), \
                    patch.object(build, 'driver_identity', return_value=self.driver_fixture()), \
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
            self.assertIn(str(build.DRIVER) + ':' + str(build.DRIVER) + ':ro', command)
            self.assertIn('LD_LIBRARY_PATH=' + str(build.DRIVER / 'driver') + ':' + str(build.DRIVER / 'common'), command)
            self.assertEqual(receipt['driver_identity_sha256'], build.sha(out / 'driver-identity.json'))
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

    def test_loader_path_uses_actual_image_and_readonly_driver_not_host_env(self):
        image = dict(Config=dict(Env=['PATH=/bin', 'LD_LIBRARY_PATH=/sdk/lib:/runtime/lib']))
        with patch.dict(os.environ, {'LD_LIBRARY_PATH': '/unapproved-host'}):
            value = build.loader_path(image)
        self.assertEqual(value, str(build.DRIVER / 'driver') + ':' + str(build.DRIVER / 'common')
                         + ':/sdk/lib:/runtime/lib')
        for invalid in ('relative', '$LD_LIBRARY_PATH'):
            with self.assertRaisesRegex(RuntimeError, 'EXPLICIT_ABSOLUTE_LOADER_PATH_REQUIRED'):
                build.loader_path(dict(Config=dict(Env=['LD_LIBRARY_PATH=' + invalid])))
        with self.assertRaisesRegex(RuntimeError, 'AMBIGUOUS_IMAGE_LOADER_PATH'):
            build.loader_path(dict(Config=dict(Env=['LD_LIBRARY_PATH=/a', 'LD_LIBRARY_PATH=/b'])))

    def test_driver_file_bytes_are_pinned_not_stubbed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); pins = {}
            for name, data in (('driver/libascend_hal.so', b'H' * 1861080), ('common/libc_sec.so', b'SEC')):
                path = root / name; path.parent.mkdir(); path.write_bytes(data)
                pins[name] = hashlib.sha256(data).hexdigest()
            with patch.object(build, 'DRIVER_PINS', pins):
                value = build.driver_identity(root)
                self.assertEqual(value['files']['driver/libascend_hal.so']['bytes'], 1861080)
                (root / 'common/libc_sec.so').write_bytes(b'changed')
                with self.assertRaisesRegex(RuntimeError, 'ACTUAL_DRIVER_DEPENDENCY_CHANGED'):
                    build.driver_identity(root)

    def test_loaded_driver_maps_require_three_exact_files_and_refuse_fourth(self):
        expected = dict(root='/driver', files={
            'hal': dict(path='/driver/driver/libascend_hal.so', inode=11, device=5, sha256='a' * 64),
            'sec': dict(path='/driver/common/libc_sec.so', inode=12, device=5, sha256='b' * 64),
            'mmpa': dict(path='/driver/common/libmmpa.so', inode=13, device=5, sha256='c' * 64)})
        maps = ('100-200 r-xp 0 00:05 11 /driver/driver/libascend_hal.so\n'
                '200-300 r-xp 0 00:05 12 /driver/common/libc_sec.so\n'
                '300-400 r-xp 0 00:05 13 /driver/common/libmmpa.so\n')
        identities = {row['path']: row['sha256'] for row in expected['files'].values()}
        with patch.object(build.os, 'major', return_value=0, create=True), \
                patch.object(build.os, 'minor', return_value=5, create=True), \
                patch.object(build, 'sha', side_effect=lambda path: identities[str(path)]):
            self.assertEqual(len(build.loaded_driver_mappings(expected, maps)), 3)
            for changed in (maps.replace('00:05', '00:06'), maps.replace(' 11 ', ' 99 '),
                            maps.replace('libc_sec.so', 'foreign.so'), maps.splitlines()[0],
                            maps.replace('libc_sec.so', 'libc_sec.so (deleted)'),
                            maps + '400-500 r-xp 0 00:05 14 /driver/common/unreviewed.so\n'):
                with self.subTest(maps=changed), self.assertRaises(RuntimeError):
                    build.loaded_driver_mappings(expected, changed)
            identities['/driver/common/libmmpa.so'] = 'd' * 64
            with self.assertRaisesRegex(RuntimeError, 'LOADED_DRIVER_MAPPING_IDENTITY_CHANGED'):
                build.loaded_driver_mappings(expected, maps)

    def contract_fixture(self, root):
        source = root / 'source'; out = root / 'out'; source.mkdir(); out.mkdir()
        (source / 'source.cpp').write_text('actual fixture source')
        (out / 'cpu_contract').write_bytes(b'CPU_EXECUTABLE_FIXTURE')
        return source, out

    def test_all_19_receipts_persist_before_marker_failure(self):
        def execute(command, **kwargs):
            kwargs['stdout'].write(('PASS_CPU_SDK_TYPED_FUNCTION_DOUBLE_NOT_NPU ' + command[-1] + '\n').encode())
            return subprocess.CompletedProcess(command, 0)
        with tempfile.TemporaryDirectory() as directory:
            source, out = self.contract_fixture(Path(directory))
            with patch.object(build, 'SOURCES', ('source.cpp',)), patch.object(build, 'run', side_effect=execute):
                build.cpu_contracts(source, out)
            before = (out / 'cpu-contract.json').read_bytes(); value = json.loads(before)
            self.assertEqual(len(value['cases']), 19)
            self.assertTrue(all(row['status'] == 'PASS' and row['returncode'] == 0 for row in value['cases']))
            self.assertEqual(value['combined_log_sha256'], build.sha(out / 'cpu-contract.log'))
            build.write_json(out / 'driver-identity.json', self.driver_fixture())
            with patch.object(build, 'driver_identity', return_value=self.driver_fixture()), \
                    patch.object(build.os, 'RTLD_LOCAL', 0, create=True), patch.object(build.os, 'RTLD_NOW', 2, create=True), \
                    patch.object(ctypes, 'CDLL', side_effect=OSError('missing actual dependency')):
                with self.assertRaisesRegex(OSError, 'missing actual dependency'):
                    build.marker_load(out, out / 'real.so', 'a' * 64)
            self.assertEqual((out / 'cpu-contract.json').read_bytes(), before)
            marker = json.loads((out / 'sdk-load.json').read_text())
            self.assertEqual(marker['status'], 'FAILED_SDK_MARKER_LOAD')
            self.assertEqual(marker['explicit_driver_api_calls'], 0)
            self.assertEqual(marker['loader_constructor_driver_calls'], 'NOT_COUNTED_MAY_PROBE_DEVICE_INFORMATION')
            self.assertFalse((out / 'build.json').exists())

    def test_mid_contract_failure_keeps_prior_results_and_first_failed_case(self):
        def execute(command, **kwargs):
            if command[-1] == build.CASES[2]:
                kwargs['stdout'].write(b'actual failure\n')
                raise subprocess.CalledProcessError(9, command)
            kwargs['stdout'].write(('PASS_CPU_SDK_TYPED_FUNCTION_DOUBLE_NOT_NPU ' + command[-1] + '\n').encode())
            return subprocess.CompletedProcess(command, 0)
        with tempfile.TemporaryDirectory() as directory:
            source, out = self.contract_fixture(Path(directory))
            with patch.object(build, 'SOURCES', ('source.cpp',)), patch.object(build, 'run', side_effect=execute):
                with self.assertRaises(subprocess.CalledProcessError):
                    build.cpu_contracts(source, out)
            value = json.loads((out / 'cpu-contract.json').read_text())
            self.assertEqual(value['status'], 'INCOMPLETE')
            self.assertEqual([row['status'] for row in value['cases']], ['PASS', 'PASS', 'FAILED'])
            self.assertEqual(value['cases'][-1]['returncode'], 9)
            self.assertEqual(value['cases'][-1]['log_sha256'], build.sha(out / value['cases'][-1]['log']))


if __name__ == '__main__':
    unittest.main()
