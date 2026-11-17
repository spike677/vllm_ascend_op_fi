"""Explicit CPU bootstrap mocks; real ELF contracts run separately on Linux."""
import copy
import ctypes
import hashlib
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

from fault_injection.deployments import writer26_loader as loader
from fault_injection.metadata_arrays import (COUNTERS, LoadedWriter26MetadataArrays,
    MetadataArrayStateV1, SCHEMA, WRITER26_SYMBOLS)


class Writer26LoaderTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(self.enterContext(tempfile.TemporaryDirectory())).resolve()
        for name in WRITER26_SYMBOLS:
            (self.root / name).write_bytes(('CPU_ONLY_NOT_ELF_' + name).encode())
        proof = self.root / 'metadata-array-ownership.json'
        proof.write_text('CPU_ONLY_NOT_NATIVE_PROOF')
        files = {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in self.root.iterdir()}
        self.build = dict(schema='demoncat.writer26.group_deployment.v2', actual_npu_calls=0,
            legacy_writers_absent=True, files=files, metadata_array_ownership=dict(schema=SCHEMA,
                state_version=1, images=WRITER26_SYMBOLS, reset_supported=False,
                runtime_quiescent_readback_required=True, parent_receipt_sha256=files[proof.name]))
        manifest = self.root / 'build.json'
        manifest.write_text(json.dumps(self.build))
        self.manifest_sha = hashlib.sha256(manifest.read_bytes()).hexdigest()
        self.paths = {name: (str(self.root / name),) for name in WRITER26_SYMBOLS}
        self.before = {name: paths for name, paths in self.paths.items() if name != 'libopapi_nn.so'}
        self.enterContext(patch.object(loader, 'sys', NS(modules={})))
        for name, value in (('RTLD_NOLOAD', 4), ('RTLD_NOW', 2), ('RTLD_LOCAL', 0)):
            self.enterContext(patch.object(os, name, value, create=True))
        self.maps = self.enterContext(patch.object(loader, '_mapped_proxies', side_effect=lambda *a: copy.deepcopy(self.before)))
        self.dlopen = self.enterContext(patch.object(loader.ctypes, 'CDLL', return_value=object()))
        self.reader = NS(paths=self.paths, read=self.read)
        self.attach = self.enterContext(patch.object(loader, 'LoadedWriter26MetadataArrays', return_value=self.reader))

    def read(self):
        rows = []
        for index, name in enumerate(WRITER26_SYMBOLS):
            state = dict.fromkeys(COUNTERS, 0)
            state.update(version=1, size=ctypes.sizeof(MetadataArrayStateV1), instance=index + 1)
            rows.append(dict(name=name, path=str(self.root / name), sha256=self.build['files'][name],
                             symbol=WRITER26_SYMBOLS[name], state=state))
        return dict(schema=SCHEMA, pid=os.getpid(), build_profile_sha256=self.manifest_sha,
            build_manifest_sha256=self.manifest_sha, images=rows, scope='SDK_GETTER_ARRAYS_ONLY', reset_supported=False)

    def initialize(self):
        return loader.StartupDependencies(bundle=self.root, manifest_sha256=self.manifest_sha)

    def test_only_missing_split_is_loaded_locally_and_four_images_remain_required(self):
        result = self.initialize()
        self.dlopen.assert_called_once_with(str(self.root / 'libopapi_nn.so'), mode=os.RTLD_NOW | os.RTLD_LOCAL)
        self.assertEqual(result.evidence['before']['missing_images'], ['libopapi_nn.so'])
        self.assertEqual(result.evidence['after']['missing_images'], [])
        self.assertEqual(len(result.evidence['metadata_arrays']['images']), 4)
        self.assertEqual(len(result.handles), 1)
        self.assertFalse(result.evidence['c2'] or result.evidence['c3'] or result.evidence['callback_installed'])

    def test_already_loaded_split_does_not_load_again(self):
        self.before = self.paths
        self.assertEqual(self.initialize().evidence['images_loaded'], [])
        self.dlopen.assert_not_called()

    def test_other_missing_image_refuses_without_loading_anything(self):
        self.before.pop('libopapi.so')
        with self.assertRaisesRegex(loader.StartupError, 'EXPECTED_CORE_TRACE_MAIN') as error:
            self.initialize()
        self.assertEqual(error.exception.evidence['before']['missing_images'], ['libopapi.so', 'libopapi_nn.so'])
        self.dlopen.assert_not_called()

    def test_actual_dependency_bytes_checked_before_any_load(self):
        (self.root / 'libopapi_nn.so').write_bytes(b'changed dependency')
        with self.assertRaisesRegex(loader.StartupError, 'DEPENDENCY_BYTES_CHANGED:libopapi_nn.so'):
            self.initialize()
        self.dlopen.assert_not_called()

    def test_foreign_existing_image_and_late_initialization_refuse(self):
        foreign = self.root / 'foreign.so'
        foreign.write_bytes((self.root / 'libopapi.so').read_bytes())
        self.before['libopapi.so'] = (str(foreign),)
        with self.assertRaisesRegex(loader.StartupError, 'FOREIGN_PREEXISTING_IMAGE'):
            self.initialize()
        loader.sys.modules['torch'] = object()
        with self.assertRaisesRegex(loader.StartupError, 'BEFORE_MODEL_IMPORT_REQUIRED'):
            self.initialize()
        self.dlopen.assert_not_called()

    def test_observer_reports_missing_name_and_never_opens_missing_dependency(self):
        with patch('fault_injection.metadata_arrays._mapped_proxies', return_value=self.before):
            with self.assertRaisesRegex(RuntimeError, 'INCOMPLETE_WRITER_PROXY_MAPPING:') as error:
                LoadedWriter26MetadataArrays(bundle=self.root, manifest_sha256=self.manifest_sha)
        diagnostic = json.loads(str(error.exception).split('INCOMPLETE_WRITER_PROXY_MAPPING:', 1)[1])
        self.assertEqual(diagnostic['missing_images'], ['libopapi_nn.so'])
        self.assertEqual(len(diagnostic['mapped_paths']), 3)
        self.dlopen.assert_not_called()

    def test_bootstrap_cannot_override_a_failed_four_counter_check(self):
        self.attach.side_effect = RuntimeError('METADATA_ARRAYS_COUNTER_SYMBOL_WRONG_IMAGE')
        with self.assertRaisesRegex(loader.StartupError, 'COUNTER_SYMBOL_WRONG_IMAGE') as error:
            self.initialize()
        self.assertEqual(error.exception.evidence['status'], 'FAILED_STARTUP_DEPENDENCY_CHECK')
        self.assertFalse(error.exception.evidence['mechanism_qualification'])


if __name__ == '__main__':
    unittest.main()
