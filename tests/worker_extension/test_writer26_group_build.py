"""Explicit CPU mocks for immutable image identity and owned build cleanup."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch
import tempfile

from fault_injection.native_build import metadata26

PATH = Path(__file__).resolve().parents[2] / 'experiments/c2c3_closure_20261117/providers26_build_group.py'
SPEC = importlib.util.spec_from_file_location('writer26_group_build', PATH)
build = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(build)


class GroupBuildTests(unittest.TestCase):
    def test_fresh_build_includes_actual_cache_getter_contract_sources(self):
        required = {'fault_injection/native_adapters/test_writer26_custom_bound_addresses.cpp',
            'fault20_fast_validation/source/fault26_readonly_trace_v1/fault26_opapi_trace_proxy.cpp',
            'fault20_fast_validation/source/fault26_readonly_trace_v1/fault26_trace_shared.h'}
        self.assertTrue(required.issubset(build.SOURCES))
        self.assertTrue(all((PATH.parents[2]/name).is_file() for name in required))

    def test_typed_fia_requires_every_new_trace_export_and_exact_sdk_layout(self):
        exports = '\n'.join('0000 T ' + name for name in build.TYPED_TRACE_EXPORTS)
        build.require_typed_trace(exports)
        for name in build.TYPED_TRACE_EXPORTS:
            with self.subTest(name=name), self.assertRaisesRegex(RuntimeError, 'FRESH_TYPED_FIA'):
                build.require_typed_trace(exports.replace(' T ' + name, ' U ' + name))
        abi = dict(version=3, inventory_version=2, descriptor=280, record=656, state=104, lease=912, snapshots=2048)
        control = dict(fia_typed_abi=abi)
        contract = dict(status='PASS_CPU_TYPED_FIA_RUNTIME', actual_npu_calls=0, cases=18)
        result = build.typed_profile(control, contract)
        self.assertEqual(result['native_format'], 47)
        self.assertFalse(result['old_reference_reusable'])
        self.assertTrue(result['stock_to_owned_reference_full_byte_comparison_required'])
        for name in abi:
            with self.subTest(name=name), self.assertRaisesRegex(RuntimeError, 'ACTUAL_SDK'):
                build.typed_profile(dict(fia_typed_abi={**abi, name: abi[name] + 1}), contract)
        with self.assertRaisesRegex(RuntimeError, 'ACTUAL_SDK'):
            build.typed_profile(control, {**contract, 'cases': 16})

    def test_old_parent_without_ownership_evidence_is_refused(self):
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaisesRegex(RuntimeError, 'OLD_PARENT_REBUILD_REQUIRED'):
                metadata26.verify_parent(PATH.parents[2], Path(folder), {})

    def test_parent_metadata_requires_exact_source_images_and_balanced_cpu_allocations(self):
        source = PATH.parents[2]
        unit, sources = metadata26.source_contract(source)
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            cpu = dict(schema='demoncat.metadata_arrays.cpu_contract.v1', status='CPU_CONTRACT_PASS',
                includes_writer26=True, actual_npu_calls=0, actual_sdk_or_numerical_acceptance=False,
                compiler_returncode=0, returncode=0,
                generated_unit_sha256=unit, production_sources={k: v for k, v in sources.items()
                    if not k.endswith('/test_metadata_ownership.py')},
                counts=dict(actual_new_arrays=3, actual_delete_arrays=3, acquired_arrays=3, released_arrays=3, live_arrays=0))
            (root / metadata26.CPU_RECEIPT).write_text(json.dumps(cpu))
            for name in metadata26.SYMBOLS:
                (root / name).write_bytes(('EXPLICIT_CPU_FIXTURE_' + name).encode())
            proof = dict(schema=metadata26.SCHEMA, sources=sources,
                images={name: dict(sha256=metadata26.sha(root / name), symbol=symbol)
                        for name, symbol in metadata26.SYMBOLS.items()},
                state_version=1, reset_supported=False, actual_loaded_readback_required=True,
                actual_npu_calls=0, native_acceptance=False, cpu_receipt_sha256=metadata26.sha(root / metadata26.CPU_RECEIPT))
            (root / metadata26.RECEIPT).write_text(json.dumps(proof))
            files = {p.name: metadata26.sha(p) for p in root.iterdir()}
            self.assertEqual(metadata26.verify_parent(source, root, files), proof)
            (root / 'libopapi.so').write_bytes(b'stale dependency')
            with self.assertRaisesRegex(RuntimeError, 'DEPENDENCY_IMAGE_OWNERSHIP_PROOF_MISMATCH'):
                metadata26.verify_parent(source, root, files)
            cpu['counts']['actual_delete_arrays'] = 2
            (root / metadata26.CPU_RECEIPT).write_text(json.dumps(cpu))
            files[metadata26.CPU_RECEIPT] = metadata26.sha(root / metadata26.CPU_RECEIPT)
            with self.assertRaisesRegex(RuntimeError, 'CPU_ALLOCATION_RELEASE_COUNTS_INCOMPLETE'):
                metadata26.verify_parent(source, root, files)

    def test_actual_image_id_is_required(self):
        actual = dict(Id='sha256:' + 'a' * 64, Architecture='arm64', Os='linux')
        with patch.object(build.subprocess, 'run', return_value=NS(stdout=json.dumps([actual]))) as run:
            self.assertEqual(build.inspect_image(), actual)
            self.assertEqual(run.call_args.args[0], ['docker', 'image', 'inspect', build.IMAGE])
        actual['Id'] = build.IMAGE
        with patch.object(build.subprocess, 'run', return_value=NS(stdout=json.dumps([actual]))):
            with self.assertRaisesRegex(RuntimeError, 'ACTUAL_LOCAL_IMAGE_ID_REQUIRED'):
                build.inspect_image()

    def test_cleanup_checks_both_owner_and_image(self):
        actual = dict(Name='/owned', Image='sha256:' + 'a' * 64,
                      Config=dict(Labels={'demoncat.writer26.build-owner': 'nonce'}))
        inspection = NS(returncode=0, stdout=json.dumps([actual]), stderr='')
        removed = NS(returncode=0, stdout='owned', stderr='')
        with patch.object(build.subprocess, 'run', side_effect=[inspection, removed]) as run:
            self.assertTrue(build.cleanup_owned('owned', 'nonce', actual['Image'])['confirmed'])
            self.assertEqual(run.call_args.args[0], ['docker', 'container', 'rm', '-f', 'owned'])
        for nonce, image in [('foreign', actual['Image']), ('nonce', 'sha256:' + 'b' * 64)]:
            with patch.object(build.subprocess, 'run', return_value=inspection) as run:
                self.assertFalse(build.cleanup_owned('owned', nonce, image)['confirmed'])
                self.assertEqual(run.call_count, 1)

    def test_absent_is_distinct_from_daemon_failure(self):
        for error, confirmed in [('Error: No such container: owned', True), ('daemon unavailable', False)]:
            with patch.object(build.subprocess, 'run', return_value=NS(returncode=1, stderr=error)):
                self.assertIs(build.cleanup_owned('owned', 'nonce', 'sha256:' + 'a' * 64)['confirmed'], confirmed)


if __name__ == '__main__':
    unittest.main()
