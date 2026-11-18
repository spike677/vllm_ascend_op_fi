"""CPU transfer contracts: exact reviewed bytes, existing implementations only."""

from copy import deepcopy
import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('bundle_exporter', ROOT / 'tools/export_integration_bundle.py')
bundle = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(bundle)


class BundleTests(unittest.TestCase):
    def test_new_attention_normal_and_index_sources_survive_export(self):
        payload = bundle._inventory(ROOT)
        required = {
            'fault_injection/native_adapters/attention28/runtime_log_index.py',
            'fault_injection/native_adapters/attention28/fia/apply_tnd28_prefill_normal.py',
            'fault_injection/native_adapters/attention28/fia/tnd28_prefill_contract.h',
            'fault_injection/native_adapters/attention28/fia/tnd28_prefill_geometry.py',
            'fault_injection/native_adapters/attention28/fia/tnd28_prefill_normal.h',
            'fault_injection/native_adapters/attention28/fia/test_tnd28_prefill_contract.cpp',
            'experiments/worker_core_v11_20261118/check_tnd28_normal_materials.py',
            'tests/test_attention_runtime_log_index.py',
            'tests/test_tnd28_prefill_normal.py',
        }
        self.assertFalse(required - payload.keys())
        for name in required:
            self.assertEqual(payload[name], (ROOT / name).read_bytes(), name)

    def test_actual_all_normal_reader_and_build_source_closure_is_exported(self):
        spec = importlib.util.spec_from_file_location('all_normal_export_contract',
            ROOT / 'experiments/c2c3_closure_20261117/arch22_normal_bridge_evidence.py')
        evidence = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(evidence)
        payload = bundle._inventory(ROOT)
        required = set(evidence.REQUIRED + evidence.S2_REQUIRED + evidence.ALL_REQUIRED + evidence.GROUP_REQUIRED)
        self.assertFalse(required - payload.keys())
        for name in required:
            self.assertEqual(payload[name], (ROOT / name).read_bytes(), name)

    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.folder = Path(temporary.name)
        self.source = self.folder / 'reviewed-source'
        for tree in bundle._TREES:
            (self.source / tree).mkdir(parents=True, exist_ok=True)
        for name in bundle._REQUIRED:
            path = self.source / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text('# CPU source fixture\n', encoding='utf-8')
        for name in bundle._DISTRIBUTIONS:
            (self.source / name).write_bytes((ROOT / name).read_bytes())
        (self.source / 'tools/export_integration_bundle.py').write_bytes(Path(bundle.__file__).read_bytes())
        self.output = self.folder / 'bundle.tar.gz'

    def export(self):
        review = bundle.create_review(self.source, 'cpu-reviewed-snapshot')
        receipt = bundle.export_bundle(self.source, review, self.output)
        return review, receipt

    def members(self, archive=None):
        with tarfile.open(archive or self.output, 'r:gz') as stream:
            return {item.name: stream.extractfile(item).read() for item in stream}

    def rewrite_archive(self, transform):
        entries = self.members()
        rewritten = self.folder / 'modified.tar.gz'
        with tarfile.open(rewritten, 'w:gz') as stream:
            for name, data in transform(entries):
                item = tarfile.TarInfo(name)
                item.size = len(data)
                stream.addfile(item, io.BytesIO(data))
        return rewritten

    def test_three_original_distributions_and_eight_indexes_share_one_source_copy(self):
        review, receipt = self.export()
        self.assertEqual(receipt['status'], 'SOURCE_BUNDLE_VERIFIED')
        self.assertFalse(receipt['qualification_inherited'])
        self.assertEqual(receipt['actual_npu_calls'], 0)
        entries = self.members()
        prefix = bundle.ARCHIVE_ROOT + '/'
        for section in bundle.SECTIONS:
            index = json.loads(entries[prefix + section + '/index.json'])
            self.assertFalse(index['copied_parallel_implementation'])
            for row in index['files']:
                self.assertEqual(bundle._sha(entries[prefix + row['bundle_path']]), row['sha256'])
        self.assertEqual([row['name'] for row in review['distributions']],
                         ['demoncatpfi', 'demoncat-worker-extension', 'demoncat-fault39-core'])
        for project in bundle._DISTRIBUTIONS:
            self.assertEqual(entries[prefix + 'sources/' + project], (self.source / project).read_bytes())
        copied = [name for name in entries if name.endswith('/worker_extension/runtime.py')]
        self.assertEqual(copied, [prefix + 'sources/src/demoncat/runtime/vllm_ascend/worker_extension/runtime.py'])
        state = json.loads(entries[prefix + 'compatibility_matrix/status.json'])
        self.assertEqual(state['inner_framework_integration'], 'PENDING')
        self.assertFalse(state['all_17_accepted'])
        self.assertEqual(state['cann_9_x'], 'NOT_ACCEPTED_BY_THIS_BUNDLE')
        self.assertEqual(state['a3'], 'NOT_ACCEPTED_BY_THIS_BUNDLE')

    def test_export_is_deterministic_and_archive_name_does_not_change_bytes(self):
        review, first = self.export()
        second_path = self.folder / 'another-name.tar.gz'
        second = bundle.export_bundle(self.source, review, second_path)
        self.assertEqual(self.output.read_bytes(), second_path.read_bytes())
        self.assertEqual(first['archive_sha256'], second['archive_sha256'])

    def test_raw_data_binaries_history_and_arbitrary_json_never_enter_review(self):
        private = ('fault_injection/private_tensor.pt', 'fault_injection/raw_statistics.json',
                   'fault_injection/tensor.npy', 'fault_injection/native_adapters/private.so',
                   'fault_injection/evidence/private.py', 'fault39/reports/private.json',
                   'native_execution_strategy/data/history/old_certificate.json',
                   'native_execution_strategy/data/evidence/private.json',
                   'tests/worker_extension/raw_values.jsonl',
                   bundle._LEGACY + 'fault26_readonly_trace_v1/create_fault26_inventory_container.sh')
        for name in private:
            path = self.source / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b'PRIVATE PAYLOAD\x00')
        review, _ = self.export()
        names = {row['source_path'] for row in review['files']}
        self.assertTrue(names.isdisjoint(private))
        self.assertFalse(any(b'PRIVATE PAYLOAD' in data for data in self.members().values()))

    def test_existing_safe_source_change_refuses_before_creating_archive(self):
        review = bundle.create_review(self.source, 'review-before-change')
        (self.source / 'fault_injection/integration.py').write_text('# changed\n', encoding='utf-8')
        with self.assertRaisesRegex(bundle.BundleError, 'REVIEWED_SOURCE_CHANGED'):
            bundle.export_bundle(self.source, review, self.output)
        self.assertFalse(self.output.exists())

    def test_new_worker_code_is_copied_from_selected_root_but_never_executed(self):
        path = self.source / 'src/demoncat/runtime/vllm_ascend/worker_extension/new_census.py'
        data = b'raise RuntimeError("packaging must not execute reviewed code")\n'
        path.write_bytes(data)
        review, _ = self.export()
        name = path.relative_to(self.source).as_posix()
        self.assertEqual(next(row['sha256'] for row in review['files'] if row['source_path'] == name),
                         bundle._sha(data))
        self.assertEqual(self.members()[bundle.ARCHIVE_ROOT + '/sources/' + name], data)

    def test_census_requires_observers_fixture_and_docs_without_engineering_history(self):
        census = self.source / bundle._CENSUS
        census.write_text('# natural call census fixture\n', encoding='utf-8')
        for name in bundle._CORE_DOCS:
            (self.source / name).write_text('# current Core contract\n', encoding='utf-8')
        project = self.source / 'pyproject.toml'
        project.write_text(project.read_text(encoding='utf-8').replace(
            '"native_execution_strategy*"]', '"native_execution_strategy*", "engineering_closure*"]'), encoding='utf-8')
        private = ('engineering_closure/historical_evidence/inherited_pass.csv',
                   'engineering_closure/engineering_tests/validation.json',
                   'engineering_closure/npu_execution/batch1.json',
                   'engineering_closure/reports/receipt.py', 'engineering_closure/queue.py')
        for name in private:
            path = self.source / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text('must not ship\n', encoding='utf-8')
        review, _ = self.export()
        names = {row['source_path'] for row in review['files']}
        self.assertTrue(bundle._OBSERVERS | bundle._CORE_DOCS <= names)
        self.assertIn('tests/worker_extension/test_serving_context.py', names)
        self.assertTrue(names.isdisjoint(private))
        (self.source / 'docs/worker_core_v1/census_contract.md').unlink()
        with self.assertRaisesRegex(bundle.BundleError, 'REQUIRED_SOURCE_MISSING'):
            bundle.create_review(self.source, 'incomplete-census')

    def test_census_refuses_source_only_observers_missing_from_wheel_declaration(self):
        (self.source / bundle._CENSUS).write_text('# census\n', encoding='utf-8')
        for name in bundle._CORE_DOCS:
            (self.source / name).write_text('# current contract\n', encoding='utf-8')
        project = self.source / 'pyproject.toml'
        project.write_text(project.read_text(encoding='utf-8').replace(', "engineering_closure*"', ''), encoding='utf-8')
        with self.assertRaisesRegex(bundle.BundleError, 'CENSUS_OBSERVER_PACKAGE_NOT_IN_ROOT_DISTRIBUTION'):
            bundle.create_review(self.source, 'source-only-observers')

    def test_rope_public_provider_requires_portable_fixture_and_current_contract_only(self):
        path = self.source / bundle._ROPE_PROVIDER
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('# reviewed RoPE RPC adapter\n', encoding='utf-8')
        with self.assertRaisesRegex(bundle.BundleError, 'REQUIRED_SOURCE_MISSING'):
            bundle.create_review(self.source, 'missing-rope-contract')
        names = bundle._ROPE_CONTRACTS | {'tests/worker_extension/test_core_rope22.py'}
        for name in names:
            (self.source / name).write_bytes((ROOT / name).read_bytes())
        history = self.source / 'experiments/unreviewed/rope_receipt.json'
        history.parent.mkdir(parents=True)
        history.write_text('{"accepted": true}', encoding='utf-8')
        review, _ = self.export()
        self.assertTrue(names <= {row['source_path'] for row in review['files']})
        self.assertFalse(any(row['source_path'].startswith('experiments/') for row in review['files']))
        (self.source / 'tests/worker_extension/rope22_fixture.py').unlink()
        with self.assertRaisesRegex(bundle.BundleError, 'REQUIRED_SOURCE_MISSING'):
            bundle.create_review(self.source, 'incomplete-rope-fixture')

    def test_adding_selected_code_after_review_is_not_silently_included(self):
        review = bundle.create_review(self.source, 'review-before-new-file')
        (self.source / 'fault_injection/new_provider.py').write_text('# new\n', encoding='utf-8')
        with self.assertRaisesRegex(bundle.BundleError, 'REVIEWED_SOURCE_CHANGED'):
            bundle.export_bundle(self.source, review, self.output)

    def test_kv41_public_provider_requires_existing_cpu_dependencies_and_current_contract(self):
        path = self.source / bundle._KV41_PROVIDER
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('# reviewed KV41 RPC adapter\n', encoding='utf-8')
        with self.assertRaisesRegex(bundle.BundleError, 'REQUIRED_SOURCE_MISSING'):
            bundle.create_review(self.source, 'missing-kv41-contract')
        names = bundle._KV41_CONTRACTS | {'tests/worker_extension/test_core_kv41.py'}
        for name in names:
            (self.source / name).write_bytes((ROOT / name).read_bytes())
        review, _ = self.export()
        self.assertTrue(names <= {row['source_path'] for row in review['files']})
        self.assertFalse(any(row['source_path'].startswith('experiments/') for row in review['files']))
        (self.source / 'tests/test_postwrite_providers.py').unlink()
        with self.assertRaisesRegex(bundle.BundleError, 'REQUIRED_SOURCE_MISSING'):
            bundle.create_review(self.source, 'incomplete-kv41-fixture')

    def test_missing_native_dependency_and_binary_disguised_as_source_refuse(self):
        dependency = self.source / sorted(bundle._SUPPORT)[0]
        saved = dependency.read_bytes()
        dependency.unlink()
        with self.assertRaisesRegex(bundle.BundleError, 'REQUIRED_SOURCE_MISSING'):
            bundle.create_review(self.source, 'missing-build-input')
        dependency.write_bytes(saved)
        (self.source / 'fault_injection/native_adapters/binary.py').write_bytes(b'\x7fELF\x00')
        with self.assertRaisesRegex(bundle.BundleError, 'BINARY_OR_OVERSIZED_SOURCE'):
            bundle.create_review(self.source, 'disguised-binary')

    def test_output39_route_requires_reused_cpu_contract_closure_without_history(self):
        path = self.source / bundle._OUTPUT39_PROVIDER
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('# reviewed output39 route\n', encoding='utf-8')
        with self.assertRaisesRegex(bundle.BundleError, 'REQUIRED_SOURCE_MISSING'):
            bundle.create_review(self.source, 'missing-output39-contract')
        names = bundle._OUTPUT39_CONTRACTS | {'tests/worker_extension/test_core_output39.py'}
        for name in names:
            (self.source / name).write_bytes((ROOT / name).read_bytes())
        review, _ = self.export()
        self.assertTrue(names <= {row['source_path'] for row in review['files']})
        self.assertFalse(any(row['source_path'].startswith('experiments/') for row in review['files']))
        (self.source / 'tests/worker_extension/test_call_binding.py').unlink()
        with self.assertRaisesRegex(bundle.BundleError, 'REQUIRED_SOURCE_MISSING'):
            bundle.create_review(self.source, 'incomplete-output39-fixture')

    def test_existing_output_is_preserved(self):
        review = bundle.create_review(self.source, 'review')
        self.output.write_bytes(b'previous delivery')
        with self.assertRaises(FileExistsError):
            bundle.export_bundle(self.source, review, self.output)
        self.assertEqual(self.output.read_bytes(), b'previous delivery')

    def test_natural_tensor_public_route_requires_complete_source_only_fixture_and_probe_closure(self):
        path = self.source / bundle._TENSOR_SERVING_PROVIDER
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('# reviewed common 8/18 public route\n', encoding='utf-8')
        with self.assertRaisesRegex(bundle.BundleError, 'REQUIRED_SOURCE_MISSING'):
            bundle.create_review(self.source, 'missing-natural-tensor-contract')
        names = bundle._TENSOR_SERVING_CONTRACTS | {'tests/worker_extension/test_core_tensor_serving.py'}
        for name in names:
            target = self.source / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes((ROOT / name).read_bytes())
        raw = self.source/'experiments/c2c3_closure_20261117/tensor-core-raw-receipt.json'
        raw.write_text('{"accepted":true}', encoding='utf-8')
        review, _ = self.export()
        included = {row['source_path'] for row in review['files']}
        self.assertTrue(names <= included)
        self.assertNotIn(raw.relative_to(self.source).as_posix(), included)
        recovery = self.source/'docs/worker_core_v1/moe_nested_mode_recovery.md'
        saved = recovery.read_bytes(); recovery.unlink()
        with self.assertRaisesRegex(bundle.BundleError, 'REQUIRED_SOURCE_MISSING'):
            bundle.create_review(self.source, 'missing-current-moe-recovery-contract')
        recovery.write_bytes(saved)
        (self.source/'tests/test_route8_index22.py').unlink()
        with self.assertRaisesRegex(bundle.BundleError, 'REQUIRED_SOURCE_MISSING'):
            bundle.create_review(self.source, 'incomplete-natural-tensor-fixture')

    def test_writer26_core_requires_typed_fixture_and_exact_build_helper_without_history(self):
        path = self.source / bundle._WRITER26_PROVIDER
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b'# reviewed Core writer bridge\n')
        with self.assertRaisesRegex(bundle.BundleError, 'REQUIRED_SOURCE_MISSING'):
            bundle.create_review(self.source, 'missing-writer26-fixture')
        names = bundle._WRITER26_CONTRACTS | {'tests/worker_extension/test_core_writer26.py'}
        for name in names:
            target = self.source / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes((ROOT / name).read_bytes())
        review, _ = self.export()
        self.assertTrue(names <= {row['source_path'] for row in review['files']})
        self.assertEqual([row['source_path'] for row in review['files'] if row['source_path'].startswith('experiments/')],
            ['experiments/c2c3_closure_20261117/providers26_build_group.py'])
        for name in ('tests/worker_extension/test_writer26_group_build.py',
                'experiments/c2c3_closure_20261117/providers26_build_group.py'):
            path = self.source/name; saved = path.read_bytes(); path.unlink()
            with self.subTest(missing=name), self.assertRaisesRegex(bundle.BundleError, 'REQUIRED_SOURCE_MISSING'):
                bundle.create_review(self.source, 'missing-typed-profile-helper')
            path.write_bytes(saved)
        (self.source / 'tests/worker_extension/test_collector_dispatch.py').unlink()
        with self.assertRaisesRegex(bundle.BundleError, 'REQUIRED_SOURCE_MISSING'):
            bundle.create_review(self.source, 'missing-typed-fia-public-observer-fixture')

    def test_v11_export_requires_endpoint_attention_writer_and_row_contracts(self):
        path = self.source / bundle._V11_ENDPOINTS
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b'# V1.1 endpoint routing\n')
        with self.assertRaisesRegex(bundle.BundleError, 'REQUIRED_SOURCE_MISSING'):
            bundle.create_review(self.source, 'missing-v11-contracts')
        for name in bundle._V11_CONTRACTS:
            target = self.source / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes((ROOT / name).read_bytes())
        review, _ = self.export()
        self.assertTrue(bundle._V11_CONTRACTS <= {row['source_path'] for row in review['files']})
        for name in ('docs/contracts/execution_adapter_contract.md',
                     'tests/worker_extension/test_core_attention_serving.py',
                     'tests/test_writer26_core_coverage_probe.py',
                     'tests/worker_extension/test_quant_mla_view_rows.py'):
            path = self.source / name
            saved = path.read_bytes()
            path.unlink()
            with self.subTest(missing=name), self.assertRaisesRegex(bundle.BundleError, 'REQUIRED_SOURCE_MISSING'):
                bundle.create_review(self.source, 'incomplete-v11-contract-set')
            path.write_bytes(saved)

    def test_changed_source_and_contract_payload_are_detected(self):
        review, _ = self.export()
        for suffix, error in (('/sources/fault_injection/integration.py', 'SOURCE_HASH_MISMATCH'),
                              ('/compatibility_matrix/status.json', 'GENERATED_CONTRACT_MISMATCH')):
            with self.subTest(suffix=suffix):
                modified = self.rewrite_archive(lambda entries: [
                    (name, data + b'changed' if name.endswith(suffix) else data) for name, data in entries.items()])
                with self.assertRaisesRegex(bundle.BundleError, error):
                    bundle.verify_bundle(modified, review)

    def test_missing_duplicate_and_traversal_archive_members_are_rejected(self):
        review, _ = self.export()
        transforms = (
            (lambda entries: list(entries.items())[1:], 'ARCHIVE_ENTRY_MISSING'),
            (lambda entries: list(entries.items()) + [next(iter(entries.items()))], 'DUPLICATE'),
            (lambda entries: list(entries.items()) + [('demoncatpfi-integration/../escaped.py', b'x')], 'UNSAFE'),
        )
        for transform, error in transforms:
            with self.subTest(error=error):
                modified = self.rewrite_archive(transform)
                with self.assertRaisesRegex(bundle.BundleError, error):
                    bundle.verify_bundle(modified, review)

    def test_review_cannot_add_private_or_traversal_source(self):
        review = bundle.create_review(self.source, 'review')
        for name in ('../outside.py', 'fault_injection/private.so', 'fault39/evidence/raw.json'):
            forged = deepcopy(review)
            forged['files'].append(dict(source_path=name, bytes=1, sha256='a' * 64))
            with self.subTest(name=name), self.assertRaises(bundle.BundleError):
                bundle.export_bundle(self.source, forged, self.output)

    def test_archive_link_is_refused_without_extracting_anything(self):
        review, _ = self.export()
        linked = self.folder / 'linked.tar.gz'
        with tarfile.open(linked, 'w:gz') as archive:
            item = tarfile.TarInfo(bundle.ARCHIVE_ROOT + '/sources/fault_injection/integration.py')
            item.type, item.linkname = tarfile.SYMTYPE, '../../outside.py'
            archive.addfile(item)
        with self.assertRaisesRegex(bundle.BundleError, 'UNSAFE_ARCHIVE_ENTRY'):
            bundle.verify_bundle(linked, review)
        self.assertFalse((self.folder / 'outside.py').exists())

    def test_malformed_distribution_review_is_explicitly_refused(self):
        review = bundle.create_review(self.source, 'review')
        review['distributions'] = None
        with self.assertRaisesRegex(bundle.BundleError, 'INVALID_DISTRIBUTION_REVIEW'):
            bundle.export_bundle(self.source, review, self.output)

    def test_linked_source_does_not_escape_review_root(self):
        external = self.folder / 'outside.py'
        external.write_text('# outside\n', encoding='utf-8')
        link = self.source / 'fault_injection/linked.py'
        try:
            os.symlink(external, link)
        except OSError as error:
            self.skipTest('Source symlink creation unavailable on this host: ' + type(error).__name__)
        with self.assertRaisesRegex(bundle.BundleError, 'LINKED_SOURCE_REFUSED'):
            bundle.create_review(self.source, 'linked-source')


class ActualSourceRelocationTests(unittest.TestCase):
    def test_every_existing_local_quoted_native_include_is_in_the_bundle(self):
        import re

        review = bundle.create_review(ROOT, 'cpu-local-header-closure-review')
        names = {row['source_path'] for row in review['files']}
        checked = 0
        for name in sorted(names):
            source = ROOT / name
            if source.suffix not in {'.cpp', '.h', '.inc'}:
                continue
            for include in re.findall(r'^\s*#\s*include\s*"([^"]+)"',
                                      source.read_text(encoding='utf-8'), re.M):
                dependency = (source.parent / include).resolve()
                # SDK/generated headers are separately supplied by native build
                # recipes. Existing repository-local headers must survive export.
                if dependency.is_file() and dependency.is_relative_to(ROOT.resolve()):
                    relative = dependency.relative_to(ROOT.resolve()).as_posix()
                    self.assertIn(relative, names, name + ' requires ' + relative)
                    checked += 1
        self.assertGreater(checked, 0)

    def test_actual_bundle_keeps_build_dependency_closure_and_imports_after_relocation(self):
        from fault_injection.native_build.recipes import recipes

        review = bundle.create_review(ROOT, 'cpu-current-worktree-review')
        names = {row['source_path'] for row in review['files']}
        for recipe in recipes():
            self.assertIn(recipe.script, names)
            for source in recipe.source_roots:
                self.assertTrue(source in names or any(name.startswith(source + '/') for name in names), source)
        temporary = tempfile.TemporaryDirectory()
        temp_path = Path(temporary.name).resolve()
        self.assertTrue(temp_path.is_relative_to(Path(tempfile.gettempdir()).resolve()))
        if os.name == 'nt':
            # Keep TemporaryDirectory's own cleanup on the same checked long
            # path used for extraction; the sandbox temp prefix is very long.
            temporary.name = '\\\\?\\' + str(temp_path)
        with temporary as directory:
            folder = Path(directory)
            archive_path = folder / 'actual.tar.gz'
            bundle.export_bundle(ROOT, review, archive_path)
            # Verification completed before this bounded test-only extraction.
            with tarfile.open(archive_path, 'r:gz') as archive:
                for item in archive:
                    path = folder / item.name
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_bytes(archive.extractfile(item).read())
            sources = folder / bundle.ARCHIVE_ROOT / 'sources'
            paths = [str(sources), str(sources / 'src'), str(sources / 'fault39/src')]
            script = (
                'import sys; sys.path[:0] = ' + repr(paths) + '; '
                'from demoncat.runtime.vllm_ascend.worker_extension.config import FaultConfig; '
                'from demoncat.runtime.vllm_ascend.worker_extension.host import HostAdapter; '
                'from fault_injection.native_build import NativeBuildManager; '
                'from fault_injection.integration import HostContract; '
                'from demoncat_fault39.contracts import HostContract as Fault39Host; '
                'from engineering_closure.path_discovery import observe_execution_path; '
                'from engineering_closure.native_observer import NativeObserver; '
                'assert FaultConfig(20,"buffer_substitute").mechanism == "20.buffer_substitute"; '
                'assert FaultConfig(39,"matmul_range").mechanism == "39.matmul_range"; '
                'assert "torch_npu" not in sys.modules; print("relocated-imports-ok")'
            )
            result = subprocess.run([sys.executable, '-I', '-B', '-c', script], cwd=ROOT,
                                    capture_output=True, text=True, timeout=30)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertEqual(result.stdout.strip(), 'relocated-imports-ok')
            # Exercise both real public CPU protocols from the exported tree.
            # Explicit paths exclude the checkout's fixtures and no NPU backend
            # is auto-loaded in this source-only portability test.
            test_paths = paths + [str(sources/'tests'), str(sources/'tests/worker_extension')]
            script = ('import sys,unittest; sys.path[:0]=' + repr(test_paths) + '; '
                'names=["test_core_tensor_serving.CoreTensorServingTests.test_public_protocol_actual_groups_original_proof_and_zero_resource_health",'
                '"test_core_tensor_serving.CoreMoEServingTests.test_public_protocol_actual_groups_original_proof_and_zero_resource_health"]; '
                'result=unittest.TextTestRunner().run(unittest.defaultTestLoader.loadTestsFromNames(names)); '
                'sys.exit(not result.wasSuccessful())')
            result = subprocess.run([sys.executable, '-I', '-B', '-c', script], cwd=sources,
                capture_output=True, text=True, timeout=60, env=dict(os.environ, TORCH_DEVICE_BACKEND_AUTOLOAD='0'))
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            # The new custom-op adapter must remain testable after relocation,
            # with its explicit CPU registration/source fixtures in the bundle.
            contract_paths = paths + [str(sources / 'tests'), str(sources / 'tests/worker_extension')]
            contract_script = ('import os,sys,unittest; os.environ["TORCH_DEVICE_BACKEND_AUTOLOAD"]="0"; '
                'sys.path[:0]=' + repr(contract_paths) + '; '
                'suite=unittest.defaultTestLoader.loadTestsFromNames('
                '["test_postwrite_custom_gemm", "test_core_output39_custom"]); '
                'result=unittest.TextTestRunner().run(suite); '
                'sys.exit(0 if result.wasSuccessful() else 1)')
            result = subprocess.run([sys.executable, '-I', '-B', '-c', contract_script], cwd=folder,
                                    capture_output=True, text=True, timeout=60)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertIn('Ran 21 tests', result.stderr)
            # The typed FIA test imports its exact build-profile fixture lazily.
            # Run that contract as well as the real MoE cleanup negatives from
            # relocated paths, with no checkout or actual container fallback.
            contract_script = ('import os,sys,unittest; os.environ["TORCH_DEVICE_BACKEND_AUTOLOAD"]="0"; '
                'sys.path[:0]=' + repr(contract_paths) + '; '
                'suite=unittest.defaultTestLoader.loadTestsFromNames('
                '["test_writer26_fia_typed", "test_writer26_group_build", "test_core_moe_mode_cleanup"]); '
                'result=unittest.TextTestRunner().run(suite); '
                'sys.exit(0 if result.wasSuccessful() else 1)')
            result = subprocess.run([sys.executable, '-I', '-B', '-c', contract_script], cwd=folder,
                                    capture_output=True, text=True, timeout=60)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == '__main__':
    unittest.main()
