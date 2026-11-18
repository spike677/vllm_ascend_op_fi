"""Local CPU fixtures. No torch import, loader operator call, or NPU execution."""
from contextlib import ExitStack
import hashlib
import json
import os
from pathlib import Path
import struct
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from experiments.worker_core_v11_20261118 import diagnose_fia_journal_visibility as D


def elf_bytes(tag):
    value = bytearray(128)
    value[:6] = b'\x7fELF\x02\x01'
    struct.pack_into('<Q', value, 32, 64)
    struct.pack_into('<HH', value, 54, 56, 1)
    struct.pack_into('<IIQQQQQQ', value, 64, 1, 5, 0, 0, 0, 128, 128, 4096)
    value[120:] = tag.encode()[:8].ljust(8, b'_')
    return bytes(value)


class Fixture:
    def __init__(self, root):
        self.root = root
        self.journal = root / 'journal'; self.journal.mkdir()
        self.source = root / 'source'; self.source.mkdir()
        self.owners = {}
        for role in ('nn', 'opapi', 'device_runtime', 'observer', 'custom', 'torch_npu', 'custom_extension'):
            path = root / (role + '.so')
            if role == 'torch_npu':
                path = root/'torch_npu/lib/libtorch_npu.so'; path.parent.mkdir(parents=True)
                (path.parent.parent/'__init__.py').write_text('# CPU fixture only\n')
            path.write_bytes(elf_bytes(role))
            self.owners[role] = dict(path=str(path.resolve()), file_bytes=128,
                                    file_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                                    segments=D.executable_segments(path))
        library = self.journal / 'libfia_input_journal.so'; library.write_bytes(elf_bytes('journal'))
        sources, rows = {}, []
        for i in range(17):
            name = f'source{i}.cpp'; (self.source/name).write_text(f'// fixture {i}\n')
            identity = D.file_identity(self.source/name); sources[name] = identity['sha256']
            rows.append(dict(path=name, sha256=identity['sha256'], bytes=identity['bytes']))
        manifest = self.source/'source_manifest.json'; manifest.write_bytes(D.canonical(dict(files=rows)))
        profile = dict(candidate='fia_input_journal', sources=sources, originals=self.owners,
                       actual_npu_calls=0, slice_pointer_lineage_proven=False)
        self.ident = hashlib.sha256(D.canonical(profile)).hexdigest(); profile['profile_sha256'] = self.ident
        (self.journal/'profile.json').write_bytes(D.canonical(profile))
        build = dict(candidate='fia_input_journal', qualified=False, actual_npu_calls=0, profile_sha256=self.ident,
                     library=library.name, files={name: D.file_identity(self.journal/name)['sha256']
                        for name in (library.name, 'profile.json')})
        (self.journal/'build.json').write_bytes(D.canonical(build))
        self.args = SimpleNamespace(journal=self.journal, build_sha256=D.file_identity(self.journal/'build.json')['sha256'],
            native_source_root=self.source, native_source_manifest_sha256=D.file_identity(manifest)['sha256'],
            out=root/'output', probe_sha256=D.file_identity(D.__file__)['sha256'],
            torch_npu_init_sha256=D.file_identity(Path(self.owners['torch_npu']['path']).parent.parent/'__init__.py')['sha256'])
        self.init = str(Path(self.owners['torch_npu']['path']).parent.parent/'__init__.py')

    def bundle(self):
        with patch.object(D, 'PROFILE', self.ident):
            return D.validate_bundle(self.args)


class FakeLoader:
    def __init__(self, bundle):
        self.bundle = bundle; self.opened = []; self.closed = []; self.looked_up = []; self.reads = []
        self.bad_memory = False; self.close_code = 0; self.bad_owner = False
        self.hidden_roles = set()
        self.images = []
        for i, (role, entry) in enumerate([('journal', bundle['library']), *bundle['originals'].items()]):
            self.images.append(dict(role=role, path=entry['path'], base=0x10000 * (i+1),
                                    data=Path(entry['path']).read_bytes()))

    def maps(self):
        return '\n'.join(f"{row['base']:x}-{row['base']+len(row['data']):x} r-xp 00000000 00:01 1 {row['path']}"
                         for row in self.images if row['role'] not in self.hidden_roles)

    def open(self, name):
        self.opened.append(name)
        handle = next((i+1 for i, row in enumerate(self.images)
                       if row['path'] == name and row['role'] not in self.hidden_roles), None)
        return handle, None if handle else 'not loaded'

    def close(self, handle):
        self.closed.append(handle)
        return self.close_code, None

    def lookup(self, handle, symbol):
        self.looked_up.append((handle, symbol))
        index = 0 if handle is None else handle-1
        return self.images[index]['base']+120, None

    def origin(self, address):
        row = next(row for row in self.images if row['base'] <= address < row['base']+len(row['data']))
        return dict(path='/missing/foreign.so' if self.bad_owner else row['path'], base=row['base'],
                    symbol='cpu_fixture_not_vendor', symbol_address=row['base']+120)

    def memory(self, address, size):
        self.reads.append((address, size))
        row = next(row for row in self.images if row['base'] <= address < row['base']+len(row['data']))
        value = row['data'][address-row['base']:address-row['base']+size]
        return b'X' * size if self.bad_memory else value

    def successful_opens(self):
        paths = {row['path'] for row in self.images}
        return sum(name in paths for name in self.opened)


class VisibilityContracts(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.fixture = Fixture(Path(self.temp.name)); self.bundle = self.fixture.bundle()

    def test_frozen_build_source_and_originals_are_all_read_back(self):
        self.assertEqual(len(self.bundle['source_files']), 17)
        self.assertEqual(set(self.bundle['originals']), set(self.fixture.owners))
        with self.assertRaisesRegex(RuntimeError, 'FROZEN_SOURCE111_PROFILE_REQUIRED'):
            D.validate_bundle(self.fixture.args)

    def test_changed_compile_source_does_not_inherit_profile(self):
        (self.fixture.source/'source2.cpp').write_text('changed')
        with patch.object(D, 'PROFILE', self.fixture.ident), self.assertRaisesRegex(RuntimeError, 'COMPILE_SOURCE_DIFFERS'):
            D.validate_bundle(self.fixture.args)

    def test_changed_native_library_is_rejected_before_loader(self):
        (self.fixture.journal/'libfia_input_journal.so').write_bytes(elf_bytes('other'))
        with patch.object(D, 'PROFILE', self.fixture.ident), self.assertRaisesRegex(RuntimeError, 'BUILD_FILE_DIFFERS'):
            D.validate_bundle(self.fixture.args)

    def test_changed_original_owner_is_rejected(self):
        Path(self.fixture.owners['opapi']['path']).write_bytes(elf_bytes('other'))
        with patch.object(D, 'PROFILE', self.fixture.ident), self.assertRaisesRegex(RuntimeError, 'ORIGINAL_DSO_DIFFERS'):
            D.validate_bundle(self.fixture.args)

    def test_all_six_lookups_and_missing_handles_are_preserved_and_references_released(self):
        loader = FakeLoader(self.bundle); report = D.observe(loader, self.bundle, 'cpu')
        self.assertTrue(report['complete']); self.assertFalse(report['cached_execution_owner_proven'])
        self.assertTrue(all(set(row['symbols']) == set(D.SYMBOLS) for row in report['handles']))
        self.assertEqual(len(loader.closed), loader.successful_opens())
        missing = next(row for row in report['handles'] if row['handle'] == 'libopapi_math.so')
        self.assertFalse(missing['loaded'])
        self.assertTrue(all(row['lookup_error'] == 'HANDLE_NOT_ALREADY_LOADED' for row in missing['symbols'].values()))
        self.assertTrue(all(row['actual_loaded_code_bytes_rehashed'] for row in report['owners']))
        self.assertLessEqual(max(size for _, size in loader.reads), D.CHUNK)

    def test_missing_symbol_is_explicit_not_executed_or_invented(self):
        loader = FakeLoader(self.bundle)
        loader.lookup = lambda handle, symbol: (None, 'undefined symbol: '+symbol)
        report = D.observe(loader, self.bundle, 'cpu')
        self.assertTrue(report['complete']); self.assertEqual(report['owners'], [])
        self.assertIsNone(report['handles'][0]['symbols'][D.SYMBOLS[0]]['address'])
        self.assertIn('undefined symbol', report['handles'][0]['symbols'][D.SYMBOLS[0]]['lookup_error'])

    def test_actual_code_drift_retains_errors_and_all_close_attempts(self):
        loader = FakeLoader(self.bundle); loader.bad_memory = True
        report = D.observe(loader, self.bundle, 'cpu')
        self.assertFalse(report['complete']); self.assertTrue(report['errors'])
        self.assertIn('MAPPED_EXECUTABLE_BYTES_DIFFER', str(report['errors']))
        self.assertEqual(len(loader.closed), loader.successful_opens())

    def test_foreign_missing_owner_is_not_relabelled_as_profile_owner(self):
        loader = FakeLoader(self.bundle); loader.bad_owner = True
        report = D.observe(loader, self.bundle, 'cpu')
        self.assertFalse(report['complete']); self.assertEqual(report['owners'], [])
        self.assertEqual(len(loader.closed), loader.successful_opens())

    def test_close_failure_is_retained_and_does_not_claim_complete(self):
        loader = FakeLoader(self.bundle); loader.close_code = 1
        report = D.observe(loader, self.bundle, 'cpu')
        self.assertFalse(report['complete'])
        self.assertIn('NOLOAD_REFERENCE_RELEASE_FAILED', str(report['errors']))

    def test_owner_error_and_reference_release_error_both_survive(self):
        loader = FakeLoader(self.bundle); loader.bad_memory = True; loader.close_code = 1
        report = D.observe(loader, self.bundle, 'cpu')
        self.assertFalse(report['complete'])
        self.assertIn('MAPPED_EXECUTABLE_BYTES_DIFFER', str(report['errors']))
        self.assertIn('NOLOAD_REFERENCE_RELEASE_FAILED', str(report['errors']))

    def test_total_code_budget_does_not_read_past_limit(self):
        loader = FakeLoader(self.bundle)
        with patch.object(D, 'MAX_STAGE_CODE', 256):
            report = D.observe(loader, self.bundle, 'cpu')
        self.assertFalse(report['complete']); self.assertIn('STAGE_CODE_READ_BOUND', str(report['errors']))
        self.assertLessEqual(sum(size for _,size in loader.reads), 256)

    def test_mapping_permissions_and_gaps_refuse_before_memory_read(self):
        loader = FakeLoader(self.bundle); maps = loader.maps
        loader.maps = lambda: maps().replace('r-xp', 'rwxp')
        report = D.observe(loader, self.bundle, 'cpu')
        self.assertFalse(report['complete']); self.assertEqual(loader.reads, [])
        with self.assertRaisesRegex(RuntimeError, 'RANGE_NOT_MAPPED'):
            D.covered_maps([], 0x10000, 128, self.bundle['library']['path'])

    def test_complete_handle_census_bound_does_not_truncate(self):
        rows = [dict(path=f'/different{i}/libopapi.so') for i in range(D.MAX_HANDLES)]
        with self.assertRaisesRegex(RuntimeError, 'HANDLE_CENSUS_BOUND'):
            D.handle_names(self.bundle, rows)

    def test_wrong_executable_geometry_and_file_bounds_refuse(self):
        path = self.fixture.root/'bad.so'; value = bytearray(elf_bytes('bad'))
        struct.pack_into('<I', value, 68, 7); path.write_bytes(value)
        with self.assertRaisesRegex(RuntimeError, 'EXECUTABLE_SEGMENT_REFUSED'):
            D.executable_segments(path)
        with patch.object(D, 'MAX_FILE', 127), self.assertRaisesRegex(RuntimeError, 'FILE_SIZE_OR_TYPE_REFUSED'):
            D.file_identity(path)

    def environment(self, stack):
        stack.enter_context(patch.object(D, 'PROFILE', self.fixture.ident))
        stack.enter_context(patch.dict(os.environ, {
            'LD_PRELOAD':self.bundle['originals']['observer']['path']+os.pathsep+self.bundle['library']['path'],
            'TORCH_DEVICE_BACKEND_AUTOLOAD':'0'}))
        stack.enter_context(patch.dict(D.sys.modules))
        for name in ('torch', 'torch_npu', 'vllm'):
            D.sys.modules.pop(name, None)
        stack.enter_context(patch.object(D.importlib.util, 'find_spec', return_value=SimpleNamespace(origin=self.fixture.init)))
        stack.enter_context(patch.object(D, 'process_identity', return_value=dict(pid=123, thread=12, start_ticks='99', boot_id='cpu')))

    def test_two_stages_keep_same_process_with_no_native_function_calls(self):
        with ExitStack() as stack:
            self.environment(stack); loader = FakeLoader(self.bundle); imports = []
            def importer(name):
                imports.append(name); return SimpleNamespace(__file__=self.fixture.init)
            code = D.run(self.fixture.args, loader_factory=lambda: loader, importer=importer)
        self.assertEqual(code, 0); self.assertEqual(imports, ['torch_npu'])
        report = json.loads((self.fixture.args.out/'result.json').read_bytes())
        self.assertEqual(len(report['observations']), 2)
        self.assertEqual(report['device_isolation'], 'REQUIRES_EXTERNAL_CONTROLLER_EVIDENCE')
        self.assertFalse(report['qualified']); self.assertFalse(report['actual_cached_resolver_or_execution_owner_proven'])
        for row in report['observations']:
            self.assertEqual(D.file_identity(self.fixture.args.out/row['path'])['sha256'], row['sha256'])

    def test_import_error_keeps_both_stages_and_first_error_without_late_load(self):
        with ExitStack() as stack:
            self.environment(stack); loader = FakeLoader(self.bundle)
            def importer(name):
                raise RuntimeError('fixture import failure after constructor')
            code = D.run(self.fixture.args, loader_factory=lambda: loader, importer=importer)
        report = json.loads((self.fixture.args.out/'result.json').read_bytes())
        self.assertEqual(code, 2); self.assertEqual(len(report['observations']), 2)
        self.assertIn('fixture import failure', report['errors'][0]['error'])
        second = json.loads((self.fixture.args.out/'02-after-import.json').read_bytes())
        self.assertEqual(second['stage'], 'after_failed_torch_npu_import')

    def test_torch_only_import_does_not_require_or_load_custom_extension(self):
        with ExitStack() as stack:
            self.environment(stack); loader = FakeLoader(self.bundle)
            loader.hidden_roles = {'torch_npu', 'custom_extension'}
            def importer(name):
                self.assertEqual(name, 'torch_npu'); loader.hidden_roles.remove('torch_npu')
                return SimpleNamespace(__file__=self.fixture.init)
            self.assertEqual(D.run(self.fixture.args, loader_factory=lambda: loader, importer=importer), 0)
        first = json.loads((self.fixture.args.out/'01-before-import.json').read_bytes())
        second = json.loads((self.fixture.args.out/'02-after-import.json').read_bytes())
        loaded = lambda report,role: next(row['loaded'] for row in report['handles']
            if row['handle'] == self.bundle['originals'][role]['path'])
        self.assertFalse(loaded(first, 'torch_npu')); self.assertTrue(loaded(second, 'torch_npu'))
        self.assertFalse(loaded(first, 'custom_extension')); self.assertFalse(loaded(second, 'custom_extension'))

    def test_missing_preloaded_journal_refuses_import_without_loading_it(self):
        with ExitStack() as stack:
            self.environment(stack); loader = FakeLoader(self.bundle); original = loader.open; imports = []
            loader.open = lambda name: (None, 'absent') if name == self.bundle['library']['path'] else original(name)
            code = D.run(self.fixture.args, loader_factory=lambda: loader, importer=lambda name: imports.append(name))
        self.assertEqual(code, 2); self.assertEqual(imports, [])
        self.assertTrue((self.fixture.args.out/'01-before-import.json').is_file())

    def test_wrong_preload_order_refuses_before_any_lookup_or_import(self):
        with ExitStack() as stack:
            self.environment(stack)
            os.environ['LD_PRELOAD'] = os.pathsep.join(reversed(os.environ['LD_PRELOAD'].split(os.pathsep)))
            calls = []
            code = D.run(self.fixture.args, loader_factory=lambda: calls.append('load'), importer=lambda _: calls.append('import'))
        self.assertEqual(code, 2); self.assertEqual(calls, [])

    def test_manifest_or_import_source_mismatch_refuses(self):
        self.fixture.args.native_source_manifest_sha256 = '0'*64
        with patch.object(D, 'PROFILE', self.fixture.ident), self.assertRaisesRegex(RuntimeError, 'SOURCE_MANIFEST_DIFFERS'):
            D.validate_bundle(self.fixture.args)

    def test_loader_only_uses_noload_and_does_not_call_looked_up_address(self):
        calls = []
        class Function:
            def __init__(self, name): self.name = name
            def __call__(self, *args):
                calls.append((self.name, args))
                return {'dlopen':99, 'dlerror':None, 'dlsym':0x1234, 'dlclose':0}.get(self.name, 0)
        libc = SimpleNamespace(**{name:Function(name) for name in ('dlopen','dlclose','dlsym','dlerror','dladdr')})
        with patch.object(D.sys, 'platform', 'linux'), patch.object(D.os, 'RTLD_NOLOAD', 4, create=True), \
             patch.object(D.os, 'RTLD_LAZY', 1, create=True), patch.object(D.C, 'CDLL', return_value=libc):
            loader = D.Loader(); handle, _ = loader.open('not-loaded.so')
            address, _ = loader.lookup(handle, D.SYMBOLS[0]); loader.close(handle)
        self.assertEqual(address, 0x1234)
        self.assertEqual(next(args for name,args in calls if name=='dlopen'), (b'not-loaded.so',5))
        self.assertEqual({name for name,_ in calls}, {'dlopen','dlsym','dlerror','dlclose'})


if __name__ == '__main__':
    unittest.main()
