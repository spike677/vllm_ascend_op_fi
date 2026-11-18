"""CPU-only read-export fixtures: no SDK, loader install, or device calls."""
import ctypes as C
import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from fault_injection import quant_group_diagnostics as diagnostic
from fault_injection.quant_group_native import (ABI_TYPES, AbiSizes, NativeEvent, RawRecord, RuntimeState,
    METADATA_ABI_TYPES, METADATA_CAPACITY, METADATA_DIMENSIONS, MetadataRecord, MetadataState)


class LoadedReadFixture:
    def __init__(self, root):
        self.path = root / diagnostic.LIBRARY
        self.path.parent.mkdir(parents=True)
        self.path.write_bytes(b'\x7fELF CPU fixture, never loaded')
        self.calls = []
        self.abi_changed = False
        self.raw_owner_changed = False
        self.raw_errors = {}
        self.state_changed = False
        self.state = RuntimeState()
        self.state.census.capacity = diagnostic.CAPACITY
        self.state.census.total = self.state.last_sequence = 5
        self.state.census.model_parents = self.state.census.native_prepares = self.state.census.native_executes = 1
        self.state.census.pending = self.state.quarantined = self.state.public_gws_entries = self.state.public_execute_entries = 1
        self.rows = {}
        self.metadata_state = MetadataState(snapshot_size=C.sizeof(METADATA_ABI_TYPES['snapshot']), record_size=C.sizeof(MetadataRecord),
            dimensions=METADATA_DIMENSIONS, capacity=METADATA_CAPACITY, total=1, retained=1,
            storage_bytes=METADATA_CAPACITY*C.sizeof(MetadataRecord))
        self.metadata = MetadataRecord(sequence=1, parent_token=1, public_entry=1, capture_complete=1)
        self.metadata.input.stage = 9
        self.metadata.input.counts[:] = [2, 2, 1]
        self.metadata.input.retained[:] = [2, 2, 1]
        self.metadata.input.shape[:2] = [8, 2048]
        self.metadata.input.strides[:2] = [2048, 1]
        self.metadata.input.storage[0] = 16384
        parent = RawRecord()
        parent.kind, parent.parent.token = 1, 1
        self.rows[1] = parent
        for sequence, kind in ((2, 1), (3, 2), (4, 3), (5, 4)):
            row = RawRecord()
            row.kind = 2
            row.native = NativeEvent(kind=kind, sequence=sequence, registration=2, executor=123 if sequence > 2 else 0,
                thread=456, device=0, status=161001 if kind == 4 else 0, original_calls=1 if kind in (2, 4) else 0)
            self.rows[sequence] = row
        self.functions = {
            'dcat_dqg_abi_v1': diagnostic.READ_EXPORTS['abi'][1](self.abi),
            'dcat_dqg_state_v1': diagnostic.READ_EXPORTS['state'][1](self.read_state),
            'dcat_dqg_raw_v1': diagnostic.READ_EXPORTS['raw'][1](self.raw),
            'dcat_dqg_metadata_state_v1': diagnostic.READ_EXPORTS['metadata_state'][1](self.read_metadata_state),
            'dcat_dqg_metadata_v1': diagnostic.READ_EXPORTS['metadata'][1](self.read_metadata),
        }
        self.addresses = {name: C.cast(value, C.c_void_p).value for name, value in self.functions.items()}
        self.owner = dict(mode='group', path=str(self.path.resolve()), base=4096, inode=10, device=20,
            sha256=hashlib.sha256(self.path.read_bytes()).hexdigest(),
            symbols={name: self.addresses[name] for name in diagnostic.isolation.MARKERS['group']})
        self.process = Mock()
        self.process.dlopen = Mock(return_value=99)
        self.process.dlsym = Mock(side_effect=lambda handle, name: self.addresses.get(name.decode(), 0))
        self.process.dladdr = Mock(side_effect=self.dladdr)
        self.process.dlclose = Mock(return_value=0)
        sources = {'fault_injection/native_adapters/' + name:
            diagnostic._sha(Path(diagnostic.__file__).parent / 'native_adapters' / name)
            for name in diagnostic.PRODUCTION_SOURCES}
        manifest = dict(schema='demoncat.quant.group_deployment.v1', cann='8.5.1', actual_npu_calls=0,
            legacy_c1_library_present=False, library=diagnostic.LIBRARY,
            files={diagnostic.LIBRARY: self.owner['sha256'], **{Path(name).name: sha for name, sha in sources.items()}},
            sources=sources, sdk_identity=dict(abi_sizes={name: C.sizeof(kind) for name, kind in ABI_TYPES.items()},
                metadata_abi_sizes={name: C.sizeof(kind) for name, kind in METADATA_ABI_TYPES.items()}))
        (root / 'build.json').write_text(json.dumps(manifest))
        self.build_sha = diagnostic._sha(root / 'build.json')

    def abi(self, output):
        self.calls.append(('abi',))
        for name, kind in ABI_TYPES.items():
            setattr(output.contents, name, C.sizeof(kind))
        if self.abi_changed:
            output.contents.runtime += 8
        return 0

    def read_state(self, output):
        if self.state_changed and ('state',) in self.calls:
            self.state.completed_calls += 1
        self.calls.append(('state',))
        C.memmove(output, C.byref(self.state), C.sizeof(self.state))
        return 0

    def raw(self, sequence, output):
        self.calls.append(('raw', sequence))
        if sequence in self.raw_errors:
            return self.raw_errors[sequence]
        if sequence not in self.rows:
            return -15
        C.memmove(output, C.byref(self.rows[sequence]), C.sizeof(RawRecord))
        return 0

    def read_metadata_state(self, output):
        self.calls.append(('metadata_state',))
        C.memmove(output, C.byref(self.metadata_state), C.sizeof(self.metadata_state))
        return 0

    def read_metadata(self, sequence, output):
        self.calls.append(('metadata', sequence))
        if sequence != 1:
            return -15
        C.memmove(output, C.byref(self.metadata), C.sizeof(self.metadata))
        return 0

    def dladdr(self, address, output):
        value = C.cast(output, C.POINTER(diagnostic.isolation._DlInfo)).contents
        value.name = os.fsencode(self.path.resolve())
        value.base = 8192 if self.raw_owner_changed and address == self.addresses['dcat_dqg_raw_v1'] else 4096
        return 1


class QuantGroupDiagnosticTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.native = LoadedReadFixture(self.root)
        self.mapped = self.enterContext(patch.object(diagnostic.isolation, 'mapped_native_modes', return_value=(self.native.owner,)))
        self.cdll = self.enterContext(patch.object(diagnostic.C, 'CDLL', return_value=self.native.process))
        for name, value in (('RTLD_NOLOAD', 4), ('RTLD_NOW', 2), ('RTLD_LOCAL', 0)):
            self.enterContext(patch.object(diagnostic.os, name, value, create=True))

    def capture(self, **kwargs):
        return diagnostic.capture_loaded_quant_state(self.root,
            approved_build_sha256=kwargs.get('approved_build_sha256', self.native.build_sha))

    def test_complete_diagnostic_preserves_vendor_failure_and_no_device_authority(self):
        report = self.capture()
        self.assertEqual(report['status'], 'READ_ONLY_SNAPSHOT_COMPLETE')
        self.assertTrue(report['snapshot_complete'])
        self.assertFalse(report['qualification_authority'])
        self.assertEqual(report['diagnostic_device_calls'], 0)
        self.assertEqual(report['state_before']['core']['installed'], 0)
        self.assertEqual(report['state_before']['census']['pending'], 1)
        summary = diagnostic.diagnostic_summary(report)
        self.assertEqual(summary['counts'], dict(parent_records=1, native_records=4, prepare_original_calls=1, execute_original_calls=1))
        self.assertEqual(summary['failed_native_exits'][0]['status'], 161001)
        self.assertEqual(summary['failed_native_exits'][0]['original_calls'], 1)
        self.cdll.assert_called_once_with(None)
        self.native.process.dlopen.assert_called_once_with(os.fsencode(self.native.path.resolve()), 6)
        self.native.process.dlclose.assert_called_once_with(99)
        self.assertEqual([call.args[1].decode() for call in self.native.process.dlsym.call_args_list],
                         ['dcat_dqg_abi_v1', 'dcat_dqg_state_v1', 'dcat_dqg_raw_v1', 'dcat_dqg_metadata_state_v1', 'dcat_dqg_metadata_v1'])
        self.assertEqual([row[1] for row in self.native.calls if row[0] == 'raw'], list(range(1, 6)))
        captured = report['metadata_records'][0]['record']['input']
        self.assertEqual(captured['stage'], 9)
        self.assertEqual(captured['counts'], [2, 2, 1])
        self.assertEqual(captured['shape'][:2], [8, 2048])
        self.assertEqual(captured['result']['rows'], 0)  # No guessed geometry repair.

    def test_shim_refusal_is_not_relabelled_a_real_execute(self):
        self.native.rows[5].native.original_calls = 0
        report = self.capture()
        summary = diagnostic.diagnostic_summary(report)
        self.assertEqual(summary['counts']['execute_original_calls'], 0)
        self.assertEqual(summary['failed_native_exits'][0]['original_calls'], 0)
        self.assertEqual(report['records'][4]['record']['native']['status'], 161001)

    def test_missing_library_does_not_open_or_load_it(self):
        self.mapped.return_value = ()
        report = self.capture()
        self.assertEqual(report['status'], 'REVIEWED_DSO_NOT_LOADED_NO_LOAD_ATTEMPTED')
        self.cdll.assert_not_called()
        self.assertEqual(self.native.calls, [])

    def test_manifest_or_actual_mapping_change_prevents_all_export_reads(self):
        for kind in ('manifest', 'mapping', 'duplicate', 'missing_marker'):
            with self.subTest(kind=kind):
                owner = {**self.native.owner, 'symbols': dict(self.native.owner['symbols'])}
                if kind == 'mapping':
                    owner['sha256'] = '0' * 64
                if kind == 'missing_marker':
                    owner['symbols'].pop('dcat_dqg_state_v1')
                self.mapped.return_value = (owner, owner) if kind == 'duplicate' else (owner,)
                report = self.capture(approved_build_sha256='0' * 64 if kind == 'manifest' else self.native.build_sha)
                self.assertEqual(report['status'], 'READ_REFUSED')
                self.cdll.assert_not_called()
                self.assertEqual(self.native.calls, [])

    def test_abi_mismatch_blocks_state_and_raw_and_releases_noload_reference(self):
        self.native.abi_changed = True
        report = self.capture()
        self.assertEqual(report['status'], 'READ_REFUSED')
        self.assertEqual(self.native.calls, [('abi',)])
        self.native.process.dlclose.assert_called_once_with(99)

    def test_foreign_raw_export_owner_prevents_native_reads_and_releases_reference(self):
        self.native.raw_owner_changed = True
        report = self.capture()
        self.assertEqual(report['status'], 'READ_REFUSED')
        self.assertEqual(self.native.calls, [])
        self.native.process.dlclose.assert_called_once_with(99)

    def test_noload_failure_does_not_fall_back_to_loading(self):
        self.native.process.dlopen.return_value = 0
        report = self.capture()
        self.assertEqual(report['status'], 'READ_REFUSED')
        self.cdll.assert_called_once_with(None)
        self.native.process.dlopen.assert_called_once()
        self.native.process.dlclose.assert_not_called()
        self.assertEqual(self.native.calls, [])

    def test_missing_raw_row_is_retained_and_later_rows_are_still_read(self):
        self.native.raw_errors[3] = -15
        report = self.capture()
        self.assertFalse(report['snapshot_complete'])
        self.assertEqual(len(report['records']), 5)
        self.assertEqual(report['records'][2]['status'], -15)
        self.assertEqual(report['records'][4]['record']['native']['status'], 161001)
        self.assertEqual(report['read_errors'][0]['sequence'], 3)

    def test_raw_identity_mismatch_is_not_a_complete_snapshot(self):
        self.native.rows[4].native.sequence = 3
        report = self.capture()
        self.assertFalse(report['snapshot_complete'])
        self.assertEqual(report['read_errors'][0]['sequence'], 4)

    def test_concurrent_state_or_mapping_change_marks_snapshot_incomplete(self):
        self.native.state_changed = True
        report = self.capture()
        self.assertFalse(report['snapshot_complete'])
        self.assertFalse(report['state_stable'])
        self.native.state_changed = False
        self.mapped.side_effect = [(self.native.owner,), ()]
        report = self.capture()
        self.assertFalse(report['snapshot_complete'])
        self.assertFalse(report['mapping_stable'])

    def test_overflow_is_preserved_and_never_treated_as_complete(self):
        self.native.state.census.overflow = 1
        report = self.capture()
        self.assertFalse(report['snapshot_complete'])
        self.assertEqual(report['state_after']['census']['overflow'], 1)
        self.assertEqual(len(report['records']), 5)

    def test_read_bound_keeps_overflow_denominator_without_unbounded_reads(self):
        self.native.state.census.capacity = 5
        self.native.state.census.total = self.native.state.last_sequence = 9
        self.native.state.census.overflow = 4
        with patch.object(diagnostic, 'CAPACITY', 5):
            report = self.capture()
        self.assertFalse(report['snapshot_complete'])
        self.assertEqual(report['sequence_range'], dict(first=1, last=5, reported_last=9, omitted=4))
        self.assertEqual([row[1] for row in self.native.calls if row[0] == 'raw'], [1, 2, 3, 4, 5])

    def test_unknown_capacity_blocks_raw_reads(self):
        self.native.state.census.capacity = diagnostic.CAPACITY + 1
        report = self.capture()
        self.assertEqual(report['status'], 'READ_REFUSED')
        self.assertEqual(self.native.calls, [('abi',), ('state',)])

    def test_reference_release_failure_is_evidence_not_success(self):
        self.native.process.dlclose.return_value = -1
        report = self.capture()
        self.assertEqual(report['status'], 'READ_REFUSED')
        self.assertFalse(report['snapshot_complete'])
        self.assertEqual(len(report['records']), 5)


if __name__ == '__main__':
    unittest.main()
