"""Real CPU Torch public calls with explicit Native-v3 observation doubles.

These contracts prove Python binding/recovery decisions, not CANN arithmetic or
actual Native descriptor ownership. The separate SDK runtime contract does that
with host SDK doubles; natural acceptance still needs the root-owned NPU run.
"""
import copy
import ctypes as C
from pathlib import Path
import sys
import tempfile
import threading
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch
import weakref
import uuid

import torch

from fault_injection.deployments.writer26_experiment import Writer26Experiment
from fault_injection.deployments.writer26_group import Call, packet
from fault_injection.deployments.writer26_fia_typed import (ABI, Abi, Descriptor, Record, TraceState, Snapshots,
    NativeTypedFia, bind_before, ncl, profile)
from demoncat.runtime.vllm_ascend.worker_extension.serving import complete_tensor
from test_writer26_inventory import CpuReader
import test_writer26_public_return as public


def descriptor(tensor, *, identity=None, raw=None):
    shape, stride = list(tensor.shape), list(tensor.stride())
    return packet(Descriptor(version=2, size=280, complete=1, contiguous=1, dtype=27, format=47,
        view_rank=3, storage_rank=3, stride_rank=3, descriptor=id(tensor) if identity is None else identity,
        raw=tensor.data_ptr() if raw is None else raw, view_bytes=tensor.numel()*2,
        storage_bytes=tensor.untyped_storage().nbytes(), element_bytes=2,
        view=(C.c_int64*8)(*(shape+[0]*5)), storage=(C.c_int64*8)(*(shape+[0]*5)),
        strides=(C.c_int64*8)(*(stride+[0]*5))))


class TypedReader:
    def __init__(self, native):
        self.native, self.rows, self.snapshots = native, [], {}
        self.preparing = self.prepared = self.executing = 0

    def state(self, *, idle=False):
        if idle:
            assert not (self.preparing or self.prepared or self.executing)
        return dict(version=2, size=104, capacity=32768, total=len(self.rows), stored=len(self.rows),
            overflow=0, errors=0, exceptions=0, before_gws=sum(r['phase']==1 for r in self.rows),
            after_gws=sum(r['phase']==2 for r in self.rows), before_execute=sum(r['phase']==3 for r in self.rows),
            preparing=self.preparing, prepared=self.prepared, executing=self.executing)

    def add(self, row):
        row = copy.deepcopy(row)
        row['sequence'] = len(self.rows)+1
        self.rows.append(row)
        return copy.deepcopy(row)

    def delta(self, before):
        return self.state(idle=True), copy.deepcopy(self.rows[before['stored']:])

    def current_before(self, call, event):
        row = copy.deepcopy(self.rows[-1])
        bind_before(row, call, event)
        return row

    def snapshot(self, token):
        return None, copy.deepcopy(self.snapshots[token])


class TypedNative(public.PublicFiaV2):
    def __init__(self, reader):
        super().__init__(reader)
        self.fia_typed = TypedReader(self)
        self.owner = 'typed-cpu-' + uuid.uuid4().hex
        self.before_mutator = self.after_mutator = self.execute_mutator = None
        self.reference_change = self.missing_owned_record = self.released_missing = False
        self.stock_change = False
        self.change_snapshot_before_consumer = False
        self.error = 0

    def execute(self, call, mode, target, allocation, *, typed_before=None):
        assert typed_before == self.fia_typed.rows[-1]
        return super().execute(call, mode, target, allocation)

    def state(self):
        row = super().state()
        row['core']['error'] = self.error
        return row

    def writer(self, x):
        self.writer_calls += 1
        output = x.clone()
        if self.stock_change and self.writer_calls == 3:
            output.reshape(-1)[-1] += 1
        self.output_ref = weakref.ref(output)
        self.output_ids.append(id(output))
        self.reader.raw.all_runs += 1
        self.reader.raw.non_target_runs += 1
        event = self.reader.append(3, output=output, residual=self.last, producer=self.sequence, dtype=27, format=47)
        call = Call(version=2, size=C.sizeof(Call), mechanism=2, dimensions=3, device=0, dtype=27, format=47,
            token=event.call, thread=threading.get_native_id(), independent_call=event.call, producer_sequence=self.sequence,
            output=output.data_ptr(), output_bytes=output.untyped_storage().nbytes(),
            residual=0 if self.last is None else self.last.data_ptr(),
            residual_bytes=0 if self.last is None else self.last.untyped_storage().nbytes(),
            shape=(C.c_int64*4)(*output.shape, 0))
        typed = self.fia_typed
        before = packet(Record(version=2, size=656, phase=1, api=1, device=0, event_sequence=event.sequence,
            call=event.call, thread=call.thread, layout_length=3, layout=(C.c_uint8*16)(84, 78, 68)))
        before['original'] = descriptor(output)
        before['effective'] = copy.deepcopy(before['original'])
        if self.before_mutator:
            self.before_mutator(before)
        before = typed.add(before)
        typed.preparing = 1
        snapshot = packet(Snapshots(version=3, size=2048, token=call.token, owner_hi=1, owner_lo=2,
                                    generation=self.config.generation))
        snapshot['before'] = before
        typed.snapshots[call.token] = snapshot
        if self.admit is not None:
            self.admit(C.pointer(call))
        selected = self.transaction is not None
        mode = self.transaction['mode'] if selected else None
        if mode == 'fault':
            self.last.view(-1)[-16:] += 1  # Explicit lifecycle-double effect, not a CANN numeric oracle.
        if mode == 'reference' and self.reference_change:
            output.reshape(-1)[-1] += 1
        raw = (999 if mode == 'lossless' else self.last.data_ptr() if mode == 'fault' else output.data_ptr())
        effective = descriptor(output, identity=id(output)+1 if selected else id(output), raw=raw)
        gws = self.reader.append(4, output=output, call=event.call, executor=456, dtype=27, format=47)
        gws.output = raw
        after = copy.deepcopy(before)
        after.update(phase=2, event_sequence=gws.sequence, executor=456, effective=effective)
        if self.after_mutator:
            self.after_mutator(after)
        after = typed.add(after)
        typed.preparing, typed.prepared = 0, 1
        snapshot.update(after=after, created=int(selected))
        enter = self.reader.append(5, output=output, residual=self.last, producer=self.sequence,
            call=event.call, executor=456, registration=1, dtype=27, format=47)
        enter.output = raw
        if selected:
            execution = copy.deepcopy(after)
            execution.update(phase=3, event_sequence=enter.sequence, registration=1)
            if self.execute_mutator:
                self.execute_mutator(execution)
            if not self.missing_owned_record:
                execution = typed.add(execution)
            snapshot.update(before_execute=execution, released=0 if self.released_missing else 1, native_return_success=1)
        self.reader.append(6, call=event.call, executor=456)
        typed.prepared = typed.executing = 0
        return output

    def receipt(self, token):
        return dict(self.receipts[token])

    def producer(self, x, residual):
        if self.transaction is not None and self.change_snapshot_before_consumer:
            token = self.transaction['call'].token
            self.fia_typed.snapshots[token]['before_execute']['effective']['raw'] += 1
        return super().producer(x, residual)

    def stop(self):
        pass

    def drain(self):
        pass

    def restore(self, token):
        assert self.transaction['call'].token == token


class FiaTypedTests(unittest.TestCase):
    def setUp(self):
        self.reader = CpuReader()
        self.native = TypedNative(self.reader)
        self.enterContext(patch.object(public, '_RUNTIME', self.native))
        self.model = public.PublicFiaModel().eval()
        self.folder = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.npu = NS(npu_fused_infer_attention_score=public._PACKET, get_npu_format=lambda _: 2)
        self.enterContext(patch.dict(sys.modules, {'torch_npu': self.npu}))
        self.enterContext(patch.object(torch.ops._C_ascend, 'npu_add_rms_norm_bias', self.native.producer, create=True))
        self.experiment = Writer26Experiment(self.model, self.native, self.reader, worker_identity=dict(rank=0),
            folder=self.folder/'typed', stage='effects', max_targets=8)
        self.addCleanup(self.cleanup)

    def cleanup(self):
        if self.experiment.current is not None:
            self.experiment.fail('CPU_FIXTURE_ABORT')
        self.experiment.close()

    def forward(self, index):
        with torch.inference_mode():
            self.experiment.begin('typed-cpu-'+str(index), 'e'*64)
            result = self.model(torch.ones(13, 576, dtype=torch.bfloat16))
            _, output, finite = complete_tensor(result)
            self.experiment.end(output, model_output_finite=finite)

    def inventory(self):
        self.experiment.install()
        self.forward(0)
        self.forward(1)

    def test_complete_two_stock_reference_lossless_fault_off_and_zero_host_bytes(self):
        self.inventory()
        self.assertEqual(len(self.experiment.targets), 2)
        for index in range(2, 40):
            if self.experiment.phase == 'complete':
                break
            self.forward(index)
        self.experiment.close()
        value = self.experiment.report()
        self.assertTrue(value['recovered'])
        self.assertFalse(value['c1'] or value['c2'] or value['c3'])
        self.assertEqual([row[1] for row in self.native.executes], ['reference','lossless','off']*2+['fault','off']*2)
        self.assertEqual(value['fia_typed_stock_bridge']['retained_host_bytes'], 0)
        self.assertEqual(len(value['candidates']), 2)
        for row in self.experiment.rows:
            self.assertEqual(len(row['candidates']), 2)
            self.assertEqual(sum(record['phase']==3 for record in row['fia_typed_records']), row['admissions'])
        for outcome in self.experiment.outcomes:
            self.assertEqual(outcome['fia_typed_consumer_snapshot']['before']['original']['format'], 47)
            self.assertEqual(outcome['fia_typed_consumer_snapshot']['before_execute']['effective']['format'], 47)
            self.assertEqual(outcome['released_receipt']['released'], 1)
            if outcome['mode'] != 'fault':
                self.assertTrue(outcome['fia_public_typed']['stock_equal'])
        for artifact in value['fia_typed_stock_bridge']['artifacts']:
            raw = Path(artifact['path']).read_bytes()
            self.assertEqual(len(raw), artifact['bytes'])

    def test_changed_storage_metadata_keeps_complete_unverified_candidates_no_admission(self):
        self.native.before_mutator = lambda row: row['original']['storage'].__setitem__(0, 14)
        self.inventory()
        self.assertEqual(len(self.experiment.candidate_inventory), 2)
        self.assertTrue(all(row['status']=='UNVERIFIED' for row in self.experiment.candidate_inventory))
        self.assertFalse(self.native.executes)

    def test_actual_flat_storage_preserved_through_stock_reference_fault_and_recovery(self):
        original_descriptor = descriptor
        def flat_descriptor(tensor, **kwargs):
            value = original_descriptor(tensor, **kwargs)
            value['storage_rank'] = 1
            value['storage'] = [tensor.numel()] + [0] * 7
            return value
        with patch(__name__ + '.descriptor', side_effect=flat_descriptor):
            self.test_complete_two_stock_reference_lossless_fault_off_and_zero_host_bytes()
        for outcome in self.experiment.outcomes:
            snapshots = outcome['fia_typed_consumer_snapshot']
            for phase in ('before', 'after', 'before_execute'):
                self.assertEqual(snapshots[phase]['original']['storage_rank'], 1)
                self.assertEqual(snapshots[phase]['effective']['storage'], snapshots[phase]['original']['storage'])

    def test_flat_storage_padding_hidden_tail_offset_and_rank_are_refused(self):
        value = descriptor(torch.ones(13, 9, 64, dtype=torch.bfloat16))
        value['storage_rank'] = 1
        value['storage'] = [7488] + [0] * 7
        ncl(value)
        changes = [('storage', [7489] + [0] * 7), ('storage', [7487] + [0] * 7),
                   ('storage', [7488, 1] + [0] * 6), ('storage_rank', 2), ('offset', 1),
                   ('storage_bytes', 14978), ('strides', [577, 64, 1] + [0] * 5)]
        for key, changed in changes:
            with self.subTest(key=key, changed=changed), self.assertRaises(RuntimeError):
                ncl({**value, key: changed})

    def test_original_after_gws_identity_change_cannot_map_public_nd(self):
        self.native.after_mutator = lambda row: row['original'].__setitem__('descriptor', 99)
        self.inventory()
        self.assertEqual(len(self.experiment.candidate_inventory), 2)
        self.assertFalse(self.experiment.targets or self.native.executes)
        self.assertTrue(any('ORIGINAL_OR_EFFECTIVE_CHANGED' in error for call in self.experiment.public_output.calls
                            for error in call['errors']))

    def test_second_stock_last_byte_change_is_not_owned_reference_qualification(self):
        self.native.stock_change = True
        self.experiment.install()
        self.forward(0)
        with self.assertRaisesRegex(RuntimeError, 'HEALTHY_COMPLETE_REQUEST_CHANGED'):
            self.forward(1)
        self.assertFalse(self.native.executes)
        self.assertTrue(any('TWO_STOCK_REQUESTS_DIFFER' in error for call in self.experiment.public_output.calls
                            for error in call['errors']))

    def test_reference_full_bytes_change_retains_lease_then_allows_owned_cleanup(self):
        self.inventory()
        self.native.reference_change = True
        with self.assertRaises(AssertionError):  # Native mock consumer records a refused completion.
            self.forward(2)
        self.assertEqual([row[1] for row in self.native.executes], ['reference'])
        self.assertIn('STOCK_TO_OWNED_DESCRIPTOR_FULL_BYTES_DIFFER', self.experiment.failure)
        self.assertTrue(self.experiment.pending and self.native.transaction)
        self.assertFalse(self.experiment.fia_bridge.references)
        self.experiment.fail(self.experiment.failure)
        self.experiment.close()
        self.assertFalse(self.experiment.pending or self.native.transaction or self.experiment.pending_lifetimes)
        self.assertEqual(self.experiment.fia_bridge.host_bytes, 0)

    def test_missing_owned_before_execute_record_never_releases_as_complete(self):
        self.inventory()
        self.native.missing_owned_record = True
        with self.assertRaises(AssertionError):
            self.forward(2)
        self.assertIn('COMPLETE_CURRENT_TYPED_FIA_RECORDS_REQUIRED', self.experiment.failure)
        self.assertTrue(self.experiment.pending)
        self.assertEqual(len(self.native.executes), 1)

    def test_owned_descriptor_not_released_is_not_a_public_success(self):
        self.inventory()
        self.native.released_missing = True
        with self.assertRaises(AssertionError):
            self.forward(2)
        self.assertIn('OWNED_DESCRIPTOR_LIFECYCLE_REQUIRED', self.experiment.failure)
        self.assertTrue(self.experiment.pending)

    def test_current_consumer_snapshot_change_cannot_acknowledge_previous_public_proof(self):
        self.inventory()
        self.native.change_snapshot_before_consumer = True
        with self.assertRaises(AssertionError):
            self.forward(2)
        self.assertIn('SNAPSHOT_CHANGED_BEFORE_CONSUMER_RELEASE', self.experiment.failure)
        self.assertTrue(self.experiment.pending and self.native.transaction)
        self.assertEqual(len(self.native.executes), 1)

    def test_before_execute_changed_storage_is_not_equivalent_public_output(self):
        self.inventory()
        self.native.execute_mutator = lambda row: row['effective']['storage'].__setitem__(1, 10)
        with self.assertRaises(AssertionError):
            self.forward(2)
        self.assertIn('NCL_VIEW_STORAGE_STRIDE_CHANGED', self.experiment.failure)
        self.assertTrue(self.experiment.pending)

    def test_final_census_cannot_drop_an_unselected_native_call(self):
        self.experiment.install()
        with torch.inference_mode():
            self.experiment.begin('typed-cpu-missing-census', 'e'*64)
            result = self.model(torch.ones(13,576,dtype=torch.bfloat16))
            _, output, finite = complete_tensor(result)
            original = self.native.fia_typed.delta
            with patch.object(self.native.fia_typed, 'delta',
                              side_effect=lambda before: (original(before)[0], original(before)[1][:-1])):
                with self.assertRaisesRegex(RuntimeError, 'COMPLETE_TYPED_FIA_NATIVE_DENOMINATOR_CHANGED'):
                    self.experiment.end(output, model_output_finite=finite)
        self.assertEqual(len(self.experiment.rows[0]['candidates']), 2)
        self.assertFalse(self.native.executes)

    def test_later_lossless_cannot_use_unverified_or_changed_stock(self):
        self.inventory()
        key = next(iter(self.experiment.targets))
        before = self.experiment.targets[key]['callback']['fia_typed_before']
        with self.assertRaisesRegex(RuntimeError, 'REFERENCE_BEFORE_DEPENDENT_MODE'):
            self.experiment.fia_bridge.before_admit(key, before, 'lossless')
        changed = copy.deepcopy(before)
        changed['layout'][0] = 66
        with self.assertRaisesRegex(RuntimeError, 'CURRENT_STOCK_PROOFS'):
            self.experiment.fia_bridge.before_admit(key, changed, 'reference')


class FiaTypedCoverageTests(unittest.TestCase):
    setUp = FiaTypedTests.setUp
    cleanup = FiaTypedTests.cleanup
    forward = FiaTypedTests.forward

    def make(self, proof=None):
        from fault_injection.deployments.writer26_coverage import Writer26Coverage
        if proof is not None:
            self.native.config.generation = proof.generation + 1
        self.experiment = Writer26Coverage(self.model, self.native, self.reader,
            worker_identity=dict(rank=0, logical_device=0, worker_start_identity=self.native.owner,
                                 model_instance_id=self.native.owner+':model'),
            folder=self.folder/('coverage-'+uuid.uuid4().hex), max_targets=8,
            trigger='once' if proof is None else 'every_match', c2_proof=proof, evidence_kind='cpu_contract')
        return self.experiment

    def all_requests(self):
        self.experiment.install()
        for index in range(40):
            if self.experiment.phase == 'complete':
                break
            self.forward(index)
        self.experiment.close()
        self.experiment.finalize()
        return self.experiment.report()

    def test_typed_complete_group_and_original_live_cpu_proof_every_match(self):
        first = self.make()
        value = self.all_requests()
        self.assertTrue(value['coverage']['contract_satisfied'])
        self.assertEqual((value['coverage']['planned_targets'], value['coverage']['hits']), (2,2))
        self.assertFalse(value['c2'])
        self.assertEqual(value['fia_typed_stock_bridge']['retained_host_bytes'], 0)
        self.make(first.proof)
        value = self.all_requests()
        self.assertTrue(value['coverage']['contract_satisfied'])
        self.assertEqual((value['coverage']['observed_eligible_calls'], value['coverage']['hits']), (4,4))
        self.assertFalse(value['c3'])

    def test_copied_cpu_proof_cannot_enable_typed_fault_admission(self):
        from dataclasses import replace
        first = self.make()
        self.all_requests()
        self.make(replace(first.proof))
        previous_faults = sum(mode == 'fault' for _,mode,_ in self.native.executes)
        self.experiment.install()
        for index in range(5):
            self.forward(index)
        self.assertEqual(self.experiment.phase, 'group_fault')
        with self.assertRaisesRegex(RuntimeError, 'MATCHING_C2_PROOF_REQUIRED'):
            self.experiment.begin('typed-copied-proof', 'e'*64)
        self.assertFalse(self.native.sealed)
        self.assertEqual(previous_faults, sum(mode == 'fault' for _,mode,_ in self.native.executes))

    def test_current_typed_failure_after_first_hit_is_existing_eligible_failure(self):
        self.make()
        self.experiment.install()
        for index in range(5):
            self.forward(index)
        def changed(row):
            if self.native.writer_calls % 2 == 0:
                row['original']['storage'][0] += 1
        self.native.before_mutator = changed
        with self.assertRaises(RuntimeError):
            self.forward(5)
        calls = self.experiment.coverage.report()['calls']
        self.assertEqual(len(calls), 2)
        self.assertEqual(sum(row['eligible'] for row in calls), 2)
        self.assertFalse(any(row.get('propagation_added') for row in calls))
        self.assertEqual(sum(mode == 'fault' for _,mode,_ in self.native.executes), 1)
        self.assertFalse(self.experiment.coverage.report()['contract_satisfied'])


class FiaTypedWireTests(unittest.TestCase):
    def test_exact_ctypes_sizes_and_strict_profile(self):
        self.assertEqual([C.sizeof(t) for t in (Descriptor, Record, TraceState, Snapshots, Abi)], [280,656,104,2048,48])
        from test_writer26_group_build import build
        valid = build.typed_profile(dict(fia_typed_abi=ABI),
            dict(status='PASS_CPU_TYPED_FIA_RUNTIME', actual_npu_calls=0, cases=18))
        profile(valid)
        for key in ('native_format', 'old_reference_reusable', 'borrowed_original_read_scope'):
            with self.subTest(key=key), self.assertRaisesRegex(RuntimeError, 'FRESH_BUILD_PROFILE'):
                profile({**valid, key: None})

    def test_nd_unknown_and_padded_native_descriptors_stay_refused(self):
        value = descriptor(torch.ones(13,9,64,dtype=torch.bfloat16))
        ncl(value)
        for key, changed in (('format',2),('format',48),('dtype',1),('storage_rank',2),('offset',1),('view_bytes',2)):
            with self.subTest(key=key, value=changed), self.assertRaises(RuntimeError):
                ncl({**value,key:changed})

    def test_typed_pending_is_not_a_completed_drain(self):
        native = NativeTypedFia.__new__(NativeTypedFia)
        native.group = NS(config=NS(thread=threading.get_native_id()))
        native.packet = packet
        def state(pointer):
            value = pointer._obj
            value.capacity=32768
            value.prepared=1
            return 0
        native.state_fn=state
        with self.assertRaisesRegex(RuntimeError, 'NOT_DRAINED'):
            native.state(idle=True)

    def test_changed_current_snapshot_is_rejected_before_native_typed_continue(self):
        from fault_injection.deployments.writer26_group import Lease
        native = NativeTypedFia.__new__(NativeTypedFia)
        native.group = NS(config=NS(thread=threading.get_native_id()))
        native.snapshot = lambda token: (None, dict(before=dict(sequence=2)))
        calls = []
        native.continue_fn = lambda proposal: calls.append(proposal)
        lease = Lease(expected=Call(token=7))
        with self.assertRaisesRegex(RuntimeError, 'BEFORE_CHANGED_SINCE_PUBLIC_ADMISSION'):
            native.execute(lease, dict(sequence=1))
        self.assertFalse(calls)


if __name__ == '__main__':
    unittest.main()
