"""CPU metadata/real public driver contracts, never an ACL/NPU certificate."""
import ctypes
from copy import deepcopy
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import struct
import sys
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

import torch

from experiments.worker_core_v11_20261118 import output39_refusal_diagnostic as diagnostic
from experiments.worker_core_v11_20261118.review39_refusal_diagnostic import review
from fault_injection.deployments.postwrite import _NativeLease, PostwriteRefused, MAX_POSTWRITE_SPAN_BYTES
from test_postwrite_providers import CpuTransport


ROOT = Path(__file__).resolve().parents[1]


class CpuOriginalABI:
    """Byte-only CPU ABI double. The separate C++ harness tests the Native code."""
    def __init__(self):
        self.leases = {}
        self.serial = 0
        self.calls = []
        self.corrupt_info = None

    def dcat_pw_open(self, address, size, device, owner, generation, result):
        self.serial += 1
        self.leases[self.serial] = dict(address=address, size=size, device=device,
            owner=owner, generation=generation, original=ctypes.string_at(address, size), applied=False)
        result._obj.value = self.serial
        self.calls.append('open')
        return 0

    def dcat_pw_apply(self, handle, owner, generation, mode, selected, count, changed):
        lease = self.leases[handle.value]
        self.calls.append('apply')
        bits = list(struct.unpack('<'+'H'*(lease['size']//2), lease['original']))
        if any(x & 0x7f80 == 0x7f80 for x in bits):
            return 4
        if ctypes.string_at(lease['address'], lease['size']) != lease['original']:
            return 5
        changes = 0
        for index in list(selected):
            value = {0:bits[index], 1:0x7f80, 2:0xff80, 3:0x7fc0}[mode]
            changes += value != bits[index]
            bits[index] = value
        ctypes.memmove(lease['address'], struct.pack('<'+'H'*len(bits), *bits), lease['size'])
        lease['applied'] = True
        changed._obj.value = changes
        return 0

    def dcat_pw_restore(self, handle, owner, generation):
        lease = self.leases[handle.value]
        self.calls.append('restore')
        ctypes.memmove(lease['address'], lease['original'], lease['size'])
        lease['applied'] = False
        return 0

    def dcat_pw_release(self, handle, owner, generation):
        self.calls.append('release')
        del self.leases[handle.value]
        return 0

    def dcat_pw_live(self, handle, owner, generation):
        return int(handle.value in self.leases)

    def dcat_pw_copy_original_v1(self, handle, owner, generation, host, capacity, result):
        lease = self.leases.get(handle.value)
        if not lease or (lease['owner'], lease['generation']) != (owner, generation):
            return 2
        if capacity != lease['size']:
            return 1
        ctypes.memmove(host, lease['original'], capacity)
        info = result._obj
        for key, value in dict(version=1, little_endian_bf16=1, handle=handle.value,
            owner=owner, generation=generation, tensor_address=lease['address'],
            elements=capacity//2, span_bytes=capacity, device=-1,
            applied=int(lease['applied'])).items():
            setattr(info, key, value)
        if self.corrupt_info:
            key, value = self.corrupt_info
            setattr(info, key, value)
        return 0


class OriginalTransport(CpuTransport):
    sha256 = 'a'*64
    lease_capacity = 256
    lease_byte_capacity = MAX_POSTWRITE_SPAN_BYTES
    def __init__(self):
        self.lib = CpuOriginalABI()
        self.opened = []
        self.restores = 0
        self.restore_failure = False

    def open(self, tensor, owner, generation):
        self.validate(tensor)
        lease = _NativeLease(self, tensor, owner, generation)
        self.opened.append(lease)
        return lease

    def snapshot(self, tensor):
        return ctypes.string_at(tensor.data_ptr(), tensor.numel()*tensor.element_size())


class Bf16SummaryTests(unittest.TestCase):
    def test_counts_signs_first_bits_and_entire_span_hash(self):
        bits = (0x0000, 0x8000, 0x0001, 0x7f7f, 0xff7f, 0x7f80, 0xff80, 0x7fc1, 0xffc0, 0x7f81, 0xffbf)
        raw = struct.pack('<'+'H'*len(bits), *bits)
        report = diagnostic.summarize_bf16_le(raw)
        self.assertEqual(report['counts'], dict(finite=5, positive_infinity=1,
            negative_infinity=1, quiet_nan=2, signaling_nan=2))
        self.assertEqual(report['first_nonfinite'], dict(index=5, bits_hex='0x7f80',
            classification='positive_infinity', sign_bit=0))
        self.assertEqual(report['first_by_class']['negative_infinity']['sign_bit'], 1)
        self.assertEqual(report['sha256'], hashlib.sha256(raw).hexdigest())
        self.assertEqual(report['elements'], 11)
        changed = raw[:-1]+bytes([raw[-1]^1])
        self.assertNotEqual(diagnostic.summarize_bf16_le(changed)['sha256'], report['sha256'])

    def test_finite_zero_empty_and_odd_bytes(self):
        summary = diagnostic.summarize_bf16_le(b'\0\0\0\x80')
        self.assertIsNone(summary['first_nonfinite'])
        self.assertEqual(summary['nonfinite_elements'], 0)
        for raw in (b'', b'a', b'aa'*(512*1024+1), bytearray(b'aa')):
            with self.assertRaises(diagnostic.DiagnosticRefused):
                diagnostic.summarize_bf16_le(raw)

    def make(self, bits):
        tensor = torch.tensor(bits, dtype=torch.uint16).view(torch.bfloat16)
        transport = OriginalTransport()
        lease = transport.open(tensor, 101, 7)
        return tensor, transport, lease

    def test_same_original_after_live_write_and_no_extra_existing_calls(self):
        tensor, transport, lease = self.make([0x3f80, 0x4000])
        lease.apply(1, (0,))
        before = list(transport.lib.calls), lease.receipt()
        report = diagnostic.copy_lease_original(lease, explicit_cpu_contract=True)
        self.assertEqual(report['span']['counts']['finite'], 2)
        self.assertTrue(torch.isinf(tensor[0]))
        self.assertEqual((transport.lib.calls, lease.receipt()), before)
        self.assertTrue(report['native']['applied'])
        lease.recover()

    def test_actual_return4_preserved_and_zero_not_treated_as_compared(self):
        tensor, transport, lease = self.make([0x3f80, 0xff81])
        with self.assertRaisesRegex(PostwriteRefused, 'POSTWRITE_APPLY_4'):
            lease.apply(1, (0,))
        before = list(transport.lib.calls)
        report = diagnostic.copy_lease_original(lease, explicit_cpu_contract=True)
        self.assertEqual(report['span']['first_nonfinite']['bits_hex'], '0xff81')
        self.assertEqual(report['apply_event']['return_code'], 4)
        self.assertFalse(report['native']['applied'])
        self.assertEqual(transport.lib.calls, before)
        lease.recover()

    def test_fresh_nonfinite_not_misrepresented_as_original_nonfinite(self):
        tensor, transport, lease = self.make([0x3f80, 0x4000])
        tensor[1] = float('nan')
        with self.assertRaisesRegex(PostwriteRefused, 'POSTWRITE_APPLY_5'):
            lease.apply(1, (0,))
        report = diagnostic.copy_lease_original(lease, explicit_cpu_contract=True)
        self.assertEqual(report['span']['nonfinite_elements'], 0)
        self.assertEqual(report['apply_event']['return_code'], 5)
        lease.recover()

    def test_metadata_missing_export_changed_storage_and_unmarked_cpu_refuse(self):
        tensor, transport, lease = self.make([0x3f80, 0x4000])
        lease.apply(0, (0,))
        with self.assertRaisesRegex(diagnostic.DiagnosticRefused, 'ACTUAL_NATIVE_LIBRARY_REQUIRED'):
            diagnostic.copy_lease_original(lease)
        for key, value in (('handle',999), ('owner',999), ('generation',999), ('tensor_address',1),
                           ('elements',1), ('span_bytes',1), ('device',1), ('applied',0)):
            transport.lib.corrupt_info = (key,value)
            with self.subTest(key=key), self.assertRaises(diagnostic.DiagnosticRefused):
                diagnostic.copy_lease_original(lease, explicit_cpu_contract=True)
        transport.lib.corrupt_info = None
        with patch.object(transport.lib, 'dcat_pw_copy_original_v1', None):
            with self.assertRaisesRegex(diagnostic.DiagnosticRefused, 'FRESH_SAME_LEASE'):
                diagnostic.copy_lease_original(lease, explicit_cpu_contract=True)
        tensor.set_(torch.ones_like(tensor))
        with self.assertRaisesRegex(PostwriteRefused, 'STORAGE_CHANGED'):
            diagnostic.copy_lease_original(lease, explicit_cpu_contract=True)


class OriginalDriverTests(unittest.TestCase):
    def fixture(self):
        import test_core_output39_group as group_tests
        import test_core_output39 as c1
        class Serial(group_tests.ParallelProjections):
            def forward(self, input_ids, positions, inputs_embeds=None, intermediate_tensors=None):
                self.forward_calls += 1
                hidden = c1.life.CpuModel.forward(self, input_ids, positions, inputs_embeds, intermediate_tensors)
                return self.other(self.projection(hidden))
        fixture = group_tests.CoreOutput39GroupTests('test_complete_c2_then_original_live_proof_c3_and_cleanup')
        with patch.object(group_tests, 'ParallelProjections', Serial):
            fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        fixture.transport = OriginalTransport()
        fixture.start()
        for index in range(6):
            fixture.request(index)
        self.assertEqual(fixture.provider.phase, 'c2_lossless')
        group = fixture.provider.group
        rows = group.discovery_reports[0]['calls']
        shape = tuple(next(row['output_geometry']['shape'] for row in rows if row['module_path'] == 'other'))
        plan = diagnostic.DiagnosticPlan(predecessor_module='projection', refused_module='other',
            operator='aten.mm.default', refused_shape=shape,
            expected_all_calls=len(group.call_order), expected_eligible=2)
        pins = {name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in diagnostic.PIN_FILES}
        return fixture, plan, pins

    def test_actual_public_driver_three_arms_and_original_refusal_recovery(self):
        fixture, plan, pins = self.fixture()
        observer = diagnostic.RefusalObserver(fixture.provider, ROOT, pins, plan=plan, explicit_cpu_contract=True)
        index = 6
        with observer:
            while fixture.provider.phase != 'c3_fault':
                fixture.request(index)
                index += 1
            with self.assertRaisesRegex(PostwriteRefused, 'POSTWRITE_APPLY_4'):
                fixture.enqueue(index)
            owned_direct_profile = sys.getprofile()
        report = observer.report()
        self.assertEqual(report['errors'], [])
        self.assertEqual(len(report['records']), 5)
        rejected = next(row for row in report['records'] if row['arm'] == 'cumulative_fault' and row['target'] == 'refused')
        self.assertEqual(rejected['origin'], 'SAME_NATIVE_LEASE_ORIGINAL')
        self.assertEqual(rejected['apply_event']['return_code'], 4)
        self.assertEqual(rejected['consumer_status'], 'NOT_REACHED_AFTER_ACTUAL_APPLY_REFUSAL')
        healthy = next(row for row in report['records'] if row['arm'] == 'healthy')
        single = next(row for row in report['records'] if row['arm'] == 'single_predecessor_fault' and row['target'] == 'refused')
        self.assertEqual(healthy['span']['nonfinite_elements'], 0)
        self.assertGreater(single['span']['nonfinite_elements'], 0)
        self.assertEqual(single['span']['sha256'], rejected['span']['sha256'])
        self.assertEqual(single['origin'], 'ACTUAL_FRESH_READ_SPAN')
        self.assertTrue(single['consumer_observations'])
        self.assertFalse(report['c2'] or report['c3'] or report['e5'])
        json.dumps(report, allow_nan=False)  # No accidental Tensor or float NaN payload.
        reviewed = review(report)
        self.assertEqual(reviewed['issues'], [], reviewed)
        self.assertFalse(reviewed['causal_qualification_granted'])
        for key, mutate in (
            ('missing control', lambda r:r['records'].pop(0)),
            ('different complete span', lambda r:r['records'][-1]['span'].update(sha256='0'*64)),
            ('wrong request', lambda r:r['records'][-1]['binding'].update(request_id='foreign')),
            ('changed input', lambda r:r['records'][0]['binding']['request_mapping'].update(input_sha256='f'*64)),
            ('invented consumer', lambda r:r['records'][-1].update(consumer_observations=[{}])),
            ('borrowed fresh origin', lambda r:r['records'][-1].update(origin='ACTUAL_FRESH_READ_SPAN')),
            ('passed guard', lambda r:r['records'][-1]['apply_event'].update(return_code=0)),
            ('unknown/null count', lambda r:r['records'][-1]['span'].update(elements=None)),
            ('changed first bits', lambda r:r['records'][-1]['span']['first_nonfinite'].update(bits_hex='0x3f80')),
            ('foreign native owner', lambda r:r['records'][-1]['native'].update(owner=999)),
            ('saved qualification', lambda r:r.update(c3=True)),
        ):
            changed = deepcopy(report)
            mutate(changed)
            with self.subTest(key=key):
                self.assertTrue(review(changed)['issues'])
        self.assertIs(sys.getprofile(), owned_direct_profile)  # Observer never removes Core's owned profile.
        self.assertIsNone(sys.gettrace())
        fixture.recovery(index+1)
        self.assertIsNone(sys.getprofile())
        self.assertFalse(fixture.transport.lib.leases)

    def test_source_drift_and_foreign_trace_refuse_before_observation(self):
        fixture, plan, pins = self.fixture()
        bad = dict(pins, **{diagnostic.PIN_FILES[0]:'0'*64})
        with self.assertRaisesRegex(diagnostic.DiagnosticRefused, 'SOURCE_NOT_PINNED'):
            diagnostic.RefusalObserver(fixture.provider, ROOT, bad, plan=plan, explicit_cpu_contract=True)
        callback = lambda *args: None
        sys.settrace(callback)
        try:
            with self.assertRaisesRegex(diagnostic.DiagnosticRefused, 'FOREIGN_TRACE'):
                diagnostic.RefusalObserver(fixture.provider, ROOT, pins, plan=plan, explicit_cpu_contract=True)
        finally:
            sys.settrace(None)
        fixture.ok(fixture.client.unregister())

    def test_wrong_frozen_denominator_refuses_before_first_lossless_request(self):
        fixture, plan, pins = self.fixture()
        before = fixture.model.forward_calls, list(fixture.transport.lib.calls)
        with self.assertRaisesRegex(diagnostic.DiagnosticRefused, 'FROZEN_DENOMINATOR_CHANGED'):
            diagnostic.RefusalObserver(fixture.provider, ROOT, pins,
                plan=replace(plan, expected_all_calls=999), explicit_cpu_contract=True)
        self.assertEqual(before, (fixture.model.forward_calls, fixture.transport.lib.calls))
        fixture.ok(fixture.client.unregister())


if __name__ == '__main__':
    unittest.main()
