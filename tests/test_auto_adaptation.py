"""Offline adaptation checks. Native receipt inputs are explicitly CPU fixtures."""
import copy
from dataclasses import replace
import hashlib
import io
import json
import os
from pathlib import Path
import struct
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import uuid

import torch

from fault_injection import (
    FaultConfig, ModelFaultBackend, ProviderProfile, UnsupportedCall, attach,
    environment_fingerprint, inspect_model, inspect_support, register_runtime, unregister_runtime,
)
from fault_injection.environment import _elf_build_id, _mapped_images, _toolkit
from fault_injection.native import NativeCall, NativeProof
from fault_injection.runtime_discovery import native_facts


def cpu_environment():
    return dict(pid=os.getpid(), machine='aarch64', python='3.11.14', packages={}, torch_cxx11_abi=False,
                driver=dict(installation=None, running_module_versions={}), maps_status='OBSERVED',
                loaded_libraries=[dict(name='libopapi.so', path='/toy/build/libopapi.so',
                                       sha256='1' * 64, elf_build_id='269e', status='MAPPED_FILE_VERIFIED')])


class EnvironmentTests(unittest.TestCase):
    def test_build_id_and_exact_mapped_fingerprint(self):
        note = struct.pack('<III', 4, 4, 3) + b'GNU\0' + bytes.fromhex('12345678')
        header = bytearray(64)
        header[:6] = b'\x7fELF\x02\x01'
        struct.pack_into('<Q', header, 32, 64)
        struct.pack_into('<HH', header, 54, 56, 1)
        elf = header + struct.pack('<IIQQQQQQ', 4, 4, 120, 0, 0, len(note), len(note), 4) + note
        self.assertEqual(_elf_build_id(io.BytesIO(elf), len(elf)), '12345678')
        self.assertIsNone(_elf_build_id(io.BytesIO(elf[:-1]), len(elf) - 1))

        fixtures = (
            '1000-2000 r-xp 00001000 08:02 4 /env A/libopapi.so\n'
            '2000-3000 r-xp 00001000 08:02 5 /env B/libnnopbase.so (deleted)\n'
            '3000-4000 r--p 00000000 08:02 4 /env A/libopapi.so\n'
        )
        self.assertEqual(list(_mapped_images(fixtures, ())), [
            ('/env A/libopapi.so', '08:02', '4', False),
            ('/env B/libnnopbase.so', '08:02', '5', True),
        ])
        environment = cpu_environment()
        required = ('libopapi.so',)
        expected = environment_fingerprint(environment, required)
        environment['toolkit'] = {'installations': [{'version': '9.9'}]}
        self.assertEqual(environment_fingerprint(environment, required), expected)
        environment['loaded_libraries'][0]['sha256'] = '2' * 64
        self.assertNotEqual(environment_fingerprint(environment, required), expected)
        environment['loaded_libraries'][0]['status'] = 'MAPPING_DELETED'
        self.assertIsNone(environment_fingerprint(environment, required))
        self.assertIsNone(environment_fingerprint(cpu_environment(), ('libnnopbase.so',)))

    def test_toolkit_uses_exact_selected_installation_and_headers(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            header = root / 'aarch64-linux/include/acl/acl.h'
            header.parent.mkdir(parents=True)
            header.write_bytes(b'fixture header')
            (root / 'ascend_toolkit_install.info').write_text('version=9.0.0\n')
            (root / 'aarch64-linux/ascend_toolkit_install.info').write_text('version=9.1.0\n')
            with patch.dict(os.environ, {'ASCEND_HOME_PATH': directory}), patch(
                'fault_injection.environment.platform.machine', return_value='aarch64'
            ):
                row, = _toolkit()['installations']
            self.assertEqual(row['fields']['version'], '9.1.0')
            self.assertEqual(row['abi_headers']['acl/acl.h'], hashlib.sha256(header.read_bytes()).hexdigest())
            self.assertIsNone(row['abi_headers']['aclnn/acl_meta.h'])


class ModelStructureTests(unittest.TestCase):
    def test_roles_use_siblings_and_geometry_and_preserve_object_aliases(self):
        model = torch.nn.Module()
        model.config = SimpleNamespace(model_type='renamed-model', num_attention_heads=4, num_key_value_heads=2)
        model.unrelated = torch.nn.Module()
        model.unrelated.down_proj = torch.nn.Linear(16, 8)
        model.arbitrary_attention = torch.nn.Module()
        model.arbitrary_attention.wqkv = torch.nn.Linear(8, 16)
        model.arbitrary_attention.wo = torch.nn.Linear(8, 8)
        model.arbitrary_ffn = torch.nn.Module()
        model.arbitrary_ffn.w13 = torch.nn.Linear(8, 32)
        model.arbitrary_ffn.w2 = torch.nn.Linear(16, 8)
        model.multiple_names = model.arbitrary_ffn.w2
        model.expert_block = torch.nn.Module()
        model.expert_block.experts = torch.nn.ModuleList([torch.nn.Identity(), torch.nn.Identity()])
        model.expert_block.gate_proj = torch.nn.Linear(8, 2)
        structure = inspect_model(model)['structure']
        modules = {row['path']: row for row in structure['modules']}
        qkv = modules['arbitrary_attention.wqkv']
        self.assertEqual(qkv['canonical_roles'], ['ATTENTION_Q_PROJ', 'ATTENTION_K_PROJ', 'ATTENTION_V_PROJ'])
        self.assertIsNone(qkv['fused_segments'])
        self.assertEqual(modules['arbitrary_ffn.w2']['canonical_roles'], ['MLP_DOWN_PROJ'])
        self.assertEqual(modules['arbitrary_ffn.w2']['aliases'], ['arbitrary_ffn.w2', 'multiple_names'])
        self.assertEqual(modules['expert_block.gate_proj']['canonical_roles'], ['MLP_GATE'])
        self.assertEqual(modules['unrelated.down_proj']['role_status'], 'UNRESOLVED')
        self.assertTrue(all(not row['actual_operator_observed'] for row in structure['modules']))


class FixtureObserver:
    """Feed synthetic call-bound receipts while running an ordinary CPU forward."""
    kinds = ('runtime_call', 'abi_handshake', 'semantic_slot', 'live_storage')

    def __init__(self, directory, operator, facts, *, omit=None, mutate=None):
        self.directory = Path(directory)
        self.operator, self.facts, self.omit, self.mutate = operator, facts, omit, mutate
        self.num_calls = 0

    def install_observer(self, callback, candidates, fault):
        restorations = []
        for _, owner, _ in candidates:
            def wrapped(*args, owner=owner, original=owner.forward, **kwargs):
                self.num_calls += 1
                call_id = uuid.uuid4().hex
                details = copy.deepcopy(self.facts)
                if self.mutate:
                    self.mutate(details, self.num_calls)
                proofs = []
                for kind in self.kinds:
                    if kind == self.omit:
                        continue
                    row = dict(schema='demoncat.native.proof.v1', pid=os.getpid(),
                               owner_id=id(owner), call_id=call_id, operator=self.operator,
                               mechanism=fault, kind=kind, status='verified',
                               details=details if kind == 'runtime_call' else {'cpu_fixture': True})
                    data = json.dumps(row).encode()
                    path = self.directory / f'{call_id}-{kind}.json'
                    path.write_bytes(data)
                    proofs.append(NativeProof(kind, str(path), hashlib.sha256(data).hexdigest()))
                call = NativeCall(owner, call_id, self.operator, proofs=tuple(proofs))
                return callback(call, lambda: original(*args, **kwargs))
            restorations.append((owner, vars(owner).get('forward')))
            owner.forward = wrapped

        def remove():
            for owner, old in reversed(restorations):
                if old is None:
                    del owner.forward
                else:
                    owner.forward = old
        return remove


class QualifiedFixture(FixtureObserver):
    """CPU lifecycle double. Device-side byte logic has a separate C++ test."""

    def __init__(self, *args, corrupt=None, corrupt_fault=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.corrupt = corrupt
        self.corrupt_fault = corrupt_fault
        self.modes = []

    def execute_scoped(self, call, invoke_once, mode):
        from fault_injection.qualification import _digest_output

        self.modes.append(mode)
        result = invoke_once()
        formals = ['x1'] if self.operator == 'aclnnQuantMatmulWeightNz' else ['self']
        replaced = [] if mode == 1 else formals
        before = ['7' * 64, '8' * 64]
        receipt = dict(mode=mode, native_status=0, original_calls=1, replacements=len(replaced),
                       changed_slots=len(replaced) if mode == 3 else 0, resources_released=True,
                       temporary_readback_verified=bool(replaced), device_copy=mode == 2,
                       replaced_formals=replaced, input_sha256_before=before,
                       input_sha256_after=list(before), output_sha256=_digest_output(result))
        if self.corrupt and mode == 2:
            receipt.update(self.corrupt)
        if self.corrupt_fault and mode == 3:
            receipt.update(self.corrupt_fault)
        return (result + 1 if mode == 3 else result), receipt


class AdaptationTests(unittest.TestCase):
    def observe_profile(self, fault, operator, model, *, level='NATIVE_VERIFIED',
                        change_environment=False, mutate=None, omit=None):
        environment = cpu_environment()
        required = ('libopapi.so',)
        facts = dict(boundary='op_exec_cache_run', native_signature='fixture:native-signature',
            build_profile_sha256='3' * 64, cache_identity='fixture:cache1', candidate_count=1,
            runtime_identity='MATCHED_BUILD',
            worker=dict(pid=os.getpid(), soc='fixture-910B4', execution_mode='eager', tp=1, rank=0, logical_device=0),
            layout=dict(dtype='BF16' if operator == 'aclnnMatmul' else 'INT8',
                        format='ND', shape=[2, 4], stride=[4, 1], offset=0),
            slots=[dict(formal='self' if operator == 'aclnnMatmul' else 'x1', slot=0, storage_bytes=16)])
        normalized = native_facts(facts, set(FixtureObserver.kinds), fault)
        self.assertEqual(normalized['worker'], facts['worker'])
        self.assertTrue(normalized['process_start_identity'])
        profile = ProviderProfile(implementation_version='cpu-fixture-only', validation_level=level,
            evidence_sha256='4' * 64, environment_fingerprint=environment_fingerprint(environment, required),
            required_libraries=required, build_profile_sha256=facts['build_profile_sha256'],
            operator=operator, boundary=facts['boundary'], native_signature=facts['native_signature'],
            layout_sha256=normalized['layout_sha256'], dtype=facts['layout']['dtype'], format='ND', ndim=2,
            soc=facts['worker']['soc'], execution_mode='eager', tp=1, rank=0, logical_device=0)
        after = copy.deepcopy(environment)
        if change_environment:
            after['loaded_libraries'][0]['sha256'] = '5' * 64

        def injection_factory(_):
            self.fail('Read-only selection loaded an injection factory')

        x = torch.ones(2, 4)
        expected = model(x)
        with tempfile.TemporaryDirectory() as directory, patch(
            'fault_injection.environment.inspect_environment', side_effect=(environment, after)
        ):
            observer = FixtureObserver(directory, operator, facts, omit=omit, mutate=mutate)
            register_runtime(fault, injection_factory, observer_factory=lambda actual: observer, profiles=(profile,))
            try:
                if fault == '20.buffer_substitute':
                    self.assertEqual(inspect_support(model, fault)['adaptation']['profiles'][0]['operator'], operator)
                backend = ModelFaultBackend(model)
                backend.register(FaultConfig(fault, arm='observe'))
                backend.enable()
                for _ in range(2):
                    self.assertTrue(torch.equal(model(x), expected))
                stopped = backend.disable()
                self.assertTrue(stopped['cleanup_confirmed'])
                self.assertTrue(torch.equal(model(x), expected))
                with self.assertRaisesRegex(UnsupportedCall, 'lossless-verified'):
                    attach(model, fault, arm='fault')
                return stopped['session']
            finally:
                unregister_runtime(fault, injection_factory)

    def test_both_mechanisms_use_same_read_only_lifecycle_and_require_lossless(self):
        for fault, operator in (('39.matmul_range', 'aclnnMatmul'),
                                ('20.buffer_substitute', 'aclnnQuantMatmulWeightNz')):
            with self.subTest(fault=fault):
                result = self.observe_profile(fault, operator, torch.nn.Sequential(torch.nn.Linear(4, 4)))
                plan = result['adaptation']
                self.assertEqual(plan['status'], 'LOSSLESS_GATE_REQUIRED')
                self.assertEqual(plan['target']['operator'], operator)
                self.assertEqual(len(plan['target']['call_ids']), 2)
                self.assertFalse(plan['native_execution_authorized'])
                self.assertTrue(all(not event['native_hit_proven'] for event in result['events']))

    def test_gaps_for_cpu_only_changed_libraries_unknown_slots_and_unstable_shapes(self):
        matrix = (
            ({'level': 'CPU_VERIFIED'}, 'NO_COMPATIBLE_STABLE_TARGET', 'NATIVE_VALIDATION_REQUIRED'),
            ({'change_environment': True}, 'NO_COMPATIBLE_STABLE_TARGET', 'LOADED_ENVIRONMENT_FINGERPRINT_MISMATCH'),
            ({'omit': 'semantic_slot'}, 'NATIVE_OPERATOR_EVIDENCE_REQUIRED', 'Missing read-only native proofs: semantic_slot'),
            ({'mutate': lambda row, _: row['layout'].update(shape=[1, 4])},
             'NO_COMPATIBLE_STABLE_TARGET', 'OPERATOR_OR_WORKER_SCOPE_MISMATCH'),
            ({'mutate': lambda row, i: row.update(cache_identity=f'cache{i}')},
             'NO_COMPATIBLE_STABLE_TARGET', 'CANDIDATE_NOT_UNIQUE_AND_STABLE'),
        )
        for kwargs, status, gap in matrix:
            with self.subTest(gap=gap):
                result = self.observe_profile('20.buffer_substitute', 'aclnnQuantMatmulWeightNz',
                                              torch.nn.Sequential(torch.nn.Linear(4, 4)), **kwargs)
                self.assertEqual(result['adaptation']['status'], status)
                self.assertIn(gap, result['adaptation']['gaps'])
                self.assertIsNone(result['adaptation']['target'])

    def test_ambiguous_runtime_is_not_reduced_to_first_layer(self):
        model = torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.Linear(4, 4))
        result = self.observe_profile('39.matmul_range', 'aclnnMatmul', model)
        self.assertEqual(result['adaptation']['status'], 'AMBIGUOUS_TARGET')
        self.assertEqual(len(result['adaptation']['candidates']), 2)
        self.assertIsNone(result['adaptation']['target'])

    def test_unproven_same_operator_does_not_make_another_layer_unique(self):
        def omit_every_other_slot(facts, index):
            if index % 2 == 0:
                facts['slots'] = []

        model = torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.Linear(4, 4))
        result = self.observe_profile('20.buffer_substitute', 'aclnnQuantMatmulWeightNz', model,
                                      mutate=omit_every_other_slot)['adaptation']
        self.assertIsNone(result['target'])
        self.assertIn('OPERATOR_CALLS_WITH_UNPROVEN_IDENTITY', result['gaps'])

    def qualified(self, fault, operator, model, *, corrupt=None, corrupt_fault=None,
                  wrong_model=False, drift=False, replace_binding=False, omit_identity=False,
                  level='NATIVE_VERIFIED', first_live=False, baseline_only=False):
        environment = cpu_environment()
        required = ('libopapi.so',)
        facts = dict(boundary='op_exec_cache_run', native_signature='fixture:native-signature',
            build_profile_sha256='3' * 64, cache_identity='fixture:cache1', candidate_count=1,
            cache_lifecycle='fixture:cache-generation-1',
            runtime_identity='MATCHED_BUILD',
            worker=dict(pid=os.getpid(), soc='fixture-910B4', execution_mode='eager', tp=1, rank=0,
                        logical_device=0, worker_id='fixture-worker', model_instance_id='fixture-model'),
            layout=dict(dtype='INT8' if operator == 'aclnnQuantMatmulWeightNz' else 'BF16',
                        format='ND', shape=[2, 4], stride=[4, 1], offset=0),
            slots=[dict(formal=name, slot=index, storage_bytes=16)
                   for index, name in enumerate(('x1', 'x2', 'out')
                   if operator == 'aclnnQuantMatmulWeightNz' else ('self', 'mat2', 'out'))])
        if omit_identity:
            del facts['worker']['model_instance_id']
        admission_modes = [1, 2, 1]
        layout = native_facts(facts, set(FixtureObserver.kinds), fault)['layout_sha256']
        profile = ProviderProfile(implementation_version='fixture', validation_level=level,
            evidence_sha256='4' * 64, environment_fingerprint=environment_fingerprint(environment, required),
            required_libraries=required, build_profile_sha256=facts['build_profile_sha256'],
            operator=operator, boundary=facts['boundary'], native_signature=facts['native_signature'],
            layout_sha256=layout, dtype=facts['layout']['dtype'], format='ND', ndim=2,
            soc=facts['worker']['soc'], execution_mode='eager', tp=1, rank=0, logical_device=0)
        if wrong_model:
            profile = replace(profile, model_class='another.Model')
        x = torch.ones(2, 4)
        with tempfile.TemporaryDirectory() as directory, patch(
            'fault_injection.environment.inspect_environment', return_value=environment
        ):
            observer = QualifiedFixture(directory, operator, facts, corrupt=corrupt, corrupt_fault=corrupt_fault)
            factory = lambda actual: observer
            replay = lambda actual: actual(x)
            def bind():
                register_runtime(fault, factory, observer_factory=factory, profiles=(profile,), replay=replay,
                                 first_live_validation=first_live)

            bind()
            try:
                with torch.inference_mode():
                    healthy = model(x)
                    backend = ModelFaultBackend(model)
                    if replace_binding:
                        backend.register(FaultConfig(fault))
                        unregister_runtime(fault, factory)
                        bind()
                        with self.assertRaisesRegex(UnsupportedCall, 'binding changed'):
                            backend.enable()
                        self.assertFalse(observer.modes)
                    elif (corrupt or wrong_model or omit_identity or len(list(model.children())) == 2
                            or (level == 'CPU_VERIFIED' and not first_live)):
                        with self.assertRaises(UnsupportedCall):
                            backend.inject(fault)
                        self.assertIn(backend.query()['state'], ('ENABLE_FAILED', 'REGISTERED_DISABLED'))
                        self.assertNotIn(3, observer.modes)
                        self.assertTrue(backend.query()['cleanup_confirmed'])
                    elif baseline_only:
                        with attach(model, fault, arm='baseline') as session:
                            self.assertTrue(torch.equal(model(x), healthy))
                            self.assertFalse(session.report()['qualification']['fault_admitted'])
                        self.assertEqual(observer.modes, admission_modes)
                    else:
                        result = backend.inject(fault)
                        self.assertEqual(result['session']['qualification']['status'], 'LOSSLESS_VERIFIED')
                        self.assertEqual(observer.modes, admission_modes)
                        candidates = result['session']['qualification']['discovery']['candidate_set']
                        self.assertEqual(candidates['eligible'], 1)
                        self.assertFalse(candidates['execution_authorized'])
                        if drift:
                            facts['cache_lifecycle'] = 'fixture:reused-cache-address'
                            with self.assertRaisesRegex(UnsupportedCall, 'cache'):
                                model(x)
                            self.assertEqual(observer.modes, admission_modes)
                        elif corrupt_fault:
                            with self.assertRaises(UnsupportedCall):
                                model(x)
                            self.assertEqual(observer.modes, admission_modes + [3])
                            self.assertTrue(torch.equal(model(x), healthy))
                            with self.assertRaises(UnsupportedCall):
                                backend.assert_coverage()
                            self.assertEqual(observer.modes, admission_modes + [3])
                        else:
                            self.assertFalse(torch.equal(model(x), healthy))
                            self.assertEqual(observer.modes, admission_modes + [3])
                            self.assertTrue(torch.equal(model(x), healthy))
                            backend.assert_coverage()
                    self.assertTrue(backend.disable()['cleanup_confirmed'])
                    self.assertTrue(torch.equal(model(x), healthy))
                    self.assertFalse(model._forward_hooks)
            finally:
                unregister_runtime(fault, factory)

    def test_inject_only_mode_qualifies_each_native_operator_and_recovers(self):
        for fault, operator in (('20.buffer_substitute', 'aclnnQuantMatmulWeightNz'),
                                ('20.buffer_substitute', 'aclnnMatmulWeightNz')):
            with self.subTest(fault=fault, operator=operator):
                self.qualified(fault, operator, torch.nn.Sequential(torch.nn.Linear(4, 4)).eval())

    def test_fault39_cannot_use_the_scoped_fault20_registry(self):
        with self.assertRaisesRegex(ValueError, 'Fault20'):
            self.qualified('39.matmul_range', 'aclnnMatmul',
                           torch.nn.Sequential(torch.nn.Linear(4, 4)).eval())

    def test_refuses_stale_binding_unproven_model_or_changed_fault_receipt(self):
        for kwargs in ({'replace_binding': True}, {'omit_identity': True},
                       {'corrupt_fault': {'input_sha256_before': ['9' * 64, '8' * 64],
                                          'input_sha256_after': ['9' * 64, '8' * 64]}}):
            with self.subTest(kwargs=kwargs):
                self.qualified('20.buffer_substitute', 'aclnnMatmulWeightNz',
                               torch.nn.Sequential(torch.nn.Linear(4, 4)).eval(), **kwargs)

    def test_inject_refuses_ambiguity_wrong_model_and_native_lossless_mismatch(self):
        fault, operator = '20.buffer_substitute', 'aclnnQuantMatmulWeightNz'
        for kwargs in ({'corrupt': {'output_sha256': 'f' * 64}},
                       {'corrupt': {'input_sha256_after': ['f' * 64]}},
                       {'corrupt': {'device_copy': False}}, {'corrupt': {'replaced_formals': ['self']}},
                       {'corrupt': {'temporary_readback_verified': False}}, {'wrong_model': True}):
            with self.subTest(kwargs=kwargs):
                self.qualified(fault, operator, torch.nn.Sequential(torch.nn.Linear(4, 4)).eval(), **kwargs)
        self.qualified(fault, operator, torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.Linear(4, 4)).eval())
        self.qualified(fault, operator, torch.nn.Sequential(torch.nn.Linear(4, 4)).eval(), drift=True)

    def test_first_device_validation_is_explicit_and_baseline_never_arms(self):
        fault, operator = '20.buffer_substitute', 'aclnnQuantMatmulWeightNz'
        for kwargs in ({'level': 'CPU_VERIFIED'}, {'level': 'CPU_VERIFIED', 'first_live': True},
                       {'baseline_only': True}):
            with self.subTest(kwargs=kwargs):
                self.qualified(fault, operator, torch.nn.Sequential(torch.nn.Linear(4, 4)).eval(), **kwargs)
