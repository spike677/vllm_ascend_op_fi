"""Real CPU module hooks + explicit mock native receipts; not NPU acceptance."""
import copy
import hashlib
import os
import sys
from types import ModuleType, SimpleNamespace as NS
import unittest
from unittest.mock import patch

import torch

from fault_injection import serving_qualification as sq
from fault_injection.deployments.matmul import CtypesMatmulRuntime, _REQUIRED_BUNDLE
from fault_injection.native import NativeCall, _LEASE, _STACK
from demoncat.runtime.vllm_ascend.worker_extension import serving20_qmm as qmm
from demoncat.runtime.vllm_ascend.worker_extension.providers import fault20
try:
    from . import test_serving_session_lifecycle as life
    from . import test_serving20_qualification as qualification
except ImportError:
    import test_serving_session_lifecycle as life
    import test_serving20_qualification as qualification


class CpuEmbedding(torch.nn.Module):
    def forward(self, input_):
        return (input_[:, None].float().repeat(1, 4) + torch.arange(4)).to(torch.bfloat16)


class CpuNorm(torch.nn.Module):
    def forward(self, x):
        return x.clone()


class AscendW8A8DynamicLinearMethod:
    """CPU fixture named explicitly to exercise reviewed method-chain discovery."""
    def apply(self, layer, x):
        import torch_npu
        quant, scale = torch_npu.npu_dynamic_quant(x)
        return torch_npu.npu_quant_matmul(quant, layer.weight, layer.weight_scale,
                                        pertoken_scale=scale, output_dtype=x.dtype)


class CpuProjection(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(4, 4, dtype=torch.int8), requires_grad=False)
        self.weight_scale = torch.nn.Parameter(torch.ones(4), requires_grad=False)
        self.quant_method = AscendW8A8DynamicLinearMethod()
        self.tp_size, self.tp_rank = 1, 0
        self.input_size_per_partition = self.output_size_per_partition = 4

    def forward(self, x):
        return self.quant_method.apply(self, x)


class CpuQmmModel(life.CpuModel):
    def __init__(self):
        super().__init__()
        self.embedding = CpuEmbedding()
        self.normalizer = CpuNorm()
        self.intermediate = CpuProjection()
        self.unused_candidate = CpuProjection()
        with torch.no_grad():
            self.head.weight.fill_(0.25)

    def forward(self, input_ids, positions, inputs_embeds=None, intermediate_tensors=None):
        return self.intermediate(self.normalizer(self.embedding(input_ids)))


class CpuQmmRuntime(qualification.CpuRuntime):
    operator = 'aclnnQuantMatmulWeightNz'

    def execute_scoped(self, call, original, mode):
        result, receipt = super().execute_scoped(call, original, mode)
        receipt['input_sha256_before'] = ['b' * 64] * 4
        receipt['input_sha256_after'] = ['b' * 64] * 4
        return result, receipt

    def emit(self, transaction):
        self.serial += 1
        formals = []
        for slot, (name, tensor) in enumerate(transaction['formals'].items()):
            formals.append(dict(formal=name, present=True, run_slot=slot,
                storage_bytes=tensor['storage_bytes'], storage_addr=tensor['storage_addr'],
                run_address=tensor['storage_addr'], view_offset=tensor['storage_offset'],
                view_shape=tensor['shape'], strides=tensor['stride'],
                element_size=tensor['element_size'], format=tensor['actual_format'],
                dtype={'torch.int8': 2, 'torch.float32': 0, 'torch.bfloat16': 27}[tensor['dtype']]))
        left = formals[0]
        for name in ('y_scale', 'x1_offset', 'x2_offset', 'y_offset', 'bias'):
            present = name in ('y_scale', 'x1_offset', 'y_offset')
            formals.append(dict(formal=name, present=present, run_slot=-1, run_address=0, storage_addr=0,
                raw_addr=0, data_addr=0, storage_bytes=0, view_bytes=0, storage_elements=0, numel=0, view_offset=0,
                dtype=27 if present else 0, format=2 if present else 0, element_size=2 if present else 0,
                contiguous=present, view_shape=[0] if present else [], storage_shape=[0] if present else [],
                strides=[1] if present else []))
        formals.append(dict(formal='out', present=True, run_slot=7, storage_addr=333,
            run_address=333, storage_bytes=32, view_shape=[4, 4], dtype=27, element_size=2, format=2))
        packet = dict(version=1, operator=self.operator, boundary='op_exec_cache_run',
            native_signature='CPU_SIGNATURE_NOT_NATIVE', cache_identity='CPU-QMM-cache', cache_lifecycle='1',
            candidate_count=1, build_profile_sha256=self.build_profile, runtime_identity='MATCHED_BUILD',
            layout=dict(dtype='INT8', format='ND', shape=left['view_shape'],
                        stride=left['strides'], offset=left['view_offset']),
            slots=[dict(formal=row['formal'], slot=row['run_slot'], storage_bytes=row['storage_bytes'])
                   for row in formals if row['storage_bytes'] > 0],
            current_typed_call=dict(source='typed_gws_execute_scope', executor_bound=True,
                formal_params_current=True, cached_layout_used=False, gws_token=self.serial, formals=formals))
        if self.packet_edit:
            self.packet_edit(packet)
        owner = _STACK.get()[-1][1]
        call_id = 'cpu-qmm-call-' + str(self.serial)
        proofs = CtypesMatmulRuntime._proofs(self, call_id, owner, packet, self.directory)
        call = NativeCall(owner, call_id, self.operator, proofs=proofs)
        def original():
            self.original_calls += 1
            return 0
        return self.callback(call, original)


class Serving20QmmTests(unittest.TestCase):
    snapshot = life.ServingSessionLifecycleTests.snapshot
    root = life.ServingSessionLifecycleTests.root
    forward = life.ServingSessionLifecycleTests.forward

    def setUp(self):
        life.ServingSessionLifecycleTests.setUp(self)
        self.backends = []
        self.enterContext(patch.dict(os.environ, {'DEMONCAT_F20_REQUIRE_TYPED_GWS': '0'}))
        environment = dict(pid=os.getpid(), maps_status='OBSERVED', machine='CPU-fixture', python='fixture',
            packages={}, torch_cxx11_abi=False, driver={'running_module_versions': {}, 'installation': None},
            loaded_libraries=[dict(name=name, sha256=hashlib.sha256(name.encode()).hexdigest(),
                elf_build_id='CPU', status='MAPPED_FILE_VERIFIED') for name in _REQUIRED_BUNDLE])
        self.enterContext(patch.object(sq, 'inspect_environment', side_effect=lambda **kw: copy.deepcopy(environment)))
        modules = {}
        for name in ('vllm.model_executor.layers.vocab_parallel_embedding', 'vllm.model_executor.layers.layernorm'):
            modules[name] = ModuleType(name)
        modules['vllm.model_executor.layers.vocab_parallel_embedding'].VocabParallelEmbedding = CpuEmbedding
        modules['vllm.model_executor.layers.vocab_parallel_embedding'].ParallelLMHead = torch.nn.Linear
        modules['vllm.model_executor.layers.layernorm'].RMSNorm = CpuNorm
        self.enterContext(patch.dict(sys.modules, modules))

    def tearDown(self):
        for session in self.sessions:
            session._restore_apis()
            session._restore_methods()
        life.ServingSessionLifecycleTests.tearDown(self)
        for backend in self.backends:
            if backend.locked:
                _LEASE.release()
                backend.locked = False

    def make_session(self, *, baseline_control=False, collector=None, start=True):
        from fault_injection.quantization import _discover
        model = CpuQmmModel().eval()
        runner = NS(model=model, vllm_config=NS(model_config=NS(hf_config=NS(vocab_size=3))))
        worker = NS(model_runner=runner, rank=0)
        folder = self.evidence_root / ('qmm-' + str(len(self.sessions)))
        folder.mkdir(parents=True)
        runtime = CpuQmmRuntime(folder)
        candidates, _ = _discover(model, (1,))
        backend = sq.ProjectionQualifiedNativeSession(model, candidates, runtime, owner='CPU-owner', generation=1,
            reviewed_manifest_path=qualification.Serving20QualificationTests.manifest(self, folder),
            first_live_validation=True, evidence_dir=folder / 'qualification', baseline_control=baseline_control)
        self.backends.append(backend)
        session = qmm.Serving20QmmSession(worker, model, backend, runtime, 'CPU-owner', candidates=candidates,
            prompt_token_ids=[1, 2, 3, 4], evidence_dir=folder / 'runtime', collector=collector)
        self.sessions.append(session)
        import torch_npu
        torch_npu.npu_dynamic_quant = lambda x: (x.round().to(torch.int8), torch.ones(x.shape[0]))
        def cpu_qmm(x1, x2, scale, *, pertoken_scale, output_dtype):
            before = len(runtime.modes)
            if not runtime.paused:
                runtime.emit(backend.current['transaction'])
            mode = runtime.modes[-1] if len(runtime.modes) != before else 0
            # Only an explicit CPU fixture models the native fault's effect.
            value = torch.zeros_like(x1) if mode == 3 else x1
            return (value.float() @ x2.float()).to(output_dtype)
        torch_npu.npu_quant_matmul = cpu_qmm
        if start:
            backend.start()
            session.install()
        return session

    def phase(self, session, index):
        self.actual_request_id = 'cpu-actual-qmm-' + str(index)
        session.expect(self.actual_request_id)
        with torch.inference_mode():
            result = self.forward(session)
        session.collect()
        return result

    def test_eight_real_cpu_hook_phases_have_intermediate_scope_and_complete_logits(self):
        collector = life.MockCollector()
        collector.wrap_forwards = True
        session = self.make_session(collector=collector)
        outputs = [self.phase(session, i) for i in range(8)]
        backend = session.backend
        self.assertIs(backend.bound_module, session.model.intermediate)
        self.assertIsNone(backend.processor)
        self.assertIsNone(backend.head)
        self.assertEqual(backend.runtime.modes, [1, 2, 1, 3, 1])
        self.assertEqual([row['native_calls'] for row in backend.rows], [1] * 7 + [0])
        self.assertTrue(all(event['verified_empty_optional_formals'] == ['x1_offset', 'y_offset', 'y_scale']
                            for event in backend.events))
        self.assertTrue(torch.equal(outputs[0], outputs[-1]))
        self.assertFalse(torch.equal(outputs[0], outputs[5]))
        self.assertTrue(all(row['transaction']['root_forward_active'] for row in backend.rows))
        self.assertTrue(all('head_argument_object_id' not in row['transaction'] for row in backend.rows))
        self.assertTrue(all(row['projection_transaction']['native_x1_equals_float_input_storage'] is False
                            and row['root_returned'] for row in session.rows))
        self.assertEqual((collector.starts, collector.stops, collector.steps), (8, 8, 8))
        self.assertTrue(backend.query()['qualification']['recovery_verified'])
        self.assertEqual(session.selection_report['structural_candidates'], 2)
        self.assertFalse(session.selection_report['all_eligible_covered'])
        session.close()
        backend.close()

    def test_public_provider_accepts_only_reviewed_projection_pair(self):
        session = self.make_session(start=False, baseline_control=True)
        identity = dict(rank=0, logical_device=0, pid=os.getpid(), worker_start_identity='CPU-worker',
                        model_instance_id='CPU-model')
        provider = fault20.Fault20Provider(session.worker, session.model, identity)
        ticket = dict(owner='CPU-owner', generation=1)
        provider.bind(**ticket)
        provider.runtime = session.native
        binding = fault20.Fault20ServingReplay('a' * 64, lambda *args: session)
        with (patch.object(fault20, 'qualified_binding', return_value=None),
              patch.object(fault20, 'resolve_runtime', return_value=binding),
              patch.object(provider, '_worker_facts', return_value=session.native.worker)):
            provider.dispatch('register', {}, **ticket)
        provider.dispatch('check', {}, **ticket)
        provider.dispatch('execute', {}, **ticket)
        for index in range(8):
            self.actual_request_id = 'cpu-provider-qmm-' + str(index)
            provider.dispatch('advance', dict(request_id=self.actual_request_id), **ticket)
            with torch.inference_mode():
                self.forward(session)
            provider.query(**ticket)
        result = provider.dispatch('finish', {}, **ticket)
        self.assertEqual(result['qualification']['status'], 'BASELINE_CONTROL_COMPLETED')
        self.assertFalse(result['native_registered'])
        self.assertFalse(result['qualification']['fault_native_attempted'])

    def test_native_scale_address_mismatch_refuses_before_original_execution(self):
        session = self.make_session()
        def edit(packet):
            row = next(row for row in packet['current_typed_call']['formals'] if row['formal'] == 'x1_scale')
            row['storage_addr'] += 4
            row['run_address'] += 4
        session.native.packet_edit = edit
        with self.assertRaisesRegex(Exception, 'ACTUAL_QUANT_FORMAL_STORAGE_OR_GEOMETRY_MISMATCH:x1_scale'):
            self.phase(session, 0)
        self.assertEqual(session.native.original_calls, 0)
        self.assertTrue(session.native.paused)
        session.close()
        session.backend.close()

    def test_cloned_quantized_api_input_is_not_accepted_as_real_quant_output(self):
        session = self.make_session()
        # Exercise the lineage guard directly within an actual selected request.
        old = session._dynamic_quant
        def replace(original_api, args, kwargs):
            result = old(original_api, args, kwargs)
            return result[0].clone(), result[1]
        session._dynamic_quant = replace
        with self.assertRaisesRegex(Exception, 'QMM_NOT_REAL_DYNAMIC_QUANT_OUTPUT'):
            self.phase(session, 0)
        self.assertEqual(session.native.original_calls, 0)
        session.close()
        session.backend.close()

    def test_missing_natural_projection_does_not_qualify_float_head(self):
        session = self.make_session()
        # Real embedding executes, but no normalizer/projection QMM consumes it.
        session.model.intermediate.forward = lambda x: x
        with self.assertRaisesRegex(Exception, 'NATURAL_PROJECTION_QMM_NOT_OBSERVED'):
            self.phase(session, 0)
        self.assertIsNone(session.backend.bound_module)
        self.assertEqual(session.native.original_calls, 0)
        session.close()
        session.backend.close()

    def test_unknown_qmm_signatures_and_missing_per_token_scale_refuse(self):
        with self.assertRaisesRegex(Exception, 'UNREVIEWED_PUBLIC_QMM_SIGNATURE'):
            qmm.bind_qmm_arguments((1, 2, 3), {'invented': 4})
        with self.assertRaisesRegex(Exception, 'DYNAMIC_QMM_SCALE_CONTRACT_REQUIRED'):
            qmm.bind_qmm_arguments((1, 2, 3), {})

    def test_empty_optional_formal_cannot_hide_storage_or_change_lowering(self):
        for field, value, reason in (
                ('storage_addr', 123, 'OPTIONAL_QMM_FORMAL_HAS_UNBOUND_STORAGE'),
                ('view_shape', [1], 'UNREVIEWED_EMPTY_QMM_FORMAL'),
                ('dtype', 0, 'UNREVIEWED_EMPTY_QMM_FORMAL'),
                ('run_slot', 4, 'OPTIONAL_QMM_FORMAL_HAS_UNBOUND_STORAGE')):
            session = self.make_session()
            def edit(packet, field=field, value=value):
                row = next(row for row in packet['current_typed_call']['formals'] if row['formal'] == 'y_scale')
                row[field] = value
            session.native.packet_edit = edit
            with self.subTest(field=field), self.assertRaisesRegex(Exception, reason):
                self.phase(session, len(self.sessions))
            self.assertEqual(session.native.original_calls, 0)
            event = session.backend.events[-1]
            self.assertEqual(event['formal_verification'], 'PENDING')
            self.assertIn('current_typed_call', event)
            self.assertIn('observed_transaction', event)
            session.close()
            session.backend.close()

    def test_unexpected_optional_formal_name_or_nonempty_slot_still_refuses(self):
        for change in ('name', 'storage'):
            session = self.make_session()
            def edit(packet, change=change):
                row = next(row for row in packet['current_typed_call']['formals'] if row['formal'] == 'y_scale')
                if change == 'name':
                    row['formal'] = 'unknown_scale'
                else:
                    row['storage_bytes'] = 2
            session.native.packet_edit = edit
            with self.subTest(change=change), self.assertRaisesRegex(Exception, 'CURRENT_FORMAL_SLOT_SET_CHANGED'):
                self.phase(session, len(self.sessions))
            self.assertEqual(session.native.original_calls, 0)
            session.close()
            session.backend.close()


if __name__ == '__main__':
    unittest.main(verbosity=2)
