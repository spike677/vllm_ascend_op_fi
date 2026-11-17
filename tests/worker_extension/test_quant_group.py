"""Actual CPU projection calls with explicit native mocks, never NPU evidence."""
import copy
import hashlib
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace as NS
from types import MethodType
import unittest
from unittest.mock import patch

import torch

from fault_injection.deployments.matmul import _REQUIRED_BUNDLE
from fault_injection.native import _LEASE, _STACK, _read_proofs
from fault_injection.quantization import _discover
from demoncat.runtime.vllm_ascend.worker_extension.quant_group import QuantGroupBackend, digest
from demoncat.runtime.vllm_ascend.worker_extension.quant_inventory import PublicQuantInventory
from demoncat.runtime.vllm_ascend.worker_extension.serving import complete_tensor
import test_serving20_qmm as qmm
from metadata_fixture import metadata_report


class TwoProjectionModel(qmm.CpuQmmModel):
    def __init__(self):
        super().__init__()
        del self.unused_candidate
        self.second = qmm.CpuProjection()

    def forward(self, input_ids, positions, inputs_embeds=None, intermediate_tensors=None):
        return self.second(super().forward(input_ids, positions, inputs_embeds, intermediate_tensors))


class GroupRuntimeMock(qmm.CpuQmmRuntime):
    def __init__(self, folder):
        super().__init__(folder)
        self.worker.update(worker_id='CPU-worker-' + str(folder), model_instance_id='CPU-model-' + str(folder))
        self.native_inventory = {}
        self.group_targets = {}
        self.sealed = False
        self.mode = 0
        self.last_error = None
        self.skip_callback = False
        self.raw_calls = 0

    def metadata_arrays(self):
        return metadata_report(self.raw_calls, self.build_profile)

    def configure_group_lifecycle(self, **kwargs):
        self.configuration = kwargs
        self.group_targets.clear()
        self.sealed = False

    def install_group_observer(self, callback, owner_reader):
        self.group_callback, self.owner_reader = callback, owner_reader
        self.paused = True
        self.callback = self._callback
        def remove():
            assert self.paused
            self.callback = None
            self.group_targets.clear()
            self.sealed = False
        return remove

    def _callback(self, call, original):
        _, details = _read_proofs(call, self.mechanism)
        packet = dict(details['runtime_call'])
        packet.pop('worker')
        stable = {key: value for key, value in packet.items() if key != 'current_typed_call'}
        key = digest(stable)
        row = self.native_inventory.setdefault(key, dict(identity_sha256=key, native=stable, observations=0))
        row['observations'] += 1
        self.raw_calls += 1  # Independent of owner, mode, admission, and hits.
        self.mode = 0
        def once():
            status = original()
            q, weight, scale, token_scale, dtype = self.api_inputs
            actual = (q.to(torch.int16) // 16 * 16).to(torch.int8) if self.mode == 3 else q.clone() if self.mode == 2 else q
            self.actual_input = actual
            self.actual_output = ((actual.float() @ weight.float()) * scale[None, :] * token_scale[:, None]).to(dtype)
            return status
        if self.paused or self.skip_callback:
            return once()
        return self.group_callback(call, once, packet)

    def independent_inventory(self):
        return copy.deepcopy(dict(complete=True, overflow_calls=0, eligible_run_calls=self.raw_calls,
            all_run_calls=self.raw_calls, non_target_runs=0, rejected_candidate_runs=0,
            targets=list(self.native_inventory.values())))

    def group_state(self):
        return dict(targets=sorted(self.group_targets), target_count=len(self.group_targets), sealed=self.sealed,
                    inflight_calls=0, device_bytes=0, host_bytes=0)

    def seal_group(self, target_keys):
        assert set(target_keys) == set(self.group_targets)
        assert all(row['lossless'] and row['off'] for row in self.group_targets.values())
        self.sealed = True
        return self.group_state()

    def execute_group(self, call, original, mode, target_key):
        row = self.group_targets.setdefault(target_key, dict(lossless=False, off=False, faults=0))
        if mode == 3:
            assert self.sealed and row['faults'] < self.configuration['max_faults']
            row['faults'] += 1
        if mode == 1 and row['lossless']:
            row['off'] = True
        if mode == 2:
            row['lossless'] = True
        before = [complete_tensor(value)[1] for value in self.api_inputs[:4]]
        self.mode = mode
        result = original()
        receipt = dict(mode=mode, original_calls=1, native_status=0, replacements=int(mode != 1),
            resources_released=True, temporary_readback_verified=mode != 1,
            replaced_formals=['x1'] if mode != 1 else [], device_copy=mode == 2,
            changed_slots=int(mode == 3 and not torch.equal(self.actual_input, self.api_inputs[0])),
            input_sha256_before=before, input_sha256_after=[complete_tensor(value)[1] for value in self.api_inputs[:4]],
            output_sha256=complete_tensor(self.actual_output)[1], target_key=target_key,
            evidence_kind='EXPLICIT_CPU_NATIVE_MOCK_NOT_NPU')
        return result, receipt


class QuantGroupTests(unittest.TestCase):
    def setUp(self):
        self.folder = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.backends = []
        self.enterContext(patch.dict(sys.modules, {'torch_npu': NS(get_npu_format=lambda tensor: 2)}))
        environment = dict(pid=os.getpid(), maps_status='OBSERVED', machine='CPU-fixture', python='fixture',
            packages={}, torch_cxx11_abi=False, driver={'running_module_versions': {}, 'installation': None},
            loaded_libraries=[dict(name=name, sha256=hashlib.sha256(name.encode()).hexdigest(),
                elf_build_id='CPU', status='MAPPED_FILE_VERIFIED') for name in _REQUIRED_BUNDLE])
        self.enterContext(patch('fault_injection.environment.inspect_environment', side_effect=lambda **kw: copy.deepcopy(environment)))

    def tearDown(self):
        for backend in self.backends:
            # CPU fixture disposal is not reported as a production OFF proof.
            PublicQuantInventory.close(backend)
            if backend.locked:
                _LEASE.release()
                backend.locked = False

    def backend(self, *, model=None, runtime=None, proof=None, unused=False):
        model = model or TwoProjectionModel().eval()
        if unused:
            model.unused_candidate = qmm.CpuProjection()
        folder = self.folder / str(len(self.backends))
        folder.mkdir()
        runtime = runtime or GroupRuntimeMock(folder)
        identity = dict(worker_start_identity=runtime.worker['worker_id'],
                        model_instance_id=runtime.worker['model_instance_id'], rank=0, logical_device=0)
        backend = QuantGroupBackend(model, _discover(model, (1,))[0], runtime=runtime, worker_identity=identity,
            owner='CPU-group-owner', generation=1 + len(self.backends), folder=folder / 'group',
            required_libraries=_REQUIRED_BUNDLE, trigger='every_match' if proof else 'once', c2_proof=proof,
            evidence_kind='cpu_contract')
        self.backends.append(backend)
        import torch_npu
        def dynamic_quant(x):
            scale = x.float().abs().amax(dim=1).clamp_min(1) / 127
            return (x.float() / scale[:, None]).round().to(torch.int8), scale
        def quant_matmul(x1, x2, scale, *, pertoken_scale, output_dtype):
            runtime.api_inputs = x1, x2, scale, pertoken_scale, output_dtype
            token = _STACK.set(((None, backend.frame['module']),))
            try:
                runtime.emit(backend.frame['transaction'])
            finally:
                _STACK.reset(token)
            return runtime.actual_output
        torch_npu.npu_dynamic_quant = dynamic_quant
        torch_npu.npu_quant_matmul = quant_matmul
        backend.install()
        backend.start()
        return backend

    def phase(self, backend):
        request = 'cpu-forward-' + str(len(backend.rows))
        with torch.inference_mode():
            backend.begin(request, 'a' * 64, request_ids=(request,))
            hidden = backend.model(torch.arange(1, 5), torch.arange(4))
            output = backend.model.processor(backend.model.head, hidden[-1:])
            _, sha, finite = complete_tensor(output)
            backend.end(sha, model_output_finite=finite)

    def complete(self, backend):
        while backend.phase != 'complete':
            self.phase(backend)
        PublicQuantInventory.close(backend)
        backend.finalize()
        backend.close()
        return backend.coverage.report()

    def test_two_actual_projections_same_native_cache_have_distinct_leases_and_off(self):
        backend = self.backend()
        report = self.complete(backend)
        self.assertTrue(report['contract_satisfied'], report['rejection_reasons'])
        self.assertFalse(report['real_worker_evidence'])
        self.assertEqual(report['planned_targets'], 2)
        self.assertEqual(report['hits'], 2)
        self.assertEqual(report['observed_eligible_calls'], 2)
        self.assertEqual(len({row['native_identity'] for row in backend.targets.values()}), 1)
        self.assertEqual(len({row['key'] for row in backend.targets.values()}), 2)
        self.assertEqual(backend.runtime.original_calls, len(backend.phases) * 2)
        self.assertIsNone(backend.runtime.callback)
        self.assertEqual(backend.runtime.group_state()['target_count'], 0)
        self.assertEqual(backend.baseline_output, backend.last_output)

    def test_group_cannot_qualify_without_actual_metadata_array_activity(self):
        runtime = GroupRuntimeMock(self.folder)
        runtime.metadata_arrays = lambda: metadata_report(0, runtime.build_profile)
        backend = self.backend(runtime=runtime)
        with self.assertRaisesRegex(RuntimeError, 'GROUP_COVERAGE_OR_RECOVERY_INCOMPLETE'):
            self.complete(backend)
        self.assertIsNone(backend.proof)
        self.assertFalse(getattr(backend, 'recovered', False))

    def test_every_match_requires_live_proof_and_counts_repeated_calls(self):
        first = self.backend()
        self.complete(first)
        second = self.backend(model=first.model, runtime=first.runtime, proof=first.proof)
        report = self.complete(second)
        self.assertTrue(report['contract_satisfied'], report['rejection_reasons'])
        self.assertEqual(report['trigger'], 'every_match')
        self.assertEqual(report['hits'], 4)
        self.assertFalse(report['real_worker_evidence'])

    def test_missing_native_callback_cannot_disappear_from_independent_denominator(self):
        backend = self.backend()
        backend.runtime.skip_callback = True
        with self.assertRaisesRegex(RuntimeError, 'INDEPENDENT_NATIVE_QMM_OUTSIDE_CALLBACK'):
            self.phase(backend)
        self.assertEqual(backend.runtime.raw_calls, 2)
        self.assertIsNone(backend.proof)

    def test_changed_cache_epoch_between_healthy_requests_refuses_group(self):
        backend = self.backend()
        self.phase(backend)
        backend.runtime.packet_edit = lambda packet: packet.update(cache_lifecycle='2')
        with self.assertRaisesRegex(RuntimeError, 'CACHE_LIFETIME_CHANGED'):
            self.phase(backend)
        self.assertIsNone(backend.proof)

    def test_unused_structural_candidate_is_not_promoted_to_all_eligible(self):
        backend = self.backend(unused=True)
        while backend.phase != 'fault':
            self.phase(backend)
        with self.assertRaisesRegex(Exception, 'UNRESOLVED_CANDIDATE_DENOMINATOR'):
            self.phase(backend)
        self.assertIsNone(backend.proof)
        self.assertFalse(backend.fault_started)

    def test_projection_return_replaced_after_native_call_does_not_qualify(self):
        model = TwoProjectionModel().eval()
        method = model.intermediate.quant_method
        original = method.apply
        def apply(_method, module, x):
            return original(module, x).clone()
        method.apply = MethodType(apply, method)
        backend = self.backend(model=model)
        with self.assertRaisesRegex(RuntimeError, 'COMPLETE_PUBLIC_PROJECTION_LINEAGE_REJECTED'):
            self.phase(backend)
        self.assertEqual(backend.runtime.raw_calls, 2)
        self.assertIsNone(backend.proof)

    def test_actual_root_and_logits_hooks_publish_only_finished_group_proof(self):
        import test_serving_session_lifecycle as life
        from demoncat.runtime.vllm_ascend.worker_extension.serving20_coverage import Serving20CoverageSession
        fixture = life.ServingSessionLifecycleTests()
        fixture.setUp()
        try:
            backend = self.backend()
            PublicQuantInventory.close(backend)
            runner = NS(model=backend.model, vllm_config=NS(model_config=NS(hf_config=NS(vocab_size=3))))
            worker = NS(model_runner=runner, rank=0)
            session = Serving20CoverageSession(worker, backend.model, backend, backend.runtime, backend.owner,
                prompt_token_ids=[1, 2, 3, 4], evidence_dir=fixture.evidence_root / 'group')
            fixture.sessions.append(session)
            session.install()
            for index in range(len(backend.phases)):
                fixture.actual_request_id = 'actual-cpu-hook-request-' + str(index)
                session.expect(fixture.actual_request_id)
                with torch.inference_mode():
                    fixture.forward(session)
            self.assertFalse(hasattr(worker, '_demoncat20_live_c2_proof'))
            session.close()
            self.assertIs(worker._demoncat20_live_c2_proof, backend.proof)
            self.assertTrue(backend.coverage.report()['contract_satisfied'])
            self.assertFalse(backend.coverage.report()['real_worker_evidence'])
            self.assertEqual(backend.owned_hooks, 0)
            self.assertEqual(session.report()['owned_hooks'], 0)
        finally:
            fixture.tearDown()
            fixture.doCleanups()


if __name__ == '__main__':
    unittest.main()
