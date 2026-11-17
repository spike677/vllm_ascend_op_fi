"""Actual CPU tensor calls with explicit mock native receipts, never NPU evidence."""
from contextlib import contextmanager
import copy
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace as NS

import torch

from demoncat_fault39.coverage_adapter import (MatmulCoverageBackend, digest, native_counts,
                                              native_delta, reconcile_discovery)
from demoncat_fault39.inspector import summarize_values
from demoncat_fault39.schema import Fault39Config


class NativeMock:
    """Simulates ABI observations to test orchestration, not fault semantics."""
    def __init__(self):
        self.counts = {}
        self.generation = 0
        self.owner = ''
        self.enabled = False
        self.hits = 0
        self.context_owner = None
        self.request = None
        self.selected = None
        self.mode = 'discover'
        self.mask = 3
        self.observation = {}
        self.configured = []
        self.calls = 0
        self.missing_hit = False
        self.stale_epoch = False

    def snapshot(self):
        return dict(runtime_identity='MATCHED_BUILD', runtime_loaded=True, runtime_ready=True,
            exact_cache_api_version=2, build_profile_sha256='b' * 64, owner=self.owner,
            generation=self.generation, enabled=self.enabled, inflight=0, leases=0,
            hits=self.hits, last_error=0, status='COMPLETED', typed_metadata_calls=2,
            interposed_run_calls=self.calls, mode={'discover': 1, 'observer': 4, 'lossless': 3, 'inject': 2}[self.mode],
            target_signature=self.selected[0] if self.selected else '',
            independent_observer_complete=True, independent_observer_max_bytes=512 * 1024**2,
            independent_eligible_runs=sum(self.counts.values()),
            independent_candidates=[dict(signature=key[0], rank=0, device=0,
                cache_identities=[dict(address=key[1], epoch=key[2], observations=count,
                                       registration_live=True)]) for key, count in self.counts.items()],
            last_observation=copy.deepcopy(self.observation))

    @contextmanager
    def context(self, owner, request_ids):
        if self.context_owner:
            raise RuntimeError('NESTED_MOCK_CONTEXT')
        self.context_owner, self.request = owner, request_ids[0]
        try:
            yield
        finally:
            self.context_owner = self.request = None

    def bind_serving_inputs(self, owner, values):
        assert owner == self.context_owner and len(values) == 13

    def configure(self, owner, config, rank, device, **kwargs):
        assert config.max_hits == 1
        self.owner = owner
        self.generation += 1
        self.enabled = True
        self.mode = 'discover'
        self.hits = 0
        return self.generation

    def resume_discovery(self, owner, generation):
        assert owner == self.owner and generation == self.generation
        self.enabled = True

    def configure_cache(self, owner, config, rank, device, **kwargs):
        key = kwargs['target_signature'], kwargs['cache_address'], kwargs['cache_epoch']
        assert owner == self.context_owner and config.max_hits == 1 and self.counts.get(key, 0) >= 2
        if self.stale_epoch:
            raise RuntimeError('MOCK_STALE_REGISTRATION_EPOCH')
        self.owner = owner
        self.generation += 1
        self.enabled = True
        self.hits = 0
        self.mode, self.mask, self.selected = kwargs['mode'], kwargs['input_mask'], key
        self.configured.append((key, self.generation, config.max_hits))
        return self.generation

    def _stop(self, owner, generation, clean):
        assert owner == self.owner and generation == self.generation
        self.enabled = False
        if clean:
            self.owner = ''
        return dict(status='DISABLED_CONFIRMED', native_code=0, owner=owner,
                    generation=generation, snapshot=self.snapshot())

    def disable(self, owner, generation, timeout):
        return self._stop(owner, generation, False)

    def clean(self, owner, generation, timeout):
        return self._stop(owner, generation, True)

    def execute(self, key):
        self.calls += 1
        self.counts[key] = self.counts.get(key, 0) + 1
        output = torch.mm(torch.ones(2, 4), torch.ones(4, 2))
        if not self.enabled or self.mode == 'discover':
            return output
        assert self.selected == key
        if self.missing_hit:
            return output
        fault = self.mode == 'inject'
        if fault:
            # A CPU-only fake effect; this is NOT the production Native fault.
            output = output + 1
        self.hits = 1
        self.enabled = False
        values = [4.0] * 8
        stats = summarize_values(values)
        tensors = [dict(original_address=100 + i, run_address=100 + i + (1000 if self.mode != 'observer' and self.mask & (1 << i) else 0),
                        shape=[2, 4] if i == 0 else [4, 2] if i == 1 else [2, 2]) for i in range(3)]
        self.observation = dict(owner=self.owner, generation=self.generation, hit=1,
            cache_address=key[1], cache_epoch=key[2], serving_live_inputs_verified=True,
            actual_request_ids_json='["' + self.request + '"]',
            original_storage_sha256_before=['a' * 64, 'c' * 64], original_storage_sha256_after=['a' * 64, 'c' * 64],
            output_storage_sha256=digest(output.tolist()), pre_sync_status=0, copy_status=0,
            run_status=0, post_sync_status=0, free_status=0,
            original_self=stats, original_mat2=stats, temporary_self=stats, temporary_mat2=stats,
            output=summarize_values([float(v) for v in output.flatten()]),
            changed=[8, 8] if fault else [0, 0], unchanged=[0, 0] if fault else [8, 8],
            output_source='post_run_device_readback', input_device_readback_verified=True,
            self_constant_positive_verified=fault, per_column_sign_counts_verified=fault,
            theoretical_output_exact_zero=fault, amplitude_equal=fault,
            proof=dict(api='aclnnMatmul', formal_order_proven=True, slots=[0, 1, 2],
                       input_mask=self.mask, generation=self.generation, hit=1, tensors=tensors))
        return output


class CoverageAdapterTests(unittest.TestCase):
    def test_independent_counter_includes_disabled_calls_and_refuses_loss(self):
        native = NativeMock()
        before = native.snapshot()
        native.execute(('same-shape', 100, 1))
        after = native.snapshot()
        self.assertEqual(native_delta(before, after), {('same-shape', 100, 1): 1})
        self.assertEqual(after['hits'], 0)
        bad = copy.deepcopy(after)
        bad['independent_eligible_runs'] += 1
        with self.assertRaisesRegex(RuntimeError, 'DENOMINATOR_INCOMPLETE'):
            native_counts(bad)
        bad = dict(after, independent_observer_complete=False)
        with self.assertRaisesRegex(RuntimeError, 'OVERFLOW'):
            native_counts(bad)
        with self.assertRaisesRegex(RuntimeError, 'COUNTER_RESET'):
            native_delta(after, before)

    def fixture(self, folder, native=None, proof=None, model=None):
        native = native or NativeMock()
        model = model or object()
        identity = dict(worker_start_identity='cpu-worker-' + str(id(model)),
                        model_instance_id='cpu-model-' + str(id(model)), rank=0, logical_device=0)
        backend = MatmulCoverageBackend(native, owner='cpu-coverage', worker_identity=identity,
            model=model, folder=folder, identity=lambda: 'actual-model', evidence_kind='cpu_contract',
            trigger='every_match' if proof else 'once', c2_proof=proof)
        backend.prepare()
        backend.register(Fault39Config())
        backend.start()
        return backend, native

    def phase(self, backend, native, index):
        backend.begin('cpu-request-' + str(index), 'd' * 64)
        outputs = []
        for module, address in ((1001, 100), (1002, 200)):
            site = module, 'aten.mm.default', 0
            row = dict(module_object_identity=module, module_path='projection' + str(module),
                       operator='aten.mm.default', tensor_signature='same-geometry',
                       input_storage_bytes=[16, 16], output_bytes=8)
            outputs.append(backend.dispatch(site, row, list(range(13)),
                lambda address=address: native.execute(('same-shape', address, 1))))
        backend.end(digest([output.tolist() for output in outputs]), model_output_finite=True)

    def test_two_same_shape_targets_all_once_native_leases_and_group_off(self):
        with tempfile.TemporaryDirectory() as folder:
            backend, native = self.fixture(folder)
            for index, _ in enumerate(backend.phases):
                self.phase(backend, native, index)
            backend.finalize()
            report = backend.coverage.report()
            self.assertTrue(report['contract_satisfied'], report['rejection_reasons'])
            self.assertFalse(report['real_worker_evidence'])
            self.assertEqual(report['planned_targets'], 2)
            self.assertEqual(report['hits'], 2)
            self.assertEqual(report['observed_eligible_calls'], 2)
            self.assertEqual(len({target['identity'] for target in backend.targets.values()}), 2)
            self.assertTrue(all(row[2] == 1 for row in native.configured))
            self.assertEqual(len({row[1] for row in native.configured}), len(native.configured))
            self.assertEqual(native.owner, '')
            self.assertEqual(report['recovery']['pending_leases'], [])
            self.assertIsNotNone(backend.proof)

    def test_unobserved_native_call_cannot_disappear_from_discovery(self):
        with tempfile.TemporaryDirectory() as folder:
            backend, native = self.fixture(folder)
            backend.begin('real-cpu-request', 'd' * 64)
            native.execute(('unwrapped-native', 500, 1))
            with self.assertRaisesRegex(RuntimeError, 'OUTSIDE_INDEPENDENT_PUBLIC_OBSERVER'):
                backend.end('a' * 64, model_output_finite=True)

    def test_c3_requires_live_c2_then_records_both_repeated_calls_per_target(self):
        with tempfile.TemporaryDirectory() as folder:
            first, native = self.fixture(Path(folder) / 'c2')
            for index in range(len(first.phases)):
                self.phase(first, native, index)
            first.finalize()
            second, native = self.fixture(Path(folder) / 'c3', native=native, proof=first.proof, model=first.model)
            for index in range(len(second.phases)):
                self.phase(second, native, index)
            second.finalize()
            report = second.coverage.report()
            self.assertTrue(report['contract_satisfied'], report['rejection_reasons'])
            self.assertEqual(report['hits'], 4)
            self.assertEqual(report['observed_eligible_calls'], 4)
            self.assertEqual(report['trigger'], 'every_match')
            self.assertEqual(report['generation'], first.logical_generation + 1)
            self.assertFalse(report['real_worker_evidence'])

    def test_stale_cache_or_missing_hit_fails_without_c2_proof(self):
        for field in ('stale_epoch', 'missing_hit'):
            with self.subTest(field=field), tempfile.TemporaryDirectory() as folder:
                backend, native = self.fixture(folder)
                self.phase(backend, native, 0)
                self.phase(backend, native, 1)
                setattr(native, field, True)
                with self.assertRaises(RuntimeError):
                    self.phase(backend, native, 2)
                self.assertIsNone(backend.proof)
                self.assertFalse(native.enabled)

    def test_discovery_rejects_unbound_inputs_and_reused_cache_address(self):
        row = dict(site=[1, 'aten.mm.default', 0], module_object_identity=1, module_path='p',
            operator='aten.mm.default', tensor_signature='shape', public_inputs_verified=True,
            native_delta=[dict(key=['sig', 12, 1], count=1)])
        changed = copy.deepcopy(row)
        changed['native_delta'][0]['key'][2] = 2
        with self.assertRaisesRegex(RuntimeError, 'LIFETIME_CHANGED'):
            reconcile_discovery([row], [changed])
        changed = dict(row, public_inputs_verified=False)
        with self.assertRaisesRegex(RuntimeError, 'NO_REVIEWED_PUBLIC_INPUTS'):
            reconcile_discovery([row], [changed])


try:
    from . import test_serving_session_lifecycle as life
except ImportError:
    import test_serving_session_lifecycle as life


class TwoProjectionModel(life.CpuModel):
    def __init__(self):
        super().__init__()
        self.projection_a = torch.nn.Linear(4, 4, bias=False, dtype=torch.bfloat16)
        self.projection_b = torch.nn.Linear(4, 4, bias=False, dtype=torch.bfloat16)

    def forward(self, input_ids, positions, inputs_embeds=None, intermediate_tensors=None):
        x = super().forward(input_ids, positions, inputs_embeds, intermediate_tensors)
        return self.projection_b(self.projection_a(x))


class ObserverSpy:
    def __init__(self):
        self.phase = 'discovery_1'
        self.requests = set()
        self.failure = None
        self.active = self.recovering = self.recovered = self.fault_started = False
        self.rows = []
        self.calls = []

    def begin(self, request_id, input_sha256):
        self.requests.add(request_id)
        self.active = True

    def dispatch(self, site, row, pair, original):
        self.calls.append(dict(site=site, row=row, pair=pair))
        return original()

    def end(self, output_sha256, *, model_output_finite):
        self.active = False
        self.phase = 'discovery_2'
        self.rows.append(dict(output=output_sha256))

    def fail(self, reason):
        self.failure = reason
        self.active = False

    def query(self):
        return {'qualification': {'status': 'CPU_OBSERVER_ONLY'}}


class WholeRequestHookTests(unittest.TestCase):
    setUp = life.ServingSessionLifecycleTests.setUp
    tearDown = life.ServingSessionLifecycleTests.tearDown
    snapshot = life.ServingSessionLifecycleTests.snapshot

    def session(self):
        from demoncat.runtime.vllm_ascend.worker_extension.serving39_coverage import Serving39CoverageSession
        model = TwoProjectionModel().eval()
        runner = NS(model=model, vllm_config=NS(model_config=NS(hf_config=NS(vocab_size=3))))
        worker = NS(model_runner=runner, rank=0)
        session = Serving39CoverageSession(worker, model, ObserverSpy(), object(), 'cpu-owner',
            prompt_token_ids=[1, 2, 3, 4], evidence_dir=self.evidence_root / 'whole-request')
        self.sessions.append(session)
        session.install()
        return session

    def test_actual_cpu_dispatch_observes_both_modules_and_real_logits(self):
        session = self.session()
        original_forward = session.model.forward
        session.expect(self.actual_request_id)
        hidden = life.ServingSessionLifecycleTests.root(self, session)
        output = session.processor(session.head, hidden[-1:].contiguous())
        mm = [row for row in session.backend.calls if row['row']['operator'] == 'aten.mm.default']
        self.assertEqual([row['row']['module_path'] for row in mm],
                         ['projection_a', 'projection_b', 'processor'])
        self.assertEqual(len({row['site'][0] for row in mm}), 3)
        self.assertTrue(all(row['pair'] is not None for row in mm))
        self.assertEqual(list(output.shape), [1, 3])
        self.assertIsNone(session.mode)
        self.assertEqual(session.module_stack, [])
        self.assertEqual(session.model.forward, original_forward)
        session.close()
        self.assertFalse(session.handles)
        self.assertEqual(session.backend.owned_hooks, 0)

    def test_real_cpu_forward_exception_removes_dispatch_mode(self):
        session = self.session()
        session.model.raise_forward = True
        session.expect(self.actual_request_id)
        with self.assertRaisesRegex(RuntimeError, 'CPU_ROOT_FORWARD_FAILED'):
            life.ServingSessionLifecycleTests.root(self, session)
        self.assertIsNone(session.mode)
        self.assertFalse(session.backend.active)
        self.assertIn('CPU_ROOT_FORWARD_FAILED', session.backend.failure)
        self.assertNotIn('COMPLETE_PREFILL_HIDDEN_STATES_REQUIRED', session.backend.failure)
        session.close()

    def test_real_cpu_logits_exception_preserves_original_failure(self):
        session = self.session()
        session.processor.raise_forward = True
        session.expect(self.actual_request_id)
        hidden = life.ServingSessionLifecycleTests.root(self, session)
        with self.assertRaisesRegex(RuntimeError, 'CPU_PROCESSOR_FORWARD_FAILED'):
            session.processor(session.head, hidden[-1:].contiguous())
        self.assertIsNone(session.mode)
        self.assertFalse(session.backend.active)
        self.assertIn('CPU_PROCESSOR_FORWARD_FAILED', session.backend.failure)
        self.assertNotIn('COMPLETE_GLOBAL_VOCAB_OUTPUT_REQUIRED', session.backend.failure)
        session.close()


if __name__ == '__main__':
    unittest.main()
