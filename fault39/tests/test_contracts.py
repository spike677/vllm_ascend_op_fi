import ctypes
import json
import math
from pathlib import Path
import sys
import tempfile
import threading
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from demoncat_fault39 import (DocumentContractAdapter, EnableError, Fault39Config,
    Fault39OutputInspector, INTEGRATION_STATUS, NativeBackend, WorkerEndpoint, integration_manifest)
from demoncat_fault39.inspector import summarize_values, validate_statistics
from demoncat_fault39.msprobe import ComparisonPolicy, first_difference, load_statistics_records, statistics_equal
from demoncat_fault39.provider import CtypesProvider, MODE_IDS, NativeConfigV1, NativeError, PATTERN_IDS
from demoncat_fault39.schema import PATTERNS
from demoncat_fault39.target import MatmulMetadata, TargetError, select_stable_target
from demoncat_fault39.testing import MockDemoncatHost, MockNativeProvider
from demoncat_fault39.transform import (TensorLayout, amplitude_exponent, cancellation_sign,
    input_value_transform, pressure_assessment)


def backend_for(tp=1, selected=(0,), max_hits=3):
    providers = {rank: MockNativeProvider(rank, rank) for rank in range(tp)}
    backend = NativeBackend({r: WorkerEndpoint(r, r, p) for r, p in providers.items()}, owner="owner-39")
    backend.prepare()
    backend.register(Fault39Config(ranks=selected, max_hits=max_hits))
    return backend, providers


def discover(backend):
    backend.discover()
    for worker in backend.workers.values():
        worker.provider.begin_call()
    backend.observe_inventory()
    for worker in backend.workers.values():
        worker.provider.begin_call()
    backend.observe_inventory()
    backend.end_discovery()


class SchemaAndTransformTests(unittest.TestCase):
    def test_one_primitive_and_five_parameters(self):
        manifest = integration_manifest()
        self.assertEqual(manifest["primitive_count"], 1)
        self.assertFalse(manifest["output_proxy"])
        self.assertEqual(len(PATTERNS), 5)
        for pattern in PATTERNS:
            self.assertEqual(Fault39Config(pattern=pattern).pattern, pattern)

    def test_config_rejects_ignored_filters_and_other_faults(self):
        for values in ({"fault_id": 38}, {"pattern": "output_inf"}, {"target": "down_proj"},
                       {"max_hits": 0}, {"ranks": []}, {"ranks": [0, 0]}, {"ranks": [True]},
                       {"amplitude_policy": "unbounded"}, {"devices": [-1]}):
            with self.subTest(values=values), self.assertRaises(ValueError):
                Fault39Config.from_dict(values)
        with self.assertRaises(TypeError):
            Fault39Config.from_dict({"callsite": "ignored"})

    def test_patterns_have_exact_zero_sum_at_unaligned_half(self):
        for pattern in PATTERNS:
            signs = [cancellation_sign(i, 1290, pattern) for i in range(1290)]
            self.assertEqual(sum(signs), 0)
            self.assertEqual(signs.count(1), 645)
            if pattern.startswith("block"):
                self.assertTrue(all(signs[k] == -signs[k + 645] for k in range(645)))

    def test_dynamic_mkn_stride_offset_and_padding(self):
        left_layout = TensorLayout((2, 8), (17, 2), 34, 1)
        right_layout = TensorLayout((8, 3), (1, 11), 34, 2)
        left_original, right_original = [7.0] * 34, [9.0] * 34
        for pattern in PATTERNS:
            left, right = input_value_transform(left_original, right_original, left_layout, right_layout, pattern, "auto")
            for row in range(2):
                for col in range(3):
                    value = sum(left[left_layout.index(row, k)] * right[right_layout.index(k, col)] for k in range(8))
                    self.assertEqual(value, 0)
            used_left = {left_layout.index(i, j) for i in range(2) for j in range(8)}
            used_right = {right_layout.index(i, j) for i in range(8) for j in range(3)}
            self.assertTrue(all(left[i] == 7 for i in range(34) if i not in used_left))
            self.assertTrue(all(right[i] == 9 for i in range(34) if i not in used_right))
        self.assertEqual(left_original, [7.0] * 34)
        self.assertEqual(right_original, [9.0] * 34)

    def test_odd_k_and_small_k_have_no_auto_pressure_solution(self):
        for k in (1, 3, 129):
            with self.assertRaisesRegex(ValueError, "EVEN_K"):
                amplitude_exponent(k, "auto")
        for k in (2, 4, 6):
            with self.assertRaisesRegex(ValueError, "NO_OVERFLOW_SOLUTION"):
                amplitude_exponent(k, "auto")
        self.assertEqual(amplitude_exponent(8, "auto"), 63)
        self.assertEqual(amplitude_exponent(12800, "auto"), 58)
        self.assertEqual(amplitude_exponent(8162, "auto"), 59)
        self.assertFalse(pressure_assessment(8162, 58)["half_pattern_pressure"])

    def test_layout_bounds_aliasing_and_exact_gcd_rule(self):
        TensorLayout((2, 3), (3, 2), 8)
        with self.assertRaisesRegex(ValueError, "OVERLAPPING"):
            TensorLayout((2, 3), (2, 1), 10)
        with self.assertRaisesRegex(ValueError, "BOUNDS"):
            TensorLayout((2, 8), (8, 1), 15)
        with self.assertRaisesRegex(ValueError, "DTYPE"):
            TensorLayout((2, 8), (8, 1), 16, dtype="INT8")

    def test_signature_is_metadata_stable_and_shape_sensitive(self):
        a = TensorLayout((1, 8), (8, 1), 8)
        b = TensorLayout((8, 3), (1, 8), 24)
        c = TensorLayout((1, 3), (3, 1), 3)
        self.assertEqual(MatmulMetadata(a, b, c).signature, MatmulMetadata(a, b, c).signature)
        changed = TensorLayout((8, 3), (3, 1), 24)
        self.assertNotEqual(MatmulMetadata(a, b, c).signature, MatmulMetadata(a, changed, c).signature)


class LifecycleTests(unittest.TestCase):
    def test_prepare_and_register_are_default_off(self):
        backend, providers = backend_for()
        self.assertFalse(providers[0].snapshot()["enabled"])
        self.assertEqual(backend.query()["state"], "REGISTERED")
        self.assertIsNone(providers[0].begin_call())
        with self.assertRaisesRegex(RuntimeError, "DISCOVERY_REQUIRED"):
            backend.enable()

    def test_stable_unique_discovery_is_required(self):
        backend, providers = backend_for()
        backend.discover()
        providers[0].begin_call()
        self.assertEqual(backend.observe_inventory()["status"], "STABILITY_WORKLOAD_REQUIRED")
        providers[0].begin_call()
        self.assertEqual(backend.observe_inventory()["status"], "STABLE_UNIQUE_TARGET")
        backend.end_discovery()
        self.assertEqual(backend.enable()["state"], "ARMED")

    def test_ambiguous_same_signature_is_not_collapsed(self):
        candidate = {"signature": "same", "candidate_count": 2}
        with self.assertRaisesRegex(TargetError, "AMBIGUOUS_TARGET"):
            select_stable_target([candidate], [candidate])
        with self.assertRaisesRegex(TargetError, "AMBIGUOUS_TARGET"):
            select_stable_target([dict(candidate, candidate_count=1)] * 2, [])

    def test_changed_target_inventory_rejects_enable(self):
        backend, providers = backend_for()
        backend.discover()
        backend.observe_inventory()
        providers[0]._candidates[0]["signature"] = "different"
        with self.assertRaisesRegex(TargetError, "UNSTABLE_TARGET"):
            backend.observe_inventory()
        backend.disable()

    def test_adjacent_control_reads_do_not_establish_stability(self):
        backend, providers = backend_for()
        backend.discover()
        providers[0].begin_call()
        backend.observe_inventory()
        with self.assertRaisesRegex(TargetError, "new real Run"):
            backend.observe_inventory()
        self.assertFalse(backend.targets)

    def test_tp1_tp2_tp4_filter_real_mock_rank_and_device(self):
        for tp in (1, 2, 4):
            with self.subTest(tp=tp):
                backend, providers = backend_for(tp, (tp - 1,))
                discover(backend)
                backend.enable()
                for rank, provider in providers.items():
                    with provider.context(backend.owner, ["actual-mock-request"]):
                        lease = provider.begin_call()
                        if rank == tp - 1:
                            self.assertIsNotNone(lease)
                            provider.complete_call(lease)
                        else:
                            self.assertIsNone(lease)
                self.assertEqual(backend.disable()["status"], "DISABLED_CONFIRMED")
                self.assertEqual(sum(p.snapshot()["counters"]["hit_count"] for p in providers.values()), 1)

    def test_runtime_identity_is_independent_of_selected_rank(self):
        provider = MockNativeProvider(rank=0, device=0)
        config = Fault39Config()
        provider.configure("owner", config, 1, 0, mode="discover")
        self.assertIsNone(provider.begin_call())
        self.assertEqual(provider.snapshot()["counters"]["candidate_count"], 0)

    def test_partial_enable_rolls_back_every_armed_worker(self):
        backend, providers = backend_for(2, (0, 1))
        discover(backend)
        providers[1].fail_next_configure = True
        with self.assertRaises(EnableError) as raised:
            backend.enable()
        self.assertEqual(raised.exception.rollback["status"], "DISABLED_CONFIRMED")
        self.assertTrue(all(not p.snapshot()["enabled"] for p in providers.values()))

    def test_lost_enable_reply_still_rolls_back_owned_generation(self):
        backend, providers = backend_for()
        discover(backend)
        original = providers[0].configure
        def lost_reply(*args, **kwargs):
            original(*args, **kwargs)
            raise TimeoutError("reply lost")
        providers[0].configure = lost_reply
        with self.assertRaises(EnableError):
            backend.enable()
        self.assertFalse(providers[0].snapshot()["enabled"])

    def test_lost_reply_and_unreachable_worker_cannot_claim_rollback(self):
        backend, providers = backend_for()
        discover(backend)
        provider = providers[0]
        original_configure, original_snapshot = provider.configure, provider.snapshot
        def unreachable():
            raise TimeoutError("worker unreachable")
        def lost_reply(*args, **kwargs):
            original_configure(*args, **kwargs)
            provider.snapshot = unreachable
            raise TimeoutError("reply lost")
        provider.configure = lost_reply
        with self.assertRaises(EnableError) as raised:
            backend.enable()
        self.assertEqual(raised.exception.rollback["status"], "DISABLE_PENDING")
        self.assertEqual(backend.clean(0)["status"], "DISABLE_PENDING")
        provider.snapshot = original_snapshot
        self.assertTrue(provider.snapshot()["enabled"])
        self.assertEqual(backend.disable()["status"], "DISABLED_CONFIRMED")

    def test_inflight_timeout_retains_lease_then_real_drain_ack(self):
        backend, providers = backend_for()
        discover(backend)
        backend.enable()
        with providers[0].context(backend.owner, ["request-1"]):
            lease = providers[0].begin_call()
        self.assertEqual(backend.disable(timeout_ms=0)["status"], "DISABLE_PENDING")
        self.assertEqual(providers[0].snapshot()["leases"], 2)
        with self.assertRaisesRegex(RuntimeError, "DISABLE_REQUIRED"):
            backend.enable()
        providers[0].complete_call(lease)
        self.assertEqual(backend.disable()["status"], "DISABLED_CONFIRMED")
        self.assertEqual(providers[0].snapshot()["leases"], 0)

    def test_flag_only_worker_ack_is_not_a_drain_confirmation(self):
        backend, providers = backend_for()
        discover(backend)
        backend.enable()
        providers[0].disable = lambda *args: {"status": "DISABLED_CONFIRMED"}
        self.assertEqual(backend.disable()["status"], "DISABLE_PENDING")
        self.assertTrue(backend.generations)

    def test_async_disable_waits_until_old_call_finishes(self):
        backend, providers = backend_for()
        discover(backend)
        backend.enable()
        with providers[0].context(backend.owner, ["request-1"]):
            lease = providers[0].begin_call()
        started = threading.Event()
        done = threading.Event()
        result = []
        def stop():
            started.set()
            result.append(backend.disable(1000))
            done.set()
        thread = threading.Thread(target=stop)
        thread.start()
        self.assertTrue(started.wait(1))
        self.assertFalse(done.wait(0.02))
        providers[0].complete_call(lease)
        thread.join(2)
        self.assertFalse(thread.is_alive())
        self.assertEqual(result[0]["status"], "DISABLED_CONFIRMED")

    def test_recovery_failure_never_releases_live_buffer(self):
        backend, providers = backend_for()
        discover(backend)
        backend.enable()
        with providers[0].context(backend.owner, ["request-1"]):
            lease = providers[0].begin_call()
        providers[0].complete_call(lease, recovery_failed=True)
        self.assertEqual(backend.clean(0)["status"], "RECOVERY_FAILED")
        self.assertEqual(providers[0].snapshot()["leases"], 2)
        self.assertTrue(backend.generations)

    def test_clean_preserves_other_fault_owner_and_reenable_generation(self):
        backend, providers = backend_for()
        providers[0].other_fault_owners["fault28-owner"] = {"enabled": True, "value": 7}
        discover(backend)
        backend.enable()
        old_generation = backend.generations[0]
        backend.disable()
        backend.enable()
        self.assertGreater(backend.generations[0], old_generation)
        self.assertEqual(backend.clean()["status"], "DISABLED_CONFIRMED")
        self.assertEqual(providers[0].other_fault_owners["fault28-owner"], {"enabled": True, "value": 7})
        self.assertEqual(providers[0].snapshot()["owner"], "")
        backend.unregister()
        backend.register(Fault39Config())
        self.assertEqual(backend.query()["state"], "REGISTERED")

    def test_owner_and_generation_mismatches_cannot_disable(self):
        backend, providers = backend_for()
        discover(backend)
        backend.enable()
        with self.assertRaisesRegex(NativeError, "OWNER_MISMATCH"):
            providers[0].disable("someone-else", backend.generations[0], 0)
        with self.assertRaisesRegex(NativeError, "GENERATION_MISMATCH"):
            providers[0].disable(backend.owner, backend.generations[0] - 1, 0)
        self.assertTrue(providers[0].snapshot()["enabled"])

    def test_request_scope_and_max_hits_are_consumed(self):
        backend, providers = backend_for(max_hits=1)
        discover(backend)
        backend.enable()
        self.assertIsNone(providers[0].begin_call())
        self.assertEqual(providers[0].snapshot()["status"], "UNPROVEN_REQUEST_MAPPING")
        with providers[0].context("other-owner", ["request"]):
            self.assertIsNone(providers[0].begin_call())
        with providers[0].context(backend.owner, ["request"]):
            lease = providers[0].begin_call()
            self.assertIsNone(providers[0].begin_call())
        providers[0].complete_call(lease)


class EvidenceTests(unittest.TestCase):
    def test_zero_is_subset_of_finite_never_double_counted(self):
        stats = summarize_values([0.0, -0.0, 3.0, -2.0, math.inf, -math.inf, math.nan])
        self.assertEqual(stats["total"], 7)
        self.assertEqual(stats["finite"], 4)
        self.assertEqual(stats["zero"], 2)
        self.assertEqual(stats["finite"] + stats["nan"] + stats["posinf"] + stats["neginf"], 7)
        with self.assertRaisesRegex(ValueError, "TOTAL_MISMATCH"):
            validate_statistics(dict(stats, total=9))

    def test_complete_input_and_output_statistics_required(self):
        inspector = Fault39OutputInspector()
        report = inspector.inspect({"runtime_loaded": True, "interposed_run_calls": 1,
            "typed_metadata_calls": 1, "counters": {"hit_count": 1}})
        self.assertEqual(report["verdict"], "INSUFFICIENT_NATIVE_EVIDENCE")
        self.assertEqual(len(report["statistics_errors"]), 5)

    def test_inspector_does_not_require_fixed_hardware_output_distribution(self):
        a = 2**63
        observation = {"original_self": summarize_values([1] * 8), "original_mat2": summarize_values([1] * 8),
            "temporary_self": summarize_values([a] * 8), "temporary_mat2": summarize_values([a] * 4 + [-a] * 4),
            "output": summarize_values([math.inf]), "exponent": 63, "generation": 2, "owner": "owner", "hit": 1,
            "changed": [8, 8], "unchanged": [0, 0], "run_status": 0, "pre_sync_status": 0,
            "copy_status": 0, "post_sync_status": 0, "free_status": 0,
            "actual_request_ids_json": '["scheduler-id"]', "output_source": "post_run_device_readback",
            "temporary_source": "host_constructed", "input_device_readback_verified": False,
            "per_column_sign_counts_verified": True, "self_constant_positive_verified": True,
            "proof": {"generation": 2, "hit": 1, "formal_order_proven": True, "slots": [0, 1, 2], "input_mask": 3,
                "tensors": [{"shape": [1, 8], "original_address": 100, "run_address": 101},
                            {"shape": [8, 1], "original_address": 200, "run_address": 201},
                            {"shape": [1, 1], "original_address": 300, "run_address": 300}]}}
        native = {"last_observation": observation, "runtime_loaded": True, "owner": "owner", "generation": 2,
            "inflight": 0, "leases": 0, "last_error": 0, "status": "COMPLETED",
            "interposed_run_calls": 1, "typed_metadata_calls": 1, "counters": {"hit_count": 1}}
        report = Fault39OutputInspector().inspect(native)
        self.assertEqual(report["verdict"], "MATMUL_INTERMEDIATE_RANGE_FAILURE_REPRODUCED")
        self.assertEqual(report["request_mapping"], "UNPROVEN_REQUEST_MAPPING")
        self.assertFalse(report["tensor_payloads"])
        self.assertFalse(report["input_device_readback_verified"])
        stale = dict(native, generation=3)
        self.assertEqual(Fault39OutputInspector().inspect(stale)["verdict"], "INSUFFICIENT_NATIVE_EVIDENCE")
        for pending in (dict(native, inflight=1), dict(native, leases=1), dict(native, last_error=-3),
                        dict(native, counters={"hit_count": 2}), dict(native, status="ARMED")):
            self.assertEqual(Fault39OutputInspector().inspect(pending)["verdict"], "INSUFFICIENT_NATIVE_EVIDENCE")

    def test_nonfinite_null_semantics_never_coerce_to_zero(self):
        pairs = [(None, 0), (None, "NaN"), (None, math.nan), (math.inf, -math.inf),
                 (math.inf, None), (math.nan, math.nan), ("NaN", "NaN"), (True, 1)]
        for a, b in pairs:
            self.assertFalse(statistics_equal(a, b), (a, b))
        self.assertFalse(statistics_equal(None, None))
        self.assertTrue(statistics_equal(math.inf, "+Inf"))
        self.assertTrue(statistics_equal(1.0, 1.0001, ComparisonPolicy(0.001, 0)))

    def test_statistics_use_additive_asymmetric_baseline_reference(self):
        self.assertTrue(statistics_equal(10.0, 12.0, ComparisonPolicy(1.0, 0.1)))
        self.assertFalse(statistics_equal(10.0, 12.01, ComparisonPolicy(1.0, 0.1)))
        self.assertFalse(statistics_equal(10.0, 11.05, ComparisonPolicy(0.0, 0.1)))
        self.assertTrue(statistics_equal(11.05, 10.0, ComparisonPolicy(0.0, 0.1)))

    def test_empty_and_null_capture_never_claim_full_observed_equality(self):
        self.assertEqual(first_difference([], [])["status"], "INSUFFICIENT_STATISTICS")
        record = [{"identity": "op", "statistics": {"Mean": None, "Max": 0}}]
        result = first_difference(record, record)
        self.assertEqual(result["status"], "NO_OBSERVABLE_STATISTICS_DIFFERENCE")
        self.assertEqual(result["unobservable_fields"][0]["field"], "/Mean")

    def test_first_difference_uses_only_collector_order(self):
        baseline = [{"identity": f"op-{i}", "statistics": {"mean": 0}} for i in range(3)]
        fault = [{"identity": f"op-{i}", "statistics": {"mean": 0 if i == 0 else 2}} for i in range(3)]
        report = first_difference(baseline, fault)
        self.assertEqual(report["identity"], "op-1")
        self.assertEqual(report["request_mapping"], "UNPROVEN_REQUEST_MAPPING")
        with self.assertRaises(TypeError):
            first_difference(baseline, fault, native_hit_log={"identity": "op-2"})

    def test_alignment_and_missing_statistics_are_not_guessed(self):
        left = [{"identity": "a", "statistics": {"min": None}}]
        right = [{"identity": "b", "statistics": {"min": None}}]
        self.assertEqual(first_difference(left, right)["status"], "ALIGNMENT_MISMATCH")
        with self.assertRaises(ValueError):
            first_difference([{"identity": "x", "statistics": {}, "tensor": [1]}], [])

    def test_statistics_json_loader_rejects_tensor_upload_formats(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "stats.json"
            payload = {"schema": "fault39.msprobe.statistics.v1", "mode": "statistics",
                       "tensor_payloads": False, "collector_identity": "test", "records": []}
            path.write_text(json.dumps(payload))
            loaded = load_statistics_records(path)
            self.assertEqual(len(loaded["source_sha256"]), 64)
            payload["tensor_payloads"] = True
            path.write_text(json.dumps(payload))
            with self.assertRaisesRegex(ValueError, "STATISTICS_ONLY"):
                load_statistics_records(path)
            with self.assertRaisesRegex(ValueError, "STATISTICS_JSON"):
                load_statistics_records(Path(directory) / "tensor.npy")

    def test_document_contract_keeps_inner_framework_pending_and_owner_isolated(self):
        host = MockDemoncatHost()
        host.catalog["other-fault"] = ("other-owner", {})
        host.bindings["other-binding"] = ("other-owner", None)
        adapter = DocumentContractAdapter(host)
        result = adapter.install()
        self.assertEqual(result["status"], INTEGRATION_STATUS)
        self.assertEqual(len(host.catalog), 2)
        adapter.uninstall()
        self.assertEqual(list(host.catalog), ["other-fault"])
        self.assertEqual(list(host.bindings), ["other-binding"])


class FakeFunction:
    def __init__(self, function):
        self.function = function
    def __call__(self, *args):
        return self.function(*args)


class FakeAbi:
    def __init__(self):
        self.provider = MockNativeProvider()
        self.context = None
        self.last_config = None
        self.fault39_abi_version_v1 = FakeFunction(lambda: 1)
        self.fault39_snapshot_v1 = FakeFunction(self.snapshot)
        self.fault39_configure_v1 = FakeFunction(self.configure)
        self.fault39_disable_v1 = FakeFunction(self.disable)
        self.fault39_clean_v1 = FakeFunction(self.clean)
        self.fault39_set_context_v1 = FakeFunction(self.set_context)
        self.fault39_clear_context_v1 = FakeFunction(self.clear_context)
        self.fault39_bind_worker_rank_v1 = FakeFunction(lambda rank: 0 if rank == 0 else -1)
    def snapshot(self, output, capacity):
        encoded = json.dumps(self.provider.snapshot()).encode() + b"\0"
        if output and capacity >= len(encoded):
            ctypes.memmove(output, encoded, len(encoded))
        return len(encoded)
    def configure(self, pointer, generation):
        raw = ctypes.cast(pointer, ctypes.POINTER(NativeConfigV1)).contents
        self.last_config = NativeConfigV1.from_buffer_copy(raw)
        config = Fault39Config(pattern={v:k for k,v in PATTERN_IDS.items()}[raw.pattern],
            amplitude_policy="auto" if raw.exponent == -1 else "historical58", max_hits=raw.max_hits, max_bytes=raw.max_bytes)
        try:
            value = self.provider.configure(raw.owner.decode(), config, raw.rank, raw.device,
                {v:k for k,v in MODE_IDS.items()}[raw.mode], raw.target_signature.decode(), raw.input_mask)
        except NativeError as exc:
            return exc.code
        ctypes.cast(generation, ctypes.POINTER(ctypes.c_uint64))[0] = value
        return 0
    def disable(self, owner, generation, timeout):
        try:
            return self.provider.disable(owner.decode(), generation, timeout)["native_code"]
        except NativeError as exc:
            return exc.code
    def clean(self, owner, generation, timeout):
        try:
            return self.provider.clean(owner.decode(), generation, timeout)["native_code"]
        except NativeError as exc:
            return exc.code
    def set_context(self, owner, request_ids):
        self.context = (owner.decode(), json.loads(request_ids))
        return 0
    def clear_context(self):
        self.context = None


class CtypesProviderTests(unittest.TestCase):
    def test_abi_size_and_tail_offsets_match_native_v1(self):
        self.assertEqual(ctypes.sizeof(NativeConfigV1), 368)
        self.assertEqual(NativeConfigV1.owner.offset, 40)
        self.assertEqual(NativeConfigV1.target_signature.offset, 104)
        self.assertEqual(NativeConfigV1.input_mask.offset, 360)

    def test_attach_only_process_namespace_never_dlopen_path(self):
        with patch("demoncat_fault39.provider.ctypes.CDLL", return_value=FakeAbi()) as loader:
            provider = CtypesProvider.attach_preloaded()
            loader.assert_called_once_with(None)
            self.assertFalse(provider.prepare()["interception_verified"])

    def test_real_ctypes_struct_roundtrip_and_owner_clean(self):
        abi = FakeAbi()
        provider = CtypesProvider(abi, evidence_kind="CPU_CONTRACT_MOCK")
        provider.bind_worker_rank(0)
        generation = provider.configure("owner", Fault39Config(), 0, 0, mode="discover")
        self.assertEqual(abi.last_config.struct_size, 368)
        self.assertEqual(abi.last_config.input_mask, 3)
        self.assertEqual(provider.disable("owner", generation, 0)["status"], "DISABLED_CONFIRMED")
        self.assertEqual(provider.clean("owner", generation, 0)["status"], "DISABLED_CONFIRMED")

    def test_lossless_input_mask_only_and_context_always_cleared(self):
        abi = FakeAbi()
        provider = CtypesProvider(abi)
        with self.assertRaises(ValueError):
            provider.configure("owner", Fault39Config(), 0, 0, input_mask=1)
        with self.assertRaisesRegex(RuntimeError, "model failed"):
            with provider.context("owner", ["scheduler-request"]):
                self.assertEqual(abi.context, ("owner", ["scheduler-request"]))
                with self.assertRaisesRegex(RuntimeError, "NESTED"):
                    with provider.context("owner", ["nested"]):
                        pass
                raise RuntimeError("model failed")
        self.assertIsNone(abi.context)

    def test_native_zero_return_cannot_fake_drain_confirmation(self):
        abi = FakeAbi()
        provider = CtypesProvider(abi)
        generation = provider.configure("owner", Fault39Config(), 0, 0, mode="discover")
        abi.fault39_disable_v1 = FakeFunction(lambda *args: 0)
        self.assertEqual(provider.disable("owner", generation, 0)["status"], "DISABLE_PENDING")


if __name__ == "__main__":
    unittest.main()
