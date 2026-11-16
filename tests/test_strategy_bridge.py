"""CPU contract checks; fixtures do not establish NPU/model validation."""
from contextlib import contextmanager
from dataclasses import replace
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import unittest

from fault_injection.strategy import NativeFAInvocation, NativeStrategyBridge, attention_context
from native_execution_strategy.adapters.flash_attention import FlashAttentionAdapter
from native_execution_strategy.config import PolicyConfig
from native_execution_strategy.models import (
    BuildFingerprints, EvidenceRef, InvocationContext, NativeStrategy, StrategyObservation,
)
from native_execution_strategy.policies import Policy
from native_execution_strategy.session import StrategySession

ROOT = Path(__file__).resolve().parents[1] / "native_execution_strategy"


def tensor(shape, dtype="torch.float16"):
    return SimpleNamespace(shape=shape, dtype=dtype)


def fixture(batch=8, query_heads=16, kv_heads=8, sequence=2048, qdim=128, vdim=128):
    call = SimpleNamespace(path="decoder.blocks.odd_name", module=object(),
        query=tensor([batch, query_heads, sequence, qdim]),
        key=tensor([batch, kv_heads, sequence, qdim]),
        value=tensor([batch, kv_heads, sequence, vdim]),
        attention_mask=object(), dropout=0.0, scaling=qdim ** -0.5, kwargs={})
    native = NativeFAInvocation(kwargs={
        "head_num": query_heads, "input_layout": "BNSD",
        "atten_mask": tensor([sequence, sequence], "torch.bool"),
        "scale": call.scaling, "keep_prob": 1, "pre_tockens": 2147483647,
        "next_tockens": 0, "inner_precise": 0, "sparse_mode": 2,
    }, sequence_lengths=[sequence] * batch, device="Ascend910B4")
    return call, native


class FixtureAdapter:
    name = "FlashAttentionScore"
    on_unsupported = "reject"

    def __init__(self):
        self.baseline = None
        self.active = False
        self.fail_observe = False
        self.last_actual = None

    def inspect(self, actual):
        self.last_actual = actual
        return InvocationContext(self.name, request_count=actual["request_count"],
            token_count=actual["token_count"], seq_lens=tuple(actual["seq_lens"]),
            dtype=actual["inputs"]["query"]["dtype"], layout=actual["layout"],
            attributes=actual["attributes"])

    def unsupported_reason(self, context, *, controlled):
        return None

    def observe_current(self, context):
        if self.baseline is None:
            raise ValueError("No native observation")
        return self.baseline

    def legal_alternatives(self, context, strategy):
        return (NativeStrategy("alt", {"tile": 64}),)

    def validate_candidate(self, context, candidate):
        return candidate.strategy_id == "alt"

    @contextmanager
    def native_scope(self, decision, invocation_id, evidence_dir, *, context, current=None):
        self.active = True
        try:
            yield decision
        finally:
            self.active = False

    def parse_native_evidence(self, context, record):
        if self.fail_observe:
            raise ValueError("Launch not bound")
        decision = record.get("decision")
        changed = decision is not None and decision.requires_execution_change
        observation = StrategyObservation(context,
            NativeStrategy("alt" if changed else "base", {"tile": 64 if changed else 128}),
            BuildFingerprints("1" * 64, "2" * 64),
            (EvidenceRef("explicit_cpu_fixture_only", "3" * 64),))
        if not changed:
            self.baseline = observation
        return observation


class StrategyBridgeTests(unittest.TestCase):
    def test_original_package_bytes_unchanged(self):
        manifest = json.loads((ROOT / "IMPORT_MANIFEST.json").read_bytes())
        self.assertTrue(manifest["byte_identical"])
        for row in manifest["files"]:
            self.assertEqual(hashlib.sha256((ROOT / row["path"]).read_bytes()).hexdigest(), row["sha256"])

    def test_validated_certificate_survives_relocation_without_widening(self):
        adapter = FlashAttentionAdapter(ROOT / "data/fa_cann91_s1s2_v1.json", evidence_root=ROOT)
        call, native = fixture()
        ctx = attention_context(call, adapter, native)
        self.assertEqual(ctx.request_count, 8)
        self.assertEqual(ctx.matrix_dims["query_heads"], 16)
        self.assertIsNone(adapter.unsupported_reason(ctx, controlled=True))
        for changed in [fixture(query_heads=32), fixture(sequence=1024), fixture(qdim=64), fixture(vdim=64)]:
            with self.assertRaises(ValueError):
                attention_context(changed[0], adapter, changed[1])

    def test_context_reads_actual_dimensions_not_model_name(self):
        adapter = FixtureAdapter()
        call, native = fixture(batch=3, query_heads=12, kv_heads=3, sequence=17, qdim=48, vdim=32)
        ctx = attention_context(call, adapter, native)
        self.assertEqual(ctx.request_count, 3)
        self.assertEqual(ctx.token_count, 51)
        self.assertEqual(adapter.last_actual["inputs"]["value"]["shape"], [3, 3, 17, 32])
        self.assertEqual(adapter.last_actual["inputs"]["atten_mask"]["shape"], [17, 17])
        self.assertNotIn("layer", ctx.attributes)

    def test_no_inference_from_leading_TND_axis_or_unmapped_batch(self):
        call, native = fixture()
        for changed in [replace(native, kwargs=dict(native.kwargs, input_layout="TND")),
                        replace(native, sequence_lengths=[]), replace(native, sequence_lengths=[2048])]:
            with self.assertRaises(ValueError):
                attention_context(call, FixtureAdapter(), changed)

    def test_preserve_mask_scale_and_optional_input_contract(self):
        call, native = fixture()
        for kwargs in [dict(native.kwargs, atten_mask=None), dict(native.kwargs, scale=1.0),
                       dict(native.kwargs, pse=object()), dict(native.kwargs, extra_field=True),
                       dict(native.kwargs, keep_prob=.9)]:
            with self.assertRaises(ValueError):
                attention_context(call, FixtureAdapter(), replace(native, kwargs=kwargs))
        with self.assertRaises(ValueError):
            attention_context(call, FixtureAdapter(), replace(native, operator="FusedInferAttentionScore"))

    def test_single_native_call_each_mode_output_identity_and_actual_observation(self):
        adapter = FixtureAdapter()
        bridge = NativeStrategyBridge(StrategySession(adapter, Policy(PolicyConfig())))
        call, native = fixture()
        output = object()
        count = []

        def invoke():
            self.assertTrue(adapter.active)
            count.append(1)
            return output

        def observer(value, plan, directory):
            self.assertFalse(adapter.active)
            self.assertIs(value, output)
            return {"decision": plan}

        first = bridge.execute_attention(call, native, invoke, observer, "n", "unused_cpu_fixture", mode="native")
        second = bridge.execute_attention(call, native, invoke, observer, "f", "unused_cpu_fixture", mode="controlled")
        self.assertIs(first.value, output)
        self.assertIs(second.value, output)
        self.assertEqual(len(count), 2)
        self.assertIsNone(first.assessment.strategy_switched)
        self.assertTrue(second.assessment.strategy_switched)
        self.assertIsNone(second.assessment.model_impact)
        self.assertFalse(bridge.report()["model_validation_extended"])
        self.assertEqual(bridge.report()["events"][0]["module_path"], call.path)

    def test_controlled_without_native_rejects_before_execution(self):
        bridge = NativeStrategyBridge(StrategySession(FixtureAdapter(), Policy(PolicyConfig())))
        call, native = fixture()
        def forbidden():
            self.fail("No hidden baseline or uncontrolled execution allowed")
        with self.assertRaisesRegex(ValueError, "No native observation"):
            bridge.execute_attention(call, native, forbidden, forbidden, "f", "unused_cpu_fixture", mode="controlled")

    def test_observation_failure_retains_unknown_and_original_value(self):
        adapter = FixtureAdapter()
        adapter.fail_observe = True
        bridge = NativeStrategyBridge(StrategySession(adapter, Policy(PolicyConfig())))
        call, native = fixture()
        output = object()
        result = bridge.execute_attention(call, native, lambda: output, lambda *args: {}, "n", "unused_cpu_fixture")
        self.assertIs(result.value, output)
        self.assertIn("Launch not bound", result.observation_error)
        self.assertIsNone(result.assessment.strategy_switched)


if __name__ == "__main__":
    unittest.main()
