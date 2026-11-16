"""Real, randomly initialized CPU models; no downloads or NPU validation.

Install torch and transformers==5.5.4 to execute. The transparent backend tests
automatic attachment and semantic preservation independently of the bounded
FIA kernel. Small models intentionally do not fit the kernel's NPU geometry.
"""
import importlib.util
import unittest

from fault_injection.attention import AttentionSession, _resolve, inspect_model
from fault_injection.core import Decision, UnsupportedCall
from fault_injection.fia import FiaBackend

HAS_DEPENDENCIES = bool(importlib.util.find_spec("torch") and importlib.util.find_spec("transformers"))


class TransparentBackend:
    def __init__(self, original_backend="eager"):
        self.calls = []
        self.original_backend = original_backend

    def check(self, call):
        call.geometry()
        return Decision(True, "CPU interface equivalence fixture, not native injection")

    def run(self, call):
        self.calls.append(call)
        resolved, _ = _resolve(call.module)
        original = resolved[0].get_interface(self.original_backend, resolved[1])
        return original(call.module, call.query, call.key, call.value,
                        call.attention_mask, dropout=call.dropout,
                        scaling=call.scaling, **call.kwargs)


@unittest.skipUnless(HAS_DEPENDENCIES, "torch and transformers required; CPU integration not executed")
class RealTransformersTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import transformers
        import torch
        if transformers.__version__ != "5.5.4":
            raise unittest.SkipTest("Integration matrix pinned to transformers==5.5.4")
        cls.torch = torch
        torch.set_num_threads(1)

    def make_model(self, family, original_backend="eager"):
        import transformers as hf
        config_type, model_type = {
            "qwen3": (hf.Qwen3Config, hf.Qwen3ForCausalLM),
            "glm": (hf.GlmConfig, hf.GlmForCausalLM),
            "deepseek_v3": (hf.DeepseekV3Config, hf.DeepseekV3ForCausalLM),
            "llama": (hf.LlamaConfig, hf.LlamaForCausalLM),
        }[family]
        args = dict(vocab_size=64, hidden_size=32, intermediate_size=64,
                    num_hidden_layers=2, num_attention_heads=4,
                    num_key_value_heads=2, max_position_embeddings=128,
                    attention_dropout=0.0, pad_token_id=0)
        if family == "qwen3":
            args.update(head_dim=8, num_hidden_layers=3)
        if family == "deepseek_v3":
            args.update(num_key_value_heads=4, q_lora_rank=8, kv_lora_rank=8,
                        qk_nope_head_dim=8, qk_rope_head_dim=4, v_head_dim=8,
                        moe_intermediate_size=16, n_shared_experts=1,
                        n_routed_experts=4, num_experts_per_tok=2, n_group=1,
                        topk_group=1, first_k_dense_replace=2)
        self.torch.manual_seed(17)
        config = config_type(**args)
        config._attn_implementation = original_backend
        return model_type(config).eval()

    def exercise(self, family, original_backend="eager"):
        import transformers as hf
        from transformers.masking_utils import AttentionMaskInterface
        torch = self.torch
        model = self.make_model(family, original_backend)
        layers = model.config.num_hidden_layers
        discovery = inspect_model(model)
        self.assertEqual(len(discovery["candidates"]), layers, discovery)
        self.assertEqual(discovery["unsupported_modules"], [])
        initial_attention = dict(hf.AttentionInterface._global_mapping)
        initial_masks = dict(AttentionMaskInterface._global_mapping)
        ids = torch.tensor([[1, 2, 3, 4, 5, 6], [0, 0, 7, 8, 9, 10]])
        mask = (ids != 0).long()
        next_ids = torch.tensor([[11], [12]])
        decode_mask = torch.cat((mask, torch.ones((2, 1), dtype=torch.long)), dim=1)
        with torch.no_grad():
            baseline = model(input_ids=ids, attention_mask=mask, use_cache=True)
            baseline_decode = model(input_ids=next_ids, attention_mask=decode_mask,
                                    past_key_values=baseline.past_key_values, use_cache=True)
            backend = TransparentBackend(original_backend)
            with AttentionSession(model, backend) as session:
                self.assertIs(AttentionMaskInterface._global_mapping[model.config._attn_implementation],
                              initial_masks[original_backend])
                observed = model(input_ids=ids, attention_mask=mask, use_cache=True)
                observed_decode = model(input_ids=next_ids, attention_mask=decode_mask,
                                        past_key_values=observed.past_key_values, use_cache=True)
                report = session.assert_coverage()
                self.assertEqual(len(report["events"]), 2 * layers)
                self.assertTrue(all(row["observed_calls"] == 2 for row in report["modules"].values()))
                self.assertTrue(all(row["request_slot_executions"] == 4 for row in report["modules"].values()))
                self.assertTrue(torch.equal(observed.logits, baseline.logits), family + " prefill")
                self.assertTrue(torch.equal(observed_decode.logits, baseline_decode.logits), family + " cached decode")
                self.assertTrue(all(call.attention_mask is not None for call in backend.calls))
                self.assertTrue(all(call.scaling is not None for call in backend.calls))
                self.assertTrue(any(call.key.shape[2] > call.query.shape[2] for call in backend.calls))
                if family == "deepseek_v3":
                    self.assertTrue(all(call.query.shape[-1] != call.value.shape[-1] for call in backend.calls))
                else:
                    self.assertTrue(all(call.query.shape[1] > call.key.shape[1] for call in backend.calls))
            off = model(input_ids=ids, attention_mask=mask, use_cache=False)
        self.assertTrue(torch.equal(off.logits, baseline.logits))
        self.assertEqual(model.config._attn_implementation, original_backend)
        self.assertEqual(hf.AttentionInterface._global_mapping, initial_attention)
        self.assertEqual(AttentionMaskInterface._global_mapping, initial_masks)
        # A generic registration must not silently bypass the native profile.
        decision = FiaBackend().check(backend.calls[0])
        self.assertFalse(decision.supported)
        with torch.no_grad(), AttentionSession(model, FiaBackend()) as session:
            with self.assertRaises(UnsupportedCall):
                model(input_ids=ids, attention_mask=mask, use_cache=False)
            self.assertEqual(session.report()["events"][0]["status"], "unsupported")

    def test_qwen3(self):
        self.exercise("qwen3")

    def test_glm(self):
        self.exercise("glm")

    def test_deepseek_v3(self):
        self.exercise("deepseek_v3")

    def test_llama(self):
        self.exercise("llama")

    def test_qwen3_sdpa(self):
        self.exercise("qwen3", "sdpa")


if __name__ == "__main__":
    unittest.main()
