"""Raw model-record fixtures. No torch, model, NPU, callback or private output input."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import compare_model as cm
from test_msprobe_compare import tensor


def dump(path, obj):
    raw = json.dumps(obj, separators=(",", ":")).encode()
    path.write_bytes(raw)
    return hashlib.sha256(raw).hexdigest()


def fixture(root, name, *, first_layer=None, early=False, missing=None, dx=False, change_stack=False, change_meta=None):
    case = root / name
    rank = case / "observation/step0/rank0"
    rank.mkdir(parents=True)
    cfg_sha = dump(case / "config.json", {"task": "statistics", "level": "mix", "step": [0], "rank": [],
              "async_dump": False, "dump_path": str(case), "statistics": {"scope": [], "list": [],
              "data_mode": ["all"], "summary_mode": "statistics", "tensor_list": []}})
    prov_sha = dump(case / "collector_provenance.json", {"precision": "low", "variant": "official_low",
                    "modified_official_collector": False, "actual_loaded": {"path": "/x.py", "method_code_file": "/x.py", "sha256": "a"*64}})
    data, graph, stack, links = {}, {}, {}, []
    qshape, kvshape = [8, 16, 1, 128], [1024, 8, 128, 128]
    def add(key, parent, frame, node):
        data[key], graph[key], stack[str(len(stack))] = node, parent, [[key], [frame]]
    add("Torch.clone.0.forward", None, "File /audit/model_adapter.py, line 27, in cpu_array, \n return t.cpu()",
        {"input_args": [tensor(qshape)], "output": [tensor(qshape, .8 if dx else .25)]})
    add("Functional.embedding.0.forward", None, "File /src/nn.py, line 4, in forward, \n embed()",
        {"input_args": [tensor(qshape)], "output": [tensor(qshape, .75 if early else .25)]})
    for layer in range(28):
        parent = f"Module.model.layers.{layer}.self_attn.Qwen3Attention.forward.0"
        add(parent, None, "File /src/qwen.py, line 8, in forward, \n attention()",
            {"input_args": [tensor(qshape)], "output": [tensor(qshape)]})
        if missing == layer:
            continue
        changed_output = first_layer is not None and layer >= first_layer
        changed_input = first_layer is not None and layer > first_layer
        native = f"NPU.npu_incre_flash_attention.{layer}.forward"
        q = tensor(qshape if layer != change_meta else [1, 16, 1, 128], .75 if changed_input else .25)
        frame = "File /audit/model_adapter.py, line 75, in call, \n return original()"
        if change_stack and layer == 0:
            frame = frame.replace("75", "76")
        add(native, parent, frame, {"input_args": [q, tensor(kvshape), tensor(kvshape)],
            "input_kwargs": {"block_size": {"type": "int", "value": 128}, "block_table": tensor([8, 128], None, "torch.int32")},
            "output": [tensor(qshape, .75 if changed_output else .25)]})
        links.append({"step": 0, "layer_index": layer, "api": native, "parent": parent})
    docs = {"dump": {"task": "statistics", "level": "mix", "framework": "pytorch", "dump_data_dir": None, "data": data},
            "construct": graph, "stack": stack}
    files = []
    for k, v in docs.items():
        path = rank / (k + ".json")
        digest = dump(path, v)
        files.append({"path": path.relative_to(case).as_posix(), "bytes": path.stat().st_size, "sha256": digest})
    dump(case / "statistics_receipt.json", {"schema": "fault33.pa.model_statistics.v1", "steps": [0], "rank": 0,
         "config_sha256": cfg_sha, "collector_sha256": prov_sha, "native_files": files, "native_api_nodes": links})
    return rank / "dump.json"


class TestModel(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.b = fixture(self.root, "B")

    def pair(self, **opts):
        return cm.compare(self.b, fixture(self.root, "F", **opts))

    def test_derived_layer_ifa_output_later_layer_inputs_can_differ(self):
        r = self.pair(first_layer=9)
        self.assertEqual(r["status"], "ALIGNED_OBSERVED_MODEL_API_DIFFERENCE")
        first = r["first_observed_model_api_difference"]
        self.assertEqual((first["B"]["layer"], first["boundary"]), (9, "output"))
        self.assertEqual(len(r["native_layers"]), 28)
        self.assertTrue(r["native_layers"][10]["input"]["statistical_differences"])
        self.assertGreater(r["compared_fields"]["unobservable"], 0)

    def test_earlier_non_attention_api_can_win(self):
        r = self.pair(first_layer=0, early=True)
        self.assertEqual(r["first_observed_model_api_difference"]["B"]["name"], "Functional.embedding.0.forward")

    def test_diagnostics_keep_unfiltered_first_as_well(self):
        r = self.pair(first_layer=15, dx=True)
        self.assertTrue(r["first_observed_raw_api_difference"]["B"]["diagnostic"])
        self.assertEqual(r["first_observed_model_api_difference"]["B"]["layer"], 15)

    def test_absent_layer_invalid_with_receipt_rewritten_consistently(self):
        with self.assertRaisesRegex(ValueError, "28 ordered"):
            self.pair(missing=22)

    def test_trace_mismatch_suppresses_rank(self):
        r = self.pair(first_layer=0, change_stack=True)
        self.assertEqual(r["status"], "INCONCLUSIVE_TRACE_OR_CONFIG")
        self.assertIsNone(r["first_observed_model_api_difference"])

    def test_earlier_geometry_difference_suppresses_rank(self):
        r = self.pair(first_layer=2, change_meta=1)
        self.assertEqual(r["status"], "INCONCLUSIVE_MODEL_API_INPUT_OR_OUTPUT_METADATA")
        self.assertIsNone(r["first_observed_model_api_difference"])

    def test_identical_statistics_do_not_establish_bit_equality(self):
        r = self.pair()
        self.assertEqual(r["status"], "ALIGNED_NO_OBSERVED_FINITE_MODEL_API_DIFFERENCE")
        self.assertFalse(r["tensor_equality_proven"])

    def test_policy_and_logits_files_never_opened(self):
        f = fixture(self.root, "F", first_layer=3)
        for name in ("result.json", "model_calls.json", "host_events.jsonl", "private_output"):
            (f.parents[3] / name).write_bytes(b"\xff")
        self.assertEqual(cm.compare(self.b, f)["first_observed_model_api_difference"]["B"]["layer"], 3)


if __name__ == "__main__":
    unittest.main()

