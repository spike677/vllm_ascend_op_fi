"""CPU fixtures for the native paged IFA statistics boundary. No framework import."""
import json
import tempfile
import unittest
from pathlib import Path

import msprobe_compare as analysis

API = "NPU.npu_incre_flash_attention.0.forward"
MODULE = "Module.replay.ReplayModule.forward.0"
CONSUMER = "Torch.add.0.forward"


def write(path, obj):
    path.write_text(json.dumps(obj), encoding="utf-8")


def tensor(shape, mean=0.25, dtype="torch.float16"):
    return {"type": "torch.Tensor", "shape": shape, "dtype": dtype,
            "Max": 1, "Min": -1, "Mean": mean, "Norm": 2, "requires_grad": False}


def fixture(root, case, *, step=0, bs=128, batch=8, mean=0.25, input_mean=0.25,
            module_mean=None, consumer_input=None):
    case_dir = root / case
    rank = case_dir / "observation" / f"step{step}" / "rank0"
    rank.mkdir(parents=True)
    write(case_dir / "config.json", {"task": "statistics", "level": "mix", "async_dump": False,
          "rank": [], "step": [0, 1], "precision": "low", "dump_path": str(case_dir),
          "statistics": {"list": [], "scope": [], "tensor_list": [], "summary_mode": "statistics", "data_mode": ["all"]}})
    q = tensor([batch, 16, 1, 128], input_mean)
    cache = tensor([batch * (16384 // bs), bs, 8, 128])
    table = tensor([batch, 16384 // bs], None, "torch.int32")
    kwargs = {"block_size": {"type": "int", "value": bs}, "block_table": table,
              "actual_seq_lengths": [{"type": "int", "value": 16384}] * batch}
    original = {"input_args": [q, cache, cache], "input_kwargs": kwargs, "output": [tensor(q["shape"], mean)]}
    parent = {"input_args": [q], "output": [tensor(q["shape"], mean if module_mean is None else module_mean)]}
    consumer = {"input_args": [tensor(q["shape"], mean if consumer_input is None else consumer_input)],
                "output": [tensor(q["shape"], mean)]}
    dump = {"task": "statistics", "level": "mix", "framework": "pytorch", "dump_data_dir": None,
            "data": {MODULE: parent, API: original, CONSUMER: consumer}}
    write(rank / "dump.json", dump)
    write(rank / "construct.json", {MODULE: None, API: MODULE, CONSUMER: MODULE})
    write(rank / "stack.json", {str(i): [[key], [f"File /audit/replay.py, line {15+i}, in forward, \n original()"]]
                                for i, key in enumerate(dump["data"])})
    return rank / "dump.json"


def modify(path, fn):
    obj = json.loads(path.read_text(encoding="utf-8"))
    fn(obj)
    write(path, obj)


class TestPair(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.b = fixture(self.root, "B")

    def pair(self, **kwargs):
        return analysis.compare(self.b, fixture(self.root, "F", **kwargs))

    def test_native_output_precedes_module_and_consumer_localization(self):
        result = self.pair(step=1, mean=0.25000003)
        first = result["first_observed_api_difference"]
        self.assertEqual((first["B_node"], first["boundary"]), (API, "output"))
        self.assertEqual(first["field_differences"][0]["path"], "$[0].Mean")
        self.assertTrue(result["native_api"]["input"]["unobservable"])  # Integer-table Mean is null.
        self.assertFalse(result["physical_fault_location_proven"])

    def test_actual_observed_input_difference_is_first(self):
        result = self.pair(input_mean=0.5, mean=0.375)
        self.assertEqual(result["first_observed_api_difference"]["boundary"], "input")

    def test_later_consumer_can_be_first_observable(self):
        result = self.pair(consumer_input=0.5)
        first = result["first_observed_api_difference"]
        self.assertEqual((first["B_node"], first["boundary"]), (CONSUMER, "input"))

    def test_module_difference_is_not_ascribed_to_native_api(self):
        result = self.pair(module_mean=0.1)
        self.assertEqual(result["status"], "ALIGNED_MODULE_ONLY_DIFFERENCE")
        self.assertIsNone(result["first_observed_api_difference"])

    def test_no_difference_with_null_does_not_prove_bit_equality(self):
        result = self.pair()
        self.assertEqual(result["status"], "ALIGNED_NO_OBSERVED_FINITE_API_DIFFERENCE")
        self.assertGreater(result["unobservable_fields_in_aligned_rows"], 0)
        self.assertFalse(result["tensor_equality_proven"])

    def test_cross_block_size_suppresses_first_difference(self):
        result = self.pair(bs=512, mean=0.5)
        self.assertEqual(result["status"], "INCONCLUSIVE_NATIVE_INPUT_METADATA")
        self.assertFalse(result["native_api"]["output_comparable"])
        self.assertIsNone(result["first_observed_api_difference"])

    def test_cross_batch_suppresses_first_difference(self):
        self.assertEqual(self.pair(batch=1)["status"], "INCONCLUSIVE_NATIVE_INPUT_METADATA")

    def test_scalar_block_size_change_counts_even_with_matching_tensor_shapes(self):
        f = fixture(self.root, "F", mean=0.5)
        modify(f, lambda obj: obj["data"][API]["input_kwargs"]["block_size"].update(value=512))
        self.assertEqual(analysis.compare(self.b, f)["status"], "INCONCLUSIVE_NATIVE_INPUT_METADATA")

    def test_stack_line_change_is_inconclusive(self):
        f = fixture(self.root, "F", mean=0.5)
        modify(f.with_name("stack.json"), lambda obj: obj["1"][1].append("File /other.py, line 4, in f, \n native()"))
        result = analysis.compare(self.b, f)
        self.assertEqual(result["status"], "INCONCLUSIVE_TRACE_OR_CONFIG")
        self.assertIsNone(result["first_observed_api_difference"])

    def test_module_alone_is_not_the_native_api(self):
        f = fixture(self.root, "F")
        modify(f, lambda obj: obj["data"].pop(API))
        with self.assertRaisesRegex(ValueError, "exactly one"):
            analysis.compare(self.b, f)

    def test_missing_real_parent_rejected(self):
        f = fixture(self.root, "F")
        modify(f.with_name("construct.json"), lambda obj: obj.update({API: None}))
        with self.assertRaisesRegex(ValueError, "Module parent"):
            analysis.compare(self.b, f)

    def test_no_mutating_void_api_substitution(self):
        f = fixture(self.root, "F")
        modify(f, lambda obj: obj["data"][API].update(output=[None]))
        with self.assertRaisesRegex(ValueError, "single Tensor"):
            analysis.compare(self.b, f)

    def test_tensor_payload_not_used(self):
        f = fixture(self.root, "F")
        modify(f, lambda obj: obj["data"][API]["output"][0].update(data_name="private.pt"))
        with self.assertRaisesRegex(ValueError, "payload"):
            analysis.compare(self.b, f)

    def test_high_precision_requires_disclosure(self):
        f = fixture(self.root, "F")
        modify(f.parents[3] / "config.json", lambda obj: obj.update(precision="high"))
        with self.assertRaisesRegex(ValueError, "collector provenance"):
            analysis.compare(self.b, f)

    def test_profile_hit_and_private_arrays_excluded(self):
        f = fixture(self.root, "F")
        for name in ("result.json", "host_events.jsonl", "target.bin", "case.log"):
            (f.parents[3] / name).write_bytes(b"\xff\xffnot-a-json-document")
        self.assertIsNone(analysis.compare(self.b, f)["first_observed_api_difference"])

    def test_nonfinite_and_missing_are_unobservable_and_json_safe(self):
        f = fixture(self.root, "F")
        modify(f, lambda obj: obj["data"][API]["output"][0].update(Mean=float("nan")))
        modify(f, lambda obj: obj["data"][API]["output"][0].pop("Max"))
        result = analysis.compare(self.b, f)
        self.assertIsNone(result["first_observed_api_difference"])
        self.assertEqual(len(result["native_api"]["output"]["unobservable"]), 2)
        json.dumps(result, allow_nan=False)


if __name__ == "__main__":
    unittest.main()

