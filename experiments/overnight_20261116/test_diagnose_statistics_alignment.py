"""Safety regressions for partial raw-statistics diagnostics (no NPU)."""
from copy import deepcopy
import json
import math
from pathlib import Path
import tempfile
import unittest

from diagnose_statistics_alignment import ComparisonPolicy, compact_report, diagnose_groups, metric


POLICY = ComparisonPolicy(1e-8, 1e-5)


def tensor(value=2.0):
    return {"type": "torch.Tensor", "dtype": "torch.float32", "shape": [2, 2],
            "Max": value, "Min": -value, "Mean": None, "Norm": None,
            "requires_grad": False}


def record(value=2.0):
    return {"input_args": [tensor()], "input_kwargs": {}, "output": [tensor(value)]}


def write_group(path, data, *, changed_stack=None, changed_construct=None):
    rank = path / "observation" / "step0" / "rank0"
    rank.mkdir(parents=True)
    documents = {path / "config.json": {"task": "statistics", "level": "mix", "dump_path": str(path),
                    "statistics": {"summary_mode": "statistics"}},
                 path / "collector_identity.json": {"files": {"collector.py": "fixture-only"}},
                 rank / "dump.json": {"task": "statistics", "level": "mix", "framework": "pytorch",
                    "dump_data_dir": None, "data": data},
                 rank / "stack.json": {str(i): [[key], [changed_stack if changed_stack and i == 0 else "same-callsite"]]
                    for i, key in enumerate(data)},
                 rank / "construct.json": {key: changed_construct if i == 0 else None for i, key in enumerate(data)}}
    for file, value in documents.items():
        file.write_text(json.dumps(value), encoding="utf-8")


class AlignmentDiagnosisTests(unittest.TestCase):
    def run_case(self, baseline, fault, **kwargs):
        with tempfile.TemporaryDirectory() as temp:
            b, f = Path(temp) / "b", Path(temp) / "f"
            write_group(b, baseline)
            write_group(f, fault, **kwargs)
            return diagnose_groups(b, f, POLICY)

    def test_unpaired_api_retains_strict_failure_and_local_consumer_change(self):
        b = {"Module.norm.RMSNorm.forward.0": record(), "Tensor.__add__.0.forward": record()}
        f = {"Module.norm.RMSNorm.forward.0": record(), "Tensor.__mul__.0.forward": record(1.0),
             "Tensor.__add__.0.forward": record(1.0)}
        report = self.run_case(b, f)
        self.assertEqual(report["strict_original_result"]["status"], "INCOMPLETE_COMPARISON")
        self.assertIsNone(report["global_first_difference"])
        step = report["steps"][0]
        self.assertEqual(len(step["unpaired"]["fault"]), 1)
        self.assertEqual(len(step["changed_paired_fields"]), 2)
        self.assertTrue(all(row["identity"] == "Tensor.__add__.0.forward/output/0" for row in step["changed_paired_fields"]))
        self.assertTrue(all(row["relative_change"] == 0.5 for row in step["changed_paired_fields"]))

    def test_scalar_change_rejects_pair_without_hiding_values(self):
        b = {"Tensor.mul.0.forward": record()}
        b["Tensor.mul.0.forward"]["input_args"].append({"type": "float", "value": 1.0})
        f = deepcopy(b)
        f["Tensor.mul.0.forward"]["input_args"][1]["value"] = 0.5
        f["Tensor.mul.0.forward"]["output"][0]["Max"] = 1.0
        item = self.run_case(b, f)["steps"][0]["paired_diagnostics"][0]
        self.assertEqual(item["pairing_status"], "REJECTED")
        self.assertEqual(item["nonstatistical_changes"][0]["path"], "Tensor.mul.0.forward/input_args/1/value")
        self.assertEqual(item["tensor_statistics"], [])
        self.assertEqual(item["raw_fault_record"]["output"][0]["Max"], 1.0)

    def test_same_api_extra_occurrence_rejects_ordinal_pairing(self):
        b = {"Tensor.mul.0.forward": record()}
        f = {"Tensor.mul.0.forward": record(1.0), "Tensor.mul.1.forward": record()}
        item = self.run_case(b, f)["steps"][0]["paired_diagnostics"][0]
        self.assertIn("OCCURRENCE_FAMILY_COUNT_MISMATCH", item["reasons"])
        self.assertEqual(item["tensor_statistics"], [])

    def test_context_change_rejects_even_equal_shapes(self):
        for args, reason in (({"changed_stack": "another-callsite"}, "STACK_UNALIGNED_OR_MISSING"),
                             ({"changed_construct": "parent.changed"}, "CONSTRUCT_UNALIGNED_OR_MISSING")):
            with self.subTest(reason=reason):
                report = self.run_case({"Module.a.RMSNorm.forward.0": record()},
                                       {"Module.a.RMSNorm.forward.0": record(1.0)}, **args)
                self.assertIn(reason, report["steps"][0]["paired_diagnostics"][0]["reasons"])

    def test_null_nan_and_zero_baseline_are_not_fabricated_equality(self):
        self.assertEqual(metric(None, None, POLICY)["status"], "UNOBSERVABLE")
        self.assertIsNone(metric(None, None, POLICY)["exact_change"])
        self.assertEqual(metric(math.nan, math.nan, POLICY)["status"], "UNCOMPARABLE")
        zero = metric(0.0, 1.0, POLICY)
        self.assertTrue(zero["exceeds_tolerance"])
        self.assertIsNone(zero["relative_change"])

    def test_fixed_tolerance_keeps_exact_subthreshold_changes(self):
        value = metric(1.0, 1.00000001, POLICY)
        self.assertTrue(value["exact_change"])
        self.assertFalse(value["exceeds_tolerance"])
        self.assertEqual(value["threshold"], 1.001e-5)

    def test_baseline_self_has_no_observable_difference_and_keeps_null(self):
        data = {"Module.a.RMSNorm.forward.0": record()}
        report = self.run_case(data, data)
        self.assertEqual(report["strict_original_result"]["status"], "NO_OBSERVABLE_STATISTICS_DIFFERENCE")
        self.assertEqual(report["steps"][0]["changed_paired_fields"], [])
        self.assertEqual(len(report["steps"][0]["unobservable_paired_fields"]), 4)

    def test_summary_labels_rejected_record_arithmetic_without_global_first(self):
        b = {"Tensor.mul.0.forward": record()}
        b["Tensor.mul.0.forward"]["input_args"].append({"type": "float", "value": 1.0})
        f = deepcopy(b)
        f["Tensor.mul.0.forward"]["input_args"][1]["value"] = 0.5
        f["Tensor.mul.0.forward"]["output"][0]["Max"] = 1.0
        summary = compact_report(self.run_case(b, f), POLICY)
        self.assertIsNone(summary["global_first_difference"])
        self.assertEqual(summary["steps"][0]["changed_paired_fields"], [])
        item = summary["steps"][0]["paired_diagnostics"][0]
        self.assertNotIn("raw_baseline_record", item)
        arithmetic = item["rejected_record_same_path_arithmetic"][0]
        self.assertFalse(arithmetic["record_pairing_accepted"])
        self.assertTrue(arithmetic["arithmetic_only_not_comparison_acceptance"])
        self.assertTrue(arithmetic["exceeds_tolerance"])

    def test_configuration_mismatch_blocks_all_local_pair_acceptance(self):
        with tempfile.TemporaryDirectory() as temp:
            b, f = Path(temp) / "b", Path(temp) / "f"
            write_group(b, {"Module.a.RMSNorm.forward.0": record()})
            write_group(f, {"Module.a.RMSNorm.forward.0": record(1.0)})
            config_path = f / "config.json"
            config = json.loads(config_path.read_text())
            config["precision"] = "changed"
            config_path.write_text(json.dumps(config), encoding="utf-8")
            report = diagnose_groups(b, f, POLICY)
        step = report["steps"][0]
        self.assertEqual(report["strict_original_result"]["steps"][0]["report"]["mismatched"], "config")
        self.assertFalse(step["capture_metadata_aligned"])
        self.assertEqual(step["changed_paired_fields"], [])
        self.assertIn("CAPTURE_CONFIGURATION_OR_COLLECTOR_UNALIGNED", step["paired_diagnostics"][0]["reasons"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
