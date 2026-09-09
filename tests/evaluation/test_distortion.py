"""Frozen-rule fixtures. Never generate an expected result from model outputs."""

import copy
import json
import tempfile
import unittest
from pathlib import Path

from ascend_fi.contracts import EvaluationResult
from ascend_fi.evaluation.distortion.contracts import DetectorResult, DistortionProbe, FaultConfigurationRecord, content_hash, load_suite
from ascend_fi.evaluation.distortion.detectors import DETECTORS
from ascend_fi.evaluation.distortion.evaluator import compare, evaluate_cycles
from ascend_fi.evaluation.distortion.live import collect_cycles
from ascend_fi.evaluation.distortion.mechanism import cleanup_gate, validate_mechanism
from ascend_fi.evaluation.distortion.report import render_dashboard, write_report


ROOT = Path(__file__).resolve().parents[2]
SUITE = load_suite(ROOT / "profiles/evaluation/distortion_probe_suite_v1.json")


def config():
    return FaultConfigurationRecord.from_dict({
        "capability": "FakeRequant", "strategy": "REFERENCE_REQUANT", "module_families": ["MLP_DOWN"],
        "actual_module_names": ["model.layers.0.mlp.down_proj", "model.layers.1.mlp.down_proj"],
        "targeted_layers": [0, 1], "available_layers": 2, "targeted_ranks": [0], "world_size": 2,
        "hit_policy": {"max_hits": 1}, "config_hash": "fixture-only",
        "frozen_arm_config": {"mode": "REFERENCE_REQUANT", "scale_provider": "tensor_absmax", "scale_multiplier": 0.25, "max_hits": 1},
        "future_field": {"preserve": True},
    })


def envelope(results):
    return {"http_status": 200, "results": results}


def clean():
    return envelope([{"pid": 100+r, "installed": False, "enabled": False} for r in range(2)])


def fault_row(row):
    row = copy.deepcopy(row)
    workers = []
    for rank in (0, 1):
        modules = [{"module": f"model.layers.{layer}.mlp.down_proj", "layer": layer, "tp_rank": rank, "tp_size": 2} for layer in range(2)]
        armed = {**config().extra["frozen_arm_config"], "request_generation": 8}
        workers.append({"pid": 100+rank, "matched_modules": modules, "installed": True, "developer_mode": True,
                        "armed_config": armed, "hit_count": 1 if rank == 0 else 0,
                        "reject_reasons": {} if rank == 0 else {"rank_filter": 1},
                        "tensor_effect_history": [{"changed_elements": 8, "total_elements": 16, "hit_index": 1,
                                                   "rank": 0, "module": modules[0]["module"], "layer": 0,
                                                   "mode": "REFERENCE_REQUANT", "scale_provider": "tensor_absmax", "scale_multiplier": 0.25}] if rank == 0 else []})
    transaction = envelope(copy.deepcopy(workers))
    transaction["disarm"] = envelope([{"ok": True, "disarmed": True, "pid": 100+r} for r in (0, 1)])
    row["fault_transaction"] = {"measurement_query": envelope(workers), "transaction_query": transaction,
                                "arm": envelope([{"ok": True, "pid": w["pid"], "armed_config": w["armed_config"]} for w in workers])}
    return row


def response(value, *, count=None, finish="stop", status=200):
    return {"response_text": value, "token_pieces": value.split(), "token_ids": None,
            "generation_length": len(value.split()) if count is None else count,
            "http_status": status, "finish_reason": finish}


def probe(name="long", expected=None, detectors=None):
    return DistortionProbe(name, "test", "test", {"max_tokens": 128}, expected or {},
                           tuple(detectors or ("repetition", "lexical_collapse", "punctuation", "length", "garbage", "crash")))


def base_for(p):
    expected = p["expected"]
    if "answer" in expected:
        return ("FINAL_ANSWER=" if expected.get("answer_type") == "number" else "") + expected["answer"]
    if expected.get("structure") == "json":
        return '{"name":"sample","count":3,"active":true}'
    if "items" in expected:
        return "\n".join(map(str, expected["items"]))
    return " ".join(f"word{i}" for i in range(96))


def cycles(distort=("exact_math", "exact_fact", "structure_json", "structure_list"), n=3):
    out = []
    for index in range(n):
        pairs = {}
        for p in SUITE["probes"]:
            baseline = response(base_for(p))
            baseline.update(sample_id=p["probe_id"], request={"prompt": p["prompt"], **p["generation_config"]})
            f = copy.deepcopy(baseline)
            if p["probe_id"] in distort:
                f.update(response("FINAL_ANSWER=0" if "answer" in p["expected"] else "broken"))
            pairs[p["probe_id"]] = {"baseline": baseline, "fault": fault_row(f), "recovery": copy.deepcopy(baseline)}
        out.append({"cycle_id": f"cycle_{index+1:02d}", "pairs": pairs, "before_baseline": clean(),
                    "after_remove": clean(), "service_healthy_before": True, "service_healthy": True})
    return out


class DistortionTests(unittest.TestCase):
    def test_fault_configuration_record(self):
        self.assertEqual(config(), FaultConfigurationRecord.from_json(config().to_json()))
        self.assertTrue(config().to_dict()["future_field"]["preserve"])

    def test_detector_unknown_roundtrip(self):
        value = DetectorResult("test", None, None, None, "UNKNOWN", "missing", extra={"future": 8})
        self.assertEqual(value, DetectorResult.from_json(value.to_json()))

    def test_recomputed_modified_suite_is_not_v1(self):
        modified = copy.deepcopy(SUITE)
        modified["thresholds"]["seq_rep_4_min"] = 0
        modified["hash"] = content_hash(modified)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/"suite.json"
            path.write_text(json.dumps(modified), encoding="utf-8")
            with self.assertRaises(ValueError):
                load_suite(path)

    def test_suite_frozen(self):
        self.assertEqual(SUITE["hash"], "abfc3c9a26597e3eefc7204306553dd381353c45dfe5b3d135632bbd8856d502")
        self.assertEqual(16, len(SUITE["probes"]))
        self.assertEqual(4, sum(p["category"] == "structured" for p in SUITE["probes"]))

    def test_mechanism_gate_requires_real_change(self):
        row = fault_row(response("ok"))
        self.assertEqual("PASS", validate_mechanism(row, config(), True)["status"])
        row["fault_transaction"]["measurement_query"]["results"][0]["tensor_effect_history"][0]["changed_elements"] = 0
        self.assertEqual("INVALID", validate_mechanism(row, config(), True)["status"])

    def test_historical_replay_missing_field_unknown(self):
        row = fault_row(response("ok"))
        del row["fault_transaction"]["measurement_query"]["results"][0]["tensor_effect_history"][0]["changed_elements"]
        self.assertEqual("UNKNOWN", validate_mechanism(row, config(), True)["status"])
        self.assertNotEqual("PASS", validate_mechanism({"hit_count": 4096, "changed_elements": 5}, config(), True)["status"])

    def test_rank_filter_required(self):
        row = fault_row(response("ok"))
        row["fault_transaction"]["measurement_query"]["results"][1]["reject_reasons"] = {}
        self.assertEqual("UNKNOWN", validate_mechanism(row, config(), True)["status"])

    def test_rank_identity_not_result_order(self):
        row = fault_row(response("ok"))
        row["fault_transaction"]["measurement_query"]["results"].reverse()
        self.assertEqual("PASS", validate_mechanism(row, config(), True)["status"])

    def test_unexpected_rank_hit_invalid(self):
        row = fault_row(response("ok"))
        row["fault_transaction"]["measurement_query"]["results"][1]["hit_count"] = 1
        self.assertEqual("INVALID", validate_mechanism(row, config(), True)["status"])

    def test_module_layer_pair_mismatch_invalid(self):
        row = fault_row(response("ok"))
        row["fault_transaction"]["measurement_query"]["results"][0]["tensor_effect_history"][0]["layer"] = 1
        self.assertEqual("INVALID", validate_mechanism(row, config(), True)["status"])

    def test_transaction_cannot_duplicate_target_rank(self):
        row = fault_row(response("ok"))
        tx = row["fault_transaction"]["transaction_query"]
        tx["results"][1] = copy.deepcopy(tx["results"][0])
        self.assertEqual("INVALID", validate_mechanism(row, config(), True)["status"])

    def test_unhealthy_or_armed_cleanup_cannot_pass(self):
        self.assertEqual("FAIL", cleanup_gate(clean(), False, 2)["status"])
        observed = clean()
        observed["results"][0]["enabled"] = True
        self.assertEqual("FAIL", cleanup_gate(observed, True, 2)["status"])

    def test_frozen_strength_not_silently_changed(self):
        row = fault_row(response("ok"))
        row["fault_transaction"]["measurement_query"]["results"][0]["armed_config"]["scale_multiplier"] = 0.5
        self.assertEqual("INVALID", validate_mechanism(row, config(), True)["status"])

    def test_padding_does_not_rescue_zero_measurement(self):
        row = fault_row(response("ok"))
        w = row["fault_transaction"]["measurement_query"]["results"][0]
        w["hit_count"], w["tensor_effect_history"] = 0, []
        self.assertEqual("INVALID", validate_mechanism(row, config(), True)["status"])

    def test_diverged_but_valid_not_distorted(self):
        p = probe(expected={"answer": "42", "answer_type": "number"}, detectors=["task_answer", "crash"])
        result = compare(p, response("The answer is 42."), response("FINAL_ANSWER=42"), SUITE["thresholds"])
        self.assertEqual("DIVERGED_VALID", result["verdict"])

    def test_wrong_answer_detector(self):
        p = probe(expected={"answer": "42", "answer_type": "number"}, detectors=["task_answer"])
        result = DETECTORS["task_answer"].evaluate(p, response("42"), response("41"), {})
        self.assertEqual("WRONG_ANSWER", result.phenotype)

    def test_unresolved_detector_first_marker(self):
        p = probe(expected={"answer": "42", "answer_type": "number"}, detectors=["task_answer"])
        d = DETECTORS["task_answer"]
        self.assertEqual("UNRESOLVED", d.evaluate(p, response("42"), response("thinking"), {}).phenotype)
        self.assertFalse(d.evaluate(p, response("42"), response("FINAL_ANSWER=42\nNext question\nFINAL_ANSWER=7"), {}).triggered)

    def test_structure_sequence_detector(self):
        p = probe(expected={"structure": "number_sequence", "items": [1, 2, 3]}, detectors=["structure"])
        result = DETECTORS["structure"].evaluate(p, response("1\n2\n3"), response("1 1 3"), {})
        self.assertTrue(result.triggered)
        self.assertIn("NEWLINE_COLLAPSE", result.fault_value["errors"])
        self.assertIn("MISSING_ITEM", result.fault_value["errors"])
        self.assertIn("DUPLICATE_ITEM", result.fault_value["errors"])

    def test_json_structure_detector(self):
        p = DistortionProbe.from_dict(next(p for p in SUITE["probes"] if p["probe_id"] == "structure_json"))
        b = response('{"name":"a","count":1,"active":true}')
        for wrong in ('{bad', '{"name":"a","count":true,"active":true}', '{"name":"a","count":1}', '{"name":"a","count":1,"active":true,"extra":2}'):
            self.assertTrue(DETECTORS["structure"].evaluate(p, b, response(wrong), {}).triggered)

    def test_repetition_detector_baseline_relative(self):
        d, p = DETECTORS["repetition"], probe()
        healthy, repeated = response(" ".join(f"x{i}" for i in range(96))), response("red green blue yellow "*24)
        self.assertTrue(d.evaluate(p, healthy, repeated, {"thresholds": SUITE["thresholds"]}).triggered)
        self.assertFalse(d.evaluate(p, repeated, repeated, {"thresholds": SUITE["thresholds"]}).triggered)

    def test_lexical_collapse_detector(self):
        result = DETECTORS["lexical_collapse"].evaluate(probe(), response(" ".join(f"x{i}" for i in range(96))), response("word "*96), {"thresholds": SUITE["thresholds"]})
        self.assertTrue(result.triggered)

    def test_punctuation_collapse_detector(self):
        d = DETECTORS["punctuation"]
        self.assertTrue(d.evaluate(probe(), response("A useful reply."), response(",,,,,,,,,,,,,,"), {"thresholds": SUITE["thresholds"]}).triggered)
        self.assertFalse(d.evaluate(probe(), response(",,,,,,,,,,,,,,"), response(",,,,,,,,,,,,,,"), {"thresholds": SUITE["thresholds"]}).triggered)

    def test_early_stop_detector(self):
        d = DETECTORS["length"]
        self.assertEqual("EARLY_TERMINATION", d.evaluate(probe(), response("text", count=100), response("text", count=30), {"thresholds": SUITE["thresholds"]}).phenotype)

    def test_runaway_detector(self):
        d = DETECTORS["length"]
        self.assertEqual("RUNAWAY", d.evaluate(probe(), response("text", count=64), response("text", count=128, finish="length"), {"thresholds": SUITE["thresholds"]}).phenotype)

    def test_garbage_heuristic_not_rare_language(self):
        d = DETECTORS["garbage"]
        self.assertFalse(d.evaluate(probe(), response("English"), response("中文 日本語 العربية"), {"thresholds": SUITE["thresholds"]}).triggered)
        self.assertTrue(d.evaluate(probe(), response("English"), response("\ufffd\ufffd"), {"thresholds": SUITE["thresholds"]}).triggered)

    def test_crash_separate_from_distortion(self):
        inputs = cycles()
        inputs[0]["pairs"]["exact_math"]["fault"]["http_status"] = 500
        d = evaluate_cycles(SUITE, config(), inputs).extra["distortion"]
        self.assertEqual(1, d["crash_count"])
        self.assertFalse(d["robust_conditions"]["no_crash"])
        self.assertFalse(d["cycles"][0]["matrix"][0]["visible_distortion"])

    def test_same_probe_repeatability(self):
        inputs = cycles()
        inputs[1] = cycles(distort=("exact_logic", "exact_copy", "structure_table", "structure_numbers"))[1]
        d = evaluate_cycles(SUITE, config(), inputs).extra["distortion"]
        self.assertEqual([], d["stable_distorted_probes"])

    def test_phenotype_breadth(self):
        d = evaluate_cycles(SUITE, config(), cycles()).extra["distortion"]
        self.assertEqual({"WRONG_ANSWER", "STRUCTURE_FAILURE"}, set(d["stable_phenotype_families"]))

    def test_recovery_only_for_fault_induced_distortion(self):
        inputs = cycles()
        for cycle in inputs:
            for phase in ("baseline", "fault", "recovery"):
                cycle["pairs"]["exact_copy"][phase].update(response("already wrong"))
        d = evaluate_cycles(SUITE, config(), inputs).extra["distortion"]
        self.assertEqual(12, d["recovery"]["fault_induced_pairs"])
        self.assertEqual("PASS", d["recovery"]["gate"])

    def test_robust_visible_distortion_gate_pass(self):
        self.assertEqual("ROBUST_VISIBLE_DISTORTION_PASS", evaluate_cycles(SUITE, config(), cycles()).extra["distortion"]["robust_gate"])

    def test_robust_visible_distortion_gate_fail(self):
        d = evaluate_cycles(SUITE, config(), cycles(distort=("exact_math",))).extra["distortion"]
        self.assertEqual("ROBUST_VISIBLE_DISTORTION_FAIL", d["robust_gate"])

    def test_legacy_cannot_pass_new_suite(self):
        suite = copy.deepcopy(SUITE)
        suite["version"] = "LEGACY_FAKE_REQUANT_EVAL_SET"
        suite["hash"] = content_hash(suite)
        self.assertFalse(evaluate_cycles(suite, config(), cycles()).extra["distortion"]["robust_conditions"]["frozen_16_probe_suite"])

    def test_unknown_detector_does_not_mean_healthy(self):
        self.assertIsNone(compare(probe(), {}, {}, SUITE["thresholds"])["verdict"])

    def test_incomplete_cycle_cannot_be_stable(self):
        inputs = cycles()
        del inputs[0]["pairs"]["open_technical"]
        d = evaluate_cycles(SUITE, config(), inputs).extra["distortion"]
        self.assertEqual([], d["stable_distorted_probes"])

    def test_request_drift_invalidates_pair(self):
        inputs = cycles()
        inputs[0]["pairs"]["exact_math"]["fault"]["request"]["max_tokens"] = 99
        d = evaluate_cycles(SUITE, config(), inputs).extra["distortion"]
        self.assertFalse(d["cycles"][0]["matrix"][0]["eligible"])

    def test_duplicate_cycle_not_repeatability(self):
        inputs = cycles()
        inputs[1]["cycle_id"] = inputs[0]["cycle_id"]
        with self.assertRaises(ValueError):
            evaluate_cycles(SUITE, config(), inputs)

    def test_legacy_analysis_compatibility(self):
        result = evaluate_cycles(SUITE, config(), cycles(), analysis={"ppl_ratio": 1.5, "legacy_level": "L4"})
        self.assertEqual(result.to_dict(), EvaluationResult.from_json(result.to_json()).to_dict())
        self.assertEqual("L4", result.extra["analysis_details"]["legacy_level"])

    def test_analysis_does_not_drive_pass(self):
        d = evaluate_cycles(SUITE, config(), cycles(distort=()), analysis={"ppl_ratio": 1000, "legacy_level": "L6"}).extra["distortion"]
        self.assertEqual("ROBUST_VISIBLE_DISTORTION_FAIL", d["robust_gate"])

    def test_dashboard_render(self):
        inputs = cycles()
        inputs[0]["pairs"]["exact_math"]["fault"]["response_text"] = '</pre><script>alert("x")</script>'
        result = evaluate_cycles(SUITE, config(), inputs)
        page = render_dashboard(result)
        self.assertIn("HOW MUCH DID WE INJECT?", page)
        self.assertIn("WHAT DID USERS ACTUALLY SEE?", page)
        self.assertIn('id="filter"', page)
        self.assertNotIn('<script>alert', page)
        self.assertIn('&lt;script&gt;', page)
        with tempfile.TemporaryDirectory() as folder:
            paths = write_report(folder, result, {"fixture": True})
            self.assertTrue(all(Path(p).is_file() for p in paths.values()))
            hits = [json.loads(x) for x in Path(paths["distortion_hits.jsonl"]).read_text(encoding="utf-8").splitlines()]
            self.assertTrue(all(x["eligible"] and x["visible_distortion"] for x in hits))
            with self.assertRaises(FileExistsError):
                write_report(folder, result, {})

    def test_dashboard_preserves_newline_without_git_trailing_space(self):
        inputs = cycles()
        inputs[0]["pairs"]["exact_math"]["fault"]["response_text"] = "with space \nnext line"
        page = render_dashboard(evaluate_cycles(SUITE, config(), inputs))
        self.assertIn("with space &#10;next line", page)
        self.assertTrue(all(line.rstrip() == line for line in page.splitlines()))

    def test_live_adapter_cleanup_on_fault_exception(self):
        class Owner:
            cleanups = 0
            exited = False
            def __enter__(self): return self
            def __exit__(self, *args): self.exited = True
            def preflight(self): return {"status": "PASS", "exclusive_lock_owned": True}
            def clean_query(self): return clean()
            def service_healthy(self): return True
            def request(self, *args): return response("baseline")
            def install(self, *args): pass
            def fault_request(self, *args): raise RuntimeError("request failed")
            def disarm_and_remove(self): self.cleanups += 1
        owner = Owner()
        with self.assertRaisesRegex(RuntimeError, "request failed"):
            collect_cycles(owner, SUITE, config(), 3)
        self.assertEqual(2, owner.cleanups)
        self.assertTrue(owner.exited)


if __name__ == "__main__":
    unittest.main()
