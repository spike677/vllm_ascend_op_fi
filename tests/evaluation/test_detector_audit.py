import copy
import json
from pathlib import Path
import tempfile
import unittest

from ascend_fi.evaluation.detector_audit import audit_cases, historical_explanation_cases, interval_union, write_audit

ROOT = Path(__file__).resolve().parents[2]
SUITE = ROOT / "profiles/evaluation/distortion_probe_suite_v1.json"
FIXTURES = Path(__file__).with_name("detector_audit_cases_v1.json")


def fixtures():
    return json.loads(FIXTURES.read_text(encoding="utf-8"))["cases"]


def real_case():
    case = next(c for c in fixtures() if c["case_id"] == "punctuation_spam")
    case["source_kind"] = "recorded_model_output"
    case["protocol"] = "chat_v2"
    case["split"] = "check"
    for name in ("baseline", "response"):
        case[name]["source_ref"] = "TEST_DOUBLE_ONLY/" + name
    case["labels"]["punctuation"] = {
        "source": "independent_human_review", "expected_trigger": True,
        "review_status": "confirmed", "reviewer": "TEST_FIXTURE_REVIEWER_NOT_REAL_GOLD",
        "review_method": "test provenance contract", "independent": True,
        "baseline_eligible": True, "not_used_for_tuning": True,
        "evidence_note": "Only tests the eligibility branch, not real label quality",
        "evidence_spans": [[0, 60]],
    }
    return case


class DetectorAuditTests(unittest.TestCase):
    def test_full_regression_fixtures(self):
        report = audit_cases(fixtures(), SUITE)
        self.assertEqual(report["regression"]["status"], "PASS")
        self.assertEqual(report["case_count"], 12)
        self.assertEqual(report["trusted_label_quality"], {})
        self.assertEqual(report["quality_status"], "INDEPENDENT_LABELS_PENDING")
        self.assertFalse(report["generalization_certified"])
        json.dumps(report, allow_nan=False)

    def test_full_tail_after_excerpt(self):
        case = next(c for c in fixtures() if c["case_id"] == "tail_only_loop")
        report = audit_cases([case], SUITE)
        value = report["cases"][0]
        self.assertGreater(len(value["response"]["response_text"]), 600)
        self.assertGreater(value["locations"]["characters"]["first_observed_offset"], 600)
        self.assertTrue(value["locations"]["not_complete_pathology_coverage"])
        self.assertEqual(value["response"]["response_text"], case["response"]["response_text"])

    def test_interval_union_no_double_count(self):
        report = interval_union([[0, 4], [2, 8], [8, 10], [12, 14]], 20)
        self.assertEqual(report["intervals"], [[0, 10], [12, 14]])
        self.assertEqual(report["covered"], 12)
        self.assertEqual(report["fraction"], .6)
        with self.assertRaises(ValueError):
            interval_union([[0, 30]], 20)
        with self.assertRaises(ValueError):
            interval_union([[True, 3]], 20)

    def test_agent_label_never_becomes_gold(self):
        case = real_case()
        case["labels"]["punctuation"]["source"] = "agent_candidate"
        report = audit_cases([case], SUITE)
        self.assertEqual(report["trusted_label_quality"], {})
        self.assertEqual(report["pending_or_excluded_labels"][0]["reason"], "LABEL_NOT_INDEPENDENT_HUMAN_REVIEW")

    def test_trusted_label_contract_and_null_denominators(self):
        report = audit_cases([real_case()], SUITE)
        quality = report["trusted_label_quality"]["chat_v2:punctuation"]
        self.assertEqual(quality["tp"], 1)
        self.assertEqual(quality["denominator"], 1)
        self.assertEqual(quality["precision"], 1)
        self.assertIsNone(quality["fpr"])
        self.assertEqual(quality, report["independent_check_subset"]["chat_v2:punctuation"])
        self.assertFalse(report["generalization_certified"])

    def test_per_detector_baseline_eligibility(self):
        case = real_case()
        case["labels"]["punctuation"]["baseline_eligible"] = False
        report = audit_cases([case], SUITE)
        self.assertEqual(report["trusted_label_quality"], {})
        self.assertEqual(report["pending_or_excluded_labels"][0]["reason"], "BASELINE_NOT_ELIGIBLE_FOR_THIS_DETECTOR")

    def test_cycles_cannot_cross_split_even_renamed_family_or_protocol(self):
        first = real_case()
        second = copy.deepcopy(first)
        second.update(case_id="cycle2", case_family="another", split="development", protocol="old_completion")
        with self.assertRaisesRegex(ValueError, "split leakage"):
            audit_cases([first, second], SUITE)

    def test_family_cannot_cross_split_even_different_prompt(self):
        first = real_case()
        second = copy.deepcopy(first)
        second.update(case_id="variant", split="development")
        second["probe"]["prompt"] = "another prompt variant in same family"
        with self.assertRaisesRegex(ValueError, "split leakage"):
            audit_cases([first, second], SUITE)

    def test_protocols_stratified_cycles_not_independent_groups(self):
        first = real_case()
        second = copy.deepcopy(first)
        second.update(case_id="cycle2", protocol="old_completion")
        report = audit_cases([first, second], SUITE)
        self.assertEqual(len(report["groups"]), 1)
        self.assertEqual(len(report["trusted_label_quality"]), 2)
        self.assertEqual(report["trusted_label_quality"]["chat_v2:punctuation"]["group_count"], 1)

    def test_exact_cycle_repeats_not_three_quality_samples(self):
        first = real_case()
        second = copy.deepcopy(first)
        second.update(case_id="cycle2", cycle_id="cycle2")
        third = copy.deepcopy(first)
        third.update(case_id="cycle3", cycle_id="cycle3")
        report = audit_cases([first, second, third], SUITE)
        self.assertEqual(report["quality_deduplicated_exact_pairs"], 2)
        self.assertEqual(report["trusted_label_quality"]["chat_v2:punctuation"]["denominator"], 1)

    def test_quality_confusion_matrix_all_four_cells(self):
        cases = []
        for index, (predicted, truth) in enumerate(((True, True), (True, False), (False, True), (False, False))):
            case = real_case()
            case.update(case_id=str(index), case_family=str(index))
            case["probe"]["prompt"] = "fixture prompt " + str(index)
            case["labels"]["punctuation"]["expected_trigger"] = truth
            if not predicted:
                case["response"]["response_text"] = "This is a normal response."
                case["labels"]["punctuation"]["evidence_spans"] = []
            cases.append(case)
        value = audit_cases(cases, SUITE)["trusted_label_quality"]["chat_v2:punctuation"]
        self.assertEqual([value[k] for k in ("tp", "fp", "fn", "tn")], [1, 1, 1, 1])
        self.assertEqual([value[k] for k in ("precision", "recall", "fpr")], [.5, .5, .5])

    def test_no_excerpt_or_unverified_raw_hash(self):
        case = real_case()
        case["response"]["text_is_complete"] = False
        with self.assertRaisesRegex(ValueError, "text_is_complete"):
            audit_cases([case], SUITE)
        case = real_case()
        case["response"]["text_sha256"] = "bad"
        with self.assertRaisesRegex(ValueError, "SHA256"):
            audit_cases([case], SUITE)

    def test_write_does_not_overwrite_and_thresholds_frozen(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "audit.json"
            report = write_audit(FIXTURES, SUITE, output)
            with self.assertRaises(FileExistsError):
                write_audit(FIXTURES, SUITE, output)
            self.assertEqual(json.loads(output.read_text())["scorer_version"], report["scorer_version"])
            changed = json.loads(SUITE.read_text())
            changed["thresholds"]["punctuation_run_min"] = 1
            path = Path(directory) / "retuned.json"
            path.write_text(json.dumps(changed))
            with self.assertRaises(ValueError):
                audit_cases(fixtures(), path)

    def test_answer_and_pathology_are_independent(self):
        report = audit_cases(fixtures(), SUITE)
        answer_loop = next(c for c in report["cases"] if c["case_id"] == "correct_answer_then_loop")
        self.assertTrue(answer_loop["comparison"]["task_outcome"]["correct"])
        self.assertTrue(answer_loop["comparison"]["output_pathology"]["confirmed"])
        budget = next(c for c in report["cases"] if c["case_id"] == "budget_no_answer_no_pathology")
        self.assertEqual(budget["comparison"]["task_outcome"]["status"], "ANSWER_UNDELIVERED_WITHIN_BUDGET")
        self.assertFalse(budget["comparison"]["output_pathology"]["confirmed"])

    def test_existing_historical_full_text_not_old_summary_label(self):
        source = ROOT / "artifacts/distortion/legacy_replay_scoring_v2_final/case_explanation_audit.json"
        if not source.exists():
            self.skipTest("optional historical evidence not in this checkout")
        cases = historical_explanation_cases(source)
        report = audit_cases(cases, SUITE)
        self.assertEqual(report["case_count"], 9)
        self.assertEqual(len(report["groups"]), 3)
        # Text-only export lacks the original token pieces: 0004's loop is
        # masked by pre-existing lexical repetition. Do not alter thresholds or
        # hide this incompatibility by using historical classifications as truth.
        self.assertEqual(report["regression"]["status"], "FAIL")
        self.assertEqual(report["regression"]["passed_assertions"], 6)
        self.assertEqual(report["trusted_label_quality"], {})
        self.assertTrue(all(c["response"]["text_sha256"] for c in report["cases"]))


if __name__ == "__main__":
    unittest.main()
