"""Small CPU checks for the complete model acceptance boundaries."""
import copy
import json
from pathlib import Path
import struct
import tempfile
import unittest

import accept_model as m


def put(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj), encoding="utf-8")


class TestAttemptComposition(unittest.TestCase):
    def setUp(self):
        parent = tempfile.TemporaryDirectory()
        self.addCleanup(parent.cleanup)
        self.root = Path(parent.name)
        self.roots = [self.make_attempt("job" + str(i), [case], i) for i, case in enumerate(m.NAMES)]

    def make_attempt(self, name, cases, slot):
        root = self.root/name
        all_configs = {n: {"case_id": n, "version": "a4"} for n in m.NAMES}
        manifest = {}
        for f in ("controller.py", "model_adapter.py", "model_probe.py", "replay.py", "vendor_exec.py"):
            p = root/"source"/f
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(b"# frozen\n")
            manifest[f] = m.sha(p)
        for n, cfg in all_configs.items():
            f = "setup/" + n + ".json"
            put(root/"source"/f, cfg)
            manifest[f] = m.sha(root/"source"/f)
        put(root/"source_manifest.json", manifest)
        sample = {"eligible": True, "require_selected_card_idle": True, "selected_card_table_idle": True,
                  "selected_card": 5, "unapproved_device_capable_containers": [], "blocking_selected_card_foreign_npu_processes": []}
        records, selected = [], []
        t = 100 + 100 * slot
        for i, case in enumerate(cases):
            f = "setup/" + case + ".json"
            put(root/case/"result.json", {"case_id": case, "case_sha256": manifest[f]})
            put(root/case/"case.json", all_configs[case])
            records.append({"case_id": case, "start": t + 2*i+1, "end": t + 2*i+2, "rc": 0,
                            "status": "CAPTURE_COMPLETED_NOT_MECHANISM_ACCEPTANCE", "result_sha256": m.sha(root/case/"result.json"),
                            "resources": [sample], "pre_case_resources": sample, "post_case_resources": sample})
            selected.append({"case_id": case, "file": f, "case": all_configs[case]})
        cid = m.digest(name.encode())
        put(root/"container_identity.json", {"id": cid})
        put(root/"selected_cases.json", selected)
        put(root/"completion.json", {"card": 5, "capture": "native-logs", "probe": "model_probe.py",
            "status": "CAPTURE_COMPLETED", "owned_container_removed": True, "owned_container_id": cid,
            "cleanup_errors": [], "device_lock_acquired": True, "device_lock_released": True, "external_container_mutations": False,
            "start_epoch": t, "end_epoch": t+2*len(cases)+1, "cases": records})
        return root

    def test_both_supported_layouts_retain_source_and_unique_binding(self):
        split, manifest = m.validate_attempts(self.roots)
        one = self.make_attempt("group", m.NAMES, 20)
        whole, whole_manifest = m.validate_attempts([one])
        self.assertEqual(manifest, whole_manifest)
        self.assertEqual(tuple(split), m.NAMES)
        self.assertEqual([whole[n][0] for n in m.NAMES], [one]*4)
        self.assertEqual([split[n][0] for n in m.NAMES], self.roots)

    def test_reject_failed_parent_even_with_successful_selected_child(self):
        path = self.roots[0]/"completion.json"
        receipt = m.read(path)
        receipt["status"] = "FAIL"
        put(path, receipt)
        with self.assertRaisesRegex(ValueError, "finish cleanly"):
            m.validate_attempts(self.roots)

    def test_reject_duplicate_missing_and_out_of_order(self):
        first, second, third, last = self.roots
        dup = self.make_attempt("duplicate", [m.NAMES[1]], 3)
        for roots in ([first,second,third], [first,second,second,last], [first,second,dup,last], [first,third,second,last]):
            with self.subTest(roots=roots), self.assertRaises(ValueError):
                m.validate_attempts(roots)

    def test_reject_overlapping_attempt_times(self):
        path = self.roots[1]/"completion.json"
        receipt = m.read(path)
        receipt["start_epoch"] = 102
        put(path, receipt)
        with self.assertRaisesRegex(ValueError, "overlap"):
            m.validate_attempts(self.roots)

    def test_reject_drift_even_in_unselected_frozen_case(self):
        root = self.roots[-1]
        f = "setup/" + m.NAMES[0] + ".json"
        put(root/"source"/f, {"case_id": m.NAMES[0], "version": "different"})
        manifest = m.read(root/"source_manifest.json")
        manifest[f] = m.sha(root/"source"/f)
        put(root/"source_manifest.json", manifest)
        with self.assertRaisesRegex(ValueError, "manifest differs between"):
            m.validate_attempts(self.roots)

    def test_reject_extra_source_unbound_result_and_failed_gate(self):
        root, case = self.roots[-1], m.NAMES[-1]
        (root/"source"/"added.py").write_text("extra", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "source inventory"):
            m.validate_attempts(self.roots)
        path = self.roots[0]/"completion.json"
        receipt = m.read(path)
        receipt["cases"][0]["resources"][0]["unapproved_device_capable_containers"] = ["foreign"]
        put(path, receipt)
        with self.assertRaisesRegex(ValueError, "selected-card resource"):
            m.validate_attempts(self.roots)
        put(self.roots[0]/m.NAMES[0]/"result.json", {"status": "replaced"})
        with self.assertRaisesRegex(ValueError, "Unbound model result"):
            m.validate_attempts(self.roots)


def value(s):
    return {"sha256": s, "request_sha256": [s]*8}


def layer(i):
    return {"index": i, "visible": {"key": 10}, "plan": {"typed": {"d": 1}, "workspace_sizes": [8], "selected_inner": 2048,
            "outer_update_loops": 8, "pre_intervention_native_inner": 2048, "raw_sha256": "a"},
            "values": {"query": "q", "kv": "kv", "attention": value("a"), "o_proj": value("o"), "o_proj_input": value("a")}}


def trace():
    case = "model_fault_r0"
    calls = []
    rows = []
    def event(tag, **fields):
        obj = {"fault33": tag, "pid": 17, "tid": 18, "case_id": case, **fields}
        rows.append(json.dumps(obj))
        return obj
    event("model_prefill_enter")
    event("model_prefill_exit")
    event("model_decode_enter")
    for i in range(28):
        event("module_enter", request_ids=m.REQUESTS, layer_index=i)
        row = {"pid": 17, "enter": event("enter", call_id=f"decode_layer_{i:02d}", layer_index=i)}
        rows.append(f"OP(17,python):[NnopbaseExecutorLaunchKernel][20] OpName:[IncreFlashAttention_{i}] "
                    "Launch dynamic kernel IncreFlashAttention task, tilingKey is 3, scheMode is 1, numBlocks is 20")
        row["exit"] = event("exit", call_id=f"decode_layer_{i:02d}", layer_index=i)
        calls.append(row)
        event("module_exit", layer_index=i)
    event("model_decode_exit")
    return case, rows, calls


class TestModelAcceptance(unittest.TestCase):
    def test_all_vocab_top1_finite_ties_and_signed_zero(self):
        a = struct.pack("<32e", *( [2, 2, 1, -0.]*8 ))
        b = struct.pack("<32e", *( [2, 3, 1, 0.]*8 ))
        result = m.compare_logits(a, b, "torch.float16", vocab=4)
        self.assertEqual(result["elements"], 32)
        self.assertEqual(result["numeric_differences"], 8)
        self.assertEqual(result["top1_changed_requests"], 8)
        for row in result["rows"]:
            self.assertEqual((row["B_top1"], row["B_argmax_ties"], row["F_top1"], row["signed_zero_only"]), (0, 2, 1, 1))

    def test_all_rows_complete_bitwise_equal_fp32(self):
        a = struct.pack("<24f", *range(24))
        self.assertTrue(m.compare_logits(a, a, "torch.float32", vocab=3)["bitwise_equal"])
        with self.assertRaisesRegex(ValueError, "geometry"):
            m.compare_logits(a[:-1], a, "torch.float32", vocab=3)

    def test_nonfinite_output_rejected(self):
        raw = struct.pack("<8e", float("inf"), *[1.]*7)
        with self.assertRaisesRegex(ValueError, "Nonfinite"):
            m.compare_logits(raw, raw, "torch.float16", vocab=1)

    def test_private_file_validates_all_eight_actual_row_digests(self):
        raw = bytes(range(24))
        meta = {"bytes": 24, "sha256": m.digest(raw), "request_sha256": [m.digest(raw[i:i+3]) for i in range(0, 24, 3)]}
        m.validate_payload(meta, raw)
        meta["request_sha256"].reverse()
        with self.assertRaisesRegex(ValueError, "row hashes"):
            m.validate_payload(meta, raw)

    def test_propagated_later_layer_inputs_can_differ(self):
        b = [layer(0), layer(1)]
        f = copy.deepcopy(b)
        f[0]["values"]["attention"] = value("f")
        f[1]["values"]["kv"] = "new"
        for row in f:
            row["plan"].update(selected_inner=1024, outer_update_loops=16)
        reports = m.compare_layers(b, f, False)
        self.assertEqual(reports[0]["changed_requests"]["attention"], m.REQUESTS)
        self.assertFalse(reports[1]["full_and_per_row_QKV_hashes_equal"])
        self.assertTrue(reports[0]["verified_partition_changed"])

    def test_changed_input_before_layer_zero_rejected(self):
        a, b = layer(0), layer(0)
        b["values"]["query"] = "bad"
        with self.assertRaisesRegex(ValueError, "first-layer"):
            m.compare_layers([a], [b], False)

    def test_repeat_must_match_projected_outputs_and_native_plan(self):
        a, b = layer(0), layer(0)
        m.compare_layers([a], [b], True)
        b["values"]["o_proj"] = value("other")
        with self.assertRaisesRegex(ValueError, "Same-mode"):
            m.compare_layers([a], [b], True)
        b = layer(0)
        b["plan"]["raw_sha256"] = "different non-projected bytes"
        with self.assertRaisesRegex(ValueError, "Same-mode"):
            m.compare_layers([a], [b], True)
        b = layer(0)
        b["plan"]["typed"]["d"] = 2
        with self.assertRaisesRegex(ValueError, "Same-mode"):
            m.compare_layers([a], [b], True)

    def test_full_log_rejects_out_of_scope_duplicate_missing_and_unparsed_launch(self):
        case, rows, calls = trace()
        launch = next(x for x in rows if x.startswith("OP("))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/"case.log"
            def check(lines):
                path.write_text("\n".join(lines)+"\n", encoding="utf-8")
                m.coverage_log(path, calls, case)
            check(rows)
            for lines in ([*rows, launch], [r for r in rows if r != launch], [launch, *rows],
                          [r.replace("scheMode is", "differentVersion says") if r == launch else r for r in rows]):
                with self.subTest(), self.assertRaises(ValueError):
                    check(lines)


if __name__ == "__main__":
    unittest.main()

