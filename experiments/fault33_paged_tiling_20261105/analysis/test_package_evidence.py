"""Publication-boundary CPU checks. Test inputs are synthetic, unrelated to real captures."""
import hashlib
import json
from pathlib import Path
import tarfile
import tempfile
import unittest
from unittest.mock import patch

import package_evidence as pkg


class TestArchive(unittest.TestCase):
    def setUp(self):
        self.t = tempfile.TemporaryDirectory()
        self.addCleanup(self.t.cleanup)
        self.tmp = Path(self.t.name)
        self.src = self.tmp/"source"
        self.src.mkdir()

    def fixture(self):
        s = pkg.Selection()
        s.roots.append(self.src)
        cases = list(pkg.GROUPS["bridge"])
        logs = {}
        for case in cases:
            log = {"source_filename": case+".log", "bytes": 12, "sha256": "a"*64}
            logs[case] = log
            prefix = "bridge/"+case+"/"
            names = pkg.CASES["common"] | pkg.CASES["replay"]
            if case == "baseline":
                names |= pkg.CASES["statistics"] | {"host_events.jsonl", "vendor_launch.json"}
                names |= {f"observation/step{i}/rank0/{kind}.json" for i in range(2) for kind in ("dump","stack","construct")}
            for n in names:
                rel = case+"/"+n
                f = self.src/rel
                f.parent.mkdir(parents=True, exist_ok=True)
                text = '{"schema":"fault33.pa.typed_host.v1"}\n' if n.endswith(".jsonl") else '{\n  "fixture": true\n}\n'
                f.write_bytes(text.encode())
                s.add(self.src, rel, prefix+n)
            s.generated(prefix+"native_execution.json", {"case_id": case, "source_log": log})
        s.groups["bridge"] = {"cases": cases, "full_logs": logs}
        return s

    def test_preserves_original_bytes_and_unsigned_bundle_inventory_detects_change(self):
        s, out, arc = self.fixture(), self.tmp/"public", self.tmp/"evidence.tar.gz"
        result = pkg.export(s, out, arc)
        self.assertEqual(pkg.verify(out)["inventory_sha256"], result["inventory_sha256"])
        name = "bridge/baseline/observation/step1/rank0/dump.json"
        original = (self.src/"baseline/observation/step1/rank0/dump.json").read_bytes()
        with tarfile.open(arc) as tar:
            self.assertEqual(tar.extractfile(name).read(), original)
            self.assertFalse(any(p.endswith((".log", ".pt", ".bin", ".so")) for p in tar.getnames()))
        (out/name).write_text("{}", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "content mismatch"):
            pkg.verify(out)

    def test_omitting_an_original_step_even_from_inventory_is_rejected(self):
        s, out = self.fixture(), self.tmp/"public"
        pkg.export(s, out)
        inv = out/pkg.INVENTORY
        data = json.loads(inv.read_text(encoding="utf-8"))
        name = "bridge/baseline/observation/step1/rank0/stack.json"
        data["files"] = [r for r in data["files"] if r["path"] != name]
        inv.write_text(json.dumps(data), encoding="utf-8")
        (out/name).unlink()
        with self.assertRaisesRegex(ValueError, "Incomplete or extra"):
            pkg.verify(out)

    def test_fresh_output_and_source_overlap_refused_before_writing(self):
        s = self.fixture()
        for output, arc in ((self.src/"sub", None), (self.tmp/"fresh", self.src/"bad.tar.gz"),
                            (self.tmp, None), (self.tmp/"fresh", self.tmp/"fresh/inner.tar.gz")):
            with self.subTest(output=output), self.assertRaises(ValueError):
                pkg.export(s, output, arc)
        self.assertFalse((self.tmp/"fresh").exists())

    def test_private_payload_and_unknown_extensions_or_roots_rejected(self):
        names = ("bridge/baseline/observation/step0/rank0/extra.json", "fault/fault/private_output/full.bin",
                 "model/model_fault_r0/native_logs/x.log", "bridge/baseline/vendor.so.1", "audit/bridge/source/setup/checkpoint.pt",
                 "../escape/result.json", "bridge/baseline/cache/data.json", "audit/bridge/source/unknown.py")
        for n in names:
            with self.subTest(path=n):
                self.assertFalse(pkg.permitted(n))
        for obj in ({"bad": [{"type": "torch.Tensor", "shape": [2], "dtype": "float16", "values": [2., 3.]}]},
                    {"child": {"raw_tensor_data": [4]}}, {"raw_hex": "ab"*3000},
                    {"type": "torch.Tensor", "dtype": "torch.float16", "shape": [1], "data_name": "exp.pt"},
                    {"child": {"shape": [2], "dtype": "float16", "values": [2., 3.]}}):
            with self.subTest(obj=str(obj)[:60]), self.assertRaises(ValueError):
                pkg.validate_json_tree(obj)

    def test_public_tiling_bytes_only_in_approved_host_record(self):
        obj = {"schema": "fault33.pa.typed_host.v1", "raw_size": 3000, "raw_hex": "af"*3000}
        pkg.validate_json_tree({"calls": [{"public_host": obj}]}, allow_host_raw=True)
        with self.assertRaisesRegex(ValueError, "bounded"):
            pkg.validate_json_tree(obj, allow_host_raw=False)
        obj["raw_size"] = 100
        with self.assertRaisesRegex(ValueError, "bounded"):
            pkg.validate_json_tree(obj, allow_host_raw=True)

    def test_statistics_tree_rejects_extra_or_missing_even_if_normal_name(self):
        root = self.src/"baseline"
        for i in range(2):
            rank = root/"observation"/f"step{i}"/"rank0"
            rank.mkdir(parents=True)
            for k in ("dump","construct","stack"):
                (rank/(k+".json")).write_text("{}", encoding="utf-8")
        self.assertEqual(len(pkg.exact_observation(root, 2)), 6)
        (rank/"shadow.json").write_text("{}", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "extra"):
            pkg.exact_observation(root, 2)

    def model_attempts(self):
        """Small real controller-layout fixtures; native kernel parsing has its own CPU tests."""
        def put(root, name, value):
            path = root/name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(json.dumps(value).encode())
            return pkg.sha(path)

        names = pkg.GROUPS["model"]
        roots = []
        for i, name in enumerate(names):
            root = self.src/f"job{i}"
            roots.append(root)
            manifest = {}
            for n in ("controller.py", "vendor_exec.py", "model_adapter.py", "model_probe.py", "replay.py"):
                (root/"source").mkdir(parents=True, exist_ok=True)
                (root/"source"/n).write_text("# fixture\n", encoding="utf-8")
                manifest[n] = pkg.sha(root/"source"/n)
            for n in names:
                f = f"setup/test/{n}.json"
                manifest[f] = put(root, "source/"+f, {"case_id": n})
            put(root, "source_manifest.json", manifest)
            file = "setup/test/"+name+".json"
            selected = {"case_id": name, "case": {"case_id": name}, "file": file}
            put(root, "selected_cases.json", [selected])
            case = root/name
            for n in pkg.CASES["common"] | pkg.CASES["model"] | {"collector_provenance.json", "config.json"}:
                put(case, n, {})
            put(case, "case.json", selected["case"])
            native = []
            for n in ("dump", "stack", "construct"):
                path = f"observation/step0/rank0/{n}.json"
                native.append({"path": path, "sha256": put(case, path, {}), "bytes": 2})
            receipt = put(case, "statistics_receipt.json", {"steps": [0], "rank": 0, "native_files": native,
                          "collector_sha256": pkg.sha(case/"collector_provenance.json"), "config_sha256": pkg.sha(case/"config.json")})
            r = put(case, "result.json", {"case_id": name, "status": "CAPTURE_COMPLETED",
                    "case_sha256": manifest[file], "statistics_receipt_sha256": receipt, "original_api_calls": 28})
            put(case, "host_events.jsonl", {"schema": "fault33.pa.typed_host.v1"})
            put(root, name+".vendor_launch.json", {"case_id": name, "case_sha256": manifest[file],
                      "launcher_sha256": manifest["vendor_exec.py"], "probe_sha256": manifest["model_probe.py"]})
            (root/(name+".log")).write_bytes(b"fixture\n")
            container = str(i+1)*64
            for n in ("resource_policy.json", "launch_command.json"):
                put(root, n, {})
            put(root, "container_identity.json", {"id": container})
            sample = {"selected_card": 5, "eligible": True, "require_selected_card_idle": True, "selected_card_table_idle": True,
                      "unapproved_device_capable_containers": [], "blocking_selected_card_foreign_npu_processes": []}
            start = 100 + i*100
            put(root, "completion.json", {"status": "CAPTURE_COMPLETED", "probe": "model_probe.py", "capture": "native-logs",
                "card": 5, "device_lock_acquired": True, "device_lock_released": True, "owned_container_removed": True,
                "owned_container_id": container, "cleanup_errors": [], "external_container_mutations": False,
                "start_epoch": start, "end_epoch": start+10, "cases": [{"case_id": name, "rc": 0,
                "start": start+1, "end": start+9, "result_sha256": r, "status":"CAPTURE_COMPLETED_NOT_MECHANISM_ACCEPTANCE",
                "pre_case_resources": sample, "post_case_resources": sample, "resources": []}]})
        return roots

    def split_selection(self, roots):
        selection = pkg.Selection()
        with patch.object(pkg.native_evidence, "parse_log", return_value=[{} for _ in range(28)]), \
                patch.object(pkg.native_evidence, "bind_host"):
            pkg.select_group(selection, "model", roots, False)
        return selection

    def test_four_preserved_controller_receipts_one_source_tree(self):
        roots = self.model_attempts()
        output = self.tmp/"public"
        pkg.export(self.split_selection(roots), output)
        record = pkg.json_data(output/pkg.INVENTORY)
        self.assertEqual(list(record["groups"]), ["model"])
        self.assertEqual(pkg.verify(output)["groups"], ["model"])
        for root, name in zip(roots, pkg.GROUPS["model"]):
            for file in pkg.AUDIT:
                audit = output/"audit/model/attempts"/name/file
                self.assertEqual(audit.read_bytes(), (root/file).read_bytes())
        self.assertEqual(len(list(output.rglob("model_probe.py"))), 1)
        self.assertNotIn("private_output", str(record["files"]))
        self.assertEqual(len(list(output.rglob("dump.json"))), 4)
        record["groups"]["model"]["attempts"]["model_restore"]["start_epoch"] = 105
        (output/pkg.INVENTORY).write_text(json.dumps(record), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "Overlapping"):
            pkg.verify(output)

    def test_split_rejects_changed_source_copy_or_wrong_attempt_count(self):
        roots = self.model_attempts()
        with self.assertRaisesRegex(ValueError, "distinct original"):
            self.split_selection(roots[:2])
        (roots[2]/"source"/"model_probe.py").write_text("# changed", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "Frozen model harness differs"):
            self.split_selection(roots)

    def test_split_rejects_receipt_swap_in_unsigned_inventory(self):
        roots = self.model_attempts()
        output = self.tmp/"public"
        pkg.export(self.split_selection(roots), output)
        record = pkg.json_data(output/pkg.INVENTORY)
        jobs = record["groups"]["model"]["attempts"]
        jobs["model_fault_r0"]["completion_sha256"] = jobs["model_fault_r1"]["completion_sha256"]
        (output/pkg.INVENTORY).write_text(json.dumps(record), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "Reused controller receipt"):
            pkg.verify(output)


if __name__ == "__main__":
    unittest.main()

