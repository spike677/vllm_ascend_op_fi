"""Offline acceptance gates. Synthetic records exercise failures without an NPU."""
import copy
import hashlib
import json
from pathlib import Path
import struct
import tempfile
import unittest

import accept_replay as ar


def synthetic_projection(inner, on):
    # Synthetic offsets deliberately do not model the Ascend ABI.
    names = [ar.INNER, ar.LOOPS, "baseParams.blockSize"]
    names.extend("fixture_" + str(i) for i in range(68))
    shape = [{"name": n, "type": "uint32_t", "offset": i * 4, "width": 4, "count": 1}
             for i, n in enumerate(names)]
    shape.append({"name": "coreSidxEnd", "type": "uint32_t", "offset": len(names)*4, "width": 4, "count": 50})
    raw = struct.pack("<" + "I" * (len(names) + 50),
                      inner, 16384 // inner, 128, *[0]*68, *range(50))
    schema = {"schema": "fault33.pa.public_serializer_layout.v1", "type": "IncreFlashAttentionTilingDataV2",
              "layout_source": "public_host_SaveToBuffer_single_field_roundtrip", "fields": shape,
              "kernel_argument_raw_observed": False, "projection_only": True, "size": len(raw)}
    mode = "inner1024" if on else "off"
    guard = {"fixed_inputs": True, "actual_lengths_all_equal": True, "native_scope": True, "eligible": True,
             "legal_request": True, "applied": on, "batch": 8, "actual_kv_length": 16384,
             "block_size": 128, "requested_sinner": 1024 if on else 0, "native_sinner": 2048}
    event = {"raw_hex": raw.hex(), "typed": ar.abi_fields(schema, raw), "mode": mode, "phase": "decode",
             "guard_reports": [guard], "selector_checks": 1, "workspace_sizes": [1024], "eligible": True,
             "policy_applied": on, "caller_declared_deterministic": True, "kernel_argument_raw_observed": False}
    return schema, {"public_host": event, "public_host_raw_sha256": hashlib.sha256(raw).hexdigest()}, mode


class TestAcceptance(unittest.TestCase):
    def test_same_projected_fields_cannot_hide_raw_restoration_mismatch(self):
        x = {"visible": {}, "output": {}, "plan": {"typed": {}, "workspace_sizes": [4], "pre_intervention_native_inner": 2048,
                                                   "raw_sha256": "first"}}
        y = copy.deepcopy(x)
        y["plan"]["raw_sha256"] = "different"
        with self.assertRaisesRegex(ValueError, "raw_sha256"):
            ar.check_equivalent(x, y, True)

    def test_public_abi_bound_to_stage_generation_and_all_three_original_files(self):
        with tempfile.TemporaryDirectory() as t:
            root = Path(t)
            schema, _, _ = synthetic_projection(1024, True)
            schema["size"] = 3000
            evidence = {"schema": schema, "selftest": {"status": "PASS_PUBLIC_SERIALIZER_CPU_ROUNDTRIP", "size": 3000,
                        "projected_fields": 72, "stock_abi_validated": False, "kernel_argument_raw_observed": False},
                        "compile": {"schema": "fault33.pa.abi_decoder_compile.v1", "npu_executed": False,
                        "abi_kind": "public_host_serializer_projection_not_device_argument_read", "source_sha256": "a"*64, "decoder_sha256": "b"*64}}
            abi = {"generation": "a3", "raw_size": 3000, "projected_fields": 72,
                   "layout_kind": "public_host_serializer_projection", "kernel_argument_raw_observed": False}
            for role, obj in evidence.items():
                name = f"ifa_abi_{role}.json"
                file = root / name
                file.write_text(json.dumps(obj), encoding="utf-8")
                abi[role+"_sha256"], abi[role+"_file"] = ar.sha(file), "build/evidence_a3/"+name
            path = root / "ifa_abi_schema.json"
            abi["schema_bytes"] = path.stat().st_size
            vendor = {"generation": "a3", "vendor_root": "vendor_stage_a3/vendors/fault33_pa_transformer", "public_abi": abi}
            self.assertEqual(ar.verified_abi(vendor, path), schema)
            for key, bad in (("schema_sha256", "a"*64), ("compile_sha256", "c"*64), ("selftest_sha256","d"*64),
                             ("schema_bytes", 0), ("raw_size", 2992), ("generation", "a2"),
                             ("schema_file", "build/evidence_a2/ifa_abi_schema.json")):
                alt = copy.deepcopy(vendor)
                alt["public_abi"][key] = bad
                with self.subTest(key=key), self.assertRaises(ValueError):
                    ar.verified_abi(alt, path)

    def test_native_log_counts_unknown_and_out_of_window_dispatches(self):
        op = "OP(17,python):[NNOP]"
        native = " OpName:[IncreFlashAttention_0] "
        entry = {"fault33": "enter", "call_id": "x", "pid": 17, "tid": 17, "block_size": 128,
                 "deterministic_algorithms": True}
        mm = [f"MatmulTiling: {f} = 2" for f in ("M", "N", "Ka", "Kb", "singleCoreN", "baseN", "baseK")]
        launch = op+"[NnopbaseExecutorLaunchKernel][88]"+native+"Launch dynamic kernel IncreFlashAttention task, tilingKey is 5, scheMode is 1, numBlocks is 20"
        lines = [json.dumps(entry), *mm, *mm, op+"[NnopbaseExecutorGenDynamicKey][17]"+native+
                 "Verbose opType is IncreFlashAttention, deterministic value is 1, highPrecision value is 0",
                 op+"[NnopbaseExecutorDoTiling][17]"+native+"Tiling outputs of IncreFlashAttention are [tilingKey: 5, numBlocks: 20, scheMode: 1, needAtomic: 0, aicpuNumBlocks: 0, dynBufSize: 0]",
                 op+"[NnopbaseExecutorPrepareParamsExt][88]"+native+"TilingDataSize is 3000 bytes",
                 launch, json.dumps({**entry, "fault33": "exit"})]
        with tempfile.TemporaryDirectory() as t:
            path = Path(t)/"case.log"
            def parse(rows):
                path.write_text("\n".join(rows)+"\n", encoding="utf-8")
                return ar.native_evidence.parse_log(path)
            own = parse(lines)[0]["whole_log_nnop_ifa_census"]
            self.assertEqual((own["calls"], own["pid"]), (1,17))
            for rows in ([launch, *lines], [*lines, launch], [*lines, launch.replace("scheMode is", "unknown version")],
                         [*lines[:-1], launch.replace("scheMode is", "unknown version"), lines[-1]]):
                with self.subTest(rows=rows[:1]), self.assertRaises(ValueError):
                    parse(rows)
            self.assertEqual(len(parse([*lines, launch.replace("OP(17,", "OP(18,")])), 1)

    def test_public_raw_and_every_decoded_typed_field_match(self):
        schema, w, mode = synthetic_projection(1024, True)
        plan = ar.private_plan(w, mode, schema)
        self.assertEqual((plan["selected_inner"], plan["outer_update_loops"], plan["verified_projected_fields"]),
                         (1024, 16, 72))
        self.assertTrue(plan["selector_applied"])

    def test_source_field_claimed_but_raw_unchanged_rejected(self):
        schema, w, mode = synthetic_projection(1024, True)
        w["public_host"]["typed"]["fixture_2"] = 90
        with self.assertRaisesRegex(ValueError, "typed observation"):
            ar.private_plan(w, mode, schema)

    def test_off_must_keep_real_native_inner(self):
        schema, w, mode = synthetic_projection(1024, False)
        with self.assertRaisesRegex(ValueError, "serialized plan"):
            ar.private_plan(w, mode, schema)
        schema, w, mode = synthetic_projection(2048, False)
        self.assertFalse(ar.private_plan(w, mode, schema)["selector_applied"])

    def test_guard_rejects_missing_match_or_duplicate_checks(self):
        schema, w, mode = synthetic_projection(1024, True)
        w["public_host"]["selector_checks"] = 2
        with self.assertRaisesRegex(ValueError, "hit count"):
            ar.private_plan(w, mode, schema)
        w["public_host"]["selector_checks"] = 1
        w["public_host"]["guard_reports"][0]["actual_lengths_all_equal"] = False
        with self.assertRaisesRegex(ValueError, "guard"):
            ar.private_plan(w, mode, schema)

    def test_wrong_size_unknown_overlapping_and_out_of_bounds_rejected(self):
        schema, w, _ = synthetic_projection(1024, True)
        raw = bytes.fromhex(w["public_host"]["raw_hex"])
        for patch in ({"type": "unknown"}, {"offset": 0}, {"offset": len(raw)}):
            altered = copy.deepcopy(schema)
            altered["fields"][1].update(patch)
            with self.subTest(patch=patch), self.assertRaises(ValueError):
                ar.abi_fields(altered, raw)
        with self.assertRaisesRegex(ValueError, "size"):
            ar.abi_fields(schema, raw[:-1])

    def test_fp16_all_rows_signed_zero_vs_real_effect(self):
        a = struct.pack("<4e", 1, -0., 2., -2.)
        b = struct.pack("<4e", 1, 0., 4., -2.)
        report = ar.difference(a, b, 2)
        self.assertFalse(report["bitwise_equal"])
        self.assertEqual(report["numeric_different_elements"], 1)
        self.assertEqual(report["signed_zero_only_differences"], 1)
        self.assertEqual(report["per_request_numeric_differences"], [0, 1])
        self.assertEqual(report["max_abs"], 2)
        with self.assertRaisesRegex(ValueError, "Nonfinite"):
            ar.difference(struct.pack("<e", float("nan")), b"\0\0", 1)

    def test_each_private_output_file_is_bound_to_own_tensor_record(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "private_output").mkdir()
            call = {"call_id": "repeat0"}
            for label, dims in ar.OUTPUT_SHAPES.items():
                payload = b"\0\0" * ar.math.prod(dims)
                meta = {"shape": dims, "dtype": "torch.float16", "byteorder": "little",
                        "logical_bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}
                call[label if label == "attention" else "synthetic_projection"] = meta
                (root / "private_output" / ("repeat0_" + label + ".bin")).write_bytes(payload)
            outputs = ar.output_bytes(root, call)
            self.assertEqual(set(outputs), {"attention", "projection"})
            path = root / "private_output/repeat0_projection.bin"
            path.write_bytes(b"\1" + path.read_bytes()[1:])
            with self.assertRaisesRegex(ValueError, "Private output"):
                ar.output_bytes(root, call)

    def test_stock_and_off_require_full_outputs_and_visible_bmm(self):
        base = {"visible": {"key": 123, "sdk_bmm": [{"baseK": 128}]}, "output": {"attention": b"a", "projection": b"b"}}
        other = copy.deepcopy(base)
        ar.check_equivalent(base, other, False)
        other["visible"]["sdk_bmm"][0]["baseK"] = 16
        with self.assertRaisesRegex(ValueError, "BMM"):
            ar.check_equivalent(base, other, False)
        other = copy.deepcopy(base)
        other["output"]["projection"] = b"c"
        with self.assertRaisesRegex(ValueError, "projection"):
            ar.check_equivalent(base, other, False)

    def test_matching_launch_key_cannot_override_bad_deterministic_flag(self):
        row = {"selected": {}, "native_raw_size": 10, "sdk_bmm": [], "native_flags": {"deterministic": 0}}
        with self.assertRaisesRegex(ValueError, "deterministic"):
            ar.visible_plan(row)

    def test_across_case_seed_or_full_cache_hash_difference_rejected(self):
        base = {"seeds": [0, 1], "software": {}, "logical": {}, "weight": "sha", "calls": [{"inputs": {"k": "foo"}}]}
        cases = {"baseline": base, "fault": copy.deepcopy(base)}
        ar.check_inputs(cases)
        cases["fault"]["calls"][0]["inputs"]["k"] = "bar"
        with self.assertRaisesRegex(ValueError, "physical"):
            ar.check_inputs(cases)
        cases["fault"] = copy.deepcopy(base)
        cases["fault"]["seeds"][1] += 1
        with self.assertRaisesRegex(ValueError, "seeds"):
            ar.check_inputs(cases)

    def test_maps_require_approved_private_tiling_library_sha(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            stage = "vendor_stage_a3/vendors/fault33_pa_transformer"
            vendor = {"vendor_root": stage, "artifacts": [{"path": "rt/liboptiling.so", "bytes": 1, "sha256": "A"},
                      {"path": "rt/libopapi.so", "bytes": 2, "sha256": "B"}],
                      "ifa_registration": {"tiling_libraries": ["rt/liboptiling.so"]}}
            case = {"operator_variant": "private_vendor"}
            result = {"calls": [{"pid": 17}]}
            f = root / "runtime_native_maps.json"
            def save(rel, size, digest):
                f.write_text(json.dumps({"pid": 17, "libraries": [{"path": "/audit/" + stage + "/" + rel,
                             "bytes": size, "sha256": digest}]}), encoding="utf-8")
                result["runtime_native_maps_sha256"] = ar.sha(f)
            save("rt/liboptiling.so", 1, "A")
            self.assertEqual(len(ar.verify_maps(root, case, result, vendor)), 1)
            save("rt/liboptiling.so", 1, "Z")
            with self.assertRaisesRegex(ValueError, "manifest"):
                ar.verify_maps(root, case, result, vendor)
            save("rt/libopapi.so", 2, "B")
            with self.assertRaisesRegex(ValueError, "Missing private tiling"):
                ar.verify_maps(root, case, result, vendor)
            with self.assertRaisesRegex(ValueError, "stock process"):
                ar.verify_maps(root, {"operator_variant": "stock"}, result, vendor)


if __name__ == "__main__":
    unittest.main()

