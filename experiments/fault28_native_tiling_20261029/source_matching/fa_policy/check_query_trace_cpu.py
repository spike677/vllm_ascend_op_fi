#!/usr/bin/env python3
"""Synthetic algorithm checks only; these profiles are not NPU evidence."""
import copy
import json
from pathlib import Path
from derive_query_trace import derive


def main():
    positive, rejected = [], 0
    for m, ratio in ((128, 8), (64, 4)):
        for reuse in (False, True):
            total, cores = 8 * 16 * (2048 // m), 40
            divisions = cores // 2 if reuse else cores
            starts = []
            for value in (i * total // divisions for i in range(divisions)):
                starts.extend([value, value] if reuse else [value])
            starts += [0] * (48 - len(starts))
            record = {
                "schema": "fault28.fa.typed_host.v1", "callback_status": 0,
                "workspace_count": 1, "workspace_sizes": [4096], "raw_size": 1, "raw_hex": "00",
                "tiling_key": str(22619644976 if reuse else 22619644976 - (1 << 34)),
                "typed": {"template": "FlashAttentionScoreS1s2Bn2gs1", "layout": "BNSD",
                          "sparse_type": 3, "sparse_mode": 2, "has_mask": 1, "has_dropout": 0,
                          "has_pse": 0, "high_precision": True, "calc_bytes": 4, "keep_prob": 1,
                          "enable_l1_reuse": reuse, "b": 8, "n1": 16, "n2": 8, "g": 2,
                          "s1": 2048, "s2": 2048, "m": m, "s2_base": 128, "n_ratio": ratio,
                          "s1_outer": 2048 // m, "s2_outer": 2048 // (128 * ratio),
                          "total_query_tasks": total, "core_num": cores, "split_factor": 128,
                          "s2_group": 128 * ratio, "d": 128, "dv": 128, "sparse_start_idx": starts}}
            result = derive(record)
            positive.append({"m": m, "ratio": ratio, "l1_reuse": reuse,
                             "queries_checked": result["queries_checked"],
                             "fake_pairs": len(result["fake_pairs_without_vec_output"]),
                             "target_key_groups": result["target_tasks"][0]["target_legal_key_groups"]})
            badstarts = starts.copy()
            badstarts[0] = 1
            mutations = [("sparse_start_idx", badstarts), ("sparse_type", 4),
                         ("total_query_tasks", total + 1), ("enable_l1_reuse", None)]
            negatives = []
            for key, value in mutations:
                bad = copy.deepcopy(record)
                bad["typed"][key] = value
                negatives.append(bad)
            bad = copy.deepcopy(record)
            bad["tiling_key"] = "1"
            negatives.append(bad)
            for bad in negatives:
                try:
                    derive(bad)
                except ValueError:
                    rejected += 1
                else:
                    raise AssertionError("Invalid profile accepted")
    result = {"status": "PASS_SYNTHETIC_PROFILE_ALGORITHM_CHECK_ONLY",
              "actual_npu_profile_used": False, "positive_profiles": positive,
              "negative_profiles_rejected": rejected,
              "limits": "Synthetic one-byte raw placeholder is not an ABI or NPU check. CLI source SHA checking is separate."}
    path = Path(__file__).with_name("trace_cpu_check.json")
    path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "positive": len(positive), "negative": rejected}))


if __name__ == "__main__":
    main()
