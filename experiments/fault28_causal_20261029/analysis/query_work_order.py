#!/usr/bin/env python3
"""Extend the audited FA oracle with every target-batch task (not only the tail query).

Copied from original oracle SHA 45a0718335cf7d61051719cf29d1890153dab0885ea1fa801082bc37a0f3d787.
Only target_batch_tasks collection is added; source semantics remain unchanged.

This is a source-derived CPU oracle, not a hardware trace or numerical oracle.
It never branches on mode, eligible, policy_applied, or an expected hit count.
Unsupported layouts/sparse modes/templates fail closed.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

SOURCE_COMMIT = "c77c0829f1f0f0b8f882db1d9b051a758b605472"
SOURCE_FILES = {
    "attention/flash_attention_score/op_kernel/arch22/flash_attention_score_s1s2_bn2gs1.h": "18bbd703454ea8cd6fbe6936329c29513745c269e7f7ad7ea5ab05d81dcd08cb",
    "attention/flash_attention_score/op_kernel/arch22/flash_attention_score_common.h": "3e074a966515df1c9cf0cc070d2315a5978ccb44e7e75a842188de5016e4cc34",
    "attention/flash_attention_score/op_kernel/arch22/flash_attention_score_tiling.h": "6ca76bce8a92fdf63d3c5646dfe6d2af3a9b367a9e6fbf7f994f0a5cee71c8d3",
    "attention/flash_attention_score/op_kernel/flash_attention_score.cpp": "7f4adb241e3f09ad3204159bcd1b963800e174230f80c78b8126e29b7481ba64",
    "attention/flash_attention_score/op_kernel/arch22/flash_attention_score_template_tiling_key.h": "a8951fea89250f0f815382e9c3b2a05a4c01a12f971985bfe5d133f04062f072",
}


def require(test, message):
    if not test:
        raise ValueError(message)


def positive_int(p, key):
    x = p[key]
    require(type(x) is int and x > 0, f"Invalid positive integer: {key}")
    return x


def ceildiv(a, b):
    return (a + b - 1) // b


def derive(record, target_batch=0, target_query=None):
    require(record["schema"] == "fault28.fa.typed_host.v1", "Unknown profile schema")
    require(record["callback_status"] == 0, "Host callback failed")
    require(record["workspace_count"] == len(record["workspace_sizes"]) == 1,
            "Only the captured one-workspace profile is supported")
    require(record["raw_size"] > 0 and len(bytes.fromhex(record["raw_hex"])) == record["raw_size"],
            "Missing or malformed raw tiling evidence")
    p = record["typed"]
    require(p["template"] == "FlashAttentionScoreS1s2Bn2gs1", "Unsupported template")
    require(p["layout"] == "BNSD" and p["sparse_type"] == 3 and p["sparse_mode"] == 2,
            "Only BNSD left-up causal is implemented")
    require(p["has_mask"] == 1 and p["has_dropout"] == 0 and p["has_pse"] == 0,
            "Mask/dropout/PSE scope mismatch")
    require(p["high_precision"] is True and p["calc_bytes"] == 4 and p["keep_prob"] == 1,
            "Precision or dropout policy mismatch")
    require(type(p["enable_l1_reuse"]) is bool, "Actual derived-template L1 reuse observation missing")
    b, n2, g = (positive_int(p, k) for k in ("b", "n2", "g"))
    s1, s2, m, s2base, ratio = (positive_int(p, k) for k in ("s1", "s2", "m", "s2_base", "n_ratio"))
    outer = positive_int(p, "s1_outer")
    total, cores = positive_int(p, "total_query_tasks"), positive_int(p, "core_num")
    split = positive_int(p, "split_factor")
    chunk = s2base * ratio
    require(s1 == s2 == 2048 and m in (64, 128) and s2base == 128 and ratio in (4, 8),
            "Outside audited 2048-token geometry")
    require(p["n1"] == n2 * g and p["d"] == p["dv"] == 128, "Head geometry mismatch")
    require(outer == ceildiv(s1, m) and p["s2_outer"] == ceildiv(ceildiv(s2, s2base), ratio),
            "Typed outer sizes are inconsistent")
    require(p["s2_group"] == chunk and total == b * n2 * g * outer, "Typed task/group count mismatch")
    require(1 <= cores <= 48, "Invalid active AIV count")
    starts = p["sparse_start_idx"]
    require(len(starts) == 48 and all(type(v) is int and 0 <= v <= total for v in starts),
            "Invalid typed sparseStartIdx[48]")
    reuse = p["enable_l1_reuse"]
    # SDK GET_TPL_TILING_KEY CPU verification gives 22619644976 for this
    # exact route with L1Reuse=true. The declared one-bit Boolean parameter
    # at bit 34 is the only allowed route difference in the non-reuse oracle.
    expected_key = 22619644976 if reuse else 22619644976 - (1 << 34)
    require(str(record["tiling_key"]) == str(expected_key), "Tiling key does not select the audited kernel")
    require(not reuse or cores % 2 == 0, "L1 reuse requires paired AIVs in audited route")
    target_query = s1 - 1 if target_query is None else target_query
    require(0 <= target_batch < b and 0 <= target_query < s1, "Target query outside profile")

    # GetS1LoopRange: fixed source 650-694. Sparse ranges override split-factor ranges.
    tasks = []
    fake_pairs = []
    intervals = []
    for core in range(cores):
        if reuse:
            pair = core - core % 2
            start = starts[pair]
            end = starts[pair + 2] if pair + 2 < cores else total
        else:
            start = starts[core]
            end = starts[core + 1] if core + 1 < cores else total
        require(0 <= start <= end <= total, "Malformed sparse task interval")
        intervals.append({"aiv": core, "start": start, "end": end})
        if start == end:
            continue
        if not reuse:
            tasks.extend((core, task, False) for task in range(start, end))
            continue
        # Process: 789-855; GetBNIdx: 974-978. Exclude pipeline flush loops.
        bng_begin, bng_end = start // outer, ceildiv(end, outer)
        residual = end % outer or outer
        for bng in range(bng_begin, bng_end):
            first = start % outer if bng == bng_begin else 0
            actual_end = residual if end < outer or bng == bng_end - 1 else outer
            need_fake = bool((actual_end - first) % 2)
            padded_end = actual_end + int(need_fake)
            for inner in range(first, padded_end):
                if (inner - first) % 2 == int(core % 2 == 0):
                    continue
                task = bng * outer + inner
                is_fake = need_fake and inner == padded_end - 1
                if is_fake:
                    # lastNotPair exits Vec1/Vec2 (1253-1256, 2164-2167).
                    fake_pairs.append({"aiv": core, "duplicated_task": task - 1})
                    continue
                use_next = (inner != padded_end - 2 or not need_fake) and core % 2 == 0
                tasks.append((core, task, use_next))
    ownership = Counter(task for _, task, _ in tasks)
    require(len(ownership) == total and set(ownership) == set(range(total)) and max(ownership.values()) == 1,
            "Missing or duplicate real query tasks across cores")

    target_tasks = []
    target_batch_tasks = []
    trace_hash = hashlib.sha256()
    queries_checked = 0
    for core, task, use_next in tasks:
        # ComputeAxisIdx 985-994 and CAUSAL GetS2LoopRange 703-709, 743-747.
        batch = task // (n2 * g * outer)
        kv_head = task % (n2 * g * outer) // (g * outer)
        group = task % (g * outer) // outer
        block = task % outer
        query_start, query_end = block * m, min((block + 1) * m, s1)
        key_end = min((block + 1) * m, s2)
        if use_next:
            next_end = min((block + 2) * m, s2)
            if ceildiv(key_end, chunk) != ceildiv(next_end, chunk):
                key_end = next_end
        # s2LoopCount 0..ceil(range/chunk)-1 (855/861 or 922/926), tail at 1057.
        groups = [[lo, min(lo + chunk, key_end)] for lo in range(0, key_end, chunk)]
        require(groups and groups[0][0] == 0 and groups[-1][1] == key_end and
                all(a[1] == z[0] for a, z in zip(groups, groups[1:])), "Key groups not a partition")
        for query in range(query_start, query_end):
            legal_end = min(query + 1, s2)
            spans = [[lo, min(hi, legal_end)] for lo, hi in groups if lo < legal_end]
            require(spans[0][0] == 0 and spans[-1][1] == legal_end and
                    sum(hi - lo for lo, hi in spans) == legal_end,
                    "A causal query's legal keys are not covered exactly once")
            queries_checked += 1
        row = {"aiv": core, "task": task, "batch": batch, "kv_head": kv_head,
               "query_head": kv_head * g + group, "query_interval": [query_start, query_end],
               "computed_key_groups": groups, "use_next_l1": use_next}
        trace_hash.update(json.dumps(row, sort_keys=True, separators=(",", ":")).encode() + b"\n")
        if batch == target_batch:
            target_batch_tasks.append(dict(row))
        if batch == target_batch and query_start <= target_query < query_end:
            row["target_query"] = target_query
            row["target_legal_key_groups"] = [[lo, min(hi, target_query + 1)]
                                               for lo, hi in groups if lo <= target_query]
            target_tasks.append(row)
    require(queries_checked == b * n2 * g * s1 and len(target_tasks) == n2 * g,
            "Incomplete all-query or target-head coverage")
    return {"status": "PASS_SOURCE_DERIVED_QUERY_KEY_COVERAGE", "source_commit": SOURCE_COMMIT,
            "source_sha256": SOURCE_FILES, "scope": "fixed BNSD causal S1S2 source semantics",
            "not_hardware_execution_trace": True, "not_numerical_equivalence_proof": True,
            "mode_or_hit_fields_used": False, "total_real_query_tasks": total,
            "queries_checked": queries_checked, "every_real_query_owned_once": True,
            "each_query_legal_keys_covered_once": True, "query_split_between_cores": False,
            "full_task_trace_sha256": trace_hash.hexdigest(), "active_aiv_intervals": intervals,
            "fake_pairs_without_vec_output": fake_pairs, "target_tasks": target_tasks, "target_batch_tasks": target_batch_tasks,
            "profile_dimensions": {k: p[k] for k in ("b", "n1", "n2", "g", "s1", "s2", "m", "s2_base", "n_ratio", "enable_l1_reuse")},
            "source_locations": {"GetS1LoopRange": "650-694", "GetS2LoopRange": "700-764",
                                 "Process": "770-950", "GetBNIdx_ComputeAxisIdx": "974-994",
                                 "fake_pair_Vec1_return": "1253-1256", "fake_pair_Vec2_return": "2164-2167",
                                 "online_softmax": "1955 onward", "online_output_accumulation": "2239-2247"}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", type=Path, required=True)
    ap.add_argument("--record-index", type=int, default=0)
    ap.add_argument("--source-root", type=Path, required=True)
    ap.add_argument("--target-batch", type=int, default=0)
    ap.add_argument("--target-query", type=int)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    for relative, expected in SOURCE_FILES.items():
        require(hashlib.sha256((args.source_root / relative).read_bytes()).hexdigest() == expected,
                f"Kernel source SHA mismatch: {relative}")
    raw = args.profile.read_bytes()
    records = [json.loads(line) for line in raw.decode().splitlines() if line.strip()]
    result = derive(records[args.record_index], args.target_batch, args.target_query)
    result.update(profile_sha256=hashlib.sha256(raw).hexdigest(), record_index=args.record_index,
                  source_files_verified=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "queries_checked": result["queries_checked"]}))


if __name__ == "__main__":
    main()
