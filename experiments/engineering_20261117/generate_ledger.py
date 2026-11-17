"""Generate a scoped task ledger from the frozen 27-node plan.

This is a reporting tool, never a runtime qualification or an admission grant.
Reviewed summaries are inputs; raw tensors and injection logs are not interpreted
to select a statistics first difference. A Git archive may supply --matrix.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import unittest


ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
SOURCE_COMMIT = "cb5b0dc120ce6c80f1469bb8b7248da27a7bb258"
MATRIX_PATH = "docs/roadmap/17_fault_acceptance_matrix.json"
MATRIX_SHA256 = "095d93744cfbca7c145b21b59f25f07060277f9d1fe2865cf4a8d5505fb0b78d"
STAGES = {
    "S1": "ENGINEERED: implementation, Provider, configuration and minimum reproduction",
    "S2": "ADAPTED: actual model, logical role, operator and environment identification",
    "S3": "NATIVE_CLOSED: actual execution, one intervention and observed effect",
    "S4": "RECOVERY_CLOSED: OFF, drain, owned resources and recovery",
    "S5": "GENERALIZED: individually accepted model/environment combinations",
    "S6": "FULL_COVERAGE: all_eligible, every_match and whole-group recovery",
}
COMMON_POLICY = {
    "unknown_identity_or_abi": "REFUSE; do not infer support from model names or CANN major versions",
    "failed_native_attempt": "Retain the original receipt; permit at most three bounded, diagnosed attempts",
    "no_effect": "Retain NO_DIFFERENCE; never manufacture a calibration or lower the statistics tolerance",
    "recovery_uncertain": "Do not release an unsafe owner or report recovery; retain pending resources and escalate",
    "scope": "CPU, constructed replay, pretrained model and public Worker evidence are not interchangeable",
    "statistics": "Preserve null, nonfinite, alignment failure and downstream-only observations",
}
PUBLIC_FILES = {
    "freeze": [MATRIX_PATH, "docs/roadmap/17_fault_full_coverage_roadmap.md"],
    "worker_lifecycle": ["src/demoncat/runtime/vllm_ascend/worker_extension/runtime.py",
                         "src/demoncat/runtime/vllm_ascend/worker_extension/extension.py"],
    "native_inventory": ["fault_injection/environment.py", "fault_injection/native_build/manager.py"],
    "build_recipes": ["fault_injection/native_build/recipes.py", "fault_injection/native_build/build26.sh",
                      "fault_injection/native_build/build_postwrite.sh"],
    "target_identity": ["fault_injection/model_discovery.py", "fault_injection/runtime_discovery.py",
                        "fault_injection/coverage.py", "fault_injection/attention.py"],
    "request_context": ["src/demoncat/runtime/vllm_ascend/worker_extension/serving_context.py",
                        "src/demoncat/runtime/vllm_ascend/worker_extension/batch_context.py",
                        "src/demoncat/runtime/vllm_ascend/worker_extension/call_binding.py"],
    "evidence_layout": ["fault_injection/evidence.py", "experiments/overnight_20261116/evidence_contract.py"],
    "unique_binding": ["fault_injection/providers.py", "fault_injection/lifecycle.py",
                       "src/demoncat/runtime/vllm_ascend/worker_extension/providers/__init__.py"],
    "independent_statistics": ["fault39/tools/compare_msprobe.py", "fault39/tools/compare_model_capture.py",
                               "fault39/src/demoncat_fault39/diagnostics.py"],
    "m1_m2_integrate": ["fault_injection/catalog.py", "docs/roadmap/verify_acceptance_matrix.py"],
}
PUBLIC_TESTS = {
    "freeze": ["docs/roadmap/verify_acceptance_matrix.py", "experiments/engineering_20261117/generate_ledger.py --self-test"],
    "worker_lifecycle": ["tests/worker_extension/test_extension.py", "tests/worker_extension/test_boundary_state.py"],
    "native_inventory": ["tests/test_native_build.py"],
    "build_recipes": ["tests/test_native_build.py", "experiments/engineering_20261117/build_manager_124.py"],
    "target_identity": ["tests/test_boundary_targets.py", "tests/test_tensor_worker_providers.py"],
    "request_context": ["tests/worker_extension/test_serving_context.py", "tests/worker_extension/test_serving_mla_context.py",
                        "tests/worker_extension/test_decode_row_context.py"],
    "evidence_layout": ["tests/test_evidence_manager.py", "tests/test_overnight_evidence_contract.py"],
    "unique_binding": ["tests/worker_extension/test_call_binding.py", "tests/test_boundary_targets.py"],
    "independent_statistics": ["experiments/engineering_20261117/analyse_batch_124.py"],
    "m1_m2_integrate": ["experiments/engineering_20261117/run_cpu_validation.py", "experiments/engineering_20261117/run_cases_124.py"],
}
PROVIDER = "src/demoncat/runtime/vllm_ascend/worker_extension/providers/"
SERVING = "src/demoncat/runtime/vllm_ascend/worker_extension/"
EXP = "experiments/engineering_20261117/"
CURRENT = {
    "8.restore_offset": {
        "files": ["fault_injection/route8.py", PROVIDER + "tensor_modules.py"],
        "tests": ["tests/test_route8_index22.py", EXP + "tensor_provider_probe.py"],
        "summary": "Actual paired I/P group remapping before native unpermute; constructed inference only.",
        "remaining": "Natural MoE producer/scheduler ownership and public Worker deployment are unverified."},
    "15.alias_detach": {
        "files": ["fault_injection/alias.py", PROVIDER + "tensor_modules.py"],
        "tests": ["tests/test_alias_auto.py", "tests/test_tensor_worker_providers.py", EXP + "tensor_provider_probe.py"],
        "summary": "NZ29 BF16 alias detach, real view copy_ and original-base consumer; fixed healthy geometry.",
        "remaining": "Natural model write-through-view chain and strict statistics alignment remain unverified."},
    "17.spatial_padding": {
        "files": ["fault_injection/module_faults.py", PROVIDER + "tensor_modules.py"],
        "tests": ["tests/test_module_faults.py", "tests/test_tensor_worker_providers.py", EXP + "tensor_provider_probe.py"],
        "summary": "Standard Conv3d actual call changes padding to (0,7,7) under the supported patch geometry.",
        "remaining": "Natural vision-model path, other patch geometries and strict statistics alignment remain unverified."},
    "18.api_half": {
        "files": ["fault_injection/module_faults.py", PROVIDER + "tensor_modules.py"],
        "tests": ["tests/test_module_faults.py", "tests/test_tensor_worker_providers.py", EXP + "tensor_provider_probe.py"],
        "summary": "Standard RMSNorm return-gain proxy; actual original forward precedes multiplication by 0.5.",
        "remaining": "Fused RMSNorm is not adapted; constructed proxy does not establish an internal Reduce defect."},
    "20.buffer_substitute": {
        "files": ["fault_injection/deployments/buffer20.py", PROVIDER + "fault20.py", SERVING + "serving20.py",
                  SERVING + "serving20_qmm.py", "fault_injection/serving_qualification.py"],
        "tests": ["tests/worker_extension/test_serving20_qmm.py", "tests/worker_extension/test_serving20_qualification.py",
                  "experiments/worker_positive_20261117/serving_probe.py"],
        "summary": "Historical WeightNz Worker scope remains; natural dynamic-QMM projection is a separate qualification.",
        "remaining": "QMM a3 has a new actual DeepSeek C1 Worker receipt. Whole-dump E5 remains blocked by baseline NaN; no all-candidate or model-family promotion."},
    "22.wrong_row": {
        "files": ["fault_injection/index22.py", PROVIDER + "tensor_modules.py"],
        "tests": ["tests/test_route8_index22.py", EXP + "tensor_provider_probe.py"],
        "summary": "Actual owned cos_sin_cache index_select return substitutes one legal source row, then a real consumer.",
        "remaining": "Legal wrong-row proxy is not uint32 overflow; natural pretrained call and scheduler mapping are unverified."},
    "25.reduce_scatter": {
        "files": ["fault_injection/collective25.py"],
        "tests": ["tests/test_collective25.py", EXP + "collective25_probe.py"],
        "summary": "Replay requires a real same-input alternate collective result; current two-rank HCCL calibration had no difference.",
        "remaining": "No valid fault sample or injected effect. Exit hung; no whole-group recovery acceptance."},
    "26.residual_stomp": {
        "files": ["fault_injection/deployments/residual26_worker.py", SERVING + "serving26.py", PROVIDER + "fault26.py"],
        "tests": [EXP + "writer26_worker_probe.py"],
        "summary": "New standard Worker QMM writer requires same-forward live residual, adjacency, consumer and recovery.",
        "remaining": "Retain historical NaN limitations and the independently reviewed current Worker scope only."},
    "26.fia_destination": {
        "files": ["fault_injection/deployments/residual26.py", SERVING + "serving26.py", PROVIDER + "fault26.py"],
        "tests": ["experiments/p3p4_closure_20261117/probe26_worker.py"],
        "summary": "Keep historical SmolLM2/Llama and GLM4 fresh-prefill actual writer/consumer acceptance.",
        "remaining": "New versions, new model implementations, decode and C2/C3 require their own qualification."},
    "quant.batch_shared_scale": {
        "files": ["fault_injection/quantization.py", SERVING + "serving_quant.py", PROVIDER + "quant.py"],
        "tests": ["experiments/p3p4_closure_20261117/probe_quant_worker.py"],
        "summary": "Keep actual DeepSeek MLA/MoE first B8-decode single projection historical acceptance.",
        "remaining": "Baseline nonfinite and later incomparable records prevent whole-dump acceptance; no family-wide grant."},
    "28.fa.legal_tiling": {
        "files": ["fault_injection/strategy.py", "native_execution_strategy/adapters/flash_attention.py"],
        "tests": ["native_execution_strategy/tests/test_fa_adapter.py"],
        "summary": "Recovered FA host/kernel sources and isolated build path; legal strategy variation retains its own mechanism.",
        "remaining": "New source builds and resulting binaries require fresh native execution and OFF; old evidence is not inherited."},
    "28.fia.legal_tiling": {
        "files": ["fault_injection/fia.py", "fault_injection/native_adapters/attention28/fia/build_fresh.py"],
        "tests": ["tests/test_fia_masks.py", "fault_injection/native_adapters/attention28/fia/test_recurrence.py"],
        "summary": "FIA legal-tiling variant rebuilds from pinned source and preserves causal coverage constraints.",
        "remaining": "New native build/execution acceptance incomplete; numerical equality is a valid outcome, not a failure to hide."},
    "28.fia.denominator_replay": {
        "files": ["fault_injection/fia.py", "fault_injection/native_adapters/attention28/fia/build_fresh.py"],
        "tests": ["tests/test_fia_masks.py", "fault_injection/native_adapters/attention28/fia/test_recurrence.py"],
        "summary": "Controlled denominator-replay error stays distinct from a historical legal-tiling discrepancy.",
        "remaining": "New native build/execution acceptance incomplete; no historical-root-cause or cross-model promotion."},
    "33.paged_decode": {
        "files": ["fault_injection/deployments/paged33.py", "fault_injection/deployments/paged33_continuous.py",
                  SERVING + "paged33_audit.py"],
        "tests": ["tests/test_paged33_worker_audit.py", EXP + "paged33_worker_audit.py"],
        "summary": "Natural Qwen Worker observed 28 TND/page16 FIA calls with actual writer, live cache, slot and consumer.",
        "remaining": "Injection refused: old CANN9.1 BNSD IFA vendor is not a TND adapter. Diagnostic equality is not B/F/OFF."},
    "39.matmul_range": {
        "files": ["fault39/native/runtime.cpp", "fault39/src/demoncat_fault39/provider.py", PROVIDER + "fault39.py"],
        "tests": ["fault39/tools/native_probe.py", "experiments/worker_positive_20261117/serving_probe.py"],
        "summary": "Keep the historical Qwen3 fresh-prefill one-output-projection public Worker acceptance.",
        "remaining": "Operator-only 8.5.1/9.0.0/9.1.0/9.2.0 evidence does not qualify every Worker/model or any A3 cell."},
    "39.output_proxy": {
        "files": ["fault_injection/deployments/postwrite.py", "fault_injection/native_adapters/postwrite/postwrite.cpp"],
        "tests": ["tests/test_postwrite_providers.py", "tests/test_postwrite_capture_window.py", EXP + "postwrite_probe.py"],
        "summary": "Public ACL writes one BF16 output element after a real mm, reads it back and restores original storage.",
        "remaining": "Constructed mechanism only; no accepted natural pretrained writer/consumer or public Worker injection."},
    "41.tail_column": {
        "files": ["fault_injection/deployments/postwrite.py", "fault_injection/native_adapters/postwrite/postwrite.cpp"],
        "tests": ["tests/test_postwrite_providers.py", EXP + "postwrite_probe.py"],
        "summary": "Constructed owned-buffer index_copy_ followed by legal-address tail-column sign changes and actual consumer.",
        "remaining": "Not a natural vLLM KV dispatcher, not an out-of-bounds-write reproduction; public Worker unverified."},
}
DEFAULT_REVIEW_SUMMARIES = (
    EXP + "tensor_collective_summary.json",
    EXP + "postwrite_constructed_summary.json",
    EXP + "residual26_worker_summary.json",
    EXP + "fault20_qmm_worker_summary.json",
    EXP + "attention28_fresh_summary.json",
)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def read_matrix(matrix_file=None):
    raw = (matrix_file.read_bytes() if matrix_file else subprocess.check_output(
        ["git", "show", SOURCE_COMMIT + ":" + MATRIX_PATH], cwd=ROOT))
    if sha(raw) != MATRIX_SHA256:
        raise ValueError("FROZEN_MATRIX_BYTES_CHANGED")
    value = json.loads(raw)
    if value.get("schema") != "demoncat.fault_coverage.acceptance_matrix.v1":
        raise ValueError("UNRECOGNIZED_FROZEN_MATRIX")
    if len(value["m1_m2_tasks"]) != 10 or len(value["mechanisms"]) != 17:
        raise ValueError("FROZEN_27_NODE_PLAN_REQUIRED")
    return value, raw


def reference(path):
    p = (ROOT / path).resolve()
    p.relative_to(ROOT)
    raw = p.read_bytes()
    return {"path": p.relative_to(ROOT).as_posix(), "sha256": sha(raw)}, json.loads(raw)


def reviewed_scopes(paths):
    """Read explicit normalized reviews; never infer PASS from process exit zero.

    New summary writers may add ``ledger_scopes`` using the documented contract.
    The legacy postwrite summary is projected through its dedicated adapter.
    """
    refs, scopes = [], []
    for path in paths:
        ref, data = reference(path)
        refs.append(ref)
        if data.get("schema") == "demoncat.postwrite.constructed_acceptance_summary.v1":
            for case in data["cases"]:
                run = case["runtime"]
                scopes.append({
                    "mechanism": case["mechanism"],
                    "scope_id": "postwrite-a2/" + case["mechanism"],
                    "scope_kind": case["scope"],
                    "scope_description": "BF16 constructed inference module; one actual bound call; ordinary format",
                    "entrypoint": "constructed_model_replay",
                    "public_worker_qualified": False,
                    "pretrained_model": False,
                    "source": ref,
                    "runtime_evidence": {k: run[k] for k in (
                        "baseline_result", "fault_result", "source_sha256", "library_sha256",
                        "actual_operator", "shape", "dtype", "logical_indices", "consumers",
                        "fault_hits", "off_hits", "exact_off", "live_leases", "changed_elements",
                        "cache_restored_exact", "non_target_exact")},
                    "E1_execution_observed": True,
                    "E2_intervention_observed": run["fault_hits"] == 1,
                    "E3_effect_observed": run["changed_elements"] > 0,
                    "E4_off_and_local_recovery": run["exact_off"] is True and run["live_leases"] == 0,
                    "independent_statistics": {
                        "status": case["independent_statistics"]["first_difference"]["status"],
                        "first_difference": case["independent_statistics"]["first_difference"],
                        "report": case["independent_report_source"],
                        "whole_dump_comparable": case["independent_statistics"]["whole_dump_comparable"],
                        "baseline_self_status": case["independent_statistics"]["baseline_self_status"],
                        "nonfinite_and_null_retained": True,
                        "request_mapping": "UNPROVEN_REQUEST_MAPPING"},
                    "all_eligible": False, "every_match": False,
                })
        else:
            rows = data.get("ledger_scopes")
            if not isinstance(rows, list) or not rows:
                raise ValueError("REVIEW_SUMMARY_NEEDS_EXPLICIT_LEDGER_SCOPES:" + path)
            for row in rows:
                row = copy.deepcopy(row)
                row["source"] = ref
                scopes.append(row)
    for row in scopes:
        validate_scope(row)
    return refs, scopes


def validate_scope(row):
    required = ("mechanism", "scope_id", "scope_kind", "scope_description", "entrypoint",
                "public_worker_qualified", "pretrained_model", "runtime_evidence", "independent_statistics",
                "E1_execution_observed", "E2_intervention_observed", "E3_effect_observed",
                "E4_off_and_local_recovery", "all_eligible", "every_match")
    if any(k not in row for k in required) or row["mechanism"] not in CURRENT:
        raise ValueError("INCOMPLETE_OR_UNKNOWN_SCOPED_REVIEW")
    for key in ("public_worker_qualified", "pretrained_model", "all_eligible", "every_match"):
        if type(row[key]) is not bool:
            raise ValueError("REVIEW_FLAG_NOT_BOOLEAN:" + key)
    if row["all_eligible"] or row["every_match"]:
        raise ValueError("C2_C3_NEED_SEPARATE_COVERAGE_REVIEW_NOT_THIS_LEDGER")
    if row["public_worker_qualified"]:
        if not row["pretrained_model"] or row["entrypoint"] != "standard_vllm_worker":
            raise ValueError("CONSTRUCTED_SCOPE_CANNOT_QUALIFY_WORKER")
        if any(row[k] is not True for k in (
                "E1_execution_observed", "E2_intervention_observed", "E3_effect_observed", "E4_off_and_local_recovery")):
            raise ValueError("WORKER_E1_E4_REQUIRED")
    if not row["runtime_evidence"] or not row["scope_description"]:
        raise ValueError("SCOPED_RUNTIME_RECEIPT_REQUIRED")


def scope_stages(row):
    native = all(row.get(k) is True for k in (
        "E1_execution_observed", "E2_intervention_observed", "E3_effect_observed"))
    return {
        "S1": "IMPLEMENTATION_PRESENT_NOT_COMPLETE_ENGINEERING_ACCEPTANCE",
        "S2": "OBSERVED_SCOPE_ONLY",
        "S3": "PASS_IN_RECORDED_SCOPE" if native else "UNVERIFIED_OR_REFUSED",
        "S4": "OFF_AND_LOCAL_RECOVERY_IN_SCOPE" if row.get("E4_off_and_local_recovery") is True else "UNVERIFIED",
        "S5": "UNVERIFIED", "S6": "UNVERIFIED",
    }


def build(matrix, matrix_raw, review_paths):
    refs, rows = reviewed_scopes(review_paths)
    cpu_validation = None
    cpu_path = EXP + "cpu_validation_summary.json"
    if (ROOT / cpu_path).is_file():
        cpu_ref, cpu = reference(cpu_path)
        if cpu.get("schema") != "demoncat.engineering.cpu_validation.v1" or cpu.get("actual_npu_calls") != 0:
            raise ValueError("UNRECOGNIZED_CPU_RECEIPT")
        skipped = sum(len(s["skips"]) for s in cpu["suites"])
        cpu_validation = {"source": cpu_ref, "status": cpu["status"], "attempted": cpu["tests"],
                          "passed": cpu["tests"] - skipped if all(s["exit_code"] == 0 for s in cpu["suites"]) else None,
                          "skipped": skipped, "suites": cpu["suites"], "actual_npu_calls": 0,
                          "grants_native_or_worker_admission": False}
    base = {m["mechanism"]: m for m in matrix["baseline_snapshot"]["mechanisms"]}
    mechanisms = {m["mechanism"]: m for m in matrix["mechanisms"]}
    if set(mechanisms) != set(CURRENT):
        raise ValueError("MECHANISM_IDS_DRIFTED_FROM_REVIEW")
    nodes = []
    file_owners = {}
    for entry in matrix["workstreams"]:
        for path in entry["owns"]:
            file_owners[path] = entry["id"]
    for path in matrix["lead_owns"]:
        file_owners[path] = "lead"
    for task in matrix["m1_m2_tasks"]:
        for path in PUBLIC_FILES[task["id"]]:
            existing = file_owners.setdefault(path, task["owner"])
            if existing != task["owner"]:
                raise ValueError("CONFLICTING_PRIMARY_FILE_OWNER:" + path)
    # Explicit controller delegation: B owns these new QMM boundary files;
    # integration ownership of mechanism 20 stays C from the frozen plan.
    delegated = {SERVING + "serving20_qmm.py": "B", "fault_injection/serving_qualification.py": "B",
                 SERVING + "serving20.py": "B", PROVIDER + "fault20.py": "B"}
    file_owners.update(delegated)
    for mid, info in CURRENT.items():
        for path in info["files"]:
            file_owners.setdefault(path, delegated.get(path, mechanisms[mid]["owner"]))
    commit = {"source_stage": SOURCE_COMMIT, "publication_commit": SOURCE_COMMIT,
              "publication_state": "IMPLEMENTATION_AND_RECEIPTS_COMMITTED"}
    for task in matrix["m1_m2_tasks"]:
        nodes.append({
            "task_id": task["id"], "kind": "public_infrastructure", "workstream": task["owner"],
            "owner": task["owner"], "dependencies": task["depends_on"],
            "owned_files": [{"path": x, "primary_owner": file_owners[x]} for x in PUBLIC_FILES[task["id"]]],
            "deliverable": task["acceptance"],
            "tests": {"required": PUBLIC_TESTS[task["id"]], "interpretation": "Test presence is not an execution receipt"},
            "exit_criteria": [task["acceptance"], "Attach reviewed implementation/test receipts; no CPU-to-NPU promotion"],
            "blocker_policy": COMMON_POLICY, "commit": commit,
            "status": "FROZEN" if task["id"] == "freeze" else "IN_PROGRESS_NOT_GLOBALLY_ACCEPTED",
        })
    for mid, m in mechanisms.items():
        info, historical = CURRENT[mid], base[mid]
        current_rows = [dict(row, stage_assessment=scope_stages(row)) for row in rows if row["mechanism"] == mid]
        historical_worker = copy.deepcopy(historical["worker_runtime_observations"])
        for row in historical_worker:
            row["historical_frozen_scope"] = True
        nodes.append({
            "task_id": mid, "kind": "mechanism", "workstream": m["owner"], "owner": m["owner"],
            "dependencies": m["m1"]["depends_on"],
            "owned_files": [{"path": x, "primary_owner": file_owners[x],
                             "delegated": file_owners[x] != m["owner"]} for x in info["files"]],
            "referenced_historical_sources": m["sources"],
            "deliverable": {"current": info["summary"], "frozen_minimum_reproduction": m["m1"]["minimum_reproduction"]},
            "tests": {"required": info["tests"], "historical_worker_scopes": historical_worker,
                      "current_reviewed_scopes": current_rows,
                      "frozen_statistics": historical["independent_statistics_observations"]},
            "exit_criteria": ["Complete S1 and S2 for a declared scope; actual single injection and effect for S3",
                              "Verify OFF, drain, local resources and whole-group obligations before claiming S4",
                              "Publish independent statistics outcome, including failure or non-observability",
                              "S5/S6 need separate model/environment and coverage matrices; no config-only promotion"],
            "blocker_policy": dict(COMMON_POLICY, current_blocker=info["remaining"]),
            "commit": commit, "status": "IN_PROGRESS_SCOPED_EVIDENCE_ONLY",
            "stage_assessment": {
                "S1": {"status": "IMPLEMENTED_PENDING_COMPLETE_ENGINEERING_REVIEW", "scope": "current implementation"},
                "S2": {"status": "PARTIAL_SCOPE_ONLY", "scope": "actual observed structure; unobserved roles remain unverified"},
                "S3": {"status": "SCOPED_EVIDENCE_ONLY" if historical_worker or current_rows else "UNVERIFIED_CURRENT_DELTA",
                       "scope": "historical and current rows separately; not all models or routes"},
                "S4": {"status": "SCOPED_EVIDENCE_ONLY" if historical_worker or current_rows else "UNVERIFIED_CURRENT_DELTA",
                       "scope": "recorded local OFF; no implied multi-rank or whole-service recovery"},
                "S5": {"status": "UNVERIFIED", "scope": "model and environment matrices"},
                "S6": {"status": "UNVERIFIED", "scope": "no C2/C3 admission"},
            },
            "historical_snapshot": copy.deepcopy(historical),
            "coverage": {"C1": "ONLY_RECORDED_QUALIFIED_SCOPES", "C2_all_eligible": False,
                         "C3_every_match": False, "whole_group_recovery": "UNVERIFIED_OUTSIDE_EXPLICIT_SCOPE"},
        })
    order = matrix["baseline_snapshot"]["m1_m2_topological_order"]
    nodes.sort(key=lambda x: order.index(x["task_id"]))
    old_workers = {k for k, v in base.items() if v["worker_e1_e4_verified_in_recorded_scope"]}
    new_workers = {r["mechanism"] for r in rows if r["public_worker_qualified"]}
    ledger = {
        "schema": "demoncat.engineering.program_ledger.v1",
        "source_stage_commit": SOURCE_COMMIT,
        "frozen_matrix": {"path": MATRIX_PATH, "git_commit": SOURCE_COMMIT, "sha256": sha(matrix_raw),
                          "baseline_commit": matrix["baseline_commit"]},
        "generator": {"path": Path(__file__).relative_to(ROOT).as_posix(), "sha256": sha(Path(__file__).read_bytes())},
        "reviewed_summary_inputs": refs,
        "cpu_validation": cpu_validation,
        "definitions": {"stages": STAGES, "scope_key": matrix["scope_key"],
                        "ownership": "Exclusive by primary workstream; multiple nodes in that workstream may use the same file",
                        "commit": "Source stage is fixed; publication_commit remains null until the controller records a real commit",
                        "counts": "Public Worker count means at least one scoped E1-E4 receipt, never universal support",
                        "stage_progression": "S3/S4 scoped evidence may precede complete S1 integration; stages are not inferred from labels",
                        "statistics": matrix["statistics_policy"]},
        "nodes": nodes,
        "topological_order": order,
        "primary_file_ownership": dict(sorted(file_owners.items())),
        "controller_delegation": {"integration_owner_unchanged": "C", "implementation_owner": "B",
                                  "mechanism": "20.buffer_substitute", "owned_files": sorted(delegated),
                                  "reason": "Explicit root delegation of natural QMM projection boundary and minimal dispatch changes"},
        "counts": {"public_nodes": 10, "mechanisms": 17, "all_nodes": 27,
                   "historical_public_worker_mechanisms": len(old_workers),
                   "public_worker_mechanisms_with_recorded_scope": len(old_workers | new_workers),
                   "new_public_worker_mechanisms": sorted(new_workers - old_workers),
                   "all_17_s1_accepted": False, "m1_complete": False, "m2_complete": False,
                   "C2_all_eligible_qualified": 0, "C3_every_match_qualified": 0},
        "model_family_matrix": [
            {"mechanism": m, "model_family": family, "status": "UNVERIFIED",
             "scope": "family-wide coverage; consult scoped receipts, not model names"}
            for m in mechanisms for family in ("Qwen", "GLM", "DeepSeek")],
        "environment_matrix": [
            {"mechanism": m, "cann": version, "soc_family": soc, "status": "UNVERIFIED",
             "scope": "complete mechanism deployment; operator-only historical tests remain separate"}
            for m in mechanisms for version in ("8.5.1", "9.0.0", "9.1.0", "9.2.0") for soc in ("A2", "A3")],
        "limitations": [
            "No model-family or environment cell is promoted by a name, import, build success or constructed tensor test.",
            "A3 hardware and the current yellow-zone host source/environment are unavailable.",
            "Every scope retains its recorded model, API, layout, version, build and Worker/rank constraints.",
            "New raw statistics, full private tensors and binaries remain outside this publication.",
        ],
    }
    validate(ledger, matrix)
    return ledger


def validate(ledger, matrix):
    nodes = ledger["nodes"]
    ids = [n["task_id"] for n in nodes]
    expected = {x["id"] for x in matrix["m1_m2_tasks"]} | {x["mechanism"] for x in matrix["mechanisms"]}
    if len(ids) != 27 or len(set(ids)) != 27 or set(ids) != expected:
        raise ValueError("EXACT_FROZEN_27_NODES_REQUIRED")
    seen = set()
    for node in nodes:
        for field in ("task_id", "workstream", "dependencies", "owner", "owned_files", "deliverable", "tests", "exit_criteria", "blocker_policy", "commit"):
            if field not in node:
                raise ValueError("MISSING_NODE_FIELD:" + field)
        if not set(node["dependencies"]) <= seen:
            raise ValueError("DEPENDENCY_MISSING_CYCLIC_OR_OUT_OF_ORDER")
        if node["commit"]["source_stage"] != SOURCE_COMMIT:
            raise ValueError("SOURCE_STAGE_DRIFT")
        seen.add(node["task_id"])
        if node["kind"] == "mechanism":
            if set(node["stage_assessment"]) != set(STAGES):
                raise ValueError("SIX_STAGES_REQUIRED")
            if node["coverage"]["C2_all_eligible"] or node["coverage"]["C3_every_match"]:
                raise ValueError("UNREVIEWED_C2_C3_PROMOTION")
            for row in node["tests"]["current_reviewed_scopes"]:
                validate_scope(row)
    if len(ledger["model_family_matrix"]) != 51 or len(ledger["environment_matrix"]) != 136:
        raise ValueError("INCOMPLETE_GENERALIZATION_MATRIX")
    if len({(x["mechanism"], x["model_family"]) for x in ledger["model_family_matrix"]}) != 51:
        raise ValueError("DUPLICATE_MODEL_MATRIX_CELL")
    if len({(x["mechanism"], x["cann"], x["soc_family"]) for x in ledger["environment_matrix"]}) != 136:
        raise ValueError("DUPLICATE_ENVIRONMENT_MATRIX_CELL")
    if any(r["status"] != "UNVERIFIED" for r in ledger["model_family_matrix"] + ledger["environment_matrix"]):
        raise ValueError("UNREVIEWED_GENERALIZATION_PROMOTION")
    originals = {m["mechanism"]: m for m in matrix["baseline_snapshot"]["mechanisms"]}
    for node in nodes:
        if node["kind"] == "mechanism" and node["historical_snapshot"] != originals[node["task_id"]]:
            raise ValueError("HISTORICAL_SCOPE_CHANGED")
    if any(ledger["counts"][k] for k in ("all_17_s1_accepted", "m1_complete", "m2_complete", "C2_all_eligible_qualified", "C3_every_match_qualified")):
        raise ValueError("UNREVIEWED_GLOBAL_COMPLETION")


def self_test(ledger, matrix):
    class LedgerTests(unittest.TestCase):
        def test_valid(self):
            validate(ledger, matrix)

        def assert_refused(self, mutate):
            value = copy.deepcopy(ledger)
            mutate(value)
            with self.assertRaises(ValueError):
                validate(value, matrix)

        def test_missing_node(self):
            self.assert_refused(lambda d: d["nodes"].pop())

        def test_dependency_cycle(self):
            self.assert_refused(lambda d: d["nodes"][0]["dependencies"].append("m1_m2_integrate"))

        def test_stage_missing(self):
            self.assert_refused(lambda d: next(n for n in d["nodes"] if n["kind"] == "mechanism")["stage_assessment"].pop("S6"))

        def test_family_name_is_not_support(self):
            self.assert_refused(lambda d: d["model_family_matrix"][0].update(status="PASS"))

        def test_environment_is_not_support(self):
            self.assert_refused(lambda d: d["environment_matrix"][0].update(status="PASS"))

        def test_duplicate_matrix_cell(self):
            self.assert_refused(lambda d: d["model_family_matrix"].__setitem__(0, copy.deepcopy(d["model_family_matrix"][1])))

        def test_historical_scope_immutable(self):
            self.assert_refused(lambda d: next(n for n in d["nodes"] if n["task_id"] == "39.matmul_range")["historical_snapshot"].update(worker_registered=False))

        def test_no_global_s1_from_files(self):
            self.assert_refused(lambda d: d["counts"].update(all_17_s1_accepted=True))

        def test_no_c2_from_config(self):
            self.assert_refused(lambda d: next(n for n in d["nodes"] if n["kind"] == "mechanism")["coverage"].update(C2_all_eligible=True))

        def test_constructed_cannot_be_worker(self):
            row = {"mechanism": "15.alias_detach", "scope_id": "test", "scope_kind": "CONSTRUCTED",
                   "scope_description": "test", "entrypoint": "constructed_model_replay", "pretrained_model": False,
                   "public_worker_qualified": True, "runtime_evidence": {"receipt": "test"},
                   "independent_statistics": {}, "all_eligible": False, "every_match": False,
                   "E1_execution_observed": True, "E2_intervention_observed": True,
                   "E3_effect_observed": True, "E4_off_and_local_recovery": True}
            with self.assertRaises(ValueError):
                validate_scope(row)

    result = unittest.TextTestRunner(verbosity=1).run(unittest.defaultTestLoader.loadTestsFromTestCase(LedgerTests))
    return result.wasSuccessful()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--matrix", type=Path, help="exact frozen matrix bytes, for an archive without Git")
    p.add_argument("--review-summary", action="append", default=[], help="repository-relative reviewed summary with ledger_scopes")
    p.add_argument("--out", type=Path, default=HERE / "program_ledger.json")
    p.add_argument("--check", action="store_true")
    p.add_argument("--self-test", action="store_true")
    args = p.parse_args()
    matrix, raw = read_matrix(args.matrix)
    paths = [x for x in DEFAULT_REVIEW_SUMMARIES if (ROOT / x).is_file()]
    paths += [x for x in args.review_summary if x not in paths]
    ledger = build(matrix, raw, paths)
    if args.self_test and not self_test(ledger, matrix):
        return 1
    serialized = (json.dumps(ledger, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    if args.check:
        if args.out.read_bytes() != serialized:
            raise ValueError("LEDGER_STALE_REGENERATE_FROM_REVIEWED_INPUTS")
    else:
        args.out.write_bytes(serialized)
    print(json.dumps({"status": "LEDGER_VERIFIED" if args.check else "LEDGER_GENERATED", **ledger["counts"],
                      "output_sha256": sha(serialized), "runtime_admission_granted": False}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
