"""Rebuild or verify the roadmap snapshot using the declared Git baseline."""

import argparse
import ast
from functools import cache
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import subprocess
from graphlib import TopologicalSorter


MATRIX = Path(__file__).with_name("17_fault_acceptance_matrix.json")
REPO = MATRIX.parents[2]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def index(rows, key):
    result = {row[key]: row for row in rows}
    require(len(result) == len(rows), f"Duplicate {key}")
    return result


def catalog_entries(source):
    # These declarations are literal data. Read them without importing any provider.
    declaration = next(node for node in ast.parse(source).body if isinstance(node, ast.Assign)
                       and any(isinstance(target, ast.Name) and target.id == "CAPABILITIES"
                               for target in node.targets))
    return {ast.literal_eval(call.args[0]): ast.literal_eval(call.args[2]) for call in declaration.value.elts}


def worker_entries(source):
    function = next(node for node in ast.parse(source).body if isinstance(node, ast.FunctionDef)
                    and node.name == "provider_capabilities")
    return index(ast.literal_eval(function.body[0].value), "mechanism")


def runtime_observation(kind, data):
    """Read the scoped execution, effect and recovery facts from each receipt format."""
    if kind in ("weight_nz", "matmul_range"):
        phases = index(data["native_phases"], "phase")
        qualification = data["qualification"]
        fault = phases["fault"]
        off = phases["disabled_off" if kind == "weight_nz" else "off"]
        keys = ["model_output_sha256"]
        lossless = ["lossless"] if kind == "weight_nz" else ["lossless_self", "lossless_mat2", "lossless_dual"]
        recovered = (qualification["recovery_verified"] is True
                     and qualification["drain_status"] == "DISABLED_CONFIRMED"
                     and data["cleanup"]["unregister"] == "ACTION_COMPLETED")
        if kind == "weight_nz":
            call, = fault["native"]
            receipt = call["receipt"]
            native = (call["operator"] == "aclnnMatmulWeightNz" and call["actual_original_calls"] == 1
                      and call["current_typed_formals_verified"] is True and call["original_storage_preserved"] is True
                      and receipt["native_status"] == 0 and receipt["replacements"] == 1
                      and receipt["temporary_readback_verified"] is True and receipt["resources_released"] is True
                      and off["native_calls"] == 0)
            hits = receipt["replacements"]
        else:
            native = (fault["actual_api"] == "aclnnMatmul" and fault["hits"] == 1
                      and fault["actual_live_inputs_verified"] is True and fault["original_storage_preserved"] is True
                      and fault["native_idle"] is True and fault["statistics"]["output"]["posinf"] > 0)
            hits = fault["hits"]
        executed = (data["status"] == "PASS_COMMON_WORKER_NATIVE_BFOFF"
                    and all(row["actual_model_request_matches_enqueue"] is True for row in data["phases"]))
    elif kind == "fia_destination":
        phases = index(data["phases"], "phase")
        fault, off = phases["comparison"], phases["off"]
        keys = ["hidden_sha256", "logits_sha256"]
        lossless = ["lossless"]
        proof = fault["writer_consumer_recovery"]
        hits = fault["native_claims"]
        native = (hits == fault["completed"] and hits > 0 and fault["all_native_pass"] is True
                  and all(proof[k] is True for k in ("all_real_kernel_writes", "all_consumers_matched",
                                                    "all_original_fia_outputs_copied", "all_residuals_changed")))
        recovered = data["recovered"] is True and proof["all_residuals_restored"] is True and fault["pending"] == 0
        executed = (data["common_worker"] is True and data["baseline_control"] is False
                    and all(row["actual_scheduler_driver_mapping"] is True for row in data["phases"]))
    elif kind == "quant_v2":
        model = data["models"]["fault"]
        phases = index(model["phases"], "phase")
        fault, off = phases["comparison"], phases["off"]
        keys = ["hidden_sha256", "logits_sha256"]
        lossless = ["lossless"]
        receipt, = fault["native_receipts"]
        hits = fault["actual_native_fault_hits"]
        native = (fault["actual_native_execution_proven"] is True and hits == 1
                  and receipt["prepare_hits"] == receipt["execute_hits"] == receipt["fault_executions"] == 1
                  and receipt["api_version"] == 2 and receipt["prepare_errors"] == receipt["execution_errors"] == 0)
        native = native and data["complete_q_changed"] is True and data["fault_scale_equals_baseline_largest"] is True
        recovered = model["recovered"] is True and off["actual_native_fault_hits"] == 0
        executed = model["core_methods_replaced"] is False and fault["actual_request_mapping"] is True
    else:
        raise ValueError(f"Unknown Worker receipt format: {kind}")

    reference = phases["reference"]
    lossless_phases = [phases[stage] for stage in lossless]
    values = [stage[key] for stage in (reference, fault, off, *lossless_phases) for key in keys]
    require(all(isinstance(value, str) and re.fullmatch("[0-9a-f]{64}", value) for value in values),
            f"Missing full output hash: {kind}")
    output_key = keys[-1]  # Complete model output, not an intermediate hidden state.
    return {
        "E1_execution_observed": executed and all(
            stage[key] == reference[key] for stage in lossless_phases for key in keys),
        "E2_native_intervention": native,
        "E3_model_output_changed": fault[output_key] != reference[output_key],
        "E4_off_recovery": recovered and all(off[key] == reference[key] for key in keys),
        "actual_fault_hits": hits,
    }


def statistics_observation(report):
    if "strict_original_result" in report:
        require(report["strict_acceptance_overridden"] is False
                and report["global_first_difference"] is None
                and report["native_logs_read"] is False and report["tensor_payloads_read"] is False,
                "A diagnostic report must preserve the original strict result")
        return {"strict_status": report["strict_original_result"]["status"], "scope": "DIAGNOSTIC_ONLY"}

    def comparable(section):
        require(section["source"] == "RAW_MSPROBE_STATISTICS_MIX"
                and section["native_logs_read"] is False and section["private_tensors_read"] is False,
                "Statistics comparison depends on non-statistical evidence")
        require(section["baseline_config_sha256"] == section["fault_config_sha256"]
                and section["baseline_collector_sha256"] == section["fault_collector_sha256"],
                "Statistics configuration or collector differs")
        result = section["comparison"]
        policy = result["comparison_policy"]
        require(policy["relative_tolerance"] == 1e-5 and policy["absolute_tolerance"] == 1e-8
                and policy["rule"] == "abs(F-B) > atol + rtol*abs(B)", "Statistics tolerance differs")
        return result

    baseline = comparable(report["baseline_self"])
    pair = comparable(report["comparison"])
    return {
        "strict_status": pair["status"],
        "baseline_self_status": baseline["status"],
        "baseline_self_reason": baseline.get("difference", {}).get("reason"),
        "first_record": pair.get("identity"),
        "difference": pair.get("difference"),
        "unobservable_fields": pair["unobservable_fields_count_before_first_or_all"],
    }


def snapshot(plan):
    require(plan["schema"] == "demoncat.fault_coverage.acceptance_matrix.v1", "Unknown matrix schema")
    commit = plan["baseline_commit"]
    require(re.fullmatch("[0-9a-f]{40}", commit) is not None, "Use a full Git commit ID")
    blob_hashes = {}

    @cache
    def read(name):
        path = PurePosixPath(name)
        require(name == path.as_posix() and not path.is_absolute()
                and ".." not in path.parts and ":" not in name and "\\" not in name,
                f"Invalid repository path: {name}")
        content = subprocess.run(["git", "show", f"{commit}:{name}"], cwd=REPO, capture_output=True)
        require(content.returncode == 0, f"Absent from baseline {commit[:8]}: {name}")
        blob_hashes[name] = hashlib.sha256(content.stdout).hexdigest()
        return content.stdout

    inputs = plan["baseline_inputs"]
    documents = {label: json.loads(read(path)) for label, path in inputs.items()
                 if label not in ("catalog", "worker_registry")}
    require(documents["latest_delta"]["previous_matrix_sha256"] == blob_hashes[inputs["overnight_matrix"]],
            "Latest delta does not refer to the imported historical matrix")
    old = index(documents["overnight_matrix"]["mechanisms"], "mechanism")
    coverage = index(documents["overnight_coverage"]["mechanisms"], "mechanism")
    delta = index(documents["latest_delta"]["mechanisms"], "mechanism")
    catalog = catalog_entries(read(inputs["catalog"]))
    registry = worker_entries(read(inputs["worker_registry"]))
    reviews = index(plan["mechanisms"], "mechanism")
    require(len(catalog) == 17 and catalog.keys() == old.keys() == coverage.keys() == delta.keys() == reviews.keys(),
            "The catalog, historical matrices and reviewed mechanisms must agree on all 17 IDs")

    tasks = index(plan["m1_m2_tasks"], "id")
    workflows = index(plan["workstreams"], "id")
    require(set(workflows) == set("ABCDE"), "Expected five workstreams")
    owned_paths = plan["lead_owns"] + [path for flow in workflows.values() for path in flow["owns"]]
    require(len(owned_paths) == len(set(owned_paths)), "A source path has multiple owners")
    require(registry.keys() <= catalog.keys(), "Worker registry contains an unknown mechanism")
    require(reviews.keys() <= set(tasks["m1_m2_integrate"]["depends_on"]), "Integration omits a mechanism")
    task_owners = set(workflows) | {"lead"}
    for path in owned_paths:
        if not path.startswith("docs/roadmap/"):
            read(path)
    dependency_graph = {}
    for task_id, task in tasks.items():
        require(task["owner"] in task_owners, f"Unknown owner: {task_id}")
        dependency_graph[task_id] = task["depends_on"]

    rows = []
    for mechanism, review in reviews.items():
        task = review["m1"]
        require(review["owner"] in ("C", "D") and mechanism not in dependency_graph,
                f"Invalid owner or task ID: {mechanism}")
        dependency_graph[mechanism] = task["depends_on"]
        require(all(task["gaps"].get(k) for k in ("implementation", "entrypoint", "recovery", "native_build")),
                f"Incomplete M1 gap review: {mechanism}")
        for name in review["sources"] + review["evidence"]:
            read(name)
        historical = old[mechanism]
        card = historical["card"]
        require(json.loads(read(card))["mechanism"] == mechanism, f"Wrong capability card: {mechanism}")
        runtime = [{"receipt": sample["path"], "scope": sample["scope"],
                    **runtime_observation(sample["kind"], json.loads(read(sample["path"])))}
                   for sample in review.get("worker_receipts", [])]
        verified = bool(runtime) and all(all(value for k, value in row.items() if k.startswith("E")) for row in runtime)
        require(verified == delta[mechanism]["new_common_worker_fault_verified"],
                f"Scoped positive receipts disagree with prior delta: {mechanism}")
        require(not verified or mechanism in registry, f"Positive receipt has no current Worker entry: {mechanism}")
        prior_every_match = coverage[mechanism]["qualification"]["every_match"]
        require(coverage[mechanism]["common_worker_all_eligible"] is False
                and prior_every_match in ("UNVERIFIED", "VERIFIED_FOUR_FORWARDS_IN_THIS_MODULE_ONLY")
                and delta[mechanism]["common_worker_all_eligible"] is False, f"Full coverage evidence changed: {mechanism}")
        require(review["statistics_status"] in plan["statistics_policy"]["statuses"],
                f"Unknown statistics annotation: {mechanism}")
        stats = [{"receipt": name, **statistics_observation(json.loads(read(name)))} for name in review["statistics_reports"]]
        rows.append({
            "mechanism": mechanism,
            "capability_card": card,
            "catalog_route": catalog[mechanism],
            "overnight_implementation": historical["implementation"],
            "overnight_level": historical["current_level"],
            "historical_scope": historical["historical_native"]["scope"],
            "overnight_every_match_scope": prior_every_match,
            "latest_evidence_delta": delta[mechanism]["change"],
            "worker_registered": mechanism in registry,
            "worker_e1_e4_verified_in_recorded_scope": verified,
            "worker_runtime_observations": runtime,
            "independent_statistics_observations": stats,
            "all_eligible_qualified": False,
            "every_match_qualified": False,
        })

    for task, dependencies in dependency_graph.items():
        require(len(dependencies) == len(set(dependencies)) and set(dependencies) <= dependency_graph.keys(),
                f"Duplicate or unknown dependency: {task}")
    task_order = list(TopologicalSorter(dependency_graph).static_order())
    positives = sum(row["worker_e1_e4_verified_in_recorded_scope"] for row in rows)
    require(positives == documents["latest_delta"]["new_common_worker_positive_mechanisms"],
            "Scoped positive count disagrees with baseline delta")
    require(documents["latest_delta"]["all_eligible_qualified_mechanisms"] == 0,
            "Baseline full coverage count changed")
    cpu = documents["cpu_receipt"]
    require(cpu["status"] == "PASS_CPU_ONLY" and cpu["actual_npu_calls"] == 0, "Expected a CPU-only receipt")
    return {
        "counts": {"mechanisms": len(rows), "worker_registered": len(registry), "worker_scoped_e1_e4": positives,
                   "all_eligible_qualified": 0, "every_match_qualified": 0,
                   "baseline_cpu_tests": sum(suite["tests"] for suite in cpu["runs"])},
        "m1_m2_topological_order": task_order,
        "mechanisms": rows,
        "baseline_git_blob_sha256": dict(sorted(blob_hashes.items())),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--refresh", action="store_true", help="Regenerate baseline_snapshot, keeping the reviewed plan")
    args = parser.parse_args()
    plan = json.loads(MATRIX.read_bytes())
    actual = snapshot(plan)
    if args.refresh:
        plan["baseline_snapshot"] = actual
        MATRIX.write_bytes((json.dumps(plan, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8"))
    else:
        require(actual == plan["baseline_snapshot"], "Baseline snapshot differs; review the sources before refreshing")
    print(json.dumps(actual["counts"], sort_keys=True))


if __name__ == "__main__":
    main()
