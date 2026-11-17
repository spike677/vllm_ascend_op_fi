"""Inspect CANN and native build inputs without initializing an NPU."""

import argparse
import json
from pathlib import Path

from fault_injection.environment import discover_installations, inspect_environment
from fault_injection.native_build import BuildError, NativeBuildManager, resolve_recipe


_BASELINES = {
    "8.5.1": {
        "profile": "profiles/environments/cann-8.5.1-prior-124.json",
        "evidence": "experiments/c2c3_closure_20261117/C2_C3_MATRIX.md",
        "scope": "Scoped Qwen3-0.6B and DeepSeek-V2-Lite-W8A8 Worker receipts on 910B4/A2.",
    },
    "9.1.0": {
        "profile": "profiles/environments/cann-9.1.0-prior-124.json",
        "evidence": "fault39/docs/cann_version_adaptation.md",
        "scope": "Historical Fault39 Qwen3 TP2; separate standalone FIA and collective receipts."
                 " Fresh standard Worker validation is required.",
    },
}


def assess_compatibility(environment, cann, *, mechanism=None, operator=None):
    """Evaluate the selected toolkit and recipe against one reviewed baseline."""
    if cann not in _BASELINES:
        raise ValueError(f"CANN_BASELINE_NOT_REVIEWED: {cann}")
    if operator is not None and mechanism is None:
        raise ValueError("OPERATOR_REQUIRES_MECHANISM")

    toolkit = environment["toolkit"]
    selected = toolkit["installations"]
    installation = selected[0] if toolkit["status"] == "FOUND" and len(selected) == 1 else None
    version = installation["fields"].get("version") if installation else None
    if toolkit["status"] == "AMBIGUOUS" or len(selected) > 1:
        toolkit_status = "AMBIGUOUS_TOOLKIT"
    elif installation is None:
        toolkit_status = "TOOLKIT_NOT_SELECTED"
    elif version != cann:
        toolkit_status = "SELECTED_VERSION_MISMATCH"
    else:
        toolkit_status = "SELECTED_VERSION_MATCH"

    links = environment["loaded_toolkit_links"]
    mapped = [match for row in links["libraries"] for match in row["matches"]]
    if links["status"] == "MULTIPLE_INSTALLATIONS_LOADED" or any(
        row["status"] == "AMBIGUOUS_INSTALLATION" for row in links["libraries"]
    ) or any(
        match["version"] != cann or (installation and match["toolkit"] != installation["toolkit"])
        for match in mapped
    ):
        mapped_status = "LOADED_TOOLKIT_MISMATCH_OR_AMBIGUITY"
    elif environment["maps_status"] != "OBSERVED" or not mapped:
        mapped_status = "NO_VERIFIED_TOOLKIT_MAPPING"
    else:
        mapped_status = "MAPPED_FILES_LINKED_TO_BASELINE"

    build = {"status": "NO_RECIPE_REQUESTED"}
    if mechanism is not None:
        try:
            recipe = resolve_recipe(mechanism, operator)
        except ValueError as error:
            build = {"status": str(error), "mechanism": mechanism, "operator": operator}
        else:
            allowed = not recipe.allowed_versions or cann in recipe.allowed_versions
            build = {
                "status": "RECIPE_AVAILABLE_ABI_CHECK_REQUIRED" if allowed else "RECIPE_ABI_NOT_ADAPTED",
                "recipe": recipe.as_dict(),
            }

    soc = environment["soc"]
    exact_a2_scope = (
        environment["soc_source"] == "INITIALIZED_TORCH_NPU_DEVICE_QUERY"
        and soc and soc["family"] == "A2"
        and soc["observed_name"].lower().replace("ascend", "").strip() == "910b4"
    )
    return {
        "schema": "demoncat.engineering.compatibility.v1",
        "requested_cann": cann,
        "historical_baseline": dict(_BASELINES[cann]),
        "toolkit": {"status": toolkit_status, "selection": toolkit},
        "loaded_toolkit": {"status": mapped_status, "links": links},
        "hardware": {
            "status": "PRIOR_910B4_SCOPE_MATCH" if exact_a2_scope else "SEPARATE_SOC_PROOF_REQUIRED",
            "observation": soc,
            "source": environment["soc_source"],
        },
        "build_inputs": build,
        "worker_environment": {
            "packages": environment["packages"],
            "driver": environment["driver"],
            "torch_cxx11_abi": environment["torch_cxx11_abi"],
            "maps_status": environment["maps_status"],
            "mapped_libraries": environment["loaded_libraries"],
        },
        "runtime_status": "EXACT_LIVE_WORKER_AND_OPERATOR_PROOF_REQUIRED",
        "runtime_qualified": False,
    }


def inspect_compatibility(cann, *, mechanism=None, operator=None, fingerprint_libraries=False,
                          inspect_build=False, ascend_root=None):
    """Read installed identities, optionally checking the selected recipe's build inputs."""
    if inspect_build and mechanism is None:
        raise ValueError("BUILD_INSPECTION_REQUIRES_MECHANISM")
    environment = inspect_environment()
    report = assess_compatibility(environment, cann, mechanism=mechanism, operator=operator)
    report["installed_toolkits"] = (
        discover_installations(ascend_root, fingerprint_libraries=fingerprint_libraries)
        if ascend_root is not None or fingerprint_libraries else environment["installed_toolkits"]
    )
    if inspect_build and report["build_inputs"]["status"] == "RECIPE_AVAILABLE_ABI_CHECK_REQUIRED":
        if report["toolkit"]["status"] != "SELECTED_VERSION_MATCH":
            report["build_inputs"]["status"] = "SELECTED_BASELINE_REQUIRED"
        elif report["loaded_toolkit"]["status"] == "LOADED_TOOLKIT_MISMATCH_OR_AMBIGUITY":
            report["build_inputs"]["status"] = "MIXED_PROCESS_TOOLKIT_REFUSED"
        else:
            installation = report["toolkit"]["selection"]["installations"][0]
            try:
                inspected = NativeBuildManager().inspect(mechanism, operator, toolkit=installation["toolkit"])
                identity = inspected["inputs"]["installation"]
                if identity["fields"]["version"] != cann or identity["sha256"] != installation["sha256"]:
                    raise BuildError("TOOLKIT_IDENTITY_CHANGED_DURING_INSPECTION")
            except (BuildError, OSError, ValueError) as error:
                report["build_inputs"].update(status="INSPECTION_REFUSED", reason=str(error))
            else:
                report["build_inputs"] = inspected
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cann", choices=(*_BASELINES, "9.1"), required=True)
    parser.add_argument("--mechanism")
    parser.add_argument("--operator")
    parser.add_argument("--ascend-root", type=Path, help="Inventory a directory separately from live toolkit selection.")
    parser.add_argument("--fingerprint-libraries", action="store_true")
    parser.add_argument("--inspect-build", action="store_true")
    args = parser.parse_args(argv)
    try:
        report = inspect_compatibility(
            "9.1.0" if args.cann == "9.1" else args.cann,
            mechanism=args.mechanism, operator=args.operator, ascend_root=args.ascend_root,
            fingerprint_libraries=args.fingerprint_libraries, inspect_build=args.inspect_build,
        )
    except (OSError, RuntimeError, ValueError) as error:
        print(json.dumps({"status": "REFUSED", "reason": str(error), "runtime_qualified": False}))
        return 2
    print(json.dumps(report, indent=2, allow_nan=False))
    inspected_states = {"INPUTS_INSPECTED_ABI_COMPILE_REQUIRED", "NO_RECIPE_REQUESTED",
                        "RECIPE_AVAILABLE_ABI_CHECK_REQUIRED"}
    return 0 if (
        report["toolkit"]["status"] == "SELECTED_VERSION_MATCH"
        and report["loaded_toolkit"]["status"] != "LOADED_TOOLKIT_MISMATCH_OR_AMBIGUITY"
        and report["build_inputs"]["status"] in inspected_states
    ) else 2


if __name__ == "__main__":
    raise SystemExit(main())
