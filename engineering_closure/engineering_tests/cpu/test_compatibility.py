import unittest
from unittest.mock import patch

from engineering_closure.compatibility import assess_compatibility, inspect_compatibility
from fault_injection.environment import classify_soc


def environment(version="8.5.1", soc=None):
    toolkit = f"/toolkit/{version}"
    selection = {"toolkit": toolkit, "fields": {"version": version}, "sha256": "identity"}
    return {
        "toolkit": {"status": "FOUND", "installations": [selection]},
        "loaded_toolkit_links": {"status": "UNRESOLVED", "libraries": []},
        "maps_status": "NONE",
        "loaded_libraries": [],
        "soc": classify_soc(soc) if soc else None,
        "soc_source": "INITIALIZED_TORCH_NPU_DEVICE_QUERY" if soc else "REQUIRES_LIVE_WORKER_OBSERVATION",
        "packages": {},
        "driver": {},
        "installed_toolkits": {"installations": [selection]},
        "torch_cxx11_abi": None,
    }


class CompatibilityTests(unittest.TestCase):
    def test_version_admission_is_recipe_specific(self):
        qmm = {"mechanism": "20.buffer_substitute", "operator": "aclnnQuantMatmulWeightNz"}
        prior = assess_compatibility(environment(), "8.5.1", **qmm)
        newer = assess_compatibility(environment("9.1.0"), "9.1.0", **qmm)
        self.assertEqual(prior["build_inputs"]["status"], "RECIPE_AVAILABLE_ABI_CHECK_REQUIRED")
        self.assertEqual(newer["build_inputs"]["status"], "RECIPE_ABI_NOT_ADAPTED")
        self.assertNotEqual(prior["historical_baseline"]["profile"], newer["historical_baseline"]["profile"])

    def test_mechanism_needs_exact_operator_for_multirecipe_path(self):
        report = assess_compatibility(environment(), "8.5.1", mechanism="20.buffer_substitute")
        self.assertEqual(report["build_inputs"]["status"], "ACTUAL_OPERATOR_REQUIRED")

    def test_toolkit_selection_and_loaded_identity_are_separate(self):
        live = environment("9.1.0")
        live["maps_status"] = "OBSERVED"
        live["loaded_toolkit_links"] = {
            "status": "PARTIALLY_LINKED",
            "libraries": [{"status": "PATH_LINKED_TO_INSTALLATION",
                           "matches": [{"toolkit": "/toolkit/8.5.1", "version": "8.5.1"}]}],
        }
        report = assess_compatibility(live, "9.1.0", mechanism="39.matmul_range")
        self.assertEqual(report["toolkit"]["status"], "SELECTED_VERSION_MATCH")
        self.assertEqual(report["loaded_toolkit"]["status"], "LOADED_TOOLKIT_MISMATCH_OR_AMBIGUITY")
        self.assertFalse(report["runtime_qualified"])

    def test_same_version_from_different_loaded_installation_refuses(self):
        live = environment()
        live["maps_status"] = "OBSERVED"
        live["loaded_toolkit_links"] = {
            "status": "PARTIALLY_LINKED",
            "libraries": [{"status": "PATH_LINKED_TO_INSTALLATION",
                           "matches": [{"toolkit": "/other/8.5.1", "version": "8.5.1"}]}],
        }
        result = assess_compatibility(live, "8.5.1")
        self.assertEqual(result["loaded_toolkit"]["status"], "LOADED_TOOLKIT_MISMATCH_OR_AMBIGUITY")

    def test_hardware_scope_does_not_promote_to_runtime_qualification(self):
        for soc in (None, "Ascend910B2", "Ascend910_9391"):
            with self.subTest(soc=soc):
                result = assess_compatibility(environment(soc=soc), "8.5.1")
                self.assertEqual(result["hardware"]["status"], "SEPARATE_SOC_PROOF_REQUIRED")
        result = assess_compatibility(environment(soc="Ascend910B4"), "8.5.1")
        self.assertEqual(result["hardware"]["status"], "PRIOR_910B4_SCOPE_MATCH")
        self.assertFalse(result["runtime_qualified"])

    @patch("engineering_closure.compatibility.NativeBuildManager")
    @patch("engineering_closure.compatibility.inspect_environment")
    def test_wrong_selected_version_never_invokes_build_inspection(self, inspect, manager):
        inspect.return_value = environment("8.5.1")
        result = inspect_compatibility("9.1.0", mechanism="39.matmul_range", inspect_build=True)
        self.assertEqual(result["build_inputs"]["status"], "SELECTED_BASELINE_REQUIRED")
        manager.assert_not_called()

    @patch("engineering_closure.compatibility.NativeBuildManager")
    @patch("engineering_closure.compatibility.inspect_environment")
    def test_build_input_inspection_is_pinned_to_selected_identity(self, inspect, manager):
        inspect.return_value = environment("9.1.0")
        inspect_build = manager.return_value.inspect
        inspect_build.return_value = {
            "status": "INPUTS_INSPECTED_ABI_COMPILE_REQUIRED",
            "inputs": {"installation": {"fields": {"version": "9.1.0"}, "sha256": "identity"}},
        }
        result = inspect_compatibility("9.1.0", mechanism="39.matmul_range", inspect_build=True)
        inspect_build.assert_called_once_with("39.matmul_range", None, toolkit="/toolkit/9.1.0")
        self.assertEqual(result["build_inputs"]["status"], "INPUTS_INSPECTED_ABI_COMPILE_REQUIRED")
        self.assertFalse(result["runtime_qualified"])

        inspect_build.return_value["inputs"]["installation"]["sha256"] = "changed"
        result = inspect_compatibility("9.1.0", mechanism="39.matmul_range", inspect_build=True)
        self.assertEqual(result["build_inputs"]["status"], "INSPECTION_REFUSED")
        self.assertEqual(result["build_inputs"]["reason"], "TOOLKIT_IDENTITY_CHANGED_DURING_INSPECTION")
