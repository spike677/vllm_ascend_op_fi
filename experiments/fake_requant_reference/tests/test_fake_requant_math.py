from __future__ import annotations

import importlib.util
import math
import sys
import unittest
from pathlib import Path


SOURCE = Path(__file__).resolve().parents[1] / "source" / "fake_requant_reference_extension.py"
SPEC = importlib.util.spec_from_file_location("fake_requant_reference_extension", SOURCE)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class FakeRequantMathTests(unittest.TestCase):
    def test_round_and_dequant_signed_int8(self) -> None:
        result = MODULE.reference_requant_scalars(
            [-0.04, -0.01, 0.01, 0.04], base_scale=0.02
        )
        self.assertEqual((-0.04, 0.0, 0.0, 0.04), result.values)
        self.assertEqual(2, result.changed_elements)
        self.assertEqual(0, result.clip_count)

    def test_saturation_reports_clip_count(self) -> None:
        result = MODULE.reference_requant_scalars([2.0, -2.0, 0.01], base_scale=0.02, scale_multiplier=0.25)
        self.assertEqual((0.635, -0.64, 0.01), result.values)
        self.assertEqual(2, result.clip_count)
        self.assertEqual(2, result.changed_elements)

    def test_squeeze_increases_bucket_collapse(self) -> None:
        baseline = MODULE.reference_requant_scalars([0.03, 0.06], base_scale=0.02, scale_multiplier=1.0)
        squeezed = MODULE.reference_requant_scalars([0.03, 0.06], base_scale=0.02, scale_multiplier=4.0)
        self.assertEqual((0.04, 0.06), baseline.values)
        self.assertEqual((0.0, 0.08), squeezed.values)
        self.assertGreater(squeezed.mean_abs_error, baseline.mean_abs_error)

    def test_invalid_scale_and_nonfinite_input_fail_closed(self) -> None:
        with self.assertRaises(MODULE.ReferenceConfigurationError):
            MODULE.reference_requant_scalars([1.0], base_scale=0.0)
        with self.assertRaises(MODULE.ReferenceTransformError):
            MODULE.reference_requant_scalars([math.inf], base_scale=0.02)
        with self.assertRaises(MODULE.ReferenceTransformError):
            MODULE.reference_requant_scalars([], base_scale=0.02)

    def test_arm_configuration_is_explicit_and_rejects_ambiguous_scale_metadata(self) -> None:
        extension = MODULE.FakeRequantReferenceExtension()
        config = extension._make_config(
            mode=MODULE.IDENTITY,
            module_regex=r"(?:^|\.)mlp\.down_proj$",
            layer_filter=[0, 8],
            rank_filter=[0],
            tensor_selector="first_float",
            scale_provider="tensor_absmax",
            scale_multiplier=1.0,
            max_hits=1,
            interval=1,
            tensor_index=None,
            runtime_scale_attr=None,
            fixed_scale=None,
            request_generation=11,
            eps=1e-12,
        )
        self.assertEqual((0, 8), config.layer_filter)
        self.assertEqual((0,), config.rank_filter)
        self.assertEqual(11, config.request_generation)
        with self.assertRaises(MODULE.ReferenceConfigurationError):
            extension._make_config(
                mode=MODULE.REFERENCE_REQUANT,
                module_regex=".*",
                layer_filter=None,
                rank_filter=None,
                tensor_selector="first_float",
                scale_provider="module_runtime_scale",
                scale_multiplier=1.0,
                max_hits=1,
                interval=1,
                tensor_index=None,
                runtime_scale_attr=None,
                fixed_scale=None,
                request_generation=None,
                eps=1e-12,
            )


if __name__ == "__main__":
    unittest.main()
