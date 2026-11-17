"""CPU admission contract tests; no native library or model is loaded."""
from contextlib import ExitStack
from copy import deepcopy
import json
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from fault_injection.core import UnsupportedCall
from fault_injection.deployments import matmul


class _Status:
    def __init__(self, row):
        self.row = row

    def __call__(self):
        return json.dumps(self.row).encode()


class PtaAdmissionTests(unittest.TestCase):
    def setUp(self):
        self.runtime = matmul.CtypesMatmulRuntime.__new__(matmul.CtypesMatmulRuntime)
        self.runtime._pta_configuration = '1'
        self.runtime.build_profile = 'a' * 64
        self.resolution = {'configuration': {'ASCEND_CUSTOM_OPP_PATH': None},
                           'loaded_torch_npu_images': [{'file_sha256': 'c' * 64}]}
        self.provider = dict(provider='/native/libopapi_math.so', provider_sha256='b' * 64,
                             address='0x1000')
        self.rows = {name: dict(schema='dcat20.pta_gate.v1', entry=name,
                    profile_sha256='a' * 64, requested=True, config_valid=True,
                    identity_valid=True, profile_match=True, original_available=True,
                    original_math_identity_valid=True, ready=True, refused=False,
                    resolver_wheel_sha256='c' * 64,
                    qmm_bypass=0, weight_nz_bypass=0, original_calls=0)
                     for name in ('libopapi_math.so', 'libopapi.so', 'libopapi_nn.so')}
        stack = self.enterContext(ExitStack())
        stack.enter_context(patch.dict(matmul.os.environ, DEMONCAT_F20_REQUIRE_TYPED_GWS='1'))
        stack.enter_context(patch.object(matmul.os, 'RTLD_NOLOAD', 4, create=True))
        stack.enter_context(patch.object(matmul.os, 'RTLD_NOW', 2, create=True))
        self.inspect = stack.enter_context(patch.object(matmul, 'inspect_loaded_pta_resolution',
                                                       side_effect=lambda: deepcopy(self.resolution)))
        self.validate = stack.enter_context(patch.object(matmul, 'validate_loaded_pta_resolution',
                                                        side_effect=lambda _: deepcopy(self.provider)))
        stack.enter_context(patch.object(matmul.ctypes, 'CDLL', side_effect=lambda name, **_: SimpleNamespace(
            dcat20_pta_gate_status_v1=_Status(self.rows[name]))))

    def test_all_three_identities_required_without_claiming_hit(self):
        result = self.runtime._verify_pta_gate()
        self.assertEqual(set(result), set(self.rows))
        self.assertEqual(result['libopapi_math.so']['weight_nz_bypass'], 0)
        self.assertEqual(self.runtime.pta_resolver_status, self.resolution)
        self.assertEqual(self.runtime._pta_provider_identity, self.provider)

    def test_two_ready_proxies_do_not_override_failed_actual_route(self):
        self.validate.side_effect = ValueError('MATH_CAN_USE_PROVIDER_IS_NOT_OWNED_GATE')
        with self.assertRaisesRegex(UnsupportedCall, 'actual PTA resolver route is unproven'):
            self.runtime._verify_pta_gate()

    def test_original_math_identity_failure_refuses(self):
        self.rows['libopapi_math.so']['original_math_identity_valid'] = False
        with self.assertRaisesRegex(UnsupportedCall, 'original math provider identity'):
            self.runtime._verify_pta_gate()

    def test_status_from_wrong_entry_refuses(self):
        self.rows['libopapi_math.so']['entry'] = 'libopapi_nn.so'
        with self.assertRaisesRegex(UnsupportedCall, 'PTA control refused'):
            self.runtime._verify_pta_gate()

    def test_resolver_configuration_drift_refuses(self):
        self.runtime._verify_pta_gate()
        self.resolution['configuration']['ASCEND_CUSTOM_OPP_PATH'] = '/foreign'
        with self.assertRaisesRegex(UnsupportedCall, 'resolver configuration changed'):
            self.runtime._verify_pta_gate()

    def test_loaded_provider_drift_refuses(self):
        self.runtime._verify_pta_gate()
        self.provider['address'] = '0x2000'
        with self.assertRaisesRegex(UnsupportedCall, 'function provider changed'):
            self.runtime._verify_pta_gate()

    def test_unrequested_mode_does_not_probe_or_rewrite_route(self):
        self.runtime._pta_configuration = '0'
        with patch.dict(matmul.os.environ, DEMONCAT_F20_REQUIRE_TYPED_GWS='0'):
            self.assertEqual(self.runtime._verify_pta_gate(), {})
        self.inspect.assert_not_called()

    def test_missing_or_different_actual_wheel_refuses(self):
        self.resolution['loaded_torch_npu_images'] = []
        with self.assertRaisesRegex(UnsupportedCall, 'actual profiled torch_npu'):
            self.runtime._verify_pta_gate()
        self.resolution['loaded_torch_npu_images'] = [{'file_sha256': 'd' * 64}]
        with self.assertRaisesRegex(UnsupportedCall, 'PTA control refused'):
            self.runtime._verify_pta_gate()


if __name__ == '__main__':
    unittest.main()
