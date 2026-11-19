"""RC1 scope decisions use one source; no native/Worker/inference execution."""
import unittest
from fault_injection.catalog import CAPABILITIES
from fault_injection.delivery_scope import describe_delivery
from fault_injection.rc1_profile import TEST_CANDIDATES, prepare_binding
from demoncat.runtime.vllm_ascend.worker_extension.config import FaultConfig
from demoncat.runtime.vllm_ascend.worker_extension.registry import worker_core_manifest


class Rc1HandoffTests(unittest.TestCase):
    def test_catalog_and_inert_binding_use_same_allow_list(self):
        calls = []
        def deployment(identity, config):
            calls.append((identity, config.mechanism))
            return {'prepare': {}, 'register': {}}
        bind = prepare_binding(deployment)
        for item in CAPABILITIES:
            prefix, variant = item.name.split('.', 1)
            config = FaultConfig(prefix if prefix == 'quant' else int(prefix), variant)
            scope = describe_delivery(item.name)
            self.assertEqual(scope['rc1_test_candidate'], item.name in TEST_CANDIDATES)
            self.assertFalse(scope['rc1_candidate_grants_live_admission'])
            before = len(calls)
            if scope['rc1_test_candidate']:
                self.assertEqual(bind({'actual': 'identity'}, config), {'prepare': {}, 'register': {}})
                self.assertEqual(len(calls), before + 1)
            else:
                with self.assertRaisesRegex(RuntimeError, 'RC1_NOT_IN_READY_TEST_SCOPE'):
                    bind({'actual': 'identity'}, config)
                self.assertEqual(len(calls), before)

    def test_live_deployment_refusal_not_replaced_with_static_acceptance(self):
        def deployment(identity, config):
            raise RuntimeError('ACTUAL_NATIVE_BUILD_MISMATCH')
        with self.assertRaisesRegex(RuntimeError, '^ACTUAL_NATIVE_BUILD_MISMATCH$'):
            prepare_binding(deployment)({}, FaultConfig(39, 'output_proxy'))
        with self.assertRaisesRegex(RuntimeError, 'RC1_NOT_IN_READY_TEST_SCOPE'):
            prepare_binding(deployment)({}, FaultConfig(39, 'output_proxy', '2'))

    def test_manifest_never_grants_admission(self):
        manifest = worker_core_manifest()
        self.assertFalse(manifest['qualification_granted'])
        selected = {r['mechanism'] for r in manifest['mechanisms']
                    if r['current_delivery']['rc1_test_candidate']}
        self.assertEqual(selected, TEST_CANDIDATES)
        for row in manifest['mechanisms']:
            self.assertFalse(row['qualification_granted'])
            self.assertEqual(row['current_delivery']['live_deployment_status'], 'NOT_EVALUATED')


if __name__ == '__main__':
    unittest.main()
