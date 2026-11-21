import unittest
from fault_injection.rc3_preview_profile import TEST_CANDIDATES,prepare_binding
from demoncat.runtime.vllm_ascend.worker_extension.config import FaultConfig


class PreviewTests(unittest.TestCase):
    def test_allow_list_is_not_a_live_grant(self):
        self.assertEqual(TEST_CANDIDATES,{'28.fia.denominator_replay','quant.batch_shared_scale'})
        group={'injection_policy':'ALL_RANK_EVERY_ELIGIBLE_CALL'}
        def refuse(*args):raise RuntimeError('SOURCE_OR_DEPLOYMENT_REJECTED')
        with self.assertRaisesRegex(RuntimeError,'SOURCE_OR_DEPLOYMENT'):
            prepare_binding(refuse)({},FaultConfig(28,'fia.denominator_replay'),group)
        calls=[];bind=prepare_binding(lambda *a:calls.append(a))
        with self.assertRaisesRegex(RuntimeError,'OUTSIDE_DELIVERY_SCOPE'):
            bind({},FaultConfig(18,'api_half'),group)
        with self.assertRaisesRegex(RuntimeError,'CONTINUOUS_GROUP'):
            bind({},FaultConfig(28,'fia.denominator_replay'),{})
        self.assertEqual(calls,[])
