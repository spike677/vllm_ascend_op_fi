import unittest
from fault_injection.rc2_profile import TEST_CANDIDATES,prepare_binding,RC_ID
from fault_injection.delivery_scope import describe_delivery
from demoncat.runtime.vllm_ascend.worker_extension.config import FaultConfig
from demoncat.runtime.vllm_ascend.worker_extension.ordinary_admission import ORDINARY

class Rc2HandoffTests(unittest.TestCase):
    def test_exact_implemented_subset_and_no_declaration_authority(self):
        self.assertEqual(TEST_CANDIDATES,{'18.api_half','39.output_proxy','8.restore_offset',
            '17.spatial_padding','22.wrong_row','quant.batch_shared_scale'})
        self.assertTrue(TEST_CANDIDATES <= ORDINARY)
        self.assertNotIn('41.tail_column',TEST_CANDIDATES)
        for name in TEST_CANDIDATES:
            r=describe_delivery(name);self.assertEqual(r['rc_id'],RC_ID)
            self.assertTrue(r['rc2_test_candidate']);self.assertFalse(r['rc2_candidate_grants_live_admission'])
    def test_callback_does_not_admit_unimplemented_or_research_variant(self):
        calls=[];bind=prepare_binding(lambda i,c:calls.append(c.mechanism))
        for name in ('20.buffer_substitute','26.residual_stomp','26.fia_destination','33.paged_decode','41.tail_column','28.fia.denominator_replay','15.alias_detach','25.reduce_scatter'):
            n,v=name.split('.',1)
            with self.assertRaisesRegex(RuntimeError,'RC2_NOT_IN_READY_TEST_SCOPE'):bind({},FaultConfig(int(n),v))
        self.assertEqual(calls,[])
    def test_deployment_gate_error_is_not_swallowed(self):
        def refuse(*_):raise RuntimeError('ORDINARY_ROUTE_NOT_ACCEPTED_FOR_DEPLOYMENT')
        with self.assertRaisesRegex(RuntimeError,'ORDINARY_ROUTE_NOT_ACCEPTED'):prepare_binding(refuse)({},FaultConfig(8,'restore_offset'))

if __name__=='__main__':unittest.main()
