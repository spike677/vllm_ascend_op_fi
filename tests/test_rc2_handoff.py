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

class Rc2SupplementTests(unittest.TestCase):
    def test_documented_bindings_exist_with_required_parameters(self):
        import inspect
        from demoncat.runtime.vllm_ascend.worker_extension import serving_tensor_binding as t
        from demoncat.runtime.vllm_ascend.worker_extension import serving39_output as o
        from pathlib import Path
        doc=(Path(__file__).resolve().parents[1]/'docs/worker_core_v1/deployment_acceptance_rc2.md').read_text(encoding='utf-8')
        for func in (t.install_rms18_task_binding,t.install_rope22_task_binding):
            self.assertIn(func.__name__,doc)
            sig=inspect.signature(func)
            required={n for n,p in sig.parameters.items() if p.default is inspect.Parameter.empty}
            self.assertEqual(required,{'worker','prompt_token_ids','weight_manifest','workload_sha256','environment_sha256'})
        sig=inspect.signature(o.install_fault39_output_serving_binding)
        self.assertEqual({n for n,p in sig.parameters.items() if p.default is inspect.Parameter.empty}, {'worker','transport','prompt_token_ids'})
        self.assertIn('runtime_only',sig.parameters)
        for func in (t.remove_tensor_serving_binding,o.remove_fault39_output_serving_binding):
            self.assertTrue(callable(func));self.assertIn(func.__name__,doc)

    def test_permit_sha_is_actual_frozen_source_review_not_release_manifest(self):
        import hashlib
        from pathlib import Path
        root=Path(__file__).resolve().parents[1]
        folder=root/'releases/demoncatpfi-yellow-rc2-20261119'
        expected='e1622a4bdc91f31576342e9d66e02f75e092f41738bb8fe8b8413fe553aecdea'
        self.assertEqual(hashlib.sha256((folder/'source-review.json').read_bytes()).hexdigest(),expected)
        self.assertNotEqual(hashlib.sha256((folder/'release-manifest.json').read_bytes()).hexdigest(),expected)
        for name in ('docs/worker_core_v1/deployment_acceptance_rc2.md','releases/RC2-HANDOFF-20261119.md'):
            self.assertIn(expected,(root/name).read_text(encoding='utf-8'))

if __name__=='__main__':unittest.main()
