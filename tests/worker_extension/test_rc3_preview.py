import unittest
import tempfile
from pathlib import Path
from tools.build_rc3_preview import write_json_bytes
from fault_injection.rc3_preview_profile import TEST_CANDIDATES,prepare_binding
from demoncat.runtime.vllm_ascend.worker_extension.config import FaultConfig


class PreviewTests(unittest.TestCase):
    def test_host_follows_plan_and_refuses_replayed_or_incomplete_phases(self):
        from fault_injection.rc3_preview_profile import next_request_phase
        for phases in (['normal','fault','off','health'],['normal','fault','off','health','extra_health']):
            plan=dict(phases=[dict(phase=p) for p in phases],max_iterations=len(phases))
            done=[]
            for phase in phases+['complete']:
                reply=dict(status='QUERY_OK',response=dict(provider_state=dict(
                    request_plan=plan,next_phase=phase,complete=phase=='complete')))
                self.assertEqual(next_request_phase(reply,plan,done),phase)
                if phase!='complete':done.append(phase)
            reply['response']['provider_state']['complete']=False
            with self.assertRaisesRegex(RuntimeError,'INCOMPLETE_RECOVERY'):next_request_phase(reply,plan,done)
            reply['response']['provider_state']['next_phase']='fault'
            with self.assertRaisesRegex(RuntimeError,'DO_NOT_REPLAY'):next_request_phase(reply,plan,done)
    def test_fault25_remains_unsupported_in_public_and_research_routes(self):
        from types import SimpleNamespace
        from demoncat.runtime.vllm_ascend.worker_extension import registry
        for version in ('1','unknown'):
            with self.assertRaisesRegex(RuntimeError,'FAULT25_UNSUPPORTED_NO_SUBSTITUTE') as caught:
                registry.create_worker_provider('25.reduce_scatter',version,SimpleNamespace(),object(),{})
            self.assertFalse(caught.exception.prerequisites['can_prepare'])
            with self.assertRaisesRegex(RuntimeError,'FAULT25_UNSUPPORTED_NO_SUBSTITUTE'):
                registry.create_research_provider('25.reduce_scatter',version,SimpleNamespace(),object(),{})

    def test_manifest_bytes_do_not_depend_on_host_newline(self):
        with tempfile.TemporaryDirectory() as root:
            p=Path(root)/'source-review.json';write_json_bytes(p,{'review':'source'})
            self.assertEqual(p.read_bytes(),b'{\n  "review": "source"\n}\n')
    def test_allow_list_is_not_a_live_grant(self):
        self.assertEqual(TEST_CANDIDATES,{'28.fia.denominator_replay','quant.batch_shared_scale','39.output_proxy','18.api_half','22.wrong_row','39.matmul_range'})
        group={'injection_policy':'ALL_RANK_EVERY_ELIGIBLE_CALL'}
        def refuse(*args):raise RuntimeError('SOURCE_OR_DEPLOYMENT_REJECTED')
        with self.assertRaisesRegex(RuntimeError,'SOURCE_OR_DEPLOYMENT'):
            prepare_binding(refuse)({},FaultConfig(28,'fia.denominator_replay'),group)
        calls=[];bind=prepare_binding(lambda *a:calls.append(a))
        with self.assertRaisesRegex(RuntimeError,'OUTSIDE_DELIVERY_SCOPE'):
            bind({},FaultConfig(33,'paged_decode'),group)
        with self.assertRaisesRegex(RuntimeError,'CONTINUOUS_GROUP'):
            bind({},FaultConfig(28,'fia.denominator_replay'),{})
        self.assertEqual(calls,[])
