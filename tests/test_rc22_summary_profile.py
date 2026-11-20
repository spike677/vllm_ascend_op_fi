import unittest
from copy import deepcopy
from demoncat.runtime.vllm_ascend.worker_extension import deployment_summary as summary
from tests.worker_extension import test_deployment_authorization as fixture
from fault_injection.rc22_profile import TEST_CANDIDATES,prepare_binding
from demoncat.runtime.vllm_ascend.worker_extension.config import FaultConfig

class Rc22Tests(unittest.TestCase):
    def receipt(self,owner):
        receipt=fixture.SummaryTests().receipt(owner)
        receipt['mechanism']='22.wrong_row'
        ev=receipt['history'][0]['response']['provider_state']['evidence']
        identity={'shape':[30,2048],'dtype':'torch.bfloat16','sha256':'a'*64}
        for row in ev['actual_requests']:
            changed=row['phase']=='fault'
            event={'applied':changed,'correct_source_row':29,'replacement_source_row':0,
                'actual_qk_before':[identity,identity],
                'actual_qk_after':[dict(identity,sha256=('b'*64 if changed else 'a'*64)),identity],
                'same_complete_inputs_as_healthy':True,'changed_from_healthy':changed,
                'independent_final_reader':{'independent_position_transform':'last_from_first' if changed else 'identity',
                    'complete_output':[dict(identity,sha256=('b'*64 if changed else 'a'*64)),identity],
                    'non_target_qk_before':[identity,identity],'cache_sha256':'c'*64,
                    'root_positions_unchanged':True,'complete_cache_unchanged':True}}
            row['mechanism']={'events':[event]}
        return receipt
    def test_rope_rows_and_full_qk_hashes_survive_bounded_summary(self):
        result=summary.summarize_deployment([self.receipt('a'),self.receipt('b')])
        self.assertTrue(result['review_ready'])
        normal,fault,off,health=result['scopes'][0]['requests']
        events=lambda row:row['mechanism']['events'][0]
        self.assertEqual(events(fault)['correct_source_row'],29)
        self.assertEqual(events(fault)['replacement_source_row'],0)
        self.assertEqual(events(normal)['actual_qk_before'],events(fault)['actual_qk_before'])
        self.assertNotEqual(events(normal)['actual_qk_after'],events(fault)['actual_qk_after'])
        self.assertEqual(events(normal)['actual_qk_after'],events(off)['actual_qk_after'])
        self.assertEqual(events(normal)['actual_qk_after'],events(health)['actual_qk_after'])
        self.assertEqual(events(fault)['independent_final_reader']['complete_output'][0]['sha256'],'b'*64)
        self.assertLess(len(summary._bytes(result)),summary.MAX_SUMMARY_BYTES)
    def test_missing_legacy_field_not_invented_and_raw_tensor_omitted(self):
        r=self.receipt('a');event=r['history'][0]['response']['provider_state']['evidence']['actual_requests'][1]['mechanism']['events'][0]
        del event['actual_qk_after'];event['raw_tensor']=[1]*10000;event['cache_unchanged']=None
        out=summary.summarize_scope(r)['requests'][1]['mechanism']['events'][0]
        self.assertNotIn('actual_qk_after',out);self.assertNotIn('raw_tensor',out)
        self.assertIsNone(out['cache_unchanged'])
    def test_new_profile_adds_41_but_does_not_admit_research(self):
        self.assertEqual(len(TEST_CANDIDATES),7)
        seen=[];bind=prepare_binding(lambda identity,config:seen.append(config.mechanism))
        bind({},FaultConfig(41,'tail_column'));self.assertEqual(seen,['41.tail_column'])
        with self.assertRaisesRegex(RuntimeError,'RC22_NOT_IN_READY_TEST_SCOPE'):bind({},FaultConfig(28,'fia.denominator_replay'))
        def denied(*args):raise RuntimeError('DEPLOYMENT_REFUSED')
        with self.assertRaisesRegex(RuntimeError,'DEPLOYMENT_REFUSED'):prepare_binding(denied)({},FaultConfig(41,'tail_column'))

if __name__=='__main__':unittest.main()
