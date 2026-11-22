from types import SimpleNamespace as NS
from unittest.mock import Mock
import unittest
from demoncat.runtime.vllm_ascend.worker_extension.product_observer import AutomaticRequests

class Runner:
    def execute_model(self,scheduler,intermediate=None):return 'execute'
    def sample_tokens(self):return 'sample'

class Session:
    def __init__(self,binding,ids,inject=True,forced_skip=None):
        self.ids=ids;self.inside=0;self.samples=0;self.skip=forced_skip;self.inject=inject
        self.site=None;self.hooks=[];self.patches=[];self.range_facts={'max_tokens':4}
        self.finished=False;self.error=None
    def install(self,managed=False):assert managed
    def managed_execute(self,original,scheduler,intermediate):return original(scheduler,intermediate)
    def managed_sample(self,original,*args,**kw):self.samples+=1;return original(*args,**kw)
    def collect(self):
        assert self.finished
        return dict(request_id=self.ids[0],reasons=[self.skip] if self.skip else [],
            completed=True,bytes_restored=True,output=[{'step':i} for i in range(self.samples)],all_finite=True)

class AutomaticTests(unittest.TestCase):
    def make(self,canary=False):
        b=NS(worker=NS(model_runner=Runner()),check=Mock(),prompts=((1,2),),warmups=[])
        m=AutomaticRequests(b,canary=canary,factory=Session);m.install();return m,b
    def schedule(self,rid='real',new=True,tokens=(8,9),finished=()):
        return NS(num_scheduled_tokens={rid:2 if new else 1},scheduled_new_reqs=[NS(req_id=rid,prompt_token_ids=list(tokens))] if new else [],finished_req_ids=set(finished))
    def steps(self,m,rid='real',tokens=(8,9)):
        r=m.runner
        for i in range(4):r.execute_model(self.schedule(rid,new=i==0,tokens=tokens));r.sample_tokens()
    def test_automatic_all_arrivals_no_host_id_hooks(self):
        m,b=self.make()
        for i in range(5):self.steps(m,str(i))
        self.assertEqual([r['request_id'] for r in m.records],[str(i) for i in range(5)])
        self.assertTrue(all(r['completion_source']=='ACTUAL_SAMPLES_REACHED_REQUEST_MAX_TOKENS' for r in m.records))
        m.close();self.assertEqual(m.snapshot()['owned_hooks'],0)
    def test_canary_is_only_selected_by_deployed_tokens(self):
        m,b=self.make(canary=True);self.steps(m,'user')
        self.assertFalse(b.warmups)
        self.steps(m,'canary',(1,2));self.assertEqual(len(b.warmups),1)
        self.assertEqual(m.records[0]['request_id'],'canary');m.close()
    def test_early_eos_uses_scheduler_finished_not_host(self):
        m,b=self.make();m.runner.execute_model(self.schedule());m.runner.sample_tokens()
        self.assertIsNotNone(m.session)
        m.runner.execute_model(self.schedule('second',finished=['real']))
        self.assertEqual(m.records[0]['completion_source'],'SCHEDULER_FINISHED_REQUEST')
        m.runner.sample_tokens();m.close()
        self.assertFalse(m.records[-1]['completed'])
    def test_enable_mid_request_skips_until_new_request(self):
        m,b=self.make();m.runner.execute_model(self.schedule(new=False));m.runner.sample_tokens();m.close()
        self.assertEqual(m.records[0]['reasons'],['REQUEST_STARTED_BEFORE_ENABLE'])
    def test_no_silent_concurrent_batch_admission_or_owner_overwrite(self):
        m,b=self.make();s=self.schedule();s.num_scheduled_tokens['other']=1
        with self.assertRaisesRegex(RuntimeError,'SERIAL'):m.runner.execute_model(s)
        self.assertIsNotNone(m.error);m.close()
        m,b=self.make();m.runner.execute_model=lambda *a:None
        with self.assertRaisesRegex(RuntimeError,'OWNER_CHANGED'):m.close()
class ProductCoverageTests(unittest.TestCase):
    def make(self,visual):
        from demoncat.runtime.vllm_ascend.worker_extension.product_request import ProductRequest
        s=ProductRequest.__new__(ProductRequest)
        s.inside=0;s.finished=True;s.error=None;s.samples=4;s.visual=visual;s.skip=[]
        s.encoder_calls=1;s.embedding_calls=1;s.ids=['r'];s.binding=NS(identity={'rank':0})
        s.external_request_id=None;s.prompts={'r':(1,2)};s.range_facts={};s.inject=True
        e=dict(site='patch',eligible=True,applied=True,consumed=True,restored=True,changed=1,before_sha256='a',restored_sha256='a')
        s.site=NS(events=[e],pending=0,uncovered=[],forward_calls=[{'patch':1}]);s.close=Mock()
        return s
    def test_visual_counts_encoder_calls_not_text_decode_steps(self):
        s=self.make(True);r=s.collect()
        self.assertEqual(r['coverage']['eligible'],1);self.assertEqual(r['observed_model_steps'],4)
        s.close.assert_called_once()
    def test_text_missing_decode_coverage_is_still_refused(self):
        s=self.make(False)
        with self.assertRaisesRegex(RuntimeError,'FORWARD_COVERAGE'):s.collect()
        s.close.assert_not_called()

if __name__=='__main__':unittest.main()
