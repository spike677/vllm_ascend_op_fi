import copy
import unittest
from tools.product_effect_probe import run, summarize, verify, canonical, sha, compare


def response(value=0):
    return dict(choices=[dict(index=0, text='a', logprobs=dict(tokens=['a'], token_logprobs=[-1.0],
        top_logprobs=[{str(i): -float(i+1)+value for i in range(5)}]))])


def identity():
    from tools.extract_registration_record import DEPLOYMENT
    d = {k: 'observed' for k in DEPLOYMENT}
    d.update(tp=2, max_model_len=1024, max_num_batched_tokens=1024, max_num_seqs=1,
             quant_model_description_sha256='a'*64, safetensors_index_sha256={'model.safetensors.index.json':'b'*64}, image_digest=None)
    return d


class Session:
    worker_identity = [{'rank':0,'pid':1}, {'rank':1,'pid':2}]
    def __init__(self, adapter, owner): self.adapter=adapter; self.owner=owner
    def attach(self): return {'status':'attached'}
    def enable(self): self.adapter.active=True; return {'status':'enabled'}
    def disable(self): self.adapter.active=False; return {'status':'disabled'}
    def status(self): return {'status':'observed'}
    def detach(self): return {'status':'detached'}
    def evidence(self):
        return dict(logits_changed=None, ranks=[dict(rank=i,owner=self.owner,provider_owner=self.owner+'-'+str(i),request_ids=['actual-id'],
            evidence_paths=['observed.requests.0'],counts=dict(observed=2,eligible=2,attempted=2,applied=2,
            changed=1,consumed=2,restored=2,failed=0),changed_elements=3,consumer=True,
            bytes_restored=True,resources_zero=True,canary_exact=True) for i in range(2)])


class Adapter:
    def __init__(self): self.active=False;self.n=0
    def identity(self): return identity()
    def infer(self, prompts, sampling): return response(.1 if self.active else 0)
    def new_session(self): self.n+=1;return Session(self,'owner-'+str(self.n))


class EffectTests(unittest.TestCase):
    def setUp(self): self.raw=run(Adapter(), ['prompt'])
    def page(self, raw=None):
        raw=raw or self.raw;return summarize(raw,canonical(raw))
    def test_two_owners_repeat_byte_exact_and_size(self):
        p=self.page();self.assertEqual(canonical(p),canonical(self.page()))
        self.assertLessEqual(len(canonical(p))+1,4096);self.assertTrue(verify(p)['registration_ready'])
        self.assertEqual(p['scopes'][0]['effect'],[False,True,False,None])
    def test_same_owner_and_worker_restart_rejected(self):
        for key,value in [('owner',self.raw['scopes'][0]['owner']),('worker_identity',[])]:
            raw=copy.deepcopy(self.raw);raw['scopes'][1][key]=value
            with self.assertRaises(ValueError):self.page(raw)
    def test_failed_eligible_and_missing_consumption_rejected(self):
        for key,value in [('failed',1),('consumed',0),('changed',3)]:
            raw=copy.deepcopy(self.raw);raw['scopes'][0]['evidence']['ranks'][0]['counts'][key]=value
            with self.assertRaises(ValueError):self.page(raw)
    def test_recovery_missing_or_false_not_coerced(self):
        for key in ['consumer','bytes_restored','resources_zero','canary_exact']:
            raw=copy.deepcopy(self.raw);raw['scopes'][0]['evidence']['ranks'][0][key]=None
            with self.assertRaises(ValueError):self.page(raw)
    def test_off_mismatch(self):
        self.raw['scopes'][0]['outputs']['off']=response(.2)
        with self.assertRaisesRegex(ValueError,'OFF_API_NOT_EXACT'):self.page()
    def test_hash_tamper(self):
        p=self.page();p['registration_ready']=False
        with self.assertRaisesRegex(ValueError,'EXTRACT_SHA_MISMATCH'):verify(p)
    def test_missing_identity_stays_not_ready(self):
        self.raw['identity']['cann']=None;p=self.page()
        self.assertFalse(verify(p)['registration_ready'])
        p['registration_ready']=True;p.pop('extract_sha256');p['extract_sha256']=sha(canonical(p))
        with self.assertRaisesRegex(ValueError,'IDENTITY_CONCLUSION'):verify(p)
    def test_no_top5_is_not_full_logits_proof(self):
        r=response();r['choices'][0]['logprobs']['top_logprobs'][0]={'a':-1}
        with self.assertRaisesRegex(ValueError,'TOP5'):compare(r,r)
    def test_overflow_fails_without_truncation(self):
        self.raw['identity']['long']='x'*4096
        with self.assertRaisesRegex(ValueError,'EXCEEDS'):self.page()
    def test_sampling_fixed_seed(self):
        class A(Adapter):
            def infer(self,prompts,sampling):
                assert sampling['seed']==42 and sampling['temperature']==.5 and sampling['logprobs']==5
                return response(.1 if self.active else 0)
        self.assertEqual(len(run(A(),['x'],temperature=.5)['scopes']),2)
    def test_error_cleanup_checkpoint(self):
        a=Adapter();a.infer=lambda p,s: (_ for _ in ()).throw(RuntimeError('kernel error'))
        checkpoints=[]
        with self.assertRaisesRegex(RuntimeError,'kernel error'):run(a,['x'],checkpoint=lambda d:checkpoints.append(copy.deepcopy(d)))
        self.assertFalse(a.active);self.assertEqual(checkpoints[-1]['control_sequence'][-1]['action'],'disable')
    def test_negative_contract_and_duplicate_owner_guard(self):
        from tools.product_effect_probe import run_negatives
        class A:
            def check_negative(self,name):
                return dict(name=name,engine_alive=True,exit_code=4,code='REFUSED',resources_zero_after_cleanup=True,
                    positive_control=True,original_session_unchanged=True,original_receipt_sha256='a'*64)
        self.assertEqual(len(run_negatives(A())['cases']),4)
        class Bad(A):
            def check_negative(self,name):return dict(super().check_negative(name),original_session_unchanged=False)
        with self.assertRaisesRegex(ValueError,'DUPLICATE_ENABLE'):run_negatives(Bad())
    def test_raw_api_reply_changes_are_not_hidden_by_text_equality(self):
        r=response();r['choices'][0]['logprobs']['token_logprobs'][0]=-.5
        self.assertFalse(compare(response(),r)['exact'])
