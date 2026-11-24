import copy,json,unittest
from tools.product_effect_probe import canonical,sha
from tools.product_effect_noise import run,summarize,verify
from test_dsv4_noise import combo,data,phases
from test_product_effect_probe import identity


def fixture():
    c=dict(identity(),**combo());workers=[dict(rank=r,pid=100+r) for r in range(8)];scopes=[]
    for i in range(2):
        owner='owner'+str(i);p=phases()
        for name,row in p.items():
            row.update(owner=owner,worker_identity_sha256=sha(canonical(workers)),request_id=f'{i}-{name}',evidence_paths=['actual.capture'])
        counts=dict(observed=4,eligible=4,attempted=4,applied=4,changed=3,consumed=4,restored=4,failed=0)
        ranks=[dict(rank=r,owner=owner,provider_owner=f'{owner}-r{r}',request_ids=[f'{i}-fault'],evidence_paths=['actual.provider'],
                    counts=copy.deepcopy(counts),changed_elements=3,consumer=True,bytes_restored=True,resources_zero=True) for r in range(8)]
        scopes.append(dict(owner=owner,worker_identity=workers,phases=p,evidence=dict(ranks=ranks)))
    return dict(schema='dcat.effect.noise.raw.v1',rule='demoncat.dsv4_0731.noise.v1',identity=c,scopes=scopes)


class NoiseProductTests(unittest.TestCase):
    def page(self,raw=None):
        raw=raw or fixture();cal=data()
        return summarize(raw,canonical(raw),cal,canonical(dict(combination=raw['identity'],calibration=cal)))
    def test_noise_is_not_exact_and_does_not_grant(self):
        p=self.page();self.assertTrue(p['evidence_passed']);self.assertFalse(p['registration_ready'])
        self.assertTrue(verify(p)['integrity_valid']);self.assertLessEqual(len(canonical(p))+1,4096)
        self.assertEqual(canonical(p),canonical(self.page()))
    def test_bytes_and_resources_never_relaxed(self):
        for field in ('bytes_restored','resources_zero','consumer'):
            d=fixture();d['scopes'][0]['evidence']['ranks'][0][field]=False
            with self.assertRaisesRegex(ValueError,'STRICT'):self.page(d)
    def test_process_owner_and_fault_request_must_match(self):
        d=fixture();d['scopes'][1]['phases']['off']['owner']='wrong'
        with self.assertRaisesRegex(ValueError,'OWNER_WORKER'):self.page(d)
        d=fixture();d['scopes'][0]['evidence']['ranks'][0]['request_ids']=['wrong']
        with self.assertRaisesRegex(ValueError,'ASSOCIATION'):self.page(d)
        d=fixture()
        for p in d['scopes'][1]['phases'].values():p['process_identity']='different'
        with self.assertRaisesRegex(ValueError,'PROCESSES_CHANGED'):self.page(d)
    def test_token_and_bound_failures(self):
        d=fixture();d['scopes'][0]['phases']['off']['tokens'][0]=99
        with self.assertRaisesRegex(ValueError,'OFF_HEALTH'):self.page(d)
        d=fixture();d['scopes'][0]['phases']['health']['ranks']['0'][0]['hidden'][0]=20.
        with self.assertRaisesRegex(ValueError,'OFF_HEALTH'):self.page(d)
    def test_missing_registration_identity_remains_not_ready(self):
        d=fixture();d['identity']['hardware_device_name']=None
        p=self.page(d);self.assertFalse(p['evidence_passed']);self.assertFalse(verify(p)['registration_ready'])
    def test_forged_counter_conclusion_is_rejected(self):
        p=self.page();p['scopes'][0]['ranks'][0][8]=1
        p.pop('extract_sha256');p['extract_sha256']=sha(canonical(p))
        with self.assertRaisesRegex(ValueError,'COUNTER'):verify(p)
    def test_qwen_cannot_select_noise_mode(self):
        d=fixture();d['identity']['model']='Qwen3.8'
        with self.assertRaisesRegex(ValueError,'COMBINATION'):self.page(d)
    def test_ep_rank_without_real_call_does_not_invent_consumer(self):
        d=fixture();r=d['scopes'][0]['evidence']['ranks'][7]
        r['counts']=dict.fromkeys(r['counts'],0);r['changed_elements']=0;r['consumer']=None
        with self.assertRaisesRegex(ValueError,'ZERO_CALL'):self.page(d)
        r['not_applicable_reason']='NO_ACTUAL_LOCAL_EXPERT_CALL'
        p=self.page(d);self.assertTrue(verify(p)['evidence_passed'])
        self.assertEqual(p['scopes'][0]['inactive_ranks'],[[7,'NO_ACTUAL_LOCAL_EXPERT_CALL']])
    def test_enable_failure_still_disables(self):
        trace=[]
        class Session:
            owner='x';worker_identity=[]
            def attach(self):trace.append('attach')
            def enable(self):trace.append('enable');raise RuntimeError('failure')
            def disable(self):trace.append('disable')
        class Adapter:
            def identity(self):return combo()
            def new_session(self):return Session()
            def capture_noise(self,*args):return phases()['normal']
        with self.assertRaisesRegex(RuntimeError,'failure'):run(Adapter(),'prompt',combo(),data())
        self.assertEqual(trace,['attach','enable','disable'])


if __name__=='__main__':unittest.main()
