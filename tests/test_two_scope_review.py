"""The offline summary must reject corrupted raw observations, regardless of PASS."""
import copy
import unittest
from experiments.worker_core_v11_20261118.review_two_scope_requests import review


def fixture(mechanism):
    worker=dict(model_instance_id='model',worker_start_identity='start',process_start_identity='proc',pid=1,rank=0,vllm_ascend_version='0.18.0')
    scopes=[]
    for index in range(2):
        records=[]
        for phase in ('normal','fault','off','health'):
            rid=str(index)+phase
            value='different' if phase=='fault' else 'normal'
            row=dict(phase=phase,request_id=rid,input_sha256='input',output=dict(hidden=value,logits=value))
            if mechanism=='18.api_half':
                event=dict(applied=phase=='fault',gain=.5,actual='actual',path='model.norm',original_execution_verified=True,return_readback_exact=True,output_changed=phase=='fault',consumers=['qkv'],effect=dict(values=dict(changed_elements=4,before_nonfinite=0,after_nonfinite=0,max_abs_diff_finite=1.)))
                row['mechanism']=dict(events=[event]) if phase in ('normal','fault') else None
            else:
                lease=dict(closed=True,first_failure=None,events=[dict(operation=k,return_code=0,returned=True,error=None) for k in ('open','apply','restore','release','live')],live_readbacks=[dict(native_live=False)])
                row.update(hidden_logits_sha256=value,report=dict(applied=int(phase=='fault'),recovered_storage_exact=True,consumers=['swiglu'],effect=dict(changed_only_selected=True,changed_elements=1,finite_to_positive_inf=1),native_lease_receipt=lease) if phase!='health' else None)
            records.append(dict(observation=row,completed=True,mapping=dict(internal_request_id=rid),output=dict(tokens=[1],text='one')))
        scopes.append(dict(worker=worker,owner='owner'+str(index),requests=records,resource_cleanup=dict(owned_hooks=0,inflight=0,leases=0,native_leases=0),ownership_retained=False))
    return dict(status='PASS',source=dict(manifest_sha256='source'),scopes=scopes)


class TwoScopeReviewTests(unittest.TestCase):
    def test_independent_raw_positive_both(self):
        for mechanism in ('18.api_half','39.output_proxy'):
            d=fixture(mechanism);d['status']='UNTRUSTED'
            self.assertEqual(review(d,mechanism)['scope_count'],2)

    def test_rejects_same_owner_or_different_worker(self):
        for bad in ('owner','worker'):
            d=fixture('18.api_half')
            d['scopes'][1][bad]=d['scopes'][0]['owner'] if bad=='owner' else dict(d['scopes'][1]['worker'],pid=2)
            with self.assertRaises(ValueError):review(d,'18.api_half')

    def test_rejects_missing_input_or_wrong_request_mapping(self):
        for field in ('input','mapping'):
            d=fixture('18.api_half');r=d['scopes'][0]['requests'][1]
            if field=='input':r['observation']['input_sha256']='changed'
            else:r['mapping']['internal_request_id']='other'
            with self.assertRaises(ValueError):review(d,'18.api_half')

    def test_rejects_hidden_recovery_failure(self):
        d=fixture('18.api_half');d['scopes'][1]['requests'][2]['observation']['output']['hidden']='fault'
        with self.assertRaises(ValueError):review(d,'18.api_half')

    def test_rejects_no_propagation(self):
        d=fixture('18.api_half');d['scopes'][0]['requests'][1]['observation']['output']=dict(hidden='normal',logits='normal')
        with self.assertRaises(ValueError):review(d,'18.api_half')

    def test_rejects_39_unclosed_native_lease(self):
        for field in ('native_live','return_code'):
            d=fixture('39.output_proxy');lease=d['scopes'][0]['requests'][1]['observation']['report']['native_lease_receipt']
            if field=='native_live':lease['live_readbacks'][-1]['native_live']=True
            else:lease['events'][2]['return_code']=1
            with self.assertRaises(ValueError):review(d,'39.output_proxy')

    def test_rejects_resource_or_owner_leak(self):
        for field in ('resource','owner'):
            d=fixture('39.output_proxy')
            if field=='resource':d['scopes'][1]['resource_cleanup']['native_leases']=1
            else:d['scopes'][1]['ownership_retained']=True
            with self.assertRaises(ValueError):review(d,'39.output_proxy')

if __name__=='__main__':unittest.main()
