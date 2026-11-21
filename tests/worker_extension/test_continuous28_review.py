from copy import deepcopy
import unittest
from tools.review_continuous28_site import review


def fixture():
    scopes=[]
    for s in range(2):
        requests=[]
        for phase in ('normal','fault','off','health'):
            ranks=[]
            for rank in (0,1):
                events=[dict(site='layer',method=method,applied=True,consumed=True,restored=True,non_target_exact=True,changed=1,before_sha256='a'*64,restored_sha256='a'*64)
                        for method in ['_forward_prefill']+['_forward_decode']*3]
                outputs=[dict(hidden='f' if phase=='fault' else 'n',logits='f' if phase=='fault' else 'n') for _ in range(4)]
                ranks.append(dict(rank=rank,pid=rank+10,model_id=rank+20,site_count=1,events=events,
                    output=outputs,requests=[dict(request_ids=[phase]) for _ in range(4)]))
            requests.append(dict(phase=phase,ranks=ranks,installed=[dict(sites=['layer'])]*2,
                cleanup=[dict(resources=dict(leases=0,hooks=0,inflight=0))]*2,
                mapping=[dict(internal_request_id=phase)],tokens=[1 if phase=='fault' else 0],seconds=1.))
        scopes.append(dict(owner=str(s),requests=requests))
    return dict(status='TWO_SCOPES_TEST_FIXTURE',scopes=scopes)


class ReceiptReviewTests(unittest.TestCase):
    def test_positive_is_not_deployment_qualification(self):
        data=fixture();before=deepcopy(data);r=review(data)
        self.assertEqual(data,before);self.assertFalse(r['qualification_granted'])

    def test_rejects_missing_consumer_restore_mapping_and_off(self):
        for field in ('consumed','restored','non_target_exact'):
            data=fixture();data['scopes'][0]['requests'][1]['ranks'][1]['events'][0][field]=False
            with self.assertRaises(ValueError):review(data)
        data=fixture();data['scopes'][1]['requests'][0]['ranks'][0]['requests'][0]['request_ids']=['wrong']
        with self.assertRaisesRegex(ValueError,'REQUEST_MAP'):review(data)
        data=fixture();data['scopes'][0]['requests'][2]['ranks'][0]['output'][0]['hidden']='corrupt'
        with self.assertRaisesRegex(ValueError,'OFF_HEALTH'):review(data)

    def test_rejects_missing_rank_and_duplicate_prefill(self):
        data=fixture();data['scopes'][0]['requests'][1]['ranks'].pop()
        with self.assertRaisesRegex(ValueError,'TWO_RANKS'):review(data)
        data=fixture();data['scopes'][0]['requests'][1]['ranks'][0]['events'][1]['method']='_forward_prefill'
        with self.assertRaisesRegex(ValueError,'PREFILL_DECODE'):review(data)

    def test_value_restore_flag_does_not_replace_byte_proof(self):
        data=fixture();data['scopes'][0]['requests'][1]['ranks'][0]['events'][0]['restored_sha256']='b'*64
        with self.assertRaisesRegex(ValueError,'RESTORED_BYTES'):review(data)

if __name__=='__main__':unittest.main()
