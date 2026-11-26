"""The same request events must aggregate identically in both evidence tools."""
import unittest
from types import SimpleNamespace as NS
from test_product_review import receipt
from tools.review_product_mode import review
from tools.product_effect_adapter import ProductSession


class CountProvenanceTests(unittest.TestCase):
    def test_same_event_set_both_paths_agree_not_a_constant_offset(self):
        for elements,expected in [([18384,18385,18385],55154),([18384]*3,55152)]:
            data=receipt();data['mechanism']='17.spatial_padding'
            data['image_pixel_sha256']=['a'*64,'b'*64,'c'*64]
            for scope in data['scopes']:
                for request,count in zip(scope['requests'][:3],elements):
                    for rank in request['result']['ranks']:rank['changed_elements']=count
            report=review(data)
            for original,reviewed in zip(data['scopes'],report['scopes']):
                session=ProductSession.__new__(ProductSession)
                session.owner='group-'+str(original['scope'])
                session.worker_identity=[dict(rank=r) for r in range(2)]
                session.client=NS(identity=NS(tenant_id='tenant'))
                session.stopped=original['disable']
                session.stopped['requests']=[r['result'] for r in original['requests']]
                for r in session.stopped['cleanup']['unregister']:r['attachment']=dict(tenant_id='tenant')
                evidence=session.evidence()
                for rank in range(2):
                    total=sum(e['changed_elements'] for e in reviewed['effects'] if e['rank']==rank)
                    self.assertEqual(total,expected)
                    self.assertEqual(evidence['ranks'][rank]['changed_elements'],total)
