import copy,unittest
from test_product_registration import record
from tools.product_registration_pipeline import prepare,stage_maintenance
from tools.extract_registration_record import canonical,sha

def fixture():
    r=record();pages=[]
    for scope,owner in enumerate(r['evidence']['owners']):
        for rank in (0,1):
            p=dict(schema='dcat.product.extract.v1',mechanism=r['mechanism'],source_manifest_sha256='3'*64,
                review_passed=True,review_error=None,identity_issues=[],scope=scope,rank=rank,
                identity={'tp':2,'image_digest':None},canary_exact=True,all_resources_zero=True,
                details={'consumers':['real_consumer']},files=[['receipt.json',100,'1'*64]],
                reviewer_sha256='2'*64,counts=r['evidence']['coverage'][scope*2+rank]['counts'])
            p['extract_sha256']=sha(canonical(p));pages.append(p)
    neg={k:dict(rejected=True,positive_control=True,resources_zero=True,ranks=[0,1],owners=['one','two'],receipt_sha256='4'*64)
         for k in ('missing_permit','unknown_profile','unknown_mechanism')}
    return dict(pages=pages,expected_ranks=r['expected_ranks'],owners={'0':'one','1':'two'},source=r['source'],
        negative_controls=neg,mechanism=r['mechanism'],source_manifest_sha256='3'*64,approved_reviewer_sha256='2'*64)

class PipelineTests(unittest.TestCase):
    def test_draft_staging_requires_exact_review_sha(self):
        args=fixture();r=prepare(**args);registry={'schema':'dcat.product.registry.v1','records':[]}
        with self.assertRaisesRegex(ValueError,'APPROVAL'):stage_maintenance(r,registry,approved_sha=None)
        staged=stage_maintenance(r,registry,approved_sha=r['record_sha256'])
        self.assertEqual(len(staged['records']),1);self.assertEqual(registry['records'],[])
    def test_missing_rank_or_unverified_page(self):
        for change in ('missing','tamper','counts','identity'):
            a=fixture()
            if change=='missing':a['pages'].pop()
            elif change=='tamper':a['pages'][0]['rank']=99
            else:
                p=a['pages'][0]
                if change=='counts':p['counts']['failed']=1
                else:p['identity']['tp']=1
                p['extract_sha256']=sha(canonical({k:v for k,v in p.items() if k!='extract_sha256'}))
            with self.assertRaises(ValueError):prepare(**a)
    def test_controls_need_same_cohort_and_positive(self):
        for key,value in [('positive_control',False),('ranks',[0]),('owners',['old','two'])]:
            a=fixture();a['negative_controls']['missing_permit'][key]=value
            with self.assertRaises(ValueError):prepare(**a)
    def test_missing_live_identity_not_filled_from_text(self):
        a=fixture();a['expected_ranks'][0]['source_fingerprints']=None
        with self.assertRaisesRegex(ValueError,'IDENTITY_MISSING'):prepare(**a)
if __name__=='__main__':unittest.main()
