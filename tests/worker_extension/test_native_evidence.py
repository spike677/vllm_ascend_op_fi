import json,unittest
from copy import deepcopy
from fault_injection.engines.native_evidence import compact_content_regions
from test_continuous_public_review import fixture
from tools.review_continuous_public import review


def region(site):
    return dict(site=site,consumed=True,restored=True,qmm_calls=1,region_elements=64,
        before_sha256='b'*64,restored_sha256='b'*64,source_sha256='s'*64,after_sha256='s'*64)


class EvidenceTests(unittest.TestCase):
    def test_1216_calls_stay_bounded_without_omitting_bad_restoration(self):
        rows=[region('language_model.model.layers.'+str(n//5)+'.projection_'+str(n)) for n in range(304) for step in range(4)]
        proof=compact_content_regions(rows)
        self.assertEqual(proof['region_count'],1216);self.assertEqual(len(proof['sites']),304)
        self.assertLess(len(json.dumps(proof).encode()),100000)
        rows[-1]['restored_sha256']='wrong'
        with self.assertRaisesRegex(ValueError,'PROOF_FAILED'):compact_content_regions(rows)

    def test_independent_review_rejects_missing_site_or_consumer(self):
        r=fixture();r['mechanism']='20.buffer_substitute'
        for scope in r['scopes']:
            for host in scope['requests']:
                for state in host['result']['response']['provider_state']['rank_states'].values():
                    state['deployment_acceptance_use']['actual']['native_library_sha256']='a'*64
                    for row in state['evidence']['actual_requests']:
                        row['native_memory_effect']=dict(calls=4,library_sha256='a'*64,
                            writer_operators=['native_owned_object_copy'],consumer_operators=['npu_dynamic_quant'],
                            region_proofs=compact_content_regions([region('actual') for _ in range(4)]))
        self.assertTrue(review(r)['passed'])
        for key in ('calls','copy_exact','restore_exact','consumed','qmm_calls'):
            bad=deepcopy(r)
            proof=bad['scopes'][1]['requests'][-1]['result']['response']['provider_state']['rank_states']['1']['evidence']['actual_requests'][1]['native_memory_effect']['region_proofs']
            proof['sites'][0][key]=3
            with self.subTest(key=key),self.assertRaisesRegex(ValueError,'INPUT20_'):review(bad)
