import copy
import unittest
from demoncat.runtime.vllm_ascend.worker_extension import product_registration as pr


class BlueMappingTests(unittest.TestCase):
    def record(self):
        rows=[r for r in pr.load_records() if r['mechanism']=='18.api_half' and r['source']=='BLUE_COMBINATION']
        self.assertEqual(len(rows),1)
        return rows[0]

    def test_reviewed_identity_and_real_group_owner_mapping(self):
        r=self.record()
        for a in r['expected_ranks']:
            self.assertEqual(pr.select_registration(r['mechanism'],'BLUE_COMBINATION',a,[r]),r)
            self.assertEqual(a['implementation_sha256'],pr.implementation_identity())
        mapped=r['evidence']['owner_rank_mapping']
        self.assertEqual({(x['group_owner'],x['rank']) for x in mapped},
                         {(o,i) for o in r['evidence']['owners'] for i in (0,1)})
        self.assertEqual(len({x['provider_owner'] for x in mapped}),4)

    def test_missing_or_drifted_mapping_never_borrows_path_digest(self):
        r=self.record();a=copy.deepcopy(r['expected_ranks'][0])
        a['runtime_libraries']={}
        with self.assertRaisesRegex(ValueError,'IDENTITY_MISSING'):
            pr.select_registration(r['mechanism'],'BLUE_COMBINATION',a,[r])
        a=copy.deepcopy(r['expected_ranks'][0]);a['runtime_libraries']['libruntime.so']['sha256']='0'*64
        with self.assertRaisesRegex(ValueError,'IDENTITY_MISMATCH'):
            pr.select_registration(r['mechanism'],'BLUE_COMBINATION',a,[r])

    def test_blue_not_yellow_not_a3_not_tp1(self):
        r=self.record();a=copy.deepcopy(r['expected_ranks'][0])
        with self.assertRaisesRegex(ValueError,'NOT_REGISTERED'):
            pr.select_registration(r['mechanism'],'YELLOW_DEDICATED_INSTANCE',a,[r])
        a['hardware_family']='A3';a['hardware_device_name']='Ascend910_9382'
        with self.assertRaisesRegex(ValueError,'IDENTITY_MISMATCH'):
            pr.select_registration(r['mechanism'],'BLUE_COMBINATION',a,[r])
        a=copy.deepcopy(r['expected_ranks'][0]);a['tp']=1
        with self.assertRaisesRegex(ValueError,'PARALLEL_SCOPE_MISMATCH'):
            pr.select_registration(r['mechanism'],'BLUE_COMBINATION',a,[r])
