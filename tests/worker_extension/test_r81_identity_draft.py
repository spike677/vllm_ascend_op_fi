import json,unittest
from pathlib import Path
from demoncat.runtime.vllm_ascend.worker_extension.product_registration import validate_identity,validate_record

class DraftTests(unittest.TestCase):
    def test_seven_actual_rank_pairs_remain_unregistered(self):
        d=json.loads((Path(__file__).resolve().parents[2]/'docs/worker_core_v1/r8_1_expected_ranks_draft_20261126.json').read_bytes())
        self.assertTrue(d['draft_only']);self.assertFalse(d['loadable_registry']);self.assertEqual(len(d['records']),7)
        for r in d['records']:
            self.assertIsNone(r['evidence']['negative_controls']);self.assertFalse(r['executable'])
            self.assertEqual([a['rank'] for a in r['expected_ranks']],[0,1])
            for a in r['expected_ranks']:validate_identity(a)
            common=lambda a:{k:v for k,v in a.items() if k not in ('rank','hardware_logical_device')}
            self.assertEqual(common(r['expected_ranks'][0]),common(r['expected_ranks'][1]))
            with self.assertRaisesRegex(ValueError,'PRODUCT_REGISTRATION_RECORD_INVALID'):validate_record(r)
