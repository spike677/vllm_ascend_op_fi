import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from reference_adapters.review_quant import review_directory


class ReviewTests(unittest.TestCase):
    def fixture(self, directory):
        data = dict(mechanism='external.row9_scale_half', scopes=[])
        chunks = {0: [], 1: []}
        for owner in range(2):
            scope = dict(duplicate_enable={'exit_code':4}, requests=[])
            data['scopes'].append(scope)
            for n in range(3):
                rid = f'{owner}-{n}'
                request = dict(result={'ranks':[]})
                scope['requests'].append(request)
                for rank in (0,1):
                    event = dict(site='linear', request_ids=[rid], changed=1, changed_elements=2,
                        before_sha256=['a','b'], restored_sha256=['a','b'], after_sha256=['c','d'],
                        consumer_sha256=['c','d'], consumer_operator='npu::npu_quant_matmul',
                        consumed=True, restored=True, qmm_reads=1, scale_multiplier=.5,
                        rounding='nearest_even', signed_range=[-127,127], saturated_elements=2,
                        clipped_elements=1, zero_elements=0, native_zero_elements=0)
                    request['result']['ranks'].append(dict(rank=rank, request_id=rid,
                        coverage={'applied':1}, sites=[event.copy()]))
                    chunks[rank].append(dict(rank=rank,events=[event]))
        (directory/'product_http_18.json').write_text(json.dumps(data))
        (directory/'product_control_negative.json').write_text(json.dumps(dict(
            missing_permit_cli={'exit_code':4}, unknown_profile_cli={'exit_code':4})))
        for rank in (0,1):
            (directory/f'external-input-rank{rank}.jsonl').write_text('\n'.join(map(json.dumps,chunks[rank])))

    def test_actual_consumer_hash_and_sidecar_association(self):
        with tempfile.TemporaryDirectory() as tmp, patch('reference_adapters.review_quant.review',return_value={'status':'PASS'}):
            directory=Path(tmp);self.fixture(directory)
            result=review_directory(directory)
            self.assertEqual(result['effects_by_rank'][0]['calls'],6)
            file=directory/'external-input-rank1.jsonl'
            lines=file.read_text().splitlines();chunk=json.loads(lines[0])
            chunk['events'][0]['consumer_sha256']=['wrong']
            lines[0]=json.dumps(chunk);file.write_text('\n'.join(lines))
            with self.assertRaisesRegex(ValueError,'QMM_PROOF'):review_directory(directory)

    def test_orphan_sidecar_rejected(self):
        with tempfile.TemporaryDirectory() as tmp, patch('reference_adapters.review_quant.review',return_value={}):
            directory=Path(tmp);self.fixture(directory)
            file=directory/'external-input-rank0.jsonl'
            chunk=json.loads(file.read_text().splitlines()[0])
            chunk['events'][0]['request_ids']=['orphan']
            with file.open('a') as stream:stream.write('\n'+json.dumps(chunk))
            with self.assertRaisesRegex(ValueError,'ORPHAN'):review_directory(directory)
