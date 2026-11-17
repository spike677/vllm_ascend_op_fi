import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from experiments.c2c3_closure_20261117 import review_vision_preparation_124 as review


class VisionPreparationReviewTests(unittest.TestCase):
    def fixture(self):
        files = {name: dict(name=name, bytes=5, sha256='s', git_blob_sha1='b') for name in review.NAMES}
        files['model.safetensors'].update(bytes=review.WEIGHT_BYTES, sha256=review.WEIGHT_SHA)
        siblings = [dict(rfilename=name, size=value['bytes'], blobId='b') for name, value in files.items()]
        next(item for item in siblings if item['rfilename'] == 'model.safetensors')['lfs'] = {
            'sha256': review.WEIGHT_SHA, 'size': review.WEIGHT_BYTES}
        metadata = dict(id=review.MODEL, sha=review.REVISION, gated=False, siblings=siblings)
        inventories = [review.exact_files(list(copy.deepcopy(files).values()))]
        return metadata, files, inventories

    def test_valid_official_sha_and_blob_comparison(self):
        review.check_model_inventory(*self.fixture())

    def test_exact_file_set_rejects_duplicate(self):
        _, files, _ = self.fixture()
        entries = list(files.values())
        entries[-1] = entries[0]
        with self.assertRaisesRegex(ValueError, 'EXACT_TEN_FILES_REQUIRED'):
            review.exact_files(entries)

    def test_mismatched_official_revision_rejected(self):
        metadata, files, inventories = self.fixture()
        metadata['sha'] = 'another-revision'
        with self.assertRaisesRegex(ValueError, 'OFFICIAL_MODEL_REVISION_MISMATCH'):
            review.check_model_inventory(metadata, files, inventories)

    def test_untrusted_receipts_do_not_override_official_blob(self):
        metadata, files, inventories = self.fixture()
        files['config.json']['git_blob_sha1'] = 'changed'
        with self.assertRaisesRegex(ValueError, 'OFFICIAL_GIT_BLOB_MISMATCH:config.json'):
            review.check_model_inventory(metadata, files, inventories)

    def test_lfs_mismatch_rejected_even_if_receipt_agrees(self):
        metadata, files, inventories = self.fixture()
        files['model.safetensors']['sha256'] = 'changed'
        inventories[0]['model.safetensors']['sha256'] = 'changed'
        with self.assertRaisesRegex(ValueError, 'OFFICIAL_LFS_WEIGHT_MISMATCH'):
            review.check_model_inventory(metadata, files, inventories)

    def test_historical_zero_field_cannot_hide_runtime_discovery(self):
        processor = dict(status='cpu', outputs={'input_ids': {'shape': [1, 901]}}, actual_npu_calls=0)
        markers = "halGetDeviceInfo failed rtGetDeviceCount aclrtGetDeviceCountImpl Can't get ascend_hal device count\n"
        report = review.inspect_runtime_log(markers + json.dumps(processor), processor)
        self.assertTrue(report['runtime_discovery_observed'])
        self.assertFalse(report['zero_runtime_api_calls_supported'])
        self.assertTrue(report['actual_npu_calls_field_is_not_a_runtime_trace'])

    def test_log_outputs_must_match_original_receipt(self):
        processor = dict(status='cpu', outputs={'input_ids': {'shape': [1, 901]}}, actual_npu_calls=0)
        markers = "halGetDeviceInfo failed rtGetDeviceCount aclrtGetDeviceCountImpl Can't get ascend_hal device count\n"
        with self.assertRaisesRegex(ValueError, 'LOG_RECEIPT_OUTPUT_MISMATCH'):
            review.inspect_runtime_log(markers + json.dumps(dict(status='cpu', outputs={})), processor)

    def test_hashes_same_bytes_as_parser_and_rejects_wrong_pin(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'receipt.json'
            data = b'{"scope":"cpu"}\n'
            path.write_bytes(data)
            record, parsed = review.load_pinned(path, hashlib.sha256(data).hexdigest())
            self.assertEqual(parsed, {'scope': 'cpu'})
            self.assertEqual(record['git_blob_sha1'], hashlib.sha1(b'blob 16\0' + data).hexdigest())
            with self.assertRaisesRegex(ValueError, 'PINNED_FILE_MISMATCH'):
                review.load_pinned(path, '0' * 64)


if __name__ == '__main__':
    unittest.main()
