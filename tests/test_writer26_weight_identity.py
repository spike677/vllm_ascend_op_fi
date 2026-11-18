"""Local loader manifests without Torch, tensor deserialization or a device."""
import json
from pathlib import Path
import tempfile
import unittest

from experiments.c2c3_postwrite_20261117.model_identity import (
    digest, file_sha, vllm_safetensors_identity,
)


class WeightIdentityTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(self.enterContext(tempfile.TemporaryDirectory()))
        (self.root/'config.json').write_text('{}')

    def shards(self, names=('quant-00001.safetensors', 'quant-00002.safetensors')):
        for name in names:
            (self.root/name).write_bytes(('fixture:'+name).encode())
        self.index = self.root/'quant_weights.safetensors.index.json'
        self.index.write_text(json.dumps({'weight_map': {str(i): n for i, n in enumerate(names)}}))

    def test_actual_nonstandard_quant_index_and_complete_loader_set(self):
        self.shards()
        q = self.root/'quant_model_description.json'
        q.write_text('{"quantization":"fixture"}')
        value = vllm_safetensors_identity(self.root)
        manifest = value['manifest']
        self.assertEqual(set(manifest['files']), {'quant-00001.safetensors', 'quant-00002.safetensors'})
        self.assertEqual(manifest['index'], dict(file=self.index.name, sha256=file_sha(self.index)))
        self.assertEqual(manifest['quantization_description']['sha256'], file_sha(q))
        self.assertEqual(value['weights_sha256'], digest(manifest))
        (self.root/'quant-00002.safetensors').write_bytes(b'changed')
        self.assertNotEqual(vllm_safetensors_identity(self.root)['weights_sha256'], value['weights_sha256'])

    def test_standard_single_file_needs_no_index(self):
        (self.root/'model.safetensors').write_bytes(b'fixture')
        self.assertIsNone(vllm_safetensors_identity(self.root)['manifest']['index'])

    def test_extra_or_missing_shard_refuses(self):
        self.shards()
        extra = self.root/'extra.safetensors'
        extra.write_bytes(b'extra')
        with self.assertRaisesRegex(ValueError, 'SELECTION_AMBIGUOUS'):
            vllm_safetensors_identity(self.root)
        extra.unlink()
        (self.root/'quant-00002.safetensors').unlink()
        with self.assertRaisesRegex(ValueError, 'SELECTION_AMBIGUOUS'):
            vllm_safetensors_identity(self.root)

    def test_ambiguous_or_missing_index_refuses(self):
        self.shards()
        other = self.root/'other.safetensors.index.json'
        other.write_bytes(self.index.read_bytes())
        with self.assertRaisesRegex(ValueError, 'UNAMBIGUOUS'):
            vllm_safetensors_identity(self.root)
        other.unlink()
        self.index.unlink()
        with self.assertRaisesRegex(ValueError, 'SHARD_INDEX_REQUIRED'):
            vllm_safetensors_identity(self.root)

    def test_index_cannot_select_an_external_or_nested_path(self):
        self.shards()
        for name in ('../outside.safetensors', '/outside.safetensors', 'C:\\outside.safetensors',
                     'nested/shard.safetensors', None):
            with self.subTest(name=name):
                self.index.write_text(json.dumps({'weight_map': {'x': name}}))
                with self.assertRaisesRegex(ValueError, 'FLAT_LOCAL'):
                    vllm_safetensors_identity(self.root)

    def test_config_required_and_bad_weight_map_refuses(self):
        self.shards()
        (self.root/'config.json').unlink()
        with self.assertRaisesRegex(ValueError, 'MODEL_CONFIG_REQUIRED'):
            vllm_safetensors_identity(self.root)
        (self.root/'config.json').write_text('{}')
        self.index.write_text('{"weight_map":{}}')
        with self.assertRaisesRegex(ValueError, 'ACTUAL_SAFETENSORS_INDEX_REQUIRED'):
            vllm_safetensors_identity(self.root)


if __name__ == '__main__':
    unittest.main()
