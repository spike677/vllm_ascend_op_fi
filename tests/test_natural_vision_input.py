import copy
import unittest

from experiments.c2c3_closure_20261117.natural_vision_input import require_geometry
from experiments.c2c3_closure_20261117.prepare_tensor_natural_cases import cases


class NaturalVisionInputTests(unittest.TestCase):
    def test_only_original_patch16_pretrained_geometry_is_accepted(self):
        config = dict(model_type='qwen3_vl', architectures=['Qwen3VLForConditionalGeneration'],
            vision_config=dict(patch_size=16, temporal_patch_size=2, in_channels=3,
                               hidden_size=1024, spatial_merge_size=2))
        self.assertEqual(require_geometry(config)['patch_size'],16)
        for key,value in (('patch_size',14),('patch_size',2),('temporal_patch_size',1),('hidden_size',32)):
            changed = copy.deepcopy(config); changed['vision_config'][key] = value
            with self.subTest(key=key,value=value), self.assertRaisesRegex(RuntimeError,'ORIGINAL_PRETRAINED'):
                require_geometry(changed)

    def test_existing_image_and_full_model_census_are_explicitly_model_local(self):
        row, = cases('/home/17-fault-c2c3-20261117/source-new','vision17-census')
        self.assertIn('--kind hf-vision',row['command'])
        self.assertEqual(row['review']['expected_hits'],0)
        self.assertIn('MODEL_LOCAL',row['review']['worker'])
        self.assertEqual(row['mounts'][-1],['/home/jw/Wan2.2/examples/i2v_input.JPG','/natural-input.JPG','ro'])
        self.assertIn('/models/Qwen3-VL-2B-Instruct',row['review']['model'])


if __name__ == '__main__':
    unittest.main()
