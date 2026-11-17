"""CPU provenance and queue preparation contracts; fixture bytes are not model evidence."""
import json
from pathlib import Path
import tempfile
import unittest

import torch

from experiments.c2c3_postwrite_20261117.model_identity import safetensors_identity, model_source_identity
from experiments.c2c3_postwrite_20261117.prepare_coverage_cases import cases


class NaturalCoverageProbeTests(unittest.TestCase):
    def test_exact_selected_weight_files_and_changes_are_hashed(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            (root/'config.json').write_text('{"model_type":"cpu-fixture"}')
            (root/'model.safetensors').write_bytes(b'explicit CPU fixture bytes')
            first=safetensors_identity(root)
            self.assertEqual(list(first['manifest']['files']),['model.safetensors'])
            (root/'model.safetensors').write_bytes(b'different explicit CPU fixture bytes')
            self.assertNotEqual(first['weights_sha256'],safetensors_identity(root)['weights_sha256'])

    def test_indexed_weights_are_complete_and_external_paths_refuse(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);(root/'config.json').write_text('{}')
            for name in ('first.safetensors','second.safetensors'):(root/name).write_bytes(name.encode())
            index=root/'model.safetensors.index.json'
            index.write_text(json.dumps({'weight_map':{'a':'first.safetensors','b':'second.safetensors'}}))
            identity=safetensors_identity(root)
            self.assertEqual(len(identity['manifest']['files']),2)
            index.write_text(json.dumps({'weight_map':{'a':'../not-a-weight.safetensors'}}))
            with self.assertRaises((FileNotFoundError,ValueError)):safetensors_identity(root)

    def test_loaded_python_model_source_is_recorded_from_actual_class(self):
        actual=model_source_identity(torch.nn.Sequential(torch.nn.Linear(2,2),torch.nn.SiLU()))
        self.assertEqual(len(actual['source_sha256']),64)
        self.assertTrue(any(item['class_name']=='torch.nn.modules.linear.Linear'
                            for item in actual['manifest']['classes']))
        self.assertTrue(all(Path(path).is_file() for path in actual['manifest']['files']))

    def test_queue_arms_share_fixed_capture_workload_and_fault_only_c3(self):
        rows=cases(source='/source-tree',native='/new-native',sha256='a'*64,build_fingerprint='b'*64,
                   model='/Qwen',cards=[5])
        self.assertEqual(len(rows),2)
        self.assertIn('--statistics',rows[0]['command'])
        self.assertNotIn('--try-c3',rows[0]['command'])
        self.assertIn('--try-c3',rows[1]['command'])
        self.assertEqual(rows[0]['review']['physical_cards'],[5])
        self.assertEqual(rows[0]['review']['expected_hits'],0)
        self.assertEqual(rows[0]['review']['resource_budget']['timeout_seconds'],1800)
        with self.assertRaises(ValueError):
            cases(source='/s',native='/n',sha256='invalid',build_fingerprint='b'*64,model='/m')


if __name__=='__main__':unittest.main()
