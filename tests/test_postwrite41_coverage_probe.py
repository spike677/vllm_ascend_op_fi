"""Probe provenance and queued budgets; no runtime/NPU qualification."""
import json
from pathlib import Path
import tempfile
import unittest

from experiments.c2c3_postwrite_20261117.natural41_coverage_probe import actual_vllm_weights,SOURCES,ROOT
from experiments.c2c3_postwrite_20261117.prepare41_coverage_cases import cases


class NaturalKVProbeTests(unittest.TestCase):
    def test_safetensors_loader_manifest_rejects_extra_ambiguous_weights(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);(root/'config.json').write_text('{}')
            (root/'model.safetensors').write_bytes(b'explicit CPU fixture, not real weights')
            record=actual_vllm_weights(root)
            self.assertIn("load_format='safetensors'",record['manifest']['loader_selection'])
            (root/'unselected.safetensors').write_bytes(b'extra file')
            with self.assertRaisesRegex(RuntimeError,'FILE_SELECTION_AMBIGUOUS'):actual_vllm_weights(root)

    def test_source_manifest_includes_independent_observer_and_group(self):
        self.assertTrue(all((ROOT/name).is_file() for name in SOURCES))
        self.assertIn('fault_injection/deployments/postwrite_kv_inventory.py',SOURCES)
        self.assertIn('src/demoncat/runtime/vllm_ascend/worker_extension/postwrite41_coverage.py',SOURCES)

    def test_both_arms_same_bounded_workload_fresh_c1_and_fault_only_c3(self):
        rows=cases(source='/new-source',native='/a2-native',sha256='a'*64,build_fingerprint='b'*64,
                   model='/Qwen',cards=[5])
        self.assertEqual(len(rows),2)
        for row in rows:
            self.assertEqual(row['timeout'],1800)
            self.assertIn('--expected-targets 28',row['command'])
            self.assertIn('--statistics',row['command'])
            self.assertEqual(row['env']['VLLM_ENABLE_V1_MULTIPROCESSING'],'0')
            self.assertIn('fresh C1 1',row['review']['expected_hits'])
        self.assertNotIn('--try-c3',rows[0]['command'])
        self.assertIn('--try-c3',rows[1]['command'])
        self.assertIn('28 once; then C3 56',rows[1]['review']['expected_hits'])
        self.assertEqual(rows[1]['review']['physical_cards'],[5])
        with self.assertRaisesRegex(ValueError,'BOUNDED_ACTUAL_UNIVERSE'):
            cases(source='/s',native='/n',sha256='a'*64,build_fingerprint='b'*64,model='/m',expected_targets=999)


if __name__=='__main__':unittest.main()
