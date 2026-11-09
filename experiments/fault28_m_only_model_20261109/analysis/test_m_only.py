import json
from pathlib import Path
import sys
import tempfile
import unittest
from types import SimpleNamespace
import numpy as np

ROOT=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(ROOT))
from numeric_observer import observe,compare_bytes
from m_only_contract import require_selection,require_plan


class ContractTests(unittest.TestCase):
    def test_joint_selection_rejected(self):
        for target in ('joint','key_group'):
            with self.assertRaises(ValueError):
                require_selection(SimpleNamespace(adapter=SimpleNamespace(version='validated'),selection_target=target))

    def test_ratio_change_rejected_even_with_correct_m(self):
        with self.assertRaises(ValueError):require_plan({'query_tile':64,'key_group_ratio':4},True)
        require_plan({'query_tile':64,'key_group_ratio':8},True)
        require_plan({'query_tile':128,'key_group_ratio':8},False)

    def test_intermediate_change_with_last_query_unchanged_is_retained(self):
        b=np.arange(32,dtype='<f2').reshape(2,2,4,2);f=b.copy();f[0,1,0,1]+=1
        r=compare_bytes(b.tobytes(),f.tobytes(),b.shape,2)
        self.assertEqual(r['changed_elements'],1);self.assertEqual(r['last_query_changed_elements'],0)
        self.assertEqual(r['per_request_changed_elements'],[1,0])

    def test_last_query_change_reported_per_request(self):
        b=np.zeros((2,2,4,2),dtype='<f2');f=b.copy();f[1,0,-1,1]=1
        r=compare_bytes(b.tobytes(),f.tobytes(),b.shape,2)
        self.assertEqual(r['last_query_changed_elements'],1)
        self.assertEqual(r['per_request_changed_elements'],[0,1])

    def test_missing_baseline_fails_instead_of_establishing_fault_reference(self):
        with tempfile.TemporaryDirectory() as d:
            c={'mode':'controlled','stage':'prefill','kind':'attention','layer_index':0,'query_axis':2,
               'reference_file':str(Path(d)/'missing.bin')}
            with self.assertRaises(FileNotFoundError):observe(bytes(16),(1,1,4,2),c)

    def test_baseline_not_overwritten_and_off_restores_exactly(self):
        with tempfile.TemporaryDirectory() as d:
            c={'mode':'native','stage':'prefill','kind':'attention','layer_index':0,'query_axis':2,
               'reference_file':str(Path(d)/'reference.bin')}
            raw=np.arange(8,dtype='<f2').tobytes();shape=(1,1,4,2)
            observe(raw,shape,c)
            with self.assertRaises(FileExistsError):observe(raw,shape,c)
            r=observe(raw,shape,dict(c,mode='off'))
            self.assertTrue(r['comparison']['bitwise_equal']);self.assertEqual(r['comparison']['changed_elements'],0)


if __name__=='__main__':unittest.main()
