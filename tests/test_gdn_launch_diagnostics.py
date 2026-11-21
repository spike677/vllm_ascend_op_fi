import importlib.util,pathlib,unittest
p=pathlib.Path(__file__).resolve().parents[1]/'experiments/worker_core_v11_20261118/gdn_launch_diagnostics.py'
spec=importlib.util.spec_from_file_location('gdn_audit',p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
class TestMetadataInstrument(unittest.TestCase):
    def test_only_adds_debug_writes(self):
        for name,slot in [('recompute_w_u_fwd_kernel','i_t_o'),('chunk_scaled_dot_kkt_fwd_kernel','task_id')]:
            source='def '+name+'(\n    k,\n):\n            T = eos - bos\n            tl.store(output, value)\n'
            changed=m.metadata_source(source,name)
            self.assertIn('    Debug,\n',changed)
            self.assertEqual(changed.count('tl.store(Debug'),4)
            self.assertIn('Debug + '+slot+' * 16 + 3, eos',changed)
            self.assertIn('tl.store(output, value)',changed)
            compile(changed,'test','exec')
    def test_unknown_source_rejected(self):
        with self.assertRaises(AssertionError):m.metadata_source('def other(): pass','recompute_w_u_fwd_kernel')
