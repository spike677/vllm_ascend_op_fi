import importlib.util,pathlib,unittest
p=pathlib.Path(__file__).resolve().parents[1]/'experiments/worker_core_v11_20261118/gdn_inline_atomic.py'
s=importlib.util.spec_from_file_location('inline',p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
class Inline(unittest.TestCase):
 def test_entry_only(self):
  ir='module attributes {} {\n'+''.join('  func.func @recompute_w_u_fwd_kernel_atomic_none_mix_'+k+'() attributes {hacc.entry, hivm.func_core_type = #hivm.func_core_type<'+k.upper()+'>} {\n    return\n  }\n' for k in ('aic','aiv'))+'}\n'
  result=m.rewrite_ir(ir)
  self.assertEqual(result.count('func.call'),2)
  self.assertEqual(result.count('    return'),2)
  self.assertNotIn('store',result)
 def test_unknown_rejected(self):
  with self.assertRaises(AssertionError):m.rewrite_ir('module {}')
