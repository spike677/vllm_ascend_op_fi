import importlib.util,json,pathlib,tempfile,unittest
p=pathlib.Path(__file__).resolve().parents[1]/'experiments/worker_core_v11_20261118/review_gdn_baseline.py'
s=importlib.util.spec_from_file_location('review',p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
class Review(unittest.TestCase):
 def fixture(self,p):
  (p/'diagnostic.json').write_text(json.dumps(dict(status='DIAGNOSTIC_COMPLETED_NOT_QUALIFICATION',rounds=[dict(batch=b,remove=[dict(hooks=0,patches=0)],outputs=[dict(tokens=[1,2])]*b) for b in (8,8,1)])))
  for rank in range(2):
   for step in range(2):
    rows=[dict(forward=i,kind=k,name=n,stats=[dict(shape=[8,16],finite=True,sha256='a'*64)]) for i in (1,2) for k,n in [('module_output',''),('api_output','compute_logits')]]
    (p/f'rank{rank}-round{step}.jsonl').write_text('\n'.join(map(json.dumps,rows)))
 def test_hash_not_tokens_controls_result(self):
  with tempfile.TemporaryDirectory() as d:
   p=pathlib.Path(d);self.fixture(p);self.assertEqual(m.review(p)['status'],'PASS_COLD_REPEAT_B8')
   f=p/'rank0-round0.jsonl';f.write_text(f.read_text().replace('a'*64,'b'*64,1))
   self.assertIn('COLD_REPEAT_FULL_ARRAY_HASH_DIFF_RANK0',m.review(p)['failures'])
 def test_nonfinite_and_missing_rejected(self):
  with tempfile.TemporaryDirectory() as d:
   p=pathlib.Path(d);self.fixture(p);f=p/'rank1-round0.jsonl';f.write_text(f.read_text().replace('true','false',1))
   self.assertIn('NONFINITE_OR_UNOBSERVABLE',m.review(p)['failures']);f.unlink()
   self.assertIn('MISSING_RANK_TRACE',m.review(p)['failures'])
