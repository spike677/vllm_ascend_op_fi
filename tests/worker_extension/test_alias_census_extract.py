import hashlib,json,subprocess,sys,tempfile,unittest
from pathlib import Path
from tools.summarize_alias_census import pages

class AliasExtractTests(unittest.TestCase):
    def test_deterministic_bounded_null_and_no_qualification(self):
        row=dict(rank=0,hooks_remaining=0,execute_wrapper_removed=True,reports=[dict(report=dict(
            counts={'view_writes':2},view_base_formats={'None':2},groups=[dict(view_op='slice',writer='copy_',consumer='add',base_format='None',count=1)]))])
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'receipt.json';p.write_bytes(json.dumps({'ranks':[row]}).encode())
            cmd=[sys.executable,'tools/summarize_alias_census.py',str(p)]
            a=subprocess.check_output(cmd);b=subprocess.check_output(cmd);self.assertEqual(a,b)
            self.assertEqual(len(list(pages(p))),2)
            for raw in a.splitlines():
                self.assertLessEqual(len(raw)+1,4096);d=json.loads(raw);digest=d.pop('extract_sha256')
                self.assertEqual(digest,hashlib.sha256(json.dumps(d,sort_keys=True,ensure_ascii=False,separators=(',',':')).encode()).hexdigest())
                self.assertFalse(d['qualification_granted']);self.assertFalse(d['injection_enabled'])
            self.assertEqual(json.loads(a.splitlines()[0])['view_base_formats'],{'None':2})

    def test_real_output_boundary_pages_keep_unknown_forward_count(self):
        row=dict(rank=0,hooks_remaining=0,execute_wrapper_removed=True,events=[dict(format=2,writer='actual',consumer='projection',base_changed=True,device_consumer_completed=True)])
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'r.json';p.write_bytes(json.dumps({'ranks':[row]}).encode())
            result=list(pages(p));self.assertEqual(result[0]['counts']['consumed'],1)
            self.assertIsNone(result[0]['forwards']);self.assertFalse(result[0]['private_format_chain_proven'])

if __name__=='__main__':unittest.main()
