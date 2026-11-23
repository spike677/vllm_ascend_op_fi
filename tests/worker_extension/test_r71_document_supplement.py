import ast,hashlib,json,re,unittest,zipfile
from pathlib import Path

class DocumentationTests(unittest.TestCase):
    def test_six_documents_exact_bytes(self):
        root=Path('releases/r71-docs-20261123-v2-r1');m=json.loads((root/'manifest.json').read_bytes())
        self.assertEqual(len(m['files']),6);self.assertFalse(m['runtime_modified'])
        z=root/'r71-docs-20261123-v2-r1.zip'
        expected=(root/'SHA256SUMS').read_text().splitlines()[0].split()[0]
        self.assertEqual(hashlib.sha256(z.read_bytes()).hexdigest(),expected)
        with zipfile.ZipFile(z) as f:
            self.assertIsNone(f.testzip())
            for row in m['files']:
                data=f.read(row['path']);self.assertEqual(len(data),row['bytes']);self.assertEqual(hashlib.sha256(data).hexdigest(),row['sha256'])
    def test_cli_and_complete_native_schema(self):
        s=Path('docs/worker_core_v1/tp2_overnight_runbook_20261123.md').read_text(encoding='utf-8')
        self.assertNotIn('dcatpfi',s);self.assertIn('python -m demoncat.cli fault enable',s)
        rows=[json.loads(b) for b in re.findall(r'```json\n(.*?)\n```',s,re.S)]
        native=[r for r in rows if r.get('profile')=='41.tail_column'][0]
        self.assertEqual(native['fault']['fault_id'],41);self.assertEqual(native['tp'],2)
        self.assertEqual(native['source_manifest_sha256'],'77e30525212d2d1e8f46313452bacdd46c81ecb751ce2e052e4d72dabbf6f7e2')
        self.assertEqual(native['native_sha256'],'d9aa0d9488c5c1e8f7fca10c297e4968687ceb6a24866d8346bd33a58afce274')
        for code in re.findall(r'```python\n(.*?)\n```',s,re.S):ast.parse(code)
    def test_documented_symbols_exist(self):
        from demoncat.runtime.vllm_ascend.worker_extension import deployment_authorization as a
        from demoncat.runtime.vllm_ascend.worker_extension.product_config import configure_product_deployments
        from demoncat.runtime.vllm_ascend.worker_extension.product_host import ProductClient
        for name in ('configure_deployment_acceptance','install_deployment_acceptance_permit','remove_deployment_acceptance_permit'):
            self.assertTrue(callable(getattr(a,name)))
        self.assertTrue(callable(configure_product_deployments))
        for name in ('from_deployment','capabilities','attach','enable','status','disable','detach'):self.assertTrue(callable(getattr(ProductClient,name)))
if __name__=='__main__':unittest.main()
