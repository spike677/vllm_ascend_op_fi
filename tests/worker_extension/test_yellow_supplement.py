import importlib.util,json,tempfile,unittest,hashlib,subprocess,sys
from pathlib import Path
from copy import deepcopy
from types import SimpleNamespace as NS
import torch
from tools.dsv4_capture_bundle import assemble_run
from tools.extract_native_build_record import extract
from fault_injection.dsv4_noise import TARGET

ROOT=Path(__file__).resolve().parents[2]
def load(name):
    s=importlib.util.spec_from_file_location(name,ROOT/'experiments/worker_core_v11_20261118'/f'{name}.py')
    m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m

class YellowSupplementTests(unittest.TestCase):
    def test_capture_through_review_and_text_extract_cli(self):
        capture=load('dsv4_noise_capture').capture
        class Model:
            def compute_logits(self,x):return x+1
        calibration=[]
        for proc in range(2):
            for sample in range(5):
                rows=[]
                for rank in range(8):
                    w=NS(rank=rank,model_runner=NS(model=Model(),input_batch=NS(req_ids=['real-cpu-call'])))
                    capture(w,'start')
                    for step in range(4):w.model_runner.model.compute_logits(torch.arange(6.).reshape(1,6)+step)
                    rows.append(capture(w,'stop'))
                calibration.append(assemble_run(rows,tokens=[1,2,3,4],prompt_token_ids=[42]*2500,single_chunk=True,sample_id=str(sample),process_identity='CPU_PROCESS_FIXTURE_'+str(proc)))
        combo=dict(TARGET,model_config_sha256='a'*64,quant_model_description_sha256='b'*64,runtime_source_sha256='c'*64,
            safetensors_index_sha256={'model.safetensors.index.json':'d'*64},max_model_len=4096,max_num_batched_tokens=4096,max_num_seqs=1)
        with tempfile.TemporaryDirectory() as td:
            p=Path(td);(p/'input.json').write_text(json.dumps(dict(combination=combo,calibration=calibration)))
            subprocess.run([sys.executable,'-m','tools.review_dsv4_noise',str(p/'input.json'),'--output',str(p/'summary.json')],check=True,capture_output=True)
            cmd=[sys.executable,'-m','tools.extract_dsv4_record','noise',str(p/'summary.json')]
            first=subprocess.check_output(cmd);self.assertEqual(first,subprocess.check_output(cmd))
            lines=first.splitlines();self.assertEqual(len(lines),8)
            for line in lines:
                self.assertLessEqual(len(line)+1,4096);r=json.loads(line)['payload'];self.assertTrue(r['finite']);self.assertTrue(r['token_stable'])
        with self.assertRaisesRegex(ValueError,'EIGHT_RANKS'):assemble_run(rows[:7],tokens=[1,2,3,4],prompt_token_ids=[42]*2500,single_chunk=True,sample_id='x',process_identity='p')

    def test_observe_probe_to_extract(self):
        m=load('dsv4_observe_probe')
        class Runner:
            requests={}
            def execute_model(self,s):return torch.ones(2,4,dtype=torch.bfloat16)@torch.ones(4,3,dtype=torch.bfloat16)
        w=NS(rank=0,model_runner=Runner());m.observe(w,'start')
        w.model_runner.execute_model(NS(num_scheduled_tokens={'r':2},scheduled_new_reqs=[NS(req_id='r')]))
        report=m.observe(w,'stop')
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'observe.json';p.write_text(json.dumps(report))
            raw=subprocess.check_output([sys.executable,'-m','tools.extract_dsv4_record','observe',str(p)])
            self.assertTrue(raw);self.assertTrue(all(len(s)+1<=4096 for s in raw.splitlines()))
        self.assertTrue(report['observer_removed']);self.assertEqual(report['injected_calls'],0)

    def test_native_extract_does_not_grant_and_detects_mismatch(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td);(p/'lib.so').write_bytes(b'cpu-fixture')
            r=dict(schema='demoncat.native_regions.build.v1',status='BUILD_AND_CPU_TESTS_ONLY',npu_calls=0,
                library=dict(file='lib.so',bytes=11,sha256=hashlib.sha256(b'cpu-fixture').hexdigest()),commands=[dict(args=['g++'],returncode=0)],sdk_version_files={})
            (p/'receipt.json').write_text(json.dumps(r));a=extract(p/'receipt.json',p/'lib.so')
            self.assertFalse(a['qualification_granted']);self.assertIn('SDK_VERSION_UNOBSERVED',a['issues'])
            command=[sys.executable,'-m','tools.extract_native_build_record',str(p/'receipt.json'),str(p/'lib.so')]
            first=subprocess.check_output(command);self.assertEqual(first,subprocess.check_output(command))
            self.assertLessEqual(len(first),4096);self.assertFalse(json.loads(first)['qualification_granted'])
            self.assertIn('SOURCE_FILE_HASHES_MISSING',json.loads(first)['issues'])
            (p/'lib.so').write_bytes(b'other');self.assertIn('LIBRARY_IDENTITY_MISMATCH',extract(p/'receipt.json',p/'lib.so')['issues'])

    def test_optional_image_keeps_other_registration_gates(self):
        from test_registration_extract import filled,RegistrationTests
        from tools.extract_registration_record import verify,seal,DEPLOYMENT
        rec=RegistrationTests().run_extract(filled());rec['d'][DEPLOYMENT.index('image_digest')]=None;rec=seal(rec)
        self.assertTrue(verify(rec)['registration_ready'])
        for key in ('hardware_device_name','vllm','torch','quant_model_description_sha256','safetensors_index_sha256','max_num_seqs'):
            altered=deepcopy(rec);altered['d'][DEPLOYMENT.index(key)]=None
            self.assertFalse(verify(seal(altered))['registration_ready'],key)

    def test_product_tp1_review_and_bounded_text_without_duplication(self):
        from test_product_review import receipt
        from test_registration_extract import filled
        from tools.extract_registration_record import DEPLOYMENT
        from tools.extract_product_record import pages
        d=receipt()
        for s in d['scopes']:
            for row in s['requests']:row['result']['ranks']=row['result']['ranks'][:1]
            for key in ('before','after'):s['disable']['canary'][key]=s['disable']['canary'][key][:1]
            for key in s['disable']['cleanup']:s['disable']['cleanup'][key]=s['disable']['cleanup'][key][:1]
            s['cleanup']=s['cleanup'][:1]
        identity=filled()['scopes'][0]['requests'][0]['result']['response']['provider_state']['rank_states']['0']['deployment_acceptance_use']['actual']
        identity.update(tp=1,image_digest=None)
        with tempfile.TemporaryDirectory() as td:
            p=Path(td);(p/'receipt.json').write_text(json.dumps(d));(p/'identity.json').write_text(json.dumps(identity))
            out=pages(p/'receipt.json',p/'identity.json');self.assertEqual(len(out),2)
            self.assertTrue(all(r['review_passed'] and not r['identity_issues'] for r in out))
            self.assertEqual(out[0]['counts']['applied'],3)
            d['scopes'][0]['disable']['canary']['after'][0]['observations'][0]['output'][0]['logits']='bad'
            (p/'receipt.json').write_text(json.dumps(d));self.assertFalse(pages(p/'receipt.json',p/'identity.json')[0]['review_passed'])
