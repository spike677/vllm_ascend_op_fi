"""Real receipt layout and Host/Runtime recovery; no device or Native calls."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path,PurePosixPath
import struct
import subprocess
import sys
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import Mock,patch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'experiments/c2c3_closure_20261117'))
import arch22_normal_bridge_evidence as common
import natural33_arch22_s2_worker_core_probe as probe
import review_arch22_normal_bridge as reader
from demoncat.runtime.vllm_ascend.worker_extension import paged33_s2_evidence as proof
from demoncat.runtime.vllm_ascend.worker_extension.serving_context import ContextRefused


class ReceiptLayoutContracts(unittest.TestCase):
    def setUp(self):
        self.folder=Path(self.enterContext(tempfile.TemporaryDirectory())).resolve()
        self.build=self.folder/'build';preflight=self.build/'attention-preflight';preflight.mkdir(parents=True)
        library=preflight/'observer.so';library.write_bytes(b'CPU fixture bytes, never loaded')
        self.receipt=preflight/'result.json'
        probe.save(self.receipt,dict(observer_compiled=True,actual_npu_calls=0,
            observer_memory_selftest={'exit_code':0},observer_source={'sha256':'a'*64},
            observer_library={'path':'/build/attention-preflight/observer.so','sha256':common.sha(library)}))
        self.sha=common.sha(self.receipt)
        self.source=dict(pins={common.NATIVE+'launch_observer.cpp':'a'*64,
                              common.EXPERIMENT+'review_arch22_normal_bridge.py':common.sha(reader.__file__)})

    def test_original_file_as_root_reproduces_and_exact_translation_uses_real_reader(self):
        # The old producer passed the Native receipt to a root-taking reader.
        with self.assertRaises(OSError):common.observer_bundle(self.receipt,self.source,self.sha)
        root=proof.normal_observer_root(self.receipt,self.sha)
        checked=common.observer_bundle(root,self.source,self.sha)
        self.assertEqual(root,self.build);self.assertEqual(checked['sha256'],self.sha)
        self.assertEqual(checked['folder'],self.receipt.parent)

    def test_no_search_or_hash_bypass_for_wrong_receipt_layout(self):
        wrong=self.folder/'result.json';wrong.write_bytes(self.receipt.read_bytes())
        for path,digest in ((wrong,self.sha),(self.build,self.sha),(self.receipt,'b'*64)):
            with self.subTest(path=str(path),sha=digest),self.assertRaises(ContextRefused):
                proof.normal_observer_root(path,digest)
        self.receipt.write_text('{}',encoding='utf-8')
        with self.assertRaisesRegex(ContextRefused,'S2_NORMAL_OBSERVER_RECEIPT_CHANGED'):
            proof.normal_observer_root(self.receipt,self.sha)

    def fixture(self):
        table=list(range(65))+[0]*15
        signature=dict(block_table_values=[table],writer_consumer_slot=[1024],
            cache_geometry={k:{'shape':[128,16,8,128]} for k in ('key','value')},
            tensors={'block_table':{'bytes':320,'sha256':hashlib.sha256(struct.pack('<80i',*table)).hexdigest()}})
        execution={'driver_sha256':'old-driver','installed_files':{'actual.py':'same'}}
        capture=dict(first_decode_boundary={'logical_input_binding':{'evidence':{'before':{'signature':signature},'outputs':[]}}},
            **{key:dict(shape=[1,2],dtype='torch.float32',sha256='c'*64,finite=True) for key in ('hidden','logits')})
        baseline=self.folder/'baseline';baseline.mkdir()
        probe.save(baseline/'result.json',dict(execution_contract=execution,phases=[{'finish':{'capture':capture}}]))
        published=self.folder/'review.json'
        summary=dict(source={'manifest_sha256':'d'*64},build={'completion_sha256':'e'*64,'variant':proof.VARIANT},
            arms={'stock':{'result_sha256':'f'*64},'baseline':{'result_sha256':common.sha(baseline/'result.json')}})
        probe.save(published,summary)
        fresh=dict(summary,issues=[],checks=1,normal_payloads_exact=True,complete_payload_pairs=435,
            observer={'receipt_sha256':self.sha},native_call_scope=dict(module_path=proof.FIRST,layout='TND',query_tokens=1,
                kv_length=1025,page_size=16,private_calls=4,natural_layers_per_phase=28,normal_only=True))
        args=dict(source=ROOT,source_manifest_sha256='d'*64,build_root=self.folder,completion_sha256='e'*64,
            observer=self.receipt,observer_receipt_sha256=self.sha,stock_run=self.folder,stock_sha256='f'*64,
            baseline_run=baseline,baseline_sha256=common.sha(baseline/'result.json'),report=published,
            report_sha256=common.sha(published),current_execution=dict(execution,driver_sha256='new-driver'))
        return args,fresh

    def replay(self,args,fresh):
        # The large payload algebra has its own negative suite. Here exercise
        # the actual frozen reader's observer contract at the loader boundary.
        def review(actual):
            common.observer_bundle(actual.observer,self.source,actual.observer_receipt_sha256)
            return deepcopy(fresh)
        with patch.object(common,'frozen_source',return_value=self.source),patch.object(reader,'review',side_effect=review):
            return proof.load_verified_normal_evidence(**args)

    def test_loader_translates_only_root_and_keeps_all_proof_and_execution_guards(self):
        args,fresh=self.fixture();value=self.replay(args,fresh)
        self.assertFalse(value.data()['qualification_granted'])
        for field,value in (('complete_payload_pairs',434),('normal_payloads_exact',False),
                            ('issues',[{'path':'source_build_observer','error':'original reader error'}])):
            bad=deepcopy(fresh);bad[field]=value
            with self.subTest(field=field),self.assertRaisesRegex(ContextRefused,'S2_FRESH_NORMAL_RAW_REPLAY_FAILED') as caught:
                self.replay(args,bad)
            self.assertEqual(caught.exception.response['replay'][field],value)
            self.assertEqual(caught.exception.response['observer_build_root'],str(self.build))
        args['current_execution']['installed_files']['actual.py']='different'
        with self.assertRaisesRegex(ContextRefused,'S2_NORMAL_MODEL_PACKAGE_OR_RUNNER_CHANGED'):
            self.replay(args,fresh)

    def test_worker_preserves_first_stage_and_nested_raw_failure_before_transport(self):
        from demoncat.runtime.vllm_ascend.worker_extension import paged33_arch22_s2_effects as effects
        from demoncat.runtime.vllm_ascend.worker_extension import serving_attention
        first=ContextRefused('S2_FRESH_NORMAL_RAW_REPLAY_FAILED')
        first.response={'replay':{'issues':[{'path':'source_build_observer','error':'original path failure'}]}}
        worker=NS(demoncat_worker_identity=lambda:dict(status='WORKER_IDENTIFIED',quarantined=False,worker={'pid':123}))
        # Model the Linux installed-module path without depending on a Windows
        # drive's interpretation of the absolute /installed deployment root.
        def path(value):
            return NS(resolve=lambda:PurePosixPath('/installed/demoncat/driver.py')) if value=='linux-installed-fixture' else Path(value)
        with patch.object(probe,'worker_object',return_value=worker),patch('inspect.getfile',return_value='linux-installed-fixture'),\
             patch.object(probe,'Path',side_effect=path),\
             patch.object(proof,'load_verified_normal_evidence',side_effect=first),\
             patch.object(effects,'Arch22S2EffectsNativeCall') as native,\
             patch.object(serving_attention,'install_attention_serving_binding') as binding:
            with self.assertRaises(ContextRefused) as caught:
                probe.setup(worker,{'pid':123},[1],{}, {'evidence_dir':self.folder/'native'}, {})
        self.assertIs(caught.exception,first);native.assert_not_called();binding.assert_not_called()
        diagnostic=json.loads((self.folder/'deployment_setup.json').read_bytes())
        self.assertEqual(diagnostic['stage'],'normal_raw_replay');self.assertFalse(diagnostic['binding_installed'])
        self.assertEqual(diagnostic['reason'],'S2_FRESH_NORMAL_RAW_REPLAY_FAILED')
        self.assertEqual(diagnostic['observed_failure']['replay']['issues'][0]['error'],'original path failure')
        report={};probe.record_setup_diagnostic(self.folder,report)
        self.assertEqual(report['deployment_setup']['receipt'],diagnostic)


class PublicUncertainPrepareContracts(unittest.TestCase):
    def setUp(self):
        from demoncat.runtime.vllm_ascend.worker_extension import identity,runtime
        from demoncat.runtime.vllm_ascend.worker_extension.config import FaultConfig
        from demoncat.runtime.vllm_ascend.worker_extension.host import HostAdapter,WorkerFaultClient
        from test_core_host import ExistingExecutor
        from test_extension import _extended_worker,ReadbackProvider
        self.worker=_extended_worker(rank=0);self.worker.load_model();self.executor=ExistingExecutor([self.worker])
        self.provider=ReadbackProvider()
        self.enterContext(patch.object(identity,'_current_device',return_value=0))
        self.resolve=self.enterContext(patch.object(runtime,'_resolve_provider',return_value=self.provider))
        self.enterContext(patch.object(runtime,'_provider_capabilities',return_value=[dict(mechanism='33.paged_decode',implementation_version='1')]))
        self.binding=Mock(return_value={})
        self.client=WorkerFaultClient(HostAdapter(self.executor,prepare_binding=self.binding),FaultConfig(33,'paged_decode'))
        self.llm=Mock();self.report={}
        self.folder=Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.addCleanup(self.clean)

    def clean(self):
        self.executor.after=None;self.provider.callbacks.clear()
        if self.client.snapshot()['attachment'] is not None:self.client.detach()

    def drive(self):
        probe.drive(self.llm,self.client,prompt={},params=None,out=self.folder,report=self.report)
        self.llm.enqueue.assert_not_called();self.llm.wait_for_completion.assert_not_called()
        self.assertEqual(self.report['requests'],[])

    def test_context_refusal_before_runtime_prepare_detaches_without_ticket_or_requests(self):
        original=ContextRefused('actual guarded setup error');original.response={'reason':'actual guarded setup error'}
        self.binding.side_effect=original;self.drive()
        self.resolve.assert_not_called()
        snapshot=self.client.snapshot()
        self.assertEqual([x['action'] for x in snapshot['history']],['attach','prepare_binding','detach'])
        self.assertEqual(snapshot['history'][1]['response']['observed_response'],original.response)
        self.assertEqual(self.report['status'],'FAILED_BEFORE_PROTOCOL_WITH_ORIGINAL_ERROR_AND_PUBLIC_DETACH')
        self.assertIn('prepare:RPC_UNCERTAIN',self.report['error']);self.assertFalse(snapshot['ownership_retained'])
        actual=vars(self.worker)['_demoncat_worker_fault_runtime']
        self.assertEqual(actual._leases,{});self.assertEqual(actual._tenants,{})
        self.assertEqual(self.worker.events,['load'])

    def lose_prepare_reply(self,*,restore_failure=False):
        def after(method,args,replies):
            if method=='demoncat_fault_rpc' and args[0]=='prepare':
                self.executor.after=None
                self.assertTrue(vars(self.worker)['_demoncat_worker_fault_runtime']._leases)
                if restore_failure:self.provider.callbacks['unregister']=lambda:(_ for _ in ()).throw(RuntimeError('actual stop failure'))
                raise TimeoutError('actual prepare response lost after Worker applied it')
            return replies
        self.executor.after=after

    def test_unknown_applied_prepare_is_recovered_by_actual_runtime_detach(self):
        self.lose_prepare_reply();self.drive()
        snapshot=self.client.snapshot();self.assertFalse(snapshot['ownership_retained'])
        self.assertEqual([x['action'] for x in snapshot['history']],['attach','prepare','detach'])
        recovered=snapshot['history'][-1]['response']['recoveries']
        self.assertEqual(len(recovered),1);self.assertEqual(recovered[0]['action'],'unregister')
        self.assertEqual(vars(self.worker)['_demoncat_worker_fault_runtime']._leases,{})
        self.assertNotIn('query',[x['action'] for x in snapshot['history']])

    def test_uncertain_detach_keeps_owner_and_first_error_then_explicit_retry_can_release(self):
        self.lose_prepare_reply(restore_failure=True);self.drive()
        snapshot=self.client.snapshot()
        self.assertTrue(snapshot['ownership_retained']);self.assertTrue(vars(self.worker)['_demoncat_worker_fault_runtime']._leases)
        self.assertEqual(self.report['status'],'FAILED_WITH_OWNER_OR_RECOVERY_EVIDENCE')
        self.assertIn('prepare:RPC_UNCERTAIN',self.report['error'])
        self.assertIn('detach:RECOVERY_REQUIRED',self.report['cleanup_error'])
        self.assertEqual([x['action'] for x in snapshot['history']],['attach','prepare','detach'])
        self.provider.callbacks.clear()
        self.assertEqual(self.client.detach()['status'],'DETACHED')
        self.assertFalse(self.client.snapshot()['ownership_retained'])


class ColdDiagnosisContracts(unittest.TestCase):
    def test_candidate_diagnosis_imports_no_device_framework_or_library(self):
        code=r'''
import runpy,sys
from pathlib import Path
root=Path(sys.argv[1]);folder=root/'experiments/c2c3_closure_20261117'
sys.path[:0]=[str(root/'src'),str(root)]
def audit(event,args):
    if event=='ctypes.dlopen':raise AssertionError('cold diagnosis loaded a DSO')
    if event=='import' and args[0].split('.')[0] in ('torch','torch_npu','vllm','vllm_ascend'):raise AssertionError(args[0])
sys.addaudithook(audit)
from demoncat.runtime.vllm_ascend.worker_extension import paged33_s2_evidence
runpy.run_path(str(folder/'diagnose_arch22_s2_prepare.py'),run_name='cold_diagnosis')
assert 'ctypes' not in sys.modules
print('CPU-only diagnosis imports')
'''
        value=subprocess.run([sys.executable,'-B','-c',code,str(ROOT)],capture_output=True,text=True,timeout=30)
        self.assertEqual(value.returncode,0,value.stdout+value.stderr)
        self.assertEqual(value.stdout.strip(),'CPU-only diagnosis imports')


if __name__=='__main__':unittest.main()
