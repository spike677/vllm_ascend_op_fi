"""CPU objects and real Python calls only. No NPU proof or original artifacts."""
from copy import deepcopy
import json
import os
from pathlib import Path
import pickle
import subprocess
import sys
import tempfile
import threading
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

import torch
from demoncat.runtime.vllm_ascend.worker_extension.paged33_normal_targets import NormalTargetScope
from demoncat.runtime.vllm_ascend.worker_extension.paged33_logical_binding import LogicalPagedInput
from demoncat.runtime.vllm_ascend.worker_extension.paged33_boundary_probe import FirstDecodeBoundaryCapture
from demoncat.runtime.vllm_ascend.worker_extension import paged33_all_normal as runtime
from test_paged33_logical_binding import Fixture,PATH
from test_paged33_arch22_s2_recipe import source_fixture
from fault_injection.native_adapters.attention28.fia import apply_tnd33_arch22_all_normal as recipe
from fault_injection.native_adapters.attention28.fia import binary_registration,build_fresh
from demoncat.runtime.vllm_ascend.worker_extension.attention_execution import AttentionExecutionAdapter
from demoncat.runtime.vllm_ascend.worker_extension.serving_attention import AttentionDeployment
from demoncat.runtime.vllm_ascend.worker_extension.call_binding import ForwardBinding
from test_core_paged33_s2 import Model,DeviceOperator


def actual_targets():
    fixture=Fixture();fixture.worker.rank=0;model=fixture.worker.model_runner.model
    candidates=[(PATH,fixture.module)]
    for index in range(1,28):
        module=torch.nn.Module();module.layer_name='actual.layer.'+str(index)
        module.impl=NS(**vars(fixture.module.impl));module.kv_cache=fixture.module.kv_cache
        layer=torch.nn.Module();layer.self_attn=torch.nn.Module();layer.self_attn.attn=module
        model.model.layers.append(layer);candidates.append((f'model.layers.{index}.self_attn.attn',module))
    return fixture,candidates


class AllNormalContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.threads=torch.get_num_threads();torch.set_num_threads(1)
    @classmethod
    def tearDownClass(cls):torch.set_num_threads(cls.threads)

    def test_all_28_real_objects_bind_separate_inputs_and_original_returns_once(self):
        f,rows=actual_targets();scope=NormalTargetScope(f.worker,rows,owner=f.owner,generation=7)
        with tempfile.TemporaryDirectory() as folder:
            calls=[];bindings=[]
            for ordinal,(path,module) in enumerate(rows):
                f.module=module;args=f.arguments('request-A');args['module_path']=path;args['writer']['module_path']=path
                args['target_scope']=scope
                # Per-target values differ: copying layer0 evidence is invalid.
                args['actual']['query'].add_(ordinal)
                bound=LogicalPagedInput(**args);bindings.append(bound)
                snapshot=deepcopy(bound.before['boundary']);snapshot.update(module_path=path,module_object_id=id(module))
                capture=FirstDecodeBoundaryCapture(f.worker,Path(folder)/str(ordinal),lambda:None,
                    allow_cpu_test=True,target_scope=scope,target_path=path)
                output=(args['actual']['query']+1,torch.empty(0,dtype=torch.float32))
                def original(**kwargs):calls.append(path);return output
                capture.start('npu_fused_infer_attention_score',original,(),args['actual'],snapshot,module,ordinal)
                returned=bound.execute(lambda:capture.invoke_original(original,(),args['actual']),owner=f.owner,generation=7)
                self.assertIs(returned,output);self.assertEqual(capture.finalize()['original_calls'],1)
                capture.release();bound.release(owner=f.owner,generation=7)
            self.assertEqual(calls,[p for p,_ in rows])
            self.assertEqual(len({b.before['tensors']['query']['sha256'] for b in bindings}),28)
            self.assertTrue(all(b.closed for b in bindings))

    def test_missing_duplicate_foreign_replaced_objects_and_serialized_scope_refused(self):
        f,rows=actual_targets()
        for bad in (rows[:-1],rows[:-1]+[rows[0]],rows[:-1]+[(rows[-1][0],rows[0][1])]):
            with self.subTest(rows=len(bad)),self.assertRaisesRegex(RuntimeError,'28_DISTINCT'):
                NormalTargetScope(f.worker,bad,owner=f.owner,generation=7)
        scope=NormalTargetScope(f.worker,rows,owner=f.owner,generation=7)
        with self.assertRaises(TypeError):pickle.dumps(scope)
        for owner,generation in ((object(),7),(f.owner,8),(f.owner,True)):
            with self.assertRaises(RuntimeError):scope.check(f.worker,PATH,rows[0][1],owner,generation)
        with self.assertRaises(RuntimeError):scope.check(f.worker,rows[1][0],rows[0][1],f.owner,7)
        errors=[]
        def foreign():
            try:scope.report()
            except Exception as error:errors.append(str(error))
        thread=threading.Thread(target=foreign);thread.start();thread.join();self.assertTrue(errors)
        rows[-1][1].impl=NS(**vars(rows[-1][1].impl))
        with self.assertRaisesRegex(RuntimeError,'OBJECT_REPLACED'):scope.report()

    def test_old_default_remains_first_layer_and_report_never_authorizes_another_target(self):
        f,rows=actual_targets();f.module=rows[1][1];args=f.arguments();args['module_path']=rows[1][0]
        args['writer']['module_path']=rows[1][0]
        with self.assertRaisesRegex(RuntimeError,'FIRST_LAYER_SCOPE_ONLY'):LogicalPagedInput(**args)
        scope=NormalTargetScope(f.worker,rows,owner=f.owner,generation=7)
        args['target_scope']=scope.report()
        with self.assertRaisesRegex(RuntimeError,'TRUSTED_NORMAL_TARGET_SCOPE'):LogicalPagedInput(**args)

    def test_host_must_match_current_target_owner_and_both_thread_domains(self):
        f,rows=actual_targets();scope=NormalTargetScope(f.worker,rows,owner=f.owner,generation=7)
        path,module=rows[17];boundary=dict(module_path=path,module_object_id=id(module))
        host=dict(schema='demoncat.paged33.arch22_all_normal_host.v1',mode='arch22_all_normal',phase='off',
            owner='a'*32,generation=7,token='b'*32,target_path=path,module_object_id=id(module),
            target_set_sha256=scope.fingerprint,target_ordinal=17,pid=os.getpid(),tid=threading.get_native_id(),
            normal_only=True,policy_applied=False,native_s2=512,selected_s2=512,actual_observer_scope_matched=True)
        kwargs=dict(owner='a'*32,generation=7,token='b'*32,phase='off')
        runtime.validate_host(host,boundary,scope,**kwargs)
        for key,value in [('owner','c'*32),('token','d'*32),('generation',8),('target_path',PATH),
                ('target_set_sha256','e'*64),('module_object_id',id(rows[0][1])),('target_ordinal',0),
                ('tid',threading.get_native_id()+1),('selected_s2',256),('policy_applied',True),('phase','fault')]:
            with self.subTest(key=key),self.assertRaises(RuntimeError):
                runtime.validate_host(dict(host,**{key:value}),boundary,scope,**kwargs)
        native=runtime.AllNormalNativeCall.__new__(runtime.AllNormalNativeCall);called=[]
        with self.assertRaisesRegex(RuntimeError,'LIVE_NORMAL_BINDING'):
            native.execute({},boundary,'fault',lambda:called.append(1),binding=None,scope=scope,owner=f.owner,generation=7)
        self.assertEqual(called,[])

    def test_fresh_native_recipe_keeps_original_arithmetic_and_has_no_fault_selection(self):
        before=source_fixture();after=recipe.patched(before);normal=recipe.normal.patched(before)
        restored=after[recipe.PATHS[1]].replace(recipe.SELECT,'').replace(recipe.CAPTURE,
            '    paged33_arch22_bridge::Capture(*fiaInfo_, context_, tilingData_, workspaceSize_, sInnerSize_);\n')
        self.assertEqual(restored,normal[recipe.PATHS[1]])
        self.assertEqual(after[recipe.normal.ROUTE],normal[recipe.normal.ROUTE])
        self.assertFalse(any('op_kernel' in name for name in after))
        header=Path(recipe.__file__).with_name('paged33_arch22_all_normal.h').read_text()
        self.assertNotIn('selectedS2=256',header);self.assertNotIn('s.phase=="fault"',header)
        self.assertIn('s.mode=="arch22_all_normal"',header)
        for name in ('DCAT33_TND_TARGET_SET','DCAT33_TND_OBJECT','DCAT33_TND_ORDINAL','dcat_attention_observer_scope_matches_v1'):
            self.assertIn(name,header)
        self.assertEqual(set(recipe.KEYS),binary_registration.KEYS[recipe.VARIANT])
        self.assertNotEqual(recipe.VARIANT,'tnd_page16_arch22_s2_256')

    def test_new_build_namespace_rejects_old_package_and_inherited_authority(self):
        builder=build_fresh.Builder.__new__(build_fresh.Builder);builder.report={}
        builder.receipt=dict(mechanism='33.paged_decode',variant=recipe.VARIANT,
            recipe_sha256=build_fresh.sha(recipe.__file__),normal_route_recipe_sha256=build_fresh.sha(recipe.normal.__file__),
            normal_route_preference_changed=True,selection_before_original_split=True,
            files=[dict(path='attention/common/op_host/arch22/'+name,
                after_sha256=build_fresh.sha(Path(recipe.__file__).with_name(name)))
                for name in ('paged33_arch22_all_normal.h','paged33_arch22_bridge.h')])
        for key in ('tensor_payload_mutated','kernel_source_modified','m_policy_changed','old_normal_certificate_reused',
                    'fault_execution_ready','stock_private_exact_proven','loading_allowed'):builder.receipt[key]=False
        builder.configure_policy(None);self.assertEqual(builder.vendor_name,'dcat_f33_arch22_all_normal')
        for key,value in (('variant','tnd_page16_arch22_s2_256'),('fault_execution_ready',True),('stock_private_exact_proven',True)):
            old=builder.receipt[key];builder.receipt[key]=value
            with self.subTest(key=key),self.assertRaises(RuntimeError):builder.configure_policy(None)
            builder.receipt[key]=old

    def test_reader_and_preparer_import_in_a_cold_stdlib_only_process(self):
        scripts=Path(__file__).resolve().parents[1]/'experiments/c2c3_closure_20261117'
        code="""import sys
sys.path.insert(0,sys.argv[1])
def audit(event,args):
 if event in ('ctypes.dlopen','subprocess.Popen','socket.connect'):raise RuntimeError(event)
sys.addaudithook(audit)
import review_arch22_all_normal,prepare_arch22_all_normal_cases
assert not any(k=='torch' or k.startswith('torch.') or k=='torch_npu' for k in sys.modules)
"""
        result=subprocess.run([sys.executable,'-B','-c',code,str(scripts)],capture_output=True,text=True)
        self.assertEqual(result.returncode,0,result.stderr)

    def test_complete_56_real_api_calls_and_all_target_refs_survive_failed_drain(self):
        f=Fixture();api_count=[]
        def original(**kwargs):api_count.append(1);return kwargs['query']+1,torch.empty(0,dtype=torch.float32)
        api=NS(npu_fused_infer_attention_score=original);model=Model(f,api,f.module.kv_cache[0]).eval()
        worker=NS(rank=0,model_runner=NS(model=model,with_prefill=True));f.worker=worker
        f.module=model.model.layers[0].self_attn.attn
        args=f.arguments('request-A');context=args['forward_context'];context.model_instance=model
        rows=[(f'model.layers.{i}.self_attn.attn',layer.self_attn.attn) for i,layer in enumerate(model.model.layers)]
        for _,module in rows:
            context.attn_metadata[module.layer_name]=args['metadata'];context.no_compile_layers[module.layer_name]=module
        f.context=context
        class FakeNative:
            def __init__(self):self.calls=[];self.fail_stop=False;self.runtime_dir=None
            def execute(self,actual,row,phase,once,*,binding,scope,owner,generation):
                scope.check(worker,row['module_path'],binding.module,owner,generation)
                result=binding.execute(once,owner=owner,generation=generation)
                receipt=dict(call_id=f'{len(self.calls)+1:032x}',simulation=True,phase=phase,module_path=row['module_path'])
                self.calls.append(receipt);return result,receipt
            def stop(self):
                if self.fail_stop:raise RuntimeError('CPU_EXPLICIT_FAILED_DRAIN')
                return dict(scope_owned=False,scope_active=False,environment_owned=0,failures=[],simulation=True,calls=len(self.calls))
        driver=FakeNative()
        with tempfile.TemporaryDirectory() as temp:
            session=runtime.AllNormalWorkerSession.__new__(runtime.AllNormalWorkerSession)
            session.folder=Path(temp)/'phase';session.folder.mkdir();(session.folder/'boundaries').mkdir()
            session.private=True;session.native_phase='off';session.owner_token='a'*32
            deployment=AttentionDeployment(worker,worker.model_runner,model,'33.paged_decode',(1,),driver,'{}',
                os.getpid(),threading.get_ident(),28,True,dict(api_module=api,candidates=rows,processor=model.logits,
                    device_operator=DeviceOperator,context=lambda:context,snapshot=lambda rid,actual,ctx:dict(
                        request_count=1,counts=[1],token_row_count=1,row_token_positions=[1024],request_ids=[rid])),
                {(session.owner_token,1)})
            AttentionExecutionAdapter.__init__(session,deployment,session._guard)
            session.targets=NormalTargetScope(worker,rows,owner=session,generation=1)
            session.logits_binding=ForwardBinding(model.logits);session.records={};session.payload_refs={}
            session.original=None;session.first_cleanup_error=None
            try:
                session.install();session.expect({'request-A':dict(prompt=[1])})
                with patch.object(runtime,'read_scoped_launches',return_value={'simulation':True}),\
                     patch.object(runtime,'mapped_api_libraries',return_value={'simulation':True}),torch.inference_mode():
                    for prefill in (True,False):
                        worker.model_runner.with_prefill=prefill
                        hidden=model(torch.tensor([1]),torch.tensor([1024]));model.logits(hidden)
                result=session.collect()
                self.assertEqual(len(api_count),56);self.assertEqual(len(driver.calls),28)
                self.assertEqual(len(result['all_decode_boundaries']),28)
                self.assertFalse(result['complete_tensor_readback_during_statistics'])
                self.assertEqual(result['audit']['rows'],result['natural_request']['forwards'][1]['calls'])
                sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'experiments/c2c3_closure_20261117'))
                import review_arch22_all_normal
                review_arch22_all_normal.actual_output_join(result)
                self.assertEqual(sum(len(fwd['calls']) for fwd in result['natural_request']['forwards']),56)
                self.assertEqual(len({r['inputs']['query']['snapshot']['sha256'] for r in result['all_decode_boundaries'].values()}),28)
                self.assertTrue(all(r['logical_input_binding']['simulation'] for r in result['all_decode_boundaries'].values()))
                driver.fail_stop=True
                with self.assertRaisesRegex(RuntimeError,'FAILED_DRAIN'):session.close()
                self.assertTrue(session.handles and session.patches)
                self.assertTrue(all(not r['binding'].closed for r in session.records.values()))
                driver.fail_stop=False;owned=api.npu_fused_infer_attention_score
                api.npu_fused_infer_attention_score=lambda **kw:owned(**kw)
                with self.assertRaisesRegex(RuntimeError,'API_OWNER_CHANGED'):session.close()
                self.assertTrue(session.handles);api.npu_fused_infer_attention_score=owned
                cleaned=session.close()
                self.assertTrue(cleaned['logical_bindings_released']);self.assertIn('FAILED_DRAIN',cleaned['first_error'])
                self.assertFalse(session.handles or session.patches);self.assertIs(api.npu_fused_infer_attention_score,original)
                self.assertEqual(original(query=torch.ones(1))[0].item(),2)
            finally:
                driver.fail_stop=False
                if session.handles:session.close()


if __name__=='__main__':unittest.main()
