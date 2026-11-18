"""CPU contracts for the new normal-only package and shared real-call probe."""
import copy
import json
import os
from pathlib import Path,PurePosixPath
import subprocess
import sys
import tempfile
import threading
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

import torch
from demoncat.runtime.vllm_ascend.worker_extension import paged33_arch22_s2_normal as runtime
from demoncat.runtime.vllm_ascend.worker_extension import paged33_boundary_probe as capture
from demoncat.runtime.vllm_ascend.worker_extension import paged33_tnd_probe as probe
from demoncat.runtime.vllm_ascend.worker_extension.paged33_logical_binding import LogicalPagedInput
from demoncat.runtime.vllm_ascend.worker_extension.paged33_audit import audit_paged_call
from test_paged33_logical_binding import Fixture,PATH
from test_paged33_arch22_s2_recipe import split_fixture
import test_paged33_arch22_bridge as normal_cases
import test_arch22_normal_bridge_review as files
from fault_injection.native_adapters.attention28.fia import arch22_s2_build_evidence as s2_evidence
from fault_injection.native_adapters.attention28.fia import arch22_s2_geometry as geometry
from fault_injection.native_adapters.attention28.fia import compile_arch22_s2_matrix as matrix_driver

evidence,review,prepare=files.evidence,files.review,files.prepare


class FakeNative:
    """Explicit CPU seam: no library, scope or runtime launch is simulated as real."""
    def __init__(self):
        self.calls=[];self.runtime_dir=None;self.library_coexistence={};self.stop_error=False
    def execute(self,actual,boundary,phase,invoke_once,*,binding,owner,generation):
        result=binding.execute(invoke_once,owner=owner,generation=generation)
        self.calls.append(dict(call_id='a'*32,module_path=PATH,phase=phase))
        return result,self.calls[-1]
    def stop(self):
        if self.stop_error:raise RuntimeError('CPU seam: uncertain stop')
        return dict(scope_owned=False,environment_owned=0,scope_active=False,first_error=None,failures=[])


def adapter(native):
    with patch.object(runtime,'Arch22S2NormalNativeCall',return_value=native):
        return runtime.S2NormalProbeAdapter(arm='baseline',experimental_qualification=True,max_calls=28)


class S2NormalContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.threads=torch.get_num_threads();torch.set_num_threads(1)
    @classmethod
    def tearDownClass(cls):torch.set_num_threads(cls.threads)

    def test_fault_and_old_proof_never_reach_native_or_original(self):
        calls=[]
        native=runtime.Arch22S2NormalNativeCall.__new__(runtime.Arch22S2NormalNativeCall)
        with self.assertRaisesRegex(RuntimeError,'NEVER_AUTHORIZES_FAULT'):
            native.execute({}, {}, 'fault',lambda:calls.append(1),binding=None,owner=None,generation=1)
        with patch.object(runtime,'Arch22S2NormalNativeCall') as constructed:
            with self.assertRaises(RuntimeError):runtime.S2NormalProbeAdapter(arm='fault',experimental_qualification=True)
            constructed.assert_not_called()
        self.assertEqual(calls,[])

    def test_final_normal_pod_matches_full_real_split_arrays_and_owned_host_scope(self):
        actual,boundary,host,launch=normal_cases.Arch22NormalContracts().native_fixture()
        row=split_fixture();checked=geometry.validate_split(row);typed=launch['typed']
        typed.update({'baseParams.s2Size':1040,'baseParams.usedCoreNum':16,'innerSplitParams.mBaseSize':512,
            'fdParams.numOfFdHead':8,'fdParams.usedVecNumOfFd':8,'fdParams.gS1BaseSizeOfFd':8})
        for name in geometry.ARRAYS:
            prefix='outerSplitParams.' if name in ('bN2End','gS1End','s2End') else 'fdParams.'
            width=52 if name in geometry.ARRAYS[-2:] else 26
            typed[prefix+name]=row[name]+[0]*(width-len(row[name]))
        typed.update({'workspaceParams.'+k:v for k,v in checked['workspace_elements'].items()})
        host.update(schema='demoncat.paged33.arch22_s2_host.v1',mode='arch22_s2_off',phase='baseline',
            target_path=PATH,owner='b'*32,generation=1,token='a'*32,pid=os.getpid(),tid=threading.get_native_id(),
            actual_observer_scope_matched=True,aic=20,aiv=40,actual_kv_length=1025,real_s2_capacity=1040,
            s2_loops=3,s2_tail=16,s2_aligned=512,block_dim=16,libapi_workspace=16*1024**2,
            workspace=16*1024**2+checked['workspace_excluding_libapi'])
        kwargs=dict(token='a'*32,owner='b'*32,generation=1,phase='baseline',rows={20:row})
        self.assertTrue(runtime.validate_owned_normal(launch,host,actual,boundary,**kwargs)['normal_only'])
        for key,value in (('owner','c'*32),('generation',True),('phase','fault'),('target_path',PATH+'x'),
                          ('actual_observer_scope_matched',False),('workspace',0),('s2_tail',15),('aiv',20)):
            bad=dict(host);bad[key]=value
            with self.subTest(key=key),self.assertRaises((RuntimeError,ValueError)):
                runtime.validate_owned_normal(launch,bad,actual,boundary,**kwargs)
        for name in geometry.ARRAYS:
            bad=copy.deepcopy(launch);prefix='outerSplitParams.' if name in ('bN2End','gS1End','s2End') else 'fdParams.'
            bad['typed'][prefix+name][-1]=1
            with self.subTest(array=name),self.assertRaisesRegex(ValueError,'ARRAY_DIFFERS'):
                runtime.validate_owned_normal(bad,host,actual,boundary,**kwargs)

    def test_readonly_build_parser_refuses_duplicate_and_nonfinite_data(self):
        for raw in ('{"a":1,"a":1}','{"a":NaN}'):
            with self.assertRaises(ValueError):s2_evidence.parse(raw)

    def test_matrix_source_pins_are_read_from_the_frozen_recipe_without_executing_it(self):
        digest=s2_evidence.sha(runtime.recipe.__file__)
        pins=s2_evidence.pristine_split_pins(digest)
        self.assertEqual(pins,{name:runtime.recipe.SOURCE_SHA[name] for name in pins})
        self.assertEqual(set(pins),{'attention/common/op_host/split_core.h','attention/common/op_host/split_core.cpp'})
        with self.assertRaisesRegex(ValueError,'RECIPE_SOURCE_DIFFERS'):s2_evidence.pristine_split_pins('0'*64)

    def test_missing_actual_matrix_rows_cannot_be_replaced_by_receipt_pass_flags(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);ev=root/'evidence/arch22_s2_matrix';(ev/'log').mkdir(parents=True)
            pins={s2_evidence.PREFIX+name:s2_evidence.sha(Path(runtime.recipe.__file__).with_name(name)) for name in
                ('apply_tnd33_arch22_s2.py','apply_tnd33_arch22_bridge.py','paged33_arch22_s2.h',
                 'paged33_arch22_bridge.h','arch22_s2_geometry.py','compile_arch22_s2_matrix.py')}
            source_pins={}
            # Explicit synthetic source files: the actual frozen-source check
            # is independently exercised above and is only replaced here to
            # reach the incomplete-row gate. No compiler or Native is run.
            for name in ('split_core.h','split_core.cpp'):
                p=root/'source/attention/common/op_host'/name;p.parent.mkdir(parents=True,exist_ok=True)
                p.write_bytes(('CPU fixture only: '+name).encode());source_pins[p.relative_to(root/'source').as_posix()]=s2_evidence.sha(p)
            lines='\n'.join('array("'+n+'",r.'+('' if n in ('bN2End','gS1End','s2End') else 'fdRes.')+n+');' for n in geometry.ARRAYS)
            (ev/'matrix.cpp').write_text(matrix_driver.PROGRAM.replace('ARRAY_LINES',lines))
            (ev/'log/log.h').write_text('#pragma once\n#define OP_LOGI(...) ((void)0)\n#define OP_LOGD(...) ((void)0)\n')
            (ev/'split_matrix.jsonl').write_text(json.dumps(split_fixture())+'\n')
            receipt=dict(schema='demoncat.paged33.arch22_s2_cpu_matrix.v1',cases=22528,
                source_sha256=source_pins,source_execution='PINNED_ORIGINAL_SPLIT_CORE_IN_CPU_SUBPROCESS',
                npu_executed=False,qualification_granted=False,actual_worker_platform_observed=False,
                complete_split_and_fd_arrays_checked=True,program_sha256=s2_evidence.sha(ev/'matrix.cpp'),
                logging_only_shim_sha256=s2_evidence.sha(ev/'log/log.h'),matrix_sha256=s2_evidence.sha(ev/'split_matrix.jsonl'),
                validator_sha256=pins[s2_evidence.PREFIX+'arch22_s2_geometry.py'])
            (ev/'receipt.json').write_text(json.dumps(receipt))
            data=dict(variant=evidence.S2_VARIANT,arch22_s2_recipe_sha256=pins[s2_evidence.PREFIX+'apply_tnd33_arch22_s2.py'],
                arch22_s2_normal_route_recipe_sha256=pins[s2_evidence.PREFIX+'apply_tnd33_arch22_bridge.py'],
                arch22_s2_headers={name:pins[s2_evidence.PREFIX+name] for name in ('paged33_arch22_s2.h','paged33_arch22_bridge.h')},
                old_normal_certificate_reused=False,fault_execution_ready=False,stock_private_exact_proven=False,
                arch22_s2_cpu_receipt_sha256=s2_evidence.sha(ev/'receipt.json'),arch22_s2_cpu_matrix_sha256=receipt['matrix_sha256'],
                arch22_s2_geometry_validator_sha256=receipt['validator_sha256'])
            with self.assertRaisesRegex(ValueError,'ORIGINAL_SPLIT_SOURCE_BYTES_CHANGED'):
                s2_evidence.verify_inputs(root,data,pins)
            with patch.object(s2_evidence,'pristine_split_pins',return_value=source_pins):
                with self.assertRaisesRegex(ValueError,'MATRIX_INCOMPLETE'):s2_evidence.verify_inputs(root,data,pins)
                (ev/'matrix.cpp').write_text('substitute successful program')
                with self.assertRaisesRegex(ValueError,'PROGRAM_OR_LOGGING_ONLY_SHIM_CHANGED'):
                    s2_evidence.verify_inputs(root,data,pins)

    def test_reader_requires_first_layer_count_all_other_returns_and_zero_references(self):
        cleanup=dict(scope_cleared=True,local_streams_drained=True,owned_hooks=0,owned_api_wrappers=0,
            boundary_refs_released=True,core_methods_replaced=False,failures=[],scope_active=False,scope_owned=False,
            environment_owned=0,first_error=None,calls=1,simulation=False,logical_binding_released=True,
            stock_launch_scope=dict(scope_cleared=True,local_streams_drained=True,errors=[]),
            unscoped_original_calls=[dict(module_path=p,original_calls=1,returned=True) for p in review.LAYERS if p!=PATH])
        finish=dict(closed=True,cleanup=cleanup)
        for private in (False,True):self.assertTrue(review.check_cleanup(finish,private,first_layer_only=True)['scope_cleared'])
        for key,value in (('logical_binding_released',False),('scope_owned',True),('calls',28),('first_error','retained'),
                          ('unscoped_original_calls',cleanup['unscoped_original_calls'][:-1])):
            bad=copy.deepcopy(finish);bad['cleanup'][key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):review.check_cleanup(bad,True,first_layer_only=True)
        bad=copy.deepcopy(finish);bad['cleanup']['unscoped_original_calls'][0]['returned']=False
        with self.assertRaisesRegex(ValueError,'ORIGINAL_CALL_CENSUS'):review.check_cleanup(bad,True,first_layer_only=True)

    def test_shared_probe_preserves_actual_first_call_in_both_arms_and_keeps_live_refs(self):
        for private in (False,True):
            with self.subTest(private=private),tempfile.TemporaryDirectory() as folder:
                f=Fixture();f.worker.rank=0;args=f.arguments();actual=args['actual'];calls=[]
                boundary=audit_paged_call('npu_fused_infer_attention_score',actual,impl=f.module.impl,
                    metadata=args['metadata'],scheduler=args['scheduler'],writer=args['writer'],allow_cpu_test=True)
                boundary.update(module_path=PATH,module_object_id=id(f.module),impl_object_id=id(f.module.impl),virtual_engine=0)
                session=probe.TNDWorkerProbeSession.__new__(probe.TNDWorkerProbeSession)
                session.worker=f.worker;session.model=f.worker.model_runner.model;session.s2_normal=True
                session.stack=[(PATH,f.module)];session.current=args['scheduler'];session.rows=[boundary]
                session.writers={id(f.module):args['writer']};session.tensor_refs={}
                session.boundary_capture=capture.FirstDecodeBoundaryCapture(f.worker,Path(folder)/'capture',lambda:None,allow_cpu_test=True)
                native=FakeNative();native.runtime_dir=Path(folder)
                session.adapter=adapter(native) if private else None
                session.logical_owner=session.adapter.owner if private else object()
                session.boundary_stock_scope=None if private else NS(token='b'*32,runtime_dir=Path(folder),invoke=lambda fn:fn())
                returned=(actual['query']+1,torch.empty(0,dtype=torch.float32))
                def original(*pos,**kw):
                    calls.append(1);self.assertFalse(pos)
                    self.assertEqual(set(kw),set(actual))
                    self.assertTrue(all(kw[k] is actual[k] for k in kw))
                    return returned
                forward=NS(get_forward_context=lambda:args['forward_context'])
                def cpu_binding(**kw):return LogicalPagedInput(**kw,allow_cpu_test=True)
                with patch.dict(sys.modules,{'vllm.forward_context':forward}), \
                     patch('demoncat.runtime.vllm_ascend.worker_extension.paged33_logical_binding.LogicalPagedInput',side_effect=cpu_binding), \
                     patch.object(capture,'mapped_api_libraries',return_value={}), \
                     patch.object(capture,'read_scoped_launches',return_value={}):
                    value=session._invoke_api('npu_fused_infer_attention_score',original,(),actual,boundary)
                self.assertIs(value,returned);self.assertEqual(calls,[1])
                report=session.boundary_capture.data['logical_input_binding']
                self.assertTrue(report['complete']);self.assertEqual(report['actual_original_calls'],1)
                self.assertTrue(report['simulation']);self.assertFalse(report['qualification_granted'])
                with self.assertRaisesRegex(ValueError,'OWNER_OR_ACTUAL_OBJECT'):
                    review.check_live_input(session.boundary_capture.data,boundary)
                # Test only the structural join with an explicitly synthetic
                # production-shaped projection. The real CPU report remains
                # simulation=True and the full reviewer cannot accept it.
                projected=copy.deepcopy(session.boundary_capture.data)
                projected['logical_input_binding']['simulation']=False
                self.assertTrue(review.check_live_input(projected,boundary)['complete_logical_kv_joined'])
                changed=copy.deepcopy(projected)
                changed['logical_input_binding']['evidence']['before']['tensors']['query']['sha256']='0'*64
                with self.assertRaises(ValueError):review.check_live_input(changed,boundary)
                self.assertIs(session.logical_binding.worker,f.worker)
                if private:
                    self.assertEqual(len(native.calls),1)
                    session.adapter.disable();self.assertTrue(session.logical_binding.closed)
                else:session.logical_binding.release(owner=session.logical_owner,generation=1)

    def test_uncertain_native_stop_retains_binding_then_retry_releases_only_owned_refs(self):
        f=Fixture();native=FakeNative();a=adapter(native);args=f.arguments();args['owner']=a.owner;args['generation']=1
        binding=LogicalPagedInput(**args);a.bind_input(binding)
        boundary=dict(module_path=PATH);result=(args['actual']['query'].clone(),torch.empty(0))
        self.assertIs(a.execute(dict(args['actual']),boundary,lambda:result),result)
        native.stop_error=True
        with self.assertRaisesRegex(RuntimeError,'uncertain stop'):a.disable()
        self.assertFalse(binding.closed);self.assertIs(binding.worker,f.worker)
        native.stop_error=False
        state=a.disable();self.assertTrue(state['logical_binding_released']);self.assertIsNone(binding.worker)
        self.assertIn('uncertain stop',state['first_error'])
        self.assertFalse(state['c1'] or state['c2'] or state['c3'])

    def test_actual_probe_finish_retains_worker_owner_and_real_hook_until_stop_retry(self):
        import attention33_worker_probe as driver
        f=Fixture();native=FakeNative();a=adapter(native);args=f.arguments();args['owner']=a.owner;args['generation']=1
        binding=LogicalPagedInput(**args);a.bind_input(binding)
        session=probe.TNDWorkerProbeSession.__new__(probe.TNDWorkerProbeSession)
        session.thread=threading.get_ident();session.current=None;session.patches=[];session.installed=True
        hook=f.worker.model_runner.model.register_forward_hook(lambda *args:None)
        session.handles=[hook];session.adapter=a;session.stop_capture=lambda:None;session.boundary_stock_scope=None
        session.boundary_capture=None;session.logical_binding=binding;session.logical_owner=a.owner
        setattr(f.worker,driver.OWNER,session);native.stop_error=True
        with patch.object(driver,'worker_object',side_effect=lambda worker:worker):
            first=driver.finish(f.worker,False)
            self.assertFalse(first['closed']);self.assertIn('uncertain stop',first['cleanup_error'])
            self.assertIs(getattr(f.worker,driver.OWNER),session)
            self.assertFalse(binding.closed);self.assertIn(hook.id,f.worker.model_runner.model._forward_hooks)
            native.stop_error=False
            second=driver.finish(f.worker,False)
        self.assertTrue(second['closed']);self.assertFalse(hasattr(f.worker,driver.OWNER))
        self.assertNotIn(hook.id,f.worker.model_runner.model._forward_hooks)
        self.assertTrue(binding.closed);self.assertIn('uncertain stop',second['cleanup']['first_error'])
        self.assertEqual(second['cleanup']['owned_hooks'],0)

    def test_a_returned_stop_with_nonzero_actual_scope_never_releases_input(self):
        f=Fixture();native=FakeNative();a=adapter(native);args=f.arguments();args['owner']=a.owner;args['generation']=1
        binding=LogicalPagedInput(**args);a.bind_input(binding)
        with patch.object(native,'stop',return_value=dict(scope_owned=True,scope_active=True,environment_owned=0)):
            with self.assertRaisesRegex(RuntimeError,'STOP_NOT_CONFIRMED'):a.disable()
        self.assertFalse(binding.closed);state=a.disable()
        self.assertTrue(binding.closed);self.assertIn('STOP_NOT_CONFIRMED',state['first_error'])

    def test_foreign_cleanup_and_duplicate_first_layer_rejected_but_other_layers_original_once(self):
        f=Fixture();native=FakeNative();a=adapter(native);args=f.arguments();args['owner']=a.owner;args['generation']=1
        binding=LogicalPagedInput(**args);a.bind_input(binding);result=(args['actual']['query'].clone(),torch.empty(0))
        a.execute(args['actual'],dict(module_path=PATH),lambda:result)
        with self.assertRaisesRegex(RuntimeError,'EXACT_ONE'):a.execute(args['actual'],dict(module_path=PATH),lambda:result)
        calls=[];other=dict(module_path='model.layers.1.self_attn.attn',module_object_id=45,scheduler={'request_ids':['A']})
        self.assertIs(a.execute({},other,lambda:(calls.append(1),result)[1]),result);self.assertEqual(calls,[1])
        self.assertEqual(len(native.calls),1)
        errors=[]
        def foreign():
            try:a.disable()
            except RuntimeError as e:errors.append(str(e))
        t=threading.Thread(target=foreign);t.start();t.join();self.assertEqual(len(errors),1)
        self.assertFalse(binding.closed);a.disable()

    def test_preparer_profiles_are_distinct_without_changing_capture_or_fault_authority(self):
        source=files.source();source.update(variant=evidence.S2_VARIANT)
        source['pins'].update({name:evidence.sha(files.ROOT/name) for name in evidence.S2_REQUIRED})
        args=NS(card=5,timeout=1800,source=PurePosixPath('/source'),installed=PurePosixPath('/installed'),
                model=PurePosixPath('/model'),build_root=PurePosixPath('/new-build'))
        observer=dict(folder=PurePosixPath('/new-observer'),alias='/output/cann851/attention-preflight',sha256='a'*64,
                      receipt={'observer_library':{'sha256':'b'*64}})
        bundle=dict(completion_sha256='c'*64,completion={'arch22_public_schema_sha256':'d'*64},summary={'variant':evidence.S2_VARIANT})
        cases=prepare.make_cases(args,source,bundle,observer,{'same':'contract'}, {})
        self.assertEqual(len(cases),2)
        self.assertEqual(cases[0]['env'],cases[1]['env'])
        for case in cases:
            self.assertIn('--arch22-s2-normal-bridge',case['command'])
            self.assertNotIn('--arch22-normal-bridge',case['command'])
            self.assertFalse(case['review']['fault_authorized'])
        self.assertNotIn('--vendor',cases[0]['command'])
        self.assertIn(evidence.S2_VENDOR,cases[1]['command'])
        self.assertIn('only layer0',cases[1]['review']['expected_hits'])
        from queue_contract import validate_cases
        self.assertEqual(validate_cases(cases,[5]),cases)

    def test_new_preparer_imports_in_cold_stdlib_only_process(self):
        script=files.EXPERIMENT/'prepare_arch22_s2_normal_bridge_cases.py'
        code='''import runpy,sys
from pathlib import Path
sys.path.insert(0,sys.argv[1])
import arch22_normal_bridge_evidence as e
sys.addaudithook(e.prohibit_runtime)
root=Path(sys.argv[1]).parents[1]
source=dict(root=root,variant=e.S2_VARIANT,pins={name:e.sha(root/name) for name in e.REQUIRED+e.S2_REQUIRED})
helper=e.helpers(source)
assert helper.arch22_s2_build_evidence.pristine_split_pins(source['pins'][e.NATIVE+'fia/apply_tnd33_arch22_s2.py'])
assert not any(name.split('.')[0] in ('torch','torch_npu') for name in sys.modules)
sys.argv=[sys.argv[2],"--help"]
runpy.run_path(sys.argv[0],run_name="__main__")
'''
        result=subprocess.run([sys.executable,'-B','-c',code,str(files.EXPERIMENT),str(script)],capture_output=True,text=True,timeout=30)
        self.assertEqual(result.returncode,0,result.stderr)
        self.assertIn('--build-root',result.stdout)


if __name__=='__main__':unittest.main()
