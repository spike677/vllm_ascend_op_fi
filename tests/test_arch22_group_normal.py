"""CPU-only contracts for the new group package's normal-only preparation."""
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'experiments/c2c3_closure_20261117'))
import arch22_normal_bridge_evidence as E
import attention33_worker_probe as probe
import review_arch22_all_normal as reader
import test_arch22_normal_bridge_review as files
from demoncat.runtime.vllm_ascend.worker_extension.paged33_group_normal import GroupNormalNativeCall,GroupNormalWorkerSession,validate_group_host
from demoncat.runtime.vllm_ascend.worker_extension.paged33_normal_targets import NormalTargetScope
from fault_injection.native_adapters.attention28.fia import apply_tnd33_arch22_group_s2 as recipe,binary_registration
from test_arch22_all_normal import actual_targets
from test_paged33_arch22_s2_recipe import source_fixture


def capture():
    value=dict(shape=[1],dtype='torch.float16',bytes=2,sha256='a'*64,finite=True)
    return dict(hidden=deepcopy(value),logits_hidden=deepcopy(value),logits=deepcopy(value),
        attention={p:{k:deepcopy(value) for k in ('output','key','value')} for p in reader.N.LAYERS})


class GroupNormalContracts(unittest.TestCase):
    def test_recipe_has_separate_namespace_and_original_split_kernel_arithmetic(self):
        before=source_fixture();actual=recipe.patched(before);normal=recipe.all_normal.patched(before)
        self.assertEqual({p:s.replace('paged33_arch22_group_s2','paged33_arch22_all_normal') for p,s in actual.items()},normal)
        self.assertFalse(any('op_kernel' in p for p in actual));self.assertEqual(recipe.SELECTED_SCOPE,[512,256])
        self.assertEqual(binary_registration.KEYS[recipe.VARIANT],set(recipe.KEYS))
        header=(Path(recipe.__file__).parent/recipe.HEADER).read_text()
        for text in ('DCAT33_TND_ADMISSION','DCAT33_TND_COVERAGE_GENERATION','dcat_attention_observer_scope_matches_v1',
                     's.selectedS2=256','s.nativeS2','s.targetSet','s.objectId','s.ordinal'):
            self.assertIn(text,header)
        self.assertNotEqual(recipe.VARIANT,E.ALL_VARIANT);self.assertNotEqual(recipe.VENDOR+'_transformer',E.ALL_VENDOR)

    def test_normal_driver_cannot_issue_fault_or_accept_foreign_actual_host(self):
        driver=GroupNormalNativeCall.__new__(GroupNormalNativeCall)
        self.assertEqual(driver._admission('off',None,None,None,1),('0'*64,0))
        with self.assertRaisesRegex(RuntimeError,'CANNOT_FAULT'):driver._admission('fault',None,None,None,1)
        fixture,rows=actual_targets();scope=NormalTargetScope(fixture.worker,rows,owner=fixture.owner,generation=7)
        path,module=rows[17];boundary=dict(module_path=path,module_object_id=id(module))
        host=dict(schema='demoncat.paged33.arch22_group_s2_host.v1',mode='arch22_group_s2_off',phase='off',owner='a'*32,
            generation=7,token='b'*32,target_path=path,module_object_id=id(module),target_set_sha256=scope.fingerprint,
            target_ordinal=17,pid=os.getpid(),tid=threading.get_native_id(),normal_only=True,policy_applied=False,
            native_s2=512,selected_s2=512,actual_observer_scope_matched=True,eligible=True,route='FiaTilingNonQuant',
            normal_route_preference_changed=True,admission_sha256='0'*64,coverage_generation=0)
        kwargs=dict(owner='a'*32,generation=7,token='b'*32,phase='off',admission='0'*64,coverage_generation=0)
        validate_group_host(host,boundary,scope,**kwargs)
        for key,value in (('owner','d'*32),('tid',host['tid']+1),('generation',8),('target_path',rows[0][0]),
            ('module_object_id',id(rows[0][1])),('admission_sha256','d'*64),('coverage_generation',1),('selected_s2',256)):
            with self.subTest(key=key),self.assertRaises(RuntimeError):validate_group_host(dict(host,**{key:value}),boundary,scope,**kwargs)

    def test_normal_rpc_returns_reference_after_actual_cleanup_and_retains_failed_owner(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp);value=capture();(folder/'runtime_result.json').write_text(json.dumps(value))
            session=GroupNormalWorkerSession.__new__(GroupNormalWorkerSession);session.folder=folder
            session.collect=lambda:value;session.close=lambda:dict(owned_hooks=0);worker=NS()
            setattr(worker,probe.OWNER,session)
            with patch.object(probe,'worker_object',side_effect=lambda worker:worker):result=probe.finish(worker,True)
            self.assertTrue(result['closed']);self.assertNotIn('capture',result);self.assertFalse(hasattr(worker,probe.OWNER))
            ref=result['capture_artifact'];self.assertEqual(ref['sha256'],E.sha(folder/'runtime_result.json'))
            def failed():raise RuntimeError('CPU_DRAIN_UNCONFIRMED')
            session.close=failed;setattr(worker,probe.OWNER,session)
            with patch.object(probe,'worker_object',side_effect=lambda worker:worker):result=probe.finish(worker,True)
            self.assertFalse(result['closed']);self.assertIs(getattr(worker,probe.OWNER),session)
            self.assertIn('CPU_DRAIN_UNCONFIRMED',result['cleanup_error'])

    def test_normal_artifact_reference_is_bounded_hashed_and_phase_specific(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);folder=root/'baseline_before';folder.mkdir();value=capture()
            path=folder/'runtime_result.json';path.write_text(json.dumps(value))
            session=GroupNormalWorkerSession.__new__(GroupNormalWorkerSession);session.folder=folder
            ref=session.capture_artifact(value);ref['path']='/evidence/run/baseline_before/runtime_result.json'
            phase=dict(phase='baseline_before',finish=dict(closed=True,capture_artifact=ref))
            self.assertEqual(reader.phase_capture(root,phase,group=True),value)
            for key,item in (('sha256','c'*64),('bytes',128*1024**2+1),('path','/evidence/run/off_after/runtime_result.json')):
                bad=deepcopy(phase);bad['finish']['capture_artifact'][key]=item
                with self.subTest(key=key),self.assertRaises(ValueError):reader.phase_capture(root,bad,group=True)
            bad=deepcopy(phase);bad['finish']['capture_artifact']['payload_identity']['hidden']['sha256']='b'*64
            with self.assertRaisesRegex(ValueError,'PAYLOAD_REFERENCE_DIFFERS'):reader.phase_capture(root,bad,group=True)
            with self.assertRaisesRegex(ValueError,'INCOMPLETE_NORMAL_PHASE:healthy_after.finish.capture_artifact'):
                reader.phase_capture(root,dict(phase='healthy_after'),group=True)

    def test_index_cost_receipt_must_match_original_prefix_owner_and_real_phase_time(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'runtime').mkdir();(root/'baseline_before').mkdir();(root/'baseline_before/runtime_result.json').write_text('{}')
            rows=[dict(event='begin',call_id='a'*32),dict(event='end',call_id='a'*32)]
            raw=b''.join((json.dumps(row)+'\n').encode() for row in rows);(root/'runtime/runtime.jsonl').write_bytes(raw)
            index=dict(schema='demoncat.attention.owned_runtime_index.v1',owner=[1,2,3],prefix_bytes=len(raw),
                prefix_sha256=hashlib.sha256(raw).hexdigest(),parsed_events=2,refreshes=28,validation_seconds=.5,
                complete_old_prefix_rehashed=True,raw_log_modified=False,qualification_granted=False)
            value=dict(target_scope=dict(pid=1,python_thread_ident=2,native_tid=3),runtime_validation_cost=dict(index=index,
                initial_runtime_prefix_bytes=0,initial_runtime_prefix_sha256=hashlib.sha256(b'').hexdigest(),owned_files_excluding_capture=[],
                indexed_history_validation_seconds=.5,raw_log_kept_complete=True,all_prior_registration_conflicts_retained=True))
            phase=dict(phase='baseline_before',elapsed_seconds=2.)
            self.assertEqual(reader.indexed_cost(value,phase,rows,root)['prefix_bytes'],len(raw))
            for key,item in (('owner',[1,3,2]),('prefix_sha256','b'*64),('parsed_events',1),('validation_seconds',3.)):
                bad=deepcopy(value);bad['runtime_validation_cost']['index'][key]=item
                with self.subTest(key=key),self.assertRaises(ValueError):reader.indexed_cost(bad,phase,rows,root)

    def test_preparer_only_queues_same_new_package_stock_and_private_normal(self):
        source=files.source();source['variant']=E.GROUP_VARIANT
        args=NS(card=5,timeout=1800,source=Path('.'),installed=Path('.'),model=Path('.'),build_root=Path('.'))
        bundle=dict(completion_sha256='a'*64,completion={'arch22_public_schema_sha256':'b'*64},summary={'variant':E.GROUP_VARIANT})
        observer=dict(sha256='c'*64,folder=Path('.'),alias='/build/attention-preflight',receipt={'observer_library':{'sha256':'d'*64}})
        cases=files.prepare.make_cases(args,source,bundle,observer,{},{});self.assertEqual(len(cases),2)
        for case in cases:
            self.assertIn('--arch22-group-normal-bridge',case['command']);self.assertNotIn('--phase fault',case['command'])
            self.assertFalse(case['review']['fault_authorized'] or case['review']['c2'] or case['review']['c3'])
        self.assertIn(E.GROUP_VENDOR,cases[1]['command'])

    def test_normal_preparer_and_reviewer_cold_import_do_not_import_torch(self):
        code="import runpy,sys; sys.path.insert(0,sys.argv[1]); runpy.run_path(sys.argv[2]); assert 'torch' not in sys.modules"
        for script in ('prepare_arch22_group_normal_cases.py','review_arch22_group_normal.py'):
            subprocess.run([sys.executable,'-B','-c',code,str(ROOT/'experiments/c2c3_closure_20261117'),
                str(ROOT/'experiments/c2c3_closure_20261117'/script)],check=True,capture_output=True)


if __name__=='__main__':unittest.main()
