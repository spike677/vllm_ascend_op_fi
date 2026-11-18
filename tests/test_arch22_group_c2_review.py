"""Independent reader contracts with constructed CPU evidence, never NPU claims."""
from copy import deepcopy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import struct
import sys
import tempfile
import threading
import unittest

ROOT=Path(__file__).resolve().parents[1]
EXPERIMENT=ROOT/'experiments/c2c3_closure_20261117'
sys.path.insert(0,str(EXPERIMENT))
import review_arch22_group_c2 as G
from fault_injection.native_adapters.attention28 import group_resource_budget as R
from test_attention_group_resource_budget import observations


def resources_fixture(folder):
    runtime=folder/'runtime.jsonl';lines=[json.dumps({'sequence':n}).encode()+b'\n' for n in range(120)]
    runtime.write_bytes(b''.join(lines));thread=[os.getpid(),threading.get_ident(),threading.get_native_id()]
    context=dict(provider_owner='a'*32,generation=7,normal_evidence_sha256='c'*64,
        thread_identity=dict(zip(('pid','python_thread_ident','native_tid'),thread)))
    source={'manifest_sha256':'a'*64};inputs=observations();plan=R.plan(inputs,variant=R.VARIANT,
        source_sha256='a'*64,build_sha256='b'*64)
    scope=dict(variant=R.VARIANT,source_manifest_sha256='a'*64,completion_sha256='b'*64,
        normal_evidence_sha256='c'*64,attempt_root='/evidence/probe',owner=thread,
        maximum_runtime_bytes=R.GROUP_LOG_BYTES,qualification_granted=False)
    def prefix(count):
        raw=b''.join(lines[:count])
        return dict(prefix_bytes=len(raw),prefix_sha256=hashlib.sha256(raw).hexdigest(),parsed_events=count,
            file_identity=[1,2],owner=thread,group_resource_scope=scope,maximum_runtime_bytes=R.GROUP_LOG_BYTES,
            complete_old_prefix_rehashed=True,raw_log_modified=False)
    phases=[dict(phase=phase,phase_index=i,started_monotonic=200.+2*i,finished_monotonic=201.+2*i,
        start=prefix(max(1,2*i)),end=prefix(2*i+2)) for i,phase in enumerate(G.PHASES)]
    rows=[];admissions=[]
    for target in range(28):
        index=3+2*target;proof={}
        for stage in ('before_request','before_native'):
            count=2*index+(stage=='before_native');now=200.+2*index+(.2 if stage=='before_native' else -.5)
            resource=dict(pid=thread[0],python_thread_ident=thread[1],native_tid=thread[2],sampled_monotonic=now,
                disk_root='/evidence/probe',free_disk_bytes=100*1024**3,available_memory_bytes=100*1024**3,
                cgroup_memory=[dict(limit_bytes=200*1024**3,current_bytes=100*1024**3)])
            original=prefix(count)
            check=R.before_fault(plan,phase_index=index,started_monotonic=100.,runtime_bytes=original['prefix_bytes'],resources=resource,now=now)
            row=dict(phase=f'fault_{target:02d}',phase_index=index,stage=stage,owner='a'*32,public_generation=7,
                scope=scope,plan_sha256=G.E.digest(plan),passed=True,runtime_index=original,check=check,
                started_monotonic=100.,checked_monotonic=now,completed_phase_resources_sha256=G.E.digest(phases[:index]))
            digest=G.E.digest(row);rows.append(dict(receipt=row,sha256=digest));proof[stage+'_resource_sha256']=digest
        admissions.append(dict(proof=proof))
    saved=dict(resource_plan=plan,resource_checks=rows,phase_resources=phases,admissions=admissions)
    return saved,context,inputs,runtime,source,'b'*64,R


def actual_cpu_ledger(control,path):
    """Use original CPU Host requests/receipts, not a hand-authored hit ledger."""
    saved=json.loads(path.read_bytes());controls=[];actual=[];artifacts={}
    def add(value,label):artifacts.setdefault(G.E.digest(value),[]).append(dict(path=label))
    def signature(row):return {k:row[k] for k in ('shape','dtype','bytes','sha256')}
    for index,ref in enumerate(control.records):
        capture=json.loads(Path(ref['path']).read_bytes());request=capture['natural_request']
        calls=[c for forward in request['forwards'] for c in forward['calls']]
        payloads={name:capture[name] for name in ('hidden','logits_hidden','logits')}
        for target,value in capture['attention'].items():payloads.update({target+'/'+kind:row for kind,row in value.items()})
        if index<3:
            controls.append(dict(scope=capture['target_scope'],calls=calls,request_id=request['request_id'],
                live={p:b['logical_input_binding'] for p,b in capture['all_decode_boundaries'].items()},
                payloads=payloads,capture_sha256=G.E.digest(capture)))
            add(capture,ref['path'])
        phase=request['phase']
        if phase.startswith(('fault_','off_')):
            selected=int(phase.split('_')[1]) if phase.startswith('fault_') else None
            for ordinal,call in enumerate(calls):
                injecting=ordinal==28+selected if selected is not None else False
                executed=call['native'] if injecting else call;add(executed,ref['path']);effect=None
                if injecting:
                    name=call['module_path']+'/output'
                    effect=dict(before=signature(controls[0]['payloads'][name]),after=signature(payloads[name]),artifact=ref)
                    add(effect,ref['path'])
                actual.append(dict(call=call,phase=phase,request_id=request['request_id'],selected=injecting,
                    artifact=ref,execution_sha256=G.E.digest(executed),effect_sha256=G.E.digest(effect) if effect else None))
        add(ref,ref['path'])
    add(control.records,'all requests')
    context=dict(provider_owner=control.owner,generation=control.public_generation,worker=control.identity,
        normal_evidence_sha256=control.normal.fingerprint,simulation=True)
    spec=importlib.util.spec_from_file_location('group_test_independent_auditor',ROOT/'fault_injection/coverage_audit.py')
    auditor=importlib.util.module_from_spec(spec);spec.loader.exec_module(auditor)
    return (saved,context,actual,controls,control.records,artifacts,{'pins':control.normal.data()['source_pins']},
        list(control.adapter.deployment.prompt),control.normal.data()['execution_contract'],auditor)


class GroupRawReviewContracts(unittest.TestCase):
    def test_all_56_resource_guards_join_original_prefixes_and_admissions(self):
        with tempfile.TemporaryDirectory() as tmp:
            args=resources_fixture(Path(tmp));result=G.resource_join(*args)
            self.assertEqual(result['checks'],56);self.assertFalse(result['qualification_granted'])

    def test_resource_source_scope_memory_prefix_and_phase_tamper_refuse(self):
        for change in ('owner','memory','prefix','stage','growth','missing_phase','admission'):
            with self.subTest(change=change),tempfile.TemporaryDirectory() as tmp:
                args=list(resources_fixture(Path(tmp)));saved=args[0];row=saved['resource_checks'][1]['receipt']
                if change=='owner':row['public_generation']+=1
                if change=='memory':row['check']['resources']['available_memory_bytes']=1
                if change=='prefix':row['runtime_index']['prefix_sha256']='d'*64
                if change=='stage':row['stage']='before_request'
                if change=='growth':saved['phase_resources'][0]['finished_monotonic']+=1000
                if change=='missing_phase':saved['phase_resources'].pop()
                if change=='admission':saved['admissions'][0]['proof']['before_native_resource_sha256']=saved['resource_checks'][0]['sha256']
                if change not in ('admission','growth','missing_phase'):
                    sha=G.E.digest(row);saved['resource_checks'][1]['sha256']=sha
                    saved['admissions'][0]['proof']['before_native_resource_sha256']=sha
                with self.assertRaises((ValueError,KeyError)):G.resource_join(*args)

    def test_missing_capture_is_named_and_pins_remain_strict(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError,'GROUP_COMPLETE_BOUNDED_CAPTURE_REQUIRED:off_03'):
                G.capture_file(Path(tmp),{},10,'off_03')

    def test_cpu_receipt_geometry_cannot_enter_native_review(self):
        table=list(range(65))+[0]*15
        signature=dict(block_table_values=[table],writer_consumer_slot=[1024],
            cache_geometry={k:dict(shape=[80,16,8,128],device='cpu') for k in ('key','value')},
            tensors={'block_table':dict(bytes=320,sha256=hashlib.sha256(struct.pack('<80i',*table)).hexdigest())})
        self.assertEqual(G.receipt_signature(signature,simulation=True)['cache_geometry']['key']['shape'][0],80)
        with self.assertRaises(ValueError):G.receipt_signature(signature)

    def test_native_query_is_a_detached_budget_snapshot_and_plan_mutation_refuses(self):
        from types import SimpleNamespace
        from demoncat.runtime.vllm_ascend.worker_extension.paged33_group_native import GroupEffectsNativeCall
        driver=GroupEffectsNativeCall.__new__(GroupEffectsNativeCall)
        driver._check=lambda:dict(active=False,owned=False,first_error=None)
        driver.observer_sha256='d'*64;driver.environment={};driver.calls=[];driver.failures=[];driver.fingerprint={}
        driver.controller=None;driver.budget=R.plan(observations(),variant=R.VARIANT,source_sha256='a'*64,build_sha256='b'*64)
        driver._budget_fingerprint=G.E.digest(driver.budget)
        state=driver.state();state['case_budget']['inputs']['phases'].clear()
        self.assertEqual(len(driver.budget['inputs']['phases']),5)
        driver.controller=SimpleNamespace(check_owner=lambda:None);driver.budget['case_seconds']+=1
        with self.assertRaisesRegex(RuntimeError,'ORIGINAL_RESOURCE_PLAN_CHANGED'):
            driver.check_resources('fault_00',3,'before_request')

    def test_complete_numeric_output_gate_distinguishes_signed_zero_and_refuses_mutated_bytes(self):
        from demoncat.runtime.vllm_ascend.worker_extension.paged33_group_control import complete_numeric_effect
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            def write(name,first):
                path=root/name;path.write_bytes(struct.pack('<e',first)+struct.pack('<e',0.)*2047)
                return dict(path=str(path),shape=[1,16,128],dtype='torch.float16',bytes=4096,
                    sha256=hashlib.sha256(path.read_bytes()).hexdigest())
            before=write('before.bin',0.);after=write('after.bin',-0.)
            self.assertNotEqual(before['sha256'],after['sha256'])
            self.assertEqual(complete_numeric_effect(before,after,root)['numeric_changed_elements'],0)
            after=write('after.bin',1.);checked=complete_numeric_effect(before,after,root)
            self.assertEqual((checked['numeric_changed_elements'],checked['max_abs_difference']),(1,1.))
            (root/'after.bin').write_bytes(b'\0'*4096)
            with self.assertRaisesRegex(RuntimeError,'OUTPUT_BYTES_CHANGED'):complete_numeric_effect(before,after,root)

    def test_cold_reader_import_does_not_load_framework_or_injector(self):
        code="import sys;sys.path.insert(0,sys.argv[1]);import review_arch22_group_c2;assert 'torch' not in sys.modules;assert not any(n.startswith('demoncat.runtime') for n in sys.modules)"
        result=subprocess.run([sys.executable,'-B','-c',code,str(EXPERIMENT)],capture_output=True,text=True,timeout=30)
        self.assertEqual(result.returncode,0,result.stderr)


if __name__=='__main__':unittest.main()
