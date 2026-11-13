"""Real eager FIA search/harvest/formal probe; raw msprobe and runtime audit are separate."""
import argparse
import hashlib
import importlib
import importlib.metadata
import json
import os
from pathlib import Path
import time
import traceback
import uuid

from collector_stage import select_collector,verify_loaded


def sha(data):return hashlib.sha256(data).hexdigest()


def file_sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*1024**2),b''):h.update(block)
    return h.hexdigest()


def save(path,value):
    with Path(path).open('x',encoding='utf-8') as f:
        json.dump(value,f,indent=2,allow_nan=False);f.write('\n')


def configure(out,collector,steps):
    from msprobe.pytorch import PrecisionDebugger
    proc=importlib.import_module('msprobe.core.dump.data_dump.data_processor.pytorch_processor')
    collector['actual_loaded']=verify_loaded(collector,proc)
    support=Path(collector['package_root'])/'msprobe/pytorch/dump/api_dump/support_wrap_ops.yaml'
    assert '- npu_fused_infer_attention_score' in support.read_text()
    collector['support_registration']={'path':str(support),'sha256':file_sha(support)}
    save(out/'collector_provenance.json',collector)
    config={'task':'statistics','level':'mix','precision':'high','rank':[],'step':steps,
            'dump_path':str(out/'observation'),'async_dump':False,
            'statistics':{'scope':[],'list':[],'data_mode':['all'],
                          'summary_mode':'statistics','tensor_list':[]}}
    save(out/'config.json',config)
    return PrecisionDebugger(config_path=str(out/'config.json'))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--case-file',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();out=a.output;out.mkdir(parents=True,exist_ok=False)
    private=out/'private_output';private.mkdir()
    case=json.loads(a.case_file.read_bytes());save(out/'case.json',case)
    result={'schema':'fault28.visible.capture.v1','status':'RUNNING','case_id':case['case_id'],'execution_mode':'eager',
            'calls':[],'accepted':False,'scope':case['capture_scope']}
    try:
        scope=case['capture_scope'];assert scope in ('harvest','calibration','model')
        assert case['batch'] in (1,8) and case['sequence_length']==2048
        steps=case.get('statistics_steps',[0,1,2] if scope=='model' else [0])
        nsteps=len(steps);assert steps==list(range(nsteps))
        assert nsteps in (1,3)
        collector=select_collector(case) if case['capture_msprobe'] else None
        token=uuid.uuid4().hex
        start=Path('/proc/self/stat').read_text().rsplit(')',1)[1].split()[19]
        os.environ['ASCEND_WORK_PATH']=str(private/'device_debug')
        if case['operator_variant']=='private_vendor':
            os.environ.update(FAULT28_FIA_GRAPH_MODE=case['mode'],FAULT28_FIA_GRAPH_TOKEN=token,
                FAULT28_FIA_GRAPH_PID=str(os.getpid()),FAULT28_FIA_GRAPH_OWNER_START=start,
                FAULT28_FIA_GRAPH_LOG=str(out/'host_events.jsonl'))
        import torch
        import torch_npu
        import numpy as np
        import model_adapter
        frozen=json.loads((out.parent/'source_manifest.json').read_bytes())
        assert file_sha(Path(__file__))==frozen['eager_probe.py']
        assert file_sha(Path(model_adapter.__file__))==frozen['model_adapter.py']
        torch.manual_seed(42);torch.use_deterministic_algorithms(True)
        def no_compile(*args,**kwargs):raise RuntimeError('torch.compile is forbidden in eager acceptance')
        torch.compile=no_compile
        torch.npu.set_device(0);torch.npu.set_compile_mode(jit_compile=False)
        torch.npu.set_device_limit(0,cube_num=16,vector_num=32)
        assert torch.npu.get_device_limit(0)=={'cube_core_num':16,'vector_core_num':32}
        save(out/'execution_config.json',{'mode':'eager','torch_compile_forbidden':True,
             'actual_process_core_limit':torch.npu.get_device_limit(0),'jit_compile':False,
             'route':'torch_npu.npu_fused_infer_attention_score / native ACLNN',
             'statistics_steps':list(range(nsteps)) if collector else []})
        mask_cpu=torch.triu(torch.ones((2048,2048),dtype=torch.bool),diagonal=1)
        batch=case['batch'];model=None
        if scope=='calibration':
            manifest=Path(case['qkv_manifest_file'])
            assert file_sha(manifest)==case['qkv_manifest_sha256']
            source=json.loads(manifest.read_bytes());cpu=[]
            for kind,heads in [('query',16),('key',8),('value',8)]:
                row=source[kind];path=manifest.parent/row['file']
                assert path.resolve().is_relative_to(manifest.parent.resolve())
                assert file_sha(path)==row['sha256']
                x=torch.from_numpy(np.fromfile(path,dtype='<f2').reshape(row['shape']).copy())
                assert tuple(x.shape)==(8,heads,2048,128)
                cpu.append(x[:batch].contiguous())
            cpu_inputs=(*cpu,mask_cpu)
            class Calibration(torch.nn.Module):
                def forward(self,q,k,v,mask):return model_adapter.native_fia(0,q,k,v,mask,128**-0.5)
            wrapper=Calibration().eval()
            save(out/'inputs.json',{'qkv_manifest_sha256':file_sha(manifest),'batch':batch,
                 'source_input_id':source['input_id'],'request_rows':source['request_rows'][:batch]})
        else:
            from transformers import AutoModelForCausalLM
            input_file=Path(case['input_manifest']);assert file_sha(input_file)==case['input_manifest_sha256']
            manifest=json.loads(input_file.read_bytes());requests=manifest['requests'][:batch]
            assert len(requests)==batch and all(len(r['prompt_token_ids'])==2048 for r in requests)
            assert all(sha(json.dumps(r['prompt_token_ids'],separators=(',',':')).encode())==r['prompt_token_ids_sha256'] for r in requests)
            mm=Path(case['model_manifest']);assert file_sha(mm)==case['model_manifest_sha256']
            for name,row in json.loads(mm.read_bytes())['files'].items():
                path=Path('/models/Qwen3-0.6B')/name
                assert path.stat().st_size==row['bytes'] and file_sha(path)==row['sha256']
            versions={n:importlib.metadata.version(n) for n in ('torch','torch-npu','transformers')}
            assert versions=={'torch':'2.10.0','torch-npu':'2.10.0.post4','transformers':'5.5.4'}
            inventory=Path(case['cpu_inventory']);assert file_sha(inventory)==case['cpu_inventory_sha256']
            inv=json.loads(inventory.read_bytes());assert versions==inv['versions']
            for name,row in inv['modules'].items():
                path=Path(importlib.import_module(name).__file__)
                assert str(path)==row['path'] and file_sha(path)==row['sha256'],name
            save(out/'versions.json',versions)
            model=AutoModelForCausalLM.from_pretrained('/models/Qwen3-0.6B',dtype=torch.float16,
                attn_implementation='eager',local_files_only=True,trust_remote_code=False).eval()
            parameters_before={n:sha(t.detach().cpu().contiguous().numpy().tobytes()) for n,t in model.named_parameters()}
            model_adapter.install(model);model=model.to('npu:0')
            wrapper=model_adapter.FullModelEager(model).eval()
            cpu_inputs=(torch.tensor([r['prompt_token_ids'] for r in requests],dtype=torch.long),
                        torch.arange(2048,dtype=torch.long).unsqueeze(0),mask_cpu)
            save(out/'inputs.json',{'input_id':manifest['input_id'],'manifest_sha256':file_sha(input_file),
                'request_rows':[{'slot':i,'request_id':r['request_id'],'token_sha256':r['prompt_token_ids_sha256']} for i,r in enumerate(requests)],
                'position_ids':'0..2047','use_cache':False,'generated_feedback':False})
        inputs=tuple(t.to('npu:0') for t in cpu_inputs);torch.npu.synchronize()
        original_inputs=[sha(t.contiguous().numpy().tobytes()) for t in cpu_inputs]
        holder=torch.nn.Module();holder.add_module('forward_region',wrapper)
        debugger=None;calls=1 if scope=='harvest' else 1+nsteps
        for call_id in range(calls):
            model_adapter.STATE['forward']=call_id;begin=len(model_adapter.STATE['calls']);started=time.monotonic()
            if call_id==1 and collector:debugger=configure(out,collector,list(range(nsteps)))
            if debugger:debugger.start(model=holder,rank_id=0)
            print(json.dumps({'schema':'fault28.eager.forward_marker.v1','event':'enter',
                              'call_id':call_id,'pid':os.getpid()}),flush=True)
            try:
                with torch.inference_mode():reply=wrapper(*inputs)
                torch.npu.synchronize()
            finally:
                if debugger:debugger.stop();debugger.step()
            print(json.dumps({'schema':'fault28.eager.forward_marker.v1','event':'exit',
                              'call_id':call_id,'pid':os.getpid()}),flush=True)
            actual=model_adapter.STATE['calls'][begin:]
            assert [c['layer'] for c in actual]==([0] if scope=='calibration' else list(range(28)))
            if scope=='calibration':
                values=[(0,'attention',reply)]
            else:
                observations=model_adapter.STATE.pop('observations')
                assert len(observations)==112
                values=[(None,'logits',reply)]+[(layer,kind,observations[4*layer+i])
                    for layer in range(28) for i,kind in enumerate(('query','key','value','attention'))]
            signatures=[];qkv={}
            for layer,kind,t in values:
                if scope=='harvest' and not (layer==0 and kind in ('query','key','value')):continue
                cpu=t.detach().cpu().contiguous()
                assert cpu.dtype==torch.float16 and cpu.shape[0]==batch and bool(torch.isfinite(cpu).all())
                raw=cpu.numpy().tobytes()
                row={'layer':layer,'kind':kind,'shape':list(cpu.shape),'dtype':str(cpu.dtype),
                     'full_sha256':sha(raw),'request_sha256':[sha(x.numpy().tobytes()) for x in cpu]}
                persist=scope=='harvest' or kind=='logits' or layer==0
                if call_id==0 and persist:
                    name=f'{kind}_{layer}.fp16.bin';(private/name).write_bytes(raw)
                    row['private_file']='private_output/'+name
                    if scope=='harvest':qkv[kind]={'file':'private_output/'+name,'sha256':sha(raw),'shape':list(cpu.shape)}
                signatures.append(row)
            assert all(sha(t.detach().cpu().contiguous().numpy().tobytes())==h for t,h in zip(inputs,original_inputs))
            if scope=='harvest':
                source_rows=json.loads((out/'inputs.json').read_bytes())['request_rows']
                save(out/'qkv_manifest.json',dict(qkv,schema='fault28.visible.qkv.v1',input_id=manifest['input_id'],request_rows=source_rows))
            result['calls'].append({'call_id':call_id,'phase':'warmup' if call_id==0 else 'statistics_forward' if debugger else 'uncaptured_repeat',
                'statistics_step':call_id-1 if debugger else None,'values':signatures,'api_markers':actual,
                'elapsed_seconds':time.monotonic()-started,'inputs_unchanged':True,
                'logits_top1':reply.detach().cpu()[:,0,:].argmax(-1).tolist() if scope=='model' else None})
        assert not torch.compiler.is_compiling() and getattr(wrapper,'_compiled_call_impl',None) is None
        maps=Path('/proc/self/maps').read_text()
        paths=sorted({x.split()[-1] for x in maps.splitlines() if '.so' in x and len(x.split())>=6})
        custom={path:file_sha(Path(path)) for path in paths if path.startswith(case.get('approved_custom_opp_path','/nonexistent')+'/') and Path(path).is_file()}
        save(out/'loaded_native_libraries.json',{'paths':paths,'custom_library_file_sha256':custom})
        if model is not None:
            assert parameters_before=={n:sha(t.detach().cpu().contiguous().numpy().tobytes()) for n,t in model.named_parameters()}
            save(out/'parameters_receipt.json',{'unchanged':True,'model_manifest_sha256':case['model_manifest_sha256']})
        if collector:
            receipts=[]
            for step in range(nsteps):
                folder=out/f'observation/step{step}/rank0';dump=json.loads((folder/'dump.json').read_bytes())
                names=[k for k in dump['data'] if k.startswith('NPU.npu_fused_infer_attention_score.')]
                assert len(names)==(1 if scope=='calibration' else 28)
                assert dump['task']=='statistics' and dump['level']=='mix' and dump.get('dump_data_dir') is None
                receipts.append({'step':step,'records':len(dump['data']),'FIA_records':names,
                    'files':{k:file_sha(folder/(k+'.json')) for k in ('dump','stack','construct')}})
            save(out/'statistics_receipt.json',{'actual_steps':list(range(nsteps)),'steps':receipts})
        def contract(call):return [(v['layer'],v['kind'],v['shape'],v['full_sha256'],v['request_sha256']) for v in call['values']]
        stable=all(contract(c)==contract(result['calls'][0]) for c in result['calls'])
        result.update(status='CAPTURE_COMPLETED',repeated_all_outputs_equal=stable,
            parameters_unchanged=model is not None,
            actual_eager_forwards=calls,actual_eager_FIA_API_calls=len(model_adapter.STATE['calls']),
            owner_pid=os.getpid(),owner_starttime_ticks=start,scope_token=token,
            statistics_steps=list(range(nsteps)) if collector else [],torch_compile_calls=0)
        assert stable
    except BaseException:
        result.update(status='FAIL',traceback=traceback.format_exc())
    save(out/'result.json',result)
    print(json.dumps({k:v for k,v in result.items() if k!='calls'}),flush=True)
    return 0 if result['status']=='CAPTURE_COMPLETED' else 1


if __name__=='__main__':raise SystemExit(main())
