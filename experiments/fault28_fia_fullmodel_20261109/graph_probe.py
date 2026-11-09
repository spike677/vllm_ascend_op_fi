"""Actual 28-layer Qwen3, one fullgraph, FIA in every attention module."""
import argparse
import hashlib
import importlib
import importlib.metadata
import json
import os
from pathlib import Path
import re
import time
import traceback
import uuid
from collector_stage import select_collector


def sha(data):return hashlib.sha256(data).hexdigest()


def file_sha(path):
    digest=hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda:stream.read(8*1024**2),b''):digest.update(chunk)
    return digest.hexdigest()


def save(path,value):
    with path.open('x',encoding='utf-8') as stream:
        json.dump(value,stream,indent=2,allow_nan=False);stream.write('\n')


def layer_of(stack):
    matches=set(int(v) for v in re.findall(r'layers(?:\.|\[|_)(\d+)',str(stack)))
    return next(iter(matches)) if len(matches)==1 else None


def parameter_digests(model):
    return {name:{'shape':list(t.shape),'dtype':str(t.dtype),
                  'sha256':sha(t.detach().cpu().contiguous().numpy().tobytes())}
            for name,t in model.named_parameters()}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case-file',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();out=args.output;out.mkdir(parents=True,exist_ok=False)
    private=out/'private_output';private.mkdir()
    case=json.loads(args.case_file.read_bytes());save(out/'case.json',case)
    result={'schema':'fault28.fia_fullmodel.capture.v1','case_id':case['case_id'],
        'status':'RUNNING','calls':[],'backend_graphs':[],'mechanism_accepted':False}
    try:
        collector=select_collector(case)
        debug=out/'private_device_debug';debug.mkdir()
        os.environ['ASCEND_WORK_PATH']=str(debug)
        starttime=Path('/proc/self/stat').read_text().rsplit(')',1)[1].split()[19]
        token=uuid.uuid4().hex
        os.environ.update(FAULT28_FIA_GRAPH_MODE=case['mode'],FAULT28_FIA_GRAPH_TOKEN=token,
            FAULT28_FIA_GRAPH_PID=str(os.getpid()),FAULT28_FIA_GRAPH_OWNER_START=starttime,
            FAULT28_FIA_GRAPH_LOG=str(out/'host_events.jsonl'))
        result.update(owner_pid=os.getpid(),owner_starttime_ticks=starttime,scope_token=token)
        import torch
        import torch_npu
        import torchair
        from transformers import AutoModelForCausalLM
        from msprobe.pytorch import set_ge_dump_config
        import model_adapter
        batch=case['batch'];assert batch in (1,8) and case['sequence_length']==2048
        source=Path(case['input_manifest']);assert file_sha(source)==case['input_manifest_sha256']
        manifest=json.loads(source.read_bytes());by_id={r['request_id']:r for r in manifest['requests']}
        requests=[by_id[name] for name in case['request_ids']]
        assert len(requests)==batch and len({r['request_id'] for r in requests})==batch
        assert len({r['prompt_token_ids_sha256'] for r in requests})==batch
        assert all(len(r['prompt_token_ids'])==2048 and sha(json.dumps(r['prompt_token_ids'],
            separators=(',',':')).encode())==r['prompt_token_ids_sha256'] for r in requests)
        inventory_path=Path(case['cpu_inventory'])
        assert file_sha(inventory_path)==case['cpu_inventory_sha256']
        inventory=json.loads(inventory_path.read_bytes())
        assert file_sha(Path(model_adapter.__file__))==inventory['adapter_source_sha256']
        actual_versions={n:importlib.metadata.version(n) for n in ('torch','torch-npu','transformers')}
        assert actual_versions==inventory['versions']
        for name,row in inventory['modules'].items():
            module=importlib.import_module(name);p=Path(module.__file__)
            assert str(p)==row['path'] and file_sha(p)==row['sha256'],name
        checkpoint_manifest=Path(case['model_manifest'])
        assert file_sha(checkpoint_manifest)==case['model_manifest_sha256']
        for name,row in json.loads(checkpoint_manifest.read_bytes())['files'].items():
            p=Path('/models/Qwen3-0.6B')/name
            assert p.stat().st_size==row['bytes'] and file_sha(p)==row['sha256']
        (out/'model_config.json').write_bytes(Path('/models/Qwen3-0.6B/config.json').read_bytes())
        torch.manual_seed(42);torch.use_deterministic_algorithms(True)
        torch.npu.set_device(0);torch.npu.set_compile_mode(jit_compile=False)
        model=AutoModelForCausalLM.from_pretrained('/models/Qwen3-0.6B',dtype=torch.float16,
            attn_implementation='eager',local_files_only=True,trust_remote_code=False).eval()
        assert all(p.dtype==torch.float16 for p in model.parameters())
        parameters_before=parameter_digests(model)
        save(out/'loaded_parameters.json',{'source':'original checkpoint, uniform FP16 conversion',
            'parameters':parameters_before,'checkpoint_sha256':file_sha(Path('/models/Qwen3-0.6B/model.safetensors'))})
        model_adapter.install(model);model=model.to('npu:0')
        wrapper=model_adapter.FullModelGraph(model,aliases=case['capture_aliases']).eval()
        ids_cpu=torch.tensor([r['prompt_token_ids'] for r in requests],dtype=torch.long)
        position_cpu=torch.arange(2048,dtype=torch.long).unsqueeze(0)
        mask_cpu=torch.triu(torch.ones((2048,2048),dtype=torch.bool),diagonal=1)
        cpu_inputs=(ids_cpu,position_cpu,mask_cpu)
        input_hashes=[sha(t.numpy().tobytes()) for t in cpu_inputs]
        inputs=tuple(t.to('npu:0') for t in cpu_inputs);torch.npu.synchronize()
        save(out/'inputs.json',{'manifest_sha256':file_sha(source),'request_rows':[
            {'slot':i,'request_id':r['request_id'],'prompt_length':2048,
             'token_sha256':r['prompt_token_ids_sha256']} for i,r in enumerate(requests)],
            'actual_input_shapes':[list(t.shape) for t in inputs], 'actual_input_sha256':input_hashes,
            'position_ids':'0..2047, identical for all requests',
            'prefill_next_token_position':2047,'use_cache':False,'generated_feedback':False})
        slots=[{'slot':0,'kind':'logits','layer':None}]
        if case['capture_aliases']:
            slots += [{'slot':1+4*layer+i,'kind':kind,'layer':layer}
                      for layer in range(28) for i,kind in enumerate(('query','key','value','attention'))]
        save(out/'output_contract.json',{'slots':slots,'scope':'Actual returned graph values; producer mapping must be verified from FX/GE, not this annotation.'})
        config=torchair.CompilerConfig();config.ge_config.aicore_num='16|32'
        ge_dir=out/'msprobe_graph'
        config=set_ge_dump_config(dump_path=str(ge_dir),dump_mode='all',compiler_config=config)
        config.dump_config.dump_data='stats'
        save(out/'graph_config.json',{'fullgraph':True,'dynamic':False,'backend':'torchair.get_npu_backend',
            'options':config.as_dict(),'attention_route':model_adapter.ROUTE,
            'capture_aliases':case['capture_aliases'],'use_cache':False})
        actual_backend=torchair.get_npu_backend(compiler_config=config)
        engine_state={'calls':0}
        def backend(gm,example_inputs):
            nodes=[]
            for node in gm.graph.nodes:
                stack=str(node.meta.get('nn_module_stack'))
                nodes.append({'name':node.name,'op':node.op,'target':str(node.target),
                    'stack_trace':node.meta.get('stack_trace'),'nn_module_stack':stack,
                    'layer':layer_of(stack),'input_nodes':[n.name for n in node.all_input_nodes]})
            fia=[n for n in nodes if n['target']=='npu.npu_fused_infer_attention_score.default']
            assert len(fia)==28 and [n['layer'] for n in fia]==list(range(28)),fia
            returned=[n for n in gm.graph.nodes if n.op=='output']
            assert len(returned)==1
            flat=returned[0].args[0]
            assert isinstance(flat,(tuple,list)) and len(flat)==len(slots), 'Backend output flattening changed'
            fia_nodes=[n for n in gm.graph.nodes if str(n.target)=='npu.npu_fused_infer_attention_score.default']
            if case['capture_aliases']:
                import operator
                for layer,node in enumerate(fia_nodes):
                    assert all(flat[1+4*layer+i] is node.args[i] for i in range(3)), 'QKV output order mismatch'
                    attn=flat[4+4*layer]
                    assert attn.target is operator.getitem and attn.args==(node,0), 'FIA output order mismatch'
            index=len(result['backend_graphs'])
            save(out/f'fx_graph_{index}.json',{'index':index,'nodes':nodes,'code':gm.code,
                'output_nodes':[n.name for n in flat],'backend_output_matches_observed_tuple':True})
            result['backend_graphs'].append({'index':index,'nodes':len(nodes),'fia_nodes':fia})
            engine=actual_backend(gm,example_inputs)
            assert 'engine' not in engine_state, 'Unexpected recompilation'
            engine_state['engine']=engine
            engine_state['placeholders']=[n.name for n in gm.graph.nodes if n.op=='placeholder']
            def execute(*arguments):
                assert 'arguments' not in engine_state, 'Dynamo front end reentered after warmup'
                engine_state['arguments']=arguments
                engine_state['calls']+=1
                return engine(*arguments)
            return execute
        compiled=torch.compile(wrapper,backend=backend,fullgraph=True,dynamic=False)
        def replay(*actual_inputs):
            # Fixed prefill workload: same live input and parameter objects.
            # Invoke the real backend, not Dynamo's hook-sensitive front end.
            assert len(actual_inputs)==len(inputs) and all(a is b for a,b in zip(actual_inputs,inputs))
            engine_state['calls']+=1
            return engine_state['engine'](*engine_state['arguments'])
        class GraphBoundary(torch.nn.Module):
            def __init__(self,call):
                super().__init__();self.__dict__['_compiled_call']=call
            def forward(self,*tensors):return self.__dict__['_compiled_call'](*tensors)
        boundary=GraphBoundary(replay).eval();holder=torch.nn.Module();holder.add_module('graph_region',boundary)
        debugger=None
        batch_private=Path('/private_batch')/case['case_id'];batch_private.mkdir(exist_ok=False)
        for call_id in range(3):
            marker={'schema':'fault28.fia_graph.call_marker.v1','event':'enter','call_id':call_id,
                'phase':'compile_warmup' if call_id==0 else 'graph_replay','pid':os.getpid(),
                'batch':batch,'layout':'BNSD','model_layers':28}
            print(json.dumps(marker),flush=True);started=time.monotonic()
            if call_id==1:
                from graph_statistics import configure
                debugger=configure(out,collector)
            if debugger is not None:debugger.start(model=holder,rank_id=0)
            try:
                with torch.inference_mode():
                    outputs=boundary(*inputs) if debugger is not None else compiled(*inputs)
            finally:
                if debugger is not None:debugger.stop();debugger.step()
            torch.npu.synchronize()
            print(json.dumps(dict(marker,event='exit',elapsed_seconds=time.monotonic()-started)),flush=True)
            assert len(outputs)==len(slots)
            if call_id==0:
                arguments=engine_state['arguments']
                assert len(arguments)==len(engine_state['placeholders'])
                bindings=[]
                for name,value in zip(engine_state['placeholders'],arguments):
                    row={'placeholder':name,'python_type':str(type(value))}
                    if isinstance(value,torch.Tensor):
                        row.update(shape=list(value.shape),dtype=str(value.dtype),
                            device=str(value.device),data_ptr=value.data_ptr())
                    bindings.append(row)
                assert all(any(isinstance(a,torch.Tensor) and a.data_ptr()==t.data_ptr()
                               and tuple(a.shape)==tuple(t.shape) for a in arguments) for t in inputs)
                save(out/'compiled_engine_bindings.json',{'arguments':bindings,
                    'all_user_inputs_bound':True,'same_live_objects_replayed':True,
                    'scope':'Actual fullgraph backend and argument tuple captured during real warmup. No eager fallback or replacement results.'})
                metadata=out/'private_jit_metadata';metadata.mkdir();files=[]
                roots=[Path('/kernel_meta'),debug/'kernel_meta']
                candidates={p.resolve() for root in roots for p in root.rglob('*') if p.is_file()}
                for p in sorted(candidates):
                    if p.stat().st_size>8*1024**2 or p.suffix not in ('.json','.o'):continue
                    if 'fusedinferattentionscore' not in p.name.lower():continue
                    row={'file':str(p),'bytes':p.stat().st_size,'sha256':file_sha(p)}
                    if p.suffix=='.json':
                        dest=metadata/(str(len(files))+'_'+p.name);dest.write_bytes(p.read_bytes())
                        row['private_copy']=str(dest.relative_to(out))
                    files.append(row)
                assert any('_isolate_tiling_' in r['file'] for r in files)
                assert any(r['file'].endswith('.o') for r in files)
                save(out/'jit_compile_inventory.json',{'files':files,'search_roots':[str(p) for p in roots]})
            values=[]
            for slot,tensor in zip(slots,outputs):
                assert tensor.dtype==torch.float16 and tensor.shape[0]==batch
                cpu=tensor.detach().cpu().contiguous();assert torch.isfinite(cpu).all()
                raw=cpu.numpy().tobytes();target=cpu[0].numpy().tobytes()
                name=f'value_{slot["slot"]:03d}.fp16.bin'
                if call_id==0:
                    (private/name).write_bytes(raw if slot['kind']=='logits' else target)
                    if slot['kind']!='logits':(batch_private/name).write_bytes(raw)
                values.append({**slot,'shape':list(cpu.shape),'dtype':str(cpu.dtype),
                    'full_sha256':sha(raw),'target_sha256':sha(target),
                    'request_sha256':[sha(row.numpy().tobytes()) for row in cpu],
                    'target_file':'private_output/'+name,
                    'private_batch_relative':case['case_id']+'/'+name if slot['kind']!='logits' else None})
            assert all(sha(t.detach().cpu().contiguous().numpy().tobytes())==h for t,h in zip(inputs,input_hashes))
            logits=outputs[0].detach().cpu().contiguous()
            assert list(logits.shape)==[batch,1,model.config.vocab_size]
            result['calls'].append({'call_id':call_id,'phase':marker['phase'],'values':values,
                'logits_top1':logits[:,0,:].argmax(-1).tolist(),'inputs_unchanged':True,
                'elapsed_seconds':time.monotonic()-started})
        assert len(result['backend_graphs'])==1
        assert engine_state['calls']==3
        assert parameters_before==parameter_digests(model),'Model parameters changed'
        save(out/'parameters_receipt.json',{'unchanged_after_three_forwards':True,
            'loaded_parameters_sha256':file_sha(out/'loaded_parameters.json')})
        from graph_statistics import receipt
        save(out/'statistics_receipt.json',receipt(out))
        host=[json.loads(line) for line in (out/'host_events.jsonl').read_text().splitlines()]
        assert host and all(r['owner_pid']==os.getpid() and r['token']==token and
            r['owner_starttime_ticks']==starttime and r['serializer_roundtrip_matches'] and
            r['raw_unchanged_by_observer'] for r in host)
        stable=all([v['full_sha256'] for v in call['values']]==
                   [v['full_sha256'] for v in result['calls'][0]['values']] for call in result['calls'])
        result.update(status='CAPTURE_COMPLETED',versions=actual_versions,torch_version=torch.__version__,
            torch_npu_version=torch_npu.__version__,host_callback_count=len(host),
            repeated_all_outputs_equal=stable,parameters_unchanged=True,
            actual_backend_calls=engine_state['calls'],single_graph_backend_replay=True,
            deterministic_algorithms=torch.are_deterministic_algorithms_enabled())
    except BaseException:
        result.update(status='FAIL',traceback=traceback.format_exc())
    finally:
        save(out/'result.json',result)
    print(json.dumps({'status':result['status'],'case_id':case['case_id'],
                      'error':result.get('traceback','')[-5000:]}),flush=True)
    return 0 if result['status']=='CAPTURE_COMPLETED' else 1


if __name__=='__main__':raise SystemExit(main())
