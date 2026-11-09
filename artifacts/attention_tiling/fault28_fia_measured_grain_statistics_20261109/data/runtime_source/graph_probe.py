"""Fixed real QKV in a full TorchAir GE graph. No eager fallback accepted."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import traceback
import uuid
from collector_stage import select_collector


def sha(data):
    return hashlib.sha256(data).hexdigest()


def save(path, value):
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case-file', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    private = args.output/'private_output'
    private.mkdir()
    case = json.loads(args.case_file.read_bytes())
    save(args.output/'case.json', case)
    result = {'schema': 'fault28.fia_graph.capture.v1', 'case_id': case['case_id'],
              'status': 'RUNNING', 'npu_mechanism_accepted': False, 'calls': [], 'backend_graphs': []}
    try:
        collector=select_collector(case)
        if case.get('device_scalar_observer'):
            debug=args.output/'private_device_debug';debug.mkdir()
            os.environ['ASCEND_WORK_PATH']=str(debug)
        if case.get('operator_variant')=='private_vendor':
            owner_start=Path('/proc/self/stat').read_text().rsplit(')',1)[1].split()[19]
            token=uuid.uuid4().hex
            os.environ.update(FAULT28_FIA_GRAPH_MODE=case['mode'],FAULT28_FIA_GRAPH_TOKEN=token,
                FAULT28_FIA_GRAPH_PID=str(os.getpid()),FAULT28_FIA_GRAPH_OWNER_START=owner_start,FAULT28_FIA_GRAPH_LOG=str(args.output/'host_events.jsonl'))
            result.update(scope_token=token,owner_pid=os.getpid(),owner_starttime_ticks=owner_start)
        import torch
        import torch_npu
        import torchair
        from msprobe.pytorch import set_ge_dump_config
        batch, length, layout = case['batch'], case['sequence_length'], case['layout']
        assert batch in (1, 8) and length == 2048 and layout in ('BNSD', 'TND')
        assert case['q_heads'] == 16 and case['kv_heads'] == 8 and case['head_dim'] == 128
        manifest_path = Path(case['input_manifest'])
        assert sha(manifest_path.read_bytes()) == case['input_manifest_sha256']
        manifest = json.loads(manifest_path.read_bytes())
        cpu = {}
        for name, row in manifest['inputs'].items():
            data = (manifest_path.parent/row['file']).read_bytes()
            assert sha(data) == row['sha256'] and len(data) == row['bytes']
            cpu[name] = torch.frombuffer(bytearray(data), dtype=torch.float16).reshape(row['shape']).repeat(batch,1,1,1)
            assert sha(cpu[name].numpy().tobytes()) == row['expected_full_sha256'][str(batch)]
        original = {name: {'shape': list(value.shape), 'dtype': str(value.dtype),
                          'full_sha256': sha(value.numpy().tobytes()),
                          'target_sha256': sha(value[0].numpy().tobytes())} for name,value in cpu.items()}
        if layout == 'TND':
            cpu = {name: value.permute(0,2,1,3).reshape(batch*length,value.shape[1],128).contiguous()
                   for name,value in cpu.items()}
        mask_cpu = torch.triu(torch.ones((2048,2048), dtype=torch.bool), diagonal=1)
        torch.use_deterministic_algorithms(True)
        torch.npu.set_device(0)
        torch.npu.set_compile_mode(jit_compile=False)
        tensors = {name: value.to('npu:0') for name,value in cpu.items()}
        mask = mask_cpu.to('npu:0')
        torch.npu.synchronize()
        save(args.output/'inputs.json', {'manifest_sha256': sha(manifest_path.read_bytes()),
            'source': manifest['source'], 'original_bnsd': original,
            'actual_graph_inputs': {name:{'shape':list(value.shape),'sha256':sha(value.numpy().tobytes())}
                                   for name,value in cpu.items()},
            'mask':{'shape':[2048,2048],'dtype':'torch.bool','sha256':sha(mask_cpu.numpy().tobytes())},
            'target_request':0,'target_queries':'all positions 0..2047','all_peers_identical_to_target':True})
        # Capture immutable op objects before any msprobe API wrappers are installed.
        fia_op=torch.ops.npu.npu_fused_infer_attention_score.default
        permute_op=torch.ops.aten.permute.default
        reshape_op=torch.ops.aten.reshape.default
        linear_op=torch.ops.aten.linear.default
        consumer=case.get('consumer')
        weight=None
        if consumer:
            assert consumer=='qwen3_layer1_o_proj'
            from safetensors import safe_open
            checkpoint=Path('/models/Qwen3-0.6B/model.safetensors')
            h=hashlib.sha256()
            with checkpoint.open('rb') as stream:
                for chunk in iter(lambda:stream.read(8*1024*1024),b''): h.update(chunk)
            assert h.hexdigest()==case['checkpoint_sha256']
            with safe_open(str(checkpoint),framework='pt',device='cpu') as stream:
                weight_cpu=stream.get_tensor('model.layers.1.self_attn.o_proj.weight').to(torch.float16).contiguous()
            assert list(weight_cpu.shape)==[1024,2048]
            weight=weight_cpu.to('npu:0')
            save(args.output/'consumer.json',{'kind':consumer,'checkpoint_sha256':h.hexdigest(),
                'tensor':'model.layers.1.self_attn.o_proj.weight','dtype':str(weight_cpu.dtype),
                'shape':list(weight_cpu.shape),'sha256':sha(weight_cpu.numpy().tobytes()),
                'scope':'Real model weight and fixed captured QKV subgraph; not complete-model acceptance.'})
        class AttentionGraph(torch.nn.Module):
            def forward(self, query, key, value, atten_mask, projection_weight=None):
                output, _ = fia_op(
                    query, key, value, atten_mask=atten_mask,
                    actual_seq_lengths=[length*(i+1) for i in range(batch)] if layout=='TND' else [length]*batch,
                    actual_seq_lengths_kv=[length*(i+1) for i in range(batch)] if layout=='TND' else [length]*batch,
                    num_heads=16, num_key_value_heads=8, scale=manifest['source_sdpa_scalar_arguments']['scale'],
                    input_layout=layout, pre_tokens=2147483647, next_tokens=0,
                    sparse_mode=case['sparse_mode'], inner_precise=case['inner_precise'])
                if consumer:
                    projected=linear_op(reshape_op(permute_op(output,[0,2,1,3]),[batch,2048,2048]),projection_weight)
                    return output,projected
                return output
        config = torchair.CompilerConfig()
        if case.get('aicore_num') is not None:
            assert case['aicore_num'] == 16
            core_option = str(case['aicore_num'])
            if case.get('vectorcore_num') is not None:
                assert case['vectorcore_num'] == 32
                core_option += '|' + str(case['vectorcore_num'])
            config.ge_config.aicore_num = core_option
        ge_dir = args.output/'msprobe_graph'
        if case['ge_statistics']:
            config = set_ge_dump_config(dump_path=str(ge_dir), dump_mode='all', compiler_config=config)
            config.dump_config.dump_data = 'stats'
        else:
            config.debug.graph_dump.type = 'txt'
            config.debug.graph_dump.path = str(ge_dir)
            ge_dir.mkdir()
        save(args.output/'graph_config.json', {'backend':'torchair.get_npu_backend','fullgraph':True,
            'dynamic':False,'mode':str(config.mode.value),'msprobe_entry':'set_ge_dump_config',
            'dump_data':str(config.dump_config.dump_data.value),'options':config.as_dict(),
            'precision_debugger_statistics_mix_requested':bool(case.get('capture_msprobe')), 
            'note':'GE native CSV plus disclosed GraphBoundary statistics/mix observer when enabled.'})
        actual_backend = torchair.get_npu_backend(compiler_config=config)
        def backend(gm, example_inputs):
            nodes = [{'name':n.name,'op':n.op,'target':str(n.target),
                      'stack_trace':n.meta.get('stack_trace'), 'nn_module_stack':str(n.meta.get('nn_module_stack'))}
                     for n in gm.graph.nodes]
            record = {'index':len(result['backend_graphs']),'nodes':nodes,'code':gm.code}
            save(args.output/f'fx_graph_{record["index"]}.json',record)
            result['backend_graphs'].append({'index':record['index'],'nodes':len(nodes),
                'fia_nodes':[n['name'] for n in nodes if 'fused_infer_attention_score' in n['target']]})
            return actual_backend(gm, example_inputs)
        model = AttentionGraph().eval()
        compiled = torch.compile(model, backend=backend, fullgraph=True, dynamic=False)
        inputs = (tensors['query'],tensors['key'],tensors['value'],mask)
        if consumer: inputs=(*inputs,weight)
        class GraphBoundary(torch.nn.Module):
            def __init__(self,call):
                super().__init__()
                self.__dict__['_compiled_call']=call
            def forward(self,*args):
                return self.__dict__['_compiled_call'](*args)
        boundary=GraphBoundary(compiled).eval()
        holder=torch.nn.Module();holder.add_module('graph_region',boundary)
        debugger=None
        expected_input = {name: sha(value.numpy().tobytes()) for name,value in cpu.items()}
        for index in range(case['repeats']+1):
            marker = {'schema':'fault28.fia_graph.call_marker.v1','event':'enter','call_id':index,
                'phase':'compile_warmup' if index==0 else 'graph_replay','pid':os.getpid(),
                'batch':batch,'layout':layout,'deterministic_algorithms':torch.are_deterministic_algorithms_enabled()}
            print(json.dumps(marker),flush=True)
            started=time.monotonic()
            if index==1 and case.get('capture_msprobe'):
                from graph_statistics import configure
                debugger=configure(args.output,collector)
            if debugger is not None: debugger.start(model=holder,rank_id=0)
            try:
                with torch.inference_mode():
                    outputs=boundary(*inputs) if debugger is not None else compiled(*inputs)
            finally:
                if debugger is not None:
                    debugger.stop()
                    debugger.step()
            output=outputs[0] if consumer else outputs
            torch.npu.synchronize()
            elapsed=time.monotonic()-started
            if index==0:
                # GE removes these compiler files on process teardown. Preserve
                # actual bounded compiler metadata privately, before cleanup.
                metadata=args.output/'private_jit_metadata';metadata.mkdir()
                files=[]
                roots=[Path('/kernel_meta')]
                if os.environ.get('ASCEND_WORK_PATH'):
                    roots.append(Path(os.environ['ASCEND_WORK_PATH'])/'kernel_meta')
                candidates={p.resolve():p for root in roots for p in root.rglob('*') if p.is_file()}
                for path in sorted(candidates):
                    if not path.is_file() or path.stat().st_size>8*1024**2:continue
                    if not any(term in path.name.lower() for term in ('te_fusedinferattentionscore','fusedinferattentionscore_isolate_tiling')):continue
                    if path.suffix not in ('.json','.o'):continue
                    item={'file':str(path),'bytes':path.stat().st_size,'sha256':sha(path.read_bytes())}
                    if path.suffix=='.json':
                        dest=metadata/(str(len(files))+'_'+path.name)
                        dest.write_bytes(path.read_bytes());item['private_copy']=str(dest.relative_to(args.output))
                    files.append(item)
                if case.get('operator_variant')=='private_vendor':
                    assert any('_isolate_tiling_' in r['file'] for r in files), 'Actual compiler tiling JSON missing'
                    assert any(r['file'].endswith('.o') for r in files), 'Actual FIA compiled binary missing'
                save(args.output/'jit_compile_inventory.json',{'files':files,'search_roots':[str(p) for p in roots],
                    'scope':'Actual GE compile metadata and binary hashes captured before GE cleanup; no tensor payload, outputs or program modified.'})
            print(json.dumps(dict(marker,event='exit',elapsed_seconds=elapsed)),flush=True)
            out_cpu=output.detach().cpu().contiguous()
            if layout=='TND':
                out_cpu=out_cpu.reshape(batch,length,16,128).permute(0,2,1,3).contiguous()
            raw=out_cpu.numpy().tobytes()
            target=out_cpu[0].numpy().tobytes()
            target_path=private/f'call{index}_target.fp16.bin'
            target_path.write_bytes(target)
            assert torch.isfinite(out_cpu).all()
            for name,tensor in tensors.items():
                assert sha(tensor.detach().cpu().contiguous().numpy().tobytes())==expected_input[name]
            consumer_result=None
            if consumer:
                projected=outputs[1].detach().cpu().contiguous()
                assert torch.isfinite(projected).all()
                assert sha(weight.detach().cpu().contiguous().numpy().tobytes())==sha(weight_cpu.numpy().tobytes())
                projected_target=projected[0].numpy().tobytes()
                projection_path=private/f'call{index}_projection.fp16.bin'
                projection_path.write_bytes(projected_target)
                consumer_result={'shape':list(projected.shape),'full_sha256':sha(projected.numpy().tobytes()),
                    'target_sha256':sha(projected_target),'target_file':str(projection_path.relative_to(args.output)),
                    'request_sha256':[sha(row.numpy().tobytes()) for row in projected]}
            result['calls'].append({'call_id':index,'phase':marker['phase'],'elapsed_seconds':elapsed,
                'shape':list(output.shape),'logical_bnsd_shape':list(out_cpu.shape),'dtype':str(output.dtype),
                'full_sha256':sha(raw),'target_sha256':sha(target),
                'request_sha256':[sha(row.numpy().tobytes()) for row in out_cpu],
                'target_file':str(target_path.relative_to(args.output)),'target_bytes':len(target),
                'inputs_unchanged':True,'consumer':consumer_result})
        assert len(result['backend_graphs'])==1 and len(result['backend_graphs'][0]['fia_nodes'])==1
        if debugger is not None:
            from graph_statistics import receipt
            save(args.output/'statistics_receipt.json',receipt(args.output))
        maps=[line for line in Path('/proc/self/maps').read_text().splitlines()
              if any(term in line for term in ('optiling','opmaster','cust_opapi','cust_opsproto'))]
        save(args.output/'loaded_libraries.json',{'maps':maps,'custom_opp_path':os.environ.get('ASCEND_CUSTOM_OPP_PATH')})
        if case.get('operator_variant')=='private_vendor':
            rows=[json.loads(line) for line in (args.output/'host_events.jsonl').read_text().splitlines()]
            assert rows and all(r['owner_pid']==os.getpid() and r['owner_starttime_ticks']==owner_start and r['token']==token and r['eligible'] and r['scope_ancestry'][-1]['pid']==os.getpid() and r['scope_ancestry'][0]['pid']==r['pid'] for r in rows)
            assert all(r['serializer_roundtrip_matches'] and r['raw_unchanged_by_observer'] for r in rows)
            assert all(r['mode']==case['mode'] for r in rows)
            result['host_callback_count']=len(rows)
        result.update(status='CAPTURE_COMPLETED', repeated_target_bytes_equal=len({x['target_sha256'] for x in result['calls']})==1,
            repeated_all_outputs_equal=len({x['full_sha256'] for x in result['calls']})==1,
            deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
            torch_version=torch.__version__,torch_npu_version=torch_npu.__version__,
            torchair_file=torchair.__file__,ge_files=[str(p.relative_to(args.output)) for p in ge_dir.rglob('*') if p.is_file()])
    except BaseException:
        result.update(status='FAIL',traceback=traceback.format_exc())
    finally:
        save(args.output/'result.json',result)
    print(json.dumps({'status':result['status'],'case_id':case['case_id'],'error':result.get('traceback','')[-4000:]}),flush=True)
    return 0 if result['status']=='CAPTURE_COMPLETED' else 1


if __name__=='__main__':
    raise SystemExit(main())
