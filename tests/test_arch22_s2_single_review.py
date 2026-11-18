"""Independent stdlib/CPU byte reader contracts; no Native execution proof."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1];EXPERIMENT=ROOT/'experiments/c2c3_closure_20261117'
sys.path.insert(0,str(EXPERIMENT))
import review_arch22_s2_single as review
import test_arch22_normal_bridge_review as fixtures


def actual_scheduler_fixture(phase='reference'):
    """Construct the two producer schemas; no private result or device data."""
    prompt=list(range(1024));rid='fixture-'+phase;pages=list(range(64))
    def tensor(values,dtype):
        raw=struct.pack('<'+('i' if dtype=='torch.int32' else 'q')*len(values),*values)
        return dict(shape=[len(values)],dtype=dtype,finite=True,sha256=hashlib.sha256(raw).hexdigest())
    def metadata(length,dtype,storage):
        return dict(object_id=storage+1,storage_identity=storage,storage_offset=0,
                    shape=[length],stride=[1],dtype=dtype,device='npu:0')
    ps=dict(internal_request_id=rid,external_request_id=None,num_requests=1,num_prompt_tokens=1024,
        num_scheduled_tokens=1024,num_computed_tokens=0,token_ids=prompt,positions=list(range(1024)),
        token_ids_sha256=hashlib.sha256(json.dumps(prompt,separators=(',',':')).encode()).hexdigest(),
        token_count_evidence=dict(request_num_tokens=1024,batch_num_tokens_no_spec=1024,batch_num_tokens_legacy=[1026],
            legacy_field_role='POST_SAMPLING_COUNTER_NOT_PREFILL_AUTHORITY'),
        input_tensor_metadata=dict(input_ids=metadata(1024,'torch.int32',101),positions=metadata(1024,'torch.int64',201)),
        block_size=16,block_ids=pages,slot_mapping=list(range(1024)),attention_metadata_keys=list(review.LAYERS),
        attention_metadata_evidence={path:dict(adapter='DENSE_EXPLICIT_LENGTHS_AND_CAUSAL',causal=True,
            actual_seq_lengths_q=[1024],seq_lens_list=[1024]) for path in review.LAYERS})
    ds=dict(request_ids=[rid],logical_ids=[rid],row_request_indices=[0],row_token_positions=[1024],counts=[1],
        request_count=1,token_row_count=1,scheduler_capacity=1,pure_decode=True,input_ids=[7],
        prefixes=[dict(prompt=prompt,output=[7])],actual_input_metadata=metadata(1,'torch.int32',101),
        actual_position_metadata=metadata(1,'torch.int64',201))
    forwards=[]
    for decode,scheduler,values,positions in ((False,ps,prompt,list(range(1024))),(True,ds,[7],[1024])):
        calls=[]
        for index,path in enumerate(review.LAYERS):
            call=dict(request_id=rid,module_path=path,module_object_id=1000+index,
                parent_path=path.rsplit('.',1)[0],aliases=[path],ordinal=index,occurrence=0,
                scheduler=deepcopy(scheduler),original_calls=1,original_returned=True,inputs_unchanged=True,
                actual_operator='torch_npu.npu_fused_infer_attention_score',native={} if decode and index==0 else None)
            if decode:
                call.update(metadata_binding='VERIFIED',layout='TND',sequence_lengths=[1025],query={'shape':[1,16,128]},
                    used_pages=[pages+[64]],slots=[1024],expected_slots=[1024],writer=dict(returned=True,module_path=path,
                        slots=[1024],slot_mapping=metadata(1,'torch.int32',301)))
                for offset,kind in enumerate(('key','value')):
                    cache=dict(shape=[128,16,8,128],stride=[16384,1024,128,1],storage_identity=10000+index*2+offset,
                        storage_offset=0,dtype='torch.float16',device='npu:0')
                    call[kind]=dict(cache,shape=[128,16,1024],stride=[16384,1024,1])
                    call['live_'+kind+'_cache']=deepcopy(cache);call['writer'][kind+'_cache']=deepcopy(cache)
            calls.append(call)
        forwards.append(dict(decode=decode,request_id=rid,phase=phase,root_returned=True,scheduler=scheduler,calls=calls,
            input=[[],dict(input_ids=tensor(values,'torch.int32'),positions=tensor(positions,'torch.int64'),
                           inputs_embeds=None,intermediate_tensors=None)]))
    request=dict(phase=phase,completed=True,mapping=dict(internal_request_id=rid,external_request_id='external-'+phase,
        request_state_object_id=99,source='ACTUAL_PUBLIC_ENQUEUE_AND_OUTPUT_PROCESSOR'),
        response=dict(request_id='external-'+phase,prompt_token_ids=prompt,outputs=[dict(tokens=[7,9])]),
        worker=dict(request_id=rid,phase=phase,forwards=forwards))
    return request,prompt


class ActualSchedulerReaderContracts(unittest.TestCase):
    def test_all_five_phases_reconcile_distinct_prefill_and_decode_schemas(self):
        for phase in review.PHASES:
            request,prompt=actual_scheduler_fixture(phase)
            prefill,decode=request['worker']['forwards']
            self.assertNotIn('request_ids',prefill['scheduler'])
            self.assertNotIn('internal_request_id',decode['scheduler'])
            row=review.request_lineage(request,prompt,'npu:0')
            self.assertEqual(row['actual_calls'],56);self.assertTrue(row['actual_root_tensor_hashes_recomputed'])
            writer=review.decode_cache_rows(prefill['scheduler'],decode['scheduler'],decode['calls'],device='npu:0')
            self.assertEqual(writer['actual_consumers'],28);self.assertTrue(writer['physical_slot_derived_from_actual_pages'])

    def test_wrong_request_row_position_token_tensor_and_call_join_fail(self):
        changes={
            'prefill_request':lambda r:r['worker']['forwards'][0]['scheduler'].update(internal_request_id='foreign'),
            'invented_common_key':lambda r:r['worker']['forwards'][0]['scheduler'].update(request_ids=['fixture-reference']),
            'prefill_position':lambda r:r['worker']['forwards'][0]['scheduler']['positions'].__setitem__(0,1),
            'prefill_slot':lambda r:r['worker']['forwards'][0]['scheduler']['slot_mapping'].__setitem__(0,3),
            'prefill_page_alias':lambda r:r['worker']['forwards'][0]['scheduler']['block_ids'].__setitem__(1,0),
            'prefill_metadata_layer':lambda r:r['worker']['forwards'][0]['scheduler']['attention_metadata_evidence'].pop(review.LAYERS[-1]),
            'decode_request':lambda r:r['worker']['forwards'][1]['scheduler'].update(request_ids=['foreign']),
            'decode_row':lambda r:r['worker']['forwards'][1]['scheduler'].update(row_request_indices=[1]),
            'decode_position':lambda r:r['worker']['forwards'][1]['scheduler'].update(row_token_positions=[1025]),
            'decode_prefix':lambda r:r['worker']['forwards'][1]['scheduler']['prefixes'][0].update(output=[9]),
            'actual_hash':lambda r:r['worker']['forwards'][1]['input'][1]['input_ids'].update(sha256='0'*64),
            'actual_dtype':lambda r:r['worker']['forwards'][1]['input'][1]['input_ids'].update(dtype='torch.int64'),
            'actual_shape':lambda r:r['worker']['forwards'][0]['input'][1]['positions'].update(shape=[1023]),
            'actual_device':lambda r:r['worker']['forwards'][1]['scheduler']['actual_input_metadata'].update(device='cpu'),
            'call_request':lambda r:r['worker']['forwards'][1]['calls'][1].update(request_id='foreign'),
            'call_scheduler':lambda r:r['worker']['forwards'][1]['calls'][1]['scheduler'].update(request_ids=['foreign']),
            'call_parent':lambda r:r['worker']['forwards'][1]['calls'][1].update(parent_path='foreign.parent'),
            'call_once':lambda r:r['worker']['forwards'][1]['calls'][1].update(original_calls=2),
            'call_missing':lambda r:r['worker']['forwards'][1]['calls'].pop(),
            'wrong_model_module':lambda r:r['worker']['forwards'][0]['calls'][1].update(module_object_id=9999),
        }
        for name,change in changes.items():
            request,prompt=actual_scheduler_fixture();change(request)
            with self.subTest(name=name),self.assertRaises((ValueError,KeyError)):
                review.request_lineage(request,prompt,'npu:0')

    def test_legacy_post_sampling_counter_is_retained_but_not_prefill_authority(self):
        request,prompt=actual_scheduler_fixture();prefill=request['worker']['forwards'][0]
        prefill['scheduler']['token_count_evidence']['batch_num_tokens_legacy']=[0]
        for call in prefill['calls']:call['scheduler']=deepcopy(prefill['scheduler'])
        self.assertFalse(review.request_lineage(request,prompt,'npu:0')['prefill_legacy_counter_used_as_authority'])
        prefill['scheduler']['token_count_evidence']['batch_num_tokens_no_spec']=1026
        with self.assertRaises(ValueError):review.request_lineage(request,prompt,'npu:0')

    def test_wrong_pages_writer_slot_or_equal_geometry_foreign_storage_fail(self):
        changes={
            'prefix_page':lambda c:c['used_pages'][0].__setitem__(0,100),
            'reused_page':lambda c:c['used_pages'][0].__setitem__(64,63),
            'out_of_range':lambda c:c['used_pages'][0].__setitem__(64,128),
            'false_derived_slot':lambda c:(c.update(slots=[0],expected_slots=[0]),c['writer'].update(slots=[0])),
            'writer_not_returned':lambda c:c['writer'].update(returned=False),
            'foreign_writer':lambda c:c['writer'].update(module_path='foreign.attn'),
            'slot_dtype':lambda c:c['writer']['slot_mapping'].update(dtype='torch.float16'),
            'flat_copy':lambda c:c['key'].update(storage_identity=999999),
            'writer_copy':lambda c:c['writer']['key_cache'].update(storage_identity=999999),
            'flat_offset':lambda c:c['value'].update(storage_offset=1),
            'writer_stride':lambda c:c['writer']['value_cache'].update(stride=[16384,1024,1,128]),
        }
        for name,change in changes.items():
            request,_=actual_scheduler_fixture();pre,dec=request['worker']['forwards'];change(dec['calls'][1])
            with self.subTest(name=name),self.assertRaises(ValueError):
                review.decode_cache_rows(pre['scheduler'],dec['scheduler'],dec['calls'],device='npu:0')


class SingleS2ReaderContracts(unittest.TestCase):
    def test_fault_raw_pod_requires_explicit_new_variant_and_actual_tile_bytes(self):
        helper=fixtures.evidence.helpers(fixtures.source())
        event,resolved,host,bundle=fixtures.pod_fixture(helper)
        bundle['completion']={'variant':review.E.S2_VARIANT}
        raw=bytearray.fromhex(host['raw_hex'])
        field=next(x for x in bundle['schema']['fields'] if x['name']=='innerSplitParams.s2BaseSize')
        struct.pack_into('<Q',raw,field['offset'],256)
        host.update(schema='demoncat.paged33.arch22_s2_host.v1',mode='arch22_s2_256',policy_applied=True,
                    selected_s2=256,raw_hex=raw.hex())
        typed=helper.abi.decode_raw(raw,bundle['schema'])
        host['typed']={k:v for k,v in typed.items() if not k.endswith('ParamsPH')}
        args=bytearray.fromhex(event['args_hex']);args[64:64+len(raw)]=raw;event['args_hex']=args.hex()
        result=review.N.final_private_pod(event,resolved,host,bundle,helper.abi,selected_s2=256)
        self.assertEqual(result['typed']['innerSplitParams.s2BaseSize'],256)
        with self.assertRaises(ValueError):review.N.final_private_pod(event,resolved,host,bundle,helper.abi)
        bad=deepcopy(bundle);bad['completion']['variant']=review.E.VARIANT
        with self.assertRaises(ValueError):review.N.final_private_pod(event,resolved,host,bad,helper.abi,selected_s2=256)
        for change in ('host_only','array','slot'):
            e,h=deepcopy(event),deepcopy(host)
            if change=='host_only':h['typed']['innerSplitParams.s2BaseSize']=512
            elif change=='array':h['typed']['actual_fixture.full_array'][27]+=1
            else:e['placeholders_hex']=struct.pack('<IIII',16,64,8,64+1576).hex()
            with self.subTest(change=change),self.assertRaises((ValueError,AssertionError)):
                review.N.final_private_pod(e,resolved,h,bundle,helper.abi,selected_s2=256)

    def test_logical_normalization_preserves_unused_pages_and_actual_kv_and_scalar_values(self):
        table=list(range(65))+[0]*15
        value=dict(block_table_values=[table],cache_geometry={k:{'shape':[128,16,8,128]} for k in ('key','value')},
            writer_consumer_slot=[1024],tensors={'block_table':dict(bytes=320,sha256=hashlib.sha256(struct.pack('<80i',*table)).hexdigest())},
            logical_kv={'key':{'sha256':'1'*64},'value':{'sha256':'2'*64}},scalars={'input_layout':'TND','sparse_mode':3})
        moved=deepcopy(value);moved['block_table_values'][0][:65]=reversed(range(65));moved['writer_consumer_slot']=[0]
        moved['tensors']['block_table']['sha256']=hashlib.sha256(struct.pack('<80i',*moved['block_table_values'][0])).hexdigest()
        self.assertNotEqual(value,moved);self.assertEqual(review.normalized(value),review.normalized(moved))
        for change in ('duplicate','outside','slot','stale_hash'):
            bad=deepcopy(moved)
            if change=='duplicate':bad['block_table_values'][0][1]=64
            elif change=='outside':bad['block_table_values'][0][-1]=128
            elif change=='slot':bad['writer_consumer_slot']=[1]
            else:bad['tensors']['block_table']['sha256']='0'*64
            with self.subTest(change=change),self.assertRaises(ValueError):review.normalized(bad)
        for change in ('unused','kv','scalar'):
            bad=deepcopy(moved)
            if change=='unused':
                bad['block_table_values'][0][-1]=1
                bad['tensors']['block_table']['sha256']=hashlib.sha256(struct.pack('<80i',*bad['block_table_values'][0])).hexdigest()
            elif change=='kv':bad['logical_kv']['key']['sha256']='3'*64
            else:bad['scalars']['pre_tokens']=2147483647
            self.assertNotEqual(review.normalized(value),review.normalized(bad))

    def test_raw_outputs_use_actual_dtype_bytes_and_reject_changed_and_nonfinite_data(self):
        helper=fixtures.evidence.helpers(fixtures.source())
        with tempfile.TemporaryDirectory() as folder:
            run=Path(folder);path=run/'logits.bin';path.write_bytes(struct.pack('<2f',1,2))
            row=dict(path='/evidence/probe/logits.bin',metadata={'shape':[1,2],'dtype':'torch.float32'},
                bytes=8,sha256=review.E.sha(path),finite=True)
            result=review.raw_tensor(run,row,[1,2],'torch.float32',helper,row['path'])
            self.assertEqual(result['bytes'],8)
            with self.assertRaises(ValueError):review.raw_tensor(run,row,[1,2],'torch.float16',helper,row['path'])
            path.write_bytes(struct.pack('<2f',1,3))
            with self.assertRaises(ValueError):review.raw_tensor(run,row,[1,2],'torch.float32',helper,row['path'])
            path.write_bytes(struct.pack('<2f',1,float('nan')));row['sha256']=review.E.sha(path)
            with self.assertRaises(ValueError):review.raw_tensor(run,row,[1,2],'torch.float32',helper,row['path'])

    def test_new_entrypoints_and_full_closure_cold_import_without_framework_or_dso(self):
        code=r'''
import runpy,sys
from pathlib import Path
root=Path(sys.argv[1]);folder=root/'experiments/c2c3_closure_20261117'
sys.path[:0]=[str(folder),str(root/'src'),str(root)]
import arch22_normal_bridge_evidence as e
sys.addaudithook(e.prohibit_runtime)
for name in e.REQUIRED+e.S2_REQUIRED:assert (root/name).is_file(),name
for name in ('prepare_arch22_s2_single_cases','review_arch22_s2_single','natural33_arch22_s2_worker_core_probe'):
    runpy.run_path(str(folder/(name+'.py')),run_name='cold_import_contract')
assert not any(x.split('.')[0] in ('torch','torch_npu','vllm','vllm_ascend') for x in sys.modules)
print('cold-cpu-source-closure-only')
'''
        result=subprocess.run([sys.executable,'-B','-c',code,str(ROOT)],capture_output=True,text=True,timeout=30)
        self.assertEqual(result.returncode,0,result.stdout+result.stderr)
        self.assertEqual(result.stdout.strip(),'cold-cpu-source-closure-only')


if __name__=='__main__':unittest.main()
