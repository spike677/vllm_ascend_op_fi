"""CPU-only byte fixtures; no model/Worker, device, shared library or qualification."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import shutil
import struct
import subprocess
import sys
import tempfile
import unittest

from experiments.c2c3_closure_20261117 import review_attention33_first_boundary_source53 as review
from experiments.c2c3_closure_20261117 import diagnose_attention33_baselines as difference_reader


def hash_bytes(value): return hashlib.sha256(value).hexdigest()


def metadata(shape, dtype='torch.float16', identity=10, storage_bytes=None):
    stride = []; count = 1
    for size in reversed(shape): stride.insert(0,count); count *= size
    return dict(shape=shape, stride=stride, storage_offset=0, dtype=dtype, device='npu:0',
        object_id=identity, storage_identity=identity+1000,
        storage_nbytes=count*review.FORMATS[dtype][1] if storage_bytes is None else storage_bytes)


def boundary_fixture(run):
    """Serialized schema fixture built from CPU bytes, never a real NPU record."""
    name = 'baseline_before'; folder = run/name/'first_boundary'; folder.mkdir(parents=True)
    private = run/name/'private'; private.mkdir()
    counter = 20
    def snapshot(raw, shape, dtype='torch.float16', filename=None):
        nonlocal counter
        counter += 1
        if filename: (folder/filename).write_bytes(raw)
        return dict(metadata=metadata(shape,dtype,counter),version=None,bytes=len(raw),sha256=hash_bytes(raw),finite=True,
            path='/evidence/run/'+name+'/first_boundary/'+filename if filename else None)
    query_bytes = struct.pack('<2048e', *(i/2048 for i in range(2048)))
    query = snapshot(query_bytes,[1,16,128],filename='query.bin')
    mask = snapshot(b'\x00\x01\x01\x00',[2,2],'torch.bool')
    values = [list(range(65))+[0]*15]
    table = snapshot(struct.pack('<80i',*values[0]),[1,80],'torch.int32')
    qout = snapshot(query_bytes,[1,16,128],filename='output-0.bin')
    empty = snapshot(b'',[0],'torch.float32',filename='output-1.bin')
    (private/'layer-0-output.bin').write_bytes(query_bytes)
    query_cache_meta = {}; logical = {}; cache_rows = {}
    for index,kind in enumerate(('key','value')):
        actual = metadata([128,16,1024],identity=50+index)
        live = metadata([128,16,8,128],identity=70+index)
        live['storage_identity'] = actual['storage_identity']
        query_cache_meta[kind] = actual
        raw = struct.pack('<e',index+1)*1025*8*128
        (private/('layer-0-'+kind+'.bin')).write_bytes(raw)
        logical[kind] = dict(metadata=live,version=None,shape=[1025,8,128],bytes=len(raw),sha256=hash_bytes(raw),finite=True,
            physical_pages=values[0][:65],readback='ACTUAL_VALID_LOGICAL_PAGE_ORDER',operator_input_repacked=False)
        cache_rows[kind] = dict(path='/evidence/run/'+name+'/private/layer-0-'+kind+'.bin',shape=[1025,8,128],
            dtype='torch.float16',bytes=len(raw),sha256=hash_bytes(raw),finite=True,physical_pages=values[0][:65],
            readback='VALID_LOGICAL_PAGE_ORDER_AFTER_CONSUMER',operator_input_repacked=False)
    bound = dict(module_path=review.FIRST,module_object_id=4,impl_object_id=5,metadata_binding='VERIFIED',
        query=query['metadata'],key=query_cache_meta['key'],value=query_cache_meta['value'],page_table=table['metadata'],
        live_key_cache=logical['key']['metadata'],live_value_cache=logical['value']['metadata'],used_pages=[values[0][:65]],
        slots=[1024],sequence_lengths=[1025],scheduler={'request_ids':['actual-fixture-request']},writer={'returned':True})
    inputs = dict(query=dict(kind='tensor',snapshot=query),atten_mask=dict(kind='tensor',snapshot=mask),
        block_table=dict(kind='tensor',snapshot=table),key=dict(kind='paged_cache',metadata=bound['key'],version=None),
        value=dict(kind='paged_cache',metadata=bound['value'],version=None),input_layout=dict(kind='scalar',value='TND'),
        block_size=dict(kind='scalar',value=16),sparse_mode=dict(kind='scalar',value=3),scale=dict(kind='scalar',value=.125),
        optional_tensor=dict(kind='scalar',value=None))
    after = {name:deepcopy(item['snapshot']) for name,item in inputs.items() if item['kind']=='tensor'}
    for item in after.values(): item['path'] = None
    output_after = deepcopy([qout,empty])
    for item in output_after: item['path'] = None
    check_names = set(inputs)|{'logical_key','logical_value','impl_key_same_object','impl_value_same_object','output_0','output_1'}
    value = dict(schema='demoncat.paged33.first_decode_boundary.v1',state='FINALIZED',execution='NATURAL_WORKER_DIAGNOSTIC',
        errors=[],original_calls=1,input_substitutions=0,operator_arguments_modified=False,fault_enabled=False,c2=False,c3=False,
        rank=0,pid=12,thread=12345,worker_object_id=1,model_object_id=2,boundary=deepcopy(bound),
        supplied_argument_names=sorted(inputs),positional_count=0,original_callable=dict(type='torch._ops.OpOverloadPacket',object_id=777),
        public_default_schema='npu::fixture(Tensor query) -> Tensor',omitted_argument_defaults={'softmax_lse_flag':False},
        default_schema_dispatch_observed=False,inputs=inputs,inputs_after=after,logical_kv_before=logical,logical_kv_after=deepcopy(logical),
        outputs=[qout,empty],outputs_after=output_after,return_container_type='tuple',block_table_values=values,subsequent_checks={name:True for name in check_names},
        all_checked_values_unchanged=True,native_call_id='a'*32,
        runtime_observation=dict(status='UNOBSERVED_OR_INCOMPLETE',error="ContextRefused('BOUNDARY_ACTUAL_FUNCTION_JOIN_MISSING')"),
        mapped_api_libraries=dict(selected_api_owner_observed=False,files=[]))
    cache_rows['output'] = dict(path='/evidence/run/'+name+'/private/layer-0-output.bin',shape=[1,16,128],dtype='torch.float16',
        bytes=len(query_bytes),sha256=hash_bytes(query_bytes),finite=True)
    phase = dict(phase=name,finish=dict(capture=dict(first_decode_boundary=value,attention={review.FIRST:cache_rows})))
    (folder/'boundary.json').write_text(json.dumps(value))
    return phase,bound


class RuntimeResolutionTests(unittest.TestCase):
    def setUp(self):
        self.run = Path(self.enterContext(tempfile.TemporaryDirectory())); (self.run/'runtime').mkdir()
        binary = self.run/'runtime/registered-12-1.bin'; binary.write_bytes(b'actual preserved CPU binary fixture')
        self.token = 'a'*32
        def row(event,now,**fields):
            return dict(schema=review.SCHEMA,event=event,pid=12,tid=34,monotonic_ns=now,call_id='',**fields)
        self.rows = [row('register',1,handle=80,generation=1,binary_file=binary.name,binary_bytes=binary.stat().st_size,result=0)]
        for now in range(2,30):
            self.rows.append(row('function_bind',now,handle=80,generation=1,function_handle=90,tiling_key='103000000010300003',result=0))
        scope = [row('begin',40),row('function_bind',41,handle=80,generation=1,function_handle=90,tiling_key='103000000010300003',result=0),
            row('launch',45,api='aclrtLaunchKernelWithHostArgs',handle=80,generation=1,function_handle=90,binary_file=binary.name,
                tiling_key='103000000010300003',block_dim=16,schedule_mode=1,launch_begin_ns=42,launch_end_ns=44,stream=9,
                result=0,snapshot_valid=True,observer_mutates_arguments=False,args_hex='00010203',args_size=4,
                placeholders_hex='',argument_layout='aclrt_host_placeholders'),row('stream_sync',46,stream=9,result=0),
            row('device_sync',47,result=0),row('end',50)]
        for record in scope: record['call_id'] = self.token
        self.rows += scope

    def resolved(self): return review.scoped_runtime(self.run,self.rows,self.token,expected_pid=12)

    def test_current_scope_resolves_29_identical_bindings_without_rewriting_gap(self):
        result = self.resolved(); launch, = result['launches']
        self.assertEqual(launch['function_binding']['repeated_identical_history'],28)
        self.assertEqual(result['event_counts']['function_bind'],1)
        self.assertEqual(launch['args_sha256'],hash_bytes(bytes.fromhex('00010203')))
        self.assertFalse(result['original_runtime_observation_rewritten'])
        self.assertFalse(result['stock_private_abi_assumption'])
        self.assertFalse(result['selected_aclnn_owner_observed'])

    def test_conflicting_old_function_key_handle_or_generation_cannot_be_filtered_away(self):
        for key,value in (('tiling_key','5100000000010200103'),('handle',81),('generation',2),('result',1)):
            with self.subTest(key=key):
                previous = self.rows[2][key]; self.rows[2][key] = value
                with self.assertRaisesRegex(ValueError,'CONFLICTING_FUNCTION'): self.resolved()
                self.rows[2][key] = previous

    def test_history_match_without_current_scope_bind_is_insufficient(self):
        self.rows[-5]['call_id'] = 'b'*32
        with self.assertRaisesRegex(ValueError,'CURRENT_SCOPE_FUNCTION_BIND'): self.resolved()

    def test_later_foreign_scope_bind_overrides_current_and_is_refused(self):
        record = deepcopy(self.rows[-5]); record.update(call_id='b'*32,monotonic_ns=42,tid=99)
        self.rows.insert(-4,record)
        with self.assertRaisesRegex(ValueError,'LATEST_EFFECTIVE_BIND'): self.resolved()

    def test_duplicate_current_scope_bind_refused(self):
        record = deepcopy(self.rows[-5]); record['monotonic_ns'] = 42; self.rows.insert(-4,record)
        with self.assertRaisesRegex(ValueError,'CURRENT_SCOPE_FUNCTION_BIND'): self.resolved()

    def test_wrong_scope_owner_and_future_binding_refused(self):
        self.rows[-5]['tid'] = 35
        with self.assertRaisesRegex(ValueError,'SCOPED_EVENT_OWNER'): self.resolved()
        self.rows[-5]['tid'] = 34; self.rows[-5]['monotonic_ns'] = 43
        with self.assertRaisesRegex(ValueError,'CURRENT_SCOPE_FUNCTION_BIND'): self.resolved()

    def test_future_other_binding_does_not_retroactively_change_observed_launch(self):
        record = deepcopy(self.rows[-5]); record.update(call_id='b'*32,monotonic_ns=60,tiling_key='777')
        self.rows.append(record)
        self.assertEqual(len(self.resolved()['launches']),1)

    def test_missing_completion_sync_refused(self):
        self.rows = [row for row in self.rows if row['event'] not in ('device_sync','stream_sync')]
        with self.assertRaisesRegex(ValueError,'COMPLETION_SYNC'): self.resolved()

    def test_invalid_argument_snapshot_or_corrupt_size_refused(self):
        launch = next(row for row in self.rows if row['event']=='launch')
        launch['snapshot_valid'] = False
        with self.assertRaisesRegex(ValueError,'SNAPSHOT_INVALID'): self.resolved()
        launch['snapshot_valid'] = True; launch['args_size'] = 8
        with self.assertRaisesRegex(ValueError,'RAW_ARGUMENT_SNAPSHOT_SIZE'): self.resolved()

    def test_stale_generation_and_deleted_binary_refused(self):
        record = deepcopy(self.rows[0]); record.update(monotonic_ns=31,generation=2)
        self.rows.insert(29,record)
        with self.assertRaisesRegex(ValueError,'STALE_BINARY'): self.resolved()
        self.rows.remove(record); (self.run/'runtime/registered-12-1.bin').unlink()
        with self.assertRaises(FileNotFoundError): self.resolved()

    def test_unrelated_unscoped_capture_gap_is_preserved_without_inventing_coverage(self):
        row = dict(schema=review.SCHEMA,event='capture_overflow',pid=12,tid=55,monotonic_ns=30,call_id='')
        self.rows.insert(29,row)
        result = self.resolved()
        self.assertEqual(result['preceding_unscoped_capture_gaps'],{'capture_overflow':1})
        self.assertFalse(result['all_runtime_entrypoints_covered'])

    def test_runtime_file_requires_complete_json_and_stable_sha(self):
        path = self.run/'runtime/runtime.jsonl'
        raw = ''.join(json.dumps(row)+'\n' for row in self.rows).encode(); path.write_bytes(raw)
        rows,report = review.read_runtime(self.run)
        self.assertEqual(rows,self.rows); self.assertEqual(report['sha256'],hash_bytes(raw))
        path.write_bytes(raw[:-1])
        with self.assertRaisesRegex(ValueError,'STABLE_COMPLETE_FILE'): review.read_runtime(self.run)

    def test_multiple_intercepted_launches_remain_visible_without_private_key_classification(self):
        bind = deepcopy(self.rows[-5]); bind.update(monotonic_ns=46,function_handle=91,tiling_key='777')
        launch = deepcopy(self.rows[-4]); launch.update(monotonic_ns=49,launch_begin_ns=47,launch_end_ns=48,
            function_handle=91,tiling_key='777',block_dim=9)
        self.rows[-3]['monotonic_ns'] = 52; self.rows[-2]['monotonic_ns'] = 53; self.rows[-1]['monotonic_ns'] = 55
        self.rows[-3:-3] = [bind,launch]
        result = self.resolved()
        self.assertEqual([row['tiling_key'] for row in result['launches']],['103000000010300003','777'])
        self.assertTrue(all(row['kernel_argument_abi']=='NOT_DECODED' for row in result['launches']))


class BoundaryByteReviewTests(unittest.TestCase):
    def setUp(self):
        self.run = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.phase,self.bound = boundary_fixture(self.run)

    def reviewed(self):
        return review.first_boundary(self.run,self.phase,self.bound,difference_reader=difference_reader)

    def persist(self):
        value = self.phase['finish']['capture']['first_decode_boundary']
        (self.run/'baseline_before/first_boundary/boundary.json').write_text(json.dumps(value))
        return value

    def test_complete_query_outputs_empty_optional_and_matching_kv_post_files(self):
        result = self.reviewed()
        self.assertEqual(result['query']['bytes'],4096)
        self.assertEqual(result['outputs'][1]['bytes'],0)
        self.assertTrue(result['summary']['block_table_full_values_rehashed'])
        self.assertFalse(result['summary']['mask_raw_bytes_rehashed'])
        self.assertFalse(result['summary']['logical_kv']['key']['pre_boundary_file_preserved'])
        self.assertEqual(result['summary']['original_runtime_observation']['status'],'UNOBSERVED_OR_INCOMPLETE')

    def test_original_gap_is_immutable_in_review_output(self):
        before = deepcopy(self.phase)
        self.reviewed()
        self.assertEqual(self.phase,before)

    def test_query_file_corruption_rejected(self):
        path = self.run/'baseline_before/first_boundary/query.bin'
        raw = bytearray(path.read_bytes()); raw[1] ^= 1; path.write_bytes(raw)
        with self.assertRaisesRegex(ValueError,'RAW_SNAPSHOT_HASH_CHANGED'): self.reviewed()

    def test_same_value_input_object_replacement_at_checkpoint_rejected(self):
        value = self.persist(); value['inputs_after']['query']['metadata']['object_id'] += 1; self.persist()
        with self.assertRaisesRegex(ValueError,'INPUT_CHECKPOINT_CHANGED'): self.reviewed()

    def test_mask_checkpoint_change_and_missing_required_mask_rejected(self):
        value = self.persist(); value['inputs_after']['atten_mask']['sha256'] = '0'*64; self.persist()
        with self.assertRaisesRegex(ValueError,'INPUT_CHECKPOINT_CHANGED:atten_mask'): self.reviewed()
        value['inputs_after']['atten_mask']['sha256'] = value['inputs']['atten_mask']['snapshot']['sha256']
        value['inputs'].pop('atten_mask'); value['supplied_argument_names'] = sorted(value['inputs']); self.persist()
        with self.assertRaisesRegex(ValueError,'ACTUAL_QKV_MASK_TABLE_REQUIRED'): self.reviewed()

    def test_complete_block_table_padding_is_rehashed_not_only_used_pages(self):
        value = self.persist(); value['block_table_values'][0][-1] = 99; self.persist()
        with self.assertRaisesRegex(ValueError,'BLOCK_TABLE_FULL_HASH'): self.reviewed()

    def test_logical_kv_pre_hash_is_not_substituted_from_post_file(self):
        value = self.persist(); value['logical_kv_before']['key']['sha256'] = '0'*64; self.persist()
        with self.assertRaisesRegex(ValueError,'PRE_POST_LOGICAL_KV'): self.reviewed()

    def test_changed_kv_post_file_rejected(self):
        path = self.run/'baseline_before/private/layer-0-key.bin'
        with path.open('r+b') as stream: stream.write(b'\x00\x00')
        with self.assertRaisesRegex(ValueError,'LOGICAL_KV_POST_FILE_HASH'): self.reviewed()

    def test_delayed_output_mutation_rejected_even_when_old_boolean_is_true(self):
        value = self.persist(); value['outputs_after'][0]['sha256'] = '0'*64; self.persist()
        self.assertTrue(value['all_checked_values_unchanged'])
        with self.assertRaisesRegex(ValueError,'IMMEDIATE_OUTPUT_CHECKPOINT_CHANGED'): self.reviewed()

    def test_wrong_layer_or_different_boundary_object_refused(self):
        value = self.persist(); value['boundary']['module_path'] = 'model.layers.1.self_attn.attn'; self.persist()
        with self.assertRaisesRegex(ValueError,'FIRST_BOUNDARY_ACTUAL_CALL_JOIN'): self.reviewed()

    def test_missing_scalar_optional_argument_changes_complete_signature(self):
        before = self.reviewed(); value = self.persist()
        value['inputs']['optional_tensor']['value'] = 7; self.persist()
        after = self.reviewed()
        self.assertNotEqual(before['input_signature'],after['input_signature'])
        self.assertNotEqual(before['input_sha256'],after['input_sha256'])

    def test_one_call_no_substitution_and_no_fault_are_required(self):
        value = self.persist(); value['original_calls'] = 2; self.persist()
        with self.assertRaisesRegex(ValueError,'ORIGINAL_ONCE'): self.reviewed()
        value['original_calls'] = 1; value['input_substitutions'] = 1; self.persist()
        with self.assertRaisesRegex(ValueError,'ORIGINAL_ONCE'): self.reviewed()

    def test_numeric_difference_retains_signed_zero_bit_difference(self):
        a = self.run/'a.bin'; b = self.run/'b.bin'
        a.write_bytes(struct.pack('<2e',0,1)); b.write_bytes(struct.pack('<2e',-0.,1.5))
        def row(path): return dict(path=str(path),shape=[2],dtype='torch.float16',bytes=4,sha256=review.sha(path))
        result = review.difference(difference_reader,row(a),row(b))
        self.assertEqual(result['changed_elements'],2)
        self.assertEqual(result['numeric_changed_elements'],1)
        self.assertEqual(result['max_abs_difference'],.5)

    def test_empty_return_has_no_fabricated_difference(self):
        item = self.reviewed()['outputs'][1]
        result = review.difference(difference_reader,item,item)
        self.assertTrue(result['exact']); self.assertEqual(result['changed_elements'],0)

    def test_path_escape_and_duplicate_json_are_refused(self):
        with self.assertRaisesRegex(ValueError,'EVIDENCE_PATH_INVALID'): review.safe_path(self.run,'/evidence/run/../file')
        path = self.run/'duplicate.json'; path.write_text('{"x":1,"x":2}')
        with self.assertRaisesRegex(ValueError,'DUPLICATE_JSON_KEY'): review.read(path)

    def test_reader_import_is_framework_free_and_blocks_runtime_actions(self):
        code = ('import sys; from experiments.c2c3_closure_20261117 import review_attention33_first_boundary_source53 as r; '
            'assert "torch" not in sys.modules; sys.addaudithook(r.prohibit_runtime); import torch')
        result = subprocess.run([sys.executable,'-B','-c',code],text=True,capture_output=True)
        self.assertNotEqual(result.returncode,0); self.assertIn('DEVICE_FRAMEWORK_IMPORT_REFUSED:torch',result.stderr)
        for event in ('ctypes.dlopen','subprocess.Popen','os.system','socket.connect'):
            with self.assertRaisesRegex(RuntimeError,'CPU_READ_ONLY_ACTION_REFUSED'): review.prohibit_runtime(event,())

    def test_frozen_byte_readers_load_only_after_manifest_and_file_verification(self):
        source = self.run/'source'; source.mkdir()
        workspace = Path(__file__).resolve().parents[1]; files = []
        for name in review.SOURCE_FILES:
            path = source/name; path.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(workspace/name,path)
            files.append(dict(path=name,sha256=review.sha(path),bytes=path.stat().st_size))
        manifest = source/'source_manifest.json'
        manifest.write_text(json.dumps(dict(base_commit=review.SOURCE,files=files)))
        code = ('import sys,numpy; from experiments.c2c3_closure_20261117 import review_attention33_first_boundary_source53 as r; '
            'r.MANIFEST=sys.argv[2]; sys.addaudithook(r.prohibit_runtime); '
            'a,b,c,s=r.load_frozen(sys.argv[1]); assert "torch" not in sys.modules; '
            'assert s["files_rehashed"]==len(r.SOURCE_FILES); print("FROZEN_CPU_READERS_ONLY")')
        command = [sys.executable,'-B','-c',code,str(source),review.sha(manifest)]
        result = subprocess.run(command,text=True,capture_output=True)
        self.assertEqual(result.returncode,0,result.stderr)
        self.assertIn('FROZEN_CPU_READERS_ONLY',result.stdout)
        self.assertEqual(list(source.rglob('__pycache__')),[])
        (source/(review.WORKER+'paged33_boundary_probe.py')).write_text('unreviewed source')
        result = subprocess.run(command,text=True,capture_output=True)
        self.assertNotEqual(result.returncode,0)
        self.assertIn('FROZEN_SOURCE_BYTES_CHANGED',result.stderr)

    def test_private_host_control_receipts_are_joined_to_runtime_without_stock_abi(self):
        folder = self.run/'baseline_before/native'; folder.mkdir()
        calls = [dict(call_id=format(i,'032x')) for i in range(28)]
        rows = [dict(schema='demoncat.paged33.fai_fd_host.v1',token=row['call_id'],pid=12,tid=34,
            eligible=True,policy_applied=False,mode='off',native_s2=512,selected_s2=512) for row in calls]
        path = folder/'host.jsonl'
        def save(): path.write_text(''.join(json.dumps(row)+'\n' for row in rows))
        save(); first = dict(call_id=calls[0]['call_id'],pid=12,tid=34)
        result = review.private_host_controls(self.run,'baseline_before',calls,first)
        self.assertEqual(result['callbacks'],28); self.assertEqual(result['policy_applied'],0)
        rows[-1]['policy_applied'] = True; save()
        with self.assertRaisesRegex(ValueError,'HOST_CONTROL_WAS_NOT_UNMODIFIED'):
            review.private_host_controls(self.run,'baseline_before',calls,first)
        rows[-1]['policy_applied'] = False; save(); first['tid'] = 99
        with self.assertRaisesRegex(ValueError,'HOST_RUNTIME_OWNER_JOIN'):
            review.private_host_controls(self.run,'baseline_before',calls,first)


if __name__ == '__main__': unittest.main()
