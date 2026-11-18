"""Metadata-only adversarial fixtures; no Torch, SDK or original private data."""
import copy
import io
import json
import math
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile
import unittest

from experiments.worker_core_v11_20261118 import review_quant_container_observations as audit


def native_fixture():
    descriptor = dict(descriptor=99, address=10000, counts=[2, 2, 1], called_mask=127,
        null_mask=0, truncated_mask=0, complete=1, offset=0, shape=[8, 2] + [0]*6,
        strides=[2, 1] + [0]*6, storage=[16] + [0]*7, dtype=27, format=2, statuses=[0]*7)
    events = []
    for call, op, handle, registration in ((1, 1, 5000, 1), (2, 2, 5000, 1),
                                          (3, 3, 6000, 3), (4, 4, 6000, 3), (5, 4, 0, 0)):
        for phase in (1, 2):
            row = dict(sequence=len(events)+1, call=call, operation=op, phase=phase,
                original_calls=phase-1, error=0, status=0, reserved=0, pid=1, thread=1, depth=1,
                scope_token=0, forward=0, operator_sequence=0, boundary_sequence=0, stage=0,
                caller_kind=1, caller_base=2000, caller_offset=100, caller=2100,
                function_base=1000, function_offset=op*16, function=1000+op*16,
                monotonic_ns=len(events)+1, handle=handle if phase==2 or op%2==0 else 0,
                registration=registration if phase==2 or op%2==0 else 0,
                count=1 if op==1 else 2 if op==3 else 0, payload_bytes=0, payload_offset=0)
            if phase==1 and op==1:
                row.update(payload_bytes=304, tensor_descriptors=[descriptor])
            if phase==1 and op==3:
                row.update(payload_bytes=16, payload_offset=304, integer_values=[0,64])
            events.append(row)
    state = dict(total_calls=5, total_events=10, retained_events=10, installed=1, stopped=1,
        resolver_ready=1, active_calls=0, scope_token=0, live_registrations=0, registration_count=0,
        first_error=0, abi_exceptions=0, getter_capture_failures=0, overflow=0, foreign_scope_thread_calls=0,
        owner_thread=1, entries=[1,1,1,2], returns=[1,1,1,2], original_calls=[1,1,1,2],
        outside_scope_calls=5, metadata_bytes=320,
        arrays=dict(capture_calls=1, getter_calls=3, acquired_arrays=3, released_arrays=3,
            active_captures=0, active_getters=0, live_arrays=0, getter_failures=0,
            getter_exceptions=0, capture_exceptions=0, ownership_errors=0))
    profile = dict(images={name: dict(segments=[dict(vaddr=0, size=1000)])
                          for name in ('nn','torch_npu','custom_extension')})
    return dict(events=events,state=state), profile


def window_fixture():
    worker = audit.ROOT / 'src/demoncat/runtime/vllm_ascend/worker_extension'
    constants = audit.literal_constants(worker/'quant_token_routes.py'); schemas=constants['SCHEMAS']
    reviewed={}; sources={}
    for file in ('quant_token_routes.py','quant_route_frames.py'):
        for module,(sha,functions) in audit.literal_constants(worker/file)['REVIEWED'].items():
            sources[module]=dict(sha256=sha)
            for function,role in functions.items():reviewed[role]=(sha,function)
    next_id=0
    def tensor(shape,dtype):
        nonlocal next_id
        next_id+=1
        width={'torch.int8':1,'torch.bfloat16':2,'torch.float32':4,'torch.int32':4,'torch.int64':8}[dtype]
        stride=[math.prod(shape[n+1:]) for n in range(len(shape))]
        return dict(tensor=dict(object_id=next_id,storage_identity=next_id,storage_address=10000*next_id,
            storage_bytes=math.prod(shape)*width,offset=0,shape=shape,stride=stride,dtype=dtype,device='npu:0',
            version=None,inference_version_unavailable=True))
    args={
        'routing': dict(x=tensor([8,2048],'torch.bfloat16'),expert_idx=tensor([8,6],'torch.int32'),
            active_num=48,expert_capacity=-1,expert_num=64,drop_pad_mode=0,expert_tokens_num_type=1,
            expert_tokens_num_flag=True,quant_mode=1,active_expert_range=[0,64],row_idx_type=0,scale=None,offset=None),
        'gmm1':dict(x=tensor([48,2048],'torch.int8'),weight=tensor([64,2048,2816],'torch.int8'),
            group_list=tensor([64],'torch.int64'),weight_scale=tensor([64,2816],'torch.float32'),
            x_scale=tensor([48],'torch.float32'),bias=None,offset=None),
        'gmm2':dict(x=[tensor([48,1408],'torch.int8')],weight=[tensor([64,1408,2048],'torch.int8')],
            scale=[tensor([64,2048],'torch.bfloat16')],per_token_scale=[tensor([48],'torch.float32')],
            group_list=tensor([64],'torch.int64'),split_item=2,group_type=0,group_list_type=1,act_type=0,
            output_dtype='torch.bfloat16',bias=None,offset=None,antiquant_scale=None,antiquant_offset=None,
            activation_input=None,activation_quant_scale=None,activation_quant_offset=None,tuning_config=None),
        'unpermute':dict(permuted_tokens=tensor([48,2048],'torch.bfloat16'),sorted_indices=tensor([48],'torch.int32'),
            probs=tensor([8,6],'torch.bfloat16'),padded_mode=False,restore_shape=None)}
    scopes=[];events=[];raw=[];frames=[dict(sequence=1,role='rchain_model',parent_source_sequence=None)]
    for number,(stage,role) in audit.STAGES.items():
        sha,function=reviewed[role];meta=audit.tensor_packets(args[stage]);operator=schemas[stage].split('(')[0].replace('::','.')+'.default'
        events.append(dict(sequence=number,operator=operator,original_calls=1,status='OBSERVED',mutable=False,
            schema=schemas[stage],schema_sha256=audit.digest(audit.canonical(schemas[stage])),inputs=meta,
            outputs=[args['unpermute']['permuted_tokens']['tensor']] if stage=='gmm2' else []))
        frames.append(dict(sequence=number+1,role=role,status='OBSERVED',function=function,source_sha256=sha,
            operator_sequences=[number],parent_source_sequence=1))
        inputs=[dict(tensor=m,write_epoch=0) for m in meta]
        scopes.append(dict(native_scope=dict(version=1,size=56,token=number,thread=1,forward=1,
            stage=number,operator_sequence=number,boundary_sequence=number+1),status='RETURNED',original_calls=1,
            original_error=None,leave_attempts=[dict(native_status=0,observed_error=None,observed_original_calls=1)],
            after_input_metadata=copy.deepcopy(inputs),evidence=dict(forward_id='f',request_ids=['r'+str(i) for i in range(8)],
                stage=stage,operator=operator,schema=schemas[stage],operator_sequence=number,boundary_sequence=number+1,
                function=function,source_sha256=sha,formals=args[stage],argument_origins={k:'KEYWORD' for k in args[stage]},
                input_objects=inputs)))
    def raw_event(stage,op,count,**extra):
        return dict(scope_token=stage,forward=1,operator_sequence=stage,boundary_sequence=stage+1,
            stage=stage,thread=1,caller_kind=2 if stage==1 else 1,operation=op,phase=1,count=count,**extra)
    raw.append(raw_event(1,3,2,integer_values=[0,64]))
    members=[args['gmm2'][name][0]['tensor'] for name in ('x','weight','scale','per_token_scale')]
    members.append(args['unpermute']['permuted_tokens']['tensor'])
    for index,member in enumerate(members):
        dtype={'torch.int8':2,'torch.bfloat16':27,'torch.float32':0}[member['dtype']]
        storage=[math.prod(member['shape'])] if index!=1 else [64,64,88,16,32]
        descriptor=dict(address=member['storage_address'],offset=0,counts=[len(member['shape']),len(member['stride']),len(storage)],
            shape=member['shape'],strides=member['stride'],storage=storage,dtype=dtype,format=29 if index==1 else 2)
        raw.append(raw_event(3,1,1,tensor_descriptors=[descriptor]))
    raw.append(raw_event(4,3,1,integer_values=[123456789]))
    boundary=dict(sources=sources,boundaries=frames,errors=[],native_qualified=False,integer_routes=None,routed_call_frames=None)
    graph=dict(forward_id='f',actual_calls=4,retained_calls=4,missing_calls=0,errors=[],native_qualified=False,
        events=events,source_boundaries=boundary)
    observation=dict(state=dict(owner_thread=1),public_scopes=scopes,events=raw,
        source_scopes=dict(calls=4,errors=[],active=0,native_qualified=False,request_rows_proven=False))
    return observation,graph,dict(forward_id='f',request_ids=['r'+str(i) for i in range(8)])


class ContainerOriginalReviewTests(unittest.TestCase):
    def test_independent_import_does_not_load_torch_or_native(self):
        script=('import sys; from experiments.worker_core_v11_20261118 import review_quant_container_observations; '
                'assert "torch" not in sys.modules; assert "ctypes" not in sys.modules')
        subprocess.run([sys.executable,'-B','-c',script],cwd=audit.ROOT,check=True,capture_output=True)

    def test_exact_pairs_count_null_destroy_separately(self):
        o,p=native_fixture();value=audit.review_calls(o,p)
        self.assertEqual(value['created_nonnull_lifetimes'],2)
        self.assertEqual(value['retired_nonnull_lifetimes'],2)
        self.assertEqual(value['null_destroy_calls'],1)

    def test_missing_original_and_late_wrong_registration_refuse(self):
        mutations=(lambda o:o['events'].pop(1),lambda o:o['events'][3].update(registration=99),
            lambda o:o['events'][1].update(original_calls=2),lambda o:o['events'][3].update(handle=999),
            lambda o:o['state'].update(outside_scope_calls=4),lambda o:o['events'][0].update(thread=2))
        for mutate in mutations:
            o,p=native_fixture();mutate(o)
            with self.assertRaises(ValueError):audit.review_calls(o,p)

    def test_reused_handle_needs_fresh_registration(self):
        o,p=native_fixture()
        for index in (5,6,7):o['events'][index]['handle']=5000
        self.assertEqual(audit.review_calls(o,p)['reused_handle_registrations'],1)
        o['events'][5]['registration']=1
        with self.assertRaisesRegex(ValueError,'FRESH_CURRENT_REGISTRATION'):audit.review_calls(o,p)

    def test_pc_profile_and_native_payload_are_not_summary_flags(self):
        mutations=(lambda o:o['events'][0].update(caller_kind=0),lambda o:o['events'][0].update(function_offset=2000),
            lambda o:o['events'][4].update(integer_values=[0]),lambda o:o['events'][0].update(payload_offset=8),
            lambda o:o['state']['arrays'].update(released_arrays=2),
            lambda o:o['events'][0]['tensor_descriptors'][0].update(truncated_mask=1))
        for mutate in mutations:
            o,p=native_fixture();mutate(o)
            with self.assertRaises(ValueError):audit.review_calls(o,p)

    def test_current_four_windows_remain_nonsemantic(self):
        o,g,f=window_fixture();v=audit.review_windows(o,g,f,audit.ROOT,expected_layers=1)
        self.assertEqual(v['source_windows'],4)
        self.assertEqual(v['gmm2_current_descriptor_matches'],dict(x=1,weight=1,scale=1,per_token_scale=1,returned_output=1))
        self.assertFalse(v['descriptor_matches_assign_gws_roles']);self.assertFalse(v['routed_rows_granted'])
        self.assertEqual(v['unpermute_integer_arrays_without_gws_parameter_role'],1)

    def test_same_shape_different_storage_or_unknown_frame_refuses(self):
        mutations=(lambda o,g:o['events'][1]['tensor_descriptors'][0].update(address=999),
            lambda o,g:o['public_scopes'][0]['evidence'].update(source_sha256='a'*64),
            lambda o,g:g['source_boundaries']['boundaries'][2].update(parent_source_sequence=None),
            lambda o,g:o['public_scopes'][1]['native_scope'].update(thread=9),
            lambda o,g:o['public_scopes'][2]['evidence']['formals'].update(group_list_type=0),
            lambda o,g:o['public_scopes'][0]['evidence'].update(request_ids=['other']))
        for mutate in mutations:
            o,g,f=window_fixture();mutate(o,g)
            with self.assertRaises(ValueError):audit.review_windows(o,g,f,audit.ROOT,expected_layers=1)

    def test_epoch_drift_refuses_without_version_counter(self):
        o,g,f=window_fixture();m=o['public_scopes'][1]['evidence']['input_objects'][0]['tensor']
        g['events'][0]['invalidated_storages']=[dict(device=m['device'],address=m['storage_address'],
            bytes=m['storage_bytes'],storage_identity=m['storage_identity'],write_epoch=1)]
        with self.assertRaisesRegex(ValueError,'CURRENT_ARGUMENT_EPOCH_CHANGED'):
            audit.review_windows(o,g,f,audit.ROOT,expected_layers=1)

    def test_foreign_source_error_or_incomplete_scope_does_not_pass(self):
        for key,value in (('errors',['old_error']),('active',1),('calls',3)):
            o,g,f=window_fixture();o['source_scopes'][key]=value
            with self.assertRaises(ValueError):audit.review_windows(o,g,f,audit.ROOT,expected_layers=1)
        o,g,f=window_fixture();o['public_scopes'][2]['leave_attempts'][0]['native_status']=1
        with self.assertRaises(ValueError):audit.review_windows(o,g,f,audit.ROOT,expected_layers=1)

    def test_gmm_weight_nz_storage_is_preserved_not_relabelled(self):
        o,g,f=window_fixture();d=o['events'][2]['tensor_descriptors'][0]
        self.assertEqual(d['format'],29);self.assertEqual(d['counts'][2],5)
        audit.review_windows(o,g,f,audit.ROOT,expected_layers=1)
        self.assertEqual(d['format'],29)
        d['storage'][-1]=33
        with self.assertRaisesRegex(ValueError,'CURRENT_CREATOR_PUBLIC_MEMBER_JOIN_AMBIGUOUS'):
            audit.review_windows(o,g,f,audit.ROOT,expected_layers=1)

    def test_archive_requires_exact_sha_members_and_regular_files(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'original.tar.gz'
            for changed in (False,True):
                with tarfile.open(path,'w:gz') as t:
                    for index,name in enumerate(audit.MEMBERS):
                        info=tarfile.TarInfo(name);info.size=2
                        if changed and index==0:
                            info.type=tarfile.SYMTYPE;info.linkname='/outside';info.size=0
                            t.addfile(info)
                        else:t.addfile(info,io.BytesIO(b'{}'))
                sha=audit.digest(path.read_bytes())
                if changed:
                    with self.assertRaisesRegex(ValueError,'REGULAR_UNIQUE'):
                        audit.load_archive(path,sha)
                else:
                    values,hashes=audit.load_archive(path,sha);self.assertEqual(len(values),8);self.assertEqual(len(hashes),8)
                    with self.assertRaisesRegex(ValueError,'ORIGINAL_PACKAGE_BYTES_CHANGED'):
                        audit.load_archive(path,'0'*64)


if __name__=='__main__':unittest.main()
