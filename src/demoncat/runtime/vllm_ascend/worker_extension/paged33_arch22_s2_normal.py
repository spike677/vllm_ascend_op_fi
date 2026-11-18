"""Only normal/OFF execution for the fresh S2-capable package; no F authority."""
import json
import os
from pathlib import Path
import threading
import uuid

from .paged33_arch22_bridge import (Arch22NormalNativeCall,validate_normal_call,
                                   validate_native,verify_normal_token_sources)
from .paged33_logical_binding import LogicalPagedInput
from .paged33_boundary_probe import FIRST_LAYER
from .paged33_tnd import _verify_bundle_artifacts
from .serving_context import require
from fault_injection.native_adapters.attention28.launch import sha,verify_launch
from fault_injection.native_adapters.attention28.fia import apply_tnd33_arch22_s2 as recipe
from fault_injection.native_adapters.attention28.fia.arch22_s2_build_evidence import verify_inputs,verify_normal_geometry,PREFIX


def verify_bundle(completion,completion_sha256,vendor,*,all_normal=False,group_s2=False):
    selected=recipe
    if all_normal:
        from fault_injection.native_adapters.attention28.fia import apply_tnd33_arch22_all_normal as selected
    if group_s2:
        from fault_injection.native_adapters.attention28.fia import apply_tnd33_arch22_group_s2 as selected
    bundle=_verify_bundle_artifacts(completion,completion_sha256,vendor,variant=selected.VARIANT,keys=selected.KEYS)
    data=bundle['completion'];folder=Path(recipe.__file__).parent
    names=('apply_tnd33_arch22_s2.py','apply_tnd33_arch22_bridge.py','paged33_arch22_s2.h',
           'paged33_arch22_bridge.h','arch22_s2_geometry.py','compile_arch22_s2_matrix.py')
    if all_normal:names+=('apply_tnd33_arch22_all_normal.py','paged33_arch22_all_normal.h')
    if group_s2:names+=('apply_tnd33_arch22_group_s2.py','paged33_arch22_group_s2.h','apply_tnd33_arch22_all_normal.py')
    bundle['s2_cpu_matrix']=verify_inputs(Path(completion).resolve().parent.parent,data,
                                         {PREFIX+name:sha(folder/name) for name in names},all_normal=all_normal,group_s2=group_s2)
    schema=Path(completion).resolve().parent/'arch22_schema'
    require(sha(schema/'compile_receipt.json')==data.get('arch22_schema_compile_sha256')
        and sha(schema/'fia_abi_schema.json')==data.get('arch22_public_schema_sha256'),
        'S2_NORMAL_FRESH_PUBLIC_SCHEMA_DIFFERS')
    compiled=json.loads((schema/'compile_receipt.json').read_bytes())
    require(compiled.get('schema')=='demoncat.fia.fresh_abi_compile.v1'
        and compiled.get('schema_sha256')==data['arch22_public_schema_sha256']
        and compiled.get('npu_executed') is False and compiled.get('loading_allowed') is False,
        'S2_NORMAL_SCHEMA_NOT_QUALIFICATION')
    bundle['canonical_token_sources']=verify_normal_token_sources(completion)
    return bundle


def validate_owned_normal(launch,host,actual,boundary,*,token,owner,generation,phase,rows):
    require(phase in ('baseline','off') and boundary['module_path']==FIRST_LAYER
        and boundary['sequence_lengths']==[1025], 'S2_NORMAL_FIRST_LAYER_FIRST_DECODE_ONLY')
    require(host.get('schema')=='demoncat.paged33.arch22_s2_host.v1'
        and host.get('mode')=='arch22_s2_off' and host.get('phase')==phase
        and host.get('owner')==owner and type(host.get('generation')) is int and host['generation']==generation
        and host.get('token')==token and host.get('target_path')==FIRST_LAYER
        and (host.get('pid'),host.get('tid'))==(os.getpid(),threading.get_native_id())
        and host.get('actual_observer_scope_matched') is True,'S2_NORMAL_ACTUAL_SCOPE_DIFFERS')
    # Reuse exactly the already reviewed normal scalar checks. This temporary
    # projection is internal; the original S2 host schema/mode is preserved.
    projected=dict(host,schema='demoncat.paged33.arch22_normal_host.v1',mode='arch22_normal')
    geometry=validate_native(launch,projected,actual,boundary)
    require(host['aic'] in rows,'S2_NORMAL_UNOBSERVED_CPU_PLATFORM_ROW')
    split=verify_normal_geometry(launch['typed'],host,rows[host['aic']])
    return dict(geometry,complete_cpu_split_join=split,normal_only=True,qualification_granted=False)


class Arch22S2NormalNativeCall(Arch22NormalNativeCall):
    bundle_verifier=staticmethod(verify_bundle)

    def execute(self,actual,boundary,phase,invoke_once,*,binding,owner,generation):
        require(phase in ('baseline','off'),'S2_NORMAL_NEVER_AUTHORIZES_FAULT')
        require(type(binding) is LogicalPagedInput and binding.owner is owner
            and binding.generation==generation and binding.actual is actual
            and binding.module_path==boundary['module_path']==FIRST_LAYER
            and boundary['sequence_lengths']==[1025],'S2_NORMAL_ACTUAL_LIVE_INPUT_REQUIRED')
        scope=validate_normal_call(actual,boundary)
        from .paged33_loading import validate_environment
        validate_environment(self.build['vendor'],platform_ready=True)
        token=uuid.uuid4().hex;log=self.folder/'host.jsonl'
        environment=dict(DCAT33_TND_MODE='arch22_s2_off',DCAT33_TND_TOKEN=token,
            DCAT33_TND_PID=str(os.getpid()),DCAT33_TND_TID=str(threading.get_native_id()),DCAT33_TND_LOG=str(log),
            DCAT33_TND_OWNER=self.owner_token,DCAT33_TND_GENERATION=str(generation),
            DCAT33_TND_PHASE=phase,DCAT33_TND_TARGET=FIRST_LAYER)
        def verify():
            hosts=[json.loads(line) for line in log.read_text().splitlines()]
            hosts=[row for row in hosts if row.get('token')==token]
            require(len(hosts)==1,'S2_NORMAL_EXACT_ONE_HOST_CALLBACK_REQUIRED')
            host=hosts[0]
            launch=verify_launch(self.runtime_dir,token,schema=self.schema,host=host,
                kernel_sha256=self.fingerprint['kernel_sha256'],kernel_metadata=self.metadata,kernel_abi=self.abi)
            geometry=validate_owned_normal(launch,host,actual,boundary,token=token,owner=self.owner_token,
                generation=generation,phase=phase,rows=self.build['s2_cpu_matrix']['normal_first_decode_rows'])
            return dict(call_id=token,token=token,phase=phase,module_path=FIRST_LAYER,
                module_object_id=boundary['module_object_id'],scheduler=boundary['scheduler'],
                virtual_engine=boundary.get('virtual_engine'),native=launch,host=host,scope=scope,
                geometry=geometry,original_calls=1,selection_applied=False,cache_repacked=False,
                api_arguments_changed=False,output_mutated=False,mechanism=self.mechanism,
                target_key=boundary.get('target_key'),stock_private_exact_proven=False,
                owner_mapping=dict(native_owner=self.owner_token,python_owner_object_id=id(owner),generation=generation),
                logical_input_binding=binding.report())
        return self._run(token,environment,
            lambda:binding.execute(invoke_once,owner=owner,generation=generation),verify)


class S2NormalProbeAdapter:
    requires_live_page_binding=True

    def __init__(self,*,arm,experimental_qualification=False,max_calls=28,**kwargs):
        require(experimental_qualification is True and arm in ('baseline','off') and max_calls==28,
                'S2_NORMAL_ONLY_FULL_DENOMINATOR_PROBE_REQUIRED')
        self.native=Arch22S2NormalNativeCall(**kwargs)
        self.arm,self.enabled,self.thread=arm,True,threading.get_ident()
        self.owner,self.generation=object(),1
        self.native.owner_token=uuid.uuid4().hex
        self.calls,self.runtime_dir=self.native.calls,self.native.runtime_dir
        self.binding=None
        self.non_target_calls=[]
        self.first_cleanup_error=None

    def bind_input(self,binding):
        require(threading.get_ident()==self.thread and self.enabled and self.binding is None
                and type(binding) is LogicalPagedInput and binding.owner is self.owner
                and binding.generation==self.generation,'S2_NORMAL_CONTEXT_ALREADY_BOUND_OR_FOREIGN')
        self.binding=binding

    def execute(self,actual,boundary,invoke_once):
        require(threading.get_ident()==self.thread and self.enabled,'S2_NORMAL_DISABLED_OR_FOREIGN')
        if boundary.get('module_path')!=FIRST_LAYER:
            row=dict(module_path=boundary.get('module_path'),module_object_id=boundary.get('module_object_id'),
                     request_ids=boundary.get('scheduler',{}).get('request_ids'),original_calls=1,returned=False)
            self.non_target_calls.append(row)
            result=invoke_once();row['returned']=True
            return result
        require(not self.calls and self.binding is not None
                and set(actual)==set(self.binding.actual)
                and all(actual[k] is self.binding.actual[k] for k in actual),
                'S2_NORMAL_EXACT_ONE_FIRST_LAYER_CALL')
        result,_=self.native.execute(self.binding.actual,boundary,self.arm,invoke_once,
                                    binding=self.binding,owner=self.owner,generation=self.generation)
        return result

    def disable(self):
        require(threading.get_ident()==self.thread,'S2_NORMAL_FOREIGN_CLEANUP')
        self.enabled=False
        try:
            state=self.native.stop()
            require(state['scope_owned'] is False and state['scope_active'] is False
                and state['environment_owned']==0,'S2_NORMAL_STOP_NOT_CONFIRMED')
            if self.binding is not None:self.binding.release(owner=self.owner,generation=self.generation)
        except BaseException as error:
            # An uncertain drain/stop keeps the live input and owner. A later
            # successful retry cannot erase the original cleanup failure.
            if self.first_cleanup_error is None:self.first_cleanup_error=repr(error)
            raise
        return dict(state,scope_cleared=not state['scope_owned'] and state['environment_owned']==0,
            first_error=state.get('first_error') or self.first_cleanup_error,
            local_streams_drained=True,library_coexistence=self.native.library_coexistence,
            public_worker_qualified=False,c1=False,c2=False,c3=False,stock_private_exact_proven=False,
            first_layer_only=True,unscoped_original_calls=list(self.non_target_calls),
            logical_binding_released=self.binding is None or self.binding.closed)
