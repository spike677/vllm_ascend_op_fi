"""All discovered decode objects, normal/OFF only; no C1/C2/C3 authority."""
from copy import deepcopy
import hashlib
import inspect
import json
import os
from pathlib import Path
import threading
import uuid

from .attention_execution import AttentionExecutionAdapter
from .call_binding import ForwardBinding
from .paged33_arch22_s2_normal import verify_bundle as _verify_bundle
from .paged33_arch22_bridge import Arch22NormalNativeCall,validate_normal_call,validate_native
from .paged33_boundary_probe import (FirstDecodeBoundaryCapture,StockLaunchScope,stock_scope_from_receipt,
    read_scoped_launches,mapped_api_libraries,MappedLibraryObservationError)
from .paged33_logical_binding import LogicalPagedInput
from .paged33_normal_targets import NormalTargetScope
from .paged33_tnd_probe import tensor_file,logical_cache_file
from .serving_attention import AttentionDeployment
from .serving_context import require
from fault_injection.native_adapters.attention28.launch import verify_launch
from fault_injection.native_adapters.attention28.fia.arch22_s2_build_evidence import verify_normal_geometry

VARIANT='tnd_page16_arch22_all_normal'


def verify_bundle(completion,completion_sha256,vendor):
    return _verify_bundle(completion,completion_sha256,vendor,all_normal=True)


def validate_host(host,boundary,scope,*,owner,generation,token,phase):
    ordinal=scope.check(scope.worker,boundary['module_path'],scope.model.get_submodule(boundary['module_path']),
                        scope.owner,generation)
    require(phase in ('baseline','off') and host.get('schema')=='demoncat.paged33.arch22_all_normal_host.v1'
        and host.get('mode')=='arch22_all_normal' and host.get('phase')==phase
        and host.get('owner')==owner and type(host.get('generation')) is int and host.get('generation')==generation and host.get('token')==token
        and host.get('target_path')==boundary['module_path'] and host.get('module_object_id')==boundary['module_object_id']
        and host.get('target_set_sha256')==scope.fingerprint and type(host.get('target_ordinal')) is int and host.get('target_ordinal')==ordinal
        and (host.get('pid'),host.get('tid'))==(os.getpid(),threading.get_native_id())
        and host.get('normal_only') is True and host.get('policy_applied') is False
        and host.get('native_s2')==host.get('selected_s2')==512
        and host.get('actual_observer_scope_matched') is True,'ALL_NORMAL_ACTUAL_HOST_SCOPE_DIFFERS')


class AllNormalNativeCall(Arch22NormalNativeCall):
    bundle_verifier=staticmethod(verify_bundle)

    def execute(self,actual,boundary,phase,invoke_once,*,binding,scope,owner,generation):
        require(phase in ('baseline','off') and type(binding) is LogicalPagedInput
            and type(scope) is NormalTargetScope and binding.target_scope is scope
            and binding.owner is owner and binding.generation==generation and binding.actual is actual,
            'ALL_NORMAL_LIVE_NORMAL_BINDING_REQUIRED')
        ordinal=scope.check(binding.worker,binding.module_path,binding.module,owner,generation)
        require(boundary['module_path']==binding.module_path and boundary['module_object_id']==id(binding.module)
            and boundary['sequence_lengths']==[1025],
                'ALL_NORMAL_ACTUAL_DECODE_GEOMETRY_REQUIRED')
        geometry=validate_normal_call(actual,boundary)
        from .paged33_loading import validate_environment
        validate_environment(self.build['vendor'],platform_ready=True)
        token=uuid.uuid4().hex;log=self.folder/'host.jsonl'
        env=dict(DCAT33_TND_MODE='arch22_all_normal',DCAT33_TND_TOKEN=token,DCAT33_TND_LOG=str(log),
            DCAT33_TND_PID=str(os.getpid()),DCAT33_TND_TID=str(threading.get_native_id()),
            DCAT33_TND_OWNER=self.owner_token,DCAT33_TND_GENERATION=str(generation),DCAT33_TND_PHASE=phase,
            DCAT33_TND_TARGET=binding.module_path,DCAT33_TND_TARGET_SET=scope.fingerprint,
            DCAT33_TND_OBJECT=str(id(binding.module)),DCAT33_TND_ORDINAL=str(ordinal))
        def verify():
            hosts=[json.loads(line) for line in log.read_text().splitlines()]
            hosts=[row for row in hosts if row.get('token')==token]
            require(len(hosts)==1,'ALL_NORMAL_ONE_HOST_CALLBACK_REQUIRED');host=hosts[0]
            validate_host(host,boundary,scope,owner=self.owner_token,generation=generation,token=token,phase=phase)
            launch=verify_launch(self.runtime_dir,token,schema=self.schema,host=host,
                kernel_sha256=self.fingerprint['kernel_sha256'],kernel_metadata=self.metadata,kernel_abi=self.abi)
            projected=dict(host,schema='demoncat.paged33.arch22_normal_host.v1',mode='arch22_normal')
            typed=validate_native(launch,projected,actual,boundary)
            split=verify_normal_geometry(launch['typed'],host,self.build['s2_cpu_matrix']['normal_first_decode_rows'][host['aic']])
            return dict(call_id=token,token=token,phase=phase,module_path=binding.module_path,
                module_object_id=id(binding.module),scheduler=boundary['scheduler'],target_ordinal=ordinal,
                virtual_engine=boundary['virtual_engine'],
                target_set_sha256=scope.fingerprint,owner=self.owner_token,generation=generation,
                native=launch,host=host,scope=geometry,geometry=dict(typed,complete_cpu_split_join=split),
                original_calls=1,selection_applied=False,cache_repacked=False,api_arguments_changed=False,
                output_mutated=False,mechanism=self.mechanism,normal_only=True,logical_input_binding=binding.report())
        return self._run(token,env,lambda:binding.execute(invoke_once,owner=owner,generation=generation),verify)


class StockNormalScopes:
    mechanism='33.paged_decode'
    def __init__(self,receipt,digest):
        self.prototype=stock_scope_from_receipt(receipt,digest)
        self.runtime_dir=self.prototype.runtime_dir;self.scopes=[];self.calls=[]
    def invoke(self,invoke_once):
        scope=StockLaunchScope(self.prototype.observer,self.prototype.synchronize,self.runtime_dir)
        self.scopes.append(scope)  # Own before begin; failed end remains retryable.
        return scope.invoke(invoke_once),scope.token
    def stop(self):
        for scope in self.scopes:scope.close()
        return dict(scope_owned=False,scope_active=False,environment_owned=0,native_loaded=False,
                    calls=0,failures=[error for scope in self.scopes for error in scope.errors])


class AllNormalWorkerSession(AttentionExecutionAdapter):
    def __init__(self,worker,*,native_args,evidence_dir,statistics=False,boundary_args):
        require(not statistics and boundary_args.get('arch22_all_normal') is True
            and (native_args is None or (native_args.get('arch22_all_normal') is True
                and native_args.get('experimental_qualification') is True
                and type(native_args.get('max_calls')) is int and native_args.get('max_calls')==28
                and not native_args.get('normal_bridge') and not native_args.get('arch22_s2_normal'))),
            'ALL_NORMAL_SAME_ZERO_FAULT_CAPTURE_REQUIRED')
        require(hashlib.sha256(Path(inspect.getfile(FirstDecodeBoundaryCapture)).read_bytes()).hexdigest()
            ==boundary_args['capture_source_sha256'],'ALL_NORMAL_INSTALLED_CAPTURE_SOURCE_CHANGED')
        require(type(boundary_args['prompt_token_ids']) is list and len(boundary_args['prompt_token_ids'])==1024
            and all(type(token) is int and token>=0 for token in boundary_args['prompt_token_ids']),
            'ALL_NORMAL_ACTUAL_1024_TOKEN_PROMPT_REQUIRED')
        self.folder=Path(evidence_dir)
        require(str(self.folder).startswith('/evidence/') and not self.folder.exists(),'ALL_NORMAL_FRESH_EVIDENCE_REQUIRED')
        self.folder.mkdir(parents=True);(self.folder/'boundaries').mkdir()
        self.private=native_args is not None
        native=dict(native_args or {})
        self.native_phase=native.pop('arm','off')
        require(self.native_phase in ('baseline','off'),'ALL_NORMAL_NEVER_AUTHORIZES_FAULT')
        for key in ('arch22_all_normal','experimental_qualification','max_calls'):native.pop(key,None)
        self.driver=(AllNormalNativeCall(**native,evidence_dir=self.folder/'native') if self.private else
                     StockNormalScopes(boundary_args['path'],boundary_args['expected_sha256']))
        self.owner_token=uuid.uuid4().hex;self.driver.owner_token=self.owner_token
        deployment=AttentionDeployment(worker,worker.model_runner,worker.model_runner.model,'33.paged_decode',tuple(boundary_args['prompt_token_ids']),
            self.driver,'{}',os.getpid(),threading.get_ident(),28,leases={(self.owner_token,1)})
        super().__init__(deployment,self._guard)
        self.targets=NormalTargetScope(worker,self.candidates,owner=self,generation=1)
        self.logits_binding=ForwardBinding(self.processor)
        self.records={};self.payload_refs={};self.original=None;self.first_cleanup_error=None
        self.source=dict(self.source)

    def _guard(self):
        require(self.deployment.leases=={(self.owner_token,1)} and self.deployment.pid==os.getpid()
            and self.deployment.thread==threading.get_ident(),'ALL_NORMAL_DEPLOYMENT_OWNER_CHANGED')
        if hasattr(self,'targets'):self.targets.report()

    def expect(self,expected):
        require(type(expected) is dict and len(expected)==1,'ALL_NORMAL_ONE_ENQUEUED_REQUEST_REQUIRED')
        request,details=next(iter(expected.items()))
        # This frozen deployment is local to this trusted probe, before any call.
        require(not self.rows and self.expected is None,'ALL_NORMAL_ONE_REQUEST_PER_OWNED_SESSION')
        require(tuple(details['prompt'])==self.deployment.prompt,'ALL_NORMAL_ACTUAL_PROMPT_CHANGED')
        return super().expect(request,self.folder.name)

    def _selected(self,row):return self.current['decode']
    def _selected_phase(self,phase):return True

    def _execute_selected(self,actual,row,phase,once):
        path,module=self.stack[-1];ordinal=self.targets.check(self.worker,path,module,self,1)
        require(path not in self.records and row['occurrence']==0 and ordinal==len(self.records),
                'ALL_NORMAL_COMPLETE_UNIQUE_NATURAL_TARGET_ORDER_REQUIRED')
        row['impl_object_id']=id(module.impl);row['consumer_returned']=False
        context=self._context();record={};self.records[path]=record
        binding=LogicalPagedInput(worker=self.worker,module=module,module_path=path,forward_context=context,
            metadata=context.attn_metadata[module.layer_name],actual=actual,scheduler=self.current['scheduler'],
            writer=self.writers[id(module)],owner=self,generation=1,target_scope=self.targets,
            allow_cpu_test=self.deployment.simulation)
        record['binding']=binding
        import torch
        capture=FirstDecodeBoundaryCapture(self.worker,self.folder/'boundaries'/str(ordinal),
            (lambda:None) if self.deployment.simulation else torch.npu.synchronize,
            allow_cpu_test=self.deployment.simulation,target_scope=self.targets,target_path=path)
        record['capture']=capture
        name,original,args,kwargs=self.original
        capture.start(name,original,args,kwargs,row,module,ordinal)
        def invoke():return capture.invoke_original(original,args,kwargs,call_once=once)
        if self.private:
            result,receipt=self.driver.execute(actual,row,self.native_phase,invoke,binding=binding,scope=self.targets,
                                               owner=self,generation=1)
            token=receipt['call_id']
        else:
            result,token=self.driver.invoke(lambda:binding.execute(invoke,owner=self,generation=1));receipt=None
        capture.data['logical_input_binding']=binding.report();capture.data['native_call_id']=token
        for key,observe in (('runtime_observation',lambda:read_scoped_launches(self.driver.runtime_dir,token,
                expected_pid=capture.pid,expected_native_tid=capture.native_tid)),('mapped_api_libraries',mapped_api_libraries)):
            try:capture.data[key]=observe()
            except Exception as error:
                capture.data[key]=dict(status='UNOBSERVED_OR_INCOMPLETE',error=repr(error))
                if isinstance(error,MappedLibraryObservationError):capture.data[key]['diagnostic']=error.evidence
        capture.write()
        return result,receipt

    def _invoke(self,name,original,args,kwargs):
        require(self.original is None,'ALL_NORMAL_API_REENTRY')
        self.original=(name,original,args,kwargs)
        try:
            result=super()._invoke(name,original,args,kwargs)
            if self.current['decode']:
                path,module=self.stack[-1]
                require(path not in self.payload_refs.setdefault('attention',{}),'ALL_NORMAL_DUPLICATE_PAYLOAD')
                self.payload_refs['attention'][path]=dict(output=result[0],key=module.impl.key_cache,
                    value=module.impl.value_cache,row=self.current['calls'][-1])
            return result
        finally:self.original=None

    def _module_post(self,module,args,kwargs,output):
        if self.current and self.current['decode']:
            rows=[row for row in self.current['calls'] if row['module_object_id']==id(module)]
            require(len(rows)==1 and rows[0]['original_returned'] and output is not None,
                    'ALL_NORMAL_ACTUAL_ATTENTION_CONSUMER_RETURN_REQUIRED')
            rows[0]['consumer_returned']=True
        super()._module_post(module,args,kwargs,output)

    def _post(self,module,args,kwargs,output):
        decode=bool(self.current and self.current['decode']);super()._post(module,args,kwargs,output)
        if decode:self.payload_refs['hidden']=output
    def _logits(self,module,args,kwargs,output):
        if self.current and self.current['decode']:
            actual=self.logits_binding.bind(module,args,kwargs)
            self.payload_refs.update(logits_hidden=actual['hidden_states'],logits=output)
        super()._logits(module,args,kwargs,output)

    def collect(self):
        row=super().collect();paths=[p for p,_,_ in self.targets.rows]
        require([f['decode'] for f in row['forwards']]==[False,True]
            and all([c['module_path'] for c in f['calls']]==paths and all(c['original_calls']==1
                    and c['original_returned'] and c['inputs_unchanged'] for c in f['calls']) for f in row['forwards'])
            and all(c['consumer_returned'] is True for c in row['forwards'][1]['calls'])
            and list(self.records)==paths,'ALL_NORMAL_COMPLETE_56_ACTUAL_CALL_DENOMINATOR_REQUIRED')
        boundaries={path:self.records[path]['capture'].finalize() for path in paths}
        refs=self.payload_refs;folder=self.folder/'private';folder.mkdir()
        report=dict(schema='demoncat.paged33.natural_probe.v1',all_decode_boundaries=boundaries,
            target_scope=self.targets.report(),natural_request=row,
            complete_tensor_readback_during_statistics=False,
            audit=dict(source=self.source,candidate_count=28,prefill_forwards=1,decode_forwards=1,core_methods_replaced=False,
                rows=deepcopy(row['forwards'][1]['calls']),errors=self.failures),
            native_calls=deepcopy(self.driver.calls),attention={},c2=False,c3=False,public_worker_qualified=False)
        for name in ('hidden','logits_hidden','logits'):report[name]=tensor_file(refs[name],folder/(name.replace('_','-')+'.bin'))
        require(all(all(report[name][key]==row['forwards'][1][name][key] for key in ('shape','dtype','sha256','finite'))
                    for name in ('hidden','logits')),'ALL_NORMAL_ACTUAL_MODEL_OUTPUT_CHANGED_AFTER_RETURN')
        require(all(report['hidden'][k]==report['logits_hidden'][k] for k in ('shape','dtype','bytes','sha256')),
                'ALL_NORMAL_ACTUAL_LOGITS_INPUT_DIFFERS')
        for ordinal,path in enumerate(sorted(paths)):
            values=refs['attention'][path];boundary=values['row']
            report['attention'][path]=dict(output=tensor_file(values['output'],folder/f'layer-{ordinal}-output.bin'))
            for name in ('key','value'):
                report['attention'][path][name]=logical_cache_file(values[name],boundary['used_pages'],1025,
                                                                  folder/f'layer-{ordinal}-{name}.bin')
        report['all_represented_tensors_finite']=all(r['finite'] for r in [report['hidden'],report['logits_hidden'],
            report['logits'],*(v for layer in report['attention'].values() for v in layer.values())])
        require(report['all_represented_tensors_finite'],'ALL_NORMAL_NONFINITE_COMPLETE_PAYLOAD')
        (self.folder/'runtime_result.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
        return report

    def report(self):return dict(source=self.source,failures=self.failures,partial_requests=self.failed_requests)

    def close(self):
        try:
            super().close()
            state=self.driver.stop()
            for record in self.records.values():
                if 'capture' in record:record['capture'].release()
                if 'binding' in record and not record['binding'].closed:record['binding'].release(owner=self,generation=1)
        except BaseException as error:
            if self.first_cleanup_error is None:self.first_cleanup_error=repr(error)
            raise
        self.payload_refs.clear()
        return dict(state,scope_cleared=True,local_streams_drained=True,owned_hooks=0,owned_api_wrappers=0,
            boundary_refs_released=all(r['capture'].refs is None for r in self.records.values() if 'capture' in r),
            core_methods_replaced=False,
            stock_launch_scope=(None if self.private else dict(scope_cleared=True,local_streams_drained=True,errors=state['failures'])),
            logical_bindings_released=all(r['binding'].closed for r in self.records.values() if 'binding' in r),
            first_error=self.first_cleanup_error,c1=False,c2=False,c3=False,public_worker_qualified=False,
            library_coexistence=getattr(self.driver,'library_coexistence',None),normal_only=True)
