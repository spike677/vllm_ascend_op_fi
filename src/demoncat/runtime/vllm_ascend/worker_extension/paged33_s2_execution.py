"""One actual first-layer effect within the complete natural Attention census."""
from copy import deepcopy
from pathlib import Path
import threading

from .attention_execution import AttentionExecutionAdapter,complete
from .call_binding import ForwardBinding
from .paged33_boundary_probe import tensor_snapshot
from .paged33_logical_binding import LogicalPagedInput
from .paged33_s2_evidence import FIRST
from .paged33_tnd_probe import logical_cache_file
from .serving_context import require

LAYERS=tuple(f'model.layers.{index}.self_attn.attn' for index in range(28))


class Paged33S2ExecutionAdapter(AttentionExecutionAdapter):
    def __init__(self,deployment,identity_guard):
        super().__init__(deployment,identity_guard)
        require(deployment.mechanism=='33.paged_decode' and len(deployment.leases)==1,
                'S2_PUBLIC_DEPLOYMENT_TICKET_REQUIRED')
        self.ticket=next(iter(deployment.leases));self.driver.bind_owner(self,self.ticket)
        self.logits_binding=ForwardBinding(self.processor)
        self.payload_refs={};self.phase_files={};self.current_binding=None;self.owned_inputs=[]
        sites=[module for path,module in self.candidates if path==FIRST]
        require(len(sites)==1 and self.model.get_submodule(FIRST) is sites[0],
                'S2_NORMAL_SITE_NOT_IN_ACTUAL_STRUCTURE')
        self.first_module=sites[0]
        self.target='FIRST_LAYER_ACTUAL_OBJECT_AND_LOGICAL_INPUT'

    def _selected(self,row):
        return self.current['decode'] and row.get('module_path')==FIRST and row.get('occurrence')==0

    def _selected_phase(self,phase):return True

    def _execute_selected(self,actual,row,phase,once):
        context=self._context();path,module=self.stack[-1]
        require(path==FIRST and module is self.first_module,'S2_FIRST_ACTUAL_MODULE_CHANGED')
        row['impl_object_id']=id(module.impl)
        try:
            binding=LogicalPagedInput(worker=self.worker,module=module,module_path=path,forward_context=context,
                metadata=context.attn_metadata[module.layer_name],actual=actual,scheduler=self.current['scheduler'],
                writer=self.writers[id(module)],owner=self,generation=self.ticket[1],allow_cpu_test=self.deployment.simulation)
            self.current_binding=binding;self.owned_inputs.append(binding)
            return self.driver.execute(actual,row,phase,once,binding=binding,owner=self,generation=self.ticket[1])
        except BaseException:
            if phase=='fault' and not self.driver.effect_attempted:self.needs_off=False
            raise

    def _invoke(self,name,original,args,kwargs):
        output=super()._invoke(name,original,args,kwargs)
        row=self.current['calls'][-1]
        if self.current['decode']:
            require(row['module_path'] in LAYERS and row['metadata_binding']=='VERIFIED'
                and type(output) is tuple and len(output)==2,'S2_COMPLETE_NATURAL_DECODE_BOUNDARY_REQUIRED')
            layer=self.payload_refs.setdefault('attention',{})
            require(row['module_path'] not in layer,'S2_DUPLICATE_NATURAL_DECODE_CALL')
            _,module=self.stack[-1]
            row['normal_proof_scope']=('ACTUAL_FIRST_LAYER_OBJECT_AND_PARENT_CALL' if row['module_path']==FIRST else
                'OTHER_KERNEL_INPUT_MATCH_WITHOUT_PER_SITE_LOGICAL_NORMAL_AND_EFFECT_PROOF' if row['input_contract_matches'] else
                'OBSERVED_INPUT_NOT_ADAPTED_NO_NATIVE_ELIGIBILITY_CONCLUSION')
            layer[row['module_path']]=dict(output=output[0],key=module.impl.key_cache,value=module.impl.value_cache,row=row)
        return output

    def _post(self,module,args,kwargs,output):
        super()._post(module,args,kwargs,output)
        if self.current['decode']:self.payload_refs['hidden']=output

    def _logits(self,module,args,kwargs,output):
        if self.current['decode']:
            actual=self.logits_binding.bind(module,args,kwargs)
            require('hidden_states' in actual,'S2_ACTUAL_LOGITS_INPUT_REQUIRED')
            self.payload_refs.update(logits=output,logits_hidden=actual['hidden_states'])
        super()._logits(module,args,kwargs,output)

    def collect(self):
        row=super().collect();phase=row['phase'];refs=self.payload_refs
        require(set(refs)=={'hidden','logits_hidden','logits','attention'} and set(refs['attention'])==set(LAYERS),
                'S2_COMPLETE_28_LAYER_PAYLOAD_SET_REQUIRED')
        folder=Path(self.driver.folder)/'public_payloads'/phase;folder.mkdir(parents=True,exist_ok=False)
        files={}
        for key in ('hidden','logits_hidden','logits'):
            files[key]=tensor_snapshot(refs[key],folder/(key+'.bin'))
        require(all(files['hidden'][key]==files['logits_hidden'][key] for key in ('bytes','sha256')),
                'S2_ROOT_TO_ACTUAL_LOGITS_INPUT_DIFFERS')
        decode=[item for item in row['forwards'] if item['decode']]
        require(len(decode)==1,'S2_ONE_ACTUAL_DECODE_FORWARD_REQUIRED')
        for key in ('hidden','logits'):
            require(complete(refs[key])==decode[0][key],'S2_RETURNED_MODEL_OUTPUT_CHANGED_BEFORE_READBACK')
        for index,path in enumerate(LAYERS):
            item=refs['attention'][path];boundary=item['row']
            files[path+'/output']=tensor_snapshot(item['output'],folder/f'layer-{index}-output.bin')
            require(complete(item['output'])==boundary['output'][0],'S2_ACTUAL_LAYER_OUTPUT_CHANGED_AFTER_RETURN')
            for name in ('key','value'):
                files[path+'/'+name]=logical_cache_file(item[name],boundary['used_pages'],boundary['sequence_lengths'][0],
                    folder/f'layer-{index}-{name}.bin')
        binding=self.current_binding
        require(binding is self.driver.live_inputs[phase] and binding.complete,'S2_CURRENT_LIVE_INPUT_MISSING')
        query=tensor_snapshot(binding.actual['query'],folder/'query.bin')
        require(query['sha256']==binding.before['tensors']['query']['sha256'],'S2_QUERY_CHANGED_AFTER_REAL_CONSUMER')
        immediate=[tensor_snapshot(value,folder/f'immediate-{index}.bin') for index,value in enumerate(binding.result)]
        require(all(value['sha256']==before['sha256'] for value,before in zip(immediate,binding._sealed['outputs']))
            and len(immediate)==2,'S2_IMMEDIATE_OUTPUT_CHANGED_AFTER_REAL_CONSUMER')
        for name in ('key','value'):
            require(files[FIRST+'/'+name]['sha256']==binding.before['logical_kv'][name]['sha256'],
                    'S2_FIRST_LOGICAL_CACHE_CHANGED_AFTER_REAL_CONSUMER')
        require(len(files)==87,'S2_ALL_87_PAYLOADS_REQUIRED')
        row['payloads']=files;row['first_query']=query;row['immediate_outputs']=immediate
        row['live_input']=binding.report()
        self.phase_files[phase]=files;self.payload_refs={};self.current_binding=None
        return row

    def prepare_off_recovery(self):
        self._owned()
        require(threading.get_ident()==self.thread and not self.root_active and not self.callbacks
            and self.needs_off and self.driver.effect_attempted,'S2_NO_IDLE_OWNED_FAULT_TO_RECOVER')
        state=self.driver.stop()
        require(not state['scope_owned'] and not state['scope_active'] and state['environment_owned']==0,
                'S2_UNCERTAIN_NATIVE_SCOPE_BEFORE_OFF')
        self.driver.prepare_recovery_off(self)
        if self.expected is not None or self.current is not None:
            partial={key:value for key,value in (self.current or {}).items() if key!='entry_error'}
            self.failed_requests.append(dict(expected=self.expected,partial=deepcopy(partial),
                returned_forwards=deepcopy(self.request_rows),complete=False,first_error=self.last_error))
        self.expected=self.current=None;self.stack.clear();self.parents.clear();self.writers.clear()
        self.payload_refs={};self.current_binding=None
        # Only the active request error is retired. failures/failed_requests
        # retain the first error and permanently prevent qualification.
        self.last_error=None

    def close(self):
        require(not self.needs_off,'S2_ACTUAL_OFF_MODEL_RECOVERY_REQUIRED')
        if self.driver.owner is None:
            state=self.driver.stop()
            require(not state['scope_owned'] and not state['scope_active'] and not state['environment_owned']
                and not state.get('pending_references') and not state.get('retained_input_objects'),
                'S2_PARTIAL_START_NOT_ZERO_RESOURCE')
        else:self.driver.release_inputs(self)
        for item in getattr(self,'owned_inputs',()):
            if not item.closed:item.release(owner=self,generation=self.ticket[1])
        super().close()
        self.payload_refs={};self.current_binding=None

    def state(self):
        value=super().state()
        value['owned_input_references']=sum(not item.closed for item in getattr(self,'owned_inputs',()))
        return value
