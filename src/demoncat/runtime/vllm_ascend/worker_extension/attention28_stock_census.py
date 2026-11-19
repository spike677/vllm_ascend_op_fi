"""Observe every actual source71-shaped TND prefill under a stock launch scope.

Trusted deployment only. No policy, private vendor, assumed stock POD or F.
Unadapted calls remain visible and execute their original arguments once.
"""
from copy import deepcopy
import json
import math
import os
from pathlib import Path
import threading
import uuid

from .attention_native import OwnedLaunchScope
from .attention_stock_launch import observe_scope, sha
from .paged33_boundary_probe import tensor_snapshot, scalar_snapshot
from .serving_context import require
from fault_injection.native_adapters.attention28.launch import LaunchObserver

MECHANISMS = ('28.fia.legal_tiling', '28.fia.denominator_replay')
SUPPLIED = {'query','key','value','atten_mask','block_table','input_layout','block_size',
            'num_heads','num_key_value_heads','scale','sparse_mode','actual_seq_lengths','actual_seq_lengths_kv'}
POLICIES = ('NES_FA_', 'FAULT28_', 'DCAT33_', 'DCAT_F33_')


def thread_identity():
    return dict(pid=os.getpid(),python_thread_ident=threading.get_ident(),native_tid=threading.get_native_id(),
                source='CURRENT_THREAD_GET_IDENT_AND_GET_NATIVE_ID')


def capture_inputs(original, args, actual, metadata):
    import torch
    schema = getattr(getattr(original, 'default', None), '_schema', None)
    values = {name:(dict(kind='tensor',snapshot=tensor_snapshot(value)) if isinstance(value,torch.Tensor)
        else dict(kind='scalar',value=scalar_snapshot(value))) for name,value in actual.items()}
    return dict(positional_count=len(args),supplied_argument_names=sorted(actual),inputs=values,
        original_callable=dict(type=type(original).__module__+'.'+type(original).__qualname__,object_id=id(original)),
        public_default_schema=str(schema) if schema is not None else None,
        omitted_argument_defaults={arg.name:scalar_snapshot(arg.default_value) for arg in schema.arguments
            if arg.name not in actual and arg.has_default_value()} if schema is not None else None,
        default_schema_dispatch_observed=False,actual_native_attributes_observed=False,
        live_metadata=dict(object_id=id(metadata),num_prefills=metadata.num_prefills,num_decodes=metadata.num_decodes,
            num_actual_tokens=metadata.num_actual_tokens,causal=metadata.causal,
            seq_lens_list=scalar_snapshot(metadata.seq_lens_list),
            actual_seq_lengths_q=scalar_snapshot(metadata.actual_seq_lengths_q),
            mask_is_actual_metadata_object=actual.get('atten_mask') is metadata.attn_mask))


def validate_tnd_prefill(captured, row, *, simulation=False):
    """Freeze the actually observed API semantics; do not guess its Native key."""
    require(row['actual_operator'] == 'torch_npu.npu_fused_infer_attention_score'
            and set(captured['inputs']) == SUPPLIED, 'TND_CENSUS_SOURCE71_ACTUAL_ARGUMENT_SET_DIFFERS')
    values=captured['inputs']; metadata=captured['live_metadata']; scheduler=row['scheduler']
    require(captured['original_callable']['type'] == 'torch._ops.OpOverloadPacket' or simulation,
            'TND_CENSUS_ACTUAL_PUBLIC_PACKET_REQUIRED')
    require(captured['public_default_schema'] and captured['omitted_argument_defaults'] is not None,
            'TND_CENSUS_PUBLIC_DEFAULT_SCHEMA_UNOBSERVED')
    defaults=captured['omitted_argument_defaults']
    # Source71 did not capture these defaults. Record the actual packet's
    # declarations now instead of assuming inner_precise=0 to select a route.
    require(not set(defaults).intersection(values)
        and all(type(defaults.get(k)) is int for k in ('pre_tokens','next_tokens','inner_precise'))
        and type(defaults.get('softmax_lse_flag')) is bool, 'TND_CENSUS_OMITTED_DEFAULTS_UNOBSERVED')
    expected=dict(input_layout='TND',block_size=128,block_table=None,num_heads=16,num_key_value_heads=8,
        sparse_mode=3,actual_seq_lengths=[2048],actual_seq_lengths_kv=[2048])
    require(all(values[k] == dict(kind='scalar',value=v) for k,v in expected.items())
        and values['scale']['kind'] == 'scalar' and type(values['scale']['value']) in (int,float)
        and math.isclose(values['scale']['value'],128**-.5,rel_tol=0,abs_tol=1e-12),
        'TND_CENSUS_SOURCE71_SEMANTICS_DIFFER')
    require(scheduler.get('status') == 'READ_ONLY_SCHEDULER_SNAPSHOT_CORROBORATED_NOT_NPU_ACCEPTANCE'
        and scheduler.get('internal_request_id') == row['request_id'] and scheduler.get('num_requests') == 1
        and scheduler.get('num_computed_tokens') == 0
        and scheduler.get('num_prompt_tokens') == scheduler.get('num_scheduled_tokens') == 2048,
        'TND_CENSUS_ACTUAL_FRESH_REQUEST_UNBOUND')
    require(metadata['num_prefills'] == 1 and metadata['num_decodes'] == 0 and metadata['num_actual_tokens'] == 2048
        and metadata['causal'] is True and metadata['seq_lens_list'] == metadata['actual_seq_lengths_q'] == [2048]
        and metadata['mask_is_actual_metadata_object'] is True, 'TND_CENSUS_LIVE_METADATA_DIFFERS')
    devices=set()
    for name,shape,dtype in (('query',[2048,16,128],'torch.float16'),('key',[2048,8,128],'torch.float16'),
            ('value',[2048,8,128],'torch.float16'),('atten_mask',[2048,2048],'torch.int8')):
        item=values[name]; snap=item['snapshot']; meta=snap['metadata']; devices.add(meta['device'])
        require(item['kind'] == 'tensor' and meta['shape'] == shape and meta['dtype'] == dtype
            and len(meta['stride']) == len(shape) and all(type(n) is int and n > 0 for n in meta['stride'])
            and type(meta['storage_offset']) is int and meta['storage_offset'] >= 0
            and type(meta['storage_identity']) is int and meta['storage_identity'] > 0
            and snap['finite'] is True, 'TND_CENSUS_ACTUAL_TENSOR_GEOMETRY_DIFFERS:'+name)
        width=1 if name == 'atten_mask' else 2
        require(snap['bytes'] == math.prod(shape)*width
            and (meta['storage_offset']+1+sum((n-1)*s for n,s in zip(shape,meta['stride'])))*width <= meta['storage_nbytes'],
            'TND_CENSUS_ACTUAL_STORAGE_EXTENT_DIFFERS:'+name)
    require(len(devices) == 1 and (next(iter(devices)).split(':')[0] == 'npu'
        or simulation and next(iter(devices)) == 'cpu'), 'TND_CENSUS_ACTUAL_DEVICE_DIFFERS')
    return dict(layout='TND',query_tokens=2048,kv_tokens=2048,api_block_size=128,paged=False,
                scheduler_cache_block_size=scheduler.get('block_size'),native_key='UNOBSERVED_UNTIL_RAW_SCOPE_JOIN',
                native_argument_tensor_pointer_join='UNOBSERVED_WITHOUT_MATCHING_STOCK_ABI')


class StockTndPrefillCensus:
    simulation=False
    observer_only=True

    def __init__(self, *, mechanism, observer_receipt, observer_receipt_sha256, evidence_dir, input_journal=None,
                 prepared_input_capture=None):
        require(mechanism in MECHANISMS, 'TND_STOCK_CENSUS_FIA_MECHANISM_REQUIRED')
        require(sha(observer_receipt) == observer_receipt_sha256, 'TND_CENSUS_OBSERVER_RECEIPT_CHANGED')
        receipt=json.loads(Path(observer_receipt).read_bytes()); library=receipt['observer_library']
        require(receipt.get('observer_compiled') is True and receipt.get('actual_npu_calls') == 0
            and receipt.get('observer_memory_selftest',{}).get('exit_code') == 0
            and receipt.get('headers') and all(sha(row['path']) == row['sha256'] for row in receipt['headers']),
            'TND_CENSUS_OBSERVER_OR_ACTUAL_SDK_CHANGED')
        self.mechanism,self.identity=mechanism,thread_identity()
        self.environment={name:os.environ.get(name) for name in
            ('LD_PRELOAD','DCAT_ATTN_OBSERVER_DIR','ASCEND_CUSTOM_OPP_PATH','ACLNN_CACHE_LIMIT','TASK_QUEUE_ENABLE')}
        require(self.environment['ACLNN_CACHE_LIMIT'] == self.environment['TASK_QUEUE_ENABLE'] == '0',
                'TND_CENSUS_SYNCHRONOUS_UNCACHED_WORKER_REQUIRED')
        self.run=Path(self.environment['DCAT_ATTN_OBSERVER_DIR']).resolve(strict=True).parent
        require(self.run.is_relative_to('/evidence') and Path(self.environment['DCAT_ATTN_OBSERVER_DIR']).name == 'runtime',
                'TND_CENSUS_OWNED_RUNTIME_PATH_REQUIRED')
        self.folder=Path(evidence_dir).resolve()
        require(self.folder.is_relative_to(self.run) and not self.folder.exists(), 'TND_CENSUS_FRESH_CALL_ARTIFACT_PATH_REQUIRED')
        self.scope=OwnedLaunchScope(LaunchObserver(library['path'],library['sha256']))
        self.fingerprint=dict(observer_receipt_sha256=observer_receipt_sha256,observer_library_sha256=library['sha256'],
            observer_source_sha256=receipt['observer_source']['sha256'],observer_only=True,private_native_build=None)
        self.calls,self.failures,self.refs=[],[],None
        self.pending_drain=False;self.first_error=None
        self.input_journal=input_journal;self._journal_current=None;self._journal_requests={}
        self.prepared_input_capture=prepared_input_capture
        if prepared_input_capture is not None:
            from .attention28_prepared_input import PreparedInputCapture
            require(type(prepared_input_capture) is PreparedInputCapture and input_journal is None,
                    'TND_INPUT_EXACT_SINGLE_CAPTURE_NO_EXTERNAL_JOURNAL')
        if input_journal is not None:self._validate_journal()
        self._check();self.folder.mkdir(parents=True)

    def _validate_journal(self):
        from fault_injection.fia_input_journal import NativeFiaInputJournal, process_identity
        journal=self.input_journal
        require(type(journal) is NativeFiaInputJournal, 'TND_CENSUS_EXACT_INPUT_JOURNAL_REQUIRED')
        require(journal.startup is not None and journal.startup['identity'] == process_identity()
            and journal.started and journal.active is None and journal.bound_worker is not None
            and journal.bound_worker['pid'] == self.identity['pid'], 'TND_CENSUS_PRESTART_JOURNAL_WORKER_REQUIRED')
        require(journal.profile['originals']['observer']['file_sha256'] == self.fingerprint['observer_library_sha256'],
            'TND_CENSUS_JOURNAL_OBSERVER_DIFFERS')

    def _journal_enter(self, token, row, actual, record):
        journal=getattr(self,'input_journal',None)
        if journal is None:return
        from fault_injection.fia_input_journal import public_tensor_metadata
        self._validate_journal()
        scope=row['native_census_scope'];phase=scope['phase']
        require(scope['worker_object_id'] == journal.bound_worker['worker_object_id']
            and scope['model_object_id'] == journal.bound_worker['model_object_id']
            and phase in ('observe','observe_repeat') and (scope['device_type'] == 'npu' or self.simulation),
            'TND_CENSUS_ACTUAL_JOURNAL_OBJECT_SCOPE_CHANGED')
        prior=self._journal_requests.setdefault(phase,row['request_id'])
        require(prior == row['request_id'] and len(set(self._journal_requests.values())) == len(self._journal_requests),
            'TND_CENSUS_JOURNAL_REQUEST_PHASE_CHANGED')
        evidence=dict(actual_input_addresses={name:public_tensor_metadata(actual[name])
                for name in ('query','key','value','atten_mask')},
            request_id=row['request_id'],actual_operator=row['actual_operator'],module_path=row['module_path'],
            ordinal=row['ordinal'],occurrence=row['occurrence'],execution_scope=deepcopy(scope),
            scheduler=deepcopy(row['scheduler']),native_census_call_id=token,
            correlation_indices='FORWARD_IS_OBSERVE_PHASE_INDEX; OPERATOR_IS_ACTUAL_ORDINAL_PLUS_ONE; BOUNDARY_IS_CENSUS_ORDER')
        record['input_journal']=dict(enabled=True,call_id=token,public_inputs=deepcopy(evidence['actual_input_addresses']),
            window_entered=False,window_left=False,internal_prepared_tensor_lineage_proven=False,slice_pointer_lineage_proven=False)
        self._journal_current=dict(row=row,record=record,original_error=None,entered=False)
        journal.enter(call_id=token,module_object_id=row['module_object_id'],
            forward=1 if phase == 'observe' else 2,sequence=row['ordinal']+1,boundary=len(self.calls)+1,evidence=evidence)
        self._journal_current['entered']=True
        record['input_journal'].update(window_entered=True,token=journal.active['identity']['scope']['token'])

    def _journal_leave(self):
        current=getattr(self,'_journal_current',None)
        if current is None:return
        journal=self.input_journal
        if journal.active is not None:
            journal.leave(original_calls=current['row']['original_calls'],original_error=current['original_error'])
            current['record']['input_journal'].update(window_left=True,original_calls=current['row']['original_calls'],
                original_error=current['original_error'])
        else:
            require(not current['entered'], 'TND_CENSUS_JOURNAL_WINDOW_LOST_WITHOUT_LEAVE')
        self._journal_current=None

    def _check(self):
        require(thread_identity() == self.identity, 'TND_CENSUS_PID_OR_THREAD_DOMAIN_CHANGED')
        require(all(os.environ.get(k) == v for k,v in self.environment.items()), 'TND_CENSUS_RUNTIME_ENVIRONMENT_CHANGED')
        require(not any(name.startswith(POLICIES) for name in os.environ), 'TND_CENSUS_FOREIGN_FAULT_POLICY_PRESENT')
        return self.scope.state()

    def state(self):
        scope=self._check()
        return dict(native_loaded=True,observer_only=True,scope_active=scope['active'],scope_owned=scope['owned'],
            environment_owned=0,pending_drains=int(self.pending_drain),pending_references=int(self.refs is not None),
            calls=len(self.calls),failures=deepcopy(self.failures),first_error=self.first_error,
            build=dict(self.fingerprint),native_launch_qualified=False,fault_execution_ready=False,simulation=self.simulation)

    def _error(self, action, error):
        if self.first_error is None:self.first_error=repr(error)
        self.failures.append(dict(action=action,error=repr(error),first_error=self.first_error))

    def _synchronize(self):
        import torch
        torch.npu.synchronize()

    def stop(self):
        self._check()
        try:
            if self.pending_drain:
                self._synchronize()
                if self.scope.state()['active']:self.scope.observer.synchronize()
                self.pending_drain=False
            self._journal_leave()
            if getattr(self,'prepared_input_capture',None) is not None:self.prepared_input_capture.stop()
            self.scope.stop()
            require(not self.scope.state()['owned'] and not self.pending_drain, 'TND_CENSUS_CLEANUP_UNCONFIRMED')
            self.refs=None
        except BaseException as error:
            self._error('stop',error);raise
        return self.state()

    def execute(self, *args):
        raise RuntimeError('TND_STOCK_CENSUS_HAS_NO_BASELINE_OR_FAULT_EXECUTION')

    def observe(self, original, args, actual, row, metadata, once):
        self._check()
        require(not self.refs and not self.pending_drain and not self.scope.state()['owned'], 'TND_CENSUS_PREVIOUS_CALL_STILL_OWNED')
        record=dict(schema='demoncat.attention28.stock_native_census.v1',status='UNOBSERVED_OR_INCOMPLETE',
            request_id=row['request_id'],module_path=row['module_path'],module_object_id=row['module_object_id'],
            execution_scope=deepcopy(row['native_census_scope']),
            actual_operator=row['actual_operator'],occurrence=row['occurrence'],ordinal=row['ordinal'],
            target_key=row['target_key'],thread_identity=dict(self.identity),fault_authorized=False,native_fault_qualified=False,
            input_substitutions=0,operator_arguments_modified=False,reasons=[])
        row['native_census']=record  # Keep incomplete capture visible on any exception.
        capture=getattr(self,'prepared_input_capture',None)
        if capture is not None and capture.used:
            record.update(status='UNOBSERVED_OUTSIDE_SINGLE_PREPARED_INPUT_WINDOW',
                reasons=['SINGLE_ACTUAL_TARGET_DIAGNOSTIC_ONLY_FULL_CALL_INVENTORY_RETAINED'])
            output=once();self.calls.append(record);return output,record
        try:
            import torch
            require(self.simulation or type(original) is torch._ops.OpOverloadPacket,
                    'TND_CENSUS_ACTUAL_PACKET_OBJECT_REQUIRED')
            record['api']=capture_inputs(original,args,actual,metadata)
            record['api_contract']=validate_tnd_prefill(record['api'],row,simulation=self.simulation)
            mask=actual['atten_mask'].detach().cpu()
            require(torch.equal(mask,torch.ones_like(mask).triu(1)), 'TND_CENSUS_ACTUAL_CAUSAL_MASK_VALUES_DIFFER')
            record['actual_causal_mask_values_verified']=True
        except Exception as error:
            record['reasons'].append(repr(error));self.calls.append(record)
            return once(),record
        token=uuid.uuid4().hex;record['call_id']=token
        self.refs=(actual,None);self.pending_drain=True
        primary=None
        try:
            self._synchronize()
            if capture is not None and getattr(capture,'borrow_query',None) is not None:
                capture.prepare_plan(row)  # All value reads precede the original Native window.
            with self.scope.scope(token):
                try:
                    if capture is not None:record['prepared_input_capture']=capture.enter(token,row,actual,record['api'])
                    self._journal_enter(token,row,actual,record)
                    try:
                        output=once();self.refs=(actual,output)
                        if capture is not None:capture.returned(output)
                    except BaseException as original_error:
                        current=getattr(self,'_journal_current',None)
                        if current is not None:current['original_error']=repr(original_error)
                        raise
                    self._synchronize();self.scope.observer.synchronize()
                    self._journal_leave();self.pending_drain=False
                except BaseException as error:
                    primary=error;self._error('original_or_completion',error);raise
        except BaseException as error:
            record['reasons'].append(repr(error));self._error('scope',error);self.calls.append(record)
            if primary is not None:raise primary.with_traceback(primary.__traceback__)
            raise
        self.stop()
        # Metadata and value readbacks occur outside the raw API scope, so
        # observation's own transfers cannot masquerade as the stock kernel.
        try:
            after=capture_inputs(original,args,actual,metadata)
            record['api_inputs_after']=after
            require(record['api'] == after, 'TND_CENSUS_ACTUAL_INPUT_CHANGED')
            record['launches']=observe_scope(self.run,token,self.identity)
            record['status']='OBSERVED_INTERCEPTED_STOCK_LAUNCHES'
        except Exception as error:record['reasons'].append(repr(error))
        record['original_calls']=row['original_calls'];record['original_returned']=True
        self.calls.append(record)
        artifact=self.folder/(token+'.json')
        with artifact.open('x',encoding='utf-8') as stream:json.dump(record,stream,allow_nan=False,indent=2)
        record['artifact']=dict(path=str(artifact),sha256=sha(artifact),bytes=artifact.stat().st_size)
        return output,record
