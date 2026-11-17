"""Bounded actual B8 decode, selected through embedding/norm/projection lineage.

Only model/module hooks and owned quant method/API observers; no Worker core
replacement, fabricated scheduler IDs or expert-row-as-request inference.
"""
from dataclasses import dataclass
import functools
import hashlib
import json
import os
from pathlib import Path
import threading

from .serving_context import require,require_serial_eager,discover_logits_boundary
from .batch_context import snapshot_first_decode
from .call_binding import ForwardBinding
from .serving import complete_tensor

MECHANISM='quant.batch_shared_scale'
PHASES=('reference','lossless','comparison','off')


@dataclass(frozen=True)
class QuantServingBinding:
    make_session: object


def same_tensor(a,b):
    if a is None or b is None:return False
    return (a.untyped_storage()._cdata==b.untyped_storage()._cdata and a.storage_offset()==b.storage_offset()
        and a.shape==b.shape and a.stride()==b.stride() and a.dtype==b.dtype and a.device==b.device)


class QuantServingSession:
    def __init__(self,worker,model,*,root,build_sha256,folder,prompts,baseline_control,collector,capture_scope='generation'):
        from fault_injection.quantization import PinnedQuantRuntime,_discover
        from vllm.model_executor.layers.logits_processor import LogitsProcessor
        self.worker,self.runner,self.model=worker,worker.model_runner,model
        require_serial_eager(self.runner,expected_max_seqs=8)
        require(not model.training and self.runner.model is model,'ACTUAL_EVAL_MODEL_REQUIRED')
        self.thread=threading.get_ident();self.folder=Path(folder);self.folder.mkdir(parents=True,exist_ok=False)
        self.prompts=prompts;self.baseline_control=baseline_control;self.collector=collector
        require(capture_scope in ('generation','first_decode'),'UNREVIEWED_CAPTURE_SCOPE')
        self.capture_scope=capture_scope;self.capture_started=False;self.quant_api=None
        self.runtime=PinnedQuantRuntime.from_portable_v2_backend(root,context_provider=self.context,
            approved_build_sha256=build_sha256)
        self.candidates,self.rejected=_discover(model,(1,))
        require(bool(self.candidates),'NO_ACTUAL_DYNAMIC_QUANT_MODULES')
        boundary=discover_logits_boundary(model,actual_logits_processor_type=LogitsProcessor,
            actual_vocab_size=self.runner.vllm_config.model_config.hf_config.vocab_size)
        self.processor=boundary['processor'];self.boundary=boundary['report'];self.guard=boundary.get('branch_guard')
        self.binding=ForwardBinding(model);self.handles=[];self.patches=[];self.rows=[];self.seen=[]
        self.expected=self.active=self.error=None;self.embedding=self.normalized=None
        self.selected=self.healthy=None;self.quant_call=None;self.quant_session=None
        self.installed=self.collecting=self.recovered=self.aborted=False;self.observations=[]

    def state(self):
        require(threading.get_ident()==self.thread and self.worker.model_runner is self.runner
            and self.runner.model is self.model,'THREAD_OR_MODEL_CHANGED')
        return self.runtime.native_state()

    def install(self):
        from vllm.model_executor.layers.vocab_parallel_embedding import VocabParallelEmbedding,ParallelLMHead
        from vllm.model_executor.layers.layernorm import RMSNorm
        require(not self.installed and not self.state()['owned'],'NATIVE_OWNER_PRESENT')
        self.handles.append(self.model.register_forward_pre_hook(self.pre,with_kwargs=True))
        self.handles.append(self.model.register_forward_hook(self.post,always_call=True,with_kwargs=True))
        self.handles.append(self.processor.register_forward_hook(self.logits,always_call=True,with_kwargs=True))
        for _,m in self.model.named_modules():
            if isinstance(m,VocabParallelEmbedding) and not isinstance(m,ParallelLMHead):
                self.handles.append(m.register_forward_hook(self.embed,with_kwargs=True))
            if isinstance(m,RMSNorm):self.handles.append(m.register_forward_hook(self.norm,with_kwargs=True))
        byid={id(m):(p,m,method) for p,m,method,_ in self.candidates}
        for method in {id(x[2]):x[2] for x in self.candidates}.values():
            original=method.apply;local='apply' in vars(method);saved=vars(method).get('apply')
            @functools.wraps(original)
            def observe(module,*args,_original=original,**kwargs):
                x=args[0] if args else kwargs.get('x');active=self.active
                match=bool(active and active.get('decode') and id(module) in byid and same_tensor(x,self.normalized))
                if match and self.selected is None:
                    require(active['phase']=='reference','HEALTHY_LINEAGE_REQUIRED')
                    self.runtime.backend.validate_input(x);self.selected=module
                    self.healthy=dict(path=byid[id(module)][0],input_sha256=self.runtime.input_sha256(x),
                        rows=self.row_signature(active['decode']))
                selected=match and module is self.selected
                require(self.quant_call is None,'NESTED_QUANT_OBSERVER')
                if selected:self.quant_call=x
                try:return _original(module,*args,**kwargs)
                finally:
                    if selected:self.quant_call=None
            method.apply=observe;self.patches.append((method,original,observe,local,saved))
        self.installed=True

    @staticmethod
    def row_signature(rows):
        return {key:rows[key] for key in ('logical_ids','row_token_positions','prefixes','input_ids')}

    def context(self,call):
        from fault_injection.core import UnsupportedCall
        if not self.active or not self.active.get('decode') or call.module is not self.selected:
            raise UnsupportedCall('Outside selected real first decode projection')
        if not same_tensor(call.input,self.normalized):raise UnsupportedCall('Embedding/norm/projection row lineage missing')
        rows=self.active['decode']
        require(self.row_signature(rows)==self.healthy['rows'],'HEALTHY_DECODE_PREFIX_OR_ROW_ORDER_CHANGED')
        return dict(rows,model=self.model,rank=0,world_size=1,healthy_run_id='actual_reference_first_decode',
            baseline_input_sha256=self.healthy['input_sha256'])

    def expect(self,request_ids):
        self.collect()
        require(self.installed and not self.aborted and not self.error and self.expected is None,'ACTIVE_IDLE_REPLAY_REQUIRED')
        require(isinstance(request_ids,list) and len(request_ids)==8 and len(set(request_ids))==8
            and all(isinstance(x,str) and x and x not in self.seen for x in request_ids),'EIGHT_FRESH_ENQUEUED_IDS_REQUIRED')
        require(len(self.rows)<4,'REPLAY_ALREADY_COMPLETE')
        self.expected={key:dict(logical_id=i,prompt=self.prompts[i]) for i,key in enumerate(request_ids)}
        self.observations=[];self.decode_result=None;self.prefill_result=None;self.forward_count=0
        self.capture_started=False
        if self.capture_scope=='generation':self.begin_observation()

    def begin_decode_capture(self):
        require(self.capture_scope=='first_decode' and self.installed and self.expected is not None
            and not self.capture_started and self.active is None and self.prefill_result is not None
            and self.decode_result is None and self.forward_count==1 and not self.error,'COMPLETED_PREFILL_BEFORE_CAPTURE_REQUIRED')
        require(set(self.runner.input_batch.req_ids)==set(self.expected),'PENDING_DECODE_REQUESTS_CHANGED')
        for key in self.expected:
            request=self.runner.requests[key]
            require(request.prompt_token_ids==self.expected[key]['prompt'] and len(request.output_token_ids)==1,
                'REAL_PREFILL_TOKEN_REQUIRED_BEFORE_CAPTURE')
        self.begin_observation()

    def begin_observation(self):
        require(not self.capture_started and self.quant_api is None,'CAPTURE_ALREADY_ACTIVE')
        self.capture_started=True
        if self.collector:self.collector.start(model=self.model,rank_id=self.worker.rank);self.collecting=True
        import torch_npu
        original=torch_npu.npu_dynamic_quant
        @functools.wraps(original)
        def quant_observe(x,*args,**kwargs):
            output=original(x,*args,**kwargs)
            if self.quant_call is not None:
                require(same_tensor(x,self.quant_call),'ACTUAL_DYNAMIC_QUANT_INPUT_CHANGED')
                q,scale=output;cpu_q,qhash,qfinite=complete_tensor(q);cpu_s,shash,sfinite=complete_tensor(scale)
                require(qfinite and sfinite and list(q.shape)==list(x.shape) and list(scale.shape)==[8],'ACTUAL_Q_SCALE_GEOMETRY')
                self.observations.append(dict(phase=PHASES[len(self.rows)],q_sha256=qhash,scale_sha256=shash,
                    scales=cpu_s.tolist(),q_rows_sha256=[hashlib.sha256(row.numpy().tobytes()).hexdigest() for row in cpu_q],
                    input_sha256=self.runtime.input_sha256(x),rows=self.active['decode']))
                # Private exact arrays are auxiliary, never statistics comparator inputs.
                import torch
                torch.save(dict(q=cpu_q,scale=cpu_s),self.folder/(PHASES[len(self.rows)]+'-quant.pt'))
            return output
        self.quant_api=(original,quant_observe);torch_npu.npu_dynamic_quant=quant_observe
        if PHASES[len(self.rows)] in ('lossless','comparison'):
            from fault_injection.quantization import QuantizationSession
            arm='fault' if PHASES[len(self.rows)]=='comparison' and not self.baseline_control else 'baseline'
            self.quant_session=QuantizationSession(self.model,arm=arm,unsupported='passthrough',runtime=self.runtime)
            self.quant_session.__enter__()

    def pre(self,module,args,kwargs):
        from vllm.forward_context import get_forward_context
        try:
            self.state();require(self.expected is not None and self.active is None and not self.error,'UNOWNED_OR_REENTRANT_FORWARD')
            actual=self.binding.bind(module,args,kwargs);context=get_forward_context()
            self.active=dict(phase=PHASES[len(self.rows)],decode=None,returned=False,logits_returned=False)
            self.embedding=self.normalized=None;self.actual=actual;self.forward_count+=1
            if self.runner.with_prefill is False:
                require(self.capture_started,'DECODE_WITHOUT_HOST_CAPTURE_ADVANCE')
                rows=snapshot_first_decode(self.runner,self.model,actual,context,self.expected)
                ids,_,_=complete_tensor(actual['input_ids']);positions,_,_=complete_tensor(actual['positions'])
                li,_,_=complete_tensor(self.runner.logits_indices)
                require(ids.tolist()==rows['input_ids'] and positions.tolist()==rows['row_token_positions']
                    and li.tolist()==list(range(8)),'DEVICE_ROW_MAPPING_MISMATCH')
                self.active['decode']=rows
            if self.guard:self.guard()
        except BaseException as exc:self.error=repr(exc);self.save();raise

    def embed(self,module,args,kwargs,output):
        if self.active and self.active.get('decode'):
            x=args[0] if args else kwargs.get('input_')
            if same_tensor(x,self.actual['input_ids']):
                require(self.embedding is None,'MULTIPLE_EMBEDDING_LINEAGES');self.embedding=output

    def norm(self,module,args,kwargs,output):
        if self.active and self.active.get('decode') and self.embedding is not None:
            x=args[0] if args else kwargs.get('x')
            if same_tensor(x,self.embedding) and self.normalized is None:
                self.normalized=output[0] if isinstance(output,tuple) else output
                require(self.normalized.ndim==2 and self.normalized.shape==self.embedding.shape,'NORM_ROW_GEOMETRY_CHANGED')

    def post(self,module,args,kwargs,output):
        if self.active is None:return
        try:
            _,digest,finite=complete_tensor(output);require(finite,'HIDDEN_NONFINITE')
            self.active.update(returned=True,hidden_sha256=digest)
        except BaseException as exc:self.error=repr(exc);self.save();raise

    def logits(self,module,args,kwargs,output):
        if self.active is None:return
        try:
            require(self.active['returned'] and not self.active['logits_returned'],'UNBOUND_LOGITS')
            cpu,digest,finite=complete_tensor(output)
            require(finite and list(cpu.shape)==[8,self.runner.vllm_config.model_config.hf_config.vocab_size],'FULL_B8_LOGITS_REQUIRED')
            self.active.update(logits_returned=True,logits_sha256=digest,top1=cpu.argmax(-1).tolist())
            if self.active['decode']:
                require(self.decode_result is None,'MULTIPLE_SELECTED_DECODES');self.decode_result=self.active
            else:
                require(self.prefill_result is None,'CHUNKED_OR_MULTIPLE_PREFILL');self.prefill_result=self.active
            self.active=None;self.embedding=self.normalized=None
        except BaseException as exc:self.error=repr(exc);self.save();raise

    def collect(self):
        if self.expected is None or self.decode_result is None:return
        require(not self.error and self.active is None and self.prefill_result and self.forward_count==2,'INCOMPLETE_GENERATION')
        if self.collecting:self.collector.stop();self.collector.step();self.collecting=False
        report=None
        if self.quant_session:
            self.quant_session.close();report=self.quant_session.report();self.quant_session=None
            routed=[x for x in report['events'] if x['status']=='routed']
            require(len(routed)==1 and routed[0]['native_execution_proven'],'ONE_ACTUAL_QUANT_CALL_REQUIRED')
        import torch_npu
        original,observer=self.quant_api
        require(torch_npu.npu_dynamic_quant is observer,'QUANT_API_OBSERVER_OWNER_CHANGED')
        torch_npu.npu_dynamic_quant=original;self.quant_api=None
        require(len(self.observations)==1 and self.healthy is not None,'ONE_Q_SCALE_OBSERVATION_REQUIRED')
        row=dict(phase=PHASES[len(self.rows)],prefill=self.prefill_result,decode=self.decode_result,
            quant=self.observations[0],native=report)
        if self.rows:
            ref=self.rows[0]
            require(row['prefill']['logits_sha256']==ref['prefill']['logits_sha256'],'PREFILL_CHANGED_BEFORE_ARM')
            if row['phase'] in ('lossless','off'):
                for key in ('hidden_sha256','logits_sha256'):
                    require(row['decode'][key]==ref['decode'][key],'COMPLETE_DECODE_NOT_RESTORED:'+key)
                require(row['quant']['q_sha256']==ref['quant']['q_sha256'] and
                    row['quant']['scale_sha256']==ref['quant']['scale_sha256'],'Q_SCALE_NOT_RESTORED')
        self.rows.append(row);self.seen.extend(self.expected);self.expected=None
        self.recovered=row['phase']=='off';self.save()

    def close(self,*,abort=False):
        if not abort:self.collect()
        if self.collecting:self.collector.stop();self.collecting=False
        if self.quant_session:self.quant_session.close();self.quant_session=None
        if getattr(self,'quant_api',None):
            import torch_npu
            original,observer=self.quant_api
            require(torch_npu.npu_dynamic_quant is observer,'QUANT_API_OWNER_CHANGED')
            torch_npu.npu_dynamic_quant=original;self.quant_api=None
        state=self.state();require(not any(state[k] for k in ('armed','pending','executing','owned')),'NATIVE_RECOVERY_PENDING')
        for method,original,observer,local,saved in reversed(self.patches):
            require(method.apply is observer,'QUANT_METHOD_OWNER_CHANGED')
            if local:method.apply=saved
            else:del method.apply
        self.patches.clear()
        for handle in reversed(self.handles):handle.remove()
        self.handles.clear();self.installed=False
        if abort:self.aborted=True;self.active=self.expected=None
        self.save()

    def report(self):
        return dict(schema='demoncat.batch_quant.serving.v1',rows=self.rows,healthy=self.healthy,
            error=self.error,recovered=self.recovered,aborted=self.aborted,baseline_control=self.baseline_control,
            capture_scope=self.capture_scope,native_build=self.runtime.build_identity,
            core_methods_replaced=False,scope='ONE_EMBED_NORM_PROJECTION_FIRST_B8_DECODE_NOT_ALL_ELIGIBLE',
            candidate_count=len(self.candidates),rejected=self.rejected)

    def save(self):(self.folder/'serving.json').write_text(json.dumps(self.report(),indent=2)+'\n')


def install_quant_binding(worker,*,prompts,root,build_sha256,evidence_dir,baseline_control,collect_statistics,capture_scope='generation'):
    from fault_injection.providers import register_runtime
    require(not hasattr(worker,'_dcat_quant_binding'),'DEPLOYMENT_ALREADY_BOUND')
    require(type(baseline_control) is bool and type(collect_statistics) is bool,'EXPLICIT_CONTROL_REQUIRED')
    require(capture_scope in ('generation','first_decode'),'UNREVIEWED_CAPTURE_SCOPE')
    require(isinstance(prompts,list) and len(prompts)==8 and all(isinstance(p,list) and len(p)>1
        and all(type(x) is int and x>=0 for x in p) for p in prompts),'EIGHT_FROZEN_PROMPTS_REQUIRED')
    require(os.environ.get('VLLM_ENABLE_V1_MULTIPROCESSING')=='0','HOST_DRIVEN_ENGINE_REQUIRED')
    require_serial_eager(worker.model_runner,expected_max_seqs=8)
    for obj in (worker,worker.model_runner):
        require(not any(n in vars(obj) for n in ('execute_model','_prepare_inputs')),'CORE_OVERRIDE_REFUSED')
    folder=Path(evidence_dir);require(folder.is_absolute() and folder.resolve().is_relative_to('/evidence'),'PRIVATE_EVIDENCE_REQUIRED')
    collector=None
    if collect_statistics:
        import msprobe
        from msprobe.pytorch import PrecisionDebugger
        cap=folder/'capture';cap.mkdir(parents=True,exist_ok=False)
        config=dict(task='statistics',level='mix',precision='high',rank=[],step=[0,1,2,3],
            dump_path=str(cap/'observation'),async_dump=False,
            statistics=dict(scope=[],list=[],data_mode=['all'],summary_mode='statistics',tensor_list=[]))
        (cap/'config.json').write_text(json.dumps(config,indent=2)+'\n')
        package=Path(msprobe.__file__).resolve().parent
        files={p.relative_to(package).as_posix():hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(package.rglob('*')) if p.is_file() and p.suffix in ('.py','.json','.yaml')}
        (cap/'collector_identity.json').write_text(json.dumps(dict(files=files,modified=False),sort_keys=True,indent=2)+'\n')
        collector=PrecisionDebugger(config_path=str(cap/'config.json'))
    model=worker.model_runner.model
    def make_session(w,m):
        require(w is worker and m is model and worker.model_runner.model is model,'FOREIGN_MODEL')
        return QuantServingSession(w,m,root=root,build_sha256=build_sha256,
            folder=folder/'runtime',prompts=prompts,baseline_control=baseline_control,collector=collector,capture_scope=capture_scope)
    def factory(m):
        require(m is model,'FOREIGN_MODEL')
        return QuantServingBinding(make_session)
    register_runtime(MECHANISM,factory);worker._dcat_quant_binding=factory
    return dict(status='DEPLOYMENT_BOUND_NOT_ACCEPTED',core_methods_replaced=False)


def remove_quant_binding(worker):
    from fault_injection.providers import unregister_runtime
    runtime=getattr(worker,'_demoncat_worker_fault_runtime',None)
    require(runtime is not None and not runtime._leases,'COMMON_OWNER_MUST_UNREGISTER_FIRST')
    binding=getattr(worker,'_dcat_quant_binding',None);require(binding is not None,'DEPLOYMENT_NOT_BOUND')
    unregister_runtime(MECHANISM,binding);del worker._dcat_quant_binding
    return dict(status='DEPLOYMENT_REMOVED')
