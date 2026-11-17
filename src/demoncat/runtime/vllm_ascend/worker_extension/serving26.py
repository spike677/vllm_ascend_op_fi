"""Bounded Fault26 FIA serving replay, preserving the real inference path.

Four fresh prefill requests: reference, lossless writer, comparison, OFF.
Only owned module hooks are installed. No core Worker/runner method changes.
Native callbacks remain process-local and are removed after each request.
"""
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import threading

from .serving_context import (require, require_serial_eager, snapshot_fresh_prefill,
                              discover_logits_boundary)
from .call_binding import ForwardBinding
from .serving import complete_tensor

MECHANISM='26.fia_destination'
PHASES=('reference','lossless','comparison','off')


@dataclass(frozen=True)
class FiaServingBinding:
    make_session: object


class FiaServingSession:
    mechanism=MECHANISM
    runtime_call_budget=None
    def __init__(self,worker,model,*,library,sha256,folder,prompt,baseline_control,collector):
        from vllm.model_executor.layers.logits_processor import LogitsProcessor
        from fault_injection.deployments.residual26 import Residual26Runtime
        self.worker,self.runner,self.model=worker,worker.model_runner,model
        require_serial_eager(self.runner)
        require(model is self.runner.model and not model.training,'LOADED_EVAL_MODEL_REQUIRED')
        self.folder=Path(folder);self.folder.mkdir(parents=True,exist_ok=False)
        self.library,self.sha256=library,sha256
        self.probe=Residual26Runtime(model,self.mechanism,library=library,sha256=sha256,
            evidence_dir=self.folder/'state-probe',log_path=self.folder/'writer.jsonl')
        boundary=discover_logits_boundary(model,actual_logits_processor_type=LogitsProcessor,
            actual_vocab_size=self.runner.vllm_config.model_config.hf_config.vocab_size)
        self.processor=boundary['processor'];self.boundary=boundary['report']
        self.boundary_guard=boundary.get('branch_guard')
        self.binding=ForwardBinding(model);self.prompt=list(prompt)
        self.baseline_control=baseline_control;self.collector=collector
        self.thread=threading.get_ident();self.handles=[];self.rows=[];self.seen=[]
        self.expected=self.current=self.error=None
        self.native_session=self.native_runtime=None
        self.collecting=False;self.aborted=False;self.installed=False;self.recovered=False

    def state(self):
        require(threading.get_ident()==self.thread,'FOREIGN_SERVING_THREAD')
        require(self.worker.model_runner is self.runner and self.runner.model is self.model,'MODEL_LOAD_CHANGED')
        return self.probe.native_state()

    def install(self):
        state=self.state()
        require(not state['registered'] and not self.handles,'NATIVE_OWNER_ALREADY_PRESENT')
        self.handles.append(self.model.register_forward_pre_hook(self.pre,prepend=True,with_kwargs=True))
        self.handles.append(self.model.register_forward_hook(self.post,always_call=True,with_kwargs=True))
        self.handles.append(self.processor.register_forward_hook(self.logits,always_call=True,with_kwargs=True))
        self.installed=True

    def expect(self,request_id):
        self.collect()
        require(self.installed and not self.aborted and not self.error,'ACTIVE_SERVING_REPLAY_REQUIRED')
        require(self.expected is None and self.current is None and len(self.rows)<len(PHASES),'PREVIOUS_REQUEST_NOT_CLOSED')
        require(isinstance(request_id,str) and 0<len(request_id)<=256 and request_id not in self.seen,
            'FRESH_ENQUEUED_ID_REQUIRED')
        self.expected=request_id
        phase=PHASES[len(self.rows)]
        # msprobe may register custom API wrappers at start. Install the owned
        # producer observer afterwards so its removal checks the actual owner.
        if self.collector:
            self.collecting=True
            self.collector.start(model=self.model,rank_id=self.worker.rank)
        if phase in ('lossless','comparison'):
            from fault_injection import attach
            from fault_injection.deployments.residual26 import Residual26Runtime
            arm='fault' if phase=='comparison' and not self.baseline_control else 'baseline'
            self.native_runtime=Residual26Runtime(self.model,self.mechanism,library=self.library,sha256=self.sha256,
                evidence_dir=self.folder/phase/'native',log_path=self.folder/'writer.jsonl',
                max_effect_calls=self.runtime_call_budget)
            self.native_session=attach(self.model,self.mechanism,arm=arm,runtime=self.native_runtime,unsupported='passthrough')
            self.native_session.__enter__()

    def pre(self,module,args,kwargs):
        from vllm.forward_context import get_forward_context
        try:
            self.state()
            require(self.expected is not None and self.current is None and not self.error,
                'NO_ADVANCE_OR_CONCURRENT_FORWARD')
            actual=self.binding.bind(module,args,kwargs)
            snapshot=snapshot_fresh_prefill(self.runner,self.model,actual,get_forward_context(),
                expected_prompt_token_ids=self.prompt,expected_internal_request_id=self.expected,
                seen_internal_request_ids=self.seen)
            cpu,identity,_=complete_tensor(actual['input_ids'])
            require(cpu.tolist()==self.prompt,'DEVICE_INPUT_BYTES_DIFFER')
            if self.boundary_guard:self.boundary_guard()
            self.current=dict(phase=PHASES[len(self.rows)],scheduler=snapshot,input_sha256=identity,
                request_id=self.expected,root_returned=False,logits_returned=False)
        except BaseException as error:
            self.error=repr(error);self.save();raise

    def post(self,module,args,kwargs,output):
        if self.current is None:return
        try:
            _,digest,finite=complete_tensor(output)
            require(finite,'MODEL_HIDDEN_NONFINITE')
            self.current.update(root_returned=True,hidden_sha256=digest,hidden_finite=finite)
        except BaseException as error:
            self.error=repr(error);self.save();raise

    def logits(self,module,args,kwargs,output):
        try:
            require(self.current is not None and self.current['root_returned'] and not self.current['logits_returned'],
                'LOGITS_OUTSIDE_COMPLETED_ROOT')
            cpu,digest,finite=complete_tensor(output)
            require(finite and list(cpu.shape)==[1,self.runner.vllm_config.model_config.hf_config.vocab_size],
                'COMPLETE_FINITE_LOGITS_REQUIRED')
            self.current.update(logits_returned=True,logits_sha256=digest,logits_finite=finite,
                top1=cpu.argmax(dim=-1).tolist())
        except BaseException as error:
            self.error=repr(error);self.save();raise

    def collect(self):
        if self.current is None:return
        require(not self.error and self.current['logits_returned'],'REQUEST_NOT_COMPLETED')
        if self.collecting:
            self.collector.stop();self.collector.step();self.collecting=False
        if self.native_session:
            self.native_session.close()
            runtime=self.native_runtime
            require(runtime.supported>0 and runtime.claims==len(runtime.completed) and not runtime.pending()
                and not runtime.error,'NATURAL_WRITER_CONSUMER_RECOVERY_UNPROVEN')
            self.current['native']=dict(supported=runtime.supported,claims=runtime.claims,
                completed=runtime.completed,coverage=self.native_session.report(),pending=runtime.pending(),
                selected_target=runtime.selected_target)
            self.native_session=self.native_runtime=None
        row=self.current
        if self.rows and row['phase'] in ('lossless','off'):
            for key in ('hidden_sha256','logits_sha256','input_sha256'):
                require(row[key]==self.rows[0][key],row['phase'].upper()+'_COMPLETE_OUTPUT_NOT_RESTORED:'+key)
        self.rows.append(row);self.seen.append(self.expected)
        self.expected=self.current=None
        self.recovered=row['phase']=='off'
        self.save()

    def close(self,*,abort=False):
        if not abort:self.collect()
        if self.collecting:
            self.collector.stop();self.collecting=False
        if self.native_session:
            self.native_session.close();self.native_session=self.native_runtime=None
        state=self.state()
        require(not any(state[k] for k in ('registered','prepared','queued_executors','pending_consumers',
            'active_consumer','callback_active','environment_enabled')),'NATIVE_RECOVERY_PENDING')
        for handle in reversed(self.handles):handle.remove()
        self.handles.clear();self.installed=False
        if abort:self.aborted=True;self.expected=self.current=None
        self.save()

    def report(self):
        return dict(schema='demoncat.fia26.serving.v1',phase=PHASES[len(self.rows)] if len(self.rows)<4 else 'complete',
            rows=self.rows,error=self.error,recovered=self.recovered,aborted=self.aborted,
            baseline_control=self.baseline_control,core_methods_replaced=False,boundary=self.boundary,
            scope='ONE_FRESH_PREFILL_PER_PHASE_NATURAL_MATCHES_NOT_PUBLIC_ALL_ELIGIBLE')

    def save(self):
        (self.folder/'serving.json').write_text(json.dumps(self.report(),indent=2)+'\n')


@dataclass(frozen=True)
class ResidualServingBinding:
    make_session: object


class ResidualServingSession(FiaServingSession):
    """One actually adjacent QMM writer on the unmodified serving path."""
    mechanism='26.residual_stomp'
    runtime_call_budget=1

    def collect(self):
        if self.current is None:return
        # Close/drain and independently compare complete original/off tensors
        # first. Failed qualification retains the concrete recorded row.
        previous=len(self.rows)
        try:
            super().collect()
            if len(self.rows)==previous:return
            row=self.rows[-1]
            if row['phase'] in ('lossless','comparison'):
                target=row['native'].get('selected_target')
                require(isinstance(target,dict) and target.get('operator')=='aclnnQuantMatmulWeightNz'
                    and target.get('allocation') is not None
                    and row['native']['supported']==1,'ONE_LIVE_ADJACENT_QMM_TARGET_REQUIRED')
                if row['phase']=='comparison':
                    lossless=next(r for r in self.rows if r['phase']=='lossless')['native']['selected_target']
                    keys=('module_path','module_object_id','operator','cache','output_bytes','residual_bytes','geometry')
                    require(all(target[k]==lossless[k] for k in keys),'QMM_TARGET_CHANGED_AFTER_LOSSLESS')
                    require(row['native']['claims']==(0 if self.baseline_control else 1),
                            'QMM_ACTUAL_CLAIM_MISMATCH')
            self.save()
        except BaseException as error:
            self.error=repr(error);self.recovered=False;self.save();raise

    def report(self):
        result=super().report()
        result.update(schema='demoncat.residual26.serving.v1',mechanism=self.mechanism,
            scope='ONE_ACTUAL_ADJACENT_QMM_WRITER_PER_FRESH_PREFILL_PHASE',
            public_all_eligible=False,public_every_match=False)
        return result


def _install_writer26_binding(worker,*,mechanism,prompt,library,sha256,evidence_dir,baseline_control,collect_statistics):
    from fault_injection.providers import register_runtime
    require(mechanism in (MECHANISM,'26.residual_stomp'),'EXPLICIT_WRITER_MECHANISM_REQUIRED')
    fia=mechanism==MECHANISM
    binding_attr='_dcat_fia26_binding' if fia else '_dcat_qmm26_binding'
    require(not hasattr(worker,binding_attr),'DEPLOYMENT_ALREADY_BOUND')
    require(type(baseline_control) is bool and type(collect_statistics) is bool,'EXPLICIT_CONTROL_REQUIRED')
    require(isinstance(prompt,list) and len(prompt)>1 and all(type(x) is int and x>=0 for x in prompt),'FROZEN_PROMPT_REQUIRED')
    require(os.environ.get('VLLM_ENABLE_V1_MULTIPROCESSING')=='0','HOST_DRIVEN_ENGINE_REQUIRED')
    require_serial_eager(worker.model_runner)
    for obj in (worker,worker.model_runner):
        require(not any(n in vars(obj) for n in ('execute_model','_prepare_inputs')),'CORE_OVERRIDE_REFUSED')
    folder=Path(evidence_dir)
    require(folder.is_absolute() and folder.resolve().is_relative_to(Path('/evidence')),'PRIVATE_EVIDENCE_REQUIRED')
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
            for p in package.rglob('*') if p.is_file() and p.suffix in ('.py','.json','.yaml')}
        (cap/'collector_identity.json').write_text(json.dumps(dict(package=str(package),files=files,
            edited_by_experiment=False,steps=dict(enumerate(PHASES))),indent=2)+'\n')
        collector=PrecisionDebugger(config_path=str(cap/'config.json'))
    model=worker.model_runner.model
    def make_session(w,m):
        require(w is worker and m is model and worker.model_runner.model is model,'FOREIGN_MODEL')
        os.environ['FAULT26_FIA_WRITER_LOG_PATH' if fia else 'FAULT26_QMM_UNDERFLOW_LOG_PATH']=str(folder/'runtime'/'writer.jsonl')
        cls=FiaServingSession if fia else ResidualServingSession
        return cls(worker,model,library=library,sha256=sha256,folder=folder/'runtime',
            prompt=prompt,baseline_control=baseline_control,collector=collector)
    def factory(m):
        require(m is model,'FOREIGN_MODEL')
        return (FiaServingBinding if fia else ResidualServingBinding)(make_session)
    register_runtime(mechanism,factory)
    setattr(worker,binding_attr,factory)
    return dict(status='DEPLOYMENT_BOUND_NOT_ACCEPTED',core_methods_replaced=False,workers_created=0)


def install_fia26_binding(worker,*,prompt,library,sha256,evidence_dir,baseline_control,collect_statistics):
    return _install_writer26_binding(worker,mechanism=MECHANISM,prompt=prompt,library=library,
        sha256=sha256,evidence_dir=evidence_dir,baseline_control=baseline_control,collect_statistics=collect_statistics)


def install_residual26_binding(worker,*,prompt,library,sha256,evidence_dir,baseline_control,collect_statistics):
    return _install_writer26_binding(worker,mechanism='26.residual_stomp',prompt=prompt,library=library,
        sha256=sha256,evidence_dir=evidence_dir,baseline_control=baseline_control,collect_statistics=collect_statistics)


def remove_fia26_binding(worker):
    return _remove_writer26_binding(worker,MECHANISM,'_dcat_fia26_binding')


def remove_residual26_binding(worker):
    return _remove_writer26_binding(worker,'26.residual_stomp','_dcat_qmm26_binding')


def _remove_writer26_binding(worker,mechanism,attribute):
    from fault_injection.providers import unregister_runtime
    runtime=getattr(worker,'_demoncat_worker_fault_runtime',None)
    require(runtime is not None and not runtime._leases,'COMMON_OWNER_MUST_UNREGISTER_FIRST')
    factory=getattr(worker,attribute,None)
    require(factory is not None,'DEPLOYMENT_NOT_BOUND')
    unregister_runtime(mechanism,factory);delattr(worker,attribute)
    return dict(status='DEPLOYMENT_REMOVED')
