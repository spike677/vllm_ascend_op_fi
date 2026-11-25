"""Read-only probe ownership, identity and row provenance. No admission changes."""
import hashlib
import inspect
import json
import os
from pathlib import Path


class ProbeRefused(ValueError):
    pass


def same_callable(left,right):
    if inspect.ismethod(left) or inspect.ismethod(right):
        return inspect.ismethod(left) and inspect.ismethod(right) and left.__func__ is right.__func__ and left.__self__ is right.__self__
    return left is right


def source(value):
    fn = getattr(value, '__func__', value)
    try:
        file = inspect.getsourcefile(fn)
        raw = Path(file).read_bytes() if file else None
    except (TypeError, OSError):
        file, raw = None, None
    try: line=inspect.getsourcelines(fn)[1]
    except (TypeError,OSError): line=None
    return dict(type=type(value).__module__+'.'+type(value).__qualname__,line=line,
                package=(getattr(fn,'__module__','') or '').split('.')[0] or None,
                module=getattr(fn, '__module__', None), qualname=getattr(fn, '__qualname__', None),
                file=file, sha256=hashlib.sha256(raw).hexdigest() if raw is not None else None)


def callable_identity(obj, name):
    current = getattr(obj, name)
    chain, seen, item = [], set(), current
    while item is not None and id(item) not in seen and len(chain) < 16:
        seen.add(id(item))
        chain.append(dict(source(item), func_id=id(getattr(item,'__func__',item)),
                          bound_self_id=id(item.__self__) if getattr(item,'__self__',None) is not None else None))
        item = getattr(item, '__wrapped__', None)
    cls_value=inspect.getattr_static(type(obj),name,None)
    record = dict(name=name, instance_override=name in vars(obj), object_class=source(type(obj)),
                  callable_chain=chain, wrapped_chain_complete=item is None,
                  bound_to_target=getattr(current, '__self__', None) is obj,
                  ordinary_bound_method=inspect.ismethod(current),class_callable=source(cls_value),
                  same_class_function=getattr(current,'__func__',None) is cls_value and cls_value is not None)
    encoded=json.dumps(record, sort_keys=True, separators=(',', ':')).encode()
    return dict(record, fingerprint=hashlib.sha256(encoded).hexdigest(), object_id=id(obj),
                callable_id=id(current), func_id=id(getattr(current, '__func__', current)))


class MethodLease:
    """Only the exact class function bound to this object is admissible.

    No source fingerprint, module name or deployment receipt can allow a wrapper.
    """
    def __init__(self, obj, name):
        self.obj, self.name = obj, name
        self.original = getattr(obj, name)
        self.had_instance = name in vars(obj)
        self.old_instance = vars(obj).get(name)
        self.identity = callable_identity(obj, name)
        self.wrapper = None
        if not callable(self.original): raise ProbeRefused('PROBE_CALLABLE_REQUIRED:'+name)
        if getattr(self.original, '_dcat_probe_owner', None) is not None:
            raise ProbeRefused('PROBE_BOUNDARY_ALREADY_OWNED:'+name)
        if not (self.identity['ordinary_bound_method'] and self.identity['bound_to_target'] and self.identity['same_class_function']):
            code='PROBE_INSTANCE_NOT_CLASS_BOUND_METHOD' if self.had_instance else 'PROBE_CLASS_METHOD_BASELINE_REQUIRED'
            raise ProbeRefused(code+':'+name)

    def install(self, wrapper):
        current=getattr(self.obj,self.name)
        same = same_callable(current,self.original) if self.had_instance else (
            self.name not in vars(self.obj) and getattr(current,'__func__',None) is getattr(self.original,'__func__',None)
            and getattr(current,'__self__',None) is self.obj)
        if not same: raise ProbeRefused('PROBE_BASELINE_CHANGED:'+self.name)
        wrapper._dcat_probe_owner=self
        self.wrapper=wrapper
        setattr(self.obj,self.name,wrapper)

    def owned(self):
        return self.wrapper is not None and vars(self.obj).get(self.name) is self.wrapper

    def restore(self):
        if self.wrapper is None: return
        if not self.owned(): raise ProbeRefused('PROBE_OWNER_CHANGED:'+self.name)
        if self.had_instance: setattr(self.obj,self.name,self.old_instance)
        else: delattr(self.obj,self.name)
        self.wrapper=None


def restore_all(leases):
    # All-or-nothing preflight: don't partly detach after a foreign replacement.
    if any(x.wrapper is not None and not x.owned() for x in leases):
        raise ProbeRefused('PROBE_OWNER_CHANGED')
    for lease in reversed(leases): lease.restore()


INNER_NAMES=('runnable','model','_inner','__wrapped__')


def object_details(obj):
    if obj is None:return None
    cls=type(obj);attrs=vars(obj) if hasattr(obj,'__dict__') else {};modules=attrs.get('_modules',{})
    refs=[]
    for name in INNER_NAMES:
        value=attrs.get(name,modules.get(name) if isinstance(modules,dict) else None)
        if value is not None and value is not obj:
            refs.append(dict(attribute=name,object_id=id(value),class_name=type(value).__module__+'.'+type(value).__qualname__))
    return dict(object_id=id(obj),class_name=cls.__module__+'.'+cls.__qualname__,
                mro=[c.__module__+'.'+c.__qualname__ for c in cls.__mro__],inner_references=refs,
                wrapper_status='INNER_REFERENCES_PRESENT_NOT_PROOF_OF_PROXY' if refs else 'NO_DIRECT_INNER_REFERENCE_OBSERVED')


def probe_targets(worker,options=None):
    paths=(options or {}).get('target_paths',{})
    if not isinstance(paths,dict) or set(paths)-{'runner','model'}:raise ProbeRefused('PROBE_TARGET_PATH_INVALID')
    def resolve(obj,path):
        if not isinstance(path,list) or len(path)>4:raise ProbeRefused('PROBE_TARGET_PATH_INVALID')
        seen={id(obj)}
        for name in path:
            if name not in INNER_NAMES:raise ProbeRefused('PROBE_TARGET_PATH_INVALID')
            attrs=vars(obj) if hasattr(obj,'__dict__') else {};modules=attrs.get('_modules',{})
            obj=attrs.get(name,modules.get(name) if isinstance(modules,dict) else None)
            if obj is None or id(obj) in seen:raise ProbeRefused('PROBE_TARGET_REFERENCE_UNPROVEN')
            seen.add(id(obj))
        return obj
    runner=resolve(worker.model_runner,paths.get('runner',[]))
    model=resolve(getattr(runner,'model',None),paths.get('model',[]))
    return runner,model


def inspect_binding(worker,options=None):
    runner,model=probe_targets(worker,options);result={}
    for label,obj,name in (('execute_model',runner,'execute_model'),('sample_tokens',runner,'sample_tokens'),('compute_logits',model,'compute_logits')):
        if obj is None or not hasattr(obj,name):result[label]=None;continue
        row=callable_identity(obj,name);current=getattr(obj,name);pins=[]
        fn=getattr(current,'__func__',current)
        # Read closure cells and object dictionaries only; never invoke a pin or descriptor.
        candidates=[]
        for key,cell in zip(getattr(getattr(fn,'__code__',None),'co_freevars',()),getattr(fn,'__closure__',()) or ()):
            try:candidates.append((key,cell.cell_contents))
            except ValueError:pass
        candidates += list(getattr(current,'__dict__',{}).items())
        for key,value in candidates[:32]:
            if callable(value):pins.append(dict(binding=key,original=source(value)))
            if hasattr(value,'__dict__'):
                for attr,original in list(vars(value).items())[:64]:
                    if callable(original) and any(s in attr.lower() for s in ('original','orig','func','method','saved','old')):
                        pins.append(dict(binding=key,holder_class=source(type(value)),attribute=attr,original=source(original)))
        row['pinned_original_candidates']=pins[:32];result[label]=row
    return dict(schema='dcat.probe.bindings.v2',rank=getattr(worker,'rank',None),bindings=result,
                objects=dict(root_runner=object_details(worker.model_runner),root_model=object_details(getattr(worker.model_runner,'model',None)),
                    selected_runner=object_details(runner),selected_model=object_details(model)),target_paths=(options or {}).get('target_paths',{}),
                observer_state_present=hasattr(worker,'_dcat_dsv4_observer'),capture_state_present=hasattr(worker,'_dcat_noise_capture'),
                read_only=True,qualification_granted=False)


def plain(value):
    if value is None or type(value) in (str,int,float,bool): return value
    if isinstance(value,(list,tuple)) and all(type(x) in (str,int,bool) for x in value): return list(value)
    return None


def deployment_identity(worker):
    runner=worker.model_runner; model=getattr(runner,'model',None)
    configs=[]
    for label,parent in (('worker',worker),('runner',runner)):
        for path,config in (('parallel_config',getattr(parent,'parallel_config',None)),
                            ('vllm_config.parallel_config',getattr(getattr(parent,'vllm_config',None),'parallel_config',None))):
            if config is not None:
                configs.append(dict(source=label+'.'+path, values={k:plain(getattr(config,k,None)) for k in
                    ('tensor_parallel_size','pipeline_parallel_size','data_parallel_size','enable_expert_parallel','expert_parallel_size')}))
    groups={}
    try:
        from vllm.distributed import parallel_state
        for kind in ('tp','dp','ep'):
            try:
                g=getattr(parallel_state,'get_'+kind+'_group')()
                groups[kind]=dict(source='vllm.distributed.parallel_state.get_'+kind+'_group',
                    size=plain(getattr(g,'world_size',None)),members=plain(getattr(g,'ranks',None)),
                    rank_in_group=plain(getattr(g,'rank_in_group',None)))
            except Exception as e: groups[kind]=dict(size=None,members=None,reason=type(e).__name__+':'+str(e))
    except ImportError: groups={k:dict(size=None,members=None,reason='VLLM_UNAVAILABLE') for k in ('tp','dp','ep')}
    objects={}
    for name,obj in (('runner',runner),('model',model),('sampler',getattr(runner,'sampler',None)),('model_sampler',getattr(model,'sampler',None))):
        objects[name]=None if obj is None else dict(object_details(obj),class_identity=source(type(obj)))
    methods={}
    for label,obj,name in (('execute_model',runner,'execute_model'),('sample_tokens',runner,'sample_tokens'),('compute_logits',model,'compute_logits')):
        if obj is not None and hasattr(obj,name):methods[label]=callable_identity(obj,name)
        else:methods[label]=None
    return dict(schema='dcat.probe.d0.v2',rank=getattr(worker,'rank',None),worker_pid=os.getpid(),
                parallel_configs=configs,groups=groups,objects=objects,methods=methods,
                qualification_granted=False,baseline_approved=False)


def scheduler_rows(runner,scheduler):
    """Use the actual InputBatch order AND actual query offsets, never request counts alone."""
    batch=getattr(runner,'input_batch',None)
    ids=list(getattr(batch,'req_ids',[]) or [])
    counts=getattr(scheduler,'num_scheduled_tokens',{})
    starts=getattr(runner,'query_start_loc',None)
    starts=getattr(starts,'cpu',None) if starts is not None else None
    if starts is None:starts=getattr(runner,'query_start_loc_cpu',None)
    if callable(starts):starts=None  # a Tensor.cpu method is not a CPU buffer
    try: offsets=starts.tolist() if hasattr(starts,'tolist') else list(starts)
    except (TypeError,AttributeError):offsets=None
    computed=getattr(batch,'num_computed_tokens_cpu',None)
    result=[];issue=None
    if {rid for rid in ids if rid is not None and rid in counts} != {rid for rid,count in counts.items() if count>0}:
        issue='SCHEDULED_REQUEST_INPUT_BATCH_JOIN_INCOMPLETE'
    for i,rid in enumerate(ids):
        if rid is None or rid not in counts:continue
        try:
            begin,end=int(offsets[i]),int(offsets[i+1]);position=int(computed[i])
            if end-begin!=counts[rid] or begin<0 or position<0:raise ValueError()
        except (TypeError,IndexError,ValueError):issue='ACTUAL_INPUT_BATCH_OFFSETS_UNPROVEN';continue
        result.extend(dict(row=row,request_id=rid,token_position=position+row-begin) for row in range(begin,end))
    if not result:issue=issue or 'ACTUAL_INPUT_BATCH_ROWS_UNOBSERVED'
    if len({x['row'] for x in result})!=len(result):issue='INPUT_BATCH_ROWS_OVERLAP'
    return dict(rows=result,source='input_batch.req_ids/query_start_loc.cpu/num_computed_tokens_cpu',
                complete=issue is None,reason=issue)
