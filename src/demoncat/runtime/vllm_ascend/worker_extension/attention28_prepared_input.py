"""Trusted, single-target normal window for the compiled TND input observer.

Holds the actual public objects. It never loads a library, calls an operator,
rewrites arguments or supplies the denominator replay gate.
"""
import hashlib
import json
import os
from pathlib import Path
import threading
import ctypes

from .serving_context import require

PREFIX='DCAT28_INPUT_'
EVIDENCE_ROOT=Path('/evidence')


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def identity():
    text=Path('/proc/self/stat').read_text();start=text[text.rfind(')')+2:].split()[19]
    return dict(pid=os.getpid(),native_tid=threading.get_native_id(),python_thread_ident=threading.get_ident(),process_start_ticks=start)


def _open_borrow_query(path,expected):
    """Use only an already loaded, source-pinned private API's read-only query."""
    from .paged33_loading import symbol_owner
    path=Path(path).resolve(strict=True)
    require(path.name=='libcust_opapi.so' and 0<path.stat().st_size<=32*1024**2
        and hashlib.sha256(path.read_bytes()).hexdigest()==expected,'TND_BORROW_EXISTING_PRIVATE_LIBRARY_CHANGED')
    require(hasattr(os,'RTLD_NOLOAD'),'TND_BORROW_NOLOAD_REQUIRED')
    library=ctypes.CDLL(str(path),mode=os.RTLD_NOLOAD|os.RTLD_LOCAL|os.RTLD_NOW)
    name='dcat_tnd28_execute_borrow_query_v1';symbol=symbol_owner(library,name)
    require(symbol['path']==str(path) and symbol['sha256']==expected,'TND_BORROW_QUERY_SYMBOL_OWNER_CHANGED')
    query=getattr(library,name);query.argtypes=[ctypes.c_char_p,ctypes.c_void_p,ctypes.c_size_t];query.restype=ctypes.c_int
    def current(token):
        # The result comes from this thread's actual Execute return, not a log.
        require(library._handle,'TND_BORROW_LOADED_LIBRARY_HANDLE_MISSING')
        out=ctypes.create_string_buffer(4096)
        require(query(token.encode(),out,len(out))==0,'TND_BORROW_LIVE_RETURN_UNAVAILABLE')
        value=json.loads(out.value);owner=value['function']
        require(owner['address']==symbol['address'] and owner['mapped_file_identity_matches'] is True
            and Path(owner['path']).resolve(strict=True)==path
            and owner['file_bytes']==path.stat().st_size
            and hashlib.sha256(path.read_bytes()).hexdigest()==expected,'TND_BORROW_QUERY_MAPPED_LIBRARY_CHANGED')
        return value
    current.library,current.path,current.sha256=library,path,expected
    return current


class PreparedInputCapture:
    """One natural matched object, one original invocation; not F authority."""
    def __init__(self,worker,*,build_completion,build_sha256,evidence_dir,
                 execute_borrow_library=None,execute_borrow_sha256=None):
        path=Path(build_completion).resolve(strict=True)
        before=path.stat();require(0<before.st_size<=4*1024**2,'TND_INPUT_BUILD_IDENTITY_OR_BOUND')
        raw=path.read_bytes();after=path.stat()
        require(hashlib.sha256(raw).hexdigest()==build_sha256 and
            (before.st_dev,before.st_ino,before.st_size,before.st_mtime_ns)==
            (after.st_dev,after.st_ino,after.st_size,after.st_mtime_ns),'TND_INPUT_BUILD_IDENTITY_OR_BOUND')
        self.build=json.loads(raw);observation=self.build.get('tnd28_compiled_input_observer',{})
        require(self.build.get('variant')=='tnd_prefill_denominator_replay'
            and self.build.get('status')=='CPU_BUILD_ONLY_REQUIRES_NATIVE_QUALIFICATION'
            and observation.get('schema')=='demoncat.tnd28.compiled_input_source.v1'
            and observation.get('observer_only') is True and observation.get('fault_authorized') is False,
            'TND_INPUT_ACTUAL_COMPILED_OBSERVER_BUILD_REQUIRED')
        self.worker=worker;self.runner=worker.model_runner;self.model=self.runner.model;self.owner=identity()
        self.root=Path(evidence_dir).resolve()
        require(self.root.is_relative_to(EVIDENCE_ROOT.resolve()) and not self.root.exists(),'TND_INPUT_FRESH_OWNED_FOLDER')
        require(not any(k.startswith(PREFIX) for k in os.environ),'TND_INPUT_FOREIGN_OBSERVER_ENVIRONMENT')
        self.root.mkdir(parents=True);self.build_sha256=build_sha256
        self.used=False;self.active=None;self.refs=None;self.first_error=None;self.receipt=None
        self.replay_binding=self.live_plan=None;self.borrow_query=None;self.typed_plan=None
        require((execute_borrow_library is None)==(execute_borrow_sha256 is None),'TND_BORROW_EXACT_LIBRARY_PAIR_REQUIRED')
        if execute_borrow_library is not None:
            require(self.build.get('tnd28_execute_borrow_header_sha256')==observation.get('execute_borrow_header_sha256')
                and isinstance(observation.get('execute_borrow_header_sha256'),str)
                and len(observation['execute_borrow_header_sha256'])==64,'TND_BORROW_FRESH_COMPILED_HEADER_REQUIRED')
            proof=self.build.get('tnd28_execute_borrow_query',{})
            require(proof.get('symbol')=='dcat_tnd28_execute_borrow_query_v1'
                and proof.get('library_sha256')==execute_borrow_sha256 and proof.get('read_only') is True
                and proof.get('fault_authorized') is False,'TND_BORROW_COMPILED_QUERY_IDENTITY_REQUIRED')
            from .attention28_typed_plan import SYMBOLS
            require(proof.get('live_public_plan_symbols')==list(SYMBOLS),
                    'TND_BORROW_COMPILED_LIVE_PLAN_ABI_REQUIRED')
            self.borrow_query=_open_borrow_query(execute_borrow_library,execute_borrow_sha256)

    def _check(self):
        require(identity()==self.owner and self.worker.model_runner is self.runner and self.runner.model is self.model,
            'TND_INPUT_WORKER_OR_THREAD_IDENTITY_CHANGED')
        self._check_policy()

    def _check_policy(self):
        require(not any(k.startswith('DCAT28_TND_') for k in os.environ),'TND_INPUT_NORMAL_WINDOW_HAS_FOREIGN_POLICY')

    def prepare_plan(self,row):
        """Called by the current adapter before its Native observer scope."""
        if self.borrow_query is None:return
        self._check();require(self.replay_binding is None,'TND_BORROW_SINGLE_PLAN_REQUIRED')
        from . import attention_execution
        from .attention28_replay_binding import ReplayInputBinding
        self.replay_binding=ReplayInputBinding(attention_execution._ACTIVE)
        try:self.live_plan=self.replay_binding.begin(row)
        except BaseException as error:
            self.first_error=self.first_error or repr(error);raise

    def enter(self,token,row,actual,api):
        self._check();require(not self.used and self.active is None,'TND_INPUT_SINGLE_CURRENT_CALL_ONLY')
        scope=row['native_census_scope']
        require(scope['worker_object_id']==id(self.worker) and scope['runner_object_id']==id(self.runner)
            and scope['model_object_id']==id(self.model) and scope['phase']=='observe',
            'TND_INPUT_ACTUAL_WORKER_NORMAL_REQUEST_REQUIRED')
        require(not any(k.startswith(PREFIX) for k in os.environ),'TND_INPUT_FOREIGN_OBSERVER_ENVIRONMENT')
        from fault_injection.fia_input_journal import public_tensor_metadata
        public={name:public_tensor_metadata(actual[name]) for name in ('query','key','value','atten_mask')}
        packet=dict(schema='demoncat.tnd28.public_input.v1',token=token,identity=self.owner,
            owner=scope['owner'],generation=scope['generation'],module_object_id=row['module_object_id'],
            worker_object_id=id(self.worker),model_object_id=id(self.model),request_id=row['request_id'],
            request_sha256=digest(row['request_id']),module_path=row['module_path'],scheduler=row['scheduler'],
            tensors=public,api=api,build_sha256=self.build_sha256,
            selection='FIRST_ACTUAL_MATCHED_OBJECT_IN_ONE_NATURAL_REQUEST',
            input_replaced=False,fault_authorized=False)
        packet['input_sha256']=digest(public)
        values=dict(MODE='observe',TOKEN=token,OWNER=scope['owner'],GENERATION=str(scope['generation']),
            MODULE=str(row['module_object_id']),REQUEST=packet['request_sha256'],HASH=packet['input_sha256'],
            PID=str(self.owner['pid']),TID=str(self.owner['native_tid']),START=self.owner['process_start_ticks'],
            LOG=str(self.root/(token+'.jsonl')))
        if self.borrow_query is not None:values['EXECUTE_BORROW']='normal_guard'
        self.refs=tuple(actual[name] for name in ('query','key','value','atten_mask'))
        self.used=True;self.active={};desired={PREFIX+k:v for k,v in values.items()}
        self.receipt=dict(packet,public_artifact=str(self.root/(token+'.public.json')),native_artifact=values['LOG'],
            window_complete=False,actual_native_observed=False,prepared_input_join=False)
        # Preserve identity before original native execution; an empty journal
        # remains an empty journal and cannot be promoted to a hit.
        try:
            if self.borrow_query is not None:
                require(self.live_plan is not None,'TND_BORROW_PRE_SCOPE_LIVE_PLAN_REQUIRED')
                self.replay_binding.check_current(self.live_plan,read_values=False)
                self.receipt['live_public_plan']=self.replay_binding.snapshot()
            with Path(self.receipt['public_artifact']).open('x',encoding='utf-8') as stream:
                json.dump(packet,stream,indent=2,allow_nan=False);stream.write('\n')
            for key,value in desired.items():
                try:os.environ[key]=value
                finally:
                    if os.environ.get(key)==value:self.active[key]=value
            if self.borrow_query is not None:
                from .attention28_typed_plan import LiveTypedPlan
                self.typed_plan=LiveTypedPlan(self.borrow_query,self.replay_binding,self.live_plan,token)
                self.typed_plan.install()
                self.receipt['live_typed_plan']=self.typed_plan.snapshot()
        except BaseException as error:
            self.first_error=self.first_error or repr(error);raise
        return self.receipt

    def returned(self,output):
        self._check();require(self.active is not None and self.refs is not None,'TND_INPUT_NO_LIVE_RETURN_SCOPE')
        require(type(output) is tuple and len(output)==2,'TND_INPUT_ORIGINAL_RETURN_CONTAINER')
        if self.live_plan is not None:
            self.replay_binding.check_current(self.live_plan,read_values=False)
            self.receipt['execute_borrow']=self._borrow_return()
            require(self.receipt['execute_borrow']['borrow']['first_error']==0,
                    'TND_BORROW_ORIGINAL_CALL_OR_CLEANUP_FAILED')
            self.receipt['live_typed_plan']=self.typed_plan.snapshot()
            require(self.typed_plan.first_error is None and len(self.typed_plan.events)==2,
                    'TND_BORROW_TWO_ACTUAL_TYPED_CHECKS_REQUIRED')
            native_plan=self.typed_plan.state()
            require(all(native_plan[name]==value for name,value in dict(bound=1,prepare_entered=1,
                prepare_accepted=1,prepare_admitted=1,execute_entered=1,safe_release=1,
                terminal_uncertain=0,first_error=0).items()),'TND_BORROW_NATIVE_TYPED_GUARD_NOT_COMPLETE')
        from fault_injection.fia_input_journal import public_tensor_metadata
        self.refs+=output
        row=dict(schema='demoncat.tnd28.public_return.v1',token=self.receipt['token'],identity=self.owner,
            owner=self.receipt['owner'],generation=self.receipt['generation'],request_sha256=self.receipt['request_sha256'],
            input_sha256=self.receipt['input_sha256'],outputs=[public_tensor_metadata(x) for x in output],
            return_container_type='tuple',original_calls=1)
        path=self.root/(self.receipt['token']+'.return.json')
        with path.open('x',encoding='utf-8') as stream:json.dump(row,stream,indent=2,allow_nan=False);stream.write('\n')
        self.receipt['returned_artifact']=str(path)

    def _borrow_return(self):
        row=self.borrow_query(self.receipt['token'])
        require(all(row.get(k)==self.receipt[k] for k in ('token','owner','generation','module_object_id',
            'request_sha256','input_sha256')),'TND_BORROW_ACTUAL_PLAN_RETURN_CHANGED')
        state=row['borrow']
        require(state.get('schema')=='demoncat.tnd28.execute_borrow.v1'
            and state['pid']==self.owner['pid'] and state['tid']==self.owner['native_tid']
            and state['left']==1 and state['complete']==1 and state['drained']==1
            and state['terminal_uncertain']==0 and state['fault_authorized'] is False,
            'TND_BORROW_TERMINAL_UNCERTAIN_OR_NOT_DRAINED')
        return row

    def stop(self):
        self._check()
        if self.active is None:
            if self.replay_binding is not None and self.replay_binding.active is not None:
                plan=self.replay_binding.active
                require(plan.row['original_calls']==0,'TND_BORROW_ORIGINAL_WITHOUT_SCOPE_UNCERTAIN')
                self.replay_binding.close_public(plan);self.live_plan=None
            return
        try:
            if self.live_plan is None and self.replay_binding is not None:
                self.live_plan=self.replay_binding.active
            if self.live_plan is not None:
                # Only the real compiled return can confirm an entered Execute
                # borrow; missing/uncertain state never triggers a pointer retry.
                if self.live_plan.row['original_calls']:
                    typed=self.typed_plan.state() if self.typed_plan is not None else None
                    refused_before_parent=(typed is not None and typed['prepare_admitted']==0
                        and typed['safe_release']==1
                        and typed['terminal_uncertain']==0)
                    if not refused_before_parent:self.receipt['execute_borrow']=self._borrow_return()
                if self.typed_plan is not None:
                    self.typed_plan.clear()
                    self.receipt['live_typed_plan']=self.typed_plan.snapshot()
                self.replay_binding.close_public(self.live_plan);self.live_plan=None
            # Retire each owned entry only after readback. Foreign values leave
            # the remaining ledger and public tensor references available.
            for key,value in list(self.active.items()):
                require(os.environ.get(key)==value,'TND_INPUT_FOREIGN_ENVIRONMENT_REPLACEMENT:'+key)
                try:del os.environ[key]
                finally:
                    if key not in os.environ:self.active.pop(key)
            self.receipt['window_complete']=True
            native=Path(self.receipt['native_artifact'])
            self.receipt['native_records_present']=native.is_file() and native.stat().st_size>0
            self.active=None;self.refs=None
        except BaseException as error:
            self.first_error=self.first_error or repr(error);raise
