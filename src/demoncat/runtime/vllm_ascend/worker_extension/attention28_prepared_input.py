"""Trusted, single-target normal window for the compiled TND input observer.

Holds the actual public objects. It never loads a library, calls an operator,
rewrites arguments or supplies the denominator replay gate.
"""
import hashlib
import json
import os
from pathlib import Path
import threading

from .serving_context import require

PREFIX='DCAT28_INPUT_'
EVIDENCE_ROOT=Path('/evidence')


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def identity():
    text=Path('/proc/self/stat').read_text();start=text[text.rfind(')')+2:].split()[19]
    return dict(pid=os.getpid(),native_tid=threading.get_native_id(),python_thread_ident=threading.get_ident(),process_start_ticks=start)


class PreparedInputCapture:
    """One natural matched object, one original invocation; not F authority."""
    def __init__(self,worker,*,build_completion,build_sha256,evidence_dir):
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

    def _check(self):
        require(identity()==self.owner and self.worker.model_runner is self.runner and self.runner.model is self.model,
            'TND_INPUT_WORKER_OR_THREAD_IDENTITY_CHANGED')
        require(not any(k.startswith('DCAT28_TND_') for k in os.environ),'TND_INPUT_NORMAL_WINDOW_HAS_FOREIGN_POLICY')

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
        self.refs=tuple(actual[name] for name in ('query','key','value','atten_mask'))
        self.used=True;self.active={};desired={PREFIX+k:v for k,v in values.items()}
        self.receipt=dict(packet,public_artifact=str(self.root/(token+'.public.json')),native_artifact=values['LOG'],
            window_complete=False,actual_native_observed=False,prepared_input_join=False)
        # Preserve identity before original native execution; an empty journal
        # remains an empty journal and cannot be promoted to a hit.
        try:
            with Path(self.receipt['public_artifact']).open('x',encoding='utf-8') as stream:
                json.dump(packet,stream,indent=2,allow_nan=False);stream.write('\n')
            for key,value in desired.items():
                try:os.environ[key]=value
                finally:
                    if os.environ.get(key)==value:self.active[key]=value
        except BaseException as error:
            self.first_error=self.first_error or repr(error);raise
        return self.receipt

    def returned(self,output):
        self._check();require(self.active is not None and self.refs is not None,'TND_INPUT_NO_LIVE_RETURN_SCOPE')
        require(type(output) is tuple and len(output)==2,'TND_INPUT_ORIGINAL_RETURN_CONTAINER')
        from fault_injection.fia_input_journal import public_tensor_metadata
        self.refs+=output
        row=dict(schema='demoncat.tnd28.public_return.v1',token=self.receipt['token'],identity=self.owner,
            owner=self.receipt['owner'],generation=self.receipt['generation'],request_sha256=self.receipt['request_sha256'],
            input_sha256=self.receipt['input_sha256'],outputs=[public_tensor_metadata(x) for x in output],
            return_container_type='tuple',original_calls=1)
        path=self.root/(self.receipt['token']+'.return.json')
        with path.open('x',encoding='utf-8') as stream:json.dump(row,stream,indent=2,allow_nan=False);stream.write('\n')
        self.receipt['returned_artifact']=str(path)

    def stop(self):
        self._check()
        if self.active is None:return
        try:
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
