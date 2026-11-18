"""Actual all-object requests and complete payloads for isolated public C2."""
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import stat

from .attention_execution import AttentionExecutionAdapter
from .call_binding import ForwardBinding
from .paged33_all_normal import AllNormalWorkerSession
from .paged33_normal_targets import NormalTargetScope
from .paged33_boundary_probe import read_scoped_launches
from .paged33_group_control import GroupC2Controller
from .paged33_s2_evidence import digest
from .serving_context import require


MAX_CAPTURE_BYTES=128*1024**2
MAX_PHASE_FILES=1024
MAX_PHASE_ARTIFACT_BYTES=512*1024**2


def _plain_stat(path):
    value=path.lstat()
    require(not stat.S_ISLNK(value.st_mode) and not getattr(value,'st_file_attributes',0)&0x400,
            'GROUP_OWNED_ARTIFACT_PATH_CHANGED:'+str(path))
    return value


def _directory_identity(path):
    value=_plain_stat(path)
    require(stat.S_ISDIR(value.st_mode) and path.resolve(strict=True)==path.absolute(),
            'GROUP_OWNED_ARTIFACT_DIRECTORY_CHANGED:'+str(path))
    return value.st_dev,value.st_ino


def _file_identity(value):
    require(stat.S_ISREG(value.st_mode) and value.st_size>=0,'GROUP_REGULAR_ARTIFACT_FILE_REQUIRED')
    return value.st_dev,value.st_ino,value.st_size,value.st_mtime_ns,value.st_ctime_ns


def _request_inventory(root,folder,root_identity,folder_identity):
    require(folder.parent==root and _directory_identity(root)==root_identity
        and _directory_identity(folder)==folder_identity,'GROUP_OWNED_REQUEST_DIRECTORY_CHANGED')
    files={};directories={'.':folder_identity};total=0;count=0
    for path in folder.rglob('*'):
        require(path.resolve(strict=True).is_relative_to(folder.resolve(strict=True)),
                'GROUP_OWNED_ARTIFACT_PATH_CHANGED:'+str(path))
        value=_plain_stat(path);relative=path.relative_to(folder).as_posix()
        if stat.S_ISDIR(value.st_mode):directories[relative]=_directory_identity(path)
        else:
            identity=_file_identity(value);files[relative]=identity
            if relative=='runtime_result.json':
                require(0<identity[2]<=MAX_CAPTURE_BYTES,'GROUP_CAPTURE_FILE_BOUND')
            else:
                count+=1;total+=identity[2]
                require(count<=MAX_PHASE_FILES and total<=MAX_PHASE_ARTIFACT_BYTES,
                        'GROUP_PHASE_ARTIFACT_RESOURCE_BOUND')
    require('runtime_result.json' in files,'GROUP_COMPLETE_CAPTURE_FILE_REQUIRED')
    require(_directory_identity(root)==root_identity and _directory_identity(folder)==folder_identity,
            'GROUP_OWNED_REQUEST_DIRECTORY_CHANGED')
    return dict(files=files,directories=directories,artifact_files=count,artifact_bytes=total)


def _read_capture(path,identity):
    # Bound allocation before reading; checking length after read_bytes is too late.
    require(0<identity[2]<=MAX_CAPTURE_BYTES,'GROUP_CAPTURE_FILE_BOUND')
    with path.open('rb') as stream:
        opened=_file_identity(os.fstat(stream.fileno()))
        # Windows path stat and handle stat expose different ctime semantics.
        # Compare dev/inode/size/mtime across APIs, and each full identity with
        # its own API before/after the read (including that API's ctime).
        require(opened[:4]==identity[:4],'GROUP_CAPTURE_FILE_IDENTITY_CHANGED')
        data=stream.read(identity[2]+1)
        require(len(data)==identity[2] and _file_identity(os.fstat(stream.fileno()))==opened,
                'GROUP_CAPTURE_FILE_CHANGED_DURING_READ')
    require(_file_identity(_plain_stat(path))==identity,'GROUP_CAPTURE_FILE_IDENTITY_CHANGED')
    return data


class GroupExecutionAdapter(AllNormalWorkerSession):
    def __init__(self,deployment,guard):
        AttentionExecutionAdapter.__init__(self,deployment,guard)
        self.owned_records=[];self.records={};self.payload_refs={};self.original=None;self.first_cleanup_error=None
        self.root_folder=self.driver.folder.parent/'group_requests'
        require(not self.root_folder.exists(),'GROUP_FRESH_REQUEST_ARTIFACT_DIRECTORY_REQUIRED')
        self.root_folder.mkdir();self.private=True
        self.artifact_root_identity=_directory_identity(self.root_folder)
        require(len(deployment.leases)==1,'GROUP_ACTUAL_PUBLIC_TICKET_REQUIRED')
        self.owner_token,self.generation=next(iter(deployment.leases))
        self.targets=NormalTargetScope(self.worker,self.candidates,owner=self,generation=self.generation)
        self.logits_binding=ForwardBinding(self.processor)
        self.control=GroupC2Controller(self,self.driver.worker_identity)
        self.driver.bind_controller(self.control)
        self.request_artifacts=[];self.native_start=0;self.folder=self.root_folder

    def _fail(self,error):
        super()._fail(error)
        if hasattr(self,'control'):self.control.failure(error)

    def expect(self,request_id,phase):
        self.control.begin_request(request_id,phase)
        if not self.deployment.simulation:self.driver.start_resource_phase(phase,self.control.index)
        self.folder=self.root_folder/(f'{len(self.request_artifacts):03d}-'+phase)
        require(_directory_identity(self.root_folder)==self.artifact_root_identity,
                'GROUP_OWNED_REQUEST_DIRECTORY_CHANGED')
        require(not self.folder.exists(),'GROUP_REQUEST_ARTIFACT_REUSE_REFUSED')
        self.folder.mkdir();(self.folder/'boundaries').mkdir()
        self.artifact_folder_identity=_directory_identity(self.folder)
        self.driver.folder=self.folder/'native';self.driver.folder.mkdir()
        self.records={};self.payload_refs={};self.original=None;self.native_start=len(self.driver.calls)
        return AttentionExecutionAdapter.expect(self,request_id,phase)

    def _execute_selected(self,actual,row,phase,once):
        try:return super()._execute_selected(actual,row,phase,once)
        finally:
            # Own partial captures/bindings before returning an exception.
            for record in self.records.values():
                if not any(record is old for old in self.owned_records):self.owned_records.append(record)

    def _native_execute(self,actual,row,invoke,binding):
        decision=self.control.observe(row,True,binding)
        native_phase='fault' if decision is not None and decision.admitted else \
            'baseline' if self.control.phase in ('reference','lossless') else 'off'
        # Native owns environment and stream until its original and independent
        # receipt return. The Coverage decision remains held until model return.
        return self.driver.execute(actual,row,native_phase,invoke,binding=binding,scope=self.targets,
                                   owner=self,generation=self.generation)

    def _invoke(self,name,original,args,kwargs):
        decode=self.current['decode']
        result=super()._invoke(name,original,args,kwargs)
        row=self.current['calls'][-1]
        if decode:
            decision=next((d for d in self.control.decisions if d.ticket.call.target.module_object_identity==row['module_object_id']
                           and d.ticket.call.target.tensor_signature=='TND_Q1_PAGE16'),None)
        else:decision=self.control.observe(row,False)
        self.control.returned(decision,row)
        return result

    def _native_calls(self):return self.driver.calls[self.native_start:]

    def _runtime_observation(self,token,capture):
        if self.deployment.simulation:return super()._runtime_observation(token,capture)
        return read_scoped_launches(self.driver.runtime_dir,token,expected_pid=capture.pid,
            expected_native_tid=capture.native_tid,runtime_index=self.driver.runtime_index)

    def collect(self):
        try:
            require(self.folder.parent==self.root_folder
                and _directory_identity(self.root_folder)==self.artifact_root_identity
                and _directory_identity(self.folder)==self.artifact_folder_identity,
                'GROUP_OWNED_REQUEST_DIRECTORY_CHANGED')
            capture=super().collect()
            inventory=_request_inventory(self.root_folder,self.folder,self.artifact_root_identity,self.artifact_folder_identity)
            path=self.folder/'runtime_result.json';data=_read_capture(path,inventory['files']['runtime_result.json'])
            require(json.loads(data)==capture,'GROUP_CAPTURE_ORIGINAL_CONTENT_DIFFERS')
            require(_request_inventory(self.root_folder,self.folder,self.artifact_root_identity,self.artifact_folder_identity)==inventory,
                    'GROUP_REQUEST_ARTIFACT_IDENTITY_CHANGED')
            row=capture['natural_request']
            artifact=dict(phase=row['phase'],request_id=row['request_id'],path=str(path),
                sha256=hashlib.sha256(data).hexdigest(),bytes=len(data),actual_calls=56,decode_calls=28,
                complete_payload_count=87,native_calls=len(capture['native_calls']),
                owned_artifacts=dict(files_excluding_capture=inventory['artifact_files'],
                    bytes_excluding_capture=inventory['artifact_bytes'],inventory_sha256=digest(inventory),
                    capture_identity=list(inventory['files']['runtime_result.json']),
                    root_identity=list(self.artifact_root_identity),request_directory_identity=list(self.artifact_folder_identity)))
            # Preserve the complete original even if admission/comparison fails.
            self.request_artifacts.append(artifact)
            self.control.complete_request(capture,artifact)
            if not self.deployment.simulation:self.driver.finish_resource_phase(row['phase'])
            # RPC/query keeps immutable artifact references rather than copying
            # 60 complete POD/boundary graphs through every Host history item.
            self.rows[-1]=deepcopy(artifact);self.request_rows=[];self.payload_refs={}
            return artifact
        except BaseException as error:self._fail(error);raise

    def prepare_off_recovery(self):
        self._owned();self.control.prepare_recovery()
        if not self.deployment.simulation:self.driver.abandon_resource_phase()
        partial={k:v for k,v in (self.current or {}).items() if k!='entry_error'}
        receipt=dict(expected=self.expected,partial=deepcopy(partial),returned_forwards=deepcopy(getattr(self,'request_rows',[])),
                     complete=False,first_error=self.last_error)
        path=self.root_folder/(f'failed-{len(self.failed_requests):03d}.json')
        path.write_text(json.dumps(receipt,allow_nan=False,indent=2)+'\n')
        self.failed_requests.append(dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),first_error=self.last_error))
        self.expected=self.current=None;self.stack.clear();self.parents.clear();self.writers.clear();self.payload_refs={}
        self.last_error=None

    def close(self):
        try:
            require(not self.needs_off,'GROUP_ACTUAL_FULL_OFF_BEFORE_CLEANUP')
            AttentionExecutionAdapter.close(self)
            state=self.driver.stop()
            require(not state['scope_owned'] and not state['scope_active'] and not state['environment_owned'],
                    'GROUP_NATIVE_ZERO_SCOPE_BEFORE_REFERENCE_RELEASE')
            for record in getattr(self,'owned_records',[]):
                if 'capture' in record:record['capture'].release()
                if 'binding' in record:record['binding'].release(owner=self,generation=self.generation)
            self.payload_refs={}
        except BaseException as error:
            if getattr(self,'first_cleanup_error',None) is None:self.first_cleanup_error=repr(error)
            raise

    def state(self):
        value=AttentionExecutionAdapter.state(self)
        value['owned_input_references']=sum(not r['binding'].closed for r in getattr(self,'owned_records',[]) if 'binding' in r)
        value['first_cleanup_error']=getattr(self,'first_cleanup_error',None)
        return value

    def write_coverage(self):
        from dataclasses import asdict
        report=self.control.session.report() if self.control.session else None
        path=self.root_folder/'coverage.json'
        path.write_text(json.dumps(dict(coverage=report,control=self.control.summary(),
            recovery_history=self.control.recovery_history,requests=self.request_artifacts,
            group=asdict(self.control.session.group) if self.control.session else None,
            admissions=self.control.admissions,resource_plan=self.driver.budget,
            resource_checks=self.driver.resource_checks,
            phase_resources=getattr(self.driver,'phase_resources',[])),indent=2,allow_nan=False)+'\n')
        return dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),bytes=path.stat().st_size)
