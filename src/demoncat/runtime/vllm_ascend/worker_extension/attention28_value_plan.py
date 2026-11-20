"""Fresh value verification inside the original drained Execute window.

No F authority. Complete logical values are hashed in memory and discarded;
no tensors are published. Device addresses come only from the compiled ABI.
"""
import ctypes as C
import hashlib
import json
import os
import threading
from pathlib import Path
from .attention28_typed_plan import LiveTypedPlan
from .serving_context import require

SIZES=(8388608,4194304,16777216,4194304,4194304)
NAMES=('query','key','value','atten_mask','prepared_value')


def verify_live_values(plan,read):
    require(plan.public_metadata['value']['storage_bytes']==16777216,
            'TND_LIVE_VALUE_EXACT_STORAGE_REQUIRED')
    result=[];value_digest=None
    for index,(name,size) in enumerate(zip(NAMES,SIZES)):
        raw=read(index,size)
        require(type(raw) is bytes and len(raw)==size,'TND_LIVE_VALUE_READ_SIZE:'+name)
        if name=='value':
            # Actual, already validated V: [2048,8,128], strides [4096,128,1],
            # offset3072. CPU logical indexing never feeds the real FIA.
            digest=hashlib.sha256()
            for row in range(2048):digest.update(memoryview(raw)[(row*4096+3072)*2:(row*4096+4096)*2])
            observed=digest.hexdigest();logical_bytes=4194304;value_digest=observed
        else:
            observed=hashlib.sha256(raw).hexdigest();logical_bytes=size
        reference_name='value' if name=='prepared_value' else name
        reference=plan.before['inputs'][reference_name]['snapshot']
        require(reference['bytes']==logical_bytes and reference['sha256']==observed,
                'TND_LIVE_VALUE_CHANGED:'+name)
        if name=='prepared_value':require(observed==value_digest,'TND_PREPARED_VALUE_NOT_CURRENT_PUBLIC_VALUE')
        result.append(dict(name=name,read_bytes=size,logical_bytes=logical_bytes,sha256=observed,
                           initial_public_sha256=reference['sha256'],unchanged=True))
    return result


# Pinned actual Slice registration/entry from the independent normal NPU join.
# Other SDK binaries require a new review; a matching key alone is insufficient.
COPY_ELF_SHA256='ceadb83454273d7a9dd67c343c2dd7ca0f756ff0f44340d42126ec584a36b63d'


def verify_copy_launch(packet,token,*,pid,tid,file_digest):
    from .attention28_typed_plan import _unique
    require(type(packet) is bytes and 0<len(packet)<131072,'TND_COPY_PACKET_BOUND')
    data=json.loads(packet,object_pairs_hook=_unique,
                    parse_constant=lambda _: (_ for _ in ()).throw(ValueError('NONFINITE_COPY_VALUE')))
    require(data['schema']=='demoncat.tnd28.current_copy_launch.v1','TND_COPY_PACKET_SCHEMA')
    row=data['launch']
    for key in ('before','after','begin_ns','end_ns','stream','prepared_v_address'):
        require(type(data[key]) is int,'TND_COPY_INTEGER:'+key)
    require(data['before']>=0 and data['after']==data['before']+1
        and row['sequence']==data['after'],'TND_COPY_SINGLE_FRESH_LAUNCH')
    require(row['schema']=='demoncat.live_launch_readback.v1' and row['event']=='launch'
        and row['call_id']==token and row['pid']==pid and row['tid']==tid,'TND_COPY_CURRENT_OWNER')
    require(0<data['begin_ns']<=row['launch_begin_ns']<=row['launch_end_ns']<=data['end_ns']
        and row['stream']==data['stream']>0 and data['prepared_v_address']>0,'TND_COPY_ACTUAL_CALL_WINDOW')
    require(row['snapshot_valid'] is True and type(row['result']) is int and row['result']==0
        and row['observer_mutates_arguments'] is False and row['generation']>0,'TND_COPY_SUCCESSFUL_READONLY_LAUNCH')
    require(row['tiling_key']=='524000000' and row['block_dim']==40
        and file_digest(row['binary_file'])==COPY_ELF_SHA256,'TND_COPY_ACTUAL_REVIEWED_SLICE')
    return dict(schema=data['schema'],pid=pid,tid=tid,token=token,before=data['before'],after=data['after'],
        begin_ns=data['begin_ns'],end_ns=data['end_ns'],tiling_key=row['tiling_key'],block_dim=row['block_dim'],
        binary_sha256=COPY_ELF_SHA256,launch_packet_sha256=hashlib.sha256(packet).hexdigest(),
        current_child_launch=True,prepared_parent_argument_join=False,fault_authorized=False)


def _current_binary_digest(name):
    path=Path(name).resolve(strict=True);before=path.stat()
    require(path.is_file() and 0<before.st_size<=128*1024**2,'TND_COPY_BINARY_BOUND')
    digest=hashlib.sha256(path.read_bytes()).hexdigest();after=path.stat()
    require((before.st_dev,before.st_ino,before.st_size,before.st_mtime_ns)==
        (after.st_dev,after.st_ino,after.st_size,after.st_mtime_ns),'TND_COPY_BINARY_CHANGED')
    return digest


class LivePreparedValuePlan(LiveTypedPlan):
    def __init__(self,*args,check_live_copy_launch=False,**kwargs):
        super().__init__(*args,**kwargs)
        from .paged33_loading import symbol_owner
        name='dcat_tnd28_value_read_v1';owner=symbol_owner(self.query_owner.library,name)
        require(owner['path']==str(self.query_owner.path) and owner['sha256']==self.query_owner.sha256,
                'TND_LIVE_VALUE_READER_OWNER_CHANGED')
        self.reader=getattr(self.query_owner.library,name)
        self.reader.argtypes=[C.c_char_p,C.c_uint32,C.c_void_p,C.c_size_t];self.reader.restype=C.c_int
        self.value_checks=[];self.copy_launch_proof=None;self.copy_reader=None
        require(type(check_live_copy_launch) is bool,'TND_COPY_EXPLICIT_CHECK_MODE')
        if check_live_copy_launch:
            name='dcat_tnd28_copy_launch_read_v1';owner=symbol_owner(self.query_owner.library,name)
            require(owner['path']==str(self.query_owner.path) and owner['sha256']==self.query_owner.sha256,
                    'TND_COPY_NATIVE_READER_OWNER_CHANGED')
            self.copy_reader=getattr(self.query_owner.library,name)
            self.copy_reader.argtypes=[C.c_char_p,C.c_void_p,C.c_size_t];self.copy_reader.restype=C.c_int
    def _read(self,index,size):
        output=C.create_string_buffer(size)
        require(self.reader(self.token.encode(),index,output,size)==0,'TND_LIVE_VALUE_NATIVE_READ_REFUSED')
        return output.raw
    def _callback(self,token,stage,pointer,length):
        status=super()._callback(token,stage,pointer,length)
        if status!=1 or stage!=2:return status
        try:
            self.value_checks=verify_live_values(self.plan,self._read)
            if self.copy_reader is not None:
                output=C.create_string_buffer(131072)
                require(self.copy_reader(self.token.encode(),output,len(output))==0,'TND_COPY_NATIVE_READ_REFUSED')
                self.copy_launch_proof=verify_copy_launch(output.value,self.token,pid=os.getpid(),
                    tid=threading.get_native_id(),file_digest=_current_binary_digest)
                self.events[-1]['current_copy_launch']=self.copy_launch_proof
            self.binding.check_current(self.plan,read_values=False)
            self.events[-1]['fresh_value_checks']=self.value_checks
            self.events[-1]['read_input_values']=True
            return 1
        except BaseException as error:
            self.first_error=self.first_error or repr(error)[:1024];return 0
    def snapshot(self):
        result=super().snapshot()
        result.update(current_value_proof=len(self.value_checks)==5 and self.first_error is None,
            value_checks=list(self.value_checks),copy_launch_proof=self.copy_launch_proof,prepared_parent_argument_join=False,
            fault_authorized=False,proof_scope='ACTUAL_TYPED_PUBLIC_VALUES_AND_DRAINED_CONTIGUOUS_CHILD_OUTPUT')
        return result
