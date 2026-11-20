"""Fresh value verification inside the original drained Execute window.

No F authority. Complete logical values are hashed in memory and discarded;
no tensors are published. Device addresses come only from the compiled ABI.
"""
import ctypes as C
import hashlib
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


class LivePreparedValuePlan(LiveTypedPlan):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        from .paged33_loading import symbol_owner
        name='dcat_tnd28_value_read_v1';owner=symbol_owner(self.query_owner.library,name)
        require(owner['path']==str(self.query_owner.path) and owner['sha256']==self.query_owner.sha256,
                'TND_LIVE_VALUE_READER_OWNER_CHANGED')
        self.reader=getattr(self.query_owner.library,name)
        self.reader.argtypes=[C.c_char_p,C.c_uint32,C.c_void_p,C.c_size_t];self.reader.restype=C.c_int
        self.value_checks=[]
    def _read(self,index,size):
        output=C.create_string_buffer(size)
        require(self.reader(self.token.encode(),index,output,size)==0,'TND_LIVE_VALUE_NATIVE_READ_REFUSED')
        return output.raw
    def _callback(self,token,stage,pointer,length):
        status=super()._callback(token,stage,pointer,length)
        if status!=1 or stage!=2:return status
        try:
            self.value_checks=verify_live_values(self.plan,self._read)
            self.binding.check_current(self.plan,read_values=False)
            self.events[-1]['fresh_value_checks']=self.value_checks
            return 1
        except BaseException as error:
            self.first_error=self.first_error or repr(error)[:1024];return 0
    def snapshot(self):
        result=super().snapshot()
        result.update(current_value_proof=len(self.value_checks)==5 and self.first_error is None,
            value_checks=list(self.value_checks),prepared_parent_argument_join=False,
            fault_authorized=False,proof_scope='ACTUAL_TYPED_PUBLIC_VALUES_AND_DRAINED_CONTIGUOUS_CHILD_OUTPUT')
        return result
