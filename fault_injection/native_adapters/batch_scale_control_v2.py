"""Portable V2 controls; missing exported resolver remains explicit and pending.

With a hidden torch_npu resolver, actual binding can only be established after
the selected real API enters this DSO's typed prepare/execute pair. Loading a
symbol or arming a pointer is never reported as an actual invocation.
"""
import ctypes
import hashlib
from pathlib import Path


class BackendControl:
    def __init__(self):
        import torch_npu
        self.lib=ctypes.CDLL(str(Path(__file__).resolve().parent/'runtime/vendor/op_api/lib/libcust_opapi.so'))
        self.lib.dcat_dq_arm.argtypes=[ctypes.c_void_p,ctypes.c_int64,ctypes.c_int64,ctypes.c_int]
        self.lib.dcat_dq_finish.argtypes=[ctypes.POINTER(ctypes.c_uint64),ctypes.c_size_t]
        path=Path(torch_npu.__file__).parent/'lib/libtorch_npu.so';runtime=ctypes.CDLL(str(path))
        names=('aclnnDynamicQuantV2GetWorkspaceSize','aclnnDynamicQuantV2','CanUsePTACache')
        resolver=getattr(runtime,'_Z16GetOpApiFuncAddrPKc',None)
        self.resolver_identity=dict(torch_npu_version=torch_npu.__version__,library_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            exported_global_resolver=resolver is not None)
        if resolver is not None:
            resolver.argtypes=[ctypes.c_char_p];resolver.restype=ctypes.c_void_p
            self.resolver_checks={name:resolver(name.encode())==ctypes.cast(getattr(self.lib,name),ctypes.c_void_p).value for name in names}
            if not all(self.resolver_checks.values()):raise RuntimeError('Exported resolver bypasses private V2 backend')
            self.resolver_strategy='EXPORTED_RESOLVER_AND_ACTUAL_ENTRY_COUNTS'
        else:
            # Symbols must exist in the approved DSO, but existence is not binding.
            for name in names:getattr(self.lib,name)
            self.resolver_checks={name:None for name in names}
            self.resolver_strategy='ACTUAL_TYPED_ENTRY_COUNTS_REQUIRED'

    @staticmethod
    def validate_input(x):
        import torch
        torch.npu.synchronize()
        if x.ndim!=2 or not x.is_contiguous() or x.storage_offset()!=0:raise ValueError('Contiguous zero-offset [T,H] required')
        if x.dtype not in (torch.bfloat16,torch.float16):raise ValueError('BF16/FP16 required')
        if not bool(torch.isfinite(x).all()) or not bool((x.float().abs().amax(-1)>0).all()):raise ValueError('Finite nonzero rows required')

    def arm_validated(self,x,fault=False):
        if self.lib.dcat_dq_arm(x.data_ptr(),*x.shape,int(fault)):raise RuntimeError('Native scope rejected')

    def finish(self):
        import torch
        torch.npu.synchronize();values=(ctypes.c_uint64*6)()
        rc=self.lib.dcat_dq_finish(values,6)
        record=dict(zip(('prepare_hits','execute_hits','fault_executions','api_version','prepare_errors','execution_errors'),values))
        if rc:raise RuntimeError('Actual selected V2 prepare/execute was not proven: '+str(record))
        return record
