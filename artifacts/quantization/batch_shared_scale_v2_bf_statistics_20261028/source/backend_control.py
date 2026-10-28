"""Host controls only; does not replace or wrap any tensor-producing API."""
import ctypes
from pathlib import Path


class BackendControl:
    def __init__(self):
        import torch_npu
        self.lib = ctypes.CDLL(str(Path(__file__).resolve().parent / 'runtime/vendor/op_api/lib/libcust_opapi.so'))
        self.lib.dcat_dq_arm.argtypes = [ctypes.c_void_p, ctypes.c_int64, ctypes.c_int64, ctypes.c_int]
        self.lib.dcat_dq_finish.argtypes = [ctypes.POINTER(ctypes.c_uint64), ctypes.c_size_t]
        runtime=ctypes.CDLL(str(Path(torch_npu.__file__).parent/'lib/libtorch_npu.so'))
        resolver=runtime._Z16GetOpApiFuncAddrPKc
        resolver.argtypes=[ctypes.c_char_p]
        resolver.restype=ctypes.c_void_p
        self.resolver_checks={name:resolver(name.encode())==ctypes.cast(getattr(self.lib,name),ctypes.c_void_p).value
            for name in ('aclnnDynamicQuantV2GetWorkspaceSize','aclnnDynamicQuantV2','CanUsePTACache')}
        if not all(self.resolver_checks.values()):
            raise RuntimeError('Actual model resolver bypasses private backend: '+str(self.resolver_checks))

    @staticmethod
    def validate_input(x):
        import torch
        torch.npu.synchronize()
        if x.ndim != 2 or not x.is_contiguous() or x.storage_offset() != 0:
            raise ValueError('Only contiguous, zero-offset [T,H] inputs are supported')
        if x.dtype not in (torch.bfloat16, torch.float16):
            raise ValueError('Only BF16/FP16 inputs are supported')
        # Validation is outside the selected quantization API and outside the
        # observer window in the model harness; it never changes the input.
        if not bool(torch.isfinite(x).all()) or not bool((x.float().abs().amax(dim=-1) > 0).all()):
            raise ValueError('Nonfinite inputs or zero rows are outside the certified scope')

    def arm_validated(self, x, fault=False):
        """Caller has validated this tensor or its exact add-zero copy."""
        if self.lib.dcat_dq_arm(x.data_ptr(), *x.shape, int(fault)):
            raise RuntimeError('Backend rejected arming request')

    def arm(self, x, fault=False):
        self.validate_input(x)
        self.arm_validated(x, fault)

    def finish(self):
        import torch
        torch.npu.synchronize()
        values = (ctypes.c_uint64 * 6)()
        rc = self.lib.dcat_dq_finish(values, 6)
        record = dict(zip(('prepare_hits', 'execute_hits', 'fault_executions', 'api_version',
                           'prepare_errors', 'execution_errors'), values))
        if rc:
            raise RuntimeError('Backend did not complete exactly one selected call: '+str(record))
        return record
