"""Disclosed, experimental CPU-FP64 backend for new msprobe statistics runs.

This never reads prior dump files or injector receipts. It replaces exactly one
reviewed collector method while a new synchronous statistics/mix capture runs.
The original package on disk, record selection, stack, construct and serializer
remain unchanged. Unsupported original tensor categories keep their original
statistics, including null/nonfinite values. This is a new collector identity,
not a repair or reinterpretation of historical precision=high dumps.
"""
import hashlib
import importlib
import os
from pathlib import Path
import threading
import types


MODULE = 'msprobe.core.dump.data_dump.data_processor.pytorch_processor'
REVIEWED_PROCESSOR_SHA256 = 'e981aa6fe9afa1e7a1f0654db44166d7c199674b0727d8d7ffa40626b1074011'
BACKEND = 'attention-msprobe-cpu-fp64-v1'
MAX_TENSOR_BYTES = 128 * 1024**2
MAX_TOTAL_BYTES = 512 * 1024**2
MAX_CALLS = 256


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def reviewed_method(module):
    """Bind the actual loaded function to the exact reviewed on-disk source."""
    source = Path(getattr(module, '__file__', ''))
    require(source.is_file() and sha(source) == REVIEWED_PROCESSOR_SHA256,
            'CPU_STATISTICS_PROCESSOR_SOURCE_UNREVIEWED')
    cls = getattr(module, 'PytorchDataProcessor', None)
    actual = vars(cls).get('get_stat_info') if isinstance(cls, type) else None
    require(isinstance(actual, types.FunctionType) and actual.__globals__ is vars(module)
            and Path(actual.__code__.co_filename).resolve() == source.resolve(),
            'CPU_STATISTICS_PROCESSOR_METHOD_UNPROVEN')
    compiled = compile(source.read_bytes(), actual.__code__.co_filename, 'exec', dont_inherit=True)
    classes = [code for code in compiled.co_consts
               if isinstance(code, types.CodeType) and code.co_name == 'PytorchDataProcessor']
    methods = [code for code in classes[0].co_consts
               if isinstance(code, types.CodeType) and code.co_name == 'get_stat_info'] if len(classes) == 1 else []
    require(len(methods) == 1 and actual.__code__ == methods[0],
            'CPU_STATISTICS_PROCESSOR_CODE_CHANGED')
    return cls, actual, source


class CpuFp64Statistics:
    def __init__(self):
        self.module = self.cls = self.original = self.wrapper = None
        self.owner = (os.getpid(), threading.get_ident())
        self.rows, self.errors = [], []
        self.total_bytes = 0
        self.installed = self.restored = False

    def identity(self):
        return dict(schema='demoncat.attention.statistics_backend.v1', backend=BACKEND,
                    adapter_sha256=sha(__file__), processor_module=MODULE,
                    processor_source_sha256=REVIEWED_PROCESSOR_SHA256,
                    method='PytorchDataProcessor.get_stat_info',
                    original_package_files_modified=False, runtime_method_replaced=True,
                    task='statistics', level='mix', precision='high', async_dump=False,
                    supported='complete nonempty dense nonscalar FP16/BF16/FP32/FP64 tensors',
                    reduction='blocking original-dtype CPU copy, then CPU float64 Max/Min/Mean/L2 norm',
                    delegated='original bool/integer/complex/scalar/empty/meta/NZ/unsupported handling',
                    null_and_nonfinite_values_preserved=True, historical_dumps_read=False,
                    record_filters_added=False, injector_information_read=False,
                    limits=dict(max_tensor_bytes=MAX_TENSOR_BYTES,
                                max_total_copied_bytes=MAX_TOTAL_BYTES, max_calls=MAX_CALLS))

    def install(self):
        require(not self.installed and not self.restored, 'CPU_STATISTICS_BACKEND_ALREADY_USED')
        self.module = importlib.import_module(MODULE)
        self.cls, self.original, self.source = reviewed_method(self.module)
        owner = self

        def get_stat_info(processor, data, async_dump=False, precision='low'):
            try:
                return owner._get_stat_info(processor, data, async_dump, precision)
            except BaseException as error:
                if len(owner.errors) < MAX_CALLS:
                    owner.errors.append(repr(error))
                raise

        self.wrapper = get_stat_info
        self.cls.get_stat_info = self.wrapper
        self.installed = True
        return self

    def _get_stat_info(self, processor, data, async_dump, precision):
        require(self.installed and self.cls.get_stat_info is self.wrapper,
                'CPU_STATISTICS_METHOD_OWNERSHIP_CHANGED')
        require((os.getpid(), threading.get_ident()) == self.owner,
                'CPU_STATISTICS_OWNER_CHANGED')
        require(async_dump is False and precision == 'high'
                and processor.config.summary_mode == 'statistics'
                and processor.config.task == 'statistics',
                'CPU_STATISTICS_SYNCHRONOUS_HIGH_STATISTICS_REQUIRED')
        require(len(self.rows) < MAX_CALLS, 'CPU_STATISTICS_CALL_BUDGET_EXCEEDED')
        torch = self.module.torch
        handler = processor.tensor_handler
        row = dict(index=len(self.rows), mode='original', complete_tensor_read=False)
        self.rows.append(row)
        if handler.is_empty_data(data) or handler.is_batchedtensor(data) or handler.is_gradtrackingtensor(data):
            return self.original(processor, data, async_dump, precision)
        row.update(shape=list(data.shape), dtype=str(data.dtype), device=str(data.device))
        if data.device.type == 'npu':
            npu = self.module.torch_npu
            if hasattr(npu, 'get_npu_format') and npu.get_npu_format(data) == npu.Format.FRACTAL_NZ:
                return self.original(processor, data, async_dump, precision)
        supported = (data.dtype in (torch.float16, torch.bfloat16, torch.float32, torch.float64)
                     and data.device.type in ('cpu', 'npu') and data.layout == torch.strided
                     and len(data.shape) > 0 and data.numel() > 0 and data.data_ptr() != 0)
        if not supported:
            return self.original(processor, data, async_dump, precision)
        size = data.numel() * data.element_size()
        require(size <= MAX_TENSOR_BYTES and self.total_bytes + size <= MAX_TOTAL_BYTES,
                'CPU_STATISTICS_COMPLETE_COPY_BUDGET_EXCEEDED')
        self.total_bytes += size
        # Conversion happens only after the complete blocking CPU copy. No
        # float64 NPU operation, sampling, input mutation or NaN replacement.
        with torch.no_grad():
            cpu = data.detach().to(device='cpu', copy=True, non_blocking=False)
            require(cpu.device.type == 'cpu' and cpu.dtype == data.dtype and cpu.shape == data.shape,
                    'CPU_STATISTICS_COMPLETE_COPY_CONTRACT_FAILED')
            values = cpu.to(dtype=torch.float64)
            result = self.module.TensorStatInfo()
            result.max = torch.max(values)
            result.min = torch.min(values)
            result.mean = torch.mean(values)
            result.norm = torch.linalg.vector_norm(values.reshape(-1), ord=2)
        row.update(mode='cpu_fp64', complete_tensor_read=True, original_bytes=size,
                   reduction_device='cpu', reduction_dtype='torch.float64')
        return result

    def close(self):
        require(self.installed, 'CPU_STATISTICS_BACKEND_NOT_INSTALLED')
        if self.cls.get_stat_info is not self.wrapper:
            self.errors.append('CPU_STATISTICS_METHOD_OWNERSHIP_CHANGED')
            raise RuntimeError('CPU_STATISTICS_METHOD_OWNERSHIP_CHANGED')
        self.cls.get_stat_info = self.original
        self.installed = False
        self.restored = True
        require(not self.errors, 'CPU_STATISTICS_COLLECTION_FAILED')

    def report(self):
        return dict(**self.identity(), calls=self.rows, copied_bytes=self.total_bytes,
                    errors=self.errors, installed=self.installed,
                    original_method_restored=self.restored,
                    status='NEW_COLLECTION_COMPLETE' if self.restored and not self.errors
                           and any(row['mode'] == 'cpu_fp64' for row in self.rows)
                           else 'NEW_COLLECTION_INCOMPLETE',
                    comparison_performed=False, e5=False, c2=False, c3=False)
