"""Explicit CPU registrations/source doubles, never an NPU qualification."""
from contextlib import ExitStack
import hashlib
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

import torch

from demoncat.runtime.vllm_ascend.worker_extension import gemm_lineage
from fault_injection.deployments import postwrite_npu_consumers as consumers


class CustomOperators:
    def __enter__(self):
        self.stack = ExitStack()
        self.folder = Path(self.stack.enter_context(tempfile.TemporaryDirectory()))
        self.calls = []
        self.library = torch.library.Library('vllm', 'FRAGMENT')
        self.stack.callback(self.library._destroy)
        self.library.define(gemm_lineage.SCHEMA.split('::', 1)[1])
        def gemm(x, weight, bias=None):
            self.calls.append('gemm')
            self.last_gemm = torch.nn.functional.linear(x, weight, bias)
            return self.last_gemm
        for key in ('CPU', 'PrivateUse1'):
            self.library.impl('unquantized_gemm', gemm, key)
        self.npu_library = torch.library.Library('npu', 'FRAGMENT')
        self.stack.callback(self.npu_library._destroy)
        for schema in (consumers.RMS_SCHEMA, consumers.SWIGLU_SCHEMA,
                'npu::output39_unknown_mutating(Tensor(a!) input) -> Tensor'):
            self.npu_library.define(schema.split('::', 1)[1])
        def rms(x, gamma, epsilon=1e-6):
            self.calls.append('rms')
            rstd = torch.rsqrt(x.float().square().mean(-1, keepdim=True) + epsilon)
            result = ((x.float() * rstd * gamma.float()).to(x.dtype), rstd)
            self.last_consumer = result
            return result
        def swiglu(x, dim=-1):
            self.calls.append('swiglu')
            left, right = x.chunk(2, dim)
            self.last_consumer = (torch.nn.functional.silu(left) * right).contiguous()
            return self.last_consumer
        def mutate(x):
            self.calls.append('unknown_mutating')
            x.add_(1)
            return x.clone()
        for name, function in (('npu_rms_norm', rms), ('npu_swiglu', swiglu),
                               ('output39_unknown_mutating', mutate)):
            for key in ('CPU', 'PrivateUse1'):
                self.npu_library.impl(name, function, key)
        sources = {
            gemm_lineage.PATCH_MODULE: ('import torch\n'
                'def unquantized_gemm(x, weight, bias=None):\n'
                '    return torch.nn.functional.linear(x, weight, bias)\n'
                'def default_unquantized_gemm(layer, x, weight, bias=None):\n'
                '    return torch.ops.vllm.unquantized_gemm(x, weight, bias)\n'),
            gemm_lineage.UTILS_MODULE: '# CPU-only utils identity fixture\n',
            gemm_lineage.REGISTER_MODULE: 'import torch\ndef direct_register_custom_op():\n    pass\n',
            consumers.META_MODULE: '# CPU-only reviewed meta-source fixture\n',
        }
        self.modules = {}
        for index, (name, source) in enumerate(sources.items()):
            file = self.folder / (str(index) + '.py')
            file.write_bytes(source.encode())
            module = ModuleType(name)
            module.__file__ = str(file)
            exec(compile(file.read_bytes(), str(file), 'exec'), vars(module))
            self.modules[name] = module
        actual_patch = self.modules[gemm_lineage.PATCH_MODULE]
        self.modules[gemm_lineage.UTILS_MODULE].default_unquantized_gemm = actual_patch.default_unquantized_gemm
        register = self.modules[gemm_lineage.REGISTER_MODULE]
        register.vllm_lib = self.library
        actual_patch.direct_register_custom_op = register.direct_register_custom_op
        self.npu = SimpleNamespace(__version__=consumers.TORCH_NPU_VERSION, get_npu_format=lambda tensor: 2)
        self.stack.enter_context(patch.dict(sys.modules, {**self.modules, 'torch_npu': self.npu}))
        for module, attribute, source_name in (
                (gemm_lineage, 'PATCH_SHA256', gemm_lineage.PATCH_MODULE),
                (gemm_lineage, 'UTILS_SHA256', gemm_lineage.UTILS_MODULE),
                (gemm_lineage, 'REGISTER_SHA256', gemm_lineage.REGISTER_MODULE),
                (consumers, 'META_SHA256', consumers.META_MODULE)):
            digest = hashlib.sha256(Path(self.modules[source_name].__file__).read_bytes()).hexdigest()
            self.stack.enter_context(patch.object(module, attribute, digest))
        self.stack.enter_context(patch.object(consumers, 'TORCH_VERSION', str(torch.__version__)))
        registrations = {}
        for operation in consumers.NPU_VALUE_CONSUMERS:
            dump = torch._C._dispatch_dump('npu::' + operation.split('.')[1])
            registrations[operation] = next(line for line in dump.splitlines() if line.startswith('PrivateUse1:'))
        self.stack.enter_context(patch.dict(consumers.PRIVATE_REGISTRATION, registrations))
        return self

    def __exit__(self, *error):
        return self.stack.__exit__(*error)


class CustomProjection(torch.nn.Module):
    def __init__(self, consumer='rms', *, rows_slice=False, mutate=False, unowned=False):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.full((4, 4), .25, dtype=torch.bfloat16), requires_grad=False)
        self.gamma = torch.nn.Parameter(torch.ones(4, dtype=torch.bfloat16), requires_grad=False)
        self.consumer, self.rows_slice, self.mutate, self.unowned = consumer, rows_slice, mutate, unowned

    def forward(self, x):
        weight = self.weight.clone() if self.unowned else self.weight
        out = torch.ops.vllm.unquantized_gemm.default(x, weight)
        if self.mutate:
            torch.ops.npu.output39_unknown_mutating.default(out)
        if self.rows_slice:
            out = out[1:]
        if self.consumer == 'rms':
            return torch.ops.npu.npu_rms_norm.default(out, self.gamma)[0]
        if self.consumer == 'swiglu':
            return torch.ops.npu.npu_swiglu.default(out)
        return torch.ops.npu.output39_unknown_mutating.default(out)
