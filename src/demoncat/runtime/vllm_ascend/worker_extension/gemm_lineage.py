"""Pinned vLLM Ascend custom-GEMM to native MatMul input binding.

This recognizes the actual registered public operator, its reviewed Python
lowering, and the owning module's weight. Native v2 still verifies both live
storage addresses, offsets, strides and registration epoch at execution.
"""
import ast
import hashlib
import inspect
from pathlib import Path
import sys
from types import CodeType

from .serving_context import require
from fault_injection.cpu_build_identity import current_cpu_identity, require_cpu_values


PATCH_MODULE = 'vllm_ascend.patch.worker.patch_unquantized_gemm'
UTILS_MODULE = 'vllm.model_executor.layers.utils'
REGISTER_MODULE = 'vllm.utils.torch_utils'
PATCH_SHA256 = '0317441758008fa74133b2ee7d3e3b2588d632d6522acfb6a8c98ee21554e5bf'
UTILS_SHA256 = '1d908a5cf4ff519fdd987b7a8798070f57510d9031f8f9c2e013149cdb060cad'
REGISTER_SHA256 = 'b8c0a763d52c0eabb7b34c14b648b6f5e43f67d806f2819755b0ea23f4250221'
SCHEMA = 'vllm::unquantized_gemm(Tensor x, Tensor weight, Tensor? bias=None) -> Tensor'
DISPATCH_TORCH_VERSION = '2.9.0+cpu'
DISPATCH_GUARD = 'TORCH_2_9_SINGLE_PRIVATEUSE1_NO_SHADOW_SOURCE_ORIGIN_V1'


def current_registration(register, raw, path, library, dispatch_dump):
    """Bind current dispatcher origin, including shadowed kernel registrations.

    Library._op_impls is an ownership ledger, not a current-kernel query. The
    pinned dispatcher reports overridden kernels as ``PrivateUse1 (inactive)``;
    no active/hidden registration other than the reviewed original is allowed.
    """
    import torch
    profile = current_cpu_identity()
    require(str(torch.__version__) == (profile.torch_version if profile else DISPATCH_TORCH_VERSION),
            'UNREVIEWED_PUBLIC_GEMM_DISPATCH_BUILD')
    assignments = [node.value for node in ast.parse(raw).body if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == 'vllm_lib' for target in node.targets)]
    require(len(assignments) == 1 and isinstance(assignments[0], ast.Call),
            'PUBLIC_GEMM_LIBRARY_CONSTRUCTION_SOURCE_UNPROVEN')
    call = assignments[0]
    require(len(call.args) == 2 and all(isinstance(arg, ast.Constant) for arg in call.args)
            and [arg.value for arg in call.args] == ['vllm', 'FRAGMENT'] and not call.keywords
            and library.kind == 'FRAGMENT' and library.dispatch_key == '' and library.m is not None,
            'PUBLIC_GEMM_LIBRARY_CONSTRUCTION_SOURCE_UNPROVEN')
    lines = dispatch_dump.splitlines()
    private = [line for line in lines if line.startswith('PrivateUse1')]
    require(len(private) == 1 and private[0].startswith('PrivateUse1: registered at ')
            and private[0].endswith(' :: (none) [ boxed ]'),
            'PUBLIC_GEMM_CURRENT_REGISTRATION_SHADOWED_OR_CHANGED')
    definitions = [line for line in lines if line.startswith('debug: registered at ')]
    require(len(definitions) == 1 and 'alias analysis kind: FROM_SCHEMA' in lines,
            'PUBLIC_GEMM_DEFINITION_ORIGIN_UNPROVEN')
    origin = private[0].removeprefix('PrivateUse1: registered at ').removesuffix(' :: (none) [ boxed ]')
    require(definitions[0] == 'debug: registered at ' + origin,
            'PUBLIC_GEMM_CURRENT_IMPL_NOT_DEFINITION_OWNER')
    filename, separator, lineno = origin.rpartition(':')
    require(separator and lineno.isdecimal() and Path(filename).resolve() == path
            and call.lineno <= int(lineno) <= call.end_lineno,
            'PUBLIC_GEMM_CURRENT_REGISTRATION_SOURCE_ORIGIN_CHANGED')
    return dict(guard_profile=profile.name if profile else DISPATCH_GUARD,
        build_identity=profile.report() if profile else dict(kind='PINNED_PRODUCTION', test_only=False),
        torch_version=str(torch.__version__),
        current_privateuse1_registration=private[0], definition_registration=definitions[0],
        source_module=register.__name__, source_path=str(path),
        source_sha256=hashlib.sha256(raw).hexdigest(), construction_lines=[call.lineno, call.end_lineno],
        actual_registration_line=int(lineno), shadowed_privateuse1_registrations=0)


class ReviewedGemmBinding:
    def __init__(self):
        self.compiled = {}
        self.last_proof = None

    def arguments(self, function, args, kwargs, expected_weight):
        import torch
        from torch._ops import OpOverload
        self.last_proof = None
        require_cpu_values((args, kwargs, expected_weight))
        profile = current_cpu_identity()
        require(isinstance(function, OpOverload) and str(function) == 'vllm.unquantized_gemm.default'
                and str(function._schema) == SCHEMA, 'UNREVIEWED_PUBLIC_GEMM_SCHEMA')
        patch = sys.modules.get(PATCH_MODULE)
        utils = sys.modules.get(UTILS_MODULE)
        register = sys.modules.get(REGISTER_MODULE)
        require(patch is not None and utils is not None and register is not None,
                'REVIEWED_ASCEND_GEMM_MODULES_NOT_LOADED')
        sources, source_bytes, source_paths = {}, {}, {}
        for module, expected in ((patch, PATCH_SHA256), (utils, UTILS_SHA256), (register, REGISTER_SHA256)):
            expected = profile.source_hashes.get(module.__name__) if profile else expected
            path = Path(module.__file__).resolve()
            raw = path.read_bytes()
            require(hashlib.sha256(raw).hexdigest() == expected, 'UNREVIEWED_PUBLIC_GEMM_SOURCE:' + module.__name__)
            sources[module.__name__] = expected
            source_bytes[module.__name__], source_paths[module.__name__] = raw, path
            if module.__name__ not in self.compiled:
                self.compiled[module.__name__] = compile(raw, str(path), 'exec', dont_inherit=True)
        for module, name in ((patch, 'unquantized_gemm'), (patch, 'default_unquantized_gemm'),
                             (register, 'direct_register_custom_op')):
            function_impl = getattr(module, name, None)
            codes = [code for code in self.compiled[module.__name__].co_consts
                     if isinstance(code, CodeType) and code.co_name == name]
            require(len(codes) == 1 and inspect.isfunction(function_impl)
                    and function_impl.__code__ == codes[0]
                    and function_impl.__globals__ is vars(module)
                    and function_impl.__globals__.get('torch') is torch,
                    'LOADED_PUBLIC_GEMM_IMPLEMENTATION_CHANGED:' + name)
        require(utils.default_unquantized_gemm is patch.default_unquantized_gemm,
                'PUBLIC_GEMM_PLATFORM_DISPATCH_CHANGED')
        library = getattr(register, 'vllm_lib', None)
        require(patch.direct_register_custom_op is register.direct_register_custom_op
                and isinstance(library, torch.library.Library) and library.ns == 'vllm'
                and 'vllm::unquantized_gemm' in library._op_defs
                and 'vllm/unquantized_gemm/PrivateUse1' in library._op_impls
                and torch._C._dispatch_has_kernel_for_dispatch_key('vllm::unquantized_gemm', 'PrivateUse1'),
                'PUBLIC_GEMM_REGISTRATION_OWNER_CHANGED')
        dispatch_dump = torch._C._dispatch_dump('vllm::unquantized_gemm')
        require(('schema: ' + SCHEMA) in dispatch_dump.splitlines()
                and len([line for line in dispatch_dump.splitlines() if line.startswith('PrivateUse1: registered at ')]) == 1,
                'PUBLIC_GEMM_DISPATCH_REGISTRATION_UNPROVEN')
        registration = current_registration(register, source_bytes[register.__name__],
            source_paths[register.__name__], library, dispatch_dump)
        require(len(args) <= 3 and not (set(kwargs) - {'x', 'weight', 'bias'}),
                'UNREVIEWED_PUBLIC_GEMM_ARGUMENTS')
        names = ('x', 'weight', 'bias')
        require(not any(name in kwargs for name in names[:len(args)]), 'DUPLICATE_PUBLIC_GEMM_ARGUMENT')
        bound = dict(zip(names, args))
        bound.update(kwargs)
        require('x' in bound and 'weight' in bound, 'MISSING_PUBLIC_GEMM_ARGUMENT')
        weight = bound['weight']
        if bound.get('bias') is not None or not isinstance(weight, torch.Tensor) or weight is not expected_weight:
            return None
        self.last_proof = dict(schema=SCHEMA, sources=sources,
            public_operator='vllm.unquantized_gemm.default',
            lowering='torch.nn.functional.linear(x, weight, None)',
            native_self='x', native_mat2='weight.t()', module_weight_object_identity=id(weight),
            registered_library_object_identity=id(library), registered_dispatch_key='PrivateUse1',
            actual_dispatch_dump=dispatch_dump,
            current_registration=registration,
            source_and_loaded_code_verified=True, owned_weight_verified=True)
        return bound['x'], weight
