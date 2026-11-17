"""Exact-image stock/custom API coexistence for the experimental TND Worker.

The vLLM adapter opens the bare name libcust_opapi.so for AddRmsNormBias.
Torch NPU can discover the private FIA package by its absolute vendor path.
Both libraries have the same SONAME, so load the unchanged stock library first
and keep the private directory out of the global library search path. Neither
operator is wrapped or replaced. Real FIA launch evidence remains mandatory.
"""
import ctypes
import hashlib
import inspect
import os
from pathlib import Path
import sys

from .serving_context import require


STOCK_ROOT = Path('/vllm-workspace/vllm-ascend/vllm_ascend')
STOCK_VENDOR = STOCK_ROOT/'_cann_ops_custom/vendors/vllm-ascend'
STOCK_FILES = {
    'platform.py': '06cff757eb3a95037f75f46966f85a4428f2688c3226f0950c184234e05e6be2',
    'vllm_ascend_C.cpython-311-aarch64-linux-gnu.so':
        'e63806486e0311c2df8ce9b120ae98f55e8c8eb91ccaf204ef0cf7d6b469ed67',
    '_cann_ops_custom/vendors/vllm-ascend/op_api/lib/libcust_opapi.so':
        'de1633c2f1f41e505277697ee82f3ffba940f730bed9986c5aeff4c8514b713f',
}
LOADER_HEADER = STOCK_ROOT.parent/'csrc/aclnn_torch_adapter/op_api_common.h'
LOADER_HEADER_SHA = '142023f0f0e682b3b79aeb987044ecbd9b7db0b63b20de967447ba6f9e3d6b16'
PRIVATE_LIB = 'op_api/lib/libcust_opapi.so'
NORM_SYMBOLS = ('aclnnAddRmsNormBias', 'aclnnAddRmsNormBiasGetWorkspaceSize')
FIA_SYMBOLS = ('aclnnFusedInferAttentionScore', 'aclnnFusedInferAttentionScoreGetWorkspaceSize',
               'aclnnFusedInferAttentionScoreV3', 'aclnnFusedInferAttentionScoreV3GetWorkspaceSize')
_ACTIVE = None


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def vendor_order(vendor):
    return [str(vendor), str(STOCK_VENDOR)]


def validate_environment(vendor, *, platform_ready):
    private = Path(vendor)
    expected = vendor_order(vendor) if platform_ready else [str(vendor)]
    require(os.environ.get('ASCEND_CUSTOM_OPP_PATH', '').split(':') == expected,
            'TND33_EXACT_STOCK_PRIVATE_VENDOR_ORDER_REQUIRED')
    for item in os.environ.get('LD_LIBRARY_PATH', '').split(':'):
        if item:
            require(not Path(item).resolve().is_relative_to(private.resolve()),
                    'TND33_PRIVATE_LIBRARY_MUST_NOT_SHADOW_STOCK_SONAME')


class DlInfo(ctypes.Structure):
    _fields_ = [('dli_fname',ctypes.c_char_p), ('dli_fbase',ctypes.c_void_p),
                ('dli_sname',ctypes.c_char_p), ('dli_saddr',ctypes.c_void_p)]


def symbol_owner(library, symbol):
    address = ctypes.cast(getattr(library, symbol), ctypes.c_void_p).value
    require(address is not None, 'TND33_REQUIRED_API_SYMBOL_MISSING:'+symbol)
    loader = ctypes.CDLL('libdl.so.2')
    loader.dladdr.argtypes = [ctypes.c_void_p,ctypes.POINTER(DlInfo)]
    loader.dladdr.restype = ctypes.c_int
    info = DlInfo()
    require(loader.dladdr(address,ctypes.byref(info)) != 0 and info.dli_fname,
            'TND33_ACTUAL_SYMBOL_LIBRARY_UNOBSERVED:'+symbol)
    path = Path(os.fsdecode(info.dli_fname)).resolve(strict=True)
    return dict(symbol=symbol, address=address, path=str(path), sha256=sha(path))


class LibraryCoexistence:
    def __init__(self, bundle):
        self.bundle, self.pid = bundle, os.getpid()
        self.private = Path(bundle['vendor'])/PRIVATE_LIB
        self.stock = STOCK_VENDOR/PRIVATE_LIB
        self.handles, self.before, self.after = {}, None, None
        self.platform_ready = False

    def check_files(self):
        for relative, digest in STOCK_FILES.items():
            path = STOCK_ROOT/relative
            require(path.is_file() and sha(path) == digest, 'TND33_STOCK_IMAGE_LIBRARY_OR_SOURCE_CHANGED:'+relative)
        require(LOADER_HEADER.is_file() and sha(LOADER_HEADER) == LOADER_HEADER_SHA,
                'TND33_STOCK_RESOLVER_SOURCE_CHANGED')
        require(self.private.is_file() and sha(self.private) == self.bundle['artifacts'][PRIVATE_LIB]['sha256'],
                'TND33_PRIVATE_API_LIBRARY_CHANGED')

    def load_before_framework(self):
        require(not any(name in sys.modules for name in ('torch','torch_npu','vllm')),
                'TND33_STOCK_SONAME_MUST_BIND_BEFORE_FRAMEWORK')
        require(not self.handles, 'TND33_LIBRARY_BOOTSTRAP_ALREADY_USED')
        validate_environment(self.bundle['vendor'],platform_ready=False)
        self.check_files()
        mode = os.RTLD_LOCAL | os.RTLD_NOW
        self.handles['stock'] = ctypes.CDLL(str(self.stock),mode=mode)
        self.handles['bare'] = ctypes.CDLL('libcust_opapi.so',mode=mode)
        self.before = self._norm_binding()
        # Absolute paths permit both distinct ELFs with the same SONAME. Keep
        # strong references for this dedicated process; never unload a live API.
        self.handles['private'] = ctypes.CDLL(str(self.private),mode=mode)
        self.after = self.inspect_bindings()
        return self.report()

    def _norm_binding(self):
        rows = [symbol_owner(self.handles['bare'],symbol) for symbol in NORM_SYMBOLS]
        require(all(row['path'] == str(self.stock.resolve())
                    and row['sha256'] == STOCK_FILES[self.stock.relative_to(STOCK_ROOT).as_posix()]
                    for row in rows), 'TND33_STOCK_NORM_SYMBOL_SHADOWED')
        return rows

    def inspect_bindings(self):
        require(os.getpid() == self.pid, 'TND33_LIBRARY_OWNER_PROCESS_CHANGED')
        self.check_files()
        bare = ctypes.CDLL('libcust_opapi.so',mode=os.RTLD_LOCAL | os.RTLD_NOW)
        require(bare._handle == self.handles['bare']._handle,
                'TND33_BARE_CUSTOM_LIBRARY_BINDING_CHANGED')
        norm = self._norm_binding()
        fia = [symbol_owner(self.handles['private'],symbol) for symbol in FIA_SYMBOLS]
        require(all(row['path'] == str(self.private.resolve())
                    and row['sha256'] == self.bundle['artifacts'][PRIVATE_LIB]['sha256'] for row in fia),
                'TND33_PRIVATE_FIA_SYMBOL_BOUND_TO_STOCK')
        return dict(stock_norm=norm,private_fia=fia, distinct_library_handles=
                    self.handles['stock']._handle != self.handles['private']._handle)

    def prepare_platform(self, platform):
        require(not self.platform_ready, 'TND33_PLATFORM_VENDOR_ORDER_ALREADY_BOUND')
        method = platform.import_kernels
        require(Path(inspect.getfile(method)).resolve() == (STOCK_ROOT/'platform.py').resolve(),
                'TND33_OFFICIAL_PLATFORM_INITIALIZER_REQUIRED')
        self.check_files()
        # The unchanged official initializer prepends its own vendor once. Run
        # it normally before any model API, then give explicit FIA package
        # lookup priority to our absolute path. Bare-name Norm remains stock.
        method()
        actual = os.environ.get('ASCEND_CUSTOM_OPP_PATH','').split(':')
        require(actual == [str(STOCK_VENDOR),self.bundle['vendor']],
                'TND33_UNEXPECTED_PLATFORM_CUSTOM_VENDOR_LIST')
        os.environ['ASCEND_CUSTOM_OPP_PATH'] = ':'.join(vendor_order(self.bundle['vendor']))
        self.platform_ready = True
        return self.verify()

    def verify(self):
        require(self.platform_ready, 'TND33_PLATFORM_VENDOR_ORDER_NOT_BOUND')
        validate_environment(self.bundle['vendor'],platform_ready=True)
        self.after = self.inspect_bindings()
        require(self.after['distinct_library_handles'], 'TND33_STOCK_PRIVATE_HANDLES_ALIASED')
        return self.report()

    def report(self):
        return dict(schema='demoncat.paged33.api_coexistence.v1',pid=self.pid,
                    stock_files=STOCK_FILES, stock_loader_source_sha256=LOADER_HEADER_SHA,
                    vendor_order=os.environ.get('ASCEND_CUSTOM_OPP_PATH','').split(':'),
                    stock_binding_before_private=self.before, bindings=self.after,
                    platform_ready=self.platform_ready, private_global_library_path_added=False,
                    operator_function_invocations=0, model_or_norm_replaced=False,
                    actual_fia_launch_proven=False, qualification='SYMBOL_BINDING_ONLY')


def bootstrap(bundle):
    global _ACTIVE
    require(_ACTIVE is None, 'TND33_EXISTING_LIBRARY_BOOTSTRAP_OWNER')
    active = LibraryCoexistence(bundle)
    active.load_before_framework()
    _ACTIVE = active
    return active


def verify_runtime(bundle):
    require(_ACTIVE is not None and _ACTIVE.bundle['vendor'] == bundle['vendor']
            and _ACTIVE.bundle['artifacts'][PRIVATE_LIB]['sha256'] == bundle['artifacts'][PRIVATE_LIB]['sha256'],
            'TND33_PREIMPORT_LIBRARY_BOOTSTRAP_MISSING_OR_DIFFERENT')
    return _ACTIVE.verify()
