"""Scoped parent environment for standard spawn Workers, no Worker patch."""
from contextlib import contextmanager
import hashlib,json,os
from pathlib import Path
from demoncat.runtime.vllm_ascend.worker_extension.paged33_loading import STOCK_ROOT,STOCK_VENDOR,STOCK_FILES


def _need(value, why):
    if not value:raise RuntimeError(why)


def _sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@contextmanager
def worker_spawn_vendor_order(root, build_sha256):
    """Initialize the unchanged platform, then inherit private-first to spawn.

    torch_npu captures its vendor search list during library initialization.
    Stock libcust has a libopapi dependency, so stock-first can resolve a stock
    DynamicQuant even when that API is absent from stock's direct exports.
    Actual per-rank Native census remains the decisive run-time test.
    """
    from vllm.platforms import current_platform
    root=Path(root).resolve(strict=True);manifest=root/'build.json'
    _need(_sha(manifest)==build_sha256,'SPAWN_QUANT_BUILD_CHANGED')
    build=json.loads(manifest.read_bytes());private=root/'runtime/vendor'
    library=root/build['library']
    _need(library.resolve()==(private/'op_api/lib/libcust_opapi.so').resolve(),'SPAWN_QUANT_LIBRARY_PATH_CHANGED')
    _need(_sha(STOCK_ROOT/'platform.py')==STOCK_FILES['platform.py'],'SPAWN_PLATFORM_SOURCE_CHANGED')
    stock_library=STOCK_VENDOR/'op_api/lib/libcust_opapi.so'
    _need(_sha(stock_library)==STOCK_FILES[stock_library.relative_to(STOCK_ROOT).as_posix()],'SPAWN_STOCK_LIBRARY_CHANGED')
    # Call the normal idempotent framework initialization; never replace it.
    current_platform.import_kernels()
    before=os.environ.get('ASCEND_CUSTOM_OPP_PATH')
    allowed={str(private),str(STOCK_VENDOR)}
    _need(before is not None and set(before.split(os.pathsep))==allowed,'SPAWN_UNKNOWN_VENDOR_PATH')
    during=os.pathsep.join([str(private),str(STOCK_VENDOR)])
    record=dict(before=before,during=during,restored=False,ordinary_platform_initializer=True,
                worker_class_patched=False,mechanism_enabled=False)
    os.environ['ASCEND_CUSTOM_OPP_PATH']=during
    try:yield record
    finally:
        current=os.environ.get('ASCEND_CUSTOM_OPP_PATH')
        # Do not overwrite an unexpected concurrent writer.
        _need(current==during,'SPAWN_VENDOR_ENV_CHANGED_CONCURRENTLY')
        os.environ['ASCEND_CUSTOM_OPP_PATH']=before;record['restored']=True
