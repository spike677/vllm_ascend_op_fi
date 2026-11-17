"""Resolve actual 8.5.1 stock Norm and private FIA libraries without NPU ops."""
import argparse
import faulthandler
import json
import os
from pathlib import Path
import sys
import traceback


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--completion',type=Path,required=True)
    parser.add_argument('--completion-sha256',required=True)
    parser.add_argument('--vendor',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--stock-extension-control',action='store_true',
                        help='Same no-device extension import with no private library loaded or vendor environment')
    parser.add_argument('--include-extension-import',action='store_true',
                        help='Optional diagnostic; the unchanged extension also aborts in the stock no-device control')
    parser.add_argument('--child',action='store_true',help=argparse.SUPPRESS)
    args = parser.parse_args()
    faulthandler.enable()
    from demoncat.runtime.vllm_ascend.worker_extension.paged33_tnd import verify_bundle
    from demoncat.runtime.vllm_ascend.worker_extension.paged33_loading import bootstrap,require
    require(not list(Path('/dev').glob('davinci*')), 'CPU_LOADING_PREFLIGHT_MUST_HAVE_NO_NPU_DEVICES')
    bundle = verify_bundle(args.completion,args.completion_sha256,args.vendor)
    if not args.child:
        require(not args.out.exists() and not os.environ.get('ASCEND_CUSTOM_OPP_PATH')
                and not os.environ.get('LD_PRELOAD'), 'FRESH_LOADING_PREFLIGHT_PROCESS_REQUIRED')
        args.out.mkdir(parents=True)
        env = dict(os.environ,ASCEND_CUSTOM_OPP_PATH='' if args.stock_extension_control else bundle['vendor'],ACLNN_CACHE_LIMIT='0',
                   TASK_QUEUE_ENABLE='0',PYTHONDONTWRITEBYTECODE='1',VLLM_ENABLE_V1_MULTIPROCESSING='0')
        os.execve(sys.executable,[sys.executable,str(Path(__file__).resolve()),*sys.argv[1:],'--child'],env)
    report = dict(schema='demoncat.paged33.loading_preflight.v1',status='STARTED',
                  actual_npu_tensor_operations=0,model_initialized=False,
                  completion_sha256=args.completion_sha256,failures=[],
                  stock_extension_control=args.stock_extension_control,
                  private_api_preloaded=False,full_extension_import_verified=False,
                  native_worker_initialization_verified=False,
                  scope='ACTUAL_API_SYMBOL_BINDINGS_AND_OFFICIAL_PLATFORM_ENVIRONMENT_ONLY')
    def checkpoint(stage):
        report['stage']=stage
        (args.out/'result.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
        print(json.dumps(dict(status=report['status'],stage=stage)),flush=True)
    try:
        coexistence = None
        if not args.stock_extension_control:
            checkpoint('BEFORE_LIBRARY_BOOTSTRAP')
            coexistence = bootstrap(bundle)
            report['before_framework'] = coexistence.report()
            report['private_api_preloaded'] = True
        checkpoint('BEFORE_PLATFORM_IMPORT')
        from vllm.platforms import current_platform
        checkpoint('BEFORE_PLATFORM_VENDOR_INITIALIZATION')
        if coexistence is not None:
            report['after_platform'] = coexistence.prepare_platform(current_platform)
        else:
            current_platform.import_kernels()
            report['stock_vendor_environment'] = os.environ.get('ASCEND_CUSTOM_OPP_PATH')
        if args.stock_extension_control or args.include_extension_import:
            checkpoint('BEFORE_ACTUAL_EXTENSION_IMPORT')
            import vllm_ascend.vllm_ascend_C
            report['full_extension_import_verified']=True
            if coexistence is not None:
                report['after_actual_extension_import'] = coexistence.verify()
        else:
            report['remaining_gate']='Device-backed ordinary Worker initialization and post-initialization binding recheck'
            report['stock_and_private_device_free_extension_import']='ABORTED_IN_SEPARATE_PRESERVED_CONTROLS'
        report['status'] = ('CPU_STOCK_EXTENSION_IMPORT_VERIFIED' if args.stock_extension_control else
                            'CPU_API_BINDINGS_AND_PLATFORM_ENV_VERIFIED')
    except BaseException as error:
        report.update(status='CPU_LOADING_PREFLIGHT_FAILED')
        report['failures'].append(dict(error=repr(error),traceback=traceback.format_exc()))
    success=report['status'] in ('CPU_STOCK_EXTENSION_IMPORT_VERIFIED','CPU_API_BINDINGS_AND_PLATFORM_ENV_VERIFIED')
    checkpoint('COMPLETE' if success else 'FAILED')
    print(json.dumps({key:report[key] for key in ('status','actual_npu_tensor_operations','failures')}))
    return 0 if success else 2


if __name__ == '__main__':
    raise SystemExit(main())
