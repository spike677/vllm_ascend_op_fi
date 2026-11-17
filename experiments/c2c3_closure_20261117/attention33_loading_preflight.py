"""Resolve actual 8.5.1 stock Norm and private FIA libraries without NPU ops."""
import argparse
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
    parser.add_argument('--child',action='store_true',help=argparse.SUPPRESS)
    args = parser.parse_args()
    from demoncat.runtime.vllm_ascend.worker_extension.paged33_tnd import verify_bundle
    from demoncat.runtime.vllm_ascend.worker_extension.paged33_loading import bootstrap,require
    require(not list(Path('/dev').glob('davinci*')), 'CPU_LOADING_PREFLIGHT_MUST_HAVE_NO_NPU_DEVICES')
    bundle = verify_bundle(args.completion,args.completion_sha256,args.vendor)
    if not args.child:
        require(not args.out.exists() and not os.environ.get('ASCEND_CUSTOM_OPP_PATH')
                and not os.environ.get('LD_PRELOAD'), 'FRESH_LOADING_PREFLIGHT_PROCESS_REQUIRED')
        args.out.mkdir(parents=True)
        env = dict(os.environ,ASCEND_CUSTOM_OPP_PATH=bundle['vendor'],ACLNN_CACHE_LIMIT='0',
                   TASK_QUEUE_ENABLE='0',PYTHONDONTWRITEBYTECODE='1',VLLM_ENABLE_V1_MULTIPROCESSING='0')
        os.execve(sys.executable,[sys.executable,str(Path(__file__).resolve()),*sys.argv[1:],'--child'],env)
    report = dict(schema='demoncat.paged33.loading_preflight.v1',status='STARTED',
                  actual_npu_tensor_operations=0,model_initialized=False,
                  completion_sha256=args.completion_sha256,failures=[])
    try:
        coexistence = bootstrap(bundle)
        report['before_framework'] = coexistence.report()
        from vllm.platforms import current_platform
        report['after_platform'] = coexistence.prepare_platform(current_platform)
        # Load the unchanged extension so that its real dependencies coexist too.
        # This imports/registers operators; no operator is called and no model is built.
        import vllm_ascend.vllm_ascend_C
        report['after_actual_extension_import'] = coexistence.verify()
        report['status'] = 'CPU_STOCK_PRIVATE_API_COEXISTENCE_VERIFIED'
    except BaseException as error:
        report.update(status='CPU_LOADING_PREFLIGHT_FAILED')
        report['failures'].append(dict(error=repr(error),traceback=traceback.format_exc()))
    with (args.out/'result.json').open('x') as stream:
        json.dump(report,stream,indent=2,allow_nan=False);stream.write('\n')
    print(json.dumps({key:report[key] for key in ('status','actual_npu_tensor_operations','failures')}))
    return 0 if report['status'] == 'CPU_STOCK_PRIVATE_API_COEXISTENCE_VERIFIED' else 2


if __name__ == '__main__':
    raise SystemExit(main())
