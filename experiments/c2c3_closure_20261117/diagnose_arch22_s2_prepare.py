"""CPU-only replay of a preserved S2 prepare's normal-artifact inputs.

Run the candidate loader against the OLD result's source/mounts/execution
contract. This diagnoses argument translation; it grants no candidate-package
NPU qualification and neither constructs a Native adapter nor loads a Worker.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import traceback


def sha(path):
    value=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(4*1024**2),b''):value.update(chunk)
    return value.hexdigest()


def diagnose(args):
    # This module is deliberately lazy about live LogicalPagedInput imports.
    from demoncat.runtime.vllm_ascend.worker_extension import paged33_s2_evidence as loader
    result=dict(schema='demoncat.paged33.prepare_cpu_diagnosis.v1',status='CPU_REPLAY_REFUSED',
        original_result_sha256=args.failed_result_sha256,candidate_loader_sha256=sha(loader.__file__),
        actual_npu_calls=0,original_results_modified=False,fault_authorized=False,
        qualification_granted=False,C1=False,C2=False,C3=False,E5='NOT_ASSESSED')
    try:
        if sha(args.failed_result)!=args.failed_result_sha256:raise ValueError('ORIGINAL_RESULT_SHA_DIFFERS')
        original=json.loads(args.failed_result.read_bytes())
        inputs=original['normal_inputs'];folder=Path(inputs['source'])/'experiments/c2c3_closure_20261117'
        sys.path.insert(0,str(folder))
        common=loader._module('arch22_normal_bridge_evidence',folder/'arch22_normal_bridge_evidence.py')
        original=common.read(args.failed_result)
        inputs=original['normal_inputs']
        frozen=common.frozen_source(inputs['source'],inputs['source_manifest_sha256'],build_variant=loader.VARIANT)
        reader=loader._module('review_arch22_normal_bridge',folder/'review_arch22_normal_bridge.py')
        common.need(sha(reader.__file__)==frozen['pins'][common.EXPERIMENT+'review_arch22_normal_bridge.py'],
                    'FROZEN_READER_SOURCE_DIFFERS')
        result['cpu_byte_reader']=reader.initialize_cpu_byte_reader()
        sys.addaudithook(common.prohibit_runtime)
        normal=loader.load_verified_normal_evidence(**inputs,current_execution=original['execution_contract'])
        data=normal.data()
        common.need(sha(args.failed_result)==args.failed_result_sha256,'ORIGINAL_RESULT_CHANGED_DURING_REPLAY')
        result.update(status='CPU_OLD_ARTIFACT_NORMAL_REPLAY_CONFIRMED_NO_NEW_PACKAGE_AUTHORITY',
            normal_evidence_sha256=normal.fingerprint,source_manifest_sha256=data['source_manifest_sha256'],
            published_report_sha256=data['published_report_sha256'],replayed_report_sha256=data['replayed_report_sha256'],
            observer_receipt=str(inputs['observer']),observer_build_root=str(loader.normal_observer_root(
                inputs['observer'],inputs['observer_receipt_sha256'])),
            complete_payload_pairs=435,
            limitation='Candidate loader CPU regression only. New source/package still requires its own normal and F/OFF evidence.')
    except Exception as error:
        result.update(error=repr(error),error_type=type(error).__name__,traceback=traceback.format_exc())
        if getattr(error,'response',None) is not None:result['observed_failure']=error.response
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--failed-result',type=Path,required=True)
    parser.add_argument('--failed-result-sha256',required=True)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args();sys.dont_write_bytecode=True
    if args.out.exists():raise ValueError('FRESH_DIAGNOSTIC_OUTPUT_REQUIRED')
    report=diagnose(args)
    with args.out.open('x',encoding='utf-8') as stream:json.dump(report,stream,allow_nan=False,indent=2);stream.write('\n')
    print(json.dumps(dict(status=report['status'],path=str(args.out),sha256=sha(args.out),fault_authorized=False)))
    return 0 if report['status']=='CPU_OLD_ARTIFACT_NORMAL_REPLAY_CONFIRMED_NO_NEW_PACKAGE_AUTHORITY' else 2


if __name__=='__main__':raise SystemExit(main())
