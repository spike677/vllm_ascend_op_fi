"""Device-free SDK compile/tests only; root supplies an isolated build container.

No Docker, SSH, torch import, aclInit, operator call, or DSO load occurs here.
The CPU executable uses the production journal with SDK-typed function doubles.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.worker_core_v11_20261118 import build_quant_container_trace as base

HEADERS = {
    'include/aclnnop/aclnn_fused_infer_attention_score_v3.h': 'bf73aea4146a9b12f91eda466927c23c5d43e56f78aa5039bd9644732f521550',
    'include/aclnn/acl_meta.h': 'be93460abfc7aa14113d7b9a3536fe2dd45d5148a357f55e15847655c9d5ffed',
    'include/acl/acl_base.h': '9346e5050048b5854bd78c4a2147761b4e0ef185c8182266d9a7ba21cabf7f57',
    'include/acl/acl_base_rt.h': '83e189e4bc63ad3e836b6c607f30152704b0c274e475e3b9d9be1689e91de07e',
    'include/acl/acl_base_mdl.h': '27fcdabe75c63155d13167b8f7ba63adbde471cccef60bec2d6c012191746515',
    'include/acl/error_codes/rt_error_codes.h': '77cce791d081ddcf1ccd6ebcc2bb399189028ef4b73f1f3b84c87ab9154b1ebf',
    'include/acl/error_codes/ge_error_codes.h': '1df2641ce314c675cb4f99d5440035711b1f09385f468885b5bfe0cead617309',
    'include/acl/acl_rt.h': '8cce153ef9eb5ed49b0f5bfb5608de706e8d466c836ad2eb50745ec2e8629bc9'}
CASES = ('inactive', 'normal', 'getter_throw', 'getter_status', 'size_mismatch', 'wrong_owner', 'wrong_caller',
    'wrong_scope', 'unknown_member', 'reused_container', 'failed_prepare', 'failed_prepare_null', 'failed_prepare_untouched', 'prepare_throw',
    'executor_reused', 'active_guard', 'foreign_thread', 'pair_budget', 'call_budget', 'no_execute',
    'execute_error', 'execute_retry', 'execute_throw', 'workspace_short')
NATIVE = 'fault_injection/native_adapters/'
SOURCES = base.SOURCES + tuple(NATIVE + 'fia_input_journal' + ext for ext in ('.h', '.cpp', '_test.cpp')) + (
    'fault_injection/fia_input_journal.py',
    'fault_injection/quant_container_native.py',
    'fault_injection/metadata_arrays.py',
    'experiments/worker_core_v11_20261118/fia_input_journal_build.py',
    'experiments/worker_core_v11_20261118/fia_input_journal_review.py')
PAIR = ('aclnnFusedInferAttentionScoreV3GetWorkspaceSize', 'aclnnFusedInferAttentionScoreV3')


def need(ok, message):
    if not ok: raise RuntimeError(message)


def run(argv, log):
    with log.open('ab') as stream:
        stream.write((json.dumps([str(x) for x in argv]) + '\n').encode()); stream.flush()
        subprocess.run([str(x) for x in argv], check=True, stdout=stream, stderr=subprocess.STDOUT)


def build(args):
    from fault39.tools.version_profile import image
    source = args.source.resolve(strict=True); sdk = args.sdk.resolve(strict=True); out = args.out.resolve()
    need(not out.exists(), 'FRESH_BUILD_DIRECTORY_REQUIRED'); out.mkdir(parents=True)
    receipt = dict(schema='demoncat.fia_input_journal.cpu_build.v1', status='INCOMPLETE',
        actual_npu_calls=0, operator_calls=0, loaded_original_function_owner_proven=False,
        qualification_granted=False, slice_pointer_lineage_proven=False, sdk=str(sdk), cases=[])
    log = out/'build.log'
    try:
        need(sdk == base.SDK.resolve(strict=True), 'ONLY_REVIEWED_851_SDK_CANDIDATE')
        need(all(base.sha(sdk/name) == digest for name, digest in HEADERS.items()), 'EXACT_SDK_HEADER_BYTES_REQUIRED')
        receipt['sdk_headers'] = HEADERS
        paths = dict(nn=sdk/'lib64/libnnopbase.so', opapi=sdk/'lib64/libopapi.so', device_runtime=sdk/'lib64/libascendcl.so',
            observer=args.observer.resolve(strict=True),
            custom=base.PACKAGE/'_cann_ops_custom/vendors/vllm-ascend/op_api/lib/libcust_opapi.so',
            torch_npu=Path('/usr/local/python3.11.14/lib/python3.11/site-packages/torch_npu/lib/libtorch_npu.so'),
            custom_extension=base.PACKAGE/'vllm_ascend_C.cpython-311-aarch64-linux-gnu.so')
        need(base.sha(paths['observer']) == args.observer_sha256, 'EXACT_OBSERVER_BYTES_REQUIRED')
        need(all(base.sha(paths[key]) == digest for key, digest in base.PINS.items()), 'UNCHANGED_CONTAINER_OWNER_PROFILE_REQUIRED')
        need(base.sha(paths['opapi']) == 'b3890625d3e0cc11ef32460083698ecaa8f702b5edd0169ea617e2e6bb76270c'
            and base.sha(paths['device_runtime']) == 'bdfa66949fc798ff21c0124a025b4227e119678cc819febed9fe1c39f2a87306',
            'REVIEWED_ORIGINAL_FIA_AND_DEVICE_IMAGES_REQUIRED')
        originals = {key: image(path) for key, path in paths.items()}
        pins = {name: base.sha(source/name) for name in dict.fromkeys(SOURCES)}
        profile = dict(schema='demoncat.quant.container_profile.v1', candidate='fia_input_journal',
            image=base.IMAGE, originals=originals, sdk_headers=HEADERS, sources=pins,
            functions=list(base.FUNCTIONS + PAIR), actual_npu_calls=0, slice_pointer_lineage_proven=False,
            scope='FIA_GWS_ARGUMENTS_CONTAINER_LIFETIME_EXECUTOR_NOT_INTERNAL_PREPARED_INPUTS')
        ident = hashlib.sha256(json.dumps(profile, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
        profile['profile_sha256'] = ident; base.write_json(out/'profile.json', profile)
        qimages = {key: originals[key] for key in ('nn', 'torch_npu', 'custom_extension')}
        qimages['custom_real'] = originals['custom']
        def header(namespace, aliases, images):
            return ('namespace ' + namespace + ' {\nstatic constexpr const char* id="' + ident + '";\n'
                + ''.join('static constexpr const char* '+key+'="'+str(paths[value].resolve())+'";\n' for key,value in aliases.items())
                + ''.join(base.declaration(key, value) for key,value in images.items()) + '}\n')
        (out/'quant_container_profile.h').write_text(header('qct_profile', {'nn_path':'nn','custom_path':'custom'}, qimages))
        (out/'fia_input_journal_profile.h').write_text(header('fij_profile',
            {'opapi_path':'opapi','device_path':'device_runtime','observer_path':'observer'},
            {key: originals[key] for key in ('opapi','device_runtime','observer')}))
        # Formatting warning suppression is limited to these compact reviewed TUs;
        # all other warnings are errors. No compile-time test bypass enters DSO.
        flags = base.flags(out) + ['-Wno-misleading-indentation']
        native = source/NATIVE; library = out/'libfia_input_journal.so'
        run(['g++', *flags, '-shared','-fPIC', native/'fia_input_journal.cpp', '-ldl','-lpthread','-Wl,-z,defs','-o',library], log)
        for key in ('nn','opapi','device_runtime','observer','custom'):
            run(['patchelf','--add-needed',paths[key].resolve(),library], log)
        executable = out/'fia_input_journal_cpu_contract'
        run(['g++', *flags, native/'fia_input_journal_test.cpp','-ldl','-lpthread','-o',executable], log)
        receipt.update(profile_sha256=ident, source_sha256=pins, executable_sha256=base.sha(executable))
        for case in CASES:
            row = dict(case=case,status='STARTED'); receipt['cases'].append(row);base.write_json(out/'receipt.json',receipt)
            result = subprocess.run([str(executable),case], capture_output=True, timeout=60)
            case_log = out/('fia_input_journal_'+case+'.log');case_log.write_bytes(result.stdout+result.stderr)
            row.update(returncode=result.returncode,log_sha256=base.sha(case_log))
            need(result.returncode == 0 and result.stdout == ('PASS_CPU_FIA_INPUT_JOURNAL_NOT_NPU '+case+'\n').encode()
                and not result.stderr,'CPU_CONTRACT_FAILED:'+case);row['status']='PASS'
        symbols = subprocess.run(['nm','-D','--defined-only',str(library)],check=True,capture_output=True,text=True).stdout
        need(set(re.findall(r'\bT (aclnn\w+)$',symbols,re.M)) == set(PAIR),'UNREQUESTED_OPERATOR_EXPORT')
        (out/'exports.txt').write_text(symbols)
        dynamic = subprocess.run(['readelf','-d',str(library)],check=True,capture_output=True,text=True).stdout
        need(all('['+str(paths[key].resolve())+']' in dynamic for key in ('nn','opapi','device_runtime','observer','custom')), 'ABSOLUTE_STARTUP_DEPENDENCIES_REQUIRED')
        (out/'dynamic.txt').write_text(dynamic)
        need(all(base.sha(path) == originals[key]['file_sha256'] for key,path in paths.items()),'ORIGINAL_IMAGES_CHANGED')
        receipt['status']='SDK_TYPED_CPU_CONTRACTS_COMPLETE_NOT_NPU'
        base.write_json(out/'receipt.json',receipt)
        files = {p.name:base.sha(p) for p in out.iterdir() if p.is_file() and p.name!='build.json'}
        base.write_json(out/'build.json',dict(schema='demoncat.quant.container_deployment.v1',candidate='fia_input_journal',
            actual_npu_calls=0,qualified=False,profile_sha256=ident,library=library.name,files=files,
            status=receipt['status'],runtime_activation_default=False,public_provider_registered=False))
        return receipt
    except BaseException as error:
        receipt.update(status='FAILED_CPU_BUILD_OR_CONTRACT',error=repr(error));raise
    finally:
        base.write_json(out/'receipt.json',receipt)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('source','sdk','observer','out'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--observer-sha256',required=True);args=p.parse_args();build(args)


if __name__=='__main__':main()
