"""Stage a verified continuation package and bind same-generation ABI evidence."""
import argparse
import json
from pathlib import Path
import subprocess
from generation import layout
from resume_support import ROOT,attempt_paths,run_cpu,save,sha,verify_new_source_freeze,verify_original


def abi_binding(paths,generation):
    ev=paths['evidence']
    schema_path=ev/'ifa_abi_schema.json';compile_path=ev/'ifa_abi_compile.json';test_path=ev/'ifa_abi_selftest.json'
    schema=json.loads(schema_path.read_text());compile_record=json.loads(compile_path.read_text());test=json.loads(test_path.read_text())
    if schema['schema']!='fault33.pa.public_serializer_layout.v1' or schema['type']!='IncreFlashAttentionTilingDataV2' or schema['size']!=3000 or len(schema['fields'])!=72:
        raise RuntimeError('ABI schema does not match the actual fixed IFA serializer')
    if test['status']!='PASS_PUBLIC_SERIALIZER_CPU_ROUNDTRIP' or test['size']!=3000 or test['projected_fields']!=72:
        raise RuntimeError('ABI CPU roundtrip gate not satisfied')
    if compile_record['source_sha256']!=sha(ROOT/'source_policy/ifa_abi_decoder.cpp') or compile_record['decoder_sha256']!=sha(ev/'ifa_abi_decoder'):
        raise RuntimeError('ABI compile receipt is not bound to the frozen decoder')
    if f'/source_{generation}/ops-transformer-v9.1.0/' not in compile_record['actual_ifa_compile_entry']['file']:
        raise RuntimeError('ABI compile entry is from another generation')
    occupied=set();names=set()
    for field in schema['fields']:
        if field['name'] in names or field['width']!=4 or field['count']<1:
            raise RuntimeError('Invalid or duplicate ABI projected field')
        names.add(field['name'])
        for position in range(field['offset'],field['offset']+field['width']*field['count']):
            if position<0 or position>=3000 or position in occupied:
                raise RuntimeError('Overlapping or out-of-range ABI field')
            occupied.add(position)
    return {'generation':generation,'schema_file':schema_path.relative_to(ROOT).as_posix(),'schema_sha256':sha(schema_path),
        'schema_bytes':schema_path.stat().st_size,'compile_file':compile_path.relative_to(ROOT).as_posix(),'compile_sha256':sha(compile_path),
        'selftest_file':test_path.relative_to(ROOT).as_posix(),'selftest_sha256':sha(test_path),'raw_size':3000,'projected_fields':72,
        'layout_kind':'public_host_serializer_projection','kernel_argument_raw_observed':False}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--generation',required=True)
    parser.add_argument('--attempt',required=True)
    args=parser.parse_args()
    paths,ev=attempt_paths(args.generation,args.attempt)
    verify_original(paths)
    freeze=json.loads((ev/'continuation_source_freeze.json').read_text())
    verify_new_source_freeze(freeze['files'])
    done=json.loads((ev/'build_completion.json').read_text())
    package=Path(done['package'])
    if done['status']!='PASS_CPU_BUILD_CONTINUATION_PACKAGE' or not done['container_removed'] or sha(package)!=done['package_sha256']:
        raise RuntimeError('Verified continuation package gate not satisfied')
    stage=paths['stage']
    if stage.exists():
        raise RuntimeError('Refuse existing private vendor stage')
    public_abi=abi_binding(paths,args.generation)
    c=layout(Path('/audit'),args.generation)
    script=f'''set -eo pipefail
source /usr/local/Ascend/cann-9.1.0/set_env.sh
bash {c['source']}/build_out/cann-ops-transformer-fault33_pa_linux-aarch64.run --install-path={c['stage']} --quiet
'''
    result=run_cpu(script,ev,f'fault33-pa-cpu-stage-resume-{args.generation}-{args.attempt}-20261105',600)
    if result['returncode'] or not result['container_removed']:
        save(ev/'stage_completion.json',{**result,'status':'FAIL_CPU_PRIVATE_STAGE'})
        raise RuntimeError('Private CPU install failed')
    base=ev/'base_vendor_manifest.json'
    subprocess.run(['python3','-B',str(ROOT/'source_policy/write_vendor_manifest.py'),'--task-root',str(ROOT),
        '--vendor-root',str(stage/'vendors/fault33_pa_transformer'),'--source',str(paths['source']),
        '--source-manifest',str(paths['evidence']/'policy_source_manifest.json'),'--package',str(package),
        '--output',str(base),'--generation',args.generation],check=True)
    manifest=json.loads(base.read_text())
    manifest['public_abi']=public_abi
    manifest['build_continuation']={'generation':args.generation,'attempt':args.attempt,
        'receipt_file':(ev/'build_completion.json').relative_to(ROOT).as_posix(),'receipt_sha256':sha(ev/'build_completion.json'),
        'source_freeze_file':(ev/'continuation_source_freeze.json').relative_to(ROOT).as_posix(),'source_freeze_sha256':sha(ev/'continuation_source_freeze.json'),
        'wheel_preflight_file':(ev/'wheel_preflight.json').relative_to(ROOT).as_posix(),'wheel_preflight_sha256':sha(ev/'wheel_preflight.json')}
    verify_original(paths);verify_new_source_freeze(freeze['files'])
    manifest_path=stage/'vendor_manifest.json'
    save(manifest_path,manifest)
    result.update(status='PASS_CPU_PRIVATE_STAGE',generation=args.generation,attempt=args.attempt,
        manifest=str(manifest_path),manifest_sha256=sha(manifest_path),public_abi=public_abi)
    save(ev/'stage_completion.json',result)
    print(json.dumps(result),flush=True)


if __name__=='__main__':
    main()
