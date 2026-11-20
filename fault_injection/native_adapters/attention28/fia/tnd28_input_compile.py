"""Exact generated-unit compiler launcher for the TND replay candidate only.

The actual original compiler executes once. No operator or device is executed.
All other compilation units pass through unchanged. This is not a runtime shim.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import shlex
import subprocess

VARIANT='tnd_prefill_denominator_replay'
GENERATED='aclnnInner_fused_infer_attention_score.cpp'
ORIGINAL_SHA='d35f8b60d1ad50d97b7cf0d2050be5c23fd0d53eba77a58c053c184d96af4fe2'
GENERATED_HEADER_SHA='fae730eb70329a1f0fd3f982fd130f80998b4d9b1339e75b6c9d021dd667c3ff'
HEADER='tnd28_input_observation.h'
BORROW_HEADER='tnd28_execute_borrow.h'
SDK_HEADERS={
    'include/aclnn/opdev/common_types.h':'4e365e2b0805beb6f97e1cbcd686724aff778110797d17ffdb8724330bbbf620',
    'include/aclnn/opdev/op_executor.h':'c79a9b705fc0e0b795bc66bc5e5c1fee15b0ca79b7ad48db5c6b6e16bc615817',
    'include/aclnn/acl_meta.h':'be93460abfc7aa14113d7b9a3536fe2dd45d5148a357f55e15847655c9d5ffed'}
COMPILE_RECEIPT='tnd28_compiled_input_compile.json'
OPTIONALS=('pseShiftOptional','dequantScale1Optional','quantScale1Optional','dequantScale2Optional',
    'quantScale2Optional','quantOffset2Optional','antiquantScaleOptional','antiquantOffsetOptional',
    'blockTableOptional','queryPaddingSizeOptional','kvPaddingSizeOptional','keyAntiquantScaleOptional',
    'keyAntiquantOffsetOptional','valueAntiquantScaleOptional','valueAntiquantOffsetOptional',
    'keySharedPrefixOptional','valueSharedPrefixOptional','actualSharedPrefixLenOptional','queryRopeOptional',
    'keyRopeOptional','keyRopeAntiquantScaleOptional','dequantScaleQueryOptional','learnableSinkOptional',
    'qStartIdxOptional','kvStartIdxOptional')
SCALARS=('numHeads','preTokens','nextTokens','numKeyValueHeads','sparseMode','innerPrecise','blockSize',
    'antiquantMode','keyAntiquantMode','valueAntiquantMode','queryQuantMode','pseType','outDtype')


def need(ok,reason):
    if not ok:raise ValueError(reason)


def sha(raw):return hashlib.sha256(raw).hexdigest()


def once(text,old,new):
    need(text.count(old)==1,'GENERATED_FIA_ANCHOR_CHANGED:'+old[:80]);return text.replace(old,new,1)


def patch_source(raw):
    need(sha(raw)==ORIGINAL_SHA,'UNREVIEWED_GENERATED_FIA_BYTES')
    text=raw.decode('utf-8')
    prefix,rest=text.split('aclnnStatus aclnnInnerFusedInferAttentionScoreGetWorkspaceSize(',1)
    prepare,rest=rest.split('aclnnStatus aclnnInnerFusedInferAttentionScoreTensorGetWorkspaceSize(',1)
    tensor,execute=rest.split('aclnnStatus aclnnInnerFusedInferAttentionScore(\n',1)
    entry='    tnd28_input::Prepare dcat_input(&aclnnInnerFusedInferAttentionScoreGetWorkspaceSize,\n'+\
      '        query,key,value,attenMaskOptional,actualSeqLengthsOptional,actualSeqLengthsKvOptional,attentionOutOut,softmaxLseOut,\n'+\
      '        {'+','.join(OPTIONALS)+'},\n        {'+','.join(SCALARS)+'},scale,inputLayoutOptional,softmaxLseFlag);\n'
    prepare=once(prepare,'{\n    uint64_t timeStamp =','{\n'+entry+
        '    if (!dcat_input.Admit()) return ACLNN_ERR_PARAM_INVALID;\n    uint64_t timeStamp =')
    prepare=once(prepare,'aclnnContiguousGetWorkspaceSize(inUnContTensors, &inContWorkspaceSize, &aclInExecutor)',
        'dcat_input.Contiguous(aclnnContiguousGetWorkspaceSize, inUnContTensors, &inContWorkspaceSize, &aclInExecutor)')
    prepare=once(prepare,'NnopbaseRunForWorkspace(*executor, workspaceSize)',
        'dcat_input.Workspace(&NnopbaseRunForWorkspace, *executor, workspaceSize)')
    prepare=once(prepare,'    return ACLNN_SUCCESS;\n}',
        '    return dcat_input.Return(ACLNN_SUCCESS, workspaceSize, executor);\n}')
    execute=once(execute,'{\n    uint64_t timeStamp =',
        '{\n    tnd28_input::Execute dcat_input(&aclnnInnerFusedInferAttentionScore, executor, workspace, workspaceSize, stream);\n    uint64_t timeStamp =')
    execute=once(execute,'aclnnContiguous(inWorkspace, inContWorkspaceSize, aclInExecutor, stream)',
        'dcat_input.Contiguous(aclnnContiguous, inWorkspace, inContWorkspaceSize, aclInExecutor, stream)')
    execute=once(execute,'NnopbaseRunWithWorkspace(executor, stream, workspace, workspaceSize)',
        'dcat_input.Run(&NnopbaseRunWithWorkspace, executor, stream, workspace, workspaceSize)')
    execute=once(execute,'    return ACLNN_SUCCESS;\n}',
        '    return dcat_input.Return(ACLNN_SUCCESS);\n}')
    prefix=once(prefix,'#include "aclnnInner_fused_infer_attention_score.h"',
        '#include "aclnnInner_fused_infer_attention_score.h"\n#include "'+HEADER+'"')
    return (prefix+'aclnnStatus aclnnInnerFusedInferAttentionScoreGetWorkspaceSize('+prepare+
        'aclnnStatus aclnnInnerFusedInferAttentionScoreTensorGetWorkspaceSize('+tensor+
        'aclnnStatus aclnnInnerFusedInferAttentionScore(\n'+execute).encode()


def file_record(path,limit):
    path=Path(path).resolve(strict=True);before=path.stat()
    need(path.is_file() and 0<before.st_size<=limit,'COMPILE_INPUT_FILE_BOUND:'+str(path))
    digest=hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda:stream.read(65536),b''):digest.update(chunk)
    after=path.stat()
    need((before.st_dev,before.st_ino,before.st_size,before.st_mtime_ns)==
        (after.st_dev,after.st_ino,after.st_size,after.st_mtime_ns),'COMPILE_INPUT_FILE_CHANGED:'+str(path))
    return dict(path=str(path),bytes=before.st_size,sha256=digest.hexdigest())


def sdk_headers(sdk):
    sdk=Path(sdk).resolve(strict=True)
    need(sdk.name=='cann-8.5.1' and (sdk/'pkg_inc').is_dir(),'EXACT_CANN851_SDK_INCLUDE_ROOT_REQUIRED')
    rows={name:file_record(sdk/name,1024**2) for name in SDK_HEADERS}
    need(all(row['sha256']==SDK_HEADERS[name] and Path(row['path']).is_relative_to(sdk)
        for name,row in rows.items()),'ACTUAL_CANN851_SDK_HEADER_BYTES_CHANGED')
    return rows


def option(argv,name):
    values=[]
    for i,arg in enumerate(argv):
        if arg==name:
            need(i+1<len(argv),'COMPILER_OPTION_VALUE_MISSING:'+name);values.append(argv[i+1])
        elif arg.startswith(name) and len(arg)>len(name):values.append(arg[len(name):])
    need(len(values)==1 and values[0],'ONE_COMPILER_OPTION_REQUIRED:'+name)
    return values[0]


def output_paths(source,argv,*,cwd=None):
    need(argv.count('-MD')==1 and not any(x in argv for x in ('-MMD','-M','-MM','-MP')),
        'ORIGINAL_MD_DEPENDENCY_OUTPUT_REQUIRED')
    root=(Path(source)/'build').resolve(strict=True)
    cwd=Path.cwd() if cwd is None else Path(cwd)
    paths=[(cwd/option(argv,name)).resolve() for name in ('-MF','-o')]
    need(paths[0]!=paths[1] and all(p.is_relative_to(root) and not p.is_symlink() for p in paths),
        'ACTUAL_COMPILE_OUTPUT_OUTSIDE_BUILD')
    return paths


def dependency_records(path,*,cwd,source,sdk,output):
    recorded=file_record(path,2*1024**2);raw=Path(path).read_bytes()
    need(sha(raw)==recorded['sha256'],'ACTUAL_DEPENDENCY_FILE_CHANGED')
    text=raw.decode().replace('\\\r\n',' ').replace('\\\n',' ')
    parts=re.split(r'(?<!\\):\s',text,maxsplit=1)
    need(len(parts)==2 and '$' not in text,'ONE_ACTUAL_DEPENDENCY_RULE_REQUIRED')
    targets=shlex.split(parts[0]);names=shlex.split(parts[1])
    resolve=lambda p:(Path(cwd)/p).resolve(strict=True)
    need(len(targets)==1 and resolve(targets[0])==Path(output).resolve(strict=True)
        and 1<=len(names)<=4096,'ACTUAL_DEPENDENCY_TARGET_OR_COUNT')
    files=[];seen=set();total=0;sdk=Path(sdk).resolve(strict=True)
    for name in names:
        p=resolve(name)
        need('/dependencies/opbase/' not in p.as_posix() and '/include/nnopbase/' not in p.as_posix(),
            'DEPENDENCY_OPBASE_INTERNAL_HEADER_FORBIDDEN:'+str(p))
        if p in seen:continue
        seen.add(p);row=file_record(p,16*1024**2);total+=row['bytes']
        need(total<=256*1024**2,'ACTUAL_DEPENDENCY_TOTAL_BYTES_BOUND')
        files.append(dict(recorded_path=name,**row))
    required=[sdk/name for name in SDK_HEADERS]
    for expected in required:
        matches=[p for p in seen if p.name==expected.name]
        need(matches==[expected.resolve(strict=True)],'ACTUAL_SDK_HEADER_SELECTION_DIFFERS:'+expected.name)
    for suffix in ('/opdev/fast_vector.h','/exe_graph/runtime/shape.h','/base/dlog_pub.h'):
        matches=[p for p in seen if p.as_posix().endswith(suffix)]
        need(len(matches)==1 and matches[0].is_relative_to(sdk),'ACTUAL_SDK_TRANSITIVE_HEADER_REQUIRED:'+suffix)
    need((Path(source)/'build/autogen/inner'/GENERATED).resolve(strict=True) in seen
        and (Path(source)/'build/autogen/inner'/HEADER).resolve(strict=True) in seen
        and (Path(source)/'build/autogen/inner'/BORROW_HEADER).resolve(strict=True) in seen,
        'ACTUAL_GENERATED_OBSERVER_DEPENDENCIES_MISSING')
    return dict(file=recorded,files=files,total_bytes=total)


def prepare_unit(source,evidence,argv,*,variant,sdk):
    need(variant==VARIANT,'INPUT_OBSERVER_EXACT_VARIANT_REQUIRED')
    source=Path(source).resolve(strict=True);evidence=Path(evidence).resolve(strict=True)
    candidates=[Path(x) for x in argv if Path(x).name==GENERATED]
    if not candidates:return None
    need(len(candidates)==1 and '-c' in argv,'ONE_ACTUAL_GENERATED_COMPILATION_REQUIRED')
    need(not any(x.startswith('@') for x in argv),'EXPLICIT_GENERATED_COMPILER_ARGUMENTS_REQUIRED')
    need(not any(name in x for x in argv for name in ('DCAT_TND28_INPUT_CPU_FIXTURE','DCAT_TND28_BORROW_CPU_FIXTURE')),
        'CPU_FIXTURE_NOT_A_PRODUCTION_COMPILE')
    pins=sdk_headers(sdk);output_paths(source,argv)
    effective=[argv[0],'-I',str(Path(sdk).resolve()/'include/aclnn'),'-I',str(Path(sdk).resolve()/'pkg_inc'),*argv[1:]]
    unit=candidates[0];expected=source/'build/autogen/inner'/GENERATED
    need(not unit.is_symlink() and unit.resolve(strict=True)==expected
        and expected.resolve(strict=True)==expected,'GENERATED_UNIT_PATH_CHANGED')
    need(unit.stat().st_size<2*1024**2,'GENERATED_UNIT_BOUND')
    raw=unit.read_bytes();receipt=evidence/'tnd28_compiled_input_observer.json'
    header=Path(__file__).with_name(HEADER).read_bytes();target=unit.with_name(HEADER)
    borrow=Path(__file__).with_name(BORROW_HEADER).read_bytes();borrow_target=unit.with_name(BORROW_HEADER)
    if receipt.exists():
        need(not receipt.is_symlink() and receipt.stat().st_size<65536,'OBSERVER_RECEIPT_PATH_OR_BOUND')
        old=json.loads(receipt.read_bytes())
        need(old['patched_sha256']==sha(raw) and old['header_sha256']==sha(header)
            and not target.is_symlink() and target.read_bytes()==header
            and not borrow_target.is_symlink() and borrow_target.read_bytes()==borrow
            and old['execute_borrow_header_sha256']==sha(borrow)
            and old['launcher_sha256']==sha(Path(__file__).read_bytes()) and old['sdk_headers']==pins
            and old['actual_compiler_argv']==argv and old['effective_compiler_argv']==effective,
            'COMPILED_OBSERVER_REENTRY_BYTES_CHANGED')
        return old
    h=unit.with_suffix('.h')
    need(not h.is_symlink() and sha(h.read_bytes())==GENERATED_HEADER_SHA,'GENERATED_HEADER_BYTES_CHANGED')
    need(not target.exists() and not borrow_target.exists(),'GENERATED_OBSERVER_HEADER_ALREADY_PRESENT')
    changed=patch_source(raw)
    report=dict(schema='demoncat.tnd28.compiled_input_source.v1',variant=variant,
        original_sha256=sha(raw),patched_sha256=sha(changed),header_sha256=sha(header),
        generated_header_sha256=GENERATED_HEADER_SHA,launcher_sha256=sha(Path(__file__).read_bytes()),
        unit=str(unit),header=str(target),actual_compiler_argv=argv,effective_compiler_argv=effective,
        execute_borrow_header=str(borrow_target),execute_borrow_header_sha256=sha(borrow),
        sdk=str(Path(sdk).resolve()),sdk_headers=pins,
        source_hooks=['typed_prepare','actual_contiguous_prepare','actual_workspace',
            'actual_prepare_return','typed_execute','actual_contiguous_execute','actual_fia_execute','actual_execute_return'],
        optional_normal_guard=dict(abi=2,live_callback=True,prepare_before_parent_allocation=True,
            post_copy_before_fia=True,reads_current_device_values=False,permits_fault=False),
        operator_arguments_changed=False,extra_operator_calls=0,observer_only=True,
        runtime_observed=False,current_input_authority=False,fault_authorized=False)
    with target.open('xb') as stream:stream.write(header)
    with borrow_target.open('xb') as stream:stream.write(borrow)
    unit.write_bytes(changed)
    with receipt.open('x',encoding='utf-8') as stream:json.dump(report,stream,indent=2);stream.write('\n')
    return report


def run_compiler(source,evidence,argv,*,variant,sdk):
    observation=prepare_unit(source,evidence,argv,variant=variant,sdk=sdk)
    if observation is None:return subprocess.run(argv,check=False).returncode
    receipt=Path(evidence)/COMPILE_RECEIPT
    report=dict(schema='demoncat.tnd28.actual_input_compile.v1',status='STARTED',
        variant=variant,source=str(Path(source).resolve()),sdk=observation['sdk'],cwd=str(Path.cwd()),
        original_compiler_argv=argv,effective_compiler_argv=observation['effective_compiler_argv'],
        sdk_headers=observation['sdk_headers'],compiler_calls=0,compiler_returncode=None,npu_executed=False,
        observation_receipt=file_record(Path(evidence)/'tnd28_compiled_input_observer.json',65536))
    # Never overwrite a failed/uncertain original compile or automatically retry.
    with receipt.open('x',encoding='utf-8') as stream:json.dump(report,stream,indent=2);stream.write('\n')
    try:
        dependency,output=output_paths(source,argv)
        need(not dependency.exists() and not output.exists(),'PREEXISTING_GENERATED_COMPILE_OUTPUT')
        report['compiler_calls']=1
        report['compiler_returncode']=subprocess.run(report['effective_compiler_argv'],check=False).returncode
        need(report['compiler_returncode']==0,'ACTUAL_ORIGINAL_COMPILER_FAILED')
        need(sdk_headers(sdk)==report['sdk_headers'],'SDK_HEADERS_CHANGED_DURING_COMPILE')
        need(file_record(observation['unit'],2*1024**2)['sha256']==observation['patched_sha256']
            and file_record(observation['header'],1024**2)['sha256']==observation['header_sha256'],
            'GENERATED_OBSERVER_BYTES_CHANGED_DURING_COMPILE')
        need(file_record(observation['execute_borrow_header'],1024**2)['sha256']==observation['execute_borrow_header_sha256'],
            'GENERATED_BORROW_BYTES_CHANGED_DURING_COMPILE')
        report['object']=file_record(output,128*1024**2)
        report['dependencies']=dependency_records(dependency,cwd=report['cwd'],source=source,sdk=sdk,output=output)
        report['status']='CPU_COMPILE_AND_ACTUAL_SDK_DEPENDENCIES_VERIFIED'
    except BaseException as error:
        report.update(status='FAILED_OR_UNCONFIRMED',first_error=repr(error))
        raise
    finally:
        with receipt.open('w',encoding='utf-8') as stream:json.dump(report,stream,indent=2);stream.write('\n')
    return 0


def verify_compile_receipt(evidence,source,sdk):
    path=Path(evidence)/COMPILE_RECEIPT;ref=file_record(path,2*1024**2);report=json.loads(path.read_bytes())
    need(report['schema']=='demoncat.tnd28.actual_input_compile.v1' and report['variant']==VARIANT
        and report['status']=='CPU_COMPILE_AND_ACTUAL_SDK_DEPENDENCIES_VERIFIED'
        and report['compiler_calls']==1 and report['compiler_returncode']==0 and report['npu_executed'] is False
        and report['source']==str(Path(source).resolve()) and report['sdk']==str(Path(sdk).resolve())
        and report['sdk_headers']==sdk_headers(sdk),'ACTUAL_INPUT_COMPILE_RECEIPT_UNCONFIRMED')
    observation=report['observation_receipt'];obj=report['object']
    need(Path(observation['path'])==(Path(evidence)/'tnd28_compiled_input_observer.json').resolve()
        and file_record(observation['path'],65536)==observation and file_record(obj['path'],128*1024**2)==obj,
        'ACTUAL_INPUT_COMPILE_ARTIFACT_CHANGED')
    prepared=json.loads(Path(observation['path']).read_bytes())
    need(report['effective_compiler_argv']==prepared['effective_compiler_argv']
        and report['original_compiler_argv']==prepared['actual_compiler_argv']
        and report['sdk_headers']==prepared['sdk_headers'],'ACTUAL_INPUT_COMPILER_ARGUMENTS_CHANGED')
    dependency,output=output_paths(source,report['original_compiler_argv'],cwd=report['cwd'])
    need(str(output)==obj['path'] and report['dependencies']==dependency_records(dependency,cwd=report['cwd'],
        source=source,sdk=sdk,output=output),'ACTUAL_INPUT_DEPENDENCY_CLOSURE_CHANGED')
    return dict(receipt=ref,object=obj,dependency_file=report['dependencies']['file'],
        dependency_count=len(report['dependencies']['files']),sdk_headers=report['sdk_headers'],
        compiler_calls=1,compiler_returncode=0,npu_executed=False)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',required=True);p.add_argument('--evidence',required=True)
    p.add_argument('--variant',required=True);p.add_argument('--sdk',required=True)
    p.add_argument('compiler',nargs=argparse.REMAINDER)
    a=p.parse_args();argv=a.compiler
    if argv and argv[0]=='--':argv=argv[1:]
    need(argv,'COMPILER_ARGUMENTS_REQUIRED')
    raise SystemExit(run_compiler(a.source,a.evidence,argv,variant=a.variant,sdk=a.sdk))


if __name__=='__main__':main()
