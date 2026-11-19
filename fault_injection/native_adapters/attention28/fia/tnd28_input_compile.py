"""Exact generated-unit compiler launcher for the TND replay candidate only.

The actual original compiler executes once. No operator or device is executed.
All other compilation units pass through unchanged. This is not a runtime shim.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

VARIANT='tnd_prefill_denominator_replay'
GENERATED='aclnnInner_fused_infer_attention_score.cpp'
ORIGINAL_SHA='d35f8b60d1ad50d97b7cf0d2050be5c23fd0d53eba77a58c053c184d96af4fe2'
GENERATED_HEADER_SHA='fae730eb70329a1f0fd3f982fd130f80998b4d9b1339e75b6c9d021dd667c3ff'
HEADER='tnd28_input_observation.h'
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
    prepare=once(prepare,'{\n    uint64_t timeStamp =','{\n'+entry+'    uint64_t timeStamp =')
    prepare=once(prepare,'aclnnContiguousGetWorkspaceSize(inUnContTensors, &inContWorkspaceSize, &aclInExecutor)',
        'dcat_input.Contiguous(aclnnContiguousGetWorkspaceSize, inUnContTensors, &inContWorkspaceSize, &aclInExecutor)')
    prepare=once(prepare,'NnopbaseRunForWorkspace(*executor, workspaceSize)',
        'dcat_input.Workspace(&NnopbaseRunForWorkspace, *executor, workspaceSize)')
    prepare=once(prepare,'    return ACLNN_SUCCESS;\n}',
        '    return dcat_input.Return(ACLNN_SUCCESS, workspaceSize, executor);\n}')
    execute=once(execute,'{\n    uint64_t timeStamp =',
        '{\n    tnd28_input::Execute dcat_input(&aclnnInnerFusedInferAttentionScore, executor);\n    uint64_t timeStamp =')
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


def prepare_unit(source,evidence,argv,*,variant):
    need(variant==VARIANT,'INPUT_OBSERVER_EXACT_VARIANT_REQUIRED')
    source=Path(source).resolve(strict=True);evidence=Path(evidence).resolve(strict=True)
    candidates=[Path(x) for x in argv if Path(x).name==GENERATED]
    if not candidates:return None
    need(len(candidates)==1 and '-c' in argv,'ONE_ACTUAL_GENERATED_COMPILATION_REQUIRED')
    need(not any(x.startswith('@') for x in argv),'EXPLICIT_GENERATED_COMPILER_ARGUMENTS_REQUIRED')
    need(not any('DCAT_TND28_INPUT_CPU_FIXTURE' in x for x in argv),'CPU_FIXTURE_NOT_A_PRODUCTION_COMPILE')
    unit=candidates[0];expected=source/'build/autogen/inner'/GENERATED
    need(not unit.is_symlink() and unit.resolve(strict=True)==expected
        and expected.resolve(strict=True)==expected,'GENERATED_UNIT_PATH_CHANGED')
    need(unit.stat().st_size<2*1024**2,'GENERATED_UNIT_BOUND')
    raw=unit.read_bytes();receipt=evidence/'tnd28_compiled_input_observer.json'
    header=Path(__file__).with_name(HEADER).read_bytes();target=unit.with_name(HEADER)
    if receipt.exists():
        need(not receipt.is_symlink() and receipt.stat().st_size<65536,'OBSERVER_RECEIPT_PATH_OR_BOUND')
        old=json.loads(receipt.read_bytes())
        need(old['patched_sha256']==sha(raw) and old['header_sha256']==sha(header)
            and not target.is_symlink() and target.read_bytes()==header
            and old['launcher_sha256']==sha(Path(__file__).read_bytes()),'COMPILED_OBSERVER_REENTRY_BYTES_CHANGED')
        return old
    h=unit.with_suffix('.h')
    need(not h.is_symlink() and sha(h.read_bytes())==GENERATED_HEADER_SHA,'GENERATED_HEADER_BYTES_CHANGED')
    need(not target.exists(),'GENERATED_OBSERVER_HEADER_ALREADY_PRESENT')
    changed=patch_source(raw)
    report=dict(schema='demoncat.tnd28.compiled_input_source.v1',variant=variant,
        original_sha256=sha(raw),patched_sha256=sha(changed),header_sha256=sha(header),
        generated_header_sha256=GENERATED_HEADER_SHA,launcher_sha256=sha(Path(__file__).read_bytes()),
        unit=str(unit),header=str(target),actual_compiler_argv=argv,
        source_hooks=['typed_prepare','actual_contiguous_prepare','actual_workspace',
            'actual_prepare_return','typed_execute','actual_contiguous_execute','actual_fia_execute','actual_execute_return'],
        operator_arguments_changed=False,extra_operator_calls=0,observer_only=True,
        runtime_observed=False,current_input_authority=False,fault_authorized=False)
    with target.open('xb') as stream:stream.write(header)
    unit.write_bytes(changed)
    with receipt.open('x',encoding='utf-8') as stream:json.dump(report,stream,indent=2);stream.write('\n')
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',required=True);p.add_argument('--evidence',required=True)
    p.add_argument('--variant',required=True);p.add_argument('compiler',nargs=argparse.REMAINDER)
    a=p.parse_args();argv=a.compiler
    if argv and argv[0]=='--':argv=argv[1:]
    need(argv,'COMPILER_ARGUMENTS_REQUIRED')
    prepare_unit(a.source,a.evidence,argv,variant=a.variant)
    raise SystemExit(subprocess.run(argv,check=False).returncode)


if __name__=='__main__':main()
