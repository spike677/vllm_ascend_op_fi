"""Derive the TND causal M128 control and true denominator replay from c77.

Source preparation only. No NPU, loading or Worker qualification is granted.
The old BNSD policy and the old TND normal package are not modified.
"""
import argparse
import difflib
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import subprocess

_spec=importlib.util.spec_from_file_location('tnd28_normal_recipe',Path(__file__).with_name('apply_tnd28_prefill_normal.py'))
normal=importlib.util.module_from_spec(_spec);_spec.loader.exec_module(normal)
base=normal.base
VARIANT='tnd_prefill_denominator_replay'
VENDOR='dcat_f28_tnd_replay'
NATIVE_KEY=normal.NORMAL_KEY
NATIVE_BRC_KEY=103000000030400003
CONTROL_KEY=109000000030400003
REPLAY_KEY=119000000030400003
KEYS=(NATIVE_KEY,NATIVE_BRC_KEY,CONTROL_KEY,REPLAY_KEY,*normal.FALLBACK_KEYS)
TRAIT=base.KERNEL+'fia_public_define.h'
KERNEL=base.KERNEL+'arch22/fia_kernel_nonquant.h'
VECTOR=base.KERNEL+'arch22/fia_block_vec_nonquant.h'
DISPATCH=base.FIA+'op_kernel/fused_infer_attention_score_v3.cpp'
ROUTE=normal.ROUTE
PATHS=(*normal.PATHS,TRAIT,KERNEL,VECTOR,DISPATCH)
SOURCE_SHA=dict(normal.SOURCE_SHA)
SOURCE_SHA.update({
    base.KERNEL+'arch22/fia_block_cube_nonquant_gqa.h':'9ea5776e25d575d911c9182dee8f41fe8d1a026622e643b07107a8f25ca293c5',
    base.KERNEL+'vector_common.h':'adeae5bbfb1b2d05557e31aa007b79ce5f5add502b270922b66aec97f592478b',
    'attention/common/op_host/split_core.h':'e7d938699bc0ee361104767dce68a65a1a2677f98623dd482bc8ce022ae556cb',
    'attention/common/op_host/split_core.cpp':'55c43a3f96f36aa51ed6a8f634a76fd6ebff7044330f965a283509424219cd02'})
EXTRA_UNCHANGED=normal.EXTRA_UNCHANGED
REPLAY_CODE='''    // Repeat only the old denominator on the second complete in-core block.
    // Max, QK, numerator/PV, final divide/cast and output stores are unchanged.
    if constexpr (FIAT::tnd28DenominatorReplay) {
        static_assert(SOFTMAX_WITH_BRC && !FLASH_DECODE && FIAT::layout == FIA_LAYOUT::TND
            && FIAT::kvLayout == FIA_LAYOUT::TND && !FIAT::pageAttention && !FIAT::enableTree,
            "TND28 replay requires the exact no-FD TND BRC specialization");
        if (!info.isFirstSInnerLoop && info.s2Idx == 1) {
            const uint32_t replayCount = dealRowCount * this->brcbNum;
            PipeBarrier<PIPE_V>();
            Muls(inSumTensor, inSumTensor, COMPUTE_T(2), replayCount);
            PipeBarrier<PIPE_V>();
        }
    }
'''
CAUSAL_CODE='''    if constexpr (FIAT::tnd28CausalPrune) {
        static_assert(LAYOUT_T == FIA_LAYOUT::TND && !PAGE_ATTENTION && !FLASH_DECODE && !ENABLE_TREE,
            "TND28 causal scope requires complete TND reductions");
        info.actualSingleProcessSInnerSize = tnd28_causal_tile::InnerWidth(
            info.gS1Idx, info.actMBaseSize, info.actS1Size, constInfo.gSize,
            s2Cur, constInfo.s2BaseSize, info.actualSingleProcessSInnerSize);
    }
'''


def patched(original):
    result=dict(original)
    result.update(normal.patched(original))
    header,host,_=normal.PATHS
    result[header]=base.once(result[header],'#include "tnd28_prefill_normal.h"',
        '#include "tnd28_denominator_policy.h"')
    result[header]=base.once(result[header],'    bool splitKVFlag_ = false;',
        '    tnd28_denominator::State tnd28Replay_;\n    bool splitKVFlag_ = false;')
    result[ROUTE]=base.once(result[ROUTE],'!tnd28_prefill_normal::Matches(context)',
        '!tnd28_denominator::UseRoute(context)')
    result[host]=base.once(result[host],'    softmaxWithBrcbFlag_ = (mBaseSize_ <= M_BASE_SIZE_128);',
        '    tnd28Replay_ = tnd28_denominator::Select(*fiaInfo_, context_, mBaseSize_, sInnerSize_, aicNum_, aivNum_);\n'
        '    if (tnd28Replay_.selected) mBaseSize_ = 128;\n'
        '    softmaxWithBrcbFlag_ = (mBaseSize_ <= M_BASE_SIZE_128);')
    result[host]=base.once(result[host],'    CalcMBaseSize();\n}',
        '    CalcMBaseSize();\n'
        '    if (tnd28Replay_.selected) {\n'
        '        sInnerSize_ = 128;\n'
        '        sInnerLoopTimes_ = (s2Size + sInnerSize_ - 1U) / sInnerSize_;\n'
        '        sInnerSizeTail_ = s2Size - (sInnerLoopTimes_ - 1U) * sInnerSize_;\n'
        '        sInnerSizeAlign_ = Align(sInnerSize_, BYTE_BLOCK);\n    }\n}')
    result[host]=base.once(result[host],'    if ((SetNumBlocks(numBlocks_) != ge::GRAPH_SUCCESS) ||',
        '    tilingKey_ = tnd28_denominator::ExtendKey(tnd28Replay_, tilingKey_, tilingData_);\n'
        '    if ((SetNumBlocks(numBlocks_) != ge::GRAPH_SUCCESS) ||')
    result[host]=base.once(result[host],normal.CAPTURE,normal.CAPTURE+
        '    tnd28_denominator::Capture(tnd28Replay_, *fiaInfo_, context_, tilingData_, workspaceSize_,\n'
        '        mBaseSize_, sInnerSize_, sInnerLoopTimes_, sInnerSizeTail_, sInnerSizeAlign_);\n')
    result[TRAIT]=base.once(result[TRAIT],'    static constexpr bool enableTree = ENABLE_TREE;',
        '    static constexpr bool enableTree = ENABLE_TREE;\n'
        '    static constexpr bool tnd28CausalPrune = false;\n'
        '    static constexpr bool tnd28DenominatorReplay = false;')
    result[KERNEL]=base.once(result[KERNEL],'using namespace matmul;',
        '#include "tnd28_causal_tile.h"\n\nusing namespace matmul;')
    anchor='    info.actualSingleProcessSInnerSizeAlign =\n        AttentionCommon::Align((uint32_t)info.actualSingleProcessSInnerSize, (uint32_t)AttentionCommon::BYTE_BLOCK);'
    result[KERNEL]=base.once(result[KERNEL],anchor,CAUSAL_CODE+anchor)
    anchor='    if constexpr (SOFTMAX_WITH_BRC) {\n        SoftmaxFlashV2<SOFTMAX_TYPE'
    result[VECTOR]=base.once(result[VECTOR],anchor,REPLAY_CODE+anchor)
    t=result[DISPATCH]
    t=base.once(t,'using namespace AscendC;', 'using namespace AscendC;\n'
        'template <class Base, bool Replay> struct Tnd28ReplayType : Base {\n'
        '    static constexpr bool tnd28CausalPrune = true;\n'
        '    static constexpr bool tnd28DenominatorReplay = Replay;\n};')
    start,end=t.index('#define INVOKE_FIA_OP_GENERAL_IMPL'),t.index('#define FIA_COPY_TILING_DATA')
    macro=t[start:end]
    macros=''
    for name,replay in (('CONTROL','false'),('REPLAY','true')):
        macros+=macro.replace('INVOKE_FIA_OP_GENERAL_IMPL','INVOKE_TND28_'+name).replace(
            'FIAType<__VA_ARGS__>','Tnd28ReplayType<FIAType<__VA_ARGS__>, '+replay+'>')
    t=t[:end]+macros+t[end:]
    match=re.search(r'\n#elif TILING_KEY_VAR == '+str(NATIVE_KEY)+r'\n(.*?)(?=\n#elif |\n#endif)',t,re.S)
    if not match: raise ValueError('TND28_ORIGINAL_TND_GQA_DISPATCH_REQUIRED')
    original_body=match[1]
    brc=base.once(original_body,'FIA_LAYOUT::TND,\n                               false);',
                  'FIA_LAYOUT::TND,\n                               true);')
    branches=''
    for key,macro_name in ((NATIVE_BRC_KEY,'INVOKE_FIA_OP_GENERAL_IMPL'),
            (CONTROL_KEY,'INVOKE_TND28_CONTROL'),(REPLAY_KEY,'INVOKE_TND28_REPLAY')):
        t=base.once(t,f'    TILING_KEY_IS({NATIVE_KEY});',f'    TILING_KEY_IS({key});\n    TILING_KEY_IS({NATIVE_KEY});')
        branches+=f'\n#elif TILING_KEY_VAR == {key}\n'+brc.replace('INVOKE_FIA_OP_GENERAL_IMPL',macro_name)
    t=base.once(t,f'\n#elif TILING_KEY_VAR == {NATIVE_KEY}\n',branches+f'\n#elif TILING_KEY_VAR == {NATIVE_KEY}\n')
    result[DISPATCH]=t
    anchor='FIA_EXTERN_C ge::graphStatus TilingFusedInferAttentionScoreV3(gert::TilingContext *context)'
    registers=''.join(f'REGISTER_TILING_DATA_CLASS(FusedInferAttentionScore_{k}, FusedInferAttentionScoreTilingData)\n'
                      for k in (NATIVE_BRC_KEY,CONTROL_KEY,REPLAY_KEY))
    result[ROUTE]=base.once(result[ROUTE],anchor,registers+anchor)
    return result


def materials(raw):
    if set(raw)!=set((*SOURCE_SHA,*EXTRA_UNCHANGED)) or any(
            hashlib.sha256(raw[p]).hexdigest()!=h for p,h in SOURCE_SHA.items()):
        raise ValueError('TND28_REPLAY_REVIEWED_PRISTINE_SOURCE_DIFFERS')
    original={p:raw[p].decode('utf-8') for p in PATHS}
    changed={p:s.encode('utf-8') for p,s in patched(original).items()}
    typed,projection,fields=base.typed_materials(raw[EXTRA_UNCHANGED[0]].decode('utf-8'))
    typed=typed.replace('fault28_fia','tnd28_prefill_normal').replace('DEMONCAT_FAULT28_FIA_TYPED_H','DEMONCAT_TND28_PREFILL_TYPED_H')
    for name in ('tnd28_prefill_normal.h','tnd28_prefill_contract.h','tnd28_denominator_policy.h','tnd28_denominator_contract.h'):
        changed[base.HOST+name]=Path(__file__).with_name(name).read_bytes()
    changed[base.HOST+'tnd28_prefill_typed.h']=typed.encode()
    changed[base.KERNEL+'arch22/tnd28_causal_tile.h']=Path(__file__).with_name('tnd28_causal_tile.h').read_bytes()
    return dict(raw,**changed),changed,projection,fields


def receipt(raw,final,projection,fields):
    sha=lambda b:hashlib.sha256(b).hexdigest()
    return dict(schema='demoncat.attention28.tnd_denominator_source.v1',mechanism='28.fia.denominator_replay',
        variant=VARIANT,vendor=VENDOR,source_commit=base.COMMIT,status='PATCHED_NOT_BUILT_NOT_NPU_VALIDATED',
        recipe_sha256=sha(Path(__file__).read_bytes()),normal_recipe_sha256=sha(Path(normal.__file__).read_bytes()),
        base_recipe_sha256=sha(Path(base.__file__).read_bytes()),projection_sha256=sha(projection.encode()),
        typed_field_count=len(fields),compile_keys=[str(k) for k in KEYS],
        normal_key=str(NATIVE_KEY),native_brc_key=str(NATIVE_BRC_KEY),control_key=str(CONTROL_KEY),replay_key=str(REPLAY_KEY),
        scope=dict(layout='TND',batch=1,s1=2048,s2=2048,heads=16,kv_heads=8,dim=128,
                   page_attention=False,api_block_size=128,m=128,s2_base=128,sparse_mode=3),
        kernel_source_modified=True,denominator_recurrence_modified=True,numerator_or_output_path_changed=False,
        actual_split_core_modified=False,selection_before_original_split=True,normal_only=False,
        fault_execution_ready=False,loading_allowed=False,stock_private_exact_proven=False,
        public_prepared_pointer_stride_join=False,actual_worker_platform_observed=False,c1=False,c2=False,c3=False,
        files=[dict(path=p,before_sha256=sha(raw[p]) if p in raw else None,after_sha256=sha(b),bytes=len(b))
               for p,b in final.items()])


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();source=args.source.resolve(strict=True);out=args.output.resolve()
    head=subprocess.check_output(['git','-C',str(source),'rev-parse','HEAD'],text=True).strip()
    dirty=subprocess.check_output(['git','-C',str(source),'status','--porcelain','--untracked-files=all'],text=True)
    if head!=base.COMMIT or dirty.strip() or out.exists() or source==out or source in out.parents:
        raise ValueError('TND28_REPLAY_FRESH_PRISTINE_SOURCE_REQUIRED')
    names=(*SOURCE_SHA,*EXTRA_UNCHANGED)
    if any((source/n).is_symlink() or (source/n).resolve(strict=True)!=source/n for n in names):
        raise ValueError('TND28_REPLAY_SOURCE_SYMLINK_REFUSED')
    raw={n:(source/n).read_bytes() for n in names}
    final,changed,projection,fields=materials(raw);report=receipt(raw,final,projection,fields)
    if any((source/n).exists() for n in changed if n not in raw): raise ValueError('TND28_REPLAY_REPEATED_PATCH')
    diff=''.join(''.join(difflib.unified_diff(raw[n].decode().splitlines(True),changed[n].decode().splitlines(True),
        fromfile='a/'+n,tofile='b/'+n)) for n in PATHS)
    out.mkdir(parents=True);written=[]
    try:
        for n,b in changed.items(): (source/n).write_bytes(b);written.append(n)
        (out/'official_sources.patch').write_text(diff,encoding='utf-8')
        (out/'fia_schema_projection.h').write_text(projection,encoding='utf-8')
        (out/'typed_fields.json').write_text(json.dumps(fields,indent=2)+'\n',encoding='utf-8')
    except BaseException as e:
        report.update(status='PATCH_FAILED_SOURCE_RETAINED',written_files=written,failure=type(e).__name__+':'+str(e));raise
    finally: (out/'policy_source_manifest.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(status=report['status'],variant=VARIANT,loading_allowed=False)))


if __name__=='__main__':main()
