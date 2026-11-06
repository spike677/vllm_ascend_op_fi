"""Add opt-in M selection and a separate legal causal-tail FIA kernel variant."""
import argparse
import difflib
import hashlib
import json
from pathlib import Path
import re
import subprocess

COMMIT='c77c0829f1f0f0b8f882db1d9b051a758b605472'
HOST='attention/common/op_host/arch22/'
KERNEL='attention/common/op_kernel/'
FIA='attention/fused_infer_attention_score/'
KEYS=(109000000000000000,109000000000400000)


def once(text,old,new):
    if text.count(old)!=1:raise ValueError('Missing/ambiguous source anchor: '+old[:100])
    return text.replace(old,new,1)


def typed_materials(header):
    classes=dict(re.findall(r'BEGIN_TILING_DATA_DEF\((\w+)\)(.*?)END_TILING_DATA_DEF',header,re.S))
    groups=re.findall(r'TILING_DATA_FIELD_DEF_STRUCT\((\w+),\s*(\w+)\)',classes['FusedInferAttentionScoreTilingData'])
    fields=[]
    for typename,group in groups:
        block=classes[typename]
        for m in re.finditer(r'TILING_DATA_FIELD_DEF(_ARR)?\(([^)]+)\)',block):
            parts=[v.strip() for v in m[2].split(',')]
            kind,name=parts[0],parts[-1]
            count=1
            if m[1]:
                expr=parts[1].replace(' ','')
                count={'FIA_MAX_AIC_CORE_NUM':26,'FIA_MAX_AIC_CORE_NUM*2':52}[expr]
            fields.append({'name':group+'.'+name,'group':group,'field':name,'dtype':kind,'count':count})
    lines=['#ifndef DEMONCAT_FAULT28_FIA_TYPED_H','#define DEMONCAT_FAULT28_FIA_TYPED_H',
           'namespace fault28_fia {','inline void AppendTyped(std::ostream& o,optiling::FusedInferAttentionScoreTilingData& d) {',
           'o<<\'{\';']
    for index,f in enumerate(fields):
        if index:lines.append("o<<',';")
        lines.append('o<<Quote("'+f['name']+'")<<\':\';')
        getter='d.'+f['group']+'.get_'+f['field']+'()'
        if f['count']==1:lines.append('o<<'+getter+';')
        else:lines.append('o<<\'[\';for(size_t j=0;j<'+str(f['count'])+';++j){if(j)o<<\',\';o<<'+getter+'[j];}o<<\']\';')
    lines+=['o<<\'}\';','}','}','#endif','']
    begin=header.index('const uint32_t FIA_MAX_AIC_CORE_NUM')
    last=re.search(r'BEGIN_TILING_DATA_DEF\(FusedInferAttentionScoreTilingData\).*?END_TILING_DATA_DEF',header,re.S).end()
    projection='#pragma once\n#include "register/tilingdata_base.h"\nnamespace optiling {\n'+header[begin:last]+'\n}\n'
    return '\n'.join(lines),projection,fields


def patched(original):
    result=dict(original)
    path=HOST+'fia_tiling_nonquant.h';t=result[path]
    t=once(t,'namespace optiling {','#include "fault28_fia_policy.h"\n\nnamespace optiling {')
    t=once(t,'    bool splitKVFlag_ = false;','    fault28_fia::State fault28State_;\n    bool splitKVFlag_ = false;')
    result[path]=t
    path=HOST+'fia_tiling_nonquant.cpp';t=result[path]
    anchor='    softmaxWithBrcbFlag_ = (mBaseSize_ <= M_BASE_SIZE_128);'
    t=once(t,anchor,'    fault28State_ = fault28_fia::Select(*fiaInfo_, context_, aicNum_, aivNum_, mBaseSize_, sInnerSize_);\n'
           '    mBaseSize_ = fault28State_.selectedM;\n'+anchor)
    t=once(t,'    OP_LOGI(fiaInfo_->opName, "FIA tilingKey_: %lu.", tilingKey_);',
           '    fault28State_.preExtensionKey = tilingKey_;\n'
           '    if (fault28State_.eligible && fault28State_.prune) tilingKey_ += 6000000000000000ULL;\n'
           '    OP_LOGI(fiaInfo_->opName, "FIA tilingKey_: %lu.", tilingKey_);')
    t=once(t,'        CalcWorkspaceSize();\n        GenTilingKey();',
           '        CalcWorkspaceSize();\n'
           '        if (fault28State_.eligible && IsFlashDecode()) {\n'
           '            OP_LOGE(fiaInfo_->opName, "Fault28 requires complete in-core KV reduction; SplitKV rejected.");\n'
           '            return ge::GRAPH_FAILED;\n'
           '        }\n        GenTilingKey();')
    anchor='    return ge::GRAPH_SUCCESS;\n}\n\n// '
    if t.count(anchor)!=1:raise ValueError('Ambiguous final tiling return')
    t=once(t,anchor,'    fault28_fia::Capture(fault28State_, context_, tilingData_, fault28State_.preExtensionKey, workspaceSize_, aicNum_, aivNum_);\n'+anchor)
    result[path]=t
    path=KERNEL+'fia_public_define.h';t=result[path]
    t=once(t,'    static constexpr bool enableTree = ENABLE_TREE;',
           '    static constexpr bool enableTree = ENABLE_TREE;\n    static constexpr bool fault28CausalPrune = false;')
    result[path]=t
    path=KERNEL+'arch22/fia_kernel_nonquant.h';t=result[path]
    t=once(t,'using namespace matmul;','#include "fault28_causal_tile.h"\n\nusing namespace matmul;')
    anchor='    info.actualSingleProcessSInnerSizeAlign =\n        AttentionCommon::Align((uint32_t)info.actualSingleProcessSInnerSize, (uint32_t)AttentionCommon::BYTE_BLOCK);'
    t=once(t,anchor,'    if constexpr (FIAT::fault28CausalPrune) {\n'
           '        static_assert(LAYOUT_T == FIA_LAYOUT::BNSD && !PAGE_ATTENTION && !FLASH_DECODE && !ENABLE_TREE, "Fault28 scope");\n'
           '        info.actualSingleProcessSInnerSize = fault28_causal_tile::InnerWidth(\n'
           '            info.gS1Idx, info.actMBaseSize, info.actS1Size, s2Cur, constInfo.s2BaseSize,\n'
           '            info.actualSingleProcessSInnerSize);\n'
           '    }\n'+anchor)
    result[path]=t
    path=FIA+'op_kernel/fused_infer_attention_score_v3.cpp';t=result[path]
    t=once(t,'using namespace AscendC;', 'using namespace AscendC;\n\n'
           'template <class Base> struct Fault28CausalFIAType : Base { static constexpr bool fault28CausalPrune = true; };')
    macro=t[t.index('#define INVOKE_FIA_OP_GENERAL_IMPL'):t.index('#define FIA_COPY_TILING_DATA')]
    new_macro=macro.replace('INVOKE_FIA_OP_GENERAL_IMPL','INVOKE_FAULT28_FIA_PRUNED').replace('FIAType<__VA_ARGS__>','Fault28CausalFIAType<FIAType<__VA_ARGS__>>')
    t=once(t,'#define FIA_COPY_TILING_DATA',new_macro+'#define FIA_COPY_TILING_DATA')
    for new in KEYS:
        old=new-6000000000000000
        t=once(t,f'    TILING_KEY_IS({old});',f'    TILING_KEY_IS({new});\n    TILING_KEY_IS({old});')
        body=re.search(r'\n#elif TILING_KEY_VAR == '+str(old)+r'\n(.*?)(?=\n#elif |\n#endif)',t,re.S)
        if not body:raise ValueError('Missing legal native key variant')
        new_body=body[1].replace('INVOKE_FIA_OP_GENERAL_IMPL','INVOKE_FAULT28_FIA_PRUNED')
        t=once(t,f'\n#elif TILING_KEY_VAR == {old}\n',f'\n#elif TILING_KEY_VAR == {new}\n'+new_body+f'\n#elif TILING_KEY_VAR == {old}\n')
    result[path]=t
    path=FIA+'op_host/arch22/fused_infer_attention_score_tiling_v3.cpp';t=result[path]
    anchor='FIA_EXTERN_C ge::graphStatus TilingFusedInferAttentionScoreV3(gert::TilingContext *context)'
    registers=''.join(f'REGISTER_TILING_DATA_CLASS(FusedInferAttentionScore_{key}, FusedInferAttentionScoreTilingData)\n' for key in KEYS)
    result[path]=once(t,anchor,registers+'\n'+anchor)
    return result


PATHS=[HOST+'fia_tiling_nonquant.h',HOST+'fia_tiling_nonquant.cpp',KERNEL+'fia_public_define.h',
       KERNEL+'arch22/fia_kernel_nonquant.h',FIA+'op_kernel/fused_infer_attention_score_v3.cpp',
       FIA+'op_host/arch22/fused_infer_attention_score_tiling_v3.cpp']


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();src=a.source.resolve();out=a.output.resolve()
    head=subprocess.check_output(['git','-C',str(src),'rev-parse','HEAD'],text=True).strip()
    dirty=subprocess.check_output(['git','-C',str(src),'status','--porcelain','--untracked-files=all'],text=True)
    if head!=COMMIT or dirty.strip() or out.exists() or src==out or src in out.parents:raise ValueError('Requires pristine pinned source and fresh external evidence')
    raw={name:(src/name).read_bytes() for name in PATHS}
    original={name:value.decode('utf-8') for name,value in raw.items()}
    final=patched(original)
    typed,projection,fields=typed_materials((src/(FIA+'op_host/fused_infer_attention_score_tiling.h')).read_text(encoding='utf-8'))
    additions={HOST+'fault28_fia_policy.h':Path(__file__).with_name('fault28_fia_policy.h').read_bytes(),
               HOST+'fault28_fia_typed.h':typed.encode(),KERNEL+'arch22/fault28_causal_tile.h':Path(__file__).with_name('fault28_causal_tile.h').read_bytes()}
    if any((src/name).exists() for name in additions):raise ValueError('Refuse repeated patch')
    # Validate every source anchor and construct all materials before mutation.
    materials={name:value.encode() for name,value in final.items()};materials.update(additions)
    for name,value in materials.items():(src/name).write_bytes(value)
    for name in [FIA+'op_host/fused_infer_attention_score_tiling.h',
                 FIA+'op_kernel/fused_infer_attention_score.cpp',FIA+'op_kernel/fused_infer_attention_score_tilingkey.h',
                 KERNEL+'memory_copy.h']:
        raw[name]=(src/name).read_bytes()
        materials[name]=raw[name]
    diff=''.join(''.join(difflib.unified_diff(original[name].splitlines(True),final[name].splitlines(True),fromfile='a/'+name,tofile='b/'+name)) for name in PATHS)
    out.mkdir(parents=True,exist_ok=False)
    (out/'official_sources.patch').write_text(diff,encoding='utf-8')
    (out/'fia_schema_projection.h').write_text(projection,encoding='utf-8')
    (out/'typed_fields.json').write_text(json.dumps(fields,indent=2)+'\n')
    receipt={'schema':'fault28.fia_graph.policy_source.v1','source_commit':head,'source_release':'v9.1.0',
             'status':'PATCHED_NOT_BUILT_NOT_NPU_VALIDATED','policy_kind':'controlled_current_source_M_selector_and_legal_causal_tail_variant',
             'kernel_source_modified':True,'new_keys':[str(x) for x in KEYS],
             'key_data':'Original public FIA serializer retained; no raw-field patch or tensor payload changes.',
             'files':[{'path':name,'before_sha256':hashlib.sha256(raw[name]).hexdigest() if name in raw else None,
                       'after_sha256':hashlib.sha256(value).hexdigest(),'bytes':len(value)} for name,value in materials.items()],
             'projection_sha256':hashlib.sha256(projection.encode()).hexdigest(),'typed_field_count':len(fields)}
    (out/'policy_source_manifest.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt,indent=2))


if __name__=='__main__':main()
