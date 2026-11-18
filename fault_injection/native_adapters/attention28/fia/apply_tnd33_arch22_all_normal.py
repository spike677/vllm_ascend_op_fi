"""Fresh all-object normal-only host; never authorize a tile change.

The source88 normal route preference is retained. The native inner S2
choice stays unchanged before the original loop/tail/SplitCore/workspace/serializer
calculations. Kernel arithmetic, public arguments and tensor bytes are intact.
The output is build input, not permission to load or run a fault.
"""
import argparse
import difflib
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('_arch22_normal_recipe',HERE/'apply_tnd33_arch22_bridge.py')
normal=importlib.util.module_from_spec(spec);spec.loader.exec_module(normal)
base=normal.base
VARIANT='tnd_page16_arch22_all_normal'
VENDOR='dcat_f33_arch22_all_normal'
KEYS=normal.KEYS
PATHS=normal.PATHS
SOURCE_SHA=dict(normal.SOURCE_SHA,**{
    'attention/common/op_host/split_core.h':'e7d938699bc0ee361104767dce68a65a1a2677f98623dd482bc8ce022ae556cb',
    'attention/common/op_host/split_core.cpp':'55c43a3f96f36aa51ed6a8f634a76fd6ebff7044330f965a283509424219cd02',
    'attention/common/op_kernel/arch22/fia_kernel_nonquant.h':'922c479ae19fb574a7ee4c1d818b87c9ea98eb19ff2bb5e937d7204999f34a0b',
    'attention/common/op_kernel/arch22/fia_block_vec_flashdecode.h':'c02cb5594e62b904afbb7d1652c43b1924497e123abe677a4ad0c1ce5e39edb5',
    'attention/fused_infer_attention_score/op_host/fused_infer_attention_score_tiling_info_parser.cpp':
        '380248471fef863d9c7ae5f77f00aafb71185330d350388652c6f01557a9f271',
    'attention/fused_infer_attention_score/op_host/fused_infer_attention_score_tiling_info_parser.h':
        '54753b541981eb176636bc0243c728830f5076a39d4e69721816d0d4eb6d44f0',
})
SELECT_ANCHOR='    sInnerLoopTimes_ = (s2Size + sInnerSize_ - static_cast<uint32_t>(1)) / sInnerSize_;'
SELECT='    paged33S2_ = paged33_arch22_all_normal::Select(*fiaInfo_, context_, s2Size, sInnerSize_, aicNum_, aivNum_);\n    sInnerSize_ = paged33S2_.selectedS2;\n'
CAPTURE=('    paged33_arch22_all_normal::Capture(paged33S2_, *fiaInfo_, context_, tilingData_, workspaceSize_,\n'
         '        libapiSize_, sInnerLoopTimes_, sInnerSizeTail_, sInnerSizeAlign_, mBaseSize_);\n')


def patched(original):
    result=normal.patched(original)
    path=PATHS[0]
    result[path]=base.once(result[path],'#include "paged33_arch22_bridge.h"','#include "paged33_arch22_all_normal.h"')
    result[path]=base.once(result[path],'    bool splitKVFlag_ = false;',
        '    paged33_arch22_all_normal::State paged33S2_;\n    bool splitKVFlag_ = false;')
    path=PATHS[1]
    result[path]=base.once(result[path],SELECT_ANCHOR,SELECT+SELECT_ANCHOR)
    result[path]=base.once(result[path],
        '    paged33_arch22_bridge::Capture(*fiaInfo_, context_, tilingData_, workspaceSize_, sInnerSize_);\n',CAPTURE)
    return result


def verify_source(raw):
    if set(raw)!=set(SOURCE_SHA) or any(hashlib.sha256(raw[p]).hexdigest()!=h for p,h in SOURCE_SHA.items()):
        raise RuntimeError('ARCH22_S2_REVIEWED_PRISTINE_SOURCE_DIFFERS')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();source=args.source.resolve(strict=True);out=args.output.resolve()
    head=subprocess.check_output(['git','-C',str(source),'rev-parse','HEAD'],text=True).strip()
    dirty=subprocess.check_output(['git','-C',str(source),'status','--porcelain','--untracked-files=all'],text=True)
    if head!=base.COMMIT or dirty.strip() or out.exists() or source==out or source in out.parents:
        raise RuntimeError('PRISTINE_PINNED_SOURCE_AND_FRESH_EXTERNAL_RECEIPT_REQUIRED')
    raw={p:(source/p).read_bytes() for p in SOURCE_SHA};verify_source(raw)
    original={p:v.decode('utf-8') for p,v in raw.items()};updated=patched(original)
    typed,projection,fields=base.typed_materials((source/(base.FIA+'op_host/fused_infer_attention_score_tiling.h')).read_text(encoding='utf-8'))
    typed=typed.replace('fault28_fia','paged33_arch22_bridge').replace('DEMONCAT_FAULT28_FIA_TYPED_H','DEMONCAT_PAGED33_ARCH22_TYPED_H')
    additions={base.HOST+'paged33_arch22_all_normal.h':(HERE/'paged33_arch22_all_normal.h').read_bytes(),
        base.HOST+'paged33_arch22_bridge.h':(HERE/'paged33_arch22_bridge.h').read_bytes(),
        base.HOST+'paged33_arch22_typed.h':typed.encode()}
    if any((source/p).exists() for p in additions):raise RuntimeError('ARCH22_S2_REPEATED_PATCH_REFUSED')
    materials=dict(raw);materials.update({p:v.encode() for p,v in updated.items()});materials.update(additions)
    patch=''.join(''.join(difflib.unified_diff(original[p].splitlines(True),updated[p].splitlines(True),
        fromfile='a/'+p,tofile='b/'+p)) for p in PATHS)
    out.mkdir(parents=True)
    for p,v in materials.items():(source/p).write_bytes(v)
    (out/'official_sources.patch').write_text(patch,encoding='utf-8')
    (out/'fia_schema_projection.h').write_text(projection,encoding='utf-8')
    (out/'typed_fields.json').write_text(json.dumps(fields,indent=2)+'\n',encoding='utf-8')
    receipt=dict(schema='demoncat.paged33.arch22_all_normal_source.v1',mechanism='33.paged_decode',variant=VARIANT,
        source_commit=head,status='PATCHED_NOT_BUILT_NOT_NPU_VALIDATED',
        recipe_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        normal_route_recipe_sha256=hashlib.sha256(Path(normal.__file__).read_bytes()).hexdigest(),
        scope=dict(layout='TND',query_tokens=1,heads=16,kv_heads=8,dim=128,page_size=16,
                   kv_length=[1025,2048],aic=[16,26],cv_ratio=2,native_s2=512,selected_s2=512),
        normal_route_preference_changed=True,selection_before_original_split=True,
        kernel_source_modified=False,tensor_payload_mutated=False,m_policy_changed=False,
        old_normal_certificate_reused=False,fault_execution_ready=False,stock_private_exact_proven=False,
        loading_allowed=False,actual_launch_verified=False,
        files=[dict(path=p,before_sha256=hashlib.sha256(raw[p]).hexdigest() if p in raw else None,
                    after_sha256=hashlib.sha256(v).hexdigest(),bytes=len(v)) for p,v in materials.items()])
    (out/'policy_source_manifest.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(receipt,indent=2))


if __name__=='__main__':main()
