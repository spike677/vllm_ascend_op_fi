"""Patch the *actual* c77 TND FAI path, with matched host/device KV tiles.

The ordinary RouteToFia and flashDecode decisions remain byte-identical. A
separate compiled key selects a 256-element tile only in the reviewed scope.
The prior arch22 experiment did not enter this path and conveys no admission.
"""
import argparse
import difflib
import hashlib
import json
from pathlib import Path
import re
import subprocess

from base_policy import COMMIT, FIA, once

VARIANT = 'tnd_page16_fai_s2_256'
NATIVE_KEYS = (5000000000000200100, 5000000000000200103,
               5000000000010200100, 5000000000010200103)
FAULT_KEYS = (5900000000010200100, 5900000000010200103)
KEYS = (*NATIVE_KEYS, *FAULT_KEYS)
PATHS = (FIA+'op_host/flash_attention_infer_tiling.h',
         FIA+'op_host/fused_infer_attention_score_tiling.cpp',
         FIA+'op_kernel/fia_kernel_common.hpp',
         FIA+'op_kernel/fused_infer_attention_score.cpp')


def typed_materials(header):
    classes = dict(re.findall(r'BEGIN_TILING_DATA_DEF\((\w+)\)(.*?)END_TILING_DATA_DEF', header, re.S))
    root = classes['FAInferTilingData']
    fields = []
    for match in re.finditer(r'TILING_DATA_FIELD_DEF(_STRUCT|_ARR)?\(([^)]+)\)', root):
        parts = [x.strip() for x in match[2].split(',')]
        if match[1] == '_STRUCT':
            kind, group = parts
            for item in re.finditer(r'TILING_DATA_FIELD_DEF_ARR\(([^)]+)\)', classes[kind]):
                dtype, count, name = [x.strip() for x in item[1].split(',')]
                if count != 'MAX_CORE_NUM_FD':
                    raise ValueError('UNREVIEWED_FAI_ARRAY_LENGTH')
                fields.append(dict(name=group+'.'+name, getter='d.'+group+'.get_'+name+'()', dtype=dtype, count=26))
        elif match[1] is None:
            dtype, name = parts
            fields.append(dict(name=name, getter='d.get_'+name+'()', dtype=dtype, count=1))
        else:
            raise ValueError('UNREVIEWED_FAI_ROOT_ARRAY')
    if len(fields) != 52:
        raise ValueError('FAI_PUBLIC_FIELD_SET_CHANGED:'+str(len(fields)))
    lines = ['#pragma once', 'namespace paged33_fai {',
             'inline void AppendTyped(std::ostream& o,optiling::FAInferTilingData& d) {', "o<<'{';"]
    for index, field in enumerate(fields):
        if index:
            lines.append("o<<',';")
        lines.append('o<<Quote("'+field['name']+'")<<\':\';')
        if field['count'] == 1:
            lines.append('o<<'+field['getter']+';')
        else:
            lines.append("o<<'[';for(size_t j=0;j<26;++j){if(j)o<<',';o<<"+field['getter']+"[j];}o<<']';")
    lines += ["o<<'}';", '}', '}', '']
    start = header.index('    constexpr int32_t MAX_CORE_NUM_FD = 26;')
    end = re.search(r'BEGIN_TILING_DATA_DEF\(FAInferTilingData\).*?END_TILING_DATA_DEF', header, re.S).end()
    projection = '#pragma once\n#include "register/tilingdata_base.h"\nnamespace optiling {\n'+header[start:end]+'\n}\n'
    return '\n'.join(lines), projection, fields


def patched(original):
    result = dict(original)
    name = PATHS[0]
    text = result[name]
    text = once(text, '        uint64_t GetTilingKey();',
        '        uint64_t GetTilingKey();\n'
        '        void DcatSetKvTile(uint32_t tile) { dcatKvTile_ = tile; }\n'
        '        uint32_t DcatWorkspaceBlockSize() const { return Q_TILE_CEIL * dcatKvTile_; }')
    text = once(text, '        uint32_t blockNum_;', '        uint32_t blockNum_;\n        uint32_t dcatKvTile_ = MAX_KV_STACK_LEN;')
    text = once(text, '        uint32_t kSBlockTile = MAX_KV_STACK_LEN;', '        uint32_t kSBlockTile = dcatKvTile_;')
    # All four workspace slices must use the selected geometry. Nothing is
    # overwritten after serialization, and no array or tensor input is edited.
    expression = 'static_cast<uint64_t>(blockNum_) * WORKSPACE_BLOCK_SIZE_DB *'
    if text.count(expression) != 4:
        raise ValueError('FAI_WORKSPACE_SLICE_SET_CHANGED')
    text = text.replace(expression, 'static_cast<uint64_t>(blockNum_) * DcatWorkspaceBlockSize() *')
    text = once(text, '        return tilingKey;',
        '        if (dcatKvTile_ == 256U) tilingKey += 900000000000000000ULL;\n        return tilingKey;')
    result[name] = text
    name = PATHS[1]
    text = once(result[name], '#include "flash_attention_infer_tiling.h"',
                '#include "flash_attention_infer_tiling.h"\n#include "paged33_fai_policy.h"')
    text = once(text, '    fai_tiling.SetCoreNum(aicoreNum);',
        '    fai_tiling.SetCoreNum(aicoreNum);\n'
        '    const auto dcatFai = paged33_fai::Select(faiContext, context, aicoreNum);\n'
        '    fai_tiling.DcatSetKvTile(dcatFai.selectedS2);')
    text = once(text,
        'static_cast<uint64_t>(fai_tiling.GetCoreNum()) * WORKSPACE_BLOCK_SIZE_DB * 4U * 3U * 4U',
        'static_cast<uint64_t>(fai_tiling.GetCoreNum()) * fai_tiling.DcatWorkspaceBlockSize() * 4U * 3U * 4U')
    text = once(text, '    context->SetTilingKey(fai_tiling.GetTilingKey());',
        '    context->SetTilingKey(fai_tiling.GetTilingKey());\n'
        '    paged33_fai::Capture(dcatFai, context, faiTilingData, faiContext.workspaces[0]);')
    anchor = 'REGISTER_TILING_DATA_CLASS(FusedInferAttentionScore_5000000000000200100, FAInferTilingData)'
    text = once(text, anchor, '\n'.join(f'REGISTER_TILING_DATA_CLASS(FusedInferAttentionScore_{key}, FAInferTilingData)'
                                      for key in FAULT_KEYS)+'\n'+anchor)
    result[name] = text
    name = PATHS[2]
    result[name] = once(result[name], '    constexpr uint32_t MAX_KV_STACK_LEN = 512;',
        '#if TILING_KEY_VAR == 5900000000010200100ULL || TILING_KEY_VAR == 5900000000010200103ULL\n'
        '    constexpr uint32_t MAX_KV_STACK_LEN = 256;\n'
        '#else\n    constexpr uint32_t MAX_KV_STACK_LEN = 512;\n#endif')
    name = PATHS[3]
    text = result[name]
    for suffix, key in (('NOMASK', FAULT_KEYS[0]), ('CAUSALMASK', FAULT_KEYS[1])):
        symbol = 'QF16_KVF16_OUTF16_NOLSEOUT_TND_PAGEDCACHE_'+suffix+'_SPLITFUSE_TILING'
        text = once(text, '        TILING_KEY_IS('+symbol+');',
                    f'        TILING_KEY_IS({key});\n        TILING_KEY_IS('+symbol+');')
        text = once(text, '#elif TILING_KEY_VAR == '+symbol+'\n',
                    '#elif TILING_KEY_VAR == '+symbol+f' || TILING_KEY_VAR == {key}ULL\n')
    result[name] = text
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    source, out = args.source.resolve(strict=True), args.output.resolve()
    head = subprocess.check_output(['git', '-C', str(source), 'rev-parse', 'HEAD'], text=True).strip()
    dirty = subprocess.check_output(['git', '-C', str(source), 'status', '--porcelain', '--untracked-files=all'], text=True)
    if head != COMMIT or dirty.strip() or out.exists() or source == out or source in out.parents:
        raise RuntimeError('PRISTINE_PINNED_SOURCE_AND_FRESH_EXTERNAL_RECEIPT_REQUIRED')
    raw = {name: (source/name).read_bytes() for name in PATHS}
    original = {name: value.decode() for name, value in raw.items()}
    updated = patched(original)
    typed, projection, fields = typed_materials(original[PATHS[0]])
    additions = {FIA+'op_host/paged33_fai_policy.h': Path(__file__).with_name('paged33_fai_policy.h').read_bytes(),
                 FIA+'op_host/paged33_fai_typed.h': typed.encode()}
    if any((source/name).exists() for name in additions):
        raise RuntimeError('REPEATED_FAI_PATCH_REFUSED')
    materials = {name: value.encode() for name, value in updated.items()}
    materials.update(additions)
    for name, value in materials.items():
        (source/name).write_bytes(value)
    out.mkdir(parents=True)
    patch = ''.join(''.join(difflib.unified_diff(original[name].splitlines(True), updated[name].splitlines(True),
                    fromfile='a/'+name, tofile='b/'+name)) for name in PATHS)
    (out/'official_sources.patch').write_text(patch)
    (out/'fai_schema_projection.h').write_text(projection)
    (out/'typed_fields.json').write_text(json.dumps(fields, indent=2)+'\n')
    receipt = dict(schema='demoncat.paged33.fai_policy_source.v1', mechanism='33.paged_decode',
        variant=VARIANT, source_commit=head, status='PATCHED_NOT_BUILT_NOT_NPU_VALIDATED',
        route='TilingProcess4SplitFuse / FAInferTiling', native_route_or_decode_flags_changed=False,
        scope=dict(layout='TND', page_size=16, batch=1, hq=16, hkv=8, dim=128,
                   kv_length=[1025, 2048], aic=16, flash_decode=False, native_s2=512, fault_s2=256),
        native_keys=list(NATIVE_KEYS), fault_keys=list(FAULT_KEYS),
        tensor_payload_mutated=False, kernel_source_modified=True,
        old_bnsd_or_arch22_qualification_reused=False, loading_allowed=False, actual_launch_verified=False,
        files=[dict(path=name, before_sha256=hashlib.sha256(raw[name]).hexdigest() if name in raw else None,
                    after_sha256=hashlib.sha256(value).hexdigest(), bytes=len(value)) for name, value in materials.items()])
    (out/'policy_source_manifest.json').write_text(json.dumps(receipt, indent=2)+'\n')
    print(json.dumps(receipt, indent=2))


if __name__ == '__main__':
    main()
