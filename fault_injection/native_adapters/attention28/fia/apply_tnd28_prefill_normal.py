"""Fresh c77 TND2048 normal-only route candidate; no shared builder changes."""
import argparse
import difflib
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess


spec = importlib.util.spec_from_file_location('tnd28_normal_base', Path(__file__).with_name('base_policy.py'))
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)
VARIANT = 'tnd_prefill_arch22_normal'
VENDOR = 'dcat_f28_tnd_normal'
NORMAL_KEY = 103000000030000003
# Required unchanged c77 warmup/out-of-scope fallback entries. This is a build
# plan, not a statement that these entries were observed or qualified.
FALLBACK_KEYS = (5000000000000200100, 5000000000000200103,
                 5000000000010200100, 5000000000010200103,
                 5100000000010200100, 5100000000010200103)
KEYS = (NORMAL_KEY, *FALLBACK_KEYS)
ROUTE = base.FIA+'op_host/arch22/fused_infer_attention_score_tiling_v3.cpp'
PATHS = (base.HOST+'fia_tiling_nonquant.h', base.HOST+'fia_tiling_nonquant.cpp', ROUTE)
SOURCE_SHA = dict(zip(PATHS, (
    '213e3ca7b3359277cdd7a2369150cde10a5b9657b2fa4a71bf9a3d1bbf06d400',
    'dc1ee4e8e6f3bc82a329e974a8147960db3617e7b17f88afac4b140224eb40f4',
    '32f0e38d0391076e7c82e4700026a2b9db72b8c96b47a78873bbb443d411710f')))
SOURCE_SHA.update({
    'attention/common/op_host/fia_tiling_info.h':
        'd201e4ed6ca93fb90a7a60c9da5fbfbb5727810768ffd6cd74e8c67115febda4',
    base.FIA+'op_host/fused_infer_attention_score_tiling_index.h':
        '36ab80c447bb001e658f7eb2bf561a04e97b1d2185f9136a676c98da52f79200',
    base.FIA+'op_host/fused_infer_attention_score_tiling_info_parser.cpp':
        '380248471fef863d9c7ae5f77f00aafb71185330d350388652c6f01557a9f271',
    base.FIA+'op_host/fused_infer_attention_score_tiling_info_parser.h':
        '54753b541981eb176636bc0243c728830f5076a39d4e69721816d0d4eb6d44f0',
    base.FIA+'op_kernel/fused_infer_attention_score_v3.cpp':
        'd87215870dc564ac2bd43578e1da67b548f313a09c07986d028f619f65317544',
    base.FIA+'op_kernel/fused_infer_attention_score_tilingkey.h':
        'ca5dd685ba91365d0ec9cdc22a4026a71789d54cb3dcc473f4e5c9e987b3e974',
    base.KERNEL+'fia_public_define.h':
        '8bf065d666a8107fa840f30791c3b565bce77e3f2c7f32f93798e52bd03dea39',
    base.KERNEL+'arch22/fia_kernel_nonquant.h':
        '922c479ae19fb574a7ee4c1d818b87c9ea98eb19ff2bb5e937d7204999f34a0b',
    base.KERNEL+'arch22/fia_block_vec_nonquant.h':
        '312bca72ac4f9ae8ee0b862427ead1aa44d7d047998c4bcd17c4b3bbbb6ceeb2',
})
EXTRA_UNCHANGED = (base.FIA+'op_host/fused_infer_attention_score_tiling.h',
                   base.FIA+'op_kernel/fused_infer_attention_score.cpp')
ROUTE_ANCHOR = '        if (CheckSpecConditions(context)) {\n            return false;\n        }'
CAPTURE = ('    tnd28_prefill_normal::Capture(*fiaInfo_, context_, tilingData_, workspaceSize_, libapiSize_,\n'
           '        aicNum_, aivNum_, mBaseSize_, sInnerSize_, sInnerLoopTimes_, sInnerSizeTail_,\n'
           '        sInnerSizeAlign_, IsFlashDecode());\n')


def patched(original):
    header, implementation, route = (original[name] for name in PATHS)
    header = base.once(header, 'namespace optiling {',
        '#include "tnd28_prefill_normal.h"\n\nnamespace optiling {')
    implementation = base.once(implementation, '    return ge::GRAPH_SUCCESS;\n}\n\n// ',
                               CAPTURE+'    return ge::GRAPH_SUCCESS;\n}\n\n// ')
    route = base.once(route, ROUTE_ANCHOR,
        '        if (CheckSpecConditions(context) && !tnd28_prefill_normal::Matches(context)) {\n'
        '            return false;\n        }')
    return dict(zip(PATHS, (header, implementation, route)))


def verify_source(raw):
    if set(raw) != set(SOURCE_SHA) or any(hashlib.sha256(raw[p]).hexdigest() != h
                                         for p, h in SOURCE_SHA.items()):
        raise ValueError('TND28_NORMAL_REVIEWED_PRISTINE_SOURCE_DIFFERS')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    source, output = args.source.resolve(strict=True), args.output.resolve()
    head = subprocess.check_output(['git', '-C', str(source), 'rev-parse', 'HEAD'], text=True).strip()
    dirty = subprocess.check_output(['git', '-C', str(source), 'status', '--porcelain', '--untracked-files=all'], text=True)
    if head != base.COMMIT or dirty.strip() or output.exists() or source == output or source in output.parents:
        raise ValueError('TND28_PRISTINE_PINNED_SOURCE_AND_FRESH_EXTERNAL_RECEIPT_REQUIRED')
    names = (*SOURCE_SHA, *EXTRA_UNCHANGED)
    if any((source/p).is_symlink() or (source/p).resolve(strict=True) != source/p for p in names):
        raise ValueError('TND28_SOURCE_SYMLINK_REFUSED')
    raw = {p: (source/p).read_bytes() for p in names}
    verify_source({p: raw[p] for p in SOURCE_SHA})
    original = {p: raw[p].decode('utf-8') for p in PATHS}
    updated = patched(original)
    typed, projection, fields = base.typed_materials(raw[EXTRA_UNCHANGED[0]].decode('utf-8'))
    typed = typed.replace('fault28_fia', 'tnd28_prefill_normal').replace(
        'DEMONCAT_FAULT28_FIA_TYPED_H', 'DEMONCAT_TND28_PREFILL_TYPED_H')
    additions = {base.HOST+'tnd28_prefill_normal.h': Path(__file__).with_name('tnd28_prefill_normal.h').read_bytes(),
                 base.HOST+'tnd28_prefill_contract.h': Path(__file__).with_name('tnd28_prefill_contract.h').read_bytes(),
                 base.HOST+'tnd28_prefill_typed.h': typed.encode()}
    if any((source/p).exists() for p in additions):
        raise ValueError('TND28_NORMAL_REPEATED_PATCH_REFUSED')
    changes = {p: s.encode() for p, s in updated.items()}
    changes.update(additions)
    materials = dict(raw, **changes)
    patch = ''.join(''.join(difflib.unified_diff(original[p].splitlines(True), updated[p].splitlines(True),
                    fromfile='a/'+p, tofile='b/'+p)) for p in PATHS)
    receipt = dict(schema='demoncat.attention28.tnd_prefill_normal_source.v1',
        mechanism='28.fia.normal_candidate', variant=VARIANT, vendor=VENDOR, source_commit=head,
        status='PATCHED_NOT_BUILT_NOT_NPU_VALIDATED', normal_only=True,
        scope=dict(layout='TND', batch=1, s1=2048, s2=2048, heads=16, kv_heads=8,
                   dim=128, page_attention=False, api_block_size=128, m=512, s2_base=512),
        normal_key=str(NORMAL_KEY), compile_keys=[str(k) for k in KEYS],
        tensor_payload_mutated=False, kernel_source_modified=False, tiling_arithmetic_modified=False,
        normal_route_preference_changed=True, public_prepared_pointer_stride_join=False,
        fault_execution_ready=False, stock_private_exact_proven=False, loading_allowed=False,
        c1=False, c2=False, c3=False,
        recipe_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        base_recipe_sha256=hashlib.sha256(Path(base.__file__).read_bytes()).hexdigest(),
        projection_sha256=hashlib.sha256(projection.encode()).hexdigest(), typed_field_count=len(fields),
        files=[dict(path=p, before_sha256=hashlib.sha256(raw[p]).hexdigest() if p in raw else None,
                    after_sha256=hashlib.sha256(v).hexdigest(), bytes=len(v)) for p, v in materials.items()])
    output.mkdir(parents=True)
    written = []
    try:
        for name, value in changes.items():
            (source/name).write_bytes(value)
            written.append(name)
        (output/'official_sources.patch').write_text(patch, encoding='utf-8')
        (output/'fia_schema_projection.h').write_text(projection, encoding='utf-8')
        (output/'typed_fields.json').write_text(json.dumps(fields, indent=2)+'\n', encoding='utf-8')
    except BaseException as error:
        receipt.update(status='PATCH_FAILED_SOURCE_RETAINED_FOR_REVIEW', written_files=written,
                       failure=type(error).__name__+':'+str(error))
        raise
    finally:
        (output/'policy_source_manifest.json').write_text(json.dumps(receipt, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(dict(status=receipt['status'], variant=VARIANT, changed_files=len(changes),
                          normal_key=str(NORMAL_KEY), loading_allowed=False)))


if __name__ == '__main__':
    main()
