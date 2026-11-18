"""Prepare a fresh, normal-only arch22 bridge candidate from pinned c77 source.

The observed stock entry belongs to arch22 FIA. c77's CheckSpecConditions
instead prefers SplitFuse for the same TND/page16 input. This patch changes
only that narrow route preference and adds an owner-scoped serializer capture.
It does not change a tiling factor, kernel source, tensor, or fault admission.
An actual exact stock/private bridge is still required after a fresh build.
"""
import argparse
import difflib
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess

spec = importlib.util.spec_from_file_location('arch22_bridge_base', Path(__file__).with_name('base_policy.py'))
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)
VARIANT = 'tnd_page16_arch22_normal_bridge'
VENDOR = 'dcat_f33_arch22_normal'
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
        '36ab80c447bb001e658f7eb2bf561a04e97b1d2185f9136a676c98da52f79200'})
# Normal unscoped prefill and out-of-scope decode remain the original c77 FAI
# paths. Their normal entries must coexist with the arch22 decode entries.
KEYS = (103000000010200003, 103000000010300003,
        104000000010200003, 104000000010300003,
        5000000000000200100, 5000000000000200103,
        5000000000010200100, 5000000000010200103,
        5100000000010200100, 5100000000010200103)
ROUTE_ANCHOR = '        if (CheckSpecConditions(context)) {\n            return false;\n        }'


def patched(original):
    """All arithmetic and original parser validation remain byte-preserved."""
    header, implementation, route = (original[path] for path in PATHS)
    header = base.once(header, 'namespace optiling {',
        '#include "paged33_arch22_bridge.h"\n\nnamespace optiling {')
    implementation = base.once(implementation, '    return ge::GRAPH_SUCCESS;\n}\n\n// ',
        '    paged33_arch22_bridge::Capture(*fiaInfo_, context_, tilingData_, workspaceSize_, sInnerSize_);\n'
        '    return ge::GRAPH_SUCCESS;\n}\n\n// ')
    route = base.once(route, ROUTE_ANCHOR,
        '        if (CheckSpecConditions(context) && !paged33_arch22_bridge::Matches(context)) {\n'
        '            return false;\n        }')
    return dict(zip(PATHS, (header, implementation, route)))


def verify_source(raw):
    if set(raw) != set(SOURCE_SHA) or any(hashlib.sha256(raw[name]).hexdigest() != digest
                                        for name, digest in SOURCE_SHA.items()):
        raise RuntimeError('ARCH22_BRIDGE_REVIEWED_PRISTINE_SOURCE_DIFFERS')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    source, output = args.source.resolve(strict=True), args.output.resolve()
    head = subprocess.check_output(['git', '-C', str(source), 'rev-parse', 'HEAD'], text=True).strip()
    dirty = subprocess.check_output(['git', '-C', str(source), 'status', '--porcelain', '--untracked-files=all'], text=True)
    if head != base.COMMIT or dirty.strip() or output.exists() or source == output or source in output.parents:
        raise RuntimeError('PRISTINE_PINNED_SOURCE_AND_FRESH_EXTERNAL_RECEIPT_REQUIRED')
    raw = {name: (source/name).read_bytes() for name in SOURCE_SHA}
    verify_source(raw)
    original = {name: value.decode('utf-8') for name, value in raw.items()}
    updated = patched(original)
    typed, projection, fields = base.typed_materials(
        (source/(base.FIA+'op_host/fused_infer_attention_score_tiling.h')).read_text(encoding='utf-8'))
    typed = typed.replace('fault28_fia', 'paged33_arch22_bridge').replace(
        'DEMONCAT_FAULT28_FIA_TYPED_H', 'DEMONCAT_PAGED33_ARCH22_TYPED_H')
    additions = {base.HOST+'paged33_arch22_bridge.h': Path(__file__).with_name('paged33_arch22_bridge.h').read_bytes(),
                 base.HOST+'paged33_arch22_typed.h': typed.encode()}
    if any((source/name).exists() for name in additions):
        raise RuntimeError('ARCH22_BRIDGE_REPEATED_PATCH_REFUSED')
    materials = dict(raw)
    materials.update({name: value.encode() for name, value in updated.items()})
    materials.update(additions)
    output.mkdir(parents=True)
    for name, value in materials.items():
        (source/name).write_bytes(value)
    patch = ''.join(''.join(difflib.unified_diff(original[name].splitlines(True), updated[name].splitlines(True),
        fromfile='a/'+name, tofile='b/'+name)) for name in PATHS)
    (output/'official_sources.patch').write_text(patch, encoding='utf-8')
    (output/'fia_schema_projection.h').write_text(projection, encoding='utf-8')
    (output/'typed_fields.json').write_text(json.dumps(fields, indent=2)+'\n', encoding='utf-8')
    receipt = dict(schema='demoncat.paged33.arch22_normal_bridge_source.v1', mechanism='33.paged_decode',
        variant=VARIANT, source_commit=head, status='PATCHED_NOT_BUILT_NOT_NPU_VALIDATED',
        scope=dict(layout='TND', batch=1, query_tokens=1, heads=16, kv_heads=8, dim=128,
                   page_size=16, kv_length=[1025, 2048], route='FiaTilingNonQuant'),
        normal_route_preference_changed=True, tensor_payload_mutated=False, kernel_source_modified=False,
        tiling_arithmetic_modified=False, fault_execution_ready=False, stock_private_exact_proven=False,
        loading_allowed=False, actual_launch_verified=False,
        recipe_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        files=[dict(path=name, before_sha256=hashlib.sha256(raw[name]).hexdigest() if name in raw else None,
                    after_sha256=hashlib.sha256(value).hexdigest(), bytes=len(value)) for name, value in materials.items()])
    (output/'policy_source_manifest.json').write_text(json.dumps(receipt, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(receipt, indent=2))


if __name__ == '__main__':
    main()
