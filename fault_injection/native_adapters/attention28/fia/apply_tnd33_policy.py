"""Patch a new pristine c77 checkout for TND/page16 native S2 selection.

This is an independent Fault33 implementation. It does not import the old BNSD
IFA policy or certificate and changes no input/cache/output tensor bytes.
"""
import argparse
import difflib
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess

spec = importlib.util.spec_from_file_location('tnd33_base_schema', Path(__file__).with_name('base_policy.py'))
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)
HOST = base.HOST
PATHS = (HOST+'fia_tiling_nonquant.h', HOST+'fia_tiling_nonquant.cpp')


def patched(original):
    header, impl = (original[x] for x in PATHS)
    header = base.once(header, 'namespace optiling {', '#include "paged33_tnd_policy.h"\n\nnamespace optiling {')
    header = base.once(header, '    bool splitKVFlag_ = false;',
                       '    paged33_tnd::State paged33State_;\n    bool splitKVFlag_ = false;')
    # Select before native loop/tail, alignment, M, split/workspace and serializer
    # calculations. Smaller S2 preserves the original M and UB upper bound.
    anchor = '    sInnerLoopTimes_ = (s2Size + sInnerSize_ - static_cast<uint32_t>(1)) / sInnerSize_;'
    impl = base.once(impl, anchor,
        '    paged33State_ = paged33_tnd::Select(*fiaInfo_, context_, s2Size, sInnerSize_);\n'
        '    sInnerSize_ = paged33State_.selectedS2;\n'+anchor)
    anchor = '    return ge::GRAPH_SUCCESS;\n}\n\n// '
    impl = base.once(impl, anchor,
        '    paged33_tnd::Capture(paged33State_, context_, tilingData_, workspaceSize_);\n'+anchor)
    return dict(zip(PATHS, (header, impl)))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    source, out = a.source.resolve(strict=True), a.output.resolve()
    head = subprocess.check_output(['git', '-C', str(source), 'rev-parse', 'HEAD'], text=True).strip()
    dirty = subprocess.check_output(['git', '-C', str(source), 'status', '--porcelain', '--untracked-files=all'], text=True)
    if head != base.COMMIT or dirty.strip() or out.exists() or source == out or source in out.parents:
        raise RuntimeError('PRISTINE_PINNED_SOURCE_AND_FRESH_EXTERNAL_RECEIPT_REQUIRED')
    raw = {name: (source/name).read_bytes() for name in PATHS}
    original = {name: value.decode() for name, value in raw.items()}
    updated = patched(original)
    typed, projection, fields = base.typed_materials((source/(base.FIA+'op_host/fused_infer_attention_score_tiling.h')).read_text())
    typed = typed.replace('fault28_fia', 'paged33_tnd').replace('DEMONCAT_FAULT28_FIA_TYPED_H', 'DEMONCAT_PAGED33_TND_TYPED_H')
    additions = {HOST+'paged33_tnd_policy.h': Path(__file__).with_name('paged33_tnd_policy.h').read_bytes(),
                 HOST+'paged33_tnd_typed.h': typed.encode()}
    if any((source/name).exists() for name in additions):
        raise RuntimeError('REPEATED_TND33_PATCH_REFUSED')
    materials = {name: value.encode() for name, value in updated.items()}
    materials.update(additions)
    for name, value in materials.items():
        (source/name).write_bytes(value)
    out.mkdir(parents=True)
    patch = ''.join(''.join(difflib.unified_diff(original[name].splitlines(True), updated[name].splitlines(True),
                    fromfile='a/'+name, tofile='b/'+name)) for name in PATHS)
    (out/'official_sources.patch').write_text(patch)
    (out/'fia_schema_projection.h').write_text(projection)
    (out/'typed_fields.json').write_text(json.dumps(fields, indent=2)+'\n')
    receipt = dict(schema='demoncat.paged33.tnd_policy_source.v1', mechanism='33.paged_decode',
        variant='tnd_page16_s2_256', source_commit=head, status='PATCHED_NOT_BUILT_NOT_NPU_VALIDATED',
        scope=dict(api='npu_fused_infer_attention_score', query_layout='TND', page_size=16,
                   batch=1, hq=16, hkv=8, dim=128, kv_length=[1025, 2048], native_s2=512, fault_s2=256),
        tensor_payload_mutated=False, kernel_source_modified=False, m_policy_changed=False,
        old_bnsd_certificate_reused=False, loading_allowed=False, actual_launch_verified=False,
        files=[dict(path=name, before_sha256=hashlib.sha256(raw[name]).hexdigest() if name in raw else None,
                    after_sha256=hashlib.sha256(value).hexdigest(), bytes=len(value)) for name, value in materials.items()])
    (out/'policy_source_manifest.json').write_text(json.dumps(receipt, indent=2)+'\n')
    print(json.dumps(receipt, indent=2))


if __name__ == '__main__':
    main()
