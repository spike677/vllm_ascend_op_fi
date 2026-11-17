"""Fresh native FAI flash-decode coverage and matched KV256 specialization.

The actual pinned decode decision is preserved byte-for-byte. Neither 16 cores
nor FAInferContext's unused max-length members are substituted for live input.
"""
import argparse
import difflib
import hashlib
import json
from pathlib import Path
import subprocess

import apply_tnd33_fai_policy as previous
from base_policy import COMMIT, FIA, once

VARIANT = 'tnd_page16_fai_fd_s2_256'
NATIVE_KEYS = (*previous.NATIVE_KEYS, 5100000000010200100, 5100000000010200103)
FAULT_KEYS = (6000000000010200100, 6000000000010200103)
KEYS = (*NATIVE_KEYS, *FAULT_KEYS)
PATHS = previous.PATHS
SOURCE_SHA = dict(zip(PATHS, (
    'ab74f1603076d55e7dcdf969c83a221a57e87c03e1504dcafe85df1090e1b121',
    '3ae7ba8858ad13e259a861d2df19e69467ab2263c5e7d12332707dd8a41e0419',
    'c8666a28c93cfe2b08c0a34c1c851bf0f70c8a68bd650cba9b06b2dc65c0a205',
    'd34068f64e515b4c8ff75f12a2222f22cf9a397fb4e5812dd3dea1c5c573f787')))


def native_decode_decision(text):
    start = text.index('static void ComputeSeqLenAndDecodeFlagsForFAI(')
    end = text.index('\nstatic ge::graphStatus ConvertContextToParamsFAI', start)
    return text[start:end]


def patched(original):
    # Reuse only the audited host tile/workspace plumbing. Replace the previous
    # non-FD device specialization with the exact native FD=true templates.
    result = previous.patched(original)
    for name in (PATHS[1], PATHS[2]):
        for old, new in zip(previous.FAULT_KEYS, FAULT_KEYS):
            result[name] = result[name].replace(str(old), str(new))
    text = original[PATHS[3]]
    for suffix, key in zip(('NOMASK', 'CAUSALMASK'), FAULT_KEYS):
        symbol = 'QF16_KVF16_OUTF16_NOLSEOUT_TND_PAGEDCACHE_' + suffix + '_SPLITFUSE_TILING_FD'
        text = once(text, '        TILING_KEY_IS(' + symbol + ');',
                    f'        TILING_KEY_IS({key});\n        TILING_KEY_IS(' + symbol + ');')
        text = once(text, '#elif TILING_KEY_VAR == ' + symbol + '\n',
                    '#elif TILING_KEY_VAR == ' + symbol + f' || TILING_KEY_VAR == {key}ULL\n')
    result[PATHS[3]] = text
    if native_decode_decision(original[PATHS[1]]) != native_decode_decision(result[PATHS[1]]):
        raise RuntimeError('NATIVE_DECODE_DECISION_CHANGED')
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, required=True);p.add_argument('--output', type=Path, required=True)
    a = p.parse_args();source, out = a.source.resolve(strict=True), a.output.resolve()
    head = subprocess.check_output(['git', '-C', str(source), 'rev-parse', 'HEAD'], text=True).strip()
    dirty = subprocess.check_output(['git', '-C', str(source), 'status', '--porcelain', '--untracked-files=all'], text=True)
    if head != COMMIT or dirty.strip() or out.exists() or source == out or source in out.parents:
        raise RuntimeError('PRISTINE_PINNED_SOURCE_AND_FRESH_EXTERNAL_RECEIPT_REQUIRED')
    raw = {name: (source / name).read_bytes() for name in PATHS}
    if any(hashlib.sha256(value).hexdigest() != SOURCE_SHA[name] for name, value in raw.items()):
        raise RuntimeError('ACTUAL_FAI_FD_SOURCE_DIFFERS')
    original = {name: value.decode() for name, value in raw.items()};updated = patched(original)
    typed, projection, fields = previous.typed_materials(original[PATHS[0]])
    additions = {FIA + 'op_host/paged33_fai_policy.h': Path(__file__).with_name('paged33_fai_fd_policy.h').read_bytes(),
                 FIA + 'op_host/paged33_fai_typed.h': typed.encode()}
    if any((source / name).exists() for name in additions):raise RuntimeError('REPEATED_FAI_PATCH_REFUSED')
    materials = {name: value.encode() for name, value in updated.items()};materials.update(additions)
    out.mkdir(parents=True)
    for name, value in materials.items():(source / name).write_bytes(value)
    diff = ''.join(''.join(difflib.unified_diff(original[name].splitlines(True), updated[name].splitlines(True),
        fromfile='a/' + name, tofile='b/' + name)) for name in PATHS)
    (out / 'official_sources.patch').write_text(diff)
    (out / 'fai_schema_projection.h').write_text(projection)
    (out / 'typed_fields.json').write_text(json.dumps(fields, indent=2) + '\n')
    receipt = dict(schema='demoncat.paged33.fai_fd_policy_source.v1', mechanism='33.paged_decode',
        variant=VARIANT, source_commit=head, status='PATCHED_NOT_BUILT_NOT_NPU_VALIDATED',
        route='TilingProcess4SplitFuse / FAInferTiling / native flash decode',
        native_route_or_decode_flags_changed=False,
        native_decode_decision_sha256=hashlib.sha256(native_decode_decision(original[PATHS[1]]).encode()).hexdigest(),
        scope=dict(layout='TND', page_size=16, batch=1, hq=16, hkv=8, dim=128, kv_length=[1025, 2048],
            aic_supported_bounds=[20, 25], actual_aic='REQUIRES_LIVE_HOST_OBSERVATION',
            flash_decode=True, length_source='actual sequence lists; native context max fields preserved',
            native_s2=512, fault_s2=256), native_keys=list(NATIVE_KEYS), fault_keys=list(FAULT_KEYS),
        tensor_payload_mutated=False, kernel_source_modified=True,
        old_bnsd_or_arch22_or_non_fd_qualification_reused=False,
        loading_allowed=False, actual_launch_verified=False,
        files=[dict(path=name, before_sha256=hashlib.sha256(raw[name]).hexdigest() if name in raw else None,
            after_sha256=hashlib.sha256(value).hexdigest(), bytes=len(value)) for name, value in materials.items()])
    (out / 'policy_source_manifest.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps(receipt, indent=2))


if __name__ == '__main__':main()
