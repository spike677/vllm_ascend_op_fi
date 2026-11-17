"""Narrow 8.5.1 OpDef registration adaptation for an Ascend910B-only build.

The newer pinned source registers two SoCs unknown to the actual 8.5.1 opbuild.
Remove only those registrations in a fresh source clone. Target tiling, kernels,
and the Ascend910B registration remain byte-identical. This is not qualification.
"""
import hashlib
from pathlib import Path
import re
import subprocess


COMMIT = 'c77c0829f1f0f0b8f882db1d9b051a758b605472'
TARGETS = {
    'fused_infer_attention_score': ('mc62', 'ascend950'),
    'incre_flash_attention': ('mc62',),
    'prompt_flash_attention': ('mc62',),
}


def adapt(source):
    source = Path(source).resolve(strict=True)
    head = subprocess.check_output(['git', '-C', str(source), 'rev-parse', 'HEAD'], text=True).strip()
    if head != COMMIT:
        raise RuntimeError('OPDEF_COMPAT_PINNED_SOURCE_REQUIRED')
    planned = []
    for operator, removed in TARGETS.items():
        relative = f'attention/{operator}/op_host/{operator}_def.cpp'
        path = source / relative
        expected = subprocess.check_output(['git', '-C', str(source), 'show', COMMIT + ':' + relative])
        raw = path.read_bytes()
        if path.is_symlink() or raw != expected:
            raise RuntimeError('OPDEF_COMPAT_PRISTINE_DEFINITION_REQUIRED:' + relative)
        target = re.findall(rb'^.*AddConfig\("ascend910b"[^\r\n]*', raw, flags=re.M)
        if len(target) != 1:
            raise RuntimeError('OPDEF_COMPAT_TARGET_REGISTRATION_AMBIGUOUS')
        changed = raw
        for soc in removed:
            pattern = rb'^.*this->AICore\(\)\.AddConfig\("' + soc.encode() + rb'", [^\r\n]*\r?\n'
            changed, count = re.subn(pattern, b'', changed, flags=re.M)
            if count != 1:
                raise RuntimeError('OPDEF_COMPAT_UNEXPECTED_SOC_REGISTRATION:' + soc)
        if re.findall(rb'^.*AddConfig\("ascend910b"[^\r\n]*', changed, flags=re.M) != target:
            raise RuntimeError('OPDEF_COMPAT_TARGET_REGISTRATION_CHANGED')
        planned.append((path, changed, dict(path=relative, removed_soc_registrations=list(removed),
            before_sha256=hashlib.sha256(raw).hexdigest(), after_sha256=hashlib.sha256(changed).hexdigest())))
    # Validate all originals before the first write; no partial compatibility patch.
    for path, changed, _ in planned:
        path.write_bytes(changed)
    return dict(schema='demoncat.opdef.compat.v1', mode='ascend910b-cann851',
        source_commit=COMMIT, target_soc='ascend910b', actual_npu_calls=0,
        target_registration_unchanged=True, files=[row for _, _, row in planned])
