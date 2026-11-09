"""CPU-only fresh private stage; no external process or source is modified."""
import hashlib
import json
from pathlib import Path
import shutil

ROOT = Path('/home/fault28-fia-fullmodel-20261109')
PRIOR = Path('/home/fault28-fia-grain-20261109')
VENDOR_SHA = '205988e230b901421f14eb0844ce435b66647a5846324608d768d313a66f8cd2'
COLLECTOR_SHA = '06ce36bae3463f44a143047d184f53915dbea4dd04b39d79960518e6878d8010'
INPUT_SHA = 'ed6b27940c3336efe8cd941fda9da191edcbd8257c34587396345f24d7994fe1'


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    assert ROOT.is_dir() and not (ROOT/'vendor_stage_a5').exists()
    assert sha(PRIOR/'vendor_stage_a5/vendor_manifest.json') == VENDOR_SHA
    assert sha(PRIOR/'collector_fp64_a1/stage_manifest.json') == COLLECTOR_SHA
    for name in ('vendor_stage_a5', 'collector_fp64_a1'):
        shutil.copytree(PRIOR/name, ROOT/name, symlinks=True)
    (ROOT/'build').mkdir()
    shutil.copytree(PRIOR/'build/evidence_a4', ROOT/'build/evidence_a4')
    for name in ('device_grain_trace.json', 'device_grain_trace.patch'):
        shutil.copyfile(PRIOR/'build'/name, ROOT/'build'/name)
    source = Path('/home/fault28-m-only-model-20261109/inputs/manifest.json')
    assert sha(source) == INPUT_SHA
    (ROOT/'private').mkdir()
    shutil.copyfile(source, ROOT/'private/requests.json')
    print(json.dumps({'status':'PASS_CPU_STAGE_NOT_NPU_ACCEPTED',
        'vendor_sha256':sha(ROOT/'vendor_stage_a5/vendor_manifest.json'),
        'collector_sha256':sha(ROOT/'collector_fp64_a1/stage_manifest.json'),
        'requests_sha256':sha(ROOT/'private/requests.json'), 'npu_calls':0}))


if __name__ == '__main__':
    main()
