"""Copy one verified fresh build into a separate immutable deployment bundle."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys


def main():
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True)
    p.add_argument('--build',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--kind',choices=['20','26'],required=True);a=p.parse_args()
    runtime='libdcat20_qmm.so' if a.kind=='20' else 'libfault26_qmm_output_underflow_abi0.so'
    paths=list(a.build.glob('cache/*/attempt-*/native/'+runtime))
    if len(paths)!=1:raise RuntimeError('AMBIGUOUS_NATIVE_BUILD')
    receipt=paths[0].parent.parent/'receipt.json'
    if not receipt.is_file():raise RuntimeError('MISSING_BUILD_RECEIPT')
    shutil.copytree(paths[0].parent,a.out)
    files={p.relative_to(a.out).as_posix():hashlib.sha256(p.read_bytes()).hexdigest()
           for p in a.out.rglob('*') if p.is_file()}
    if a.kind=='20':
        subprocess.run([sys.executable,str(a.source/'experiments/worker_positive_20261117/make_fault20_review.py'),
                        '--build',str(a.out)],check=True)
    else:
        (a.out/'build.json').write_text(json.dumps(dict(schema='demoncat.writer26.deployment.v1',
            source_build_receipt=str(receipt),source_build_receipt_sha256=hashlib.sha256(receipt.read_bytes()).hexdigest(),
            files=files,actual_npu_calls=0),indent=2)+'\n')
    print(json.dumps(dict(runtime=str(a.out/runtime),sha256=files[runtime],source_build_receipt=str(receipt))))


if __name__=='__main__':main()
