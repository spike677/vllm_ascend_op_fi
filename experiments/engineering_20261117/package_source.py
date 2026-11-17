"""Freeze selected reviewed source bytes for a fresh remote attempt, no weights/data."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import subprocess
import tarfile


def main():
    p=argparse.ArgumentParser(); p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    root=Path(__file__).resolve().parents[2]
    sources=['fault_injection','src/demoncat','fault39/native','fault39/tools','fault39/src',
             'native_execution_strategy','experiments/engineering_20261117','tests/native',
             'pyproject.toml','src/pyproject.toml','fault39/pyproject.toml',
             'experiments/worker_positive_20261117/build_packages_124.py',
             'experiments/worker_positive_20261117/make_fault20_review.py',
             'experiments/worker_positive_20261117/serving_probe.py',
             'experiments/provider_adaptation_20261116/capture.py',
             'fault20_fast_validation/source/l4_buffer_substitute_allhit_v1',
             'fault20_fast_validation/source/l4_metadata_lifecycle_proxy',
             'fault20_fast_validation/source/fault26_readonly_trace_v1',
             'fault20_fast_validation/source/fault26_fia_writer_proxy_v1',
             'fault20_fast_validation/source/fault26_qmm_writer_proxy_v1',
             'fault20_fast_validation/source/fault26_add_rms_norm_bias_proxy_v1']
    paths=[]
    for source in sources:
        location=root/source
        for path in ([location] if location.is_file() else location.rglob('*')):
            if path.is_file() and path.suffix in ('.py','.cpp','.h','.inc','.sh','.json','.toml') and '__pycache__' not in path.parts:
                paths.append(path)
    snapshot={path.relative_to(root).as_posix():path.read_bytes() for path in sorted(set(paths))}
    files=[dict(path=name,sha256=hashlib.sha256(raw).hexdigest(),bytes=len(raw)) for name,raw in snapshot.items()]
    manifest=dict(base_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip(),files=files,
                  contains_model_weights=False,contains_native_binaries=False,contains_tensor_payloads=False)
    a.out.parent.mkdir(parents=True,exist_ok=True)
    with a.out.open('xb') as stream, tarfile.open(fileobj=stream,mode='w:gz') as tar:
        for name,raw in snapshot.items():
            info=tarfile.TarInfo(name);info.size=len(raw);info.mode=0o644
            tar.addfile(info,io.BytesIO(raw))
        raw=(json.dumps(manifest,indent=2)+'\n').encode(); info=tarfile.TarInfo('source_manifest.json');info.size=len(raw)
        tar.addfile(info,io.BytesIO(raw))
    print(json.dumps(dict(path=str(a.out),files=len(files),bytes=a.out.stat().st_size,sha256=hashlib.sha256(a.out.read_bytes()).hexdigest())))


if __name__=='__main__':main()
