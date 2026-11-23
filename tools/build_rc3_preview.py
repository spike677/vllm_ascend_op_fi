"""Build from a committed source tree into a new, never-overwritten release."""
import argparse,hashlib,json,os,subprocess,sys,tarfile,tempfile
from pathlib import Path

VERSIONS={'demoncatpfi':'0.4.0rc5','demoncat-worker-extension':'0.3.0rc5','demoncat-fault39-core':'0.3.0rc5'}
ROOTS=('src/','fault_injection/','fault39/src/','native_execution_strategy/','engineering_closure/',
       'tools/','tests/','docs/worker_core_v1/')
EXACT=('pyproject.toml','fault39/pyproject.toml','README.md',
       'releases/RC3-PREVIEW-R5-EXTRACT-TOOL-20261123.md',
       'experiments/worker_core_v11_20261118/continuous_public_probe.py',
       'experiments/worker_core_v11_20261118/product_http_probe.py',
       'experiments/worker_core_v11_20261118/product_mode_probe.py',
       'experiments/worker_core_v11_20261118/paged33_launch_capture.cpp',
       'experiments/worker_core_v11_20261118/paged33_native_probe.py',
       'experiments/worker_core_v11_20261118/continuous_matmul39_source_review_20261122.json')
SUFFIXES={'.py','.json','.toml','.md','.cpp','.h','.inc','.sh','.yaml','.yml'}


def sha(data):return hashlib.sha256(data).hexdigest()


def write_json_bytes(path,value):
    # Explicit bytes: Windows text-mode CRLF conversion invalidates Git hashes.
    Path(path).write_bytes((json.dumps(value,indent=2)+'\n').encode('utf-8'))


def build(repo,output):
    repo=Path(repo).resolve();output=Path(output).resolve()
    if output.exists():raise RuntimeError('RELEASE_OUTPUT_ALREADY_EXISTS')
    def git(*args):return subprocess.check_output(['git','-C',str(repo),*args])
    commit=git('rev-parse','HEAD').decode().strip()
    paths=[p for p in git('ls-files','-z').decode().split('\0') if p and
        (p in EXACT or p.startswith(ROOTS) or
         (p.startswith('experiments/worker_core_v11_20261118/') and ('_review_' in p or p.endswith('native_regions_build_20261122.json'))))
        and Path(p).suffix in SUFFIXES]
    # The source package intentionally excludes old releases, private evidence,
    # model tensors, credentials, external intake and all untracked scratch work.
    if not paths:raise RuntimeError('EMPTY_SOURCE_REVIEW')
    # Git normalizes configured CRLF; package bytes always come from the commit.
    # Avoid Windows command-line length limits by checking bounded path batches.
    for start in range(0,len(paths),40):
        dirty=subprocess.run(['git','-C',str(repo),'diff','--quiet',commit,'--',*paths[start:start+40]])
        if dirty.returncode:raise RuntimeError('UNCOMMITTED_RELEASE_SOURCE')
    output.mkdir(parents=True);wheels=output/'wheels';wheels.mkdir()
    (repo/'.codex-tmp').mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='rc3-preview-',dir=repo/'.codex-tmp') as temp:
        stage=Path(temp);files=[]
        for name in sorted(paths):
            raw=git('show',commit+':'+name);dest=stage/name
            dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(raw)
            files.append(dict(path=name,bytes=len(raw),sha256=sha(raw)))
        review=dict(schema='demoncat.source_review.v1',source_commit=commit,files=files,
            delivery_scope=['28.fia.denominator_replay','quant.batch_shared_scale','39.output_proxy','18.api_half','22.wrong_row','39.matmul_range','17.spatial_padding','41.tail_column','26.fia_destination','20.buffer_substitute','33.paged_decode'],
            product_candidates=['28.fia.denominator_replay','quant.batch_shared_scale','39.output_proxy','18.api_half','22.wrong_row','39.matmul_range','17.spatial_padding','33.paged_decode'],
            restrictions={'33.paged_decode':'A2 measured range only; A3 launch/control review pending',
                          '41/26fia/20':'Acceptance requires reviewed Native build; product adapter closed',
                          '8/DeepSeek':'New deployment and product acceptance pending'},qualification_granted=False)
        write_json_bytes(output/'source-review.json',review)
        archive=output/(output.name+'.tar.gz')
        with tarfile.open(archive,'w:gz') as tf:
            for row in files:tf.add(stage/row['path'],arcname=row['path'],recursive=False)
        env=dict(os.environ,PIP_NO_INDEX='1',PYTHONDONTWRITEBYTECODE='1')
        for package in (stage,stage/'src',stage/'fault39'):
            # Direct PEP517 backend avoids pip's deeply nested Windows cache.
            subprocess.run([sys.executable,'-c',
                'from setuptools.build_meta import build_wheel; import sys; build_wheel(sys.argv[1])',
                str(wheels)],cwd=package,env=env,check=True)
    if len(list(wheels.glob('*.whl')))!=3:raise RuntimeError('THREE_WHEELS_REQUIRED')
    manifest=dict(schema='demoncat.rc3_preview.release.v1',source_commit=commit,versions=VERSIONS,
        source_manifest_file='source-review.json',source_manifest_sha256=sha((output/'source-review.json').read_bytes()),
        target='YELLOW_DEDICATED_INSTANCE / A3 / Qwen3.8 W8A8 / TP1 and TP2 independently / acceptance pending',
        blue_results_are_not_a3_acceptance=True,files=[])
    for p in sorted(output.rglob('*')):
        if p.is_file():manifest['files'].append(dict(path=p.relative_to(output).as_posix(),bytes=p.stat().st_size,sha256=sha(p.read_bytes())))
    write_json_bytes(output/'release-manifest.json',manifest)
    sums=''.join(sha(p.read_bytes())+'  '+p.relative_to(output).as_posix()+'\n' for p in sorted(output.rglob('*')) if p.is_file())
    (output/'SHA256SUMS').write_bytes(sums.encode('utf-8'))
    print(json.dumps(dict(source_commit=commit,source_files=len(paths),output=str(output),source_manifest_sha256=manifest['source_manifest_sha256'])))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--repo',default='.');p.add_argument('--output',required=True);a=p.parse_args();build(a.repo,a.output)
