"""Download the explicitly authorized public GLM checkpoint; no NPU/imports.

Pins a Hub commit, verifies LFS SHA256/size, and never runs remote Python code.
Weights and partial files stay in the private 124 experiment directory.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import time
import urllib.request

REPO = 'zai-org/GLM-4-9B-0414'
ROOT = Path('/home/worker-positive-20261117/models/GLM-4-9B-0414')


def open_url(url, **kwargs):
    request = urllib.request.Request(url, headers={'User-Agent': 'DemonCATPFI-acceptance/1'}, **kwargs)
    return urllib.request.urlopen(request, timeout=45)


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for data in iter(lambda: stream.read(8*1024*1024), b''):
            digest.update(data)
    return digest.hexdigest()


def verify(path, item):
    if path.stat().st_size != item['size']:
        raise RuntimeError('FILE_SIZE_MISMATCH:'+item['rfilename'])
    digest = sha256(path)
    expected = item.get('lfs', {}).get('sha256')
    if expected:
        if digest != expected:
            raise RuntimeError('LFS_SHA256_MISMATCH:'+item['rfilename'])
    else:
        # Canonical Hub Git blob identity also covers config/tokenizer files.
        data = path.read_bytes()
        git_hash = hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest()
        if git_hash != item['blobId']:
            raise RuntimeError('GIT_BLOB_ID_MISMATCH:'+item['rfilename'])
    return digest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--metadata-only', action='store_true')
    parser.add_argument('--official-metadata', type=Path)
    parser.add_argument('--endpoint', choices=['https://huggingface.co','https://hf-mirror.com'],
                        default='https://huggingface.co')
    args = parser.parse_args()
    ROOT.mkdir(parents=True, exist_ok=True)
    manifest_path = ROOT/'download_manifest.json'
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
    else:
        if args.official_metadata:
            doc = json.loads(args.official_metadata.read_text())
        else:
            with open_url('https://huggingface.co/api/models/'+REPO+'?blobs=true') as stream:
                doc = json.load(stream)
        if doc.get('gated') or not re.fullmatch('[0-9a-f]{40}', doc['sha']):
            raise RuntimeError('PUBLIC_FIXED_COMMIT_REQUIRED')
        files = [f for f in doc['siblings'] if '/' not in f['rfilename'] and
                 (f['rfilename'].endswith(('.json', '.safetensors', '.model', '.tiktoken', '.jinja'))
                  or f['rfilename'] in ('README.md', 'LICENSE', 'LICENSE.txt', 'merges.txt', 'vocab.txt'))]
        if not 0 < len(files) <= 40 or not any(f['rfilename'].endswith('.safetensors') for f in files):
            raise RuntimeError('EXPECTED_BOUNDED_SAFETENSORS_CHECKPOINT')
        total = sum(f['size'] for f in files)
        if total > 30*1024**3 or shutil.disk_usage(ROOT).free < total+10*1024**3:
            raise RuntimeError('CHECKPOINT_STORAGE_BUDGET_EXCEEDED')
        manifest = dict(repository=REPO, revision=doc['sha'], files=files, total_bytes=total,
                        source='https://huggingface.co/'+REPO, remote_code_executed=False,
                        official_metadata_sha256=sha256(args.official_metadata) if args.official_metadata else None)
        manifest_path.write_text(json.dumps(manifest, indent=2)+'\n')
    print(json.dumps({k:manifest[k] for k in ('repository','revision','total_bytes')}), flush=True)
    if args.metadata_only:
        return
    receipt = dict(status='DOWNLOADING', revision=manifest['revision'], completed=[], endpoint=args.endpoint)
    for item in manifest['files']:
        name = item['rfilename']
        if Path(name).name != name or name.startswith('.'):
            raise RuntimeError('UNSAFE_MANIFEST_NAME')
        path = ROOT/name
        expected = item.get('lfs', {}).get('sha256')
        if path.exists():
            digest = verify(path, item)
        else:
            part = ROOT/(name+'.partial')
            # Restart a partial file only within this explicitly owned download.
            for attempt in range(3):
                try:
                    url = args.endpoint+'/'+REPO+'/resolve/'+manifest['revision']+'/'+name
                    with open_url(url) as source, part.open('wb') as target:
                        while data := source.read(8*1024*1024):
                            target.write(data)
                    digest = verify(part, item)
                    part.rename(path)
                    break
                except Exception:
                    if attempt == 2:
                        raise
                    time.sleep(2)
        receipt['completed'].append(dict(name=name, bytes=path.stat().st_size, sha256=digest,
                                         expected_lfs_sha256=expected))
        (ROOT/'download_receipt.json').write_text(json.dumps(receipt, indent=2)+'\n')
        print(json.dumps(receipt['completed'][-1]), flush=True)
    receipt['status']='COMPLETE_CHECKSUM_VERIFIED_NOT_NPU_ACCEPTANCE'
    (ROOT/'download_receipt.json').write_text(json.dumps(receipt, indent=2)+'\n')


if __name__ == '__main__':
    main()
