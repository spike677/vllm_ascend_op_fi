"""Private identical-byte bind mounts avoid Docker overlay maps/stat dev mismatch.

No NPU devices, no library imports, no host toolkit edits. Files come from the
same pinned acceptance image; runtime identity gates remain unchanged.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path('/home/worker-positive-20261117')
IMAGE = 'quay.io/ascend/vllm-ascend:v0.18.0'
IMAGE_ID = 'sha256:35f9e569dbe3dc8db90cb361886d7889489efd8fa065860c8de319e13ab025d0'


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


parser = argparse.ArgumentParser()
parser.add_argument('--environment', type=Path, required=True)
parser.add_argument('--native', type=Path, required=True)
parser.add_argument('--out', required=True)
args = parser.parse_args()
if not args.out.replace('-', '').isalnum():
    raise ValueError('Invalid snapshot label')
image = json.loads(subprocess.check_output(['docker', 'image', 'inspect', IMAGE]))[0]
if image['Id'] != IMAGE_ID:
    raise RuntimeError('PINNED_IMAGE_CHANGED')
environment = json.loads(args.environment.read_text())
profile = json.loads((args.native / 'version_profile.json').read_text())
output = ROOT / args.out
output.mkdir()
allowed = ('/usr/local/Ascend/cann-8.5.1/', '/usr/local/python3.11.14/lib/python3.11/site-packages/torch_npu/',
           '/vllm-workspace/vllm-ascend/vllm_ascend/_cann_ops_custom/')
rows = [row for row in environment['loaded_libraries'] if row['status'] == 'MAPPED_FILE_IDENTITY_CHANGED']
if not 1 <= len(rows) <= 16 or len({r['path'] for r in rows}) != len(rows):
    raise RuntimeError('BOUNDED_DISTINCT_OVERLAY_INPUTS_REQUIRED')
container = subprocess.check_output(['docker', 'create', '--runtime=runc', '--network=none',
    '--label', 'dcat.task=worker-positive-20261117', '-e', 'ASCEND_VISIBLE_DEVICES=void',
    '--entrypoint', '/bin/true', IMAGE_ID], text=True).strip()
manifest = dict(schema='demoncat.private_mapped_libraries.v1', image_id=IMAGE_ID,
    source_environment_sha256=digest(args.environment), actual_npu_calls=0,
    reason='OVERLAY_MAPS_DEVICE_DIFFERS_FROM_STAT_DEVICE', libraries=[])
try:
    for index, row in enumerate(rows):
        name, source = row['name'], row['path']
        if not source.startswith(allowed) or Path(source).name != name or '..' in Path(source).parts:
            raise RuntimeError('UNREVIEWED_IMAGE_LIBRARY_PATH')
        destination = output / (str(index) + '-' + name)
        subprocess.run(['docker', 'cp', '-L', container + ':' + source, str(destination)], check=True,
                       stdout=subprocess.DEVNULL)
        if not destination.is_file() or destination.is_symlink() or destination.stat().st_size > 2 * 1024**3:
            raise RuntimeError('EXPECTED_BOUNDED_REGULAR_LIBRARY')
        sha = digest(destination)
        expected = None
        if name in ('libnnopbase.so', 'libascendcl.so'):
            expected = profile['libraries']['nn' if name == 'libnnopbase.so' else 'acl']['file_sha256']
        elif name == 'libtorch_npu.so':
            expected = profile['controlled_pta_cache']['resolver_wheel']['file_sha256']
        if expected is not None and expected != sha:
            raise RuntimeError('BUILD_LIBRARY_CONTENT_MISMATCH:' + name)
        manifest['libraries'].append(dict(name=name, source=str(destination), destination=source,
            bytes=destination.stat().st_size, sha256=sha, profile_sha256=expected))
finally:
    subprocess.run(['docker', 'rm', container], check=True, stdout=subprocess.DEVNULL)
manifest['status'] = 'SNAPSHOT_ONLY_REQUIRES_ACTUAL_MAPPED_IDENTITY'
(output / 'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
print(json.dumps(dict(status=manifest['status'], files=len(rows),
    total_bytes=sum(row['bytes'] for row in manifest['libraries']))))
