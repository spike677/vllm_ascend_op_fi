"""Stage unchanged public image libraries for explicit read-only test mounts.

This does not change host libraries or start a device worker. A fresh worker
must still prove its actual loaded mappings and symbol bindings independently.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import uuid

IMAGE = 'sha256:35f9e569dbe3dc8db90cb361886d7889489efd8fa065860c8de319e13ab025d0'
FILES = [
    '/usr/local/Ascend/cann-8.5.1/aarch64-linux/lib64/' + name
    for name in ('libacl_rt.so', 'libascendcl.so', 'libnnopbase.so', 'libopapi_math.so', 'libopapi.so',
                 'libopapi_cv.so', 'libopapi_nn.so', 'libopapi_oam.so')
] + [
    '/usr/local/python3.11.14/lib/python3.11/site-packages/torch_npu/lib/libtorch_npu.so',
    '/vllm-workspace/vllm-ascend/vllm_ascend/_cann_ops_custom/vendors/vllm-ascend/op_api/lib/libcust_opapi.so',
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    owner = uuid.uuid4().hex
    script = '''import hashlib,json,pathlib,shutil,sys
rows=[]
for name in json.loads(sys.argv[1]):
 p=pathlib.Path(name); q=pathlib.Path('/output/files')/p.relative_to('/')
 q.parent.mkdir(parents=True,exist_ok=True)
 with p.open('rb') as f: before=hashlib.file_digest(f,'sha256').hexdigest()
 shutil.copyfile(p,q)
 with q.open('rb') as f: after=hashlib.file_digest(f,'sha256').hexdigest()
 with p.open('rb') as f: reread=hashlib.file_digest(f,'sha256').hexdigest()
 assert before==after==reread
 q.chmod(0o444)
 rows.append(dict(original_path=name,relative_stage_path=str(q.relative_to('/output')),sha256=before,bytes=q.stat().st_size))
pathlib.Path('/output/files.json').write_text(json.dumps(rows,indent=2))
'''
    command = ['docker', 'create', '--name', 'dcat-image-stage-' + owner,
        '--label', 'dcat.image-stage-owner=' + owner, '--runtime=runc', '--network=none',
        '--cpus=2', '--memory=2g', '-e', 'ASCEND_VISIBLE_DEVICES=void',
        '--mount', 'type=bind,src=' + str(out) + ',dst=/output',
        '--entrypoint', 'python3', IMAGE, '-B', '-c', script, json.dumps(FILES)]
    cid = subprocess.check_output(command, text=True, timeout=30).strip()
    def owned():
        row, = json.loads(subprocess.check_output(['docker', 'inspect', cid], timeout=30))
        host = row['HostConfig']
        if (row['Image'] != IMAGE or row['Config']['Labels'].get('dcat.image-stage-owner') != owner
                or host['Privileged'] or host['Runtime'] != 'runc' or host['NetworkMode'] != 'none'
                or host.get('Devices') or host.get('DeviceRequests')):
            raise RuntimeError('OWNED_DEVICE_FREE_STAGE_CHANGED')
        return row
    receipt = dict(schema='demoncat.unchanged_image_library_stage.v1', image=IMAGE,
                   source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                   actual_npu_calls=0, qualification_granted=False, command=command)
    try:
        owned()
        result = subprocess.run(['docker', 'start', '-a', cid], capture_output=True, timeout=180)
        (out / 'process.log').write_bytes(result.stdout + result.stderr)
        receipt['exit_code'] = result.returncode
        result.check_returncode()
        receipt['files'] = json.loads((out / 'files.json').read_bytes())
        for row in receipt['files']:
            path = out / row['relative_stage_path']
            with path.open('rb') as stream:
                if hashlib.file_digest(stream, 'sha256').hexdigest() != row['sha256']:
                    raise RuntimeError('HOST_STAGE_BYTES_CHANGED')
    finally:
        state = owned()['State']
        if state['Running']:
            subprocess.run(['docker', 'stop', '--time', '10', cid], check=True, timeout=30)
            state = owned()['State']
        receipt['final_state'] = state
        subprocess.run(['docker', 'rm', cid], check=True, timeout=30, stdout=subprocess.DEVNULL)
        receipt['owned_container_removed'] = True
        (out / 'receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps(dict(stage=str(out), files=len(receipt['files']), actual_npu_calls=0)))


if __name__ == '__main__':
    main()
