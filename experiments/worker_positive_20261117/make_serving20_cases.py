"""Use the same host-driven acceptance harness with Fault20's own native path."""
import argparse
import hashlib
import json
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('--base', required=True)
parser.add_argument('--out', required=True)
parser.add_argument('--native', required=True)
parser.add_argument('--statistics', action='store_true')
parser.add_argument('--baseline-control', action='store_true')
parser.add_argument('--model-path', default='/home/gyl/models/Qwen3-0.6B')
parser.add_argument('--model-label', default='qwen')
parser.add_argument('--operator', choices=['aclnnMatmulWeightNz','aclnnQuantMatmulWeightNz'], default='aclnnMatmulWeightNz')
parser.add_argument('--memory', type=float, default=.35)
parser.add_argument('--mapped-libraries', type=Path)
parser.add_argument('--weight-nz-output', action='store_true')
args = parser.parse_args()
root = Path('/home/worker-positive-20261117')
for label in (args.base, args.out, args.native, args.model_label):
    if not label.replace('-', '').replace('.', '').isalnum():
        raise ValueError('Invalid label')
model_path = Path(args.model_path).resolve(strict=True)
if not model_path.is_dir():
    raise ValueError('Existing pretrained model directory required')
case = json.loads((root / args.base).read_text())[0]
case['id'] = 'fault20-serving-' + args.model_label + ('-statistics' if args.statistics else '')
case['env'].update(VLLM_ASCEND_ENABLE_NZ='2', DEMONCAT_F20_REQUIRE_TYPED_GWS='1',
    DEMONCAT_F20_DIAGNOSTICS_PATH='/evidence/current-call.jsonl')
for mount in case['mounts']:
    if mount[1] == '/native': mount[0] = str(root / args.native / 'build20')
    elif mount[1].endswith('/libopapi_nn.so'): mount[0] = str(root / args.native / 'build20/libopapi_nn.so')
    elif mount[1] == '/probe.py': mount[0] = str(root / 'serving_probe.py')
    elif mount[1] == '/model': mount[0] = str(model_path)
case['mounts'].append([str(root / args.native / 'build20/libopapi_math.so'),
    '/usr/local/Ascend/cann-8.5.1/aarch64-linux/lib64/libopapi_math.so', 'ro'])
if args.mapped_libraries:
    snapshot = json.loads(args.mapped_libraries.read_text())
    if snapshot['schema'] != 'demoncat.private_mapped_libraries.v1':
        raise ValueError('Reviewed library snapshot required')
    targets = {mount[1] for mount in case['mounts']}
    for row in snapshot['libraries']:
        path = Path(row['source'])
        if row['destination'] in targets or path.is_symlink() or not path.is_file():
            raise ValueError('Library mount conflicts or source changed')
        h = hashlib.sha256()
        with path.open('rb') as stream:
            for chunk in iter(lambda: stream.read(8*1024*1024), b''):
                h.update(chunk)
        if h.hexdigest() != row['sha256']:
            raise ValueError('Library snapshot bytes changed')
        targets.add(row['destination'])
        case['mounts'].append([str(path), row['destination'], 'ro'])
case['command'] = case['command'].replace('libfault39_native_v1.so','libdcat20_qmm.so')
case['command'] = case['command'].replace(' --statistics','').replace(' --baseline-control','')
case['command'] += ' --mechanism 20.buffer_substitute --operator ' + args.operator + ' --memory ' + str(args.memory)
if args.statistics: case['command'] += ' --statistics'
if args.baseline_control: case['command'] += ' --baseline-control'
if args.weight_nz_output: case['command'] += ' --weight-nz-output'
with (root / args.out).open('x') as stream:
    json.dump([case], stream, indent=2)
