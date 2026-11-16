"""Reconstruct the exact historical read-only dependency bytes used on 124.

This does not import the legacy module or activate its old injector/collector.
The current provider uses only validate_runtime, capture_rows and digest.
Keep CRLF: deployment identity checks cover bytes, not normalized source text.
"""
import argparse
import hashlib
from pathlib import Path
import subprocess

COMMIT = '297e03d1a03b08b31ec82b26b1940d89d9e49478'
PREFIX = 'artifacts/quantization/batch_shared_scale_v2_bf_statistics_20261028/source/'
HASHES = {
    'backend_control.py': '5ce739c4272840f110044cfe898827fe403b5730273258e8ba0eda437685b38c',
    'model_hook.py': 'a9024f015e468074b1fab8dbc15124c2ab3a609ba946f572cd58a02fce9678ef',
}
DEPLOYED_HOOK_SHA = '33b650ef368de1f5e8beb771e64aa6e06f1c64f99746dcc07a3b151253b8cdd0'


def materialize(repo, output):
    contents = {}
    for name, expected in HASHES.items():
        raw = subprocess.check_output(['git', '-C', str(repo), 'show', COMMIT + ':' + PREFIX + name])
        if hashlib.sha256(raw).hexdigest() != expected:
            raise ValueError('Pinned dependency differs: ' + name)
        contents[name] = raw
    # Only this unused historical collector configuration differs from the
    # published reference. No read-only scheduler helper changes are involved.
    old = ("        config.write_text(json.dumps({'task':'statistics','level':'mix','rank':[],'step':[],\r\n"
           "            'dump_path':str(Path(os.environ['DCAT_V2_OUTPUT'])/'observation'),\r\n"
           "            'async_dump':False,'statistics':{'scope':scope,'list':[],'data_mode':['all'],\r\n"
           "                'summary_mode':'md5','tensor_list':['.']}}))\r\n").encode()
    new = ("        config.write_text(json.dumps({'task':'tensor','level':'mix','rank':[],'step':[],\r\n"
           "            'dump_path':str(Path(os.environ['DCAT_V2_OUTPUT'])/'observation'),\r\n"
           "            'async_dump':False,'tensor':{'scope':scope,'list':[],'data_mode':['all'],'summary_mode':'md5'}}))\r\n").encode()
    raw = contents['model_hook.py']
    if raw.count(old) != 1:
        raise ValueError('Historical source anchor differs')
    contents['model_hook.py'] = raw.replace(old, new)
    if hashlib.sha256(contents['model_hook.py']).hexdigest() != DEPLOYED_HOOK_SHA:
        raise ValueError('Deployed dependency identity differs')
    output.mkdir(parents=True, exist_ok=False)
    for name, raw in contents.items():
        with (output / name).open('xb') as stream:
            stream.write(raw)
    return {name: hashlib.sha256(raw).hexdigest() for name, raw in contents.items()}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    print(materialize(args.repo, args.output))
