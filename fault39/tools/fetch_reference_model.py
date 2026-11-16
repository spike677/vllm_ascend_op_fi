"""Download a small public Llama-architecture reference, without loading code."""
import argparse
import hashlib
import json
from pathlib import Path
import urllib.request
import subprocess

REPO = 'HuggingFaceTB/SmolLM2-135M-Instruct'
REVISION = '12fd25f77366fa6b3b4b768ec3050bf629380bac'
FILES = ('config.json', 'generation_config.json', 'model.safetensors', 'tokenizer.json',
         'tokenizer_config.json', 'special_tokens_map.json', 'merges.txt', 'vocab.json')


def get(url):
    return urllib.request.urlopen(urllib.request.Request(url, headers={'User-Agent': 'fault39-blue-validation'}), timeout=30)


def main():
    p = argparse.ArgumentParser(); p.add_argument('--output', type=Path, required=True)
    p.add_argument('--mirror', choices=('huggingface.co', 'hf-mirror.com'), default='huggingface.co')
    a = p.parse_args()
    revision = REVISION
    with get('https://huggingface.co/api/models/' + REPO + '/tree/' + revision) as response:
        tree = {item['path']: item for item in json.load(response)}
    selected = [name for name in FILES if name in tree]
    if 'model.safetensors' not in selected or sum(tree[n].get('size', 0) for n in selected) > 512 * 1024**2:
        raise RuntimeError('REFERENCE_MODEL_SIZE_OR_LAYOUT_MISMATCH')
    a.output.mkdir(parents=True, exist_ok=True)
    manifest_path = a.output / 'download_manifest.json'
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text())
        if previous.get('revision') != revision or previous.get('source') != 'https://huggingface.co/' + REPO:
            raise RuntimeError('RESUME_SOURCE_OR_REVISION_MISMATCH')
    receipt = {'source': 'https://huggingface.co/' + REPO, 'revision': revision, 'files': {}}
    (a.output / 'download_manifest.json').write_text(json.dumps({'source': receipt['source'], 'revision': revision,
        'transport_host': a.mirror, 'files': {name: tree[name] for name in selected}}, indent=2))
    for name in selected:
        target = a.output / name
        temporary = target.with_suffix(target.suffix + '.part')
        if not target.exists():
            subprocess.run(['curl', '--fail', '--location', '--silent', '--show-error', '--connect-timeout', '15',
                        '--max-time', '180', '--speed-limit', '1024', '--speed-time', '30', '--max-filesize', str(512 * 1024**2),
                        '--continue-at', '-',
                        '--output', str(temporary), 'https://' + a.mirror + '/' + REPO + '/resolve/' + revision + '/' + name + '?download=true'],
                       check=True, timeout=190)
        candidate = target if target.exists() else temporary
        digest = hashlib.sha256(); size = candidate.stat().st_size
        with candidate.open('rb') as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b''):
                digest.update(block)
        actual = digest.hexdigest()
        expected = tree[name].get('lfs', {}).get('oid')
        if expected and expected != actual:
            raise RuntimeError('MODEL_LFS_SHA256_MISMATCH')
        if tree[name].get('size') != size:
            raise RuntimeError('MODEL_FILE_SIZE_MISMATCH')
        if candidate != target:
            candidate.replace(target)
        receipt['files'][name] = {'bytes': size, 'sha256': actual, 'lfs_sha256_verified': bool(expected)}
        print(json.dumps({'downloaded': name, 'bytes': size}), flush=True)
    config = json.loads((a.output / 'config.json').read_text())
    if config.get('architectures') != ['LlamaForCausalLM'] or config.get('model_type') != 'llama':
        raise RuntimeError('SECOND_ARCHITECTURE_NOT_LLAMA')
    (a.output / 'download_receipt.json').write_text(json.dumps(receipt, indent=2))


if __name__ == '__main__': main()
