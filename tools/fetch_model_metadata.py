"""Fetch allowlisted ModelScope metadata only; never download model weights."""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
from urllib.parse import quote, urlencode
from urllib.request import urlopen


def allowed(path):
    name = PurePosixPath(path).name.lower()
    if '..' in PurePosixPath(path).parts or path.startswith('/'):
        return False
    return (name in {'config.json', 'configuration.json', 'generation_config.json',
                     'quant_model_description.json', 'readme.md', 'readme',
                     'special_tokens_map.json', 'added_tokens.json', 'vocab.json',
                     'vocab.txt', 'merges.txt', 'chat_template.jinja',
                     'preprocessor_config.json', 'processor_config.json',
                     'video_preprocessor_config.json', 'vision_config.json'}
            or name.endswith('.safetensors.index.json')
            or name.endswith('best_practice.yaml')
            or (name.startswith('tokenizer') and name.endswith(('.json', '.model'))))


def fetch(repo, root):
    base = 'https://modelscope.cn/api/v1/models/' + repo
    listing_url = base + '/repo/files?' + urlencode(dict(Revision='master', Recursive='true'))
    with urlopen(listing_url, timeout=60) as response:
        listing = json.load(response)
    assert listing['Code'] == 200
    files = listing['Data']['Files']
    selected = [f for f in files if f['Type'] == 'blob' and allowed(f['Path'])]
    assert selected and sum(f['Size'] for f in selected) <= 256 * 1024**2
    listing_sha = hashlib.sha256(json.dumps(listing, sort_keys=True).encode()).hexdigest()
    destination = root / repo / ('20261121-' + listing_sha[:12])
    destination.mkdir(parents=True, exist_ok=True)
    (destination / 'repository-listing.json').write_text(json.dumps(listing, indent=2), encoding='utf-8')
    records = []
    for item in selected:
        assert item['Size'] <= 64 * 1024**2, item['Path']
        url = 'https://modelscope.cn/models/' + repo + '/resolve/' + quote(item['Revision']) + '/' + quote(item['Path'])
        with urlopen(url, timeout=120) as response:
            data = response.read(64 * 1024**2 + 1)
        sha = hashlib.sha256(data).hexdigest()
        assert len(data) == item['Size'] and sha == item['Sha256'], item['Path']
        path = destination / item['Path']
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        records.append(dict(path=item['Path'], bytes=len(data), sha256=sha,
                            revision=item['Revision'], source_url=url))
        print(json.dumps(dict(repo=repo, downloaded=item['Path'], bytes=len(data))), flush=True)
    manifest = dict(repository=repo, listing_url=listing_url, listing_sha256=listing_sha,
                    files=records, total_bytes=sum(r['bytes'] for r in records), weights_downloaded=False)
    (destination / 'metadata-manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print(json.dumps(dict(destination=str(destination), total_bytes=manifest['total_bytes'])), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    for variant in ('w4a8', 'w8a8'):
        fetch('Eco-Tech/DeepSeek-V4.1-Flash-1017-' + variant, args.root)
