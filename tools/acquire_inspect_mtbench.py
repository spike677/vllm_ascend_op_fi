"""Download pinned official MT-Bench bytes over verified HTTPS, never model data.

Optional --proxy applies only to this public acquisition process. No global env
or SSH/service settings are changed. Outputs are exclusive and never overwritten.
"""
import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path
import re
import sys
import urllib.request

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from ascend_fi.inspect.dataset import SOURCE_FILE, QUESTION_IDS, SCOPE, load_samples


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--proxy')
    parser.add_argument('--revision', help='Full commit SHA; otherwise resolve main first')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    opener = urllib.request.build_opener(urllib.request.ProxyHandler(
        {'http': args.proxy, 'https': args.proxy} if args.proxy else {}))
    def fetch(url, name):
        req = urllib.request.Request(url, headers={'User-Agent': 'DemonCATPFI-public-dataset-acquisition'})
        with opener.open(req, timeout=45) as r:
            data = r.read()
            receipt = {'url': url, 'final_url': r.url, 'status': r.status,
                       'etag': r.headers.get('ETag'), 'sha256': hashlib.sha256(data).hexdigest(),
                       'bytes': len(data)}
        with (args.output / name).open('xb') as f:
            f.write(data)
        return data, receipt
    revision = args.revision
    info, resolution = fetch('https://api.github.com/repos/lm-sys/FastChat/commits/' + (revision or 'main'), 'revision.json')
    resolved = json.loads(info)['sha']
    if not re.fullmatch('[0-9a-f]{40}', resolved) or revision and revision != resolved:
        raise ValueError('INVALID_REPOSITORY_REVISION')
    root = 'https://raw.githubusercontent.com/lm-sys/FastChat/' + resolved + '/'
    _, questions = fetch(root + SOURCE_FILE, 'question.jsonl')
    _, license_info = fetch(root + 'LICENSE', 'LICENSE')
    _, readme = fetch(root + 'fastchat/llm_judge/README.md', 'UPSTREAM_README.md')
    manifest = {'repository': 'lm-sys/FastChat', 'source_file': SOURCE_FILE, 'revision': resolved,
                'sha256': questions['sha256'], 'scope': SCOPE, 'question_ids': QUESTION_IDS,
                'turn_index': 0, 'acquired_at': dt.datetime.now(dt.timezone.utc).isoformat(),
                'acquisition': 'verified HTTPS public files only', 'proxy_used': bool(args.proxy),
                'receipts': [resolution, questions, license_info, readme],
                'license': {'repository_license': 'Apache-2.0', 'path': 'LICENSE',
                            'sha256': license_info['sha256'],
                            'scope_note': 'Repository license captured; no claim of separate dataset-specific grant beyond upstream materials.'},
                'official_benchmark_score': 'NOT_COMPUTED; first-turn targeted subset, no judge'}
    with (args.output / 'dataset_manifest.json').open('x', encoding='utf-8') as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    _, samples = load_samples(args.output / 'question.jsonl', args.output / 'dataset_manifest.json')
    with (args.output / 'selected_samples.jsonl').open('x', encoding='utf-8') as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')
    print(json.dumps(manifest, ensure_ascii=False))


if __name__ == '__main__':
    main()
