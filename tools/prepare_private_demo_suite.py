"""Freeze a new small-model demo protocol from existing verified public samples.

Does not change the source suite, prompts, rules or detector thresholds. This is
a new 512-token demo, not a replay of the historical 1024-token Qwen32B evidence.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from ascend_fi.evaluation.distortion.contracts import content_hash


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-suite', type=Path, required=True)
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--max-tokens', type=int, default=512)
    a = p.parse_args()
    original = json.loads(a.source_suite.read_text())
    if original['hash'] != content_hash(original):
        raise ValueError('SOURCE_SUITE_HASH_MISMATCH')
    suite = copy.deepcopy(original)
    ids = [82, 86, 149]
    suite['probes'] = [next(p for p in original['probes'] if p['dataset_sample']['question_id'] == n) for n in ids]
    suite['probe_count'] = len(ids)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(a.model), local_files_only=True)
    counts = {}
    for probe in suite['probes']:
        probe['generation_config']['max_tokens'] = a.max_tokens
        counts[probe['probe_id']] = len(tokenizer.apply_chat_template(
            [{'role': 'user', 'content': probe['prompt']}], tokenize=True,
            add_generation_prompt=True, **suite['request_protocol']['chat_template_kwargs']))
    suite['demo_scope'] = {'model': str(a.model), 'samples': ids, 'cycles': 1,
        'source_suite_sha256': hashlib.sha256(a.source_suite.read_bytes()).hexdigest(),
        'prompt_token_counts': counts, 'max_prompt_tokens': max(counts.values()),
        'native_weight_quantization': False, 'not_official_mtbench_score': True}
    suite['hash'] = content_hash(suite)
    with a.out.open('x', encoding='utf-8') as f:
        json.dump(suite, f, ensure_ascii=False, indent=2)
    print(json.dumps(suite['demo_scope']))


if __name__ == '__main__':
    main()
