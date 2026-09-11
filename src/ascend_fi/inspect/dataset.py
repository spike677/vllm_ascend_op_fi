"""Verified local public samples; no prompt generation, translation or judge."""
import hashlib
import json
from pathlib import Path
import re

QUESTION_IDS = [82, 83, 86, 88, 141, 143, 146, 149]
SOURCE_FILE = 'fastchat/llm_judge/data/mt_bench/question.jsonl'
SCOPE = 'MT_BENCH_FIRST_TURN_TARGETED_SUBSET'


def load_samples(dataset_file, manifest_file, question_ids=QUESTION_IDS, turn_index=0):
    raw = Path(dataset_file).read_bytes()
    manifest = json.loads(Path(manifest_file).read_text(encoding='utf-8'))
    if (manifest.get('repository') != 'lm-sys/FastChat' or manifest.get('source_file') != SOURCE_FILE
            or not re.fullmatch('[0-9a-f]{40}', manifest.get('revision', ''))
            or hashlib.sha256(raw).hexdigest() != manifest.get('sha256')):
        raise ValueError('DATASET_PROVENANCE_OR_HASH_MISMATCH')
    if type(turn_index) is not int or turn_index < 0 or len(set(question_ids)) != len(question_ids):
        raise ValueError('INVALID_SAMPLE_SELECTION')
    records = {}
    for line in raw.decode('utf-8').splitlines():
        row = json.loads(line)
        key = row['question_id']
        if key in records:
            raise ValueError('DUPLICATE_QUESTION_ID')
        records[key] = row
    selected = []
    for key in question_ids:
        if key not in records or turn_index >= len(records[key]['turns']):
            raise ValueError('DATASET_SAMPLE_MISSING:' + str(key))
        row = records[key]
        prompt = row['turns'][turn_index]
        if not isinstance(prompt, str) or not prompt:
            raise ValueError('INVALID_ORIGINAL_PROMPT')
        selected.append({'question_id': key, 'turn_index': turn_index, 'category': row['category'],
                         'prompt': prompt, 'prompt_sha256': hashlib.sha256(prompt.encode()).hexdigest(),
                         'source_revision': manifest['revision'], 'source_sha256': manifest['sha256']})
    return manifest, selected


def generation_config(max_tokens):
    return {'temperature': 0, 'seed': 42, 'max_tokens': max_tokens, 'top_p': 1.0,
            'top_k': -1, 'min_p': 0.0, 'presence_penalty': 0.0, 'frequency_penalty': 0.0,
            'repetition_penalty': 1.0, 'stop': [], 'stop_token_ids': [], 'ignore_eos': False,
            'min_tokens': 0, 'n': 1, 'add_generation_prompt': True}


def as_probe(sample, max_tokens):
    return {'probe_id': f"mtbench_{sample['question_id']}_turn{sample['turn_index']}",
            'category': sample['category'], 'prompt': sample['prompt'], 'dataset_sample': sample,
            'generation_config': generation_config(max_tokens), 'expected': {},
            'applicable_detectors': ['length', 'garbage', 'crash']}
