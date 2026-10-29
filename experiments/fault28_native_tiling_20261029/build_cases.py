"""CPU-only tokenizer freezing for native FP16 attention calibration."""
import argparse
import hashlib
import json
from pathlib import Path


def token_sha(tokens):
    return hashlib.sha256(json.dumps(list(tokens), separators=(',', ':')).encode()).hexdigest()


def build(tokenizer, length, model, tokenizer_sources, capture_msprobe=False):
    if length not in (256, 512, 1024, 2048):
        raise ValueError('Use a bounded, explicit supported prompt length')
    sentence = ('Describe a public library, its reading rooms, book shelves, daily activities, '
                'and the practical services it offers to visitors. Use clear complete sentences. ')
    text = sentence
    while len(tokenizer.encode(text, add_special_tokens=False)) < length:
        text += sentence
    ids = list(tokenizer.encode(text, add_special_tokens=False)[:length])
    vocabulary = set(tokenizer.get_vocab().values())
    if len(ids) != length or any(type(value) is not int or value not in vocabulary for value in ids):
        raise ValueError('Invalid frozen tokenizer output')
    digest = token_sha(ids)
    cases = []
    for batch in (1, 8):
        for repeat in (False, True):
            cases.append({'schema': 'fault28.native_tiling.case.v1',
                          'case_id': 'stock_b' + str(batch) + ('_repeat' if repeat else ''),
                          'batch_size': batch, 'prompt_length': length, 'target_index': 0,
                          'layer_index': 1, 'capture_msprobe': capture_msprobe,
                          'capture_full_inputs': not repeat, 'operator_variant': 'stock',
                          'model': model, 'dtype': 'half', 'tensor_parallel_size': 1,
                          'prompt_token_ids': [list(ids) for _ in range(batch)],
                          'prompt_token_sha256': [digest] * batch,
                          'expected_target_token_sha256': digest,
                          'expected_target_last_position': length - 1,
                          'prompt_rule': 'Frozen token prefix of repeated ordinary prose; all peers duplicate the target.',
                          'decoded_inspection_text': tokenizer.decode(ids),
                          'tokenizer_sources': tokenizer_sources,
                          'scope': 'Calibration only; no altered tiling, tensor replacement or mechanism acceptance.'})
    return cases


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tokenizer', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--prompt-length', type=int, default=1024)
    parser.add_argument('--model', default='/models/Qwen3-0.6B')
    parser.add_argument('--capture-msprobe', action='store_true')
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('Cases output must be fresh')
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(args.tokenizer), local_files_only=True, trust_remote_code=False)
    names = ('config.json', 'tokenizer.json', 'tokenizer_config.json', 'vocab.json', 'merges.txt', 'special_tokens_map.json')
    sources = {name: hashlib.sha256((args.tokenizer / name).read_bytes()).hexdigest()
               for name in names if (args.tokenizer / name).is_file()}
    cases = build(tokenizer, args.prompt_length, args.model, sources, args.capture_msprobe)
    args.output.mkdir(parents=True, exist_ok=False)
    rows = []
    for case in cases:
        name = case['case_id'] + '.json'
        data = (json.dumps(case, indent=2, ensure_ascii=False) + '\n').encode()
        (args.output / name).write_bytes(data)
        rows.append({'case_id': case['case_id'], 'file': name, 'sha256': hashlib.sha256(data).hexdigest(),
                     'batch_size': case['batch_size'], 'prompt_length': args.prompt_length})
    (args.output / 'manifest.json').write_text(json.dumps({'schema': 'fault28.native_tiling.cases.v1',
        'status': 'INPUTS_FROZEN_NOT_NPU_VALIDATED', 'cases': rows}, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'status': 'INPUTS_FROZEN_NOT_NPU_VALIDATED', 'cases': len(cases),
                      'prompt_length': args.prompt_length, 'target_token_sha256': cases[0]['expected_target_token_sha256']}))


if __name__ == '__main__':
    main()
