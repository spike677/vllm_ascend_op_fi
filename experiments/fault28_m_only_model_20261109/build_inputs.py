"""CPU tokenizer preparation of eight distinct bounded request sequences."""
import argparse
import hashlib
import json
from pathlib import Path

SEEDS = [
    'Explain how a compiler translates a loop into machine instructions. Include an example and a short correctness argument. ',
    'A library records the titles, authors and publication years of books. Describe a reliable catalog and search procedure. ',
    'Solve a sequence of elementary algebra problems and check each answer by substitution. State the assumptions explicitly. ',
    'Write a small Python function that validates a nested configuration and returns clear error messages. Discuss edge cases. ',
    'Explain the water cycle to a student, from evaporation and clouds to rainfall and rivers. Use concrete examples. ',
    'Describe a journey through a city with museums, parks and a railway station. Put the events in chronological order. ',
    'Summarize a set of observations about plant growth under different amounts of light and water. Separate observation from inference. ',
    'Compare two sorting algorithms using a worked example. Explain their memory use and behavior with repeated keys. ',
]


def digest(values):
    return hashlib.sha256(json.dumps(list(values), separators=(',', ':')).encode()).hexdigest()


def build(model, destination):
    from transformers import AutoTokenizer
    assert not destination.exists()
    tokenizer = AutoTokenizer.from_pretrained(str(model), local_files_only=True, trust_remote_code=False)
    vocabulary = set(tokenizer.get_vocab().values())
    requests = []
    for index, seed in enumerate(SEEDS):
        multiplier = 1
        while True:
            values = tokenizer.encode(seed * multiplier, add_special_tokens=False)
            if len(values) >= 2049:
                break
            multiplier *= 2
        values = values[:2049]
        assert all(value in vocabulary for value in values)
        requests.append({'request_id': f'request_{index:02d}', 'prompt_seed': seed, 'repetitions': multiplier,
            'prompt_length': 2048, 'prompt_token_ids': values[:2048], 'teacher_forced_next_token': values[2048],
            'prompt_token_ids_sha256': digest(values[:2048]), 'known_prefix_after_decode_sha256': digest(values)})
    assert len({row['prompt_token_ids_sha256'] for row in requests}) == 8
    value = {'schema': 'nes.model_requests.v1', 'requests': requests, 'prefill_length': 2048, 'decode_steps': 1,
             'request_sets': {'b1': [requests[0]['request_id']], 'b8': [row['request_id'] for row in requests]},
             'policy': 'Natural text tokenization, fixed unpadded lengths. Decode token is teacher forced identically in all arms; no generated-token feedback.',
             'new_model_request_execution': True, 'npu_calls_during_preparation': 0}
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'requests': 8, 'manifest_sha256': hashlib.sha256(destination.read_bytes()).hexdigest(), 'npu_calls': 0}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    build(args.model, args.output)
