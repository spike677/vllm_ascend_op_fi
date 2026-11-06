"""Verify model release, tokenizer and fixed requests without exposing NPU devices."""
import json
from pathlib import Path
from model_probe import MODEL, make_requests, verify_release
from stock_probe import write


def main():
    release = verify_release()
    from transformers import AutoConfig, AutoTokenizer
    config = AutoConfig.from_pretrained(str(MODEL), local_files_only=True, trust_remote_code=False)
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL), local_files_only=True, trust_remote_code=False)
    mapping, prompts, teachers = make_requests(tokenizer)
    result = {'status': 'PASS_CPU_MODEL_PREREQUISITES', 'npu_executed': False,
              'release': release, 'request_mapping': mapping, 'prompt_lengths': [len(x) for x in prompts],
              'teacher_tokens': teachers, 'model_geometry': {k: getattr(config, k) for k in
                  ('num_hidden_layers', 'num_attention_heads', 'num_key_value_heads', 'head_dim', 'hidden_size',
                   'vocab_size', 'max_position_embeddings', 'sliding_window')}}
    write(Path('/evidence/result.json'), result)
    print(json.dumps({'status': result['status'], 'npu_executed': False, 'requests': len(mapping),
                      'prompt_lengths': result['prompt_lengths'], 'model_geometry': result['model_geometry']}))


if __name__ == '__main__':
    main()
