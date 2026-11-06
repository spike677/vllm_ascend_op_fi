"""One fresh HF Qwen3 prefill and one teacher-forced decode with a paged-IFA adapter.

The model produces and consumes the actual Q, cached K/V and logits. The decode
adapter explicitly selects paged IFA in place of HF's SDPA. A completed capture
has to be compared with independent baseline, fault and recovery cases.
"""
import argparse
import hashlib
import importlib
import importlib.metadata
import inspect
import json
import os
from pathlib import Path
import re

from collector_stage import select_collector, verify_loaded
from model_adapter import PagedDecode, cache_layer, event, observe_prefill, tensor_report
from replay import SUPPORT_SHA, loaded_native_libraries
from stock_probe import require, write
from vendor_exec import sha, validate_vendor


MODEL = Path('/models/Qwen3-0.6B')
# Same release, checkpoint and tokenizer as the published all-layer Qwen capture.
FILES = {
    'config.json': (726, '660db3b73d788119c04535e48cf9be5f55bc3100841a718637ae695b442f27dd'),
    'generation_config.json': (239, '2325da0f15bb848e018c5ae071b7943332e9f871d6b60e2ed22ca97d4cb993d2'),
    'model.safetensors': (1503300328, 'f47f71177f32bcd101b7573ec9171e6a57f4f4d31148d38e382306f42996874b'),
    'tokenizer.json': (11422654, 'aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4'),
    'tokenizer_config.json': (9732, 'd5d09f07b48c3086c508b30d1c9114bd1189145b74e982a265350c923acd8101'),
    'vocab.json': (2776833, 'ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910'),
}
MODULES = {
    'torch.nn.attention': '96afe3932505426b14f9511605f31764fa8eab8aa8b7af95fbe2d26f70e8faa6',
    'transformers.integrations.sdpa_attention': '87f933d1a2d8508df572da5c0748c6b24c22ff2b625796949957dcd86cc57564',
    'transformers.masking_utils': '7fd2bf34f3abc87953a500c0d53e3e633e5efba10aebcc4dc11422e5769092dd',
    'transformers.models.qwen3.modeling_qwen3': 'fbdcfeeb1b54135ca67ba7df924da92f4b264e1252b517eea2de989289ebaeab',
}
SEEDS = (
    'Explain how a compiler translates a loop into machine instructions. Include an example and a short correctness argument. ',
    'A library records the titles, authors and publication years of books. Describe a reliable catalog and search procedure. ',
    'Solve a sequence of elementary algebra problems and check each answer by substitution. State the assumptions explicitly. ',
    'Write a small Python function that validates a nested configuration and returns clear error messages. Discuss edge cases. ',
    'Explain the water cycle to a student, from evaporation and clouds to rainfall and rivers. Use concrete examples. ',
    'Describe a journey through a city with museums, parks and a railway station. Put the events in chronological order. ',
    'Summarize a set of observations about plant growth under different amounts of light and water. Separate observation from inference. ',
    'Compare two sorting algorithms using a worked example. Explain their memory use and behavior with repeated keys. ',
)


def verify_release():
    results = {}
    for name, (size, expected) in FILES.items():
        path = MODEL / name
        require(path.is_file() and not path.is_symlink() and path.stat().st_size == size and sha(path) == expected,
                'Model/tokenizer fingerprint differs: ' + name)
        results[name] = {'sha256': expected, 'bytes': size}
    versions = {name: importlib.metadata.version(name) for name in ('torch', 'torch-npu', 'transformers')}
    require(versions == {'torch': '2.10.0', 'torch-npu': '2.10.0.post4', 'transformers': '5.5.4'}, 'Unknown stack')
    modules = {}
    for name, expected in MODULES.items():
        path = Path(importlib.import_module(name).__file__).resolve()
        require(path.is_relative_to('/work/venv') and sha(path) == expected, 'Changed model Python: ' + name)
        modules[name] = {'path': str(path), 'sha256': expected}
    return {'root': str(MODEL), 'files': results, 'versions': versions, 'modules': modules}


def validate_case(case):
    require(case['execution_kind'] == 'model_paged_adapter' and case['mode'] in ('off', 'inner1024', 'inner512'),
            'Unknown model execution policy')
    expected = dict(batch=8, kv_length=16384, q_heads=16, kv_heads=8, head_dim=128, block_size=128,
                    dtype='float16', inner_precise=0, deterministic_algorithms=True)
    require(all(type(case.get(k)) is type(v) and case[k] == v for k, v in expected.items()), 'Unexpected model geometry')
    require(case.get('repeats', 1) == 1 and type(case['capture_msprobe']) is bool,
            'Each fresh model process runs one decode')


def verify_vendor(case, case_path, out):
    require(not os.environ.get('LD_PRELOAD') and not any(k.startswith('FAULT33_PA_') for k in os.environ),
            'Unexpected inherited native scope')
    if case['operator_variant'] == 'stock':
        require(case['mode'] == 'off' and not os.environ.get('ASCEND_CUSTOM_OPP_PATH'), 'Unexpected stock policy')
        return None
    vendor = validate_vendor(case)
    launch_path = out.with_name(out.name + '.vendor_launch.json')
    require(os.environ.get('FAULT33_VENDOR_LAUNCH_RECEIPT') == str(launch_path)
            and sha(launch_path) == os.environ.get('FAULT33_VENDOR_LAUNCH_RECEIPT_SHA256'), 'Wrong vendor launcher')
    launch = json.loads(launch_path.read_bytes())
    require(launch['probe_sha256'] == sha(Path(__file__)) and launch['case_sha256'] == sha(case_path)
            and launch['verified_vendor']['manifest_sha256'] == vendor['manifest_sha256'], 'Wrong launch evidence')
    require(os.environ.get('ACLNN_CACHE_LIMIT') == '0' and all(
        launch['child_environment'].get(k) == os.environ.get(k) for k in
        ('ASCEND_CUSTOM_OPP_PATH', 'LD_LIBRARY_PATH', 'LD_PRELOAD', 'ACLNN_CACHE_LIMIT', 'PYTHONPATH')),
        'Live vendor environment differs')
    return vendor


def tokens_digest(token_ids):
    return hashlib.sha256(json.dumps(token_ids, separators=(',', ':')).encode()).hexdigest()


def make_requests(tokenizer):
    requests, prompts, teachers = [], [], []
    for i, seed in enumerate(SEEDS):
        encoded = tokenizer.encode(seed * 2048, add_special_tokens=False)
        require(len(encoded) >= 16384 and all(0 <= n < 151936 for n in encoded[:16384]), 'Short/invalid prompt')
        ids = encoded[:16384]
        prompts.append(ids[:-1])
        teachers.append([ids[-1]])
        requests.append({'batch_row': i, 'request_id': f'request_{i:02d}', 'prefill_length': 16383,
                         'prompt_token_sha256': tokens_digest(ids[:-1]), 'known_prefix_sha256': tokens_digest(ids),
                         'teacher_token_id': ids[-1], 'decode_position': 16383})
    require(len({r['known_prefix_sha256'] for r in requests}) == 8, 'Expected eight distinct requests')
    return requests, prompts, teachers


def memory(npu):
    result = {}
    for name in ('memory_allocated', 'max_memory_allocated', 'memory_reserved', 'max_memory_reserved', 'mem_get_info'):
        getter = getattr(npu, name, None)
        if getter:
            try:
                result[name] = getter()
            except (RuntimeError, TypeError, NotImplementedError) as exc:
                result[name] = {'unavailable': type(exc).__name__}
    return result


def configure_debugger(out, collector):
    from msprobe.pytorch import PrecisionDebugger
    processor = importlib.import_module('msprobe.core.dump.data_dump.data_processor.pytorch_processor')
    collector['actual_loaded'] = verify_loaded(collector, processor)
    support = Path(inspect.getsourcefile(PrecisionDebugger)).parents[1] / 'api_dump/support_wrap_ops.yaml'
    require(sha(support) == SUPPORT_SHA and '- npu_incre_flash_attention' in support.read_text(), 'Changed IFA registration')
    write(out / 'collector_provenance.json', collector)
    config = {'task': 'statistics', 'level': 'mix', 'rank': [], 'step': [0], 'dump_path': str(out / 'observation'),
              'async_dump': False, 'statistics': {'scope': [], 'list': [], 'data_mode': ['all'],
                                                   'summary_mode': 'statistics', 'tensor_list': []}}
    if collector['precision'] == 'high':
        config['precision'] = 'high'
    write(out / 'config.json', config)
    return PrecisionDebugger(config_path=str(out / 'config.json'))


def capture_receipt(out):
    directory = out / 'observation' / 'step0' / 'rank0'
    require(not any(file.is_file() and file.suffix.lower() in {'.pt', '.pth', '.npy', '.npz', '.bin', '.safetensors'}
                    for file in (out / 'observation').rglob('*')), 'Full tensors in statistics capture')
    objects, files = {}, []
    for name in ('dump', 'construct', 'stack'):
        path = directory / (name + '.json')
        objects[name] = json.loads(path.read_bytes())
        files.append({'path': path.relative_to(out).as_posix(), 'bytes': path.stat().st_size, 'sha256': sha(path)})
    dump = objects['dump']
    require(dump['task'] == 'statistics' and dump['level'] == 'mix' and dump.get('dump_data_dir') is None, 'Wrong dump mode')
    apis = [key for key in dump['data'] if re.fullmatch(r'NPU\.npu_incre_flash_attention\.\d+\.forward', key)]
    stack_nodes = {name for group in objects['stack'].values() if isinstance(group, list) and len(group) == 2
                   and isinstance(group[0], list) and isinstance(group[1], list) and group[1]
                   for name in group[0]}
    linked = {}
    for api in apis:
        parent = objects['construct'].get(api)
        match = re.fullmatch(r'Module\.model\.layers\.(\d+)\.self_attn\.Qwen3Attention\.forward\.\d+', parent or '')
        require(match and parent in dump['data'] and api in stack_nodes, 'IFA lacks original layer ancestry/stack')
        layer = int(match[1])
        require(layer not in linked, 'Duplicate native IFA on one layer')
        linked[layer] = {'step': 0, 'layer_index': layer, 'api': api, 'parent': parent}
    require(sorted(linked) == list(range(28)), 'Incomplete native all-layer coverage')
    return {'schema': 'fault33.pa.model_statistics.v1', 'native_files': files,
            'native_api_nodes': [linked[k] for k in sorted(linked)], 'steps': [0], 'rank': 0,
            'config_sha256': sha(out / 'config.json'), 'collector_sha256': sha(out / 'collector_provenance.json'),
            'scope': 'One complete decode model call; prefill is outside the statistics window.'}


def verify_host(out, calls, mode):
    rows = [json.loads(line) for line in (out / 'host_events.jsonl').read_text().splitlines()]
    require(len(rows) == len(calls) == 28, 'Private host callbacks do not cover all decode attention layers')
    for event_row, call in zip(rows, calls):
        require(all(event_row[k] == call[f] for k, f in
                    [('pid', 'pid'), ('tid', 'tid'), ('call_id', 'call_id'), ('token', 'scope_token')])
                and event_row['mode'] == mode and event_row['phase'] == 'decode'
                and event_row['callback_status'] == 0 and event_row['eligible'] is True
                and event_row['policy_applied'] is (mode != 'off')
                and event_row['selector_checks'] == 1 and event_row['serialized_captures'] == 1
                and event_row['typed_matches_final_serialized_raw'] is True
                and call['enter']['monotonic_ns'] <= event_row['callback_begin_monotonic_ns']
                <= event_row['callback_end_monotonic_ns'] <= call['exit']['monotonic_ns'],
                'Host callback/real IFA scope mismatch for ' + call['call_id'])
    return {'sha256': sha(out / 'host_events.jsonl'), 'callbacks': len(rows),
            'applied': sum(row['policy_applied'] for row in rows)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case-file', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    case = json.loads(args.case_file.read_bytes())
    validate_case(case)
    out = args.output.resolve()
    out.mkdir(exist_ok=False, parents=True)
    (out / 'private_output').mkdir()
    write(out / 'case.json', case)
    vendor = verify_vendor(case, args.case_file, out)
    collector = select_collector(case)
    import torch
    import torch_npu
    from torch.nn import functional
    from torch.nn.attention import sdpa_kernel, SDPBackend
    from transformers import AutoModelForCausalLM, AutoTokenizer
    release = verify_release()
    write(out / 'release.json', release)
    torch.set_num_threads(2)
    torch.manual_seed(42)
    torch.use_deterministic_algorithms(True)
    torch_npu.npu.set_device(0)
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL), local_files_only=True, trust_remote_code=False)
    requests, prompts, teachers = make_requests(tokenizer)
    write(out / 'request_mapping.json', {'schema': 'fault33.model_known_prefix.v1', 'rows': requests})
    model = AutoModelForCausalLM.from_pretrained(str(MODEL), dtype=torch.float16,
        attn_implementation='sdpa', local_files_only=True, trust_remote_code=False).eval().to('npu:0')
    cfg = model.config
    require(type(model).__name__ == 'Qwen3ForCausalLM' and cfg._attn_implementation == 'sdpa'
            and len(model.model.layers) == cfg.num_hidden_layers == 28 and cfg.num_key_value_heads == 8
            and cfg.num_attention_heads == 16 and cfg.hidden_size == 1024 and cfg.head_dim == 128
            and cfg.vocab_size == 151936 and cfg.max_position_embeddings >= 16384
            and not cfg.sliding_window and all(param.dtype == torch.float16 for param in model.parameters()),
            'Unexpected model or attention configuration')
    ids = torch.tensor(prompts, dtype=torch.long, device='npu:0')
    next_ids = torch.tensor(teachers, dtype=torch.long, device='npu:0')
    require(ids.cpu().tolist() == prompts and next_ids.cpu().tolist() == teachers, 'Token batch differs')
    measurements = {'loaded': memory(torch_npu.npu)}
    with torch.inference_mode(), sdpa_kernel(SDPBackend.FLASH_ATTENTION):
        event('model_prefill_enter', case_id=case['case_id'], batch=8, prompt_length=16383)
        with observe_prefill(model, functional, torch) as prefill_calls:
            pre = model(input_ids=ids, position_ids=torch.arange(16383, device='npu:0').unsqueeze(0),
                        attention_mask=None, past_key_values=None, use_cache=True, logits_to_keep=1, return_dict=True)
        torch_npu.npu.synchronize()
        cache = pre.past_key_values
        require(cache is not None and len(cache.layers) == 28 and int(cache.get_seq_length()) == 16383,
                'Model did not return the exact real prefill cache')
        require(list(pre.logits.shape) == [8, 1, 151936], 'Unexpected prefill logits shape')
        prefill_logits = tensor_report(pre.logits, finite=True)
        del pre
        event('model_prefill_exit', case_id=case['case_id'], cache_layers=len(cache.layers))
        measurements['prefill'] = memory(torch_npu.npu)
        prefix = []
        for idx in range(28):
            k, v = cache_layer(cache, idx)
            require(list(k.shape) == list(v.shape) == [8, 8, 16383, 128] and k.dtype == v.dtype == torch.float16,
                    'Unexpected real cache shape or dtype')
            prefix.append({'key': tensor_report(k), 'value': tensor_report(v)})
        del k, v
        write(out / 'prefill_cache.json', {'layers': prefix, 'original_sdpa_calls': prefill_calls,
                                          'actual_cache_class': type(cache).__name__,
                                          'length': 16383, 'prefill_logits': prefill_logits})
        table = torch.arange(1024, dtype=torch.int32, device='npu:0').view(8, 128)
        write(out / 'block_table.json', {'values': table.cpu().tolist(), **tensor_report(table)})
        adapter = PagedDecode(torch, torch_npu, model, cache, prefix, table, case, out,
                              [row['request_id'] for row in requests])
        debugger = configure_debugger(out, collector) if case['capture_msprobe'] else None
        if debugger:
            debugger.start(model=model, rank_id=0)
        try:
            with adapter.install(functional):
                event('model_decode_enter', case_id=case['case_id'], cache_length_before=16383, mode=case['mode'])
                reply = model(input_ids=next_ids, position_ids=torch.tensor([[16383]], device='npu:0'),
                    attention_mask=None, past_key_values=cache, use_cache=True, logits_to_keep=1, return_dict=True)
                torch_npu.npu.synchronize()
                event('model_decode_exit', case_id=case['case_id'], status='returned')
        finally:
            if debugger:
                debugger.stop()
                debugger.step()
        require(adapter.visits == adapter.exits == list(range(28)) and len(adapter.calls) == 28
                and next_ids.cpu().tolist() == teachers, 'Incomplete model coverage or changed teacher inputs')
        require(reply.past_key_values is cache and all(int(cache.get_seq_length(i)) == 16384 for i in range(28))
                and list(reply.logits.shape) == [8, 1, 151936], 'Incomplete decode cache/logits')
        private = out / 'private_output' / 'decode_logits.bin'
        logits = tensor_report(reply.logits, destination=private, finite=True)
        measurements['decode'] = memory(torch_npu.npu)
    write(out / 'runtime_native_maps.json', {'pid': os.getpid(), 'libraries': loaded_native_libraries()})
    host = verify_host(out, adapter.calls, case['mode']) if vendor else None
    if case['capture_msprobe']:
        write(out / 'statistics_receipt.json', capture_receipt(out))
    write(out / 'model_calls.json', {'calls': adapter.calls, 'module_enters': adapter.visits, 'module_exits': adapter.exits})
    write(out / 'result.json', {'schema': 'fault33.model_paged_adapter.result.v1', 'case_id': case['case_id'],
          'status': 'CAPTURE_COMPLETED', 'mode': case['mode'], 'execution_kind': 'model_paged_adapter', 'model_forward': True,
          'scope': 'Complete HF model, stock SDPA prefill, explicitly adapted paged-IFA decode. No serving scheduler.',
          'mechanism_accepted': False, 'original_api_calls': len(adapter.calls), 'all_layers': adapter.exits,
          'original_prefill_sdpa_calls': len(prefill_calls), 'eligible_decode_ifa_calls': len(adapter.calls),
          'request_count': 8, 'covered_request_layer_pairs': 8 * len(adapter.calls),
          'deterministic_algorithms': torch.are_deterministic_algorithms_enabled(), 'host': host,
          'vendor_manifest_sha256': vendor['manifest_sha256'] if vendor else None,
          'private_logits_file': private.relative_to(out).as_posix(), 'decode_logits': logits,
          'case_sha256': sha(args.case_file), 'calls_sha256': sha(out / 'model_calls.json'), 'memory_bytes': measurements,
          'runtime_native_maps_sha256': sha(out / 'runtime_native_maps.json'),
          'statistics_receipt_sha256': sha(out / 'statistics_receipt.json') if case['capture_msprobe'] else None,
          'prefill_cache_sha256': sha(out / 'prefill_cache.json'), 'block_table_sha256': sha(out / 'block_table.json'),
          'request_mapping_sha256': sha(out / 'request_mapping.json'), 'release_sha256': sha(out / 'release.json')})
    print(json.dumps({'status': 'CAPTURE_COMPLETED', 'case_id': case['case_id'], 'original_api_calls': 28}), flush=True)


if __name__ == '__main__':
    main()
