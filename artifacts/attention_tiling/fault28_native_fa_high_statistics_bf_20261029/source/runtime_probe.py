"""One stock/native-attention calibration prefill. No fault injection here."""
import argparse
import hashlib
import importlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import time
import traceback


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n', encoding='utf-8')
    temporary.replace(path)


def validate_case(case):
    require(case['batch_size'] in (1, 8) and case['prompt_length'] in (256, 512, 1024, 2048), 'Unsupported bounded case')
    require(case['target_index'] == 0 and case['layer_index'] == 1 and
            case.get('tensor_parallel_size', 1) == 1 and case.get('dtype', 'half') == 'half',
            'Initial calibration requires target0/layer1/TP1/FP16')
    require(type(case['capture_msprobe']) is bool and type(case['capture_full_inputs']) is bool, 'Capture flags must be booleans')
    tokens = case['prompt_token_ids']
    require(len(tokens) == case['batch_size'] and all(isinstance(row, list) and len(row) == case['prompt_length']
            and all(type(value) is int and value >= 0 for value in row) for row in tokens), 'Malformed frozen tokens')
    require(all(row == tokens[0] for row in tokens), 'Calibration peers must duplicate the exact target token IDs')
    require(os.environ.get('ACLNN_CACHE_LIMIT') == '0', 'ACLNN_CACHE_LIMIT=0 must be fixed for all calibration cases')
    require(not os.environ.get('LD_PRELOAD'), 'External LD_PRELOAD is not allowed')
    for name in ('DCAT_DQ_V2_ENABLE', 'DCAT_V2_HOOK', 'BATCH_OUTLIER_ENABLED', 'BATCH_INPUT_ENABLED'):
        require(os.environ.get(name, '0') == '0', 'Another experiment hook is enabled: ' + name)
    variant = case.get('operator_variant', 'stock')
    custom = os.environ.get('ASCEND_CUSTOM_OPP_PATH')
    if variant == 'stock':
        require(not custom, 'Stock calibration forbids external ASCEND_CUSTOM_OPP_PATH before native initialization')
    elif variant == 'private_vendor':
        require(custom and custom == case.get('approved_custom_opp_path'), 'Private vendor path is not explicitly approved by the case')
        manifest = Path(case['approved_vendor_manifest_file'])
        require(manifest.is_file() and sha(manifest) == case['approved_vendor_manifest_sha256'], 'Private vendor manifest fingerprint differs')
    else:
        raise RuntimeError('Unknown operator variant')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case-file', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    case = json.loads(args.case_file.read_bytes())
    validate_case(case)
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    here = Path(__file__).resolve().parent
    before_environment = {key: os.environ.get(key) for key in
                          ('LD_PRELOAD', 'ASCEND_CUSTOM_OPP_PATH', 'ACLNN_CACHE_LIMIT', 'ASCEND_GLOBAL_LOG_LEVEL',
                           'ASCEND_SLOG_PRINT_TO_STDOUT', 'ASCEND_LAUNCH_BLOCKING')}
    os.environ.update(FAULT28_CALIBRATION='1', FAULT28_OUTPUT=str(args.output),
                      FAULT28_CONTROL=str(args.output / 'control.json'),
                      FAULT28_MSPROBE=str(int(case['capture_msprobe'])),
                      VLLM_WORKER_MULTIPROC_METHOD='spawn', VLLM_DISABLE_REQUEST_ID_RANDOMIZATION='1',
                      VLLM_ENABLE_V1_MULTIPROCESSING='0', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
                      PYTHONPATH=str(here) + os.pathsep + os.environ.get('PYTHONPATH', ''))
    import model_hook
    model_hook.install()
    model = case.get('model', '/models/Qwen3-0.6B')
    options = {'model': model, 'dtype': 'half', 'tensor_parallel_size': 1,
               'cpu_offload_gb': 0, 'gpu_memory_utilization': .5, 'max_model_len': 4096,
               'max_num_batched_tokens': 16384, 'max_num_seqs': 8, 'enforce_eager': True,
               'enable_prefix_caching': False, 'enable_chunked_prefill': False, 'async_scheduling': False,
               'additional_config': {'enable_flashcomm1': False}, 'distributed_executor_backend': 'mp',
               'seed': 42, 'block_size': 128, 'trust_remote_code': False, 'disable_log_stats': True}
    result = {'schema': 'fault28.native_tiling.calibration.v1', 'status': 'STARTING',
              'case_id': case['case_id'], 'batch_size': case['batch_size'], 'prompt_length': case['prompt_length'],
              'target_index': 0, 'layer_index': 1, 'source_case_sha256': sha(args.case_file),
              'operator_variant': case.get('operator_variant', 'stock'), 'options': options,
              'environment_before_native_init': before_environment,
              'source_sha256': {name: sha(here / name) for name in
                                ('runtime_probe.py', 'model_hook.py', 'sitecustomize.py', 'build_cases.py')},
              'checkpoint_conversion': 'Original checkpoint config is recorded below; native vLLM dtype=half converts loading to FP16.',
              'scope': 'Read-only calibration. CAPTURE_COMPLETED is not a tiling or fault-mechanism PASS.',
              'capture_full_inputs': case['capture_full_inputs'], 'capture_msprobe': case['capture_msprobe'],
              'private_raw_policy': 'private_replay/*.bin contains actual logical activations/logits, not public statistics; exclude from GitHub.',
              'requests': [], 'native_calls': [], 'start_epoch': time.time()}
    write(args.output / 'control.json', {'active': False})
    save = lambda: write(args.output / 'result.json', result)
    save()
    try:
        config_file = Path(model) / 'config.json'
        config = json.loads(config_file.read_bytes())
        result['checkpoint_config'] = {'sha256': sha(config_file), 'torch_dtype': config.get('torch_dtype'),
                                       'dtype': config.get('dtype'), 'architectures': config.get('architectures'),
                                       'num_hidden_layers': config.get('num_hidden_layers'),
                                       'quantization_config': config.get('quantization_config')}
        require(config.get('quantization_config') is None, 'Initial probe requires an unquantized checkpoint')
        require((config.get('dtype') or config.get('torch_dtype')) in ('bfloat16', 'torch.bfloat16'),
                'Expected original BF16 checkpoint; record conversion explicitly')
        from vllm import LLM, SamplingParams
        native_sources = {
            'vllm.v1.engine.input_processor': 'c5673988c0f7cfec268220e3f044e718702c015a4f236c020937cfd40a793f15',
            'vllm.v1.engine.llm_engine': '2900f6732a1e507f6ced6f75f05f1fcaf2cc2929df0f5066574e9077ad3fbde1',
            'vllm.v1.engine.core_client': 'e41fe15caa59dfd289ae212b3eaefc20c34be87a72e779df03d970f9714b793d',
            'vllm.entrypoints.offline_utils': '688fbad0af9c2180b83aa77dcd0dbda85ca076a6c72bffa61840896d950cf458',
            'vllm.v1.executor.abstract': '419002195d13cf050c3ea9705d18a12456c890c27c470a1efadce7de27ded0b1',
            'vllm.v1.executor.multiproc_executor': 'a2837f18010a1f3539ea062b0824ae6dacaced0ec5813910a4b6edef7875af18'}
        result['native_submission_sources'] = {}
        for name, expected in native_sources.items():
            path = Path(importlib.import_module(name).__file__)
            require(sha(path) == expected, 'Native submission source changed: ' + name)
            result['native_submission_sources'][name] = {'path': str(path), 'sha256': expected}
        result['versions'] = {name: importlib.metadata.version(name) for name in
                              ('vllm', 'vllm-ascend', 'torch', 'torch-npu', 'mindstudio-probe')}
        save()
        from vllm.v1.engine.core_client import InprocClient
        from vllm.v1.executor.multiproc_executor import MultiprocExecutor
        llm = LLM(**options)
        client = llm.llm_engine.engine_core
        require(isinstance(client, InprocClient), 'Native synchronous InprocClient required')
        executor = client.engine_core.model_executor
        require(isinstance(executor, MultiprocExecutor) and llm.llm_engine.model_executor is executor and
                executor.parallel_config.tensor_parallel_size == 1, 'Native TP1 mp worker executor required')
        result['submission'] = {'strategy': 'enqueue_complete_batch_then_step',
                                'actual_client': type(client).__module__ + '.' + type(client).__qualname__,
                                'actual_executor': type(executor).__module__ + '.' + type(executor).__qualname__,
                                'VLLM_ENABLE_V1_MULTIPROCESSING': os.environ['VLLM_ENABLE_V1_MULTIPROCESSING'],
                                'VLLM_DISABLE_REQUEST_ID_RANDOMIZATION': os.environ['VLLM_DISABLE_REQUEST_ID_RANDOMIZATION'],
                                'ACLNN_CACHE_LIMIT': os.environ['ACLNN_CACHE_LIMIT']}
        tokenizer = llm.get_tokenizer()
        vocabulary = set(tokenizer.get_vocab().values())
        tokens = case['prompt_token_ids']
        require(all(token in vocabulary for row in tokens for token in row), 'Frozen tokens outside actual model vocabulary')
        for name, expected in case['tokenizer_sources'].items():
            require(sha(Path(model) / name) == expected, 'Actual tokenizer/config source differs: ' + name)
        require([model_hook.token_sha(row) for row in tokens] == case['prompt_token_sha256'] and
                model_hook.token_sha(tokens[0]) == case['expected_target_token_sha256'] and
                case['expected_target_last_position'] == len(tokens[0])-1, 'Frozen token fingerprints changed')
        require(sum(map(len, tokens)) <= options['max_num_batched_tokens'] and
                all(len(row)+1 <= options['max_model_len'] for row in tokens), 'No truncation/chunking permitted')
        control = {**case, 'active': True, 'source_case_sha256': result['source_case_sha256'],
                   'request_id_to_prompt_index': {str(i): i for i in range(len(tokens))}}
        write(args.output / 'control.json', control)
        save()
        params = SamplingParams(temperature=0, seed=42, max_tokens=1, ignore_eos=True, logprobs=20)
        replies = llm.generate([{'prompt_token_ids': row} for row in tokens], params, use_tqdm=False)
        write(args.output / 'control.json', {'active': False})
        require(len(replies) == len(tokens), 'Request output count differs')
        for reply in replies:
            index = control['request_id_to_prompt_index'][reply.request_id]
            require(reply.prompt_token_ids == tokens[index] and len(reply.outputs) == 1 and
                    len(reply.outputs[0].token_ids) == 1, 'Actual request/prefix/generated token count differs')
            generated = reply.outputs[0]
            result['requests'].append({'prompt_index': index, 'request_id': reply.request_id,
                'prompt_token_sha256': model_hook.token_sha(reply.prompt_token_ids),
                'output_token_ids': list(generated.token_ids), 'text': generated.text,
                'finish_reason': generated.finish_reason,
                'logprobs': [{str(k): v.logprob for k, v in row.items()} for row in generated.logprobs]})
        events = [json.loads(line) for path in args.output.glob('pid*.jsonl')
                  for line in path.read_text(encoding='utf-8').splitlines()]
        completed = [event for event in events if event['event'] == 'capture_completed']
        require(len(completed) == 1 and completed[0]['rank'] == 0 and completed[0]['logits_saved'] and
                completed[0]['prefill_certified'], 'Native rank0 capture did not complete exactly once')
        for path in sorted((args.output / 'native_calls').glob('call_*.json')):
            row = json.loads(path.read_bytes())
            result['native_calls'].append({key: row[key] for key in
                ('call_id', 'native_index', 'layer_index', 'layer_name', 'api', 'status')} |
                {'file': str(path.relative_to(args.output)), 'sha256': sha(path),
                 'target_mapping_certified': row['target_mapping_certified']})
        require(len(result['native_calls']) == int(config['num_hidden_layers']), 'Incomplete native call ledger')
        result.update(status='CAPTURE_COMPLETED', prefill_certified=True, all_requests_in_one_prefill=True,
                      completed=completed[0], receipt_sha256=sha(args.output / 'receipt.json'), end_epoch=time.time())
        result['requests'].sort(key=lambda row: row['prompt_index'])
        save()
        print(json.dumps({'status': result['status'], 'case_id': case['case_id'],
                          'native_calls': len(result['native_calls']), 'prefill_certified': True}), flush=True)
        return 0
    except BaseException:
        result.update(status='CAPTURE_FAILED', traceback=traceback.format_exc(), end_epoch=time.time())
        save()
        print(result['traceback'], file=sys.stderr)
        return 1
    finally:
        write(args.output / 'control.json', {'active': False})


if __name__ == '__main__':
    raise SystemExit(main())
