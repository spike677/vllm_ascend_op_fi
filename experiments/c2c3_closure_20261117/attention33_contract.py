"""CPU-only identities and prerequisites for the natural TND FD experiment."""
import hashlib
import json
import math
from pathlib import Path

PHASES = ('baseline_before', 'baseline_repeat', 'measured', 'off_after', 'healthy_after')
RUNNER = dict(dtype='float16', tensor_parallel_size=1, enforce_eager=True,
    max_model_len=1280, max_num_seqs=1, max_num_batched_tokens=1280, block_size=16,
    num_gpu_blocks_override=128, enable_prefix_caching=False, enable_chunked_prefill=False,
    async_scheduling=False, seed=42,
    worker_extension_cls='demoncat.runtime.vllm_ascend.controller.DemoncatWorkerExtension',
    disable_log_stats=True)
ENVIRONMENT = ('VLLM_ENABLE_V1_MULTIPROCESSING', 'VLLM_WORKER_MULTIPROC_METHOD',
    'VLLM_ASCEND_ENABLE_NZ', 'VLLM_ALLOW_INSECURE_SERIALIZATION', 'OMP_NUM_THREADS',
    'VLLM_HOST_IP', 'GLOO_SOCKET_IFNAME', 'HCCL_CONNECT_TIMEOUT', 'HCCL_EXEC_TIMEOUT',
    'ACLNN_CACHE_LIMIT', 'TASK_QUEUE_ENABLE', 'DCAT33_REVIEWED_PHYSICAL_CARD')


def require(ok, reason):
    if not ok:raise RuntimeError(reason)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for data in iter(lambda: stream.read(4*1024**2), b''):digest.update(data)
    return digest.hexdigest()


def read_json(path):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            require(key not in result, 'TND33_DUPLICATE_JSON_KEY:' + key);result[key] = value
        return result
    def invalid(value):raise RuntimeError('TND33_NONFINITE_JSON:' + value)
    return json.loads(Path(path).read_bytes(), object_pairs_hook=unique, parse_constant=invalid)


def file_inventory(root, *, suffixes, max_files, max_bytes):
    root = Path(root).resolve(strict=True);files = {}
    for path in sorted(root.rglob('*')):
        relative = path.relative_to(root)
        if any(part.startswith('.') or part == '__pycache__' for part in relative.parts):continue
        if not path.is_file() or path.suffix not in suffixes:continue
        require(path.resolve().is_relative_to(root), 'TND33_INPUT_FILE_ESCAPES_ROOT')
        files[relative.as_posix()] = dict(bytes=path.stat().st_size, sha256=sha(path))
        require(len(files) <= max_files and sum(row['bytes'] for row in files.values()) <= max_bytes,
                'TND33_INPUT_INVENTORY_BOUND')
    require(files, 'TND33_INPUT_INVENTORY_EMPTY')
    return files


def execution_contract(model, installed, driver, memory, environment):
    require(type(memory) is float and math.isfinite(memory) and 0 < memory <= .8,
            'TND33_BOUNDED_MEMORY_REQUIRED')
    model, installed, driver = Path(model), Path(installed), Path(driver)
    model_files = file_inventory(model, suffixes={'.json', '.safetensors', '.bin', '.model', '.tiktoken', '.txt', '.py'},
                                 max_files=128, max_bytes=4*1024**3)
    require('config.json' in model_files and any(name.endswith('.safetensors') for name in model_files),
            'TND33_ORIGINAL_MODEL_FILES_REQUIRED')
    config = read_json(model/'config.json')
    layout = {name: config[name] for name in ('model_type', 'hidden_size', 'num_hidden_layers',
        'num_attention_heads', 'num_key_value_heads', 'head_dim', 'vocab_size')}
    require(layout == dict(model_type='qwen3', hidden_size=1024, num_hidden_layers=28,
        num_attention_heads=16, num_key_value_heads=8, head_dim=128, vocab_size=151936),
        'TND33_REVIEWED_QWEN_LAYOUT_REQUIRED')
    package_files = {}
    for name in ('demoncat', 'fault_injection'):
        package_files[name] = file_inventory(installed/name, suffixes={'.py', '.json', '.h', '.hpp', '.cpp'},
                                             max_files=4096, max_bytes=128*1024**2)
    return dict(schema='demoncat.paged33.execution_contract.v1', model_files=model_files,
        model_layout=layout, installed_files=package_files, driver_sha256=sha(driver),
        contract_source_sha256=sha(Path(__file__)), runner=dict(RUNNER, gpu_memory_utilization=memory),
        sampling=dict(temperature=0, max_tokens=2, ignore_eos=True), prompt_tokens=1024,
        environment={name: environment.get(name) for name in ENVIRONMENT})


def verify_bridge_header(path, expected_sha, contract, completion_sha, reviewer_source):
    require(path is not None and expected_sha and sha(path) == expected_sha, 'TND33_EXACT_BRIDGE_REQUIRED')
    data = read_json(path)
    require(data.get('schema') == 'demoncat.paged33.stock_private_bridge.v1'
        and data.get('status') == 'PASS_EXACT_COMPLETE_STOCK_PRIVATE_BRIDGE'
        and data.get('execution_contract') == contract and data.get('completion_sha256') == completion_sha
        and data.get('reviewer_source_sha256') == sha(reviewer_source)
        and data.get('full_payload_file_pairs') == 435 and data.get('native_baseline_launches') == 112
        and data.get('native_keys') == [5100000000010200103]
        and len(data.get('actual_aic', [])) == 1 and type(data['actual_aic'][0]) is int
        and 20 <= data['actual_aic'][0] <= 25 and data.get('raw_payloads_finite') is True
        and data.get('runtime_launches_reverified') is True
        and data.get('c2') is False and data.get('c3') is False,
        'TND33_BRIDGE_INPUT_BUILD_OR_ACTUAL_PLATFORM_DIFFERS')
    return data
