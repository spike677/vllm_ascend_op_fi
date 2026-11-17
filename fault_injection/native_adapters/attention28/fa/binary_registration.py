"""Register only the newly compiled FA S1S2 key using the pinned source generator."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path

BINARY = 'FlashAttentionScore_d510be03e39ccdbb4abc4f0db5af9547'
KEY = 22619644976
CORE_TYPES = {'MIX': 0, 'AiCore': 1, 'VectorCore': 2, 'MIX_AICORE': 3, 'MIX_VECTOR_CORE': 4, 'MIX_AIV': 4}


def require(ok, message):
    if not ok:raise RuntimeError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_metadata(metadata, binary):
    require(metadata.get('binFileName') == BINARY and metadata.get('binFileSuffix') == '.o'
            and metadata.get('sha256') == sha(binary), 'FRESH_FA_KERNEL_IDENTITY_DIFFERS')
    require(metadata.get('coreType') in CORE_TYPES, 'UNRECOGNIZED_COMPILED_FA_CORE_TYPE')
    support = metadata.get('supportInfo', {})
    require(support.get('simplifiedKeyMode') == 0 and support.get('simplifiedKey')
            and support.get('optionalInputMode') == 'gen_placeholder'
            and support.get('optionalOutputMode') == 'gen_placeholder', 'FA_SUPPORT_CONTRACT_DIFFERS')
    keys = support.get('tilingKey', [])
    require(len(keys) == 1 and int(keys[0]) == KEY, 'FA_SUPPORT_KEY_SET_DIFFERS')
    # Some compiler representations have a single kernel, others a kernelList.
    # Preserve the actual representation; never copy FIA's MIX launch convention.
    rows = metadata.get('kernelList')
    if rows is not None:
        require(len(rows) == 1 and int(rows[0]['tilingKey']) == KEY, 'FA_COMPILED_KEY_SET_DIFFERS')


def validate_generated(generated, existing, metadata):
    require(set(generated) == {'binary_info_config.json', 'flash_attention_score.json'}, 'FA_GENERATED_OPERATORS_DIFFER')
    require(generated['flash_attention_score.json'] == existing, 'FA_GENERATED_MATCHER_DIFFERS_FROM_SDK')
    index = generated['binary_info_config.json']
    require(set(index) == {'FlashAttentionScore'}, 'FA_SUMMARY_OPERATOR_DIFFERS')
    op = index['FlashAttentionScore'];rows = op.get('binaryList', [])
    path = 'ascend910b/flash_attention_score/' + BINARY
    require(len(rows) == 1 and rows[0].get('binPath') == path + '.o'
            and rows[0].get('jsonPath') == path + '.json'
            and rows[0].get('coreType') == CORE_TYPES[metadata['coreType']]
            and rows[0].get('simplifiedKey') == metadata['supportInfo']['simplifiedKey'], 'FA_SUMMARY_BINARY_DIFFERS')
    multi = 1 if metadata['coreType'] == 'MIX' and metadata.get('taskRation') == 'tilingKey' else None
    require(rows[0].get('multiKernelType') == multi, 'FA_SUMMARY_MULTIKERNEL_DISAGREES_WITH_COMPILER')
    require(op.get('simplifiedKeyMode') == 0 and op.get('optionalInputMode') == 'gen_placeholder'
            and op.get('optionalOutputMode') == 'gen_placeholder', 'FA_SUMMARY_MODE_DIFFERS')
    return index


def register_index(source, vendor):
    require(os.name == 'posix', 'LINUX_OFFICIAL_REGISTRATION_REQUIRED')
    helper = Path(__file__).resolve().parents[1] / 'fia/binary_registration.py'
    spec = importlib.util.spec_from_file_location('dcat_fa_official_generator_loader', helper)
    module = importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    vendor = Path(vendor).resolve(strict=True)
    root = vendor / 'op_impl/ai_core/tbe/kernel'
    metadata_path = root / ('ascend910b/flash_attention_score/' + BINARY + '.json')
    binary = metadata_path.with_suffix('.o')
    registry = root / 'config/ascend910b/flash_attention_score.json'
    output = registry.with_name('binary_info_config.json')
    require(not output.exists() and not output.is_symlink(), 'FA_SUMMARY_ALREADY_EXISTS')
    protected = {p: sha(p) for p in (metadata_path, binary, registry)}
    metadata = json.loads(metadata_path.read_bytes());validate_metadata(metadata, binary)
    generated = {'binary_info_config.json': {}}
    module.load_official(source).gen_ops_config(str(metadata_path), 'ascend910b', 'binary_info_config.json', generated)
    index = validate_generated(generated, json.loads(registry.read_bytes()), metadata)
    require(all(sha(p) == digest for p, digest in protected.items()), 'FA_BUILD_INPUT_CHANGED_DURING_REGISTRATION')
    with output.open('x', encoding='utf-8') as stream:
        json.dump(index, stream, indent=2, allow_nan=False);stream.write('\n')
    return dict(schema='demoncat.fa.binary_summary_registration.v1', status='REGISTERED_REQUIRES_NPU_VALIDATION',
        operator='FlashAttentionScore', compiled_keys=[KEY], generated_index=dict(path=output.relative_to(vendor).as_posix(),
        sha256=sha(output), bytes=output.stat().st_size), generator_sha256=module.SCRIPT_SHA,
        constants_sha256=module.CONSTANTS_SHA, metadata_sha256=protected[metadata_path], binary_sha256=protected[binary],
        existing_registry_sha256=protected[registry], host_kernel_bytes_changed=False, old_binaries_used=False,
        npu_executed=False, actual_launch_verified=False, production_loading_allowed=False)
