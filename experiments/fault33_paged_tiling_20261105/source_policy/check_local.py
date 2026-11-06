"""Local CPU source checks and explicit source/version receipt; never compiles/runs NPU."""
import ast
import contextlib
import hashlib
import io
import json
from pathlib import Path
import re
import unittest

import apply_policy
import make_abi_decoder
import make_offline_contract
import test_source_policy


def main():
    here = Path(__file__).resolve().parent
    for p in here.glob('*.py'):
        ast.parse(p.read_text(encoding='utf-8'), filename=str(p))
    expected = make_abi_decoder.generate()
    if (here/'ifa_abi_decoder.cpp').read_text(encoding='utf-8') != expected:
        raise RuntimeError('Generated ABI decoder differs from checked-in source')
    if json.loads((here/'offline_cmake_contract.json').read_text())!=make_offline_contract.generate():
        raise RuntimeError('Reviewed CMake dependency contract differs from source fixtures')
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(test_source_policy.PatchChecks)
    log = io.StringIO()
    with contextlib.redirect_stdout(log):
        result = unittest.TextTestRunner(stream=log, verbosity=2).run(suite)
    if not result.wasSuccessful():
        raise RuntimeError(log.getvalue())
    source = here.parent/'source_matching'
    paths = [apply_policy.ENTRY,
        'attention/incre_flash_attention/op_host/incre_flash_attention_tiling.h',
        'attention/incre_flash_attention/op_host/incre_flash_attention_tiling_check.cpp',
        'attention/incre_flash_attention/op_host/incre_flash_attention_tiling_base.h',
        'attention/incre_flash_attention/op_kernel/incre_flash_attention_tilingkey.h',
        'attention/incre_flash_attention/op_kernel/incre_flash_attention_arch22.h',
        'attention/incre_flash_attention/op_kernel/arch22/incre_flash_attention_split_Bbn2s2_Us2.h']
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    projected = len(re.findall(r'fields.push_back\(Probe', expected))+1
    receipt = {'schema': 'fault33.pa.source_version.v1', 'source_release': 'v9.1.0',
        'source_commit': apply_policy.COMMIT, 'status': 'PREPARED_LOCAL_CPU_CHECKS_PASS_SDK_BUILD_PENDING',
        'policy_kind': 'controlled_current_source_host_selector_not_historical_stock_version',
        'matched_source_files': [{'path': p, 'sha256': sha(source/p)} for p in paths],
        'policy_files': [{'path': p.name, 'sha256': sha(p)} for p in sorted(here.iterdir())
            if p.is_file() and p.name not in ('source_version.json',)],
        'local_checks': {'python_syntax': 'PASS', 'source_anchor_and_order_tests': result.testsRun,
            'generated_abi_projection_consistency': 'PASS', 'projected_fields': projected,
            'sdk_compilation_executed': False, 'npu_executed': False},
        'scope': {'batch': 8, 'q_heads': 16, 'kv_heads': 8, 'q_length': 1, 'kv_length': 16384,
            'head_dim': 128, 'block_size': 128, 'dtype': 'float16', 'inner_precise': 0,
            'modes': ['off', 'inner1024', 'inner512'], 'flash_decode': False},
        'raw_observation': 'final_public_TilingContext_only_device_argument_not_directly_observed',
        'abi_decoder': 'public_SaveToBuffer_derived_projection_not_claimed_kernel_POD'}
    (here/'source_version.json').write_text(json.dumps(receipt, indent=2)+'\n', encoding='utf-8')
    print(json.dumps({'status': receipt['status'], 'tests': result.testsRun,
        'projected_fields': projected, 'version_receipt_sha256': sha(here/'source_version.json')}))


if __name__ == '__main__':
    main()
