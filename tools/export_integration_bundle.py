"""Export a pinned, source-only Worker Core integration bundle using stdlib only.

No packaged module is imported, no build is run, and no serving/device state is
queried. A caller-supplied review reference identifies the approved snapshot;
the exporter checks bytes and scope, not the truth of that review reference.
"""

import argparse
import ast
import gzip
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
import tarfile


POLICY = 'demoncat.worker_core.source_only.v2'
ARCHIVE_ROOT = 'demoncatpfi-integration'
MAX_FILE = 16 * 1024 * 1024
MAX_TOTAL = 128 * 1024 * 1024
SECTIONS = ('contracts', 'worker_extension', 'providers', 'native_build',
            'adapters', 'tests', 'integration_guide', 'compatibility_matrix')
_DENIED = {'.git', '.codex', '.codex-tmp', '.publish', '__pycache__', '.local',
           'artifacts', 'evidence', 'reports', 'logs', 'history', 'data', 'weights',
           'checkpoints', 'datasets', 'statistics', 'dump', 'dumps', 'dist', 'build',
           'vendor', 'third_party', 'node_modules'}
_SUFFIXES = {'.py', '.cpp', '.h', '.inc', '.sh', '.md'}
_TREES = ('fault_injection', 'src/demoncat/runtime', 'fault39/src/demoncat_fault39',
          'fault39/native', 'native_execution_strategy', 'tests/native')
_JSON = {
    'fault_injection/native_adapters/attention28/source_import.json',
    'fault_injection/native_adapters/attention28/fia/build_contract.json',
    'fault_injection/native_adapters/attention28/fia/fia_abi_decoder.provenance.json',
    'native_execution_strategy/adapters/fa_cann91_schema.json',
    'native_execution_strategy/native/tcube_fields.json',
    'native_execution_strategy/native/cpu_fixture_plan.json',
}
_DISTRIBUTIONS = {
    'pyproject.toml': ('demoncatpfi', '.'),
    'src/pyproject.toml': ('demoncat-worker-extension', 'src'),
    'fault39/pyproject.toml': ('demoncat-fault39-core', 'fault39'),
}
_LEGACY = 'fault20_fast_validation/source/'
_OBSERVERS = {'engineering_closure/__init__.py', 'engineering_closure/path_discovery.py',
              'engineering_closure/native_observer.py'}
_CORE_DOCS = {'docs/worker_core_v1/README.md', 'docs/worker_core_v1/census_contract.md'}
_CENSUS = 'src/demoncat/runtime/vllm_ascend/worker_extension/call_census.py'
_ROPE_PROVIDER = 'src/demoncat/runtime/vllm_ascend/worker_extension/providers/fault22.py'
_ROPE_CONTRACTS = {'tests/worker_extension/rope22_fixture.py',
                   'docs/worker_core_v1/rope22_host_contract.md'}
_KV41_PROVIDER = 'src/demoncat/runtime/vllm_ascend/worker_extension/providers/fault41.py'
_KV41_CONTRACTS = {'tests/worker_extension/kv41_fixture.py',
                   'tests/test_postwrite_natural_kv.py', 'tests/test_postwrite_providers.py',
                   'tests/test_postwrite_native_recovery.py',
                   'docs/worker_core_v1/kv41_host_contract.md'}
_OUTPUT39_PROVIDER = 'src/demoncat/runtime/vllm_ascend/worker_extension/providers/fault39_output.py'
_OUTPUT39_CONTRACTS = {'tests/test_postwrite_providers.py',
    'tests/test_postwrite_native_recovery.py',
    'tests/test_postwrite_custom_gemm.py', 'tests/worker_extension/output39_custom_fixture.py',
    'tests/worker_extension/test_core_output39_custom.py',
    'tests/worker_extension/test_serving_session_lifecycle.py',
    'tests/worker_extension/test_call_binding.py', 'docs/worker_core_v1/output39_host_contract.md'}
_TENSOR_SERVING_PROVIDER = 'src/demoncat/runtime/vllm_ascend/worker_extension/providers/tensor_serving.py'
_TENSOR_SERVING_CONTRACTS = {
    'src/demoncat/runtime/vllm_ascend/worker_extension/serving_tensor_binding.py',
    'src/demoncat/runtime/vllm_ascend/worker_extension/serving_tensor_request.py',
    'docs/worker_core_v1/tensor_group_host_contract.md',
    'docs/worker_core_v1/moe_nested_mode_recovery.md',
    'tests/worker_extension/test_core_moe_mode_cleanup.py',
    'tests/test_serving_tensor_group.py', 'tests/test_serving_moe_group.py',
    'tests/test_rmsnorm_natural.py', 'tests/test_route8_index22.py',
    'tests/test_tensor_statistics_window.py', 'tests/test_natural_tensor_group.py',
    'tests/test_tensor_worker_core_probe.py',
    'experiments/c2c3_closure_20261117/natural_tensor_worker_core_probe.py',
    'experiments/c2c3_closure_20261117/prepare_tensor_worker_core.py',
}
_WRITER26_PROVIDER = 'src/demoncat/runtime/vllm_ascend/worker_extension/providers/fault26_core.py'
_WRITER26_CONTRACTS = {'tests/worker_extension/'+name+'.py' for name in (
    'test_writer26_inventory', 'test_writer26_pending_consumer', 'test_writer26_group_experiment',
    'test_writer26_fia_typed', 'test_writer26_public_return', 'test_writer26_group_build', 'test_collector_dispatch',
    'test_serving_session_lifecycle', 'test_call_binding', 'metadata_fixture')} | {
    'docs/worker_core_v1/writer26_host_contract.md', 'docs/worker_core_v1/route_boundary_audit_37bbeb4f.md',
    'experiments/c2c3_closure_20261117/providers26_build_group.py'}
_V11_ENDPOINTS = 'src/demoncat/runtime/vllm_ascend/worker_extension/endpoints.py'
_V11_CONTRACTS = {'docs/contracts/execution_adapter_contract.md',
    'docs/worker_extension/worker_endpoints.md', 'docs/worker_core_v1/attention_v11.md',
    'docs/worker_core_v1/writer26_complete_set_contract.md',
    'docs/worker_core_v1/writer26_isolated_contract.md',
    'experiments/c2c3_closure_20261117/providers26_isolated_budget.py',
    'tests/test_writer26_core_isolated_transition_probe.py',
    'tests/test_writer26_isolated_transition_budget.py',
    'tests/worker_extension/test_core_writer26_isolated_transition.py',
    'docs/worker_core_v1/writer26_qmm_dynamic_eligibility_design.md',
    'tests/worker_extension/test_core_endpoints.py',
    'tests/test_environment_mapping_diagnostics.py',
    'tests/worker_extension/test_core_writer26_environment.py',
    'docs/worker_core_v1/output39_complete_set_contract.md',
    'docs/worker_core_v1/bound_environment_contract.md',
    'tests/test_bound_environment.py', 'tests/test_native_environment_holders.py',
    'docs/worker_core_v1/attention33_arch22_normal_bridge.md',
    'docs/worker_core_v1/attention33_arch22_s2_candidate.md',
    'tests/test_paged33_arch22_s2_recipe.py',
    'tests/test_paged33_logical_binding.py',
    'docs/worker_core_v1/attention33_arch22_normal_review.md',
    'docs/worker_core_v1/attention33_arch22_source79_geometry_review.md',
    'docs/worker_core_v1/attention33_source83_reader_review.md',
    'docs/worker_core_v1/attention33_source85_capture_review.md',
    'tests/test_paged33_mapping_diagnostic.py',
    'tests/test_output39_group_native_provenance.py',
    'tests/test_arch22_reader_initialization.py',
    'tests/test_arch22_normal_repeat_review.py',
    'docs/worker_core_v1/attention28_tnd_source_design.md',
    'docs/worker_core_v1/attention28_stock_native_census.md',
    'docs/worker_core_v1/attention28_source87_raw_review.md',
    'experiments/c2c3_closure_20261117/attention28_census_raw.py',
    'experiments/c2c3_closure_20261117/review_attention28_stock_census.py',
    'tests/test_attention28_stock_census_review.py',
    'tests/worker_extension/test_attention28_review_protocol.py',
    'tests/test_quant_route_audit_source87.py',
    'tests/test_statistics_readiness.py',
    'tests/test_worker18_statistics_boundary.py',
    'experiments/worker_core_v11_20261118/review_statistics_readiness.py',
    'experiments/worker_core_v11_20261118/AGENT7_WORKER18_E5_READINESS.md',
    'experiments/worker_core_v11_20261118/core_tensor_statistics_capture.py',
    'experiments/worker_core_v11_20261118/collector_packet_compat.py',
    'experiments/worker_core_v11_20261118/core_tensor_statistics_probe.py',
    'experiments/worker_core_v11_20261118/prepare_core_tensor_statistics.py',
    'experiments/worker_core_v11_20261118/review_core_tensor_statistics.py',
    'experiments/worker_core_v11_20261118/CORE_TENSOR_STATISTICS_SIDECAR.md',
    'experiments/c2c3_closure_20261117/tensor_statistics_capture.py',
    'experiments/c2c3_closure_20261117/audit_statistics_series.py',
    'tests/test_core_tensor_statistics_capture.py',
    'tests/test_core_tensor_statistics_probe.py',
    'tests/test_core_tensor_statistics_raw_review.py',
    'tests/test_core_tensor_statistics_msprobe.py',
    'experiments/worker_core_v11_20261118/audit_quant_source87_routes.py',
    'experiments/worker_core_v11_20261118/review_quant_routed_frames.py',
    'experiments/worker_core_v11_20261118/QUANT_SOURCE93_ROUTED_FRAMES_REVIEW.md',
    'tests/worker_extension/test_quant_routed_frames_review.py',
    'experiments/worker_core_v11_20261118/QUANT_SOURCE87_INTEGER_DIAGNOSTIC.md',
    'tests/worker_extension/test_attention28_stock_census.py',
    'tests/test_attention28_stock_launch.py',
    'experiments/c2c3_closure_20261117/natural_attention28_stock_census_probe.py',
    'experiments/c2c3_closure_20261117/prepare_attention28_stock_census.py',
    'tests/test_arch22_normal_bridge_review.py',
    'experiments/c2c3_closure_20261117/arch22_normal_bridge_evidence.py',
    'experiments/c2c3_closure_20261117/prepare_arch22_normal_bridge_cases.py',
    'experiments/c2c3_closure_20261117/prepare_arch22_s2_normal_bridge_cases.py',
    'fault_injection/native_adapters/attention28/fia/arch22_s2_build_evidence.py',
    'docs/worker_core_v1/attention33_arch22_s2_normal.md',
    'tests/test_arch22_s2_normal.py',
    'docs/worker_core_v1/attention33_arch22_s2_single.md',
    'experiments/c2c3_closure_20261117/natural33_arch22_s2_worker_core_probe.py',
    'experiments/c2c3_closure_20261117/prepare_arch22_s2_single_cases.py',
    'experiments/c2c3_closure_20261117/review_arch22_s2_single.py',
    'src/demoncat/runtime/vllm_ascend/worker_extension/paged33_s2_evidence.py',
    'src/demoncat/runtime/vllm_ascend/worker_extension/paged33_s2_execution.py',
    'src/demoncat/runtime/vllm_ascend/worker_extension/paged33_arch22_s2_effects.py',
    'src/demoncat/runtime/vllm_ascend/worker_extension/providers/paged33_s2.py',
    'tests/test_arch22_s2_effects.py',
    'tests/test_arch22_s2_probe.py',
    'tests/test_arch22_s2_single_review.py',
    'tests/worker_extension/test_core_paged33_s2.py',
    'experiments/worker_core_v11_20261118/review_quant_native_route_materials.py',
    'experiments/worker_core_v11_20261118/QUANT_NATIVE_ROUTE_GATE.md',
    'tests/worker_extension/test_quant_native_route_materials.py',
    'experiments/worker_core_v11_20261118/build_quant_container_trace.py',
    'experiments/worker_core_v11_20261118/QUANT_CONTAINER_TRACE.md',
    'fault_injection/native_adapters/quant_container_trace.cpp',
    'fault_injection/native_adapters/quant_container_trace.h',
    'fault_injection/native_adapters/test_quant_container_trace.cpp',
    'tests/worker_extension/test_quant_container_build.py',
    'fault_injection/quant_container_native.py',
    'src/demoncat/runtime/vllm_ascend/worker_extension/quant_container_scopes.py',
    'experiments/worker_core_v11_20261118/prepare_quant_container_case.py',
    'experiments/worker_core_v11_20261118/probe_quant_containers.py',
    'tests/worker_extension/test_quant_containers.py',
    'experiments/worker_core_v11_20261118/output39_refusal_diagnostic.py',
    'experiments/worker_core_v11_20261118/output39_refusal_probe.py',
    'experiments/worker_core_v11_20261118/prepare_output39_refusal.py',
    'experiments/worker_core_v11_20261118/review39_refusal_diagnostic.py',
    'experiments/worker_core_v11_20261118/review39_refusal_originals.py',
    'experiments/worker_core_v11_20261118/review39_group_originals.py',
    'experiments/worker_core_v11_20261118/OUTPUT39_REFUSAL_ORIGINAL_REVIEW_REQUIREMENTS.md',
    'tests/test_output39_refusal_original_review.py',
    'experiments/worker_core_v11_20261118/review39_refusal_prefix.py',
    'tests/test_output39_refusal_prefix_review.py',
    'experiments/worker_core_v11_20261118/OUTPUT39_SOURCE101_PARTIAL_REVIEW.md',
    'experiments/worker_core_v11_20261118/output39_source101_partial_summary.json',
    'experiments/worker_core_v11_20261118/CONVERGENCE_DECISIONS.md',
    'experiments/worker_core_v11_20261118/VISION17_SOURCE77_ACCEPTANCE.md',
    'experiments/worker_core_v11_20261118/PUBLIC18_SOURCE99_STATISTICS.md',
    'experiments/worker_core_v11_20261118/public18_source99_statistics_summary.json',
    'experiments/worker_core_v11_20261118/public18_source99_runtime_controls.json',
    'experiments/worker_core_v11_20261118/OUTPUT39_FIXED_C3_REFUSAL_DIAGNOSTIC.md',
    'fault_injection/native_adapters/postwrite/test_original_snapshot.cpp',
    'tests/test_output39_original_native_source.py',
    'tests/test_output39_refusal_diagnostic.py',
    'tests/test_output39_refusal_probe.py',
    'experiments/c2c3_closure_20261117/review_arch22_normal_bridge.py',
    'tests/test_worker_core26_c3_original_review.py',
    'tests/test_worker_core26_build_binding_review.py',
    'tests/test_worker_core26_isolated_original_review.py',
    'tests/test_worker_core26_isolated_c3_original_review.py',
    'experiments/worker_core_v11_20261118/review26_isolated_originals.py',
    'experiments/worker_core_v11_20261118/review26_isolated_c3_originals.py',
    'experiments/worker_core_v11_20261118/AGENT7_NATIVE26_ISOLATED_C3_ORIGINAL_REVIEW.md',
    'experiments/worker_core_v11_20261118/AGENT7_NATIVE26_ISOLATED_ORIGINAL_REVIEW.md',
    'experiments/worker_core_v11_20261118/AGENT7_NATIVE26_SOURCE77_AUDIT_CORRECTION.md',
    'experiments/worker_core_v11_20261118/review26_c3_originals.py',
    'experiments/worker_core_v11_20261118/review26_environment_originals.py',
    'experiments/worker_core_v11_20261118/review17_worker_originals.py',
    'experiments/worker_core_v11_20261118/group_review_common.py',
    'experiments/worker_core_v11_20261118/AGENT7_VISION17_ORIGINALS_REVIEW.md',
    'tests/test_worker_core17_original_review.py',
    'tests/test_paged33_arch22_bridge.py',
    'tests/worker_extension/test_core_output39_group.py',
    'tests/worker_extension/test_core_output39_group_custom.py',
    'tests/worker_extension/test_core_output39_discovery_evidence.py',
    'tests/worker_extension/test_core_output39_direct_discovery.py',
    'docs/worker_core_v1/output39_direct_discovery_contract.md',
    'experiments/worker_core_v11_20261118/output39_failed_discovery_evidence.md',
    'tests/test_output39_group_worker_core_probe.py',
    'tests/test_worker_core39_direct_original_review.py',
    'experiments/worker_core_v11_20261118/review39_forward_originals.py',
    'experiments/worker_core_v11_20261118/AGENT7_OUTPUT39_DIRECT_ORIGINAL_REVIEW.md',
    'tests/worker_extension/test_core_vision17.py',
    'tests/worker_extension/test_core_vision17_group.py',
    'tests/test_vision17_context.py', 'tests/test_vision17_trace.py',
    'tests/test_vision17_cache_control.py',
    'experiments/worker_core_v11_20261118/vision17_cache_control.py',
    'tests/worker_extension/test_quant_embedding_rows.py',
    'tests/worker_extension/test_quant_addrms_diagnostic.py',
    'tests/worker_extension/test_quant_inference_parameters.py',
    'tests/worker_extension/test_quant_moe_rows.py',
    'tests/worker_extension/test_quant_token_routes.py',
    'tests/worker_extension/test_quant_route_frames.py',
    'experiments/worker_core_v11_20261118/QUANT_ROUTED_FRAME_OBSERVER.md',
    'experiments/worker_core_v11_20261118/quant_integer_route_observer.md',
    'experiments/worker_core_v11_20261118/quant_shared_moe_source79.md',
    'experiments/worker_core_v11_20261118/allocator26_availability.py',
    'tests/test_allocator26_availability.py',
    'experiments/worker_core_v11_20261118/AGENT7_QMM26_AVAILABILITY_REVIEW.md',
    'experiments/worker_core_v11_20261118/qmm26_availability_independent_summary.json',
    'experiments/c2c3_closure_20261117/providers_quant_v11_inference_parameters.md',
    'experiments/c2c3_closure_20261117/V11_VISION17_SCOPE_20261118.md',
    'experiments/c2c3_closure_20261117/vision17_worker_census_probe.py',
    'experiments/c2c3_closure_20261117/prepare_vision17_worker_census.py',
    'experiments/c2c3_closure_20261117/vision17_worker_group_probe.py',
    'experiments/c2c3_closure_20261117/prepare_vision17_worker_group.py',
    'experiments/c2c3_closure_20261117/V11_VISION17_GROUP_20261118.md',
    'tests/worker_extension/test_core_attention_serving.py',
    'tests/worker_extension/test_core_writer26_coverage.py',
    'tests/worker_extension/test_core_writer26_every_match.py',
    'tests/worker_extension/test_core_writer26_isolated.py',
    'tests/test_writer26_core_isolated_probe.py',
    'tests/test_writer26_core_every_match_probe.py',
    'tests/test_writer26_core_coverage_probe.py', 'tests/test_writer26_worker_core_probe.py',
    'experiments/c2c3_closure_20261117/providers26_worker_core_probe.py',
    'experiments/c2c3_closure_20261117/natural_attention_worker_core_probe.py',
} | {'tests/worker_extension/'+name+'.py' for name in (
    'test_quant_mla_view_rows', 'test_quant_mla_decode_rows', 'test_quant_source_rows',
    'test_quant_row_provenance', 'test_quant_group_c1', 'test_quant_group_inventory')}
# Current build inputs only: no whole legacy source tree or experiment history.
_SUPPORT = {
    _LEGACY + 'l4_buffer_substitute_allhit_v1/' + name for name in (
        'buffer_substitute_preload.cpp', 'buffer_control.cpp', 'fault20_buffer_api.h',
        'extreme_transform.h', 'int_equivalent_transform.h', 'mantissa_truncate.h', 'build.sh')
} | {
    _LEGACY + 'l4_metadata_lifecycle_proxy/' + name for name in ('libopapi_metadata_proxy.cpp', 'build.sh')
} | {
    _LEGACY + 'fault26_readonly_trace_v1/' + name for name in (
        'fault26_trace_shared.h', 'fault26_inventory.h', 'fault26_fia_typed.h',
        'fault26_trace_preload.cpp', 'fault26_opapi_trace_proxy.cpp',
        'fault26_qmm_underflow_api.h', 'fault26_qmm_output_underflow.cpp', 'build.sh',
        'build_fault26_qmm_output_underflow.sh')
} | {
    _LEGACY + 'fault26_fia_writer_proxy_v1/' + name for name in ('fault26_fia_writer_lossless.cpp', 'build.sh')
}
_EXACT = set(_DISTRIBUTIONS) | _SUPPORT | _OBSERVERS | _CORE_DOCS | _KV41_CONTRACTS | _OUTPUT39_CONTRACTS | _TENSOR_SERVING_CONTRACTS | _WRITER26_CONTRACTS | _V11_CONTRACTS | {
    'fault39/tools/version_profile.py', 'fault39/tools/deployment_profile.py',
    'native_execution_strategy/native/build_requirements.txt',
    'fault39/tests/test_contracts.py', 'tests/test_integration_contract.py',
    'tests/test_native_build.py', 'tests/test_integration_bundle.py',
    'tests/worker_extension/__init__.py', 'tests/worker_extension/test_extension.py',
    'tests/worker_extension/test_serving_context.py',
    'tests/worker_extension/rope22_fixture.py',
    'tests/worker_extension/metadata_fixture.py', 'tools/export_integration_bundle.py',
    'docs/worker_core_v1/host_contract.md', 'docs/worker_core_v1/integration_bundle.md',
    'docs/worker_core_v1/rope22_host_contract.md',
}
_REQUIRED = (_EXACT - _CORE_DOCS - _ROPE_CONTRACTS - _KV41_CONTRACTS - _OUTPUT39_CONTRACTS - _TENSOR_SERVING_CONTRACTS - _WRITER26_CONTRACTS - _V11_CONTRACTS) | {
    'fault_injection/integration.py', 'fault_injection/lifecycle.py',
    'fault_injection/native_build/manager.py', 'fault_injection/native_build/recipes.py',
    'fault_injection/native_adapters/build_qmm.sh',
    'fault39/src/demoncat_fault39/contracts.py', 'fault39/src/demoncat_fault39/provider.py',
    'src/demoncat/runtime/vllm_ascend/worker_extension/host.py',
    'src/demoncat/runtime/vllm_ascend/worker_extension/config.py',
    'src/demoncat/runtime/vllm_ascend/worker_extension/runtime.py',
    'src/demoncat/runtime/vllm_ascend/worker_extension/extension.py',
    'tests/worker_extension/test_core_host.py',
    'tests/worker_extension/metadata_fixture.py',
}


def _required(names):
    # Older reviewed snapshots may lack census entirely. A snapshot that ships
    # it must also ship the actual observer package, fixture and its contracts.
    return (_REQUIRED | (_CORE_DOCS if _CENSUS in names else set())
            | (_V11_CONTRACTS if _V11_ENDPOINTS in names else set())
            | (_ROPE_CONTRACTS | {'tests/worker_extension/test_core_rope22.py'}
               if _ROPE_PROVIDER in names else set())
            | (_KV41_CONTRACTS | {'tests/worker_extension/test_core_kv41.py'}
               if _KV41_PROVIDER in names else set())
            | (_OUTPUT39_CONTRACTS | {'tests/worker_extension/test_core_output39.py'}
               if _OUTPUT39_PROVIDER in names else set())
            | (_TENSOR_SERVING_CONTRACTS | {'tests/worker_extension/test_core_tensor_serving.py'}
               if _TENSOR_SERVING_PROVIDER in names else set())
            | (_WRITER26_CONTRACTS | {'tests/worker_extension/test_core_writer26.py'}
               if _WRITER26_PROVIDER in names else set()))


class BundleError(ValueError):
    pass


def _json(value):
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + '\n').encode('utf-8')


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def _path(name):
    if (not isinstance(name, str) or not name or '\\' in name or ':' in name
            or any(ord(char) < 32 for char in name)):
        raise BundleError('UNSAFE_RELATIVE_PATH')
    part = PurePosixPath(name)
    if part.is_absolute() or '..' in part.parts or part.as_posix() != name or name == '.':
        raise BundleError('UNSAFE_RELATIVE_PATH')
    return part


def _denied(parts):
    return any(part.lower() in _DENIED or part.lower().endswith('.egg-info') for part in parts)


def _allowed(name):
    part = _path(name)
    if _denied(part.parts):
        return False
    if name in _EXACT or name in _JSON:
        return True
    if (name.startswith('tests/worker_extension/test_core_') and part.suffix == '.py'
            and len(part.parts) == 3):
        return True
    if name.startswith('native_execution_strategy/tests/'):
        return False
    return (any(name.startswith(tree + '/') for tree in _TREES)
            and part.suffix in _SUFFIXES)


def _linked(path):
    details = path.lstat()
    return stat.S_ISLNK(details.st_mode) or bool(getattr(details, 'st_file_attributes', 0) & 0x400)


def _read(root, name):
    part = _path(name)
    path = root
    for component in part.parts:
        path = path / component
        if _linked(path):
            raise BundleError('LINKED_SOURCE_REFUSED: ' + name)
    if not path.is_file() or not path.resolve(strict=True).is_relative_to(root):
        raise BundleError('SOURCE_OUTSIDE_REVIEW_ROOT: ' + name)
    if path.stat().st_size > MAX_FILE:
        raise BundleError('SOURCE_FILE_TOO_LARGE: ' + name)
    with path.open('rb') as stream:
        data = stream.read(MAX_FILE + 1)
    if len(data) > MAX_FILE or b'\0' in data:
        raise BundleError('BINARY_OR_OVERSIZED_SOURCE: ' + name)
    try:
        data.decode('utf-8')
    except UnicodeDecodeError as error:
        raise BundleError('NON_UTF8_SOURCE: ' + name) from error
    return data


def _inventory(source_root):
    root = Path(source_root).resolve(strict=True)
    if not root.is_dir():
        raise BundleError('SOURCE_DIRECTORY_REQUIRED')
    names = {name for name in _EXACT if (root / name).exists()}
    for tree in (*_TREES, 'tests/worker_extension'):
        base = root / tree
        if not base.is_dir():
            raise BundleError('SOURCE_TREE_MISSING: ' + tree)
        # Never enumerate unrelated repositories, experiments or data trees.
        for directory, children, files in os.walk(base, followlinks=False):
            relative = Path(directory).relative_to(root)
            if _linked(Path(directory)):
                raise BundleError('LINKED_SOURCE_REFUSED: ' + relative.as_posix())
            children[:] = sorted(child for child in children
                                  if not _denied((*relative.parts, child))
                                  and (*relative.parts, child) != ('native_execution_strategy', 'tests'))
            for child in children:
                if _linked(Path(directory) / child):
                    raise BundleError('LINKED_SOURCE_REFUSED: ' + (relative / child).as_posix())
            names.update((relative / name).as_posix() for name in files
                         if _allowed((relative / name).as_posix()))
    missing = _required(names) - names
    if missing:
        raise BundleError('REQUIRED_SOURCE_MISSING: ' + ', '.join(sorted(missing)))
    payload = {name: _read(root, name) for name in sorted(names) if _allowed(name)}
    if sum(map(len, payload.values())) > MAX_TOTAL:
        raise BundleError('SOURCE_SNAPSHOT_TOO_LARGE')
    if payload['tools/export_integration_bundle.py'] != Path(__file__).read_bytes():
        raise BundleError('EXPORTER_SOURCE_VERSION_MISMATCH')
    return payload


def _distributions(payload):
    result = []
    for project, (expected, root) in _DISTRIBUTIONS.items():
        text = payload[project].decode('utf-8')
        block = re.search(r'(?ms)^\[project\]\s*\n(.*?)(?=^\[|\Z)', text)
        fields = {} if block is None else dict(re.findall(
            r'(?m)^\s*(name|version)\s*=\s*[\"\']([^\"\'\n]+)[\"\']\s*$', block[1]))
        if fields.get('name') != expected or not fields.get('version'):
            raise BundleError('DISTRIBUTION_METADATA_MISMATCH: ' + project)
        result.append(dict(name=expected, version=fields['version'], source_root=root,
                           bundle_root='sources' if root == '.' else 'sources/' + root,
                           pyproject=project))
    if _CENSUS in payload:
        block = re.search(r'(?ms)^\[tool\.setuptools\.packages\.find\]\s*\n(.*?)(?=^\[|\Z)',
                          payload['pyproject.toml'].decode('utf-8'))
        include = None if block is None else re.search(r'(?m)^include\s*=\s*(\[[^\n]*\])\s*$', block[1])
        try:
            patterns = [] if include is None else ast.literal_eval(include[1])
        except (SyntaxError, ValueError):
            patterns = []
        if not isinstance(patterns, list) or 'engineering_closure*' not in patterns:
            raise BundleError('CENSUS_OBSERVER_PACKAGE_NOT_IN_ROOT_DISTRIBUTION')
    return result


def create_review(source_root, review_id):
    if not isinstance(review_id, str) or not review_id.strip() or len(review_id) > 200:
        raise BundleError('EXPLICIT_SOURCE_REVIEW_REFERENCE_REQUIRED')
    payload = _inventory(source_root)
    return dict(schema='demoncat.integration.source_review.v1', policy=POLICY,
                review_id=review_id, review_reference_is_caller_supplied=True,
                distributions=_distributions(payload),
                files=[dict(source_path=name, bytes=len(data), sha256=_sha(data))
                       for name, data in payload.items()])


def _validate_review(review):
    if (not isinstance(review, dict) or review.get('schema') != 'demoncat.integration.source_review.v1'
            or set(review) != {'schema', 'policy', 'review_id', 'review_reference_is_caller_supplied',
                               'distributions', 'files'}
            or review.get('policy') != POLICY or review.get('review_reference_is_caller_supplied') is not True
            or not isinstance(review.get('review_id'), str) or not review['review_id'].strip()
            or len(review['review_id']) > 200
            or not isinstance(review.get('files'), list) or len(review['files']) > 4000):
        raise BundleError('INVALID_SOURCE_REVIEW')
    distributions = review.get('distributions')
    if not isinstance(distributions, list) or len(distributions) != len(_DISTRIBUTIONS):
        raise BundleError('INVALID_DISTRIBUTION_REVIEW')
    for row, (project, (name, root)) in zip(distributions, _DISTRIBUTIONS.items()):
        if (not isinstance(row, dict) or set(row) != {'name', 'version', 'source_root', 'bundle_root', 'pyproject'}
                or row.get('name') != name or row.get('pyproject') != project
                or row.get('source_root') != root
                or row.get('bundle_root') != ('sources' if root == '.' else 'sources/' + root)
                or not isinstance(row.get('version'), str) or not row['version']):
            raise BundleError('INVALID_DISTRIBUTION_REVIEW')
    names = []
    for row in review['files']:
        if (not isinstance(row, dict) or set(row) != {'source_path', 'bytes', 'sha256'}
                or not _allowed(row['source_path']) or type(row['bytes']) is not int
                or not 0 <= row['bytes'] <= MAX_FILE
                or not isinstance(row['sha256'], str) or re.fullmatch('[a-f0-9]{64}', row['sha256']) is None):
            raise BundleError('INVALID_REVIEWED_SOURCE_ENTRY')
        names.append(row['source_path'])
    if (names != sorted(set(names)) or len({name.lower() for name in names}) != len(names)
            or not _required(names) <= set(names)):
        raise BundleError('REVIEWED_SOURCE_SET_INVALID')
    if sum(row['bytes'] for row in review['files']) > MAX_TOTAL:
        raise BundleError('SOURCE_SNAPSHOT_TOO_LARGE')


def _sections(name):
    sections = set()
    if name.startswith('src/demoncat/runtime/'):
        sections.add('worker_extension')
    if name in _OBSERVERS:
        sections.update(('worker_extension', 'adapters'))
    if '/providers/' in name or '/deployments/' in name or name.endswith(('/providers.py', '/provider.py', '/backend.py')):
        sections.add('providers')
    if (name.startswith(('fault_injection/native_', 'fault39/native/', 'fault39/tools/', _LEGACY,
                         'native_execution_strategy/native/', 'tests/native/'))):
        sections.add('native_build')
    if (name.startswith(('fault_injection/', 'fault39/src/', 'native_execution_strategy/'))
            or name.endswith(('/host.py', '/config.py'))):
        sections.add('adapters')
    if name.startswith(('tests/', 'fault39/tests/')):
        sections.add('tests')
    if name.endswith(('/integration.py', '/contracts.py', '/schema.py', '/host.py', '/config.py')) or name.endswith('_api.h'):
        sections.add('contracts')
    if name.startswith('docs/') or name in _DISTRIBUTIONS or name.startswith('tools/'):
        sections.add('integration_guide')
    if name in _TENSOR_SERVING_CONTRACTS and name.startswith('experiments/'):
        sections.add('integration_guide')
    if name in _WRITER26_CONTRACTS and name.startswith('experiments/'):
        sections.update(('integration_guide', 'native_build'))
    if name.endswith(('/environment.py', '/recipes.py', '/version_profile.py')):
        sections.add('compatibility_matrix')
    return sorted(sections)


_GUIDE = """# Source integration handoff

This bundle preserves one canonical implementation under `sources/`. The eight
section directories contain entrypoint/source indexes, not replacement code.
Three original pyproject.toml files define demoncatpfi, demoncat-worker-extension
and demoncat-fault39-core. Build/install them independently from sources/,
sources/src/ and sources/fault39/ using your approved Python packaging tools.

The inner-framework Registry/Controller implementation is unavailable here:
INNER_FRAMEWORK_INTEGRATION_PENDING. Map the two existing HostContracts and
Worker Core HostAdapter to your actual host. Serving owns Worker creation,
startup and shutdown. Keep Worker RPC nonce/owner/generation, one-attempt rules,
scope checks and owner retention on uncertain/failed recovery unchanged.

The current Worker source may include additional discovery/census modules from
the explicitly reviewed source root. Inventory and C0 observations never grant
runtime or native qualification. Read sources/docs/worker_core_v1/host_contract.md.
When census is present, the package also includes the exact existing
engineering_closure observer modules, the serving-context CPU fixture, and
sources/docs/worker_core_v1/{README,census_contract}.md. Unrelated engineering
history, queues, execution receipts and reports are not included.

NativeBuildManager remains fault_injection.native_build.NativeBuildManager.
Pass --repository pointing to sources/ when installed outside this source tree.
Current recipe dependencies retain their original relative paths, including
explicitly selected legacy C++/header/build inputs. No CANN/vendor libraries,
prebuilt .so, tensors, statistics, experiment reports or historical certificates
are included. Obtain environment-specific dependencies through the host's
approved deployment and produce new build fingerprints. A successful build is
not runtime qualification. No native build runs during export or verification.

CANN 8.5.1/A2 has mechanism-specific prior evidence only; this artifact grants
no acceptance. CANN 9.x and A3 remain NOT_ACCEPTED_BY_THIS_BUNDLE. Compatibility
must be established for an exact build/environment/operator/Worker instance.
C1/C2/C3, resource recovery, healthy-output recovery and independent E5 each
require their own current evidence. This is not a 17/17 completion statement.

Native strategy certificate files and their historical evidence are excluded.
Provide independently reviewed certificates and bound evidence explicitly;
absence must continue to refuse arming rather than use historical acceptance.

Before extraction, verify against the separately retained source review file:
python tools/export_integration_bundle.py verify --archive bundle.tar.gz --review review.json
The archive SHA returned by export should also be retained independently.
The verifier does not extract or execute archive entries. Hash verification
establishes integrity against supplied review bytes, not authorship or approval.
"""


def _generated(review):
    entries = [dict(row, bundle_path='sources/' + row['source_path'], sections=_sections(row['source_path']))
               for row in review['files']]
    payload = {'source_review.json': _json(review), 'integration_guide/README.md': _GUIDE.encode('utf-8')}
    for section in SECTIONS:
        payload[section + '/index.json'] = _json(dict(
            schema='demoncat.integration.source_index.v1', section=section,
            files=[row for row in entries if section in row['sections']],
            implementation_location='sources/', copied_parallel_implementation=False))
    payload['contracts/entrypoints.json'] = _json(dict(
        catalog=['fault_injection.integration.HostContract', 'demoncat_fault39.contracts.HostContract'],
        host='demoncat.runtime.vllm_ascend.worker_extension.host.HostAdapter',
        client='demoncat.runtime.vllm_ascend.worker_extension.host.WorkerFaultClient',
        config='demoncat.runtime.vllm_ascend.worker_extension.config.FaultConfig',
        lifecycle_authority='demoncat.runtime.vllm_ascend.worker_extension.runtime.WorkerFaultRuntime',
        native_build='fault_injection.native_build.NativeBuildManager',
        source_distributions=review['distributions'], inner_framework_integration='PENDING',
        serving_lifecycle_owned_by='vLLM/vLLM-Ascend', automatic_fault_retries=False))
    payload['compatibility_matrix/status.json'] = _json(dict(
        schema='demoncat.integration.compatibility.v1', qualification_inherited=False,
        cann_8_5_1_a2='MECHANISM_SPECIFIC_EVIDENCE_NOT_BUNDLED',
        cann_9_x='NOT_ACCEPTED_BY_THIS_BUNDLE', a3='NOT_ACCEPTED_BY_THIS_BUNDLE',
        inner_framework_integration='PENDING', all_17_accepted=False,
        assessments={name: 'NOT_ASSESSED' for name in ('C1', 'C2', 'C3', 'Recovery', 'HealthyOutput', 'E5')}))
    payload['tests/README.md'] = (
        '# Relocated CPU contracts\n\n'
        'From sources/, set PYTHONPATH to src, fault39/src, the current directory, '
        'tests and tests/worker_extension using the platform path separator.\n\n'
        'Run `python -B -m unittest discover -s tests/worker_extension -p "test_core_*.py"`.\n'
        'The serving-context fixture and the exact engineering_closure observer modules are included.\n'
        'Existing host contracts: `python -B -m unittest discover -s tests -p test_integration_contract.py`.\n'
        'Existing Fault39: `python -B -m unittest discover -s fault39/tests -p test_contracts.py`.\n'
        'CPU readback fixtures grant no NPU or compatibility acceptance.\n'
    ).encode('utf-8')
    return payload


def _manifest(review, payload):
    source_paths = {'sources/' + row['source_path']: row['source_path'] for row in review['files']}
    return dict(schema='demoncat.integration.bundle.v1', policy=POLICY,
                source_review_sha256=_sha(_json(review)), review_id=review['review_id'],
                source_only=True, native_build_executed=False, npu_execution=False,
                files=[dict(path=name, source_path=source_paths.get(name), bytes=len(data), sha256=_sha(data))
                       for name, data in sorted(payload.items())])


def export_bundle(source_root, review, output):
    _validate_review(review)
    expected = create_review(source_root, review['review_id'])
    if _json(review) != _json(expected):
        raise BundleError('REVIEWED_SOURCE_CHANGED')
    sources = _inventory(source_root)
    for row in review['files']:
        data = sources[row['source_path']]
        if len(data) != row['bytes'] or _sha(data) != row['sha256']:
            raise BundleError('SOURCE_CHANGED_DURING_EXPORT')
    if set(sources) != {row['source_path'] for row in review['files']}:
        raise BundleError('SOURCE_SET_CHANGED_DURING_EXPORT')
    payload = {'sources/' + name: data for name, data in sources.items()}
    payload.update(_generated(review))
    payload['source_manifest.json'] = _json(_manifest(review, payload))
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation preserves any earlier artifact. On failure only this
    # newly created file is removed; no directory or source path is deleted.
    with output.open('xb') as raw:
        try:
            with gzip.GzipFile(filename='', mode='wb', fileobj=raw, mtime=0) as compressed:
                with tarfile.open(fileobj=compressed, mode='w', format=tarfile.PAX_FORMAT) as archive:
                    for name, data in sorted(payload.items()):
                        item = tarfile.TarInfo(ARCHIVE_ROOT + '/' + name)
                        item.size, item.mtime = len(data), 0
                        item.mode = 0o755 if name.endswith('.sh') else 0o644
                        archive.addfile(item, io.BytesIO(data))
        except BaseException:
            raw.close()
            output.unlink()
            raise
    return verify_bundle(output, review)


def verify_bundle(archive_path, review):
    _validate_review(review)
    if Path(archive_path).stat().st_size > MAX_TOTAL:
        raise BundleError('ARCHIVE_SIZE_LIMIT')
    generated = _generated(review)
    expected = {'sources/' + row['source_path']: row for row in review['files']}
    payload = {}
    total = 0
    with tarfile.open(archive_path, mode='r:gz') as archive:
        for item in archive:
            part = _path(item.name)
            if part.parts[0] != ARCHIVE_ROOT or not item.isfile() or len(part.parts) < 2:
                raise BundleError('UNSAFE_ARCHIVE_ENTRY')
            name = PurePosixPath(*part.parts[1:]).as_posix()
            if name in payload or name not in set(expected) | set(generated) | {'source_manifest.json'}:
                raise BundleError('UNEXPECTED_OR_DUPLICATE_ARCHIVE_ENTRY')
            total += item.size
            if item.size < 0 or item.size > MAX_FILE or total > MAX_TOTAL:
                raise BundleError('ARCHIVE_SIZE_LIMIT')
            data = archive.extractfile(item).read(MAX_FILE + 1)
            if len(data) != item.size:
                raise BundleError('TRUNCATED_ARCHIVE_ENTRY')
            if name in expected:
                row = expected[name]
                if len(data) != row['bytes'] or _sha(data) != row['sha256']:
                    raise BundleError('SOURCE_HASH_MISMATCH: ' + name)
            elif name in generated and data != generated[name]:
                raise BundleError('GENERATED_CONTRACT_MISMATCH: ' + name)
            payload[name] = data
    if set(payload) != set(expected) | set(generated) | {'source_manifest.json'}:
        raise BundleError('ARCHIVE_ENTRY_MISSING')
    manifest = payload.pop('source_manifest.json')
    if manifest != _json(_manifest(review, payload)):
        raise BundleError('SOURCE_MANIFEST_MISMATCH')
    source_payload = {row['source_path']: payload['sources/' + row['source_path']] for row in review['files']}
    if _distributions(source_payload) != review['distributions']:
        raise BundleError('DISTRIBUTION_METADATA_MISMATCH')
    return dict(status='SOURCE_BUNDLE_VERIFIED', archive_sha256=_sha(Path(archive_path).read_bytes()),
                source_files=len(expected), files=len(payload) + 1,
                source_review_sha256=_sha(_json(review)), qualification_inherited=False,
                inner_framework_integration='PENDING', actual_npu_calls=0)


def _load_review(path):
    with Path(path).open('rb') as stream:
        data = stream.read(4 * 1024 * 1024 + 1)
    if len(data) > 4 * 1024 * 1024:
        raise BundleError('REVIEW_FILE_TOO_LARGE')
    return json.loads(data)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='action', required=True)
    plan = commands.add_parser('plan', help='Freeze candidate bytes for caller review')
    plan.add_argument('--source-root', type=Path, required=True)
    plan.add_argument('--review-id', required=True)
    plan.add_argument('--output', type=Path, required=True)
    export = commands.add_parser('export')
    export.add_argument('--source-root', type=Path, required=True)
    export.add_argument('--review', type=Path, required=True)
    export.add_argument('--output', type=Path, required=True)
    verify = commands.add_parser('verify')
    verify.add_argument('--archive', type=Path, required=True)
    verify.add_argument('--review', type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.action == 'plan':
            review = create_review(args.source_root, args.review_id)
            args.output.parent.mkdir(parents=True, exist_ok=True)
            with args.output.open('xb') as stream:
                stream.write(_json(review))
            result = dict(status='SOURCE_SNAPSHOT_RECORDED_FOR_REVIEW', files=len(review['files']),
                          review_sha256=_sha(_json(review)), review_reference_is_caller_supplied=True)
        elif args.action == 'export':
            result = export_bundle(args.source_root, _load_review(args.review), args.output)
        else:
            result = verify_bundle(args.archive, _load_review(args.review))
    except (BundleError, OSError, ValueError, tarfile.TarError) as error:
        print(json.dumps(dict(status='REFUSED', reason=str(error), actual_npu_calls=0)))
        return 2
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
