"""E4 build and final launch checks for the existing exact operator routes."""
from dataclasses import dataclass
import hashlib
from pathlib import Path
from types import MappingProxyType

from fault_injection.core import UnsupportedCall
from fault_injection.native_build import NativeBuildManager, resolve_recipe
from native_execution_strategy.certificates import strict_json


@dataclass(frozen=True)
class KernelPath:
    operator: str
    change: str
    modification_site: str
    build_path: str | None
    expected_cann: str | None


KERNEL_PATHS = MappingProxyType({
    '17.spatial_padding': KernelPath('torch.nn.functional.conv3d', 'host_parameter',
        'Conv3d padding argument with shape guard', None, None),
    '28.fa.legal_tiling': KernelPath('FlashAttentionScore', 'tiling_computation',
        'FA S1S2 native route, selected M tile and/or N grouping', 'FA strategy certificate', 'V100R001C11SPC001B243'),
    '28.fia.legal_tiling': KernelPath('FusedInferAttentionScore', 'tiling_computation',
        'FIA host derivation of M/S2 loops, tails and final healthy tiling key', 'fresh FIA legal_tiling', '9.1'),
    '28.fia.denominator_replay': KernelPath('FusedInferAttentionScore', 'device_kernel',
        'FIA online softmax second S2 block reuses previous denominator mass', 'fresh FIA denominator_replay', '9.1'),
    '33.paged_decode': KernelPath('IncreFlashAttention', 'tiling_computation',
        'Paged IFA SInner choice on the historical BNSD route', 'Pinned Paged33 vendor', '9.1'),
    '25.reduce_scatter': KernelPath('reduce_scatter_tensor', 'reduce_measured_output',
        'Synchronized process-group SUM returns before measured delta replay', None, None),
})
PAGED33_TND_PATH = KernelPath('FusedInferAttentionScore / FAInferTiling', 'tiling_computation',
    'TND page16 S2 compiled specialization with matching workspace', 'fresh FIA tnd_page16_fai_s2_256', '8.5.1')


def verify_native_build(mechanism, operator, artifact, *, pid=None, manager=None):
    """Bind an existing standard build and optional live process maps to the route."""
    recipe = resolve_recipe(mechanism, operator)
    if manager is None:
        manager = NativeBuildManager()
    build = manager.verify(artifact)
    if (build['mechanism'], build['operator'], build['recipe']) != (mechanism, recipe.operator, recipe.id):
        raise UnsupportedCall('OPERATOR_NATIVE_BUILD_RECEIPT_DOES_NOT_MATCH')
    if pid is None:
        return dict(build=build, loaded=None, execution_verified=False)
    loaded = manager.verify_loaded(artifact, pid=pid)
    if loaded.get('build_key') != build['build_key']:
        raise UnsupportedCall('OPERATOR_NATIVE_BUILD_CHANGED_DURING_LOAD_CHECK')
    return dict(build=build, loaded=loaded, execution_verified=False)


def verify_fa_launch(adapter, context, record):
    """FA uses its own S1S2 certificate, POD decoder and scoped native receipt."""
    from native_execution_strategy.adapters.flash_attention import FlashAttentionAdapter

    if not isinstance(adapter, FlashAttentionAdapter) or context.operator != adapter.name:
        raise UnsupportedCall('FA_S1S2_ADAPTER_AND_CONTEXT_REQUIRED')
    if not isinstance(record.get('host_event'), dict) or not isinstance(record.get('native_launch'), dict):
        raise UnsupportedCall('FA_CERTIFIED_HOST_AND_NATIVE_RECORD_REQUIRED')
    observation = adapter.parse_native_evidence(context, record)
    return dict(mechanism='28.fa.legal_tiling', operator=adapter.name,
                certificate_runtime=dict(adapter.certificate['runtime']),
                build_identity=observation.fingerprints.as_dict(),
                final_pod_sha256=observation.raw_sha256,
                strategy=observation.strategy.as_dict(), numerical_effect=None,
                public_qualification_granted=False)


def verify_fia_launch(mechanism, *, phase, completion, completion_sha256, vendor,
                      public_schema, schema_sha256, runtime_directory, host):
    """Verify the fresh FIA vendor and observed final call without opening a device."""
    from experiments.engineering_20261117 import fresh_fia_qualification as fresh
    from fault_injection.native_adapters.attention28.launch import bundle_kernel_abi, verify_launch

    if mechanism not in ('28.fia.legal_tiling', '28.fia.denominator_replay'):
        raise UnsupportedCall('FIA_MECHANISM_REQUIRED')
    if phase not in ('baseline', 'fault', 'off', 'restore'):
        raise UnsupportedCall('FIA_CANDIDATE_PHASE_REQUIRED')
    build = fresh.verify_build(completion, completion_sha256, vendor)
    if build['variant'] != mechanism.removeprefix('28.fia.'):
        raise UnsupportedCall('FIA_BUILD_VARIANT_DOES_NOT_MATCH')
    raw_schema = Path(public_schema).read_bytes()
    if hashlib.sha256(raw_schema).hexdigest() != schema_sha256:
        raise UnsupportedCall('FIA_PUBLIC_SCHEMA_HASH_DIFFERS')
    schema = strict_json(raw_schema, 'FIA public serializer schema')
    mode, query_tile, key_tile, key = fresh.plan(build['variant'], phase)
    if (host.get('mode') != mode or host.get('eligible') is not True
            or host.get('selected_m') != query_tile or int(host.get('tiling_key', 0)) != key
            or host.get('serializer_roundtrip_matches') is not True
            or host.get('raw_unchanged_by_observer') is not True
            or host.get('kernel_argument_raw_observed') is not False):
        raise UnsupportedCall('FIA_ACTUAL_HOST_ROUTE_DOES_NOT_MATCH_PHASE')
    decoded = fresh.decode_host(host, schema)
    if (decoded.get('innerSplitParams.mBaseSize'), decoded.get('innerSplitParams.s2BaseSize')) != (query_tile, key_tile):
        raise UnsupportedCall('FIA_HOST_M_S2_DO_NOT_MATCH_PHASE')
    binary = build['kernel_binary_sha256']
    metadata, abi = bundle_kernel_abi(build['vendor_root'], build['files'], binary)
    launch = verify_launch(runtime_directory, host['token'], kernel_sha256=binary, schema=schema,
                           host=host, kernel_metadata=metadata, kernel_abi=abi)
    return dict(mechanism=mechanism, operator=KERNEL_PATHS[mechanism].operator, phase=phase,
                build_identity=dict(completion_sha256=completion_sha256, schema_sha256=schema_sha256,
                                    kernel_sha256=binary), launch=launch, new_npu_calls=0,
                numerical_effect=None, public_qualification_granted=False)


def verify_paged33_tnd_launch(*, phase, kv_length, completion, completion_sha256, vendor,
                              public_schema, schema_sha256, runtime_directory, host):
    """Check the independent TND/page16 FAInferTiling route offline."""
    from demoncat.runtime.vllm_ascend.worker_extension.paged33_tnd import verify_bundle, validate_fai_launch
    from fault_injection.native_adapters.attention28.launch import bundle_kernel_abi, verify_launch

    if phase not in ('baseline', 'fault', 'off') or type(kv_length) is not int or not 1025 <= kv_length <= 2048:
        raise UnsupportedCall('TND33_PHASE_OR_LIVE_KV_LENGTH_OUT_OF_SCOPE')
    build = verify_bundle(completion, completion_sha256, vendor)
    raw_schema = Path(public_schema).read_bytes()
    if hashlib.sha256(raw_schema).hexdigest() != schema_sha256 or schema_sha256 != build['completion'].get('fai_public_schema_sha256'):
        raise UnsupportedCall('TND33_PUBLIC_SCHEMA_HASH_DIFFERS')
    schema = strict_json(raw_schema, 'TND33 public serializer schema')
    if schema.get('schema') != 'demoncat.paged33.fai_public_serializer.v1' or schema.get('type') != 'FAInferTilingData':
        raise UnsupportedCall('TND33_ACTUAL_FAI_PUBLIC_SCHEMA_REQUIRED')
    fault = phase == 'fault'
    if (host.get('eligible') is not True or host.get('policy_applied') is not fault
            or host.get('mode') != ('s2_256' if fault else 'off')
            or host.get('native_s2') != 512 or host.get('selected_s2') != (256 if fault else 512)):
        raise UnsupportedCall('TND33_ACTUAL_HOST_STRATEGY_DIFFERS')
    binary = build['completion']['kernel_binary_sha256']
    metadata, abi = bundle_kernel_abi(build['vendor'], build['artifacts'], binary)
    launch = verify_launch(runtime_directory, host['token'], kernel_sha256=binary, schema=schema,
                           host=host, kernel_metadata=metadata, kernel_abi=abi)
    geometry = validate_fai_launch(launch, host, arm=phase, kv_length=kv_length)
    return dict(mechanism='33.paged_decode', operator=PAGED33_TND_PATH.operator, phase=phase,
                build_identity=dict(completion_sha256=completion_sha256, schema_sha256=schema_sha256,
                                    kernel_sha256=binary), launch=launch, geometry=geometry,
                new_npu_calls=0, numerical_effect=None, public_qualification_granted=False)
