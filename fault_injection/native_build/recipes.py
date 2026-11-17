"""Explicit operator-specific build recipes; no filename/version heuristics."""
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class Recipe:
    id: str
    mechanism: str
    operator: str
    backend: str
    script: str
    output_variable: str
    runtime: str
    runtime_symbols: tuple
    source_roots: tuple
    api_headers: tuple
    operator_exports: tuple
    allowed_versions: tuple = ()
    requires_custom: bool = False
    requires_pta_wheel: bool = False
    contract_version: str = "1"

    def as_dict(self):
        return asdict(self)


_COMMON_SOURCES = ("fault_injection/native_build", "fault_injection/environment.py")
_PRIVATE_HEADERS = ("include/acl/acl.h", "include/aclnn/acl_meta.h", "include/aclnn/opdev/op_cache.h")
_F20_SOURCES = (
    "fault_injection/native_adapters", "fault_injection/deployments/pta_resolver.py",
    "fault_injection/deployments/matmul.py", "fault39/native", "fault39/tools/version_profile.py",
    "fault39/tools/deployment_profile.py", "tests/native",
    "fault20_fast_validation/source/l4_buffer_substitute_allhit_v1",
    "fault20_fast_validation/source/l4_metadata_lifecycle_proxy",
    "fault20_fast_validation/source/fault26_readonly_trace_v1",
)
_F26_SOURCES = (
    "fault20_fast_validation/source/l4_buffer_substitute_allhit_v1/fault20_buffer_api.h",
    "fault_injection/native_adapters/legacy26_fia_bridge.inc",
    "fault_injection/native_adapters/legacy26_qmm_bridge.inc",
    "fault20_fast_validation/source/fault26_readonly_trace_v1",
    "fault20_fast_validation/source/fault26_fia_writer_proxy_v1",
    "fault20_fast_validation/source/l4_metadata_lifecycle_proxy",
)
_F20_EXPORTS = ("aclnnMatmulWeightNz", "aclnnMatmulWeightNzGetWorkspaceSize",
                "aclnnQuantMatmulWeightNz", "aclnnQuantMatmulWeightNzGetWorkspaceSize")
_F26_EXPORTS = ("aclnnFusedInferAttentionScoreV3", "aclnnFusedInferAttentionScoreV3GetWorkspaceSize",
                "aclnnQuantMatmulWeightNz", "aclnnQuantMatmulWeightNzGetWorkspaceSize", "aclnnReduceSum")

_RECIPES = (
    Recipe("fault39.matmul.abi0.v1", "39.matmul_range", "aclnnMatmul", "opcache_proxy",
           "fault39/native/build.sh", "FAULT39_OUTPUT_DIR", "libfault39_native_v1.so",
           ("fault39_abi_version_v1", "fault39_build_profile_v1", "fault39_configure_v1",
            "fault39_disable_v1", "fault39_clean_v1", "fault39_snapshot_v1"),
           _COMMON_SOURCES + ("fault39/native", "fault39/tools/version_profile.py", "fault39/tools/deployment_profile.py",
               "fault20_fast_validation/source/l4_metadata_lifecycle_proxy/libopapi_metadata_proxy.cpp",
               "fault20_fast_validation/source/l4_buffer_substitute_allhit_v1/fault20_buffer_api.h"),
           _PRIVATE_HEADERS + ("include/aclnnop/aclnn_matmul.h",),
           ("aclnnMatmul", "aclnnMatmulGetWorkspaceSize")),
    *tuple(Recipe("fault20." + label + ".abi0.v1", "20.buffer_substitute", api, "opcache_proxy",
           "fault_injection/native_adapters/build_qmm.sh", "DEMONCAT_QMM_OUTPUT", "libdcat20_qmm.so",
           tuple(prefix + "_" + action + "_v2" for action in ("install", "pause", "resume", "remove", "state")),
           _COMMON_SOURCES + _F20_SOURCES,
           _PRIVATE_HEADERS + ("include/aclnnop/aclnn_matmul.h", "include/aclnnop/aclnn_quant_matmul_weight_nz.h"),
           _F20_EXPORTS, allowed_versions=("8.5.1",), requires_pta_wheel=True)
           for label, api, prefix in (("weight_nz", "aclnnMatmulWeightNz", "dcat20_weight_nz"),
                                     ("qmm", "aclnnQuantMatmulWeightNz", "dcat20_qmm"))),
    *tuple(Recipe("fault26." + label + ".abi0.v1", mechanism, api, "operator_writer",
           "fault_injection/native_build/build26.sh", "DCAT26_BUILD", runtime,
           tuple(prefix + "_" + action + "_v1" for action in actions),
           _COMMON_SOURCES + _F26_SOURCES,
           _PRIVATE_HEADERS + ("include/aclnnop/aclnn_matmul.h", "include/aclnnop/aclnn_quant_matmul_weight_nz.h",
                               "include/aclnnop/aclnn_fused_infer_attention_score_v3.h"),
           _F26_EXPORTS, allowed_versions=("8.5.1",), requires_custom=True)
           for label, mechanism, api, runtime, prefix, actions in (
               ("qmm", "26.residual_stomp", "aclnnQuantMatmulWeightNz", "libfault26_qmm_output_underflow_abi0.so",
                "dcat26q", ("install", "remove", "continue", "pending", "state")),
               ("fia", "26.fia_destination", "aclnnFusedInferAttentionScoreV3", "libfault26_fia_writer_proxy_abi0.so",
                "dcat26f", ("install", "remove", "continue", "pending", "state")))),
    *tuple(Recipe("postwrite." + label + ".acl.v1", mechanism, operator, "public_acl_postwrite",
           "fault_injection/native_build/build_postwrite.sh", "DCAT_POSTWRITE_BUILD", "libdcat_postwrite.so",
           tuple("dcat_pw_" + action for action in ("abi", "open", "apply", "restore", "release", "live")),
           _COMMON_SOURCES + ("fault_injection/native_adapters/postwrite",),
           ("include/acl/acl.h",), ("aclrtGetDevice", "aclrtSynchronizeDevice", "aclrtMemcpy"))
           for label, mechanism, operator in (("output", "39.output_proxy", "aten.mm.default"),
                                               ("cache_tail", "41.tail_column", "aten.index_copy_.default"))),
)


def recipes():
    return tuple(_RECIPES)


def resolve_recipe(mechanism, operator=None):
    matching = [r for r in _RECIPES if r.mechanism == mechanism and (operator is None or r.operator == operator)]
    if len(matching) != 1:
        raise ValueError("ACTUAL_OPERATOR_REQUIRED" if matching else "NATIVE_RECIPE_NOT_ADAPTED")
    return matching[0]
