#include "fault26_trace_shared.h"

#include <acl/acl.h>
#include <aclnn/opdev/common_types.h>
#include <aclnnop/aclnn_fused_infer_attention_score_v3.h>

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <pthread.h>

namespace {

using ExecuteFn = aclnnStatus (*)(void *, uint64_t, aclOpExecutor *, aclrtStream);
using FiaGwsFn = decltype(&::aclnnFusedInferAttentionScoreV3GetWorkspaceSize);
using TraceBeginFn = void (*)(uint32_t, const char *, void *, uint64_t,
                              aclOpExecutor *, aclrtStream);
using TraceEndFn = void (*)(uint32_t, const char *, aclnnStatus);
using TraceWriterGwsFn = void (*)(const char *, const char *,
                                  const aclTensor *, uint64_t,
                                  aclOpExecutor *, aclnnStatus);
using TraceAddRmsConsumeFn = void (*)(aclrtStream);
using InventoryFiaBeforeFn = void (*)(const aclTensor *);
using InventoryFiaBeforeV2Fn = void (*)(const aclTensor *,const char *);
using InventoryFiaAfterV2Fn = void (*)(const aclTensor *,const aclTensor *,aclOpExecutor *,aclnnStatus);
using InventoryFiaEndV2Fn = void (*)(aclOpExecutor *,aclnnStatus);
using AddRmsNormBiasGwsFn = aclnnStatus (*)(
    const aclTensor *, const aclTensor *, const aclTensor *, const aclTensor *,
    double, const aclTensor *, const aclTensor *, const aclTensor *, uint64_t *,
    aclOpExecutor **);
using TraceAddRmsNormBiasGwsFn = void (*)(
    const aclTensor *, const aclTensor *, const aclTensor *, const aclTensor *,
    double, const aclTensor *, const aclTensor *, const aclTensor *, uint64_t,
    aclOpExecutor *, aclnnStatus);
using WriteGateAddRmsNormBiasGwsFn = void (*)(
    const aclTensor *, const aclTensor *, const aclTensor *, const aclTensor *,
    double, const aclTensor *, const aclTensor *, const aclTensor *, uint64_t,
    aclOpExecutor *, aclnnStatus);
using WriteGateBeforeExecuteFn = void (*)(void *, uint64_t, aclOpExecutor *,
                                          aclrtStream);
using WriteGateAfterExecuteFn = void (*)(aclnnStatus, aclrtStream);
using FiaWriterPrepareFn = const aclTensor *(*)(const aclTensor *);
using FiaWriterGwsCompleteFn = void (*)(const aclTensor *, const aclTensor *,
                                        aclOpExecutor *, aclnnStatus);
using FiaWriterBeforeExecuteFn = bool (*)(aclOpExecutor *, aclrtStream);
using FiaWriterAfterExecuteFn = void (*)(aclOpExecutor *, aclrtStream,
                                         aclnnStatus);

pthread_once_t g_resolve_once = PTHREAD_ONCE_INIT;
#if !defined(FAULT26_MAIN_PROXY_ONLY)
pthread_once_t g_resolve_custom_once = PTHREAD_ONCE_INIT;
#endif
ExecuteFn g_real_fia_v3 = nullptr;
FiaGwsFn g_real_fia_v3_gws = nullptr;
ExecuteFn g_real_reduce_sum = nullptr;
#if !defined(FAULT26_MAIN_PROXY_ONLY)
ExecuteFn g_real_add_rms_norm_bias = nullptr;
AddRmsNormBiasGwsFn g_real_add_rms_norm_bias_gws = nullptr;
#endif
TraceBeginFn g_trace_begin = nullptr;
TraceEndFn g_trace_end = nullptr;
TraceWriterGwsFn g_trace_writer_gws = nullptr;
InventoryFiaBeforeFn g_inventory_fia_before = nullptr;
InventoryFiaBeforeV2Fn g_inventory_fia_before_v2 = nullptr;
InventoryFiaAfterV2Fn g_inventory_fia_after_v2 = nullptr;
InventoryFiaEndV2Fn g_inventory_fia_end_v2 = nullptr;
TraceAddRmsConsumeFn g_trace_add_rms_consume = nullptr;
TraceAddRmsNormBiasGwsFn g_trace_add_rms_norm_bias_gws = nullptr;
WriteGateAddRmsNormBiasGwsFn g_write_gate_add_rms_norm_bias_gws = nullptr;
WriteGateBeforeExecuteFn g_write_gate_before_execute = nullptr;
WriteGateAfterExecuteFn g_write_gate_after_execute = nullptr;
FiaWriterPrepareFn g_fia_writer_prepare = nullptr;
FiaWriterGwsCompleteFn g_fia_writer_gws_complete = nullptr;
FiaWriterBeforeExecuteFn g_fia_writer_before_execute = nullptr;
FiaWriterAfterExecuteFn g_fia_writer_after_execute = nullptr;
#if !defined(FAULT26_MAIN_PROXY_ONLY)
void *g_custom_opapi_handle = nullptr;
#endif
#if !defined(FAULT26_CUSTOM_PROXY_ONLY)
thread_local uint32_t g_fia_gws_depth = 0;
thread_local uint32_t g_fia_execute_depth = 0;
#endif

void resolve_symbols() {
    g_real_fia_v3 = reinterpret_cast<ExecuteFn>(
        dlsym(RTLD_NEXT, "aclnnFusedInferAttentionScoreV3"));
    g_real_fia_v3_gws = reinterpret_cast<FiaGwsFn>(
        dlsym(RTLD_NEXT, "aclnnFusedInferAttentionScoreV3GetWorkspaceSize"));
    g_real_reduce_sum =
        reinterpret_cast<ExecuteFn>(dlsym(RTLD_NEXT, "aclnnReduceSum"));
    g_trace_begin = reinterpret_cast<TraceBeginFn>(
        dlsym(RTLD_DEFAULT, "fault26_trace_execute_begin"));
    g_trace_end = reinterpret_cast<TraceEndFn>(
        dlsym(RTLD_DEFAULT, "fault26_trace_execute_end"));
    g_trace_writer_gws = reinterpret_cast<TraceWriterGwsFn>(
        dlsym(RTLD_DEFAULT, "fault26_trace_writer_gws"));
    g_inventory_fia_before = reinterpret_cast<InventoryFiaBeforeFn>(
        dlsym(RTLD_DEFAULT, "dcat26_inventory_fia_before_v1"));
    g_inventory_fia_before_v2 = reinterpret_cast<InventoryFiaBeforeV2Fn>(
        dlsym(RTLD_DEFAULT, "dcat26_inventory_fia_before_v2"));
    g_inventory_fia_after_v2 = reinterpret_cast<InventoryFiaAfterV2Fn>(
        dlsym(RTLD_DEFAULT, "dcat26_inventory_fia_after_v2"));
    g_inventory_fia_end_v2 = reinterpret_cast<InventoryFiaEndV2Fn>(
        dlsym(RTLD_DEFAULT, "dcat26_inventory_fia_end_v2"));
    g_trace_add_rms_consume = reinterpret_cast<TraceAddRmsConsumeFn>(
        dlsym(RTLD_DEFAULT, "fault26_trace_add_rms_consume"));
    g_trace_add_rms_norm_bias_gws =
        reinterpret_cast<TraceAddRmsNormBiasGwsFn>(dlsym(
            RTLD_DEFAULT, "fault26_trace_add_rms_norm_bias_gws"));
    g_write_gate_add_rms_norm_bias_gws =
        reinterpret_cast<WriteGateAddRmsNormBiasGwsFn>(dlsym(
            RTLD_DEFAULT, "fault26_write_gate_add_rms_norm_bias_gws"));
    g_write_gate_before_execute = reinterpret_cast<WriteGateBeforeExecuteFn>(
        dlsym(RTLD_DEFAULT, "fault26_write_gate_before_add_rms_norm_bias"));
    g_write_gate_after_execute = reinterpret_cast<WriteGateAfterExecuteFn>(
        dlsym(RTLD_DEFAULT, "fault26_write_gate_after_add_rms_norm_bias"));
    g_fia_writer_prepare = reinterpret_cast<FiaWriterPrepareFn>(
        dlsym(RTLD_DEFAULT, "fault26_fia_writer_prepare_output"));
    g_fia_writer_gws_complete = reinterpret_cast<FiaWriterGwsCompleteFn>(
        dlsym(RTLD_DEFAULT, "fault26_fia_writer_gws_complete"));
    g_fia_writer_before_execute =
        reinterpret_cast<FiaWriterBeforeExecuteFn>(dlsym(
            RTLD_DEFAULT, "fault26_fia_writer_before_execute"));
    g_fia_writer_after_execute = reinterpret_cast<FiaWriterAfterExecuteFn>(
        dlsym(RTLD_DEFAULT, "fault26_fia_writer_after_execute"));
}

#if !defined(FAULT26_MAIN_PROXY_ONLY)
void resolve_custom_symbols() {
    g_real_add_rms_norm_bias = reinterpret_cast<ExecuteFn>(
        dlsym(RTLD_NEXT, "aclnnAddRmsNormBias"));
    g_real_add_rms_norm_bias_gws = reinterpret_cast<AddRmsNormBiasGwsFn>(
        dlsym(RTLD_NEXT, "aclnnAddRmsNormBiasGetWorkspaceSize"));
    if (g_real_add_rms_norm_bias != nullptr &&
        g_real_add_rms_norm_bias_gws != nullptr) {
        return;
    }
    const char *path = std::getenv("FAULT26_REAL_CUSTOM_OPAPI");
    if (path == nullptr || path[0] == '\0') {
        return;
    }
    g_custom_opapi_handle = dlopen(path, RTLD_NOW | RTLD_LOCAL);
    if (g_custom_opapi_handle == nullptr) {
        return;
    }
    g_real_add_rms_norm_bias = reinterpret_cast<ExecuteFn>(
        dlsym(g_custom_opapi_handle, "aclnnAddRmsNormBias"));
    g_real_add_rms_norm_bias_gws = reinterpret_cast<AddRmsNormBiasGwsFn>(dlsym(
        g_custom_opapi_handle, "aclnnAddRmsNormBiasGetWorkspaceSize"));
}
#endif

#if !defined(FAULT26_CUSTOM_PROXY_ONLY)
aclnnStatus forward_execute(ExecuteFn real, uint32_t kind, const char *api,
                            void *workspace, uint64_t workspace_size,
                            aclOpExecutor *executor, aclrtStream stream) {
    pthread_once(&g_resolve_once, resolve_symbols);
    if (real == nullptr) {
        return static_cast<aclnnStatus>(1);
    }
    if (g_trace_begin != nullptr) {
        g_trace_begin(kind, api, workspace, workspace_size, executor, stream);
    }
    const aclnnStatus status =
        real(workspace, workspace_size, executor, stream);
    if (kind == FAULT26_EXECUTE_FIA_V3 &&
        g_fia_writer_after_execute != nullptr) {
        g_fia_writer_after_execute(executor, stream, status);
    }
    if (g_trace_end != nullptr) {
        g_trace_end(kind, api, status);
    }
    return status;
}
#endif

}  // namespace

#if !defined(FAULT26_CUSTOM_PROXY_ONLY)
extern "C" aclnnStatus aclnnFusedInferAttentionScoreV3GetWorkspaceSize(
    const aclTensor *query, const aclTensorList *key,
    const aclTensorList *value, const aclTensor *pse_shift_optional,
    const aclTensor *atten_mask_optional,
    const aclIntArray *actual_seq_lengths_optional,
    const aclIntArray *actual_seq_lengths_kv_optional,
    const aclTensor *deq_scale1_optional,
    const aclTensor *quant_scale1_optional,
    const aclTensor *deq_scale2_optional,
    const aclTensor *quant_scale2_optional,
    const aclTensor *quant_offset2_optional,
    const aclTensor *antiquant_scale_optional,
    const aclTensor *antiquant_offset_optional,
    const aclTensor *block_table_optional,
    const aclTensor *query_padding_size_optional,
    const aclTensor *kv_padding_size_optional,
    const aclTensor *key_antiquant_scale_optional,
    const aclTensor *key_antiquant_offset_optional,
    const aclTensor *value_antiquant_scale_optional,
    const aclTensor *value_antiquant_offset_optional,
    const aclTensor *key_shared_prefix_optional,
    const aclTensor *value_shared_prefix_optional,
    const aclIntArray *actual_shared_prefix_len_optional,
    const aclTensor *query_rope_optional,
    const aclTensor *key_rope_optional,
    const aclTensor *key_rope_antiquant_scale_optional, int64_t num_heads,
    double scale_value, int64_t pre_tokens, int64_t next_tokens,
    char *input_layout, int64_t num_key_value_heads, int64_t sparse_mode,
    int64_t inner_precise, int64_t block_size, int64_t antiquant_mode,
    bool softmax_lse_flag, int64_t key_antiquant_mode,
    int64_t value_antiquant_mode, const aclTensor *attention_out,
    const aclTensor *softmax_lse, uint64_t *workspace_size,
    aclOpExecutor **executor) {
    pthread_once(&g_resolve_once, resolve_symbols);
    if (g_real_fia_v3_gws == nullptr) {
        return static_cast<aclnnStatus>(1);
    }
    const bool outermost = g_fia_gws_depth++ == 0;
    if(outermost) {
        if(g_inventory_fia_before_v2)g_inventory_fia_before_v2(attention_out,input_layout);
        else if(g_inventory_fia_before)g_inventory_fia_before(attention_out);
    }
    const aclTensor *effective_attention_out = attention_out;
    if (outermost && g_fia_writer_prepare != nullptr) {
        const aclTensor *candidate = g_fia_writer_prepare(attention_out);
        if (candidate != nullptr) {
            effective_attention_out = candidate;
        }
    }
    const aclnnStatus status = g_real_fia_v3_gws(
        query, key, value, pse_shift_optional, atten_mask_optional,
        actual_seq_lengths_optional, actual_seq_lengths_kv_optional,
        deq_scale1_optional, quant_scale1_optional, deq_scale2_optional,
        quant_scale2_optional, quant_offset2_optional,
        antiquant_scale_optional, antiquant_offset_optional,
        block_table_optional, query_padding_size_optional,
        kv_padding_size_optional, key_antiquant_scale_optional,
        key_antiquant_offset_optional, value_antiquant_scale_optional,
        value_antiquant_offset_optional, key_shared_prefix_optional,
        value_shared_prefix_optional, actual_shared_prefix_len_optional,
        query_rope_optional, key_rope_optional,
        key_rope_antiquant_scale_optional, num_heads, scale_value, pre_tokens,
        next_tokens, input_layout, num_key_value_heads, sparse_mode,
        inner_precise, block_size, antiquant_mode, softmax_lse_flag,
        key_antiquant_mode, value_antiquant_mode, effective_attention_out,
        softmax_lse, workspace_size, executor);
    --g_fia_gws_depth;
    if (outermost && !g_inventory_fia_after_v2 && g_fia_writer_gws_complete != nullptr) {
        g_fia_writer_gws_complete(
            attention_out, effective_attention_out,
            executor == nullptr ? nullptr : *executor, status);
    }
    if (outermost && g_trace_writer_gws != nullptr) {
        g_trace_writer_gws(
            "aclnnFusedInferAttentionScoreV3", "attentionOut",
            effective_attention_out,
            workspace_size == nullptr ? 0 : *workspace_size,
            executor == nullptr ? nullptr : *executor, status);
    }
    if(outermost && g_inventory_fia_after_v2) {
        // Typed after-GWS evidence precedes the writer's current-call audit.
        // The borrowed original is still an argument on this actual GWS stack.
        g_inventory_fia_after_v2(attention_out,effective_attention_out,executor==nullptr?nullptr:*executor,status);
        if(g_fia_writer_gws_complete)g_fia_writer_gws_complete(
            attention_out,effective_attention_out,executor==nullptr?nullptr:*executor,status);
    }
    return status;
}

extern "C" aclnnStatus aclnnFusedInferAttentionScoreV3(
    void *workspace, uint64_t workspace_size, aclOpExecutor *executor,
    aclrtStream stream) {
    pthread_once(&g_resolve_once, resolve_symbols);
    if (g_real_fia_v3 == nullptr) {
        return static_cast<aclnnStatus>(1);
    }
    const bool outermost = g_fia_execute_depth++ == 0;
    if (outermost && g_trace_begin != nullptr) {
        g_trace_begin(FAULT26_EXECUTE_FIA_V3,
                      "aclnnFusedInferAttentionScoreV3", workspace,
                      workspace_size, executor, stream);
    }
    const bool execute_allowed =
        !outermost || g_fia_writer_before_execute == nullptr ||
        g_fia_writer_before_execute(executor, stream);
    const aclnnStatus status =
        execute_allowed
            ? g_real_fia_v3(workspace, workspace_size, executor, stream)
            : static_cast<aclnnStatus>(1);
    --g_fia_execute_depth;
    if (outermost && g_fia_writer_after_execute != nullptr) {
        g_fia_writer_after_execute(executor, stream, status);
    }
    if (outermost && g_trace_end != nullptr) {
        g_trace_end(FAULT26_EXECUTE_FIA_V3,
                    "aclnnFusedInferAttentionScoreV3", status);
    }
    if(outermost && g_inventory_fia_end_v2)g_inventory_fia_end_v2(executor,status);
    return status;
}

extern "C" aclnnStatus aclnnReduceSum(void *workspace, uint64_t workspace_size,
                                       aclOpExecutor *executor,
                                       aclrtStream stream) {
    pthread_once(&g_resolve_once, resolve_symbols);
    return forward_execute(g_real_reduce_sum, FAULT26_EXECUTE_REDUCE_SUM,
                           "aclnnReduceSum", workspace, workspace_size,
                           executor, stream);
}
#endif

#if !defined(FAULT26_MAIN_PROXY_ONLY)
// Serialized Worker diagnostic: these are the pointers already cached by the
// actual continuation. Do not initialize pthread_once or look up another symbol.
extern "C" int dcat26_custom_cached_address_v1(uint32_t slot,uint64_t* output) {
    if(!output)return 1;
    *output=0;
    uintptr_t address=0;
    if(slot==0)address=reinterpret_cast<uintptr_t>(g_real_add_rms_norm_bias);
    else if(slot==1)address=reinterpret_cast<uintptr_t>(g_real_add_rms_norm_bias_gws);
    else return 2;
    if(!address)return 3;
    *output=address;return 0;
}

extern "C" aclnnStatus aclnnAddRmsNormBiasGetWorkspaceSize(
    const aclTensor *x1, const aclTensor *x2, const aclTensor *gamma,
    const aclTensor *beta_optional, double epsilon, const aclTensor *y_out,
    const aclTensor *rstd_out, const aclTensor *x_out,
    uint64_t *workspace_size, aclOpExecutor **executor) {
    pthread_once(&g_resolve_once, resolve_symbols);
    pthread_once(&g_resolve_custom_once, resolve_custom_symbols);
    if (g_real_add_rms_norm_bias_gws == nullptr) {
        return static_cast<aclnnStatus>(1);
    }
    const aclnnStatus status = g_real_add_rms_norm_bias_gws(
        x1, x2, gamma, beta_optional, epsilon, y_out, rstd_out, x_out,
        workspace_size, executor);
    if (g_trace_add_rms_norm_bias_gws != nullptr) {
        g_trace_add_rms_norm_bias_gws(
            x1, x2, gamma, beta_optional, epsilon, y_out, rstd_out, x_out,
            workspace_size == nullptr ? 0 : *workspace_size,
            executor == nullptr ? nullptr : *executor, status);
    }
    if (g_write_gate_add_rms_norm_bias_gws != nullptr) {
        g_write_gate_add_rms_norm_bias_gws(
            x1, x2, gamma, beta_optional, epsilon, y_out, rstd_out, x_out,
            workspace_size == nullptr ? 0 : *workspace_size,
            executor == nullptr ? nullptr : *executor, status);
    }
    return status;
}

extern "C" aclnnStatus aclnnAddRmsNormBias(
    void *workspace, uint64_t workspace_size, aclOpExecutor *executor,
    aclrtStream stream) {
    pthread_once(&g_resolve_custom_once, resolve_custom_symbols);
    pthread_once(&g_resolve_once, resolve_symbols);
    if (g_real_add_rms_norm_bias == nullptr) {
        return static_cast<aclnnStatus>(1);
    }
    if (g_trace_begin != nullptr) {
        g_trace_begin(FAULT26_EXECUTE_ADD_RMS_NORM_BIAS,
                      "aclnnAddRmsNormBias", workspace, workspace_size,
                      executor, stream);
    }
    if (g_write_gate_before_execute != nullptr) {
        g_write_gate_before_execute(workspace, workspace_size, executor,
                                    stream);
    }
    if (g_trace_add_rms_consume != nullptr) {
        g_trace_add_rms_consume(stream);
    }
    const aclnnStatus status =
        g_real_add_rms_norm_bias(workspace, workspace_size, executor, stream);
    if (g_write_gate_after_execute != nullptr) {
        g_write_gate_after_execute(status, stream);
    }
    if (g_trace_end != nullptr) {
        g_trace_end(FAULT26_EXECUTE_ADD_RMS_NORM_BIAS,
                    "aclnnAddRmsNormBias", status);
    }
    return status;
}
#endif
