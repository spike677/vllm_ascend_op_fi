// This executable compiles actual installed signatures and makes no CANN call.
#include <acl/acl.h>
#include <aclnn/acl_meta.h>
#include <aclnn/opdev/op_cache.h>
#include <aclnnop/aclnn_quant_matmul_weight_nz.h>
#include <aclnnop/aclnn_fused_infer_attention_score_v3.h>
#include <aclnn_add_rms_norm_bias.h>
#include <iostream>
#include <type_traits>

using Cache = op::internal::OpExecCache;
using Run = aclnnStatus (Cache::*)(void*, aclrtStream, const std::vector<void*>&);
using Execute = aclnnStatus (*)(void*, uint64_t, aclOpExecutor*, aclrtStream);
using Qmm = aclnnStatus (*)(const aclTensor*, const aclTensor*, const aclTensor*, const aclTensor*,
    const aclTensor*, const aclTensor*, const aclTensor*, const aclTensor*, const aclTensor*,
    bool, bool, int64_t, aclTensor*, uint64_t*, aclOpExecutor**);
using AddRms = aclnnStatus (*)(const aclTensor*, const aclTensor*, const aclTensor*, const aclTensor*,
    double, const aclTensor*, const aclTensor*, const aclTensor*, uint64_t*, aclOpExecutor**);
using Fia = aclnnStatus (*)(const aclTensor*, const aclTensorList*, const aclTensorList*,
    const aclTensor*, const aclTensor*, const aclIntArray*, const aclIntArray*,
    const aclTensor*, const aclTensor*, const aclTensor*, const aclTensor*, const aclTensor*,
    const aclTensor*, const aclTensor*, const aclTensor*, const aclTensor*, const aclTensor*,
    const aclTensor*, const aclTensor*, const aclTensor*, const aclTensor*, const aclTensor*,
    const aclTensor*, const aclIntArray*, const aclTensor*, const aclTensor*, const aclTensor*,
    int64_t, double, int64_t, int64_t, char*, int64_t, int64_t, int64_t, int64_t, int64_t,
    bool, int64_t, int64_t, const aclTensor*, const aclTensor*, uint64_t*, aclOpExecutor**);
static_assert(std::is_same_v<decltype(&Cache::Run), Run>);
static_assert(std::is_same_v<decltype(&op::internal::AddOpExecCache), bool (*)(Cache*)>);
static_assert(std::is_same_v<decltype(static_cast<void (*)(const aclTensor*)>(&op::internal::AddParamToBuf)),
                             void (*)(const aclTensor*)>);
static_assert(std::is_same_v<decltype(static_cast<void (*)(aclTensor*)>(&op::internal::AddParamToBuf)),
                             void (*)(aclTensor*)>);
static_assert(std::is_same_v<decltype(&aclnnQuantMatmulWeightNzGetWorkspaceSize), Qmm>);
static_assert(std::is_same_v<decltype(&aclnnQuantMatmulWeightNz), Execute>);
static_assert(std::is_same_v<decltype(&aclnnFusedInferAttentionScoreV3GetWorkspaceSize), Fia>);
static_assert(std::is_same_v<decltype(&aclnnFusedInferAttentionScoreV3), Execute>);
static_assert(std::is_same_v<decltype(&aclnnAddRmsNormBiasGetWorkspaceSize), AddRms>);
static_assert(std::is_same_v<decltype(&aclnnAddRmsNormBias), Execute>);
static_assert(ACL_BF16 == 27 && ACL_FORMAT_ND == 2 && ACL_INT8 == 2);
int main() {
    std::cout << "{\"private_cache_signatures\":\"verified\",\"qmm_signature\":\"verified\","
                 "\"fia_v3_signature\":\"verified\",\"custom_signature\":\"verified\",\"actual_npu_calls\":0}";
}
