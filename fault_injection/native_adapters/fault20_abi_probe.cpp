// Compile-only signature checks and ABI facts. There are no CANN calls.
#include <acl/acl.h>
#include <aclnn/acl_meta.h>
#include <aclnn/opdev/op_cache.h>
#include <aclnnop/aclnn_matmul.h>
#include <aclnnop/aclnn_quant_matmul_weight_nz.h>
#include <iostream>
#include <type_traits>
#include <utility>

using Cache = op::internal::OpExecCache;
using Run = aclnnStatus (Cache::*)(void*, aclrtStream, const std::vector<void*>&);
using Execute = aclnnStatus (*)(void*, uint64_t, aclOpExecutor*, aclrtStream);
using WeightNzGetws = aclnnStatus (*)(const aclTensor*, const aclTensor*, aclTensor*, int8_t,
                                     uint64_t*, aclOpExecutor**);
using QmmGetws = aclnnStatus (*)(const aclTensor*, const aclTensor*, const aclTensor*, const aclTensor*,
                                const aclTensor*, const aclTensor*, const aclTensor*, const aclTensor*,
                                const aclTensor*, bool, bool, int64_t, aclTensor*, uint64_t*, aclOpExecutor**);
static_assert(std::is_same_v<decltype(&aclnnMatmulWeightNzGetWorkspaceSize), WeightNzGetws>);
static_assert(std::is_same_v<decltype(&aclnnMatmulWeightNz), Execute>);
static_assert(std::is_same_v<decltype(&aclnnQuantMatmulWeightNzGetWorkspaceSize), QmmGetws>);
static_assert(std::is_same_v<decltype(&aclnnQuantMatmulWeightNz), Execute>);
static_assert(std::is_same_v<decltype(&Cache::Run), Run>);
static_assert(std::is_same_v<decltype(&op::internal::AddOpExecCache), bool(*)(Cache*)>);
static_assert(std::is_same_v<decltype(static_cast<void(*)(const aclTensor*)>(&op::internal::AddParamToBuf)),
                             void(*)(const aclTensor*)>);
static_assert(std::is_same_v<decltype(static_cast<void(*)(aclTensor*)>(&op::internal::AddParamToBuf)),
                             void(*)(aclTensor*)>);
static_assert(ACL_BF16 == 27 && ACL_FORMAT_ND == 2 && ACL_INT8 == 2);
static_assert(std::is_pointer_v<decltype(std::declval<const aclTensor&>().GetStorage())>);
static_assert(std::is_pointer_v<decltype(std::declval<const aclTensor&>().GetStorageAddr())>);
static_assert(std::is_pointer_v<decltype(std::declval<const aclTensor&>().GetData())>);

int main() {
    std::cout << "{\"aclTensor_size\":" << sizeof(aclTensor)
              << ",\"OpExecCache_size\":" << sizeof(Cache)
              << ",\"weight_nz_signature\":\"verified\",\"qmm_signature\":\"verified\""
                 ",\"private_cache_signatures\":\"verified\",\"actual_npu_calls\":0}";
}
