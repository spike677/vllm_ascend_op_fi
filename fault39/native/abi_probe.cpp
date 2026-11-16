// Header-only ABI checks. The executable has no CANN linkage or NPU calls.
#include <acl/acl.h>
#include <aclnn/acl_meta.h>
#include <aclnn/opdev/op_cache.h>
#include <aclnnop/aclnn_matmul.h>
#include <iostream>
#include <type_traits>
#include <utility>
using Cache=op::internal::OpExecCache;
using Getws=aclnnStatus(*)(const aclTensor*,const aclTensor*,aclTensor*,int8_t,uint64_t*,aclOpExecutor**);
using Execute=aclnnStatus(*)(void*,uint64_t,aclOpExecutor*,aclrtStream);
using Run=aclnnStatus(Cache::*)(void*,aclrtStream,const std::vector<void*>&);
static_assert(std::is_same<decltype(&aclnnMatmulGetWorkspaceSize),Getws>::value,"Matmul GWS signature changed");
static_assert(std::is_same<decltype(&aclnnMatmul),Execute>::value,"Matmul execute signature changed");
static_assert(std::is_same<decltype(&Cache::Run),Run>::value,"Cache Run signature changed");
static_assert(std::is_same<decltype(&op::internal::AddOpExecCache),bool(*)(Cache*)>::value,"Cache add changed");
static_assert(std::is_same<decltype(static_cast<void(*)(const aclTensor*)>(&op::internal::AddParamToBuf)),void(*)(const aclTensor*)>::value,"Const parameter changed");
static_assert(std::is_same<decltype(static_cast<void(*)(aclTensor*)>(&op::internal::AddParamToBuf)),void(*)(aclTensor*)>::value,"Parameter changed");
static_assert(ACL_BF16==27 && ACL_FORMAT_ND==2,"Required enums changed");
static_assert(std::is_pointer<decltype(std::declval<const aclTensor&>().GetStorage())>::value,"Storage accessor changed");
static_assert(std::is_pointer<decltype(std::declval<const aclTensor&>().GetStorageAddr())>::value,"Storage address changed");
static_assert(std::is_pointer<decltype(std::declval<const aclTensor&>().GetData())>::value,"Data accessor changed");
int main(){std::cout << "{\"aclTensor_size\":" << sizeof(aclTensor)
 << ",\"OpExecCache_size\":" << sizeof(Cache)
 << ",\"ACL_BF16\":" << int(ACL_BF16) << ",\"ACL_FORMAT_ND\":" << int(ACL_FORMAT_ND)
 << ",\"matmul_signature\":\"verified\",\"private_cache_signatures\":\"verified\",\"actual_npu_calls\":0}";}
