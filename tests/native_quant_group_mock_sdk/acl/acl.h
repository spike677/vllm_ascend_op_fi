// HOST TEST ONLY. These declarations are NOT an SDK ABI validation.
#ifndef DCAT_QUANT_TEST_ACL_H
#define DCAT_QUANT_TEST_ACL_H
#include <cstddef>
#include <cstdint>
using aclError=int32_t;
using aclnnStatus=uint32_t;
constexpr aclError ACL_SUCCESS=0;
enum aclDataType { ACL_FLOAT=0,ACL_FLOAT16=1,ACL_INT8=2,ACL_BF16=27 };
enum aclFormat { ACL_FORMAT_ND=2 };
enum aclrtMemMallocPolicy { ACL_MEM_MALLOC_HUGE_FIRST=0 };
enum aclrtMemcpyKind { ACL_MEMCPY_HOST_TO_DEVICE=1,ACL_MEMCPY_DEVICE_TO_HOST=2,ACL_MEMCPY_DEVICE_TO_DEVICE=3 };
struct aclTensor;
struct aclOpExecutor;
using aclrtStream=void*;
#endif
