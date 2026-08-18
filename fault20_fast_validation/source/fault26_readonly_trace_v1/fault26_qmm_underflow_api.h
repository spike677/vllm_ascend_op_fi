#ifndef FAULT26_QMM_UNDERFLOW_API_H
#define FAULT26_QMM_UNDERFLOW_API_H

#include <acl/acl.h>
#include <aclnn/opdev/common_types.h>
#include <cstdint>

constexpr uint32_t FAULT26_QMM_UNDERFLOW_ABI_VERSION = 1;

struct Fault26QmmUnderflowContext {
    uint32_t abi_version;
    int32_t device;
    uint64_t run_index;
    uint64_t producer_sequence;
    uint64_t consumer_sequence;
    void *cache;
    void *original_output;
    uint64_t output_bytes;
    int64_t output_m;
    int64_t output_n;
    void *residual;
    uint64_t residual_bytes;
    int64_t residual_m;
    int64_t residual_k;
};

struct Fault26QmmUnderflowDecision {
    uint32_t abi_version;
    uint32_t claimed;
    uint64_t token;
    void *replacement_output;
};

using Fault26QmmUnderflowPrepareFn = bool (*)(
    const Fault26QmmUnderflowContext *, aclrtStream,
    Fault26QmmUnderflowDecision *);
using Fault26QmmUnderflowCompleteFn = void (*)(
    const Fault26QmmUnderflowContext *, const Fault26QmmUnderflowDecision *,
    aclrtStream, aclnnStatus);
using Fault26QmmUnderflowConsumerFn = void (*)(
    int32_t, uint64_t, void *, uint64_t, void *, uint64_t, aclrtStream,
    aclnnStatus);

#endif
