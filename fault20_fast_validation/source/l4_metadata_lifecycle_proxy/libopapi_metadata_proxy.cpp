#include <acl/acl.h>
#include <aclnn/acl_meta.h>
#include <aclnn/opdev/common_types.h>
#include <aclnnop/aclnn_ascend_quant_v3.h>
#include <aclnnop/aclnn_matmul.h>
#include <aclnnop/aclnn_quant_matmul_weight_nz.h>

#include "../l4_buffer_substitute_allhit_v1/fault20_buffer_api.h"

#include <atomic>
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <fcntl.h>
#include <limits>
#include <pthread.h>
#include <sys/syscall.h>
#include <sys/uio.h>
#include <time.h>
#include <unistd.h>

namespace {

constexpr const char *kDefaultLogPath =
    "/fault20/runtime/l4_metadata_lifecycle_proxy/metadata.jsonl";
constexpr size_t kMaxDims = 16;

using QuantMatmulGetWorkspaceSizeFn = aclnnStatus (*)(
    const aclTensor *, const aclTensor *, const aclTensor *, const aclTensor *, const aclTensor *,
    const aclTensor *, const aclTensor *, const aclTensor *, const aclTensor *, bool, bool, int64_t,
    aclTensor *, uint64_t *, aclOpExecutor **);
using AscendQuantGetWorkspaceSizeFn = aclnnStatus (*)(
    const aclTensor *, const aclTensor *, const aclTensor *, bool, const char *, int32_t, int32_t,
    const aclTensor *, uint64_t *, aclOpExecutor **);
using MatmulWeightNzGetWorkspaceSizeFn = aclnnStatus (*)(
    const aclTensor *, const aclTensor *, aclTensor *, int8_t, uint64_t *,
    aclOpExecutor **);
using ExecuteFn = aclnnStatus (*)(void *, uint64_t, aclOpExecutor *, aclrtStream);
using Fault20GwsBeginFn = void (*)();
using Fault20GwsCommitFn = bool (*)(const Fault20X1Metadata *);
using Fault20MatmulGwsCommitFn = bool (*)(const Fault20MatmulMetadata *);
using Fault20GwsAbortFn = void (*)();

pthread_once_t g_resolve_once = PTHREAD_ONCE_INIT;
QuantMatmulGetWorkspaceSizeFn g_real_qmm_getws = nullptr;
AscendQuantGetWorkspaceSizeFn g_real_quant_getws = nullptr;
MatmulWeightNzGetWorkspaceSizeFn g_real_matmul_getws = nullptr;
ExecuteFn g_real_qmm_execute = nullptr;
ExecuteFn g_real_quant_execute = nullptr;
ExecuteFn g_real_matmul_execute = nullptr;
Fault20GwsBeginFn g_fault20_gws_begin = nullptr;
Fault20GwsCommitFn g_fault20_gws_commit = nullptr;
Fault20GwsAbortFn g_fault20_gws_abort = nullptr;
Fault20GwsBeginFn g_fault20_matmul_gws_begin = nullptr;
Fault20MatmulGwsCommitFn g_fault20_matmul_gws_commit = nullptr;
Fault20GwsAbortFn g_fault20_matmul_gws_abort = nullptr;
std::atomic<uint64_t> g_sequence{0};

uint64_t monotonic_ns() {
    timespec value{};
    if (clock_gettime(CLOCK_MONOTONIC, &value) != 0) {
        return 0;
    }
    return static_cast<uint64_t>(value.tv_sec) * 1000000000ULL +
           static_cast<uint64_t>(value.tv_nsec);
}

void write_line(const char *line, size_t size) {
    const char *configured = std::getenv("FAULT20_METADATA_LOG_PATH");
    const char *path = configured == nullptr || configured[0] == '\0' ? kDefaultLogPath : configured;
    const int fd = open(path, O_WRONLY | O_CREAT | O_APPEND | O_CLOEXEC, 0644);
    if (fd < 0) {
        return;
    }
    char newline = '\n';
    iovec parts[2] = {{const_cast<char *>(line), size}, {&newline, 1}};
    (void)writev(fd, parts, 2);
    (void)close(fd);
}

void resolve_real() {
    g_real_qmm_getws = reinterpret_cast<QuantMatmulGetWorkspaceSizeFn>(
        dlsym(RTLD_NEXT, "aclnnQuantMatmulWeightNzGetWorkspaceSize"));
    g_real_qmm_execute =
        reinterpret_cast<ExecuteFn>(dlsym(RTLD_NEXT, "aclnnQuantMatmulWeightNz"));
    g_real_quant_getws = reinterpret_cast<AscendQuantGetWorkspaceSizeFn>(
        dlsym(RTLD_NEXT, "aclnnAscendQuantV3GetWorkspaceSize"));
    g_real_quant_execute =
        reinterpret_cast<ExecuteFn>(dlsym(RTLD_NEXT, "aclnnAscendQuantV3"));
    g_real_matmul_getws = reinterpret_cast<MatmulWeightNzGetWorkspaceSizeFn>(
        dlsym(RTLD_NEXT, "aclnnMatmulWeightNzGetWorkspaceSize"));
    g_real_matmul_execute =
        reinterpret_cast<ExecuteFn>(dlsym(RTLD_NEXT, "aclnnMatmulWeightNz"));
    g_fault20_gws_begin = reinterpret_cast<Fault20GwsBeginFn>(
        dlsym(RTLD_DEFAULT, "fault20_qmm_gws_begin"));
    g_fault20_gws_commit = reinterpret_cast<Fault20GwsCommitFn>(
        dlsym(RTLD_DEFAULT, "fault20_qmm_gws_commit"));
    g_fault20_gws_abort = reinterpret_cast<Fault20GwsAbortFn>(
        dlsym(RTLD_DEFAULT, "fault20_qmm_gws_abort"));
    g_fault20_matmul_gws_begin = reinterpret_cast<Fault20GwsBeginFn>(
        dlsym(RTLD_DEFAULT, "fault20_matmul_gws_begin"));
    g_fault20_matmul_gws_commit = reinterpret_cast<Fault20MatmulGwsCommitFn>(
        dlsym(RTLD_DEFAULT, "fault20_matmul_gws_commit"));
    g_fault20_matmul_gws_abort = reinterpret_cast<Fault20GwsAbortFn>(
        dlsym(RTLD_DEFAULT, "fault20_matmul_gws_abort"));
}

uint64_t checked_bytes(int64_t elements, size_t element_size);

bool checked_product(const int64_t *dims, uint64_t count, int64_t *product) {
    if (dims == nullptr || count == 0 || count > FAULT20_MAX_DIMS) {
        return false;
    }
    uint64_t value = 1;
    for (uint64_t i = 0; i < count; ++i) {
        if (dims[i] < 0) {
            return false;
        }
        const uint64_t dim = static_cast<uint64_t>(dims[i]);
        if (dim != 0 && value > static_cast<uint64_t>(std::numeric_limits<int64_t>::max()) / dim) {
            return false;
        }
        value *= dim;
    }
    *product = static_cast<int64_t>(value);
    return true;
}

bool is_contiguous(const int64_t *shape, uint64_t shape_count, const int64_t *strides,
                   uint64_t stride_count) {
    if (shape == nullptr || strides == nullptr || shape_count == 0 ||
        shape_count != stride_count || shape_count > FAULT20_MAX_DIMS) {
        return false;
    }
    int64_t expected = 1;
    for (uint64_t reverse = 0; reverse < shape_count; ++reverse) {
        const uint64_t i = shape_count - reverse - 1;
        if (strides[i] != expected || shape[i] < 0) {
            return false;
        }
        if (shape[i] != 0 &&
            expected > std::numeric_limits<int64_t>::max() / shape[i]) {
            return false;
        }
        expected *= shape[i];
    }
    return true;
}

bool capture_x1_metadata(const aclTensor *x1, const aclTensor *out, bool transpose_x1,
                         bool transpose_x2, int64_t group_size, Fault20X1Metadata *metadata) {
    if (x1 == nullptr || out == nullptr || metadata == nullptr) {
        return false;
    }
    aclDataType dtype = ACL_DT_UNDEFINED;
    aclFormat format = ACL_FORMAT_UNDEFINED;
    int64_t view_offset = 0;
    int64_t *view_dims = nullptr;
    int64_t *storage_dims = nullptr;
    int64_t *strides = nullptr;
    int64_t *out_dims = nullptr;
    uint64_t view_count = 0;
    uint64_t storage_count = 0;
    uint64_t stride_count = 0;
    uint64_t out_count = 0;
    if (aclGetDataType(x1, &dtype) != ACL_SUCCESS ||
        aclGetFormat(x1, &format) != ACL_SUCCESS ||
        aclGetViewOffset(x1, &view_offset) != ACL_SUCCESS ||
        aclGetViewShape(x1, &view_dims, &view_count) != ACL_SUCCESS ||
        aclGetStorageShape(x1, &storage_dims, &storage_count) != ACL_SUCCESS ||
        aclGetViewStrides(x1, &strides, &stride_count) != ACL_SUCCESS ||
        aclGetViewShape(out, &out_dims, &out_count) != ACL_SUCCESS ||
        view_count == 0 || storage_count == 0 || stride_count == 0 || out_count == 0 ||
        view_count > FAULT20_MAX_DIMS || storage_count > FAULT20_MAX_DIMS ||
        stride_count > FAULT20_MAX_DIMS || out_count > FAULT20_MAX_DIMS) {
        return false;
    }
    int64_t numel = 0;
    int64_t storage_elements = 0;
    if (!checked_product(view_dims, view_count, &numel) ||
        !checked_product(storage_dims, storage_count, &storage_elements)) {
        return false;
    }
    const size_t element_size = aclDataTypeSize(dtype);
    const uint64_t view_bytes = checked_bytes(numel, element_size);
    const uint64_t storage_bytes = checked_bytes(storage_elements, element_size);
    if (element_size == 0 || view_bytes == 0 || storage_bytes == 0) {
        return false;
    }
    Fault20X1Metadata value{};
    value.dtype = static_cast<int32_t>(dtype);
    value.format = static_cast<int32_t>(format);
    value.view_rank = static_cast<uint32_t>(view_count);
    value.storage_rank = static_cast<uint32_t>(storage_count);
    value.stride_rank = static_cast<uint32_t>(stride_count);
    value.contiguous = is_contiguous(view_dims, view_count, strides, stride_count) ? 1U : 0U;
    for (uint64_t i = 0; i < view_count; ++i) {
        value.view_shape[i] = view_dims[i];
    }
    for (uint64_t i = 0; i < storage_count; ++i) {
        value.storage_shape[i] = storage_dims[i];
    }
    for (uint64_t i = 0; i < stride_count; ++i) {
        value.strides[i] = strides[i];
    }
    value.view_offset = view_offset;
    value.numel = numel;
    value.storage_elements = storage_elements;
    value.element_size = element_size;
    value.view_bytes = view_bytes;
    value.storage_bytes = storage_bytes;
    value.output_last_dim = out_dims[out_count - 1];
    value.group_size = group_size;
    value.transpose_x1 = transpose_x1 ? 1U : 0U;
    value.transpose_x2 = transpose_x2 ? 1U : 0U;
    *metadata = value;
    return true;
}

bool capture_tensor_metadata(const aclTensor *tensor, Fault20TensorMetadata *metadata) {
    if (tensor == nullptr || metadata == nullptr) {
        return false;
    }
    aclDataType dtype = ACL_DT_UNDEFINED;
    aclFormat format = ACL_FORMAT_UNDEFINED;
    int64_t view_offset = 0;
    int64_t *view_dims = nullptr;
    int64_t *storage_dims = nullptr;
    int64_t *strides = nullptr;
    uint64_t view_count = 0;
    uint64_t storage_count = 0;
    uint64_t stride_count = 0;
    void *raw_addr = nullptr;
    if (aclGetDataType(tensor, &dtype) != ACL_SUCCESS ||
        aclGetFormat(tensor, &format) != ACL_SUCCESS ||
        aclGetViewOffset(tensor, &view_offset) != ACL_SUCCESS ||
        aclGetViewShape(tensor, &view_dims, &view_count) != ACL_SUCCESS ||
        aclGetStorageShape(tensor, &storage_dims, &storage_count) != ACL_SUCCESS ||
        aclGetViewStrides(tensor, &strides, &stride_count) != ACL_SUCCESS ||
        aclGetRawTensorAddr(tensor, &raw_addr) != ACL_SUCCESS || view_count == 0 ||
        storage_count == 0 || stride_count == 0 || view_count > FAULT20_MAX_DIMS ||
        storage_count > FAULT20_MAX_DIMS || stride_count > FAULT20_MAX_DIMS) {
        return false;
    }
    int64_t numel = 0;
    int64_t storage_elements = 0;
    if (!checked_product(view_dims, view_count, &numel) ||
        !checked_product(storage_dims, storage_count, &storage_elements)) {
        return false;
    }
    const size_t element_size = aclDataTypeSize(dtype);
    const uint64_t view_bytes = checked_bytes(numel, element_size);
    const uint64_t storage_bytes = checked_bytes(storage_elements, element_size);
    if (element_size == 0 || view_bytes == 0 || storage_bytes == 0) {
        return false;
    }
    Fault20TensorMetadata value{};
    value.dtype = static_cast<int32_t>(dtype);
    value.format = static_cast<int32_t>(format);
    value.present = 1;
    value.contiguous = is_contiguous(view_dims, view_count, strides, stride_count) ? 1U : 0U;
    value.view_rank = static_cast<uint32_t>(view_count);
    value.storage_rank = static_cast<uint32_t>(storage_count);
    value.stride_rank = static_cast<uint32_t>(stride_count);
    for (uint64_t i = 0; i < view_count; ++i) {
        value.view_shape[i] = view_dims[i];
    }
    for (uint64_t i = 0; i < storage_count; ++i) {
        value.storage_shape[i] = storage_dims[i];
    }
    for (uint64_t i = 0; i < stride_count; ++i) {
        value.strides[i] = strides[i];
    }
    value.view_offset = view_offset;
    value.numel = numel;
    value.storage_elements = storage_elements;
    value.element_size = element_size;
    value.view_bytes = view_bytes;
    value.storage_bytes = storage_bytes;
    value.tensor_ptr = reinterpret_cast<uintptr_t>(tensor);
    value.storage_ptr = reinterpret_cast<uintptr_t>(tensor->GetStorage());
    value.raw_addr = reinterpret_cast<uintptr_t>(raw_addr);
    value.storage_addr = reinterpret_cast<uintptr_t>(tensor->GetStorageAddr());
    value.data_addr = reinterpret_cast<uintptr_t>(tensor->GetData());
    *metadata = value;
    return true;
}

uint64_t checked_bytes(int64_t elements, size_t element_size) {
    if (elements < 0 || element_size == 0) {
        return 0;
    }
    const uint64_t count = static_cast<uint64_t>(elements);
    if (count > std::numeric_limits<uint64_t>::max() / element_size) {
        return 0;
    }
    return count * element_size;
}

size_t append_dims(char *line, size_t capacity, size_t used, const int64_t *dims, uint64_t count) {
    const uint64_t limit = count < kMaxDims ? count : kMaxDims;
    for (uint64_t i = 0; i < limit && used < capacity; ++i) {
        const int written = std::snprintf(line + used, capacity - used, "%s%" PRId64,
                                          i == 0 ? "" : ",", dims[i]);
        if (written < 0 || static_cast<size_t>(written) >= capacity - used) {
            return capacity;
        }
        used += static_cast<size_t>(written);
    }
    return used;
}

void log_call(const char *event, uint64_t invocation, aclnnStatus status, aclOpExecutor *executor,
              void *workspace, uint64_t workspace_size, aclrtStream stream) {
    char line[1024];
    const int written = std::snprintf(
        line, sizeof(line),
        "{\"event\":\"%s\",\"pid\":%ld,\"tid\":%ld,\"ts_ns\":%" PRIu64
        ",\"sequence\":%" PRIu64 ",\"invocation\":%" PRIu64
        ",\"status\":%d,\"executor\":\"%p\",\"cache\":\"%p\","
        "\"workspace\":\"%p\",\"workspace_size\":%" PRIu64 ",\"stream\":\"%p\"}",
        event, static_cast<long>(getpid()), static_cast<long>(syscall(SYS_gettid)),
        monotonic_ns(), g_sequence.fetch_add(1, std::memory_order_relaxed) + 1, invocation,
        static_cast<int>(status), executor, nullptr, workspace, workspace_size, stream);
    if (written > 0 && static_cast<size_t>(written) < sizeof(line)) {
        write_line(line, static_cast<size_t>(written));
    }
}

void log_tensor(const char *api, const char *role, uint64_t invocation, aclOpExecutor *executor,
                const aclTensor *tensor) {
    char line[4096];
    if (tensor == nullptr) {
        const int written = std::snprintf(
            line, sizeof(line),
            "{\"event\":\"tensor_metadata\",\"api\":\"%s\",\"role\":\"%s\","
            "\"pid\":%ld,\"tid\":%ld,\"ts_ns\":%" PRIu64
            ",\"sequence\":%" PRIu64 ",\"invocation\":%" PRIu64
            ",\"executor\":\"%p\",\"cache\":\"%p\",\"tensor\":\"%p\","
            "\"present\":false}",
            api, role, static_cast<long>(getpid()), static_cast<long>(syscall(SYS_gettid)),
            monotonic_ns(), g_sequence.fetch_add(1, std::memory_order_relaxed) + 1, invocation,
            executor, nullptr, tensor);
        if (written > 0 && static_cast<size_t>(written) < sizeof(line)) {
            write_line(line, static_cast<size_t>(written));
        }
        return;
    }

    aclDataType dtype = ACL_DT_UNDEFINED;
    aclFormat format = ACL_FORMAT_UNDEFINED;
    int64_t view_offset = 0;
    int64_t *view_dims = nullptr;
    int64_t *storage_dims = nullptr;
    int64_t *strides = nullptr;
    uint64_t view_count = 0;
    uint64_t storage_count = 0;
    uint64_t stride_count = 0;
    void *raw_addr = nullptr;
    (void)aclGetDataType(tensor, &dtype);
    (void)aclGetFormat(tensor, &format);
    (void)aclGetViewOffset(tensor, &view_offset);
    (void)aclGetViewShape(tensor, &view_dims, &view_count);
    (void)aclGetStorageShape(tensor, &storage_dims, &storage_count);
    (void)aclGetViewStrides(tensor, &strides, &stride_count);
    (void)aclGetRawTensorAddr(tensor, &raw_addr);

    const int64_t numel = tensor->Numel();
    const int64_t storage_elements = tensor->Size();
    const size_t element_size = aclDataTypeSize(dtype);
    const uint64_t nominal_view_bytes = checked_bytes(numel, element_size);
    const uint64_t storage_bytes = checked_bytes(storage_elements, element_size);
    const aclStorage *storage = tensor->GetStorage();
    void *storage_addr = tensor->GetStorageAddr();
    void *data_addr = tensor->GetData();

    int written = std::snprintf(
        line, sizeof(line),
        "{\"event\":\"tensor_metadata\",\"api\":\"%s\",\"role\":\"%s\","
        "\"pid\":%ld,\"tid\":%ld,\"ts_ns\":%" PRIu64
        ",\"sequence\":%" PRIu64 ",\"invocation\":%" PRIu64
        ",\"executor\":\"%p\",\"cache\":\"%p\",\"tensor\":\"%p\","
        "\"storage\":\"%p\",\"storage_addr\":\"%p\",\"raw_addr\":\"%p\","
        "\"data_addr\":\"%p\",\"present\":true,\"dtype\":%d,\"format\":%d,"
        "\"element_size\":%zu,\"numel\":%" PRId64 ",\"nominal_view_bytes\":%" PRIu64
        ",\"storage_elements\":%" PRId64 ",\"storage_bytes\":%" PRIu64
        ",\"view_offset\":%" PRId64 ",\"view_shape\":[",
        api, role, static_cast<long>(getpid()), static_cast<long>(syscall(SYS_gettid)),
        monotonic_ns(), g_sequence.fetch_add(1, std::memory_order_relaxed) + 1, invocation,
        executor, nullptr, tensor, storage, storage_addr, raw_addr, data_addr,
        static_cast<int>(dtype), static_cast<int>(format), element_size, numel,
        nominal_view_bytes, storage_elements, storage_bytes, view_offset);
    if (written < 0 || static_cast<size_t>(written) >= sizeof(line)) {
        return;
    }
    size_t used = static_cast<size_t>(written);
    used = append_dims(line, sizeof(line), used, view_dims, view_count);
    if (used >= sizeof(line)) {
        return;
    }
    written = std::snprintf(line + used, sizeof(line) - used, "],\"storage_shape\":[");
    if (written < 0 || static_cast<size_t>(written) >= sizeof(line) - used) {
        return;
    }
    used += static_cast<size_t>(written);
    used = append_dims(line, sizeof(line), used, storage_dims, storage_count);
    if (used >= sizeof(line)) {
        return;
    }
    written = std::snprintf(line + used, sizeof(line) - used, "],\"strides\":[");
    if (written < 0 || static_cast<size_t>(written) >= sizeof(line) - used) {
        return;
    }
    used += static_cast<size_t>(written);
    used = append_dims(line, sizeof(line), used, strides, stride_count);
    if (used >= sizeof(line)) {
        return;
    }
    written = std::snprintf(line + used, sizeof(line) - used,
                            "],\"view_rank\":%" PRIu64 ",\"storage_rank\":%" PRIu64
                            ",\"stride_rank\":%" PRIu64 "}",
                            view_count, storage_count, stride_count);
    if (written < 0 || static_cast<size_t>(written) >= sizeof(line) - used) {
        return;
    }
    used += static_cast<size_t>(written);
    write_line(line, used);
}

void log_matmul_summary(uint64_t invocation, aclOpExecutor *executor,
                        const Fault20MatmulMetadata &metadata) {
    char line[2048];
    const int written = std::snprintf(
        line, sizeof(line),
        "{\"event\":\"matmulweightnz_gws_summary\",\"pid\":%ld,\"tid\":%ld,"
        "\"ts_ns\":%" PRIu64 ",\"sequence\":%" PRIu64
        ",\"invocation\":%" PRIu64 ",\"executor\":\"%p\","
        "\"cube_math_type\":%d,\"all_present\":%s,\"any_empty_storage\":%s,"
        "\"any_shared_storage\":%s,\"self_storage\":\"0x%" PRIx64
        "\",\"mat2_storage\":\"0x%" PRIx64 "\",\"out_storage\":\"0x%" PRIx64
        "\",\"self_storage_addr\":\"0x%" PRIx64
        "\",\"mat2_storage_addr\":\"0x%" PRIx64
        "\",\"out_storage_addr\":\"0x%" PRIx64 "\"}",
        static_cast<long>(getpid()), static_cast<long>(syscall(SYS_gettid)), monotonic_ns(),
        g_sequence.fetch_add(1, std::memory_order_relaxed) + 1, invocation, executor,
        metadata.cube_math_type, metadata.all_present ? "true" : "false",
        metadata.any_empty_storage ? "true" : "false",
        metadata.any_shared_storage ? "true" : "false", metadata.self.storage_ptr,
        metadata.mat2.storage_ptr, metadata.out.storage_ptr, metadata.self.storage_addr,
        metadata.mat2.storage_addr, metadata.out.storage_addr);
    if (written > 0 && static_cast<size_t>(written) < sizeof(line)) {
        write_line(line, static_cast<size_t>(written));
    }
}

}  // namespace

extern "C" aclnnStatus aclnnQuantMatmulWeightNzGetWorkspaceSize(
    const aclTensor *x1, const aclTensor *x2, const aclTensor *x1_scale, const aclTensor *x2_scale,
    const aclTensor *y_scale, const aclTensor *x1_offset, const aclTensor *x2_offset,
    const aclTensor *y_offset, const aclTensor *bias, bool transpose_x1, bool transpose_x2,
    int64_t group_size, aclTensor *out, uint64_t *workspace_size, aclOpExecutor **executor) {
    pthread_once(&g_resolve_once, resolve_real);
    const uint64_t invocation = g_sequence.fetch_add(1, std::memory_order_relaxed) + 1;
    if (g_real_qmm_getws == nullptr) {
        log_call("qmm_getws_resolve_failed", invocation, 1, nullptr, nullptr, 0, nullptr);
        return 1;
    }
    const bool registry_hooks_ready = g_fault20_gws_begin != nullptr &&
                                      g_fault20_gws_commit != nullptr &&
                                      g_fault20_gws_abort != nullptr;
    if (registry_hooks_ready) {
        g_fault20_gws_begin();
    }
    const aclnnStatus status = g_real_qmm_getws(
        x1, x2, x1_scale, x2_scale, y_scale, x1_offset, x2_offset, y_offset, bias,
        transpose_x1, transpose_x2, group_size, out, workspace_size, executor);
    if (registry_hooks_ready) {
        Fault20X1Metadata metadata{};
        if (status == ACL_SUCCESS && capture_x1_metadata(x1, out, transpose_x1, transpose_x2,
                                                         group_size, &metadata)) {
            (void)g_fault20_gws_commit(&metadata);
        } else {
            g_fault20_gws_abort();
        }
    }
    aclOpExecutor *resolved_executor = executor == nullptr ? nullptr : *executor;
    log_call("qmm_getws", invocation, status, resolved_executor, nullptr,
             workspace_size == nullptr ? 0 : *workspace_size, nullptr);
    log_tensor("aclnnQuantMatmulWeightNz", "x1", invocation, resolved_executor, x1);
    log_tensor("aclnnQuantMatmulWeightNz", "x2", invocation, resolved_executor, x2);
    log_tensor("aclnnQuantMatmulWeightNz", "x1_scale", invocation, resolved_executor, x1_scale);
    log_tensor("aclnnQuantMatmulWeightNz", "x2_scale", invocation, resolved_executor, x2_scale);
    log_tensor("aclnnQuantMatmulWeightNz", "y_scale", invocation, resolved_executor, y_scale);
    log_tensor("aclnnQuantMatmulWeightNz", "x1_offset", invocation, resolved_executor, x1_offset);
    log_tensor("aclnnQuantMatmulWeightNz", "x2_offset", invocation, resolved_executor, x2_offset);
    log_tensor("aclnnQuantMatmulWeightNz", "y_offset", invocation, resolved_executor, y_offset);
    log_tensor("aclnnQuantMatmulWeightNz", "bias", invocation, resolved_executor, bias);
    log_tensor("aclnnQuantMatmulWeightNz", "out", invocation, resolved_executor, out);
    return status;
}

extern "C" aclnnStatus aclnnQuantMatmulWeightNz(void *workspace, uint64_t workspace_size,
                                                  aclOpExecutor *executor, aclrtStream stream) {
    pthread_once(&g_resolve_once, resolve_real);
    const uint64_t invocation = g_sequence.fetch_add(1, std::memory_order_relaxed) + 1;
    log_call("qmm_execute_enter", invocation, 0, executor, workspace, workspace_size, stream);
    if (g_real_qmm_execute == nullptr) {
        log_call("qmm_execute_resolve_failed", invocation, 1, executor, workspace,
                 workspace_size, stream);
        return 1;
    }
    const aclnnStatus status = g_real_qmm_execute(workspace, workspace_size, executor, stream);
    log_call("qmm_execute_exit", invocation, status, executor, workspace, workspace_size,
             stream);
    return status;
}

extern "C" aclnnStatus aclnnMatmulWeightNzGetWorkspaceSize(
    const aclTensor *self, const aclTensor *mat2, aclTensor *out, int8_t cube_math_type,
    uint64_t *workspace_size, aclOpExecutor **executor) {
    pthread_once(&g_resolve_once, resolve_real);
    const uint64_t invocation = g_sequence.fetch_add(1, std::memory_order_relaxed) + 1;
    if (g_real_matmul_getws == nullptr) {
        log_call("matmulweightnz_getws_resolve_failed", invocation, 1, nullptr, nullptr, 0,
                 nullptr);
        return 1;
    }
    const bool registry_hooks_ready = g_fault20_matmul_gws_begin != nullptr &&
                                      g_fault20_matmul_gws_commit != nullptr &&
                                      g_fault20_matmul_gws_abort != nullptr;
    if (registry_hooks_ready) {
        g_fault20_matmul_gws_begin();
    }
    const aclnnStatus status = g_real_matmul_getws(self, mat2, out, cube_math_type,
                                                   workspace_size, executor);
    Fault20MatmulMetadata metadata{};
    const bool metadata_ok = status == ACL_SUCCESS &&
                             capture_tensor_metadata(self, &metadata.self) &&
                             capture_tensor_metadata(mat2, &metadata.mat2) &&
                             capture_tensor_metadata(out, &metadata.out);
    if (metadata_ok) {
        metadata.cube_math_type = static_cast<int32_t>(cube_math_type);
        metadata.all_present = 1;
        metadata.any_empty_storage =
            metadata.self.storage_ptr == 0 || metadata.mat2.storage_ptr == 0 ||
                    metadata.out.storage_ptr == 0 || metadata.self.storage_addr == 0 ||
                    metadata.mat2.storage_addr == 0 || metadata.out.storage_addr == 0
                ? 1U
                : 0U;
        metadata.any_shared_storage =
            (metadata.self.storage_ptr != 0 &&
             metadata.self.storage_ptr == metadata.mat2.storage_ptr) ||
                    (metadata.self.storage_ptr != 0 &&
                     metadata.self.storage_ptr == metadata.out.storage_ptr) ||
                    (metadata.mat2.storage_ptr != 0 &&
                     metadata.mat2.storage_ptr == metadata.out.storage_ptr)
                ? 1U
                : 0U;
    }
    if (registry_hooks_ready) {
        if (metadata_ok) {
            (void)g_fault20_matmul_gws_commit(&metadata);
        } else {
            g_fault20_matmul_gws_abort();
        }
    }
    aclOpExecutor *resolved_executor = executor == nullptr ? nullptr : *executor;
    log_call("matmulweightnz_getws", invocation, status, resolved_executor, nullptr,
             workspace_size == nullptr ? 0 : *workspace_size, nullptr);
    log_tensor("aclnnMatmulWeightNz", "self", invocation, resolved_executor, self);
    log_tensor("aclnnMatmulWeightNz", "mat2", invocation, resolved_executor, mat2);
    log_tensor("aclnnMatmulWeightNz", "out", invocation, resolved_executor, out);
    if (metadata_ok) {
        log_matmul_summary(invocation, resolved_executor, metadata);
    }
    return status;
}

extern "C" aclnnStatus aclnnMatmulWeightNz(void *workspace, uint64_t workspace_size,
                                             aclOpExecutor *executor, aclrtStream stream) {
    pthread_once(&g_resolve_once, resolve_real);
    const uint64_t invocation = g_sequence.fetch_add(1, std::memory_order_relaxed) + 1;
    log_call("matmulweightnz_execute_enter", invocation, 0, executor, workspace,
             workspace_size, stream);
    if (g_real_matmul_execute == nullptr) {
        log_call("matmulweightnz_execute_resolve_failed", invocation, 1, executor, workspace,
                 workspace_size, stream);
        return 1;
    }
    const aclnnStatus status =
        g_real_matmul_execute(workspace, workspace_size, executor, stream);
    log_call("matmulweightnz_execute_exit", invocation, status, executor, workspace,
             workspace_size, stream);
    return status;
}

extern "C" aclnnStatus aclnnAscendQuantV3GetWorkspaceSize(
    const aclTensor *x, const aclTensor *scale, const aclTensor *offset, bool sqrt_mode,
    const char *round_mode, int32_t dst_type, int32_t axis, const aclTensor *y,
    uint64_t *workspace_size, aclOpExecutor **executor) {
    pthread_once(&g_resolve_once, resolve_real);
    const uint64_t invocation = g_sequence.fetch_add(1, std::memory_order_relaxed) + 1;
    if (g_real_quant_getws == nullptr) {
        log_call("quant_getws_resolve_failed", invocation, 1, nullptr, nullptr, 0, nullptr);
        return 1;
    }
    const aclnnStatus status = g_real_quant_getws(x, scale, offset, sqrt_mode, round_mode, dst_type,
                                                   axis, y, workspace_size, executor);
    aclOpExecutor *resolved_executor = executor == nullptr ? nullptr : *executor;
    log_call("quant_getws", invocation, status, resolved_executor, nullptr,
             workspace_size == nullptr ? 0 : *workspace_size, nullptr);
    log_tensor("aclnnAscendQuantV3", "x", invocation, resolved_executor, x);
    log_tensor("aclnnAscendQuantV3", "scale", invocation, resolved_executor, scale);
    log_tensor("aclnnAscendQuantV3", "offset", invocation, resolved_executor, offset);
    log_tensor("aclnnAscendQuantV3", "y", invocation, resolved_executor, y);
    return status;
}

extern "C" aclnnStatus aclnnAscendQuantV3(void *workspace, uint64_t workspace_size,
                                            aclOpExecutor *executor, aclrtStream stream) {
    pthread_once(&g_resolve_once, resolve_real);
    const uint64_t invocation = g_sequence.fetch_add(1, std::memory_order_relaxed) + 1;
    log_call("quant_execute_enter", invocation, 0, executor, workspace, workspace_size, stream);
    if (g_real_quant_execute == nullptr) {
        log_call("quant_execute_resolve_failed", invocation, 1, executor, workspace,
                 workspace_size, stream);
        return 1;
    }
    const aclnnStatus status = g_real_quant_execute(workspace, workspace_size, executor, stream);
    log_call("quant_execute_exit", invocation, status, executor, workspace, workspace_size,
             stream);
    return status;
}
