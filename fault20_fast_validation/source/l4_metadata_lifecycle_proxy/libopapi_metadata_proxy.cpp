#include <acl/acl.h>
#include <aclnn/acl_meta.h>
#include <aclnn/opdev/common_types.h>
#include <aclnnop/aclnn_ascend_quant_v3.h>
#include <aclnnop/aclnn_copy.h>
#include <aclnnop/aclnn_matmul.h>
#include <aclnnop/aclnn_quant_matmul_weight_nz.h>

#include "../l4_buffer_substitute_allhit_v1/fault20_buffer_api.h"
#include "../../../fault39/native/metadata_array_owner.h"
#include "../fault26_readonly_trace_v1/fault26_trace_shared.h"

#include <atomic>
#include <cmath>
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
#ifdef DEMONCAT_QMM_AUTO
#include "fault39_build_profile.h"
#endif

namespace {

#ifdef DEMONCAT_QMM_AUTO
using CurrentGwsBegin = uint64_t (*)(uint32_t);
using CurrentGwsCommit = bool (*)(uint64_t, uint32_t, const Fault20MetadataAbiDescriptor*, const void*);
using CurrentGwsAbort = void (*)(uint64_t, uint32_t);
using CurrentExecuteBegin = uint64_t (*)(uint32_t, const void*);
using CurrentExecuteEnd = void (*)(uint64_t, uint32_t, const void*);
CurrentGwsBegin g_current_gws_begin = nullptr;
CurrentGwsCommit g_current_gws_commit = nullptr;
CurrentGwsAbort g_current_gws_abort = nullptr;
CurrentExecuteBegin g_current_execute_begin = nullptr;
CurrentExecuteEnd g_current_execute_end = nullptr;

struct CurrentGwsScope {
    uint32_t kind;
    uint64_t token;
    bool finished = false;
    explicit CurrentGwsScope(uint32_t value) : kind(value),
        token(g_current_gws_begin ? g_current_gws_begin(kind) : 0) {}
    ~CurrentGwsScope() {
        if (!finished && token && g_current_gws_abort) g_current_gws_abort(token, kind);
    }
    void commit(const Fault20MetadataAbiDescriptor& metadata, const void* executor) {
        if (token && g_current_gws_commit) (void)g_current_gws_commit(token, kind, &metadata, executor);
        finished = true;
    }
};

struct CurrentExecuteScope {
    uint32_t kind;
    const void* executor;
    uint64_t token;
    CurrentExecuteScope(uint32_t value, const void* actual) : kind(value), executor(actual),
        token(g_current_execute_begin ? g_current_execute_begin(kind, executor) : 0) {}
    ~CurrentExecuteScope() {
        if (token && g_current_execute_end) g_current_execute_end(token, kind, executor);
    }
};
#endif

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
using InplaceCopyGetWorkspaceSizeFn = aclnnStatus (*)(
    aclTensor *, const aclTensor *, uint64_t *, aclOpExecutor **);
using ExecuteFn = aclnnStatus (*)(void *, uint64_t, aclOpExecutor *, aclrtStream);
using Fault20GwsBeginFn = void (*)();
using Fault20GwsCommitFn = bool (*)(const Fault20QmmMetadata *);
using Fault20MatmulGwsCommitFn = bool (*)(const Fault20MatmulMetadata *);
using Fault20VersionedGwsCommitFn = bool (*)(
    const Fault20MetadataAbiDescriptor *);
using Fault20GwsAbortFn = void (*)();
using Fault26TraceBeginFn = void (*)(uint32_t, const char *, void *, uint64_t,
                                     aclOpExecutor *, aclrtStream);
using Fault26TraceEndFn = void (*)(uint32_t, const char *, aclnnStatus);
using Fault26TraceWriterGwsFn = void (*)(
    const char *, const char *, const aclTensor *, uint64_t, aclOpExecutor *,
    aclnnStatus);

pthread_once_t g_resolve_once = PTHREAD_ONCE_INIT;
QuantMatmulGetWorkspaceSizeFn g_real_qmm_getws = nullptr;
AscendQuantGetWorkspaceSizeFn g_real_quant_getws = nullptr;
MatmulWeightNzGetWorkspaceSizeFn g_real_matmul_getws = nullptr;
MatmulWeightNzGetWorkspaceSizeFn g_real_float_matmul_getws = nullptr;
InplaceCopyGetWorkspaceSizeFn g_real_inplace_copy_getws = nullptr;
ExecuteFn g_real_qmm_execute = nullptr;
ExecuteFn g_real_quant_execute = nullptr;
ExecuteFn g_real_matmul_execute = nullptr;
ExecuteFn g_real_float_matmul_execute = nullptr;
ExecuteFn g_real_inplace_copy_execute = nullptr;
Fault20GwsBeginFn g_fault20_gws_begin = nullptr;
Fault20VersionedGwsCommitFn g_fault20_gws_commit_v17 = nullptr;
Fault20GwsCommitFn g_fault20_gws_commit = nullptr;
Fault20GwsAbortFn g_fault20_gws_abort = nullptr;
Fault20GwsBeginFn g_fault20_matmul_gws_begin = nullptr;
Fault20VersionedGwsCommitFn g_fault20_matmul_gws_commit_v17 = nullptr;
Fault20MatmulGwsCommitFn g_fault20_matmul_gws_commit = nullptr;
Fault20GwsAbortFn g_fault20_matmul_gws_abort = nullptr;
Fault20GwsBeginFn g_fault20_float_matmul_gws_begin = nullptr;
Fault20VersionedGwsCommitFn g_fault20_float_matmul_gws_commit_v17 = nullptr;
Fault20MatmulGwsCommitFn g_fault20_float_matmul_gws_commit = nullptr;
Fault20GwsAbortFn g_fault20_float_matmul_gws_abort = nullptr;
Fault26TraceBeginFn g_fault26_trace_begin = nullptr;
Fault26TraceEndFn g_fault26_trace_end = nullptr;
Fault26TraceWriterGwsFn g_fault26_trace_writer_gws = nullptr;
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
    const ssize_t written = writev(fd, parts, 2);
    (void)written;
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
    g_real_float_matmul_getws = reinterpret_cast<MatmulWeightNzGetWorkspaceSizeFn>(
        dlsym(RTLD_NEXT, "aclnnMatmulGetWorkspaceSize"));
    g_real_float_matmul_execute =
        reinterpret_cast<ExecuteFn>(dlsym(RTLD_NEXT, "aclnnMatmul"));
    g_real_inplace_copy_getws = reinterpret_cast<InplaceCopyGetWorkspaceSizeFn>(
        dlsym(RTLD_NEXT, "aclnnInplaceCopyGetWorkspaceSize"));
    g_real_inplace_copy_execute =
        reinterpret_cast<ExecuteFn>(dlsym(RTLD_NEXT, "aclnnInplaceCopy"));
    g_fault20_gws_begin = reinterpret_cast<Fault20GwsBeginFn>(
        dlsym(RTLD_DEFAULT, "fault20_qmm_gws_begin"));
    g_fault20_gws_commit_v17 = reinterpret_cast<Fault20VersionedGwsCommitFn>(
        dlsym(RTLD_DEFAULT, "fault20_qmm_gws_commit_v17"));
    g_fault20_gws_commit = reinterpret_cast<Fault20GwsCommitFn>(
        dlsym(RTLD_DEFAULT, "fault20_qmm_gws_commit"));
    g_fault20_gws_abort = reinterpret_cast<Fault20GwsAbortFn>(
        dlsym(RTLD_DEFAULT, "fault20_qmm_gws_abort"));
    g_fault20_matmul_gws_begin = reinterpret_cast<Fault20GwsBeginFn>(
        dlsym(RTLD_DEFAULT, "fault20_matmul_gws_begin"));
    g_fault20_matmul_gws_commit_v17 =
        reinterpret_cast<Fault20VersionedGwsCommitFn>(
            dlsym(RTLD_DEFAULT, "fault20_matmul_gws_commit_v17"));
    g_fault20_matmul_gws_commit = reinterpret_cast<Fault20MatmulGwsCommitFn>(
        dlsym(RTLD_DEFAULT, "fault20_matmul_gws_commit"));
    g_fault20_matmul_gws_abort = reinterpret_cast<Fault20GwsAbortFn>(
        dlsym(RTLD_DEFAULT, "fault20_matmul_gws_abort"));
    g_fault20_float_matmul_gws_begin = reinterpret_cast<Fault20GwsBeginFn>(
        dlsym(RTLD_DEFAULT, "fault20_float_matmul_gws_begin"));
    g_fault20_float_matmul_gws_commit_v17 =
        reinterpret_cast<Fault20VersionedGwsCommitFn>(
            dlsym(RTLD_DEFAULT, "fault20_float_matmul_gws_commit_v17"));
    g_fault20_float_matmul_gws_commit =
        reinterpret_cast<Fault20MatmulGwsCommitFn>(
            dlsym(RTLD_DEFAULT, "fault20_float_matmul_gws_commit"));
    g_fault20_float_matmul_gws_abort = reinterpret_cast<Fault20GwsAbortFn>(
        dlsym(RTLD_DEFAULT, "fault20_float_matmul_gws_abort"));
    g_fault26_trace_begin = reinterpret_cast<Fault26TraceBeginFn>(
        dlsym(RTLD_DEFAULT, "fault26_trace_execute_begin"));
    g_fault26_trace_end = reinterpret_cast<Fault26TraceEndFn>(
        dlsym(RTLD_DEFAULT, "fault26_trace_execute_end"));
    g_fault26_trace_writer_gws = reinterpret_cast<Fault26TraceWriterGwsFn>(
        dlsym(RTLD_DEFAULT, "fault26_trace_writer_gws"));
#ifdef DEMONCAT_QMM_AUTO
#ifdef DEMONCAT_QMM_SPLIT
    const char* library = "libopapi_nn_real.so";
    const auto& image = f39profile::split;
#else
    const char* library = "libopapi_real.so";
    const auto& image = f39profile::opapi;
#endif
    void* handle = dlopen(library, RTLD_LAZY | RTLD_NOLOAD);
    auto* getws = handle ? dlsym(handle, "aclnnQuantMatmulWeightNzGetWorkspaceSize") : nullptr;
    auto* execute = handle ? dlsym(handle, "aclnnQuantMatmulWeightNz") : nullptr;
    auto* weight_nz_getws = handle ? dlsym(handle, "aclnnMatmulWeightNzGetWorkspaceSize") : nullptr;
    auto* weight_nz_execute = handle ? dlsym(handle, "aclnnMatmulWeightNz") : nullptr;
    using Profile = const char* (*)();
    const auto profile = reinterpret_cast<Profile>(dlsym(RTLD_DEFAULT, "dcat20_qmm_profile_v1"));
    bool verified = profile && std::strcmp(profile(), f39profile::id) == 0;
    if (verified) {
        g_current_gws_begin = reinterpret_cast<CurrentGwsBegin>(dlsym(RTLD_DEFAULT, "dcat20_auto_gws_begin_v1"));
        g_current_gws_commit = reinterpret_cast<CurrentGwsCommit>(dlsym(RTLD_DEFAULT, "dcat20_auto_gws_commit_v1"));
        g_current_gws_abort = reinterpret_cast<CurrentGwsAbort>(dlsym(RTLD_DEFAULT, "dcat20_auto_gws_abort_v1"));
        g_current_execute_begin = reinterpret_cast<CurrentExecuteBegin>(dlsym(RTLD_DEFAULT, "dcat20_auto_execute_begin_v1"));
        g_current_execute_end = reinterpret_cast<CurrentExecuteEnd>(dlsym(RTLD_DEFAULT, "dcat20_auto_execute_end_v1"));
        if (!g_current_gws_begin || !g_current_gws_commit || !g_current_gws_abort
            || !g_current_execute_begin || !g_current_execute_end) {
            g_current_gws_begin = nullptr;
            g_current_execute_begin = nullptr;
            verified = false;
        }
    }
    for (void* symbol : {reinterpret_cast<void*>(&aclGetDataType), reinterpret_cast<void*>(&aclGetFormat),
            reinterpret_cast<void*>(&aclGetViewOffset), reinterpret_cast<void*>(&aclGetViewShape),
            reinterpret_cast<void*>(&aclGetStorageShape), reinterpret_cast<void*>(&aclGetViewStrides),
            reinterpret_cast<void*>(&aclGetRawTensorAddr)}) {
        verified = f39identity::matches(symbol, f39profile::nn) && verified;
    }
    if (verified && f39identity::matches(getws, image) && f39identity::matches(execute, image)) {
        g_real_qmm_getws = reinterpret_cast<QuantMatmulGetWorkspaceSizeFn>(getws);
        g_real_qmm_execute = reinterpret_cast<ExecuteFn>(execute);
    } else {
        g_fault20_gws_begin = nullptr;
        g_fault20_gws_commit_v17 = nullptr;
    }
    if (verified && f39identity::matches(weight_nz_getws, image)
        && f39identity::matches(weight_nz_execute, image)) {
        g_real_matmul_getws = reinterpret_cast<MatmulWeightNzGetWorkspaceSizeFn>(weight_nz_getws);
        g_real_matmul_execute = reinterpret_cast<ExecuteFn>(weight_nz_execute);
    } else {
        g_fault20_matmul_gws_begin = nullptr;
        g_fault20_matmul_gws_commit_v17 = nullptr;
    }
    if (handle) dlclose(handle);
#endif
}

Fault20MetadataAbiDescriptor make_metadata_descriptor(uint32_t payload_kind,
                                                      uint32_t payload_size,
                                                      const void *payload) {
    Fault20MetadataAbiDescriptor descriptor{};
    descriptor.magic = FAULT20_METADATA_ABI_MAGIC;
    descriptor.version = FAULT20_METADATA_ABI_VERSION;
    descriptor.descriptor_size = sizeof(Fault20MetadataAbiDescriptor);
    descriptor.payload_kind = payload_kind;
    descriptor.payload_size = payload_size;
    descriptor.payload_address = reinterpret_cast<uint64_t>(payload);
    return descriptor;
}

bool commit_qmm_metadata(const Fault20QmmMetadata &metadata) {
    if (g_fault20_gws_commit_v17 != nullptr) {
        const Fault20MetadataAbiDescriptor descriptor = make_metadata_descriptor(
            FAULT20_METADATA_PAYLOAD_QMM, sizeof(metadata), &metadata);
        return g_fault20_gws_commit_v17(&descriptor);
    }
    return g_fault20_gws_commit != nullptr && g_fault20_gws_commit(&metadata);
}

bool commit_matmul_metadata(const Fault20MatmulMetadata &metadata,
                            bool floating_api) {
    Fault20VersionedGwsCommitFn versioned =
        floating_api ? g_fault20_float_matmul_gws_commit_v17
                     : g_fault20_matmul_gws_commit_v17;
    if (versioned != nullptr) {
        const uint32_t kind = floating_api
                                  ? FAULT20_METADATA_PAYLOAD_FLOAT_MATMUL
                                  : FAULT20_METADATA_PAYLOAD_MATMUL_WEIGHT_NZ;
        const Fault20MetadataAbiDescriptor descriptor =
            make_metadata_descriptor(kind, sizeof(metadata), &metadata);
        return versioned(&descriptor);
    }
    Fault20MatmulGwsCommitFn legacy =
        floating_api ? g_fault20_float_matmul_gws_commit
                     : g_fault20_matmul_gws_commit;
    return legacy != nullptr && legacy(&metadata);
}

void notify_fault26_writer_gws(const char *api, const char *role,
                               const aclTensor *writable,
                               uint64_t workspace_size,
                               aclOpExecutor *executor,
                               aclnnStatus status) {
    if (g_fault26_trace_writer_gws != nullptr) {
        g_fault26_trace_writer_gws(api, role, writable, workspace_size,
                                   executor, status);
    }
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
                         bool transpose_x2, int64_t group_size, Fault20X1Metadata *metadata) try {
    dcat_metadata::CaptureScope capture_scope;
    if (x1 == nullptr || out == nullptr || metadata == nullptr) {
        return false;
    }
    aclDataType dtype = ACL_DT_UNDEFINED;
    aclFormat format = ACL_FORMAT_UNDEFINED;
    int64_t view_offset = 0;
    dcat_metadata::OwnedArray view_dims;
    dcat_metadata::OwnedArray storage_dims;
    dcat_metadata::OwnedArray strides;
    dcat_metadata::OwnedArray out_dims;
    uint64_t view_count = 0;
    uint64_t storage_count = 0;
    uint64_t stride_count = 0;
    uint64_t out_count = 0;
    if (aclGetDataType(x1, &dtype) != ACL_SUCCESS ||
        aclGetFormat(x1, &format) != ACL_SUCCESS ||
        aclGetViewOffset(x1, &view_offset) != ACL_SUCCESS ||
        view_dims.read(aclGetViewShape, x1, &view_count) != ACL_SUCCESS ||
        storage_dims.read(aclGetStorageShape, x1, &storage_count) != ACL_SUCCESS ||
        strides.read(aclGetViewStrides, x1, &stride_count) != ACL_SUCCESS ||
        out_dims.read(aclGetViewShape, out, &out_count) != ACL_SUCCESS ||
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
} catch (...) {
    dcat_metadata::capture_exception();
    return false;
}

bool capture_tensor_metadata(const aclTensor *tensor, Fault20TensorMetadata *metadata) try {
    dcat_metadata::CaptureScope capture_scope;
    if (tensor == nullptr || metadata == nullptr) {
        return false;
    }
    aclDataType dtype = ACL_DT_UNDEFINED;
    aclFormat format = ACL_FORMAT_UNDEFINED;
    int64_t view_offset = 0;
    dcat_metadata::OwnedArray view_dims;
    dcat_metadata::OwnedArray storage_dims;
    dcat_metadata::OwnedArray strides;
    uint64_t view_count = 0;
    uint64_t storage_count = 0;
    uint64_t stride_count = 0;
    void *raw_addr = nullptr;
    if (aclGetDataType(tensor, &dtype) != ACL_SUCCESS ||
        aclGetFormat(tensor, &format) != ACL_SUCCESS ||
        aclGetViewOffset(tensor, &view_offset) != ACL_SUCCESS ||
        view_dims.read(aclGetViewShape, tensor, &view_count) != ACL_SUCCESS ||
        storage_dims.read(aclGetStorageShape, tensor, &storage_count) != ACL_SUCCESS ||
        strides.read(aclGetViewStrides, tensor, &stride_count) != ACL_SUCCESS ||
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
    if (element_size == 0) {
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
} catch (...) {
    dcat_metadata::capture_exception();
    return false;
}

bool capture_optional_tensor_metadata(const aclTensor *tensor,
                                      Fault20TensorMetadata *metadata) {
    if (metadata == nullptr) {
        return false;
    }
    *metadata = Fault20TensorMetadata{};
    return tensor == nullptr || capture_tensor_metadata(tensor, metadata);
}

void copy_output_metadata_to_legacy(const Fault20TensorMetadata &out,
                                    Fault20X1Metadata *legacy) {
    if (legacy == nullptr || out.present == 0) {
        return;
    }
    legacy->output_dtype = out.dtype;
    legacy->output_format = out.format;
    legacy->output_view_rank = out.view_rank;
    legacy->output_storage_rank = out.storage_rank;
    legacy->output_stride_rank = out.stride_rank;
    legacy->output_contiguous = out.contiguous;
    for (size_t i = 0; i < FAULT20_MAX_DIMS; ++i) {
        legacy->output_view_shape[i] = out.view_shape[i];
        legacy->output_storage_shape[i] = out.storage_shape[i];
        legacy->output_strides[i] = out.strides[i];
    }
    legacy->output_view_offset = out.view_offset;
    legacy->output_numel = out.numel;
    legacy->output_storage_elements = out.storage_elements;
    legacy->output_element_size = out.element_size;
    legacy->output_view_bytes = out.view_bytes;
    legacy->output_storage_bytes = out.storage_bytes;
    legacy->output_tensor_ptr = out.tensor_ptr;
    legacy->output_storage_ptr = out.storage_ptr;
    legacy->output_raw_addr = out.raw_addr;
    legacy->output_storage_addr = out.storage_addr;
    legacy->output_data_addr = out.data_addr;
}

bool capture_qmm_metadata(
    const aclTensor *x1, const aclTensor *x2, const aclTensor *x1_scale,
    const aclTensor *x2_scale, const aclTensor *y_scale, const aclTensor *x1_offset,
    const aclTensor *x2_offset, const aclTensor *y_offset, const aclTensor *bias,
    bool transpose_x1, bool transpose_x2, int64_t group_size, aclTensor *out,
    Fault20QmmMetadata *metadata) {
    if (x1 == nullptr || x2 == nullptr || out == nullptr || metadata == nullptr) {
        return false;
    }
    Fault20QmmMetadata value{};
    if (!capture_x1_metadata(x1, out, transpose_x1, transpose_x2, group_size,
                             &value.x1_legacy) ||
        !capture_tensor_metadata(x1, &value.x1) ||
        !capture_tensor_metadata(x2, &value.x2) ||
        !capture_optional_tensor_metadata(x1_scale, &value.x1_scale) ||
        !capture_optional_tensor_metadata(x2_scale, &value.x2_scale) ||
        !capture_optional_tensor_metadata(y_scale, &value.y_scale) ||
        !capture_optional_tensor_metadata(x1_offset, &value.x1_offset) ||
        !capture_optional_tensor_metadata(x2_offset, &value.x2_offset) ||
        !capture_optional_tensor_metadata(y_offset, &value.y_offset) ||
        !capture_optional_tensor_metadata(bias, &value.bias) ||
        !capture_tensor_metadata(out, &value.out)) {
        return false;
    }
    copy_output_metadata_to_legacy(value.out, &value.x1_legacy);
    value.group_size = group_size;
    value.transpose_x1 = transpose_x1 ? 1U : 0U;
    value.transpose_x2 = transpose_x2 ? 1U : 0U;
    const Fault20TensorMetadata *tensors[] = {
        &value.x1,       &value.x2,        &value.x1_scale, &value.x2_scale,
        &value.y_scale,  &value.x1_offset, &value.x2_offset,
        &value.y_offset, &value.bias,      &value.out,
    };
    for (size_t i = 0; i < sizeof(tensors) / sizeof(tensors[0]); ++i) {
        const Fault20TensorMetadata &current = *tensors[i];
        if (current.present == 0) {
            continue;
        }
        ++value.present_count;
        value.any_empty_storage |=
            current.storage_bytes == 0 || current.storage_addr == 0 ? 1U : 0U;
        for (size_t j = i + 1; j < sizeof(tensors) / sizeof(tensors[0]); ++j) {
            const Fault20TensorMetadata &other = *tensors[j];
            if (other.present != 0 && current.storage_ptr != 0 &&
                current.storage_ptr == other.storage_ptr) {
                value.any_shared_storage = 1;
            }
        }
    }
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
    if (count && dims == nullptr) return capacity;
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

void log_quant_parameters(uint64_t invocation, aclnnStatus status, aclOpExecutor *executor,
                          bool sqrt_mode, const char *round_mode, int32_t dst_type, int32_t axis) {
    char line[1024];
    const char *mode = round_mode == nullptr ? "" : round_mode;
    const int written = std::snprintf(
        line, sizeof(line),
        "{\"event\":\"quant_parameters\",\"api\":\"aclnnAscendQuantV3\","
        "\"pid\":%ld,\"tid\":%ld,\"ts_ns\":%" PRIu64
        ",\"sequence\":%" PRIu64 ",\"invocation\":%" PRIu64
        ",\"status\":%d,\"executor\":\"%p\",\"sqrt_mode\":%s,"
        "\"round_mode\":\"%s\",\"dst_type\":%d,\"axis\":%d}",
        static_cast<long>(getpid()), static_cast<long>(syscall(SYS_gettid)), monotonic_ns(),
        g_sequence.fetch_add(1, std::memory_order_relaxed) + 1, invocation,
        static_cast<int>(status), executor, sqrt_mode ? "true" : "false", mode, dst_type, axis);
    if (written > 0 && static_cast<size_t>(written) < sizeof(line)) {
        write_line(line, static_cast<size_t>(written));
    }
}

bool snapshot_enabled() {
    const char *value = std::getenv("FAULT15_CAPTURE_SCALE_VALUES");
    return value != nullptr && std::strcmp(value, "1") == 0;
}

float bf16_to_float(uint16_t value) {
    const uint32_t bits = static_cast<uint32_t>(value) << 16U;
    float result = 0.0F;
    std::memcpy(&result, &bits, sizeof(result));
    return result;
}

void log_tensor_value_summary(const char *role, uint64_t invocation, aclOpExecutor *executor,
                              const aclTensor *tensor) {
    if (!snapshot_enabled() || tensor == nullptr) {
        return;
    }

    aclDataType dtype = ACL_DT_UNDEFINED;
    int64_t view_offset = 0;
    void *raw_addr = nullptr;
    if (aclGetDataType(tensor, &dtype) != ACL_SUCCESS ||
        aclGetViewOffset(tensor, &view_offset) != ACL_SUCCESS || view_offset != 0 ||
        aclGetRawTensorAddr(tensor, &raw_addr) != ACL_SUCCESS || raw_addr == nullptr) {
        return;
    }
    const int64_t numel = tensor->Numel();
    const size_t element_size = aclDataTypeSize(dtype);
    const uint64_t bytes = checked_bytes(numel, element_size);
    constexpr uint64_t kMaxSnapshotBytes = 1024U * 1024U;
    if (numel <= 0 || element_size == 0 || bytes == 0 || bytes > kMaxSnapshotBytes) {
        return;
    }

    void *host = std::malloc(static_cast<size_t>(bytes));
    if (host == nullptr) {
        return;
    }
    const aclError copy_status = aclrtMemcpy(host, bytes, raw_addr, bytes, ACL_MEMCPY_DEVICE_TO_HOST);

    uint64_t hash = 1469598103934665603ULL;
    const auto *raw = static_cast<const uint8_t *>(host);
    if (copy_status == ACL_SUCCESS) {
        for (uint64_t i = 0; i < bytes; ++i) {
            hash ^= raw[i];
            hash *= 1099511628211ULL;
        }
    }

    const int dtype_value = static_cast<int>(dtype);
    const bool numeric_supported = dtype_value == 0 || dtype_value == 27;
    double minimum = std::numeric_limits<double>::infinity();
    double maximum = -std::numeric_limits<double>::infinity();
    double total = 0.0;
    uint64_t finite_count = 0;
    uint64_t nonfinite_count = 0;
    if (copy_status == ACL_SUCCESS && numeric_supported) {
        for (int64_t i = 0; i < numel; ++i) {
            float value = 0.0F;
            if (dtype_value == 27) {
                uint16_t bits = 0;
                std::memcpy(&bits, raw + static_cast<size_t>(i) * sizeof(bits), sizeof(bits));
                value = bf16_to_float(bits);
            } else {
                std::memcpy(&value, raw + static_cast<size_t>(i) * sizeof(value), sizeof(value));
            }
            if (std::isfinite(value)) {
                minimum = std::fmin(minimum, static_cast<double>(value));
                maximum = std::fmax(maximum, static_cast<double>(value));
                total += static_cast<double>(value);
                ++finite_count;
            } else {
                ++nonfinite_count;
            }
        }
    }

    char line[2048];
    int written = std::snprintf(
        line, sizeof(line),
        "{\"event\":\"tensor_value_summary\",\"api\":\"aclnnAscendQuantV3\","
        "\"role\":\"%s\",\"pid\":%ld,\"tid\":%ld,\"ts_ns\":%" PRIu64
        ",\"sequence\":%" PRIu64 ",\"invocation\":%" PRIu64
        ",\"executor\":\"%p\",\"dtype\":%d,\"numel\":%" PRId64
        ",\"bytes\":%" PRIu64 ",\"copy_status\":%d,\"fnv1a64\":\"%016" PRIx64
        "\",\"numeric_supported\":%s,\"finite_count\":%" PRIu64
        ",\"nonfinite_count\":%" PRIu64,
        role, static_cast<long>(getpid()), static_cast<long>(syscall(SYS_gettid)), monotonic_ns(),
        g_sequence.fetch_add(1, std::memory_order_relaxed) + 1, invocation, executor, dtype_value,
        numel, bytes, static_cast<int>(copy_status), hash,
        numeric_supported ? "true" : "false", finite_count, nonfinite_count);
    if (written > 0 && static_cast<size_t>(written) < sizeof(line)) {
        size_t used = static_cast<size_t>(written);
        if (copy_status == ACL_SUCCESS && numeric_supported && finite_count > 0) {
            written = std::snprintf(line + used, sizeof(line) - used,
                                    ",\"min\":%.9g,\"max\":%.9g,\"mean\":%.9g}",
                                    minimum, maximum, total / static_cast<double>(finite_count));
        } else {
            written = std::snprintf(line + used, sizeof(line) - used,
                                    ",\"min\":null,\"max\":null,\"mean\":null}");
        }
        if (written > 0 && static_cast<size_t>(written) < sizeof(line) - used) {
            used += static_cast<size_t>(written);
            write_line(line, used);
        }
    }
    std::free(host);
}

void log_tensor(const char *api, const char *role, uint64_t invocation, aclOpExecutor *executor,
                const aclTensor *tensor) try {
    dcat_metadata::CaptureScope capture_scope;
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
    dcat_metadata::OwnedArray view_dims;
    dcat_metadata::OwnedArray storage_dims;
    dcat_metadata::OwnedArray strides;
    uint64_t view_count = 0;
    uint64_t storage_count = 0;
    uint64_t stride_count = 0;
    void *raw_addr = nullptr;
    if (aclGetDataType(tensor, &dtype) != ACL_SUCCESS ||
        aclGetFormat(tensor, &format) != ACL_SUCCESS ||
        aclGetViewOffset(tensor, &view_offset) != ACL_SUCCESS ||
        view_dims.read(aclGetViewShape, tensor, &view_count) != ACL_SUCCESS ||
        storage_dims.read(aclGetStorageShape, tensor, &storage_count) != ACL_SUCCESS ||
        strides.read(aclGetViewStrides, tensor, &stride_count) != ACL_SUCCESS ||
        aclGetRawTensorAddr(tensor, &raw_addr) != ACL_SUCCESS) return;

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
} catch (...) {
    dcat_metadata::capture_exception();
}

void log_matmul_summary(const char *event, uint64_t invocation,
                        aclOpExecutor *executor,
                        const Fault20MatmulMetadata &metadata) {
    char line[2048];
    const int written = std::snprintf(
        line, sizeof(line),
        "{\"event\":\"%s\",\"pid\":%ld,\"tid\":%ld,"
        "\"ts_ns\":%" PRIu64 ",\"sequence\":%" PRIu64
        ",\"invocation\":%" PRIu64 ",\"executor\":\"%p\","
        "\"cube_math_type\":%d,\"all_present\":%s,\"any_empty_storage\":%s,"
        "\"any_shared_storage\":%s,\"self_storage\":\"0x%" PRIx64
        "\",\"mat2_storage\":\"0x%" PRIx64 "\",\"out_storage\":\"0x%" PRIx64
        "\",\"self_storage_addr\":\"0x%" PRIx64
        "\",\"mat2_storage_addr\":\"0x%" PRIx64
        "\",\"out_storage_addr\":\"0x%" PRIx64 "\"}",
        event, static_cast<long>(getpid()), static_cast<long>(syscall(SYS_gettid)), monotonic_ns(),
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

extern "C" int dcat20_metadata_arrays_v1(DcatMetadataArrayStateV1* output) {
    return dcat_metadata::snapshot(output);
}

extern "C" uint32_t fault20_metadata_proxy_abi_v17() {
    return FAULT20_METADATA_ABI_VERSION;
}

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
                                       (g_fault20_gws_commit_v17 != nullptr ||
                                        g_fault20_gws_commit != nullptr) &&
                                       g_fault20_gws_abort != nullptr;
#ifdef DEMONCAT_QMM_AUTO
    CurrentGwsScope current(FAULT20_METADATA_PAYLOAD_QMM);
#endif
    if (registry_hooks_ready) {
        g_fault20_gws_begin();
    }
    const aclnnStatus status = g_real_qmm_getws(
        x1, x2, x1_scale, x2_scale, y_scale, x1_offset, x2_offset, y_offset, bias,
        transpose_x1, transpose_x2, group_size, out, workspace_size, executor);
    if (registry_hooks_ready) {
        Fault20QmmMetadata metadata{};
        if (status == ACL_SUCCESS &&
            capture_qmm_metadata(x1, x2, x1_scale, x2_scale, y_scale, x1_offset,
                                 x2_offset, y_offset, bias, transpose_x1,
                                 transpose_x2, group_size, out, &metadata)) {
#ifdef DEMONCAT_QMM_AUTO
            const auto descriptor = make_metadata_descriptor(FAULT20_METADATA_PAYLOAD_QMM, sizeof(metadata), &metadata);
            current.commit(descriptor, executor ? *executor : nullptr);
#else
            (void)commit_qmm_metadata(metadata);
#endif
        } else {
            g_fault20_gws_abort();
        }
    }
    aclOpExecutor *resolved_executor = executor == nullptr ? nullptr : *executor;
    notify_fault26_writer_gws(
        "aclnnQuantMatmulWeightNz", "out", out,
        workspace_size == nullptr ? 0 : *workspace_size, resolved_executor,
        status);
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
    if (g_fault26_trace_begin != nullptr) {
        g_fault26_trace_begin(FAULT26_EXECUTE_QMM,
                              "aclnnQuantMatmulWeightNz", workspace,
                              workspace_size, executor, stream);
    }
#ifdef DEMONCAT_QMM_AUTO
    CurrentExecuteScope current(FAULT20_METADATA_PAYLOAD_QMM, executor);
#endif
    const aclnnStatus status = g_real_qmm_execute(workspace, workspace_size, executor, stream);
    if (g_fault26_trace_end != nullptr) {
        g_fault26_trace_end(FAULT26_EXECUTE_QMM,
                            "aclnnQuantMatmulWeightNz", status);
    }
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
                                      (g_fault20_matmul_gws_commit_v17 != nullptr ||
                                       g_fault20_matmul_gws_commit != nullptr) &&
                                      g_fault20_matmul_gws_abort != nullptr;
#ifdef DEMONCAT_QMM_AUTO
    CurrentGwsScope current(FAULT20_METADATA_PAYLOAD_MATMUL_WEIGHT_NZ);
#endif
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
#ifdef DEMONCAT_QMM_AUTO
            const auto descriptor = make_metadata_descriptor(FAULT20_METADATA_PAYLOAD_MATMUL_WEIGHT_NZ, sizeof(metadata), &metadata);
            current.commit(descriptor, executor ? *executor : nullptr);
#else
            (void)commit_matmul_metadata(metadata, false);
#endif
        } else {
            g_fault20_matmul_gws_abort();
        }
    }
    aclOpExecutor *resolved_executor = executor == nullptr ? nullptr : *executor;
    notify_fault26_writer_gws(
        "aclnnMatmulWeightNz", "out", out,
        workspace_size == nullptr ? 0 : *workspace_size, resolved_executor,
        status);
    log_call("matmulweightnz_getws", invocation, status, resolved_executor, nullptr,
             workspace_size == nullptr ? 0 : *workspace_size, nullptr);
    log_tensor("aclnnMatmulWeightNz", "self", invocation, resolved_executor, self);
    log_tensor("aclnnMatmulWeightNz", "mat2", invocation, resolved_executor, mat2);
    log_tensor("aclnnMatmulWeightNz", "out", invocation, resolved_executor, out);
    if (metadata_ok) {
        log_matmul_summary("matmulweightnz_gws_summary", invocation,
                           resolved_executor, metadata);
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
    if (g_fault26_trace_begin != nullptr) {
        g_fault26_trace_begin(FAULT26_EXECUTE_MATMUL,
                              "aclnnMatmulWeightNz", workspace,
                              workspace_size, executor, stream);
    }
#ifdef DEMONCAT_QMM_AUTO
    CurrentExecuteScope current(FAULT20_METADATA_PAYLOAD_MATMUL_WEIGHT_NZ, executor);
#endif
    const aclnnStatus status =
        g_real_matmul_execute(workspace, workspace_size, executor, stream);
    if (g_fault26_trace_end != nullptr) {
        g_fault26_trace_end(FAULT26_EXECUTE_MATMUL,
                            "aclnnMatmulWeightNz", status);
    }
    log_call("matmulweightnz_execute_exit", invocation, status, executor, workspace,
             workspace_size, stream);
    return status;
}

extern "C" aclnnStatus aclnnMatmulGetWorkspaceSize(
    const aclTensor *self, const aclTensor *mat2, aclTensor *out,
    int8_t cube_math_type, uint64_t *workspace_size,
    aclOpExecutor **executor) {
    pthread_once(&g_resolve_once, resolve_real);
    const uint64_t invocation =
        g_sequence.fetch_add(1, std::memory_order_relaxed) + 1;
    if (g_real_float_matmul_getws == nullptr) {
        log_call("matmul_getws_resolve_failed", invocation, 1, nullptr,
                 nullptr, 0, nullptr);
        return 1;
    }
    const bool registry_hooks_ready =
        g_fault20_float_matmul_gws_begin != nullptr &&
        (g_fault20_float_matmul_gws_commit_v17 != nullptr ||
         g_fault20_float_matmul_gws_commit != nullptr) &&
        g_fault20_float_matmul_gws_abort != nullptr;
    if (registry_hooks_ready) {
        g_fault20_float_matmul_gws_begin();
    }
    const aclnnStatus status = g_real_float_matmul_getws(
        self, mat2, out, cube_math_type, workspace_size, executor);
    Fault20MatmulMetadata metadata{};
    const bool metadata_ok =
        status == ACL_SUCCESS &&
        capture_tensor_metadata(self, &metadata.self) &&
        capture_tensor_metadata(mat2, &metadata.mat2) &&
        capture_tensor_metadata(out, &metadata.out);
    if (metadata_ok) {
        metadata.cube_math_type = static_cast<int32_t>(cube_math_type);
        metadata.all_present = 1;
        metadata.any_empty_storage =
            metadata.self.storage_ptr == 0 || metadata.mat2.storage_ptr == 0 ||
                    metadata.out.storage_ptr == 0 ||
                    metadata.self.storage_addr == 0 ||
                    metadata.mat2.storage_addr == 0 ||
                    metadata.out.storage_addr == 0
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
            (void)commit_matmul_metadata(metadata, true);
        } else {
            g_fault20_float_matmul_gws_abort();
        }
    }
    aclOpExecutor *resolved_executor = executor == nullptr ? nullptr : *executor;
    notify_fault26_writer_gws(
        "aclnnMatmul", "out", out,
        workspace_size == nullptr ? 0 : *workspace_size, resolved_executor,
        status);
    log_call("matmul_getws", invocation, status, resolved_executor, nullptr,
             workspace_size == nullptr ? 0 : *workspace_size, nullptr);
    log_tensor("aclnnMatmul", "self", invocation, resolved_executor, self);
    log_tensor("aclnnMatmul", "mat2", invocation, resolved_executor, mat2);
    log_tensor("aclnnMatmul", "out", invocation, resolved_executor, out);
    if (metadata_ok) {
        log_matmul_summary("matmul_gws_summary", invocation,
                           resolved_executor, metadata);
    }
    return status;
}

extern "C" aclnnStatus aclnnMatmul(void *workspace, uint64_t workspace_size,
                                     aclOpExecutor *executor,
                                     aclrtStream stream) {
    pthread_once(&g_resolve_once, resolve_real);
    const uint64_t invocation =
        g_sequence.fetch_add(1, std::memory_order_relaxed) + 1;
    log_call("matmul_execute_enter", invocation, 0, executor, workspace,
             workspace_size, stream);
    if (g_real_float_matmul_execute == nullptr) {
        log_call("matmul_execute_resolve_failed", invocation, 1, executor,
                 workspace, workspace_size, stream);
        return 1;
    }
    if (g_fault26_trace_begin != nullptr) {
        g_fault26_trace_begin(FAULT26_EXECUTE_MATMUL, "aclnnMatmul",
                              workspace, workspace_size, executor, stream);
    }
    const aclnnStatus status = g_real_float_matmul_execute(
        workspace, workspace_size, executor, stream);
    if (g_fault26_trace_end != nullptr) {
        g_fault26_trace_end(FAULT26_EXECUTE_MATMUL, "aclnnMatmul", status);
    }
    log_call("matmul_execute_exit", invocation, status, executor, workspace,
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
    notify_fault26_writer_gws(
        "aclnnAscendQuantV3", "y", y,
        workspace_size == nullptr ? 0 : *workspace_size, resolved_executor,
        status);
    log_call("quant_getws", invocation, status, resolved_executor, nullptr,
             workspace_size == nullptr ? 0 : *workspace_size, nullptr);
    log_quant_parameters(invocation, status, resolved_executor, sqrt_mode, round_mode, dst_type,
                         axis);
    log_tensor("aclnnAscendQuantV3", "x", invocation, resolved_executor, x);
    log_tensor("aclnnAscendQuantV3", "scale", invocation, resolved_executor, scale);
    log_tensor("aclnnAscendQuantV3", "offset", invocation, resolved_executor, offset);
    log_tensor("aclnnAscendQuantV3", "y", invocation, resolved_executor, y);
    log_tensor_value_summary("scale", invocation, resolved_executor, scale);
    log_tensor_value_summary("offset", invocation, resolved_executor, offset);
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
    if (g_fault26_trace_begin != nullptr) {
        g_fault26_trace_begin(FAULT26_EXECUTE_ASCEND_QUANT_V3,
                              "aclnnAscendQuantV3", workspace,
                              workspace_size, executor, stream);
    }
    const aclnnStatus status = g_real_quant_execute(workspace, workspace_size, executor, stream);
    if (g_fault26_trace_end != nullptr) {
        g_fault26_trace_end(FAULT26_EXECUTE_ASCEND_QUANT_V3,
                            "aclnnAscendQuantV3", status);
    }
    log_call("quant_execute_exit", invocation, status, executor, workspace, workspace_size,
             stream);
    return status;
}

extern "C" aclnnStatus aclnnInplaceCopyGetWorkspaceSize(
    aclTensor *self_ref, const aclTensor *src, uint64_t *workspace_size,
    aclOpExecutor **executor) {
    pthread_once(&g_resolve_once, resolve_real);
    const uint64_t invocation =
        g_sequence.fetch_add(1, std::memory_order_relaxed) + 1;
    if (g_real_inplace_copy_getws == nullptr) {
        log_call("inplace_copy_getws_resolve_failed", invocation, 1, nullptr,
                 nullptr, 0, nullptr);
        return 1;
    }
    const aclnnStatus status = g_real_inplace_copy_getws(
        self_ref, src, workspace_size, executor);
    aclOpExecutor *resolved_executor = executor == nullptr ? nullptr : *executor;
    notify_fault26_writer_gws(
        "aclnnInplaceCopy", "selfRef", self_ref,
        workspace_size == nullptr ? 0 : *workspace_size, resolved_executor,
        status);
    log_call("inplace_copy_getws", invocation, status, resolved_executor,
             nullptr, workspace_size == nullptr ? 0 : *workspace_size,
             nullptr);
    log_tensor("aclnnInplaceCopy", "selfRef", invocation,
               resolved_executor, self_ref);
    log_tensor("aclnnInplaceCopy", "src", invocation, resolved_executor,
               src);
    return status;
}

extern "C" aclnnStatus aclnnInplaceCopy(
    void *workspace, uint64_t workspace_size, aclOpExecutor *executor,
    aclrtStream stream) {
    pthread_once(&g_resolve_once, resolve_real);
    const uint64_t invocation =
        g_sequence.fetch_add(1, std::memory_order_relaxed) + 1;
    log_call("inplace_copy_execute_enter", invocation, 0, executor,
             workspace, workspace_size, stream);
    if (g_real_inplace_copy_execute == nullptr) {
        log_call("inplace_copy_execute_resolve_failed", invocation, 1,
                 executor, workspace, workspace_size, stream);
        return 1;
    }
    if (g_fault26_trace_begin != nullptr) {
        g_fault26_trace_begin(FAULT26_EXECUTE_INPLACE_COPY,
                              "aclnnInplaceCopy", workspace,
                              workspace_size, executor, stream);
    }
    const aclnnStatus status = g_real_inplace_copy_execute(
        workspace, workspace_size, executor, stream);
    if (g_fault26_trace_end != nullptr) {
        g_fault26_trace_end(FAULT26_EXECUTE_INPLACE_COPY,
                            "aclnnInplaceCopy", status);
    }
    log_call("inplace_copy_execute_exit", invocation, status, executor,
             workspace, workspace_size, stream);
    return status;
}
