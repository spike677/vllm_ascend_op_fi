#include "fault26_trace_shared.h"
#include "fault26_qmm_underflow_api.h"
#include "../l4_buffer_substitute_allhit_v1/fault20_buffer_api.h"

#include <acl/acl.h>
#include <aclnn/acl_meta.h>
#include <aclnn/opdev/common_types.h>
#include <aclnn/opdev/op_cache.h>

#include <algorithm>
#include <atomic>
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <dlfcn.h>
#include <fcntl.h>
#include <limits>
#include <mutex>
#include <pthread.h>
#include <string>
#include <sys/syscall.h>
#include <sys/uio.h>
#include <time.h>
#include <unistd.h>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#if !defined(__aarch64__)
#error "Fault26 read-only tracer is version-locked to aarch64"
#endif

#if !defined(_GLIBCXX_USE_CXX11_ABI) || _GLIBCXX_USE_CXX11_ABI != 0
#error "Fault26 read-only tracer requires _GLIBCXX_USE_CXX11_ABI=0"
#endif

namespace {

constexpr const char *kDefaultLogPath =
    "/fault20/runtime/fault26_readonly_trace_v1/fault26_trace.jsonl";
constexpr const char *kDefaultFilter =
    "FusedInferAttention,InferAttention,ReduceSum,Add,RmsNorm,Residual";
constexpr const char *kRunSymbol =
    "_ZN2op8internal11OpExecCache3RunEPvS2_RKSt6vectorIS2_SaIS2_EE";
constexpr const char *kAddSymbol =
    "_ZN2op8internal14AddOpExecCacheEPNS0_11OpExecCacheE";
constexpr const char *kAddParamConstTensorSymbol =
    "_ZN2op8internal13AddParamToBufEPK9aclTensor";
constexpr const char *kAddParamTensorSymbol =
    "_ZN2op8internal13AddParamToBufEP9aclTensor";
constexpr const char *kGetLogApiInfoSymbol =
    "_ZN2op8internal13GetLogApiInfoEv";
constexpr size_t kMaxDims = 8;
constexpr size_t kMaxTensors = 64;
constexpr size_t kMaxExecuteDepth = 8;
constexpr size_t kQmmFormalCount = 10;
constexpr const char *kQmmPrefix = "aclnnQuantMatmulWeightNz_";

#ifndef FAULT26_EXPECTED_LIBRARY_PATH_FRAGMENT
#define FAULT26_EXPECTED_LIBRARY_PATH_FRAGMENT "/cann-8.5.1/"
#endif

using RunFn = aclnnStatus (*)(op::internal::OpExecCache *, void *, aclrtStream,
                              const std::vector<void *> &);
using AddFn = bool (*)(op::internal::OpExecCache *);
using AddParamConstTensorFn = void (*)(const aclTensor *);
using AddParamTensorFn = void (*)(aclTensor *);
using GetLogApiInfoFn = std::string (*)();
using GetDeviceFn = aclError (*)(int32_t *);
using SynchronizeStreamFn = aclError (*)(aclrtStream);
using MemcpyFn = aclError (*)(void *, size_t, const void *, size_t,
                              aclrtMemcpyKind);
using GetDataTypeFn = aclError (*)(const aclTensor *, aclDataType *);
using GetFormatFn = aclError (*)(const aclTensor *, aclFormat *);
using GetViewOffsetFn = aclError (*)(const aclTensor *, int64_t *);
using GetShapeFn = aclError (*)(const aclTensor *, int64_t **, uint64_t *);
using GetRawTensorAddrFn = aclError (*)(const aclTensor *, void **);
using DataTypeSizeFn = size_t (*)(aclDataType);
using QmmAfterRunFn = void (*)(const char *, void *, aclrtStream, aclnnStatus);

struct TensorMetadata {
    uint32_t present{0};
    uint32_t metadata_ok{0};
    int32_t dtype{static_cast<int32_t>(ACL_DT_UNDEFINED)};
    int32_t format{static_cast<int32_t>(ACL_FORMAT_UNDEFINED)};
    uint32_t view_rank{0};
    uint32_t storage_rank{0};
    uint32_t stride_rank{0};
    uint32_t contiguous{0};
    int64_t view_shape[kMaxDims]{};
    int64_t storage_shape[kMaxDims]{};
    int64_t strides[kMaxDims]{};
    int64_t view_offset{0};
    int64_t numel{0};
    int64_t storage_elements{0};
    uint64_t element_size{0};
    uint64_t view_bytes{0};
    uint64_t storage_bytes{0};
    uintptr_t tensor_ptr{0};
    uintptr_t storage_ptr{0};
    uintptr_t raw_addr{0};
    uintptr_t storage_addr{0};
    uintptr_t data_addr{0};
};

struct CaptureState {
    std::vector<TensorMetadata> tensors;
    bool overflow{false};
};

struct RegisteredOperation {
    std::vector<TensorMetadata> tensors;
    bool overflow{false};
    uint64_t run_count{0};
};

struct QmmCaptureState {
    bool active{false};
    uint32_t add_count{0};
    op::internal::OpExecCache *cache{nullptr};
    uint32_t tensor_param_count{0};
    uintptr_t tensor_params[16]{};
};

struct RegisteredQmmOperation {
    Fault20QmmMetadata metadata{};
    int32_t device{-1};
    uint64_t run_count{0};
    uint32_t tensor_param_count{0};
    uintptr_t tensor_params[16]{};
};

struct AddRmsAddressRecord {
    aclOpExecutor *executor{nullptr};
    TensorMetadata x2{};
    TensorMetadata x_out{};
    int32_t device{-1};
    uint64_t sequence{0};
};

struct ActiveAddRmsAddress {
    bool valid{false};
    AddRmsAddressRecord record{};
};

struct WriterTopologyRecord {
    std::string api;
    std::string role;
    aclOpExecutor *executor{nullptr};
    TensorMetadata writable{};
    uint64_t workspace_size{0};
    int32_t device{-1};
    uint64_t registration_sequence{0};
};

struct WriterResidualState {
    bool valid{false};
    uint64_t session{0};
    uint64_t producer_sequence{0};
    TensorMetadata metadata{};
    uint64_t p0_hash{0};
    uint64_t last_hash{0};
    uint64_t checkpoint_index{0};
    std::vector<unsigned char> p0_bytes;
    std::vector<unsigned char> last_bytes;
};

struct WriterQmmWindow {
    bool active{false};
    uint64_t session{0};
    int32_t device{-1};
    uint64_t producer_sequence{0};
    uint64_t consumer_sequence{0};
    uint64_t qmm_run_index{0};
    op::internal::OpExecCache *cache{nullptr};
    TensorMetadata residual{};
};

struct ExecuteFrame {
    uint32_t kind{FAULT26_EXECUTE_NONE};
    const char *api{nullptr};
    void *workspace{nullptr};
    uint64_t workspace_size{0};
    aclOpExecutor *executor{nullptr};
    aclrtStream stream{nullptr};
    uint64_t begin_ns{0};
};

struct ExecuteStack {
    ExecuteFrame frames[kMaxExecuteDepth]{};
    size_t depth{0};
    bool overflow{false};
};

pthread_once_t g_resolve_once = PTHREAD_ONCE_INIT;
RunFn g_real_run = nullptr;
AddFn g_real_add = nullptr;
AddParamConstTensorFn g_real_add_param_const_tensor = nullptr;
AddParamTensorFn g_real_add_param_tensor = nullptr;
GetLogApiInfoFn g_get_log_api_info = nullptr;
GetDeviceFn g_get_device = nullptr;
SynchronizeStreamFn g_synchronize_stream = nullptr;
MemcpyFn g_memcpy = nullptr;
GetDataTypeFn g_get_data_type = nullptr;
GetFormatFn g_get_format = nullptr;
GetViewOffsetFn g_get_view_offset = nullptr;
GetShapeFn g_get_view_shape = nullptr;
GetShapeFn g_get_storage_shape = nullptr;
GetShapeFn g_get_view_strides = nullptr;
GetRawTensorAddrFn g_get_raw_tensor_addr = nullptr;
DataTypeSizeFn g_data_type_size = nullptr;
QmmAfterRunFn g_qmm_after_run = nullptr;
Fault26QmmUnderflowPrepareFn g_qmm_underflow_prepare = nullptr;
Fault26QmmUnderflowCompleteFn g_qmm_underflow_complete = nullptr;
Fault26QmmUnderflowConsumerFn g_qmm_underflow_consumer = nullptr;
void *g_nnopbase_handle = nullptr;
void *g_ascendcl_handle = nullptr;
bool g_abi_ok = false;
bool g_trace_enabled = false;
bool g_trace_all_apis = false;
bool g_capture_tensors = true;
bool g_writer_topology_enabled = false;
bool g_writer_audit_allowed = false;
bool g_writer_audit_abi_ok = false;
std::atomic<uint64_t> g_sequence{0};
std::atomic<uint64_t> g_logged_events{0};
uint64_t g_max_events = 200000;
thread_local CaptureState g_capture;
thread_local QmmCaptureState g_qmm_capture;
thread_local ExecuteStack g_execute_stack;
thread_local bool g_in_hook = false;
thread_local ActiveAddRmsAddress g_active_add_rms;
std::atomic<uint64_t> g_add_rms_sequence{0};
std::mutex g_writer_audit_mutex;
uint64_t g_writer_session{0};
uint64_t g_writer_qmm_windows{0};
std::unordered_map<int32_t, WriterResidualState> g_writer_residuals;

std::unordered_map<op::internal::OpExecCache *, RegisteredOperation> &registry() {
    static auto *value =
        new std::unordered_map<op::internal::OpExecCache *, RegisteredOperation>();
    return *value;
}

std::mutex &registry_mutex() {
    static auto *value = new std::mutex();
    return *value;
}

std::unordered_map<op::internal::OpExecCache *, RegisteredQmmOperation> &
qmm_registry() {
    static auto *value = new std::unordered_map<op::internal::OpExecCache *,
                                                RegisteredQmmOperation>();
    return *value;
}

std::unordered_map<aclOpExecutor *, std::deque<AddRmsAddressRecord>> &
add_rms_queues() {
    static auto *value = new std::unordered_map<
        aclOpExecutor *, std::deque<AddRmsAddressRecord>>();
    return *value;
}

std::unordered_map<aclOpExecutor *, std::deque<WriterTopologyRecord>> &
writer_topology_queues() {
    static auto *value = new std::unordered_map<
        aclOpExecutor *, std::deque<WriterTopologyRecord>>();
    return *value;
}

std::unordered_map<int32_t, TensorMetadata> &current_residuals() {
    static auto *value =
        new std::unordered_map<int32_t, TensorMetadata>();
    return *value;
}

std::unordered_map<int32_t, AddRmsAddressRecord> &
current_residual_records() {
    static auto *value =
        new std::unordered_map<int32_t, AddRmsAddressRecord>();
    return *value;
}

uint64_t monotonic_ns() {
    timespec value{};
    if (clock_gettime(CLOCK_MONOTONIC, &value) != 0) {
        return 0;
    }
    return static_cast<uint64_t>(value.tv_sec) * 1000000000ULL +
           static_cast<uint64_t>(value.tv_nsec);
}

bool env_is_one(const char *name) {
    const char *value = std::getenv(name);
    return value != nullptr && std::strcmp(value, "1") == 0;
}

uint64_t parse_max_events() {
    const char *value = std::getenv("FAULT26_TRACE_MAX_EVENTS");
    if (value == nullptr || value[0] == '\0') {
        return 200000;
    }
    char *end = nullptr;
    const unsigned long long parsed = std::strtoull(value, &end, 10);
    if (end == value || *end != '\0' || parsed == 0 || parsed > 10000000ULL) {
        return 200000;
    }
    return static_cast<uint64_t>(parsed);
}

uint64_t parse_env_u64(const char *name, uint64_t fallback, uint64_t minimum,
                       uint64_t maximum) {
    const char *value = std::getenv(name);
    if (value == nullptr || value[0] == '\0') {
        return fallback;
    }
    char *end = nullptr;
    const unsigned long long parsed = std::strtoull(value, &end, 10);
    if (end == value || *end != '\0' || parsed < minimum ||
        parsed > maximum) {
        return fallback;
    }
    return static_cast<uint64_t>(parsed);
}

std::string json_escape(const std::string &value) {
    std::string escaped;
    escaped.reserve(value.size() + 16);
    for (const unsigned char character : value) {
        switch (character) {
            case '\\':
                escaped += "\\\\";
                break;
            case '"':
                escaped += "\\\"";
                break;
            case '\n':
                escaped += "\\n";
                break;
            case '\r':
                escaped += "\\r";
                break;
            case '\t':
                escaped += "\\t";
                break;
            default:
                if (character >= 0x20U) {
                    escaped.push_back(static_cast<char>(character));
                }
                break;
        }
    }
    return escaped;
}

bool reserve_log_event() {
    if (!g_trace_enabled) {
        return false;
    }
    const uint64_t index = g_logged_events.fetch_add(1, std::memory_order_relaxed);
    return index < g_max_events;
}

void write_line(const std::string &line) {
    if (!reserve_log_event()) {
        return;
    }
    const char *configured = std::getenv("FAULT26_TRACE_LOG_PATH");
    const char *path = configured == nullptr || configured[0] == '\0'
                           ? kDefaultLogPath
                           : configured;
    const int fd = open(path, O_WRONLY | O_CREAT | O_APPEND | O_CLOEXEC, 0644);
    if (fd < 0) {
        return;
    }
    char newline = '\n';
    iovec parts[2] = {
        {const_cast<char *>(line.data()), line.size()}, {&newline, 1}};
    const ssize_t written = writev(fd, parts, 2);
    (void)written;
    (void)close(fd);
}

std::string dims_json(const int64_t *dims, uint32_t count) {
    std::string value = "[";
    const uint32_t limit = std::min<uint32_t>(count, kMaxDims);
    for (uint32_t i = 0; i < limit; ++i) {
        if (i != 0) {
            value += ',';
        }
        value += std::to_string(dims[i]);
    }
    value += ']';
    return value;
}

bool checked_product(const int64_t *dims, uint32_t count, int64_t *product) {
    if (dims == nullptr || product == nullptr || count == 0 || count > kMaxDims) {
        return false;
    }
    uint64_t current = 1;
    for (uint32_t i = 0; i < count; ++i) {
        if (dims[i] < 0) {
            return false;
        }
        const uint64_t dim = static_cast<uint64_t>(dims[i]);
        if (dim != 0 && current >
                            static_cast<uint64_t>(
                                std::numeric_limits<int64_t>::max()) /
                                dim) {
            return false;
        }
        current *= dim;
    }
    *product = static_cast<int64_t>(current);
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

bool is_contiguous(const int64_t *shape, uint32_t shape_count,
                   const int64_t *strides, uint32_t stride_count) {
    if (shape == nullptr || strides == nullptr || shape_count == 0 ||
        shape_count != stride_count || shape_count > kMaxDims) {
        return false;
    }
    int64_t expected = 1;
    for (uint32_t reverse = 0; reverse < shape_count; ++reverse) {
        const uint32_t index = shape_count - reverse - 1;
        if (shape[index] < 0 || strides[index] != expected) {
            return false;
        }
        if (shape[index] != 0 &&
            expected > std::numeric_limits<int64_t>::max() / shape[index]) {
            return false;
        }
        expected *= shape[index];
    }
    return true;
}

TensorMetadata capture_tensor(const aclTensor *tensor) {
    TensorMetadata metadata{};
    metadata.tensor_ptr = reinterpret_cast<uintptr_t>(tensor);
    if (tensor == nullptr) {
        return metadata;
    }
    metadata.present = 1;
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
    if (g_get_data_type == nullptr || g_get_format == nullptr ||
        g_get_view_offset == nullptr || g_get_view_shape == nullptr ||
        g_get_storage_shape == nullptr || g_get_view_strides == nullptr ||
        g_get_raw_tensor_addr == nullptr || g_data_type_size == nullptr ||
        g_get_data_type(tensor, &dtype) != ACL_SUCCESS ||
        g_get_format(tensor, &format) != ACL_SUCCESS ||
        g_get_view_offset(tensor, &view_offset) != ACL_SUCCESS ||
        g_get_view_shape(tensor, &view_dims, &view_count) != ACL_SUCCESS ||
        g_get_storage_shape(tensor, &storage_dims, &storage_count) != ACL_SUCCESS ||
        g_get_view_strides(tensor, &strides, &stride_count) != ACL_SUCCESS ||
        g_get_raw_tensor_addr(tensor, &raw_addr) != ACL_SUCCESS ||
        view_count == 0 || view_count > kMaxDims || storage_count == 0 ||
        storage_count > kMaxDims || stride_count == 0 ||
        stride_count > kMaxDims) {
        return metadata;
    }
    metadata.dtype = static_cast<int32_t>(dtype);
    metadata.format = static_cast<int32_t>(format);
    metadata.view_rank = static_cast<uint32_t>(view_count);
    metadata.storage_rank = static_cast<uint32_t>(storage_count);
    metadata.stride_rank = static_cast<uint32_t>(stride_count);
    for (uint64_t i = 0; i < view_count; ++i) {
        metadata.view_shape[i] = view_dims[i];
    }
    for (uint64_t i = 0; i < storage_count; ++i) {
        metadata.storage_shape[i] = storage_dims[i];
    }
    for (uint64_t i = 0; i < stride_count; ++i) {
        metadata.strides[i] = strides[i];
    }
    if (!checked_product(metadata.view_shape, metadata.view_rank,
                         &metadata.numel) ||
        !checked_product(metadata.storage_shape, metadata.storage_rank,
                         &metadata.storage_elements)) {
        return metadata;
    }
    metadata.element_size = g_data_type_size(dtype);
    metadata.view_bytes = checked_bytes(metadata.numel, metadata.element_size);
    metadata.storage_bytes =
        checked_bytes(metadata.storage_elements, metadata.element_size);
    metadata.view_offset = view_offset;
    metadata.contiguous =
        is_contiguous(metadata.view_shape, metadata.view_rank, metadata.strides,
                      metadata.stride_rank)
            ? 1U
            : 0U;
    // Public metadata APIs expose the cold storage address but not a stable
    // storage object identity. It is sufficient for uniqueness checks here;
    // the live Run address remains the only address used for mapping.
    metadata.storage_ptr = reinterpret_cast<uintptr_t>(raw_addr);
    metadata.raw_addr = reinterpret_cast<uintptr_t>(raw_addr);
    metadata.storage_addr = reinterpret_cast<uintptr_t>(raw_addr);
    metadata.data_addr = reinterpret_cast<uintptr_t>(raw_addr);
    metadata.metadata_ok = metadata.element_size != 0 ? 1U : 0U;
    return metadata;
}

void log_typed_gws_summary(const char *api, double epsilon,
                           uint64_t workspace_size,
                           aclOpExecutor *executor, aclnnStatus status) {
    char line[2048];
    const std::string escaped_api = json_escape(api == nullptr ? "" : api);
    const int written = std::snprintf(
        line, sizeof(line),
        "{\"event\":\"fault26_typed_gws\",\"pid\":%ld,"
        "\"tid\":%ld,\"ts_ns\":%" PRIu64
        ",\"sequence\":%" PRIu64 ",\"api\":\"%s\","
        "\"epsilon\":%.17g,\"workspace_size\":%" PRIu64
        ",\"executor\":\"%p\",\"status\":%d}",
        static_cast<long>(getpid()),
        static_cast<long>(syscall(SYS_gettid)), monotonic_ns(),
        g_sequence.fetch_add(1, std::memory_order_relaxed) + 1,
        escaped_api.c_str(), epsilon, workspace_size, executor,
        static_cast<int>(status));
    if (written > 0 && static_cast<size_t>(written) < sizeof(line)) {
        write_line(std::string(line, static_cast<size_t>(written)));
    }
}

void log_typed_gws_tensor(const char *api, const char *role,
                          const aclTensor *tensor, uint64_t workspace_size,
                          aclOpExecutor *executor, aclnnStatus status) {
    const TensorMetadata metadata = capture_tensor(tensor);
    const std::string escaped_api = json_escape(api == nullptr ? "" : api);
    const std::string escaped_role = json_escape(role == nullptr ? "" : role);
    const std::string view_shape =
        dims_json(metadata.view_shape, metadata.view_rank);
    const std::string storage_shape =
        dims_json(metadata.storage_shape, metadata.storage_rank);
    const std::string strides =
        dims_json(metadata.strides, metadata.stride_rank);
    char line[6144];
    const int written = std::snprintf(
        line, sizeof(line),
        "{\"event\":\"fault26_typed_gws_tensor\",\"pid\":%ld,"
        "\"tid\":%ld,\"ts_ns\":%" PRIu64
        ",\"sequence\":%" PRIu64 ",\"api\":\"%s\","
        "\"role\":\"%s\",\"executor\":\"%p\","
        "\"workspace_size\":%" PRIu64 ",\"status\":%d,"
        "\"present\":%s,\"metadata_ok\":%s,\"dtype\":%d,"
        "\"format\":%d,\"view_shape\":%s,\"storage_shape\":%s,"
        "\"strides\":%s,\"view_offset\":%" PRId64
        ",\"element_size\":%" PRIu64 ",\"view_bytes\":%" PRIu64
        ",\"storage_bytes\":%" PRIu64 ",\"tensor\":\"0x%" PRIxPTR
        "\",\"storage\":\"0x%" PRIxPTR "\"}",
        static_cast<long>(getpid()),
        static_cast<long>(syscall(SYS_gettid)), monotonic_ns(),
        g_sequence.fetch_add(1, std::memory_order_relaxed) + 1,
        escaped_api.c_str(), escaped_role.c_str(), executor, workspace_size,
        static_cast<int>(status), metadata.present != 0 ? "true" : "false",
        metadata.metadata_ok != 0 ? "true" : "false", metadata.dtype,
        metadata.format, view_shape.c_str(), storage_shape.c_str(),
        strides.c_str(), metadata.view_offset, metadata.element_size,
        metadata.view_bytes, metadata.storage_bytes, metadata.tensor_ptr,
        metadata.storage_ptr);
    if (written > 0 && static_cast<size_t>(written) < sizeof(line)) {
        write_line(std::string(line, static_cast<size_t>(written)));
    }
}

void *resolve_from_nnopbase(const char *symbol) {
    if (g_nnopbase_handle == nullptr) {
        g_nnopbase_handle =
            dlopen("libnnopbase.so", RTLD_NOW | RTLD_LOCAL | RTLD_NOLOAD);
    }
    return g_nnopbase_handle == nullptr ? nullptr
                                        : dlsym(g_nnopbase_handle, symbol);
}

void *resolve_from_ascendcl(const char *symbol) {
    if (g_ascendcl_handle == nullptr) {
        constexpr const char *kCandidates[] = {
            "libascendcl.so",
            "/usr/local/Ascend/cann-8.5.1/lib64/libascendcl.so",
            "/usr/local/Ascend/cann-8.5.1/aarch64-linux/lib64/"
            "libascendcl.so",
        };
        for (const char *candidate : kCandidates) {
            g_ascendcl_handle =
                dlopen(candidate, RTLD_NOW | RTLD_LOCAL | RTLD_NOLOAD);
            if (g_ascendcl_handle != nullptr) {
                break;
            }
        }
    }
    return g_ascendcl_handle == nullptr ? nullptr
                                        : dlsym(g_ascendcl_handle, symbol);
}

void resolve_real() {
    g_real_run = reinterpret_cast<RunFn>(dlsym(RTLD_NEXT, kRunSymbol));
    g_real_add = reinterpret_cast<AddFn>(dlsym(RTLD_NEXT, kAddSymbol));
    g_real_add_param_const_tensor = reinterpret_cast<AddParamConstTensorFn>(
        dlsym(RTLD_NEXT, kAddParamConstTensorSymbol));
    g_real_add_param_tensor = reinterpret_cast<AddParamTensorFn>(
        dlsym(RTLD_NEXT, kAddParamTensorSymbol));
    g_get_log_api_info = reinterpret_cast<GetLogApiInfoFn>(
        dlsym(RTLD_NEXT, kGetLogApiInfoSymbol));
    if (g_real_run == nullptr) {
        g_real_run = reinterpret_cast<RunFn>(resolve_from_nnopbase(kRunSymbol));
    }
    if (g_real_add == nullptr) {
        g_real_add = reinterpret_cast<AddFn>(resolve_from_nnopbase(kAddSymbol));
    }
    if (g_real_add_param_const_tensor == nullptr) {
        g_real_add_param_const_tensor = reinterpret_cast<AddParamConstTensorFn>(
            resolve_from_nnopbase(kAddParamConstTensorSymbol));
    }
    if (g_real_add_param_tensor == nullptr) {
        g_real_add_param_tensor = reinterpret_cast<AddParamTensorFn>(
            resolve_from_nnopbase(kAddParamTensorSymbol));
    }
    if (g_get_log_api_info == nullptr) {
        g_get_log_api_info = reinterpret_cast<GetLogApiInfoFn>(
            resolve_from_nnopbase(kGetLogApiInfoSymbol));
    }
    g_get_device = reinterpret_cast<GetDeviceFn>(
        dlsym(RTLD_DEFAULT, "aclrtGetDevice"));
    g_synchronize_stream = reinterpret_cast<SynchronizeStreamFn>(
        dlsym(RTLD_DEFAULT, "aclrtSynchronizeStream"));
    g_memcpy =
        reinterpret_cast<MemcpyFn>(dlsym(RTLD_DEFAULT, "aclrtMemcpy"));
    g_get_data_type = reinterpret_cast<GetDataTypeFn>(
        dlsym(RTLD_DEFAULT, "aclGetDataType"));
    g_get_format = reinterpret_cast<GetFormatFn>(
        dlsym(RTLD_DEFAULT, "aclGetFormat"));
    g_get_view_offset = reinterpret_cast<GetViewOffsetFn>(
        dlsym(RTLD_DEFAULT, "aclGetViewOffset"));
    g_get_view_shape = reinterpret_cast<GetShapeFn>(
        dlsym(RTLD_DEFAULT, "aclGetViewShape"));
    g_get_storage_shape = reinterpret_cast<GetShapeFn>(
        dlsym(RTLD_DEFAULT, "aclGetStorageShape"));
    g_get_view_strides = reinterpret_cast<GetShapeFn>(
        dlsym(RTLD_DEFAULT, "aclGetViewStrides"));
    g_get_raw_tensor_addr = reinterpret_cast<GetRawTensorAddrFn>(
        dlsym(RTLD_DEFAULT, "aclGetRawTensorAddr"));
    g_data_type_size = reinterpret_cast<DataTypeSizeFn>(
        dlsym(RTLD_DEFAULT, "aclDataTypeSize"));
    g_qmm_after_run = reinterpret_cast<QmmAfterRunFn>(
        dlsym(RTLD_DEFAULT, "fault26_qmm_trigger_after_run"));
    g_qmm_underflow_prepare = reinterpret_cast<Fault26QmmUnderflowPrepareFn>(
        dlsym(RTLD_DEFAULT, "fault26_qmm_underflow_prepare"));
    g_qmm_underflow_complete = reinterpret_cast<Fault26QmmUnderflowCompleteFn>(
        dlsym(RTLD_DEFAULT, "fault26_qmm_underflow_complete"));
    g_qmm_underflow_consumer = reinterpret_cast<Fault26QmmUnderflowConsumerFn>(
        dlsym(RTLD_DEFAULT, "fault26_qmm_underflow_consumer_complete"));
    if (g_get_data_type == nullptr) {
        g_get_data_type = reinterpret_cast<GetDataTypeFn>(
            resolve_from_nnopbase("aclGetDataType"));
    }
    if (g_get_format == nullptr) {
        g_get_format = reinterpret_cast<GetFormatFn>(
            resolve_from_nnopbase("aclGetFormat"));
    }
    if (g_get_view_offset == nullptr) {
        g_get_view_offset = reinterpret_cast<GetViewOffsetFn>(
            resolve_from_nnopbase("aclGetViewOffset"));
    }
    if (g_get_view_shape == nullptr) {
        g_get_view_shape = reinterpret_cast<GetShapeFn>(
            resolve_from_nnopbase("aclGetViewShape"));
    }
    if (g_get_storage_shape == nullptr) {
        g_get_storage_shape = reinterpret_cast<GetShapeFn>(
            resolve_from_nnopbase("aclGetStorageShape"));
    }
    if (g_get_view_strides == nullptr) {
        g_get_view_strides = reinterpret_cast<GetShapeFn>(
            resolve_from_nnopbase("aclGetViewStrides"));
    }
    if (g_get_raw_tensor_addr == nullptr) {
        g_get_raw_tensor_addr = reinterpret_cast<GetRawTensorAddrFn>(
            resolve_from_nnopbase("aclGetRawTensorAddr"));
    }
    // torch_npu may load libascendcl with a local scope. In that case it is
    // present in the process but its public C symbols are not visible through
    // RTLD_DEFAULT. Use RTLD_NOLOAD so the tracer never introduces a new CANN
    // library or changes loader order.
    if (g_get_device == nullptr) {
        g_get_device = reinterpret_cast<GetDeviceFn>(
            resolve_from_ascendcl("aclrtGetDevice"));
    }
    if (g_synchronize_stream == nullptr) {
        g_synchronize_stream = reinterpret_cast<SynchronizeStreamFn>(
            resolve_from_ascendcl("aclrtSynchronizeStream"));
    }
    if (g_memcpy == nullptr) {
        g_memcpy = reinterpret_cast<MemcpyFn>(
            resolve_from_ascendcl("aclrtMemcpy"));
    }
    if (g_data_type_size == nullptr) {
        g_data_type_size = reinterpret_cast<DataTypeSizeFn>(
            resolve_from_ascendcl("aclDataTypeSize"));
    }

    Dl_info info{};
    const bool path_resolved =
        g_real_run != nullptr &&
        dladdr(reinterpret_cast<void *>(g_real_run), &info) != 0 &&
        info.dli_fname != nullptr;
    const bool path_ok =
        path_resolved &&
        std::strstr(info.dli_fname,
                    FAULT26_EXPECTED_LIBRARY_PATH_FRAGMENT) != nullptr;
    g_max_events = parse_max_events();
    g_trace_all_apis = env_is_one("FAULT26_TRACE_ALL_APIS");
    const char *capture_tensors = std::getenv("FAULT26_TRACE_CAPTURE_TENSORS");
    g_capture_tensors =
        capture_tensors == nullptr || std::strcmp(capture_tensors, "0") != 0;
    const bool inventory_abi_ok = path_ok && g_real_run != nullptr &&
                                  g_get_log_api_info != nullptr;
    const bool metadata_abi_ok =
        g_real_add != nullptr && g_real_add_param_const_tensor != nullptr &&
        g_real_add_param_tensor != nullptr && g_get_data_type != nullptr &&
        g_get_format != nullptr && g_get_view_offset != nullptr &&
        g_get_view_shape != nullptr && g_get_storage_shape != nullptr &&
        g_get_view_strides != nullptr && g_get_raw_tensor_addr != nullptr &&
        g_data_type_size != nullptr;
    // Phase 1 deliberately disables tensor capture and only inventories Run
    // API names. Do not require unused metadata symbols in that phase. Phase 2
    // remains fail-closed on the complete metadata ABI.
    g_abi_ok = inventory_abi_ok && (!g_capture_tensors || metadata_abi_ok);
    g_trace_enabled = env_is_one("VLLM_SERVER_DEV_MODE") &&
                      env_is_one("FAULT26_TRACE_ENABLED") && g_abi_ok;
    g_writer_topology_enabled =
        g_trace_enabled && g_capture_tensors &&
        env_is_one("FAULT26_WRITER_TOPOLOGY_ENABLED");
    g_writer_audit_abi_ok =
        g_synchronize_stream != nullptr && g_memcpy != nullptr;
    g_writer_audit_allowed = env_is_one("VLLM_SERVER_DEV_MODE") &&
                             env_is_one("FAULT26_WRITER_AUDIT_ALLOWED") &&
                             g_writer_audit_abi_ok;

    if (env_is_one("FAULT26_TRACE_ENABLED")) {
        char line[2048];
        const int written = std::snprintf(
            line, sizeof(line),
            "{\"event\":\"fault26_trace_resolution\",\"pid\":%ld,"
            "\"tid\":%ld,\"ts_ns\":%" PRIu64
            ",\"sequence\":%" PRIu64 ",\"abi_ok\":%s,"
            "\"developer_mode_ok\":%s,\"trace_enabled\":%s,"
            "\"trace_all_apis\":%s,\"capture_tensors\":%s,"
            "\"max_events\":%" PRIu64
            ",\"writer_topology_enabled\":%s"
            ",\"writer_audit_allowed\":%s,\"writer_audit_abi_ok\":%s,"
            "\"real_library\":\"%s\",\"expected_fragment\":\"%s\"}",
            static_cast<long>(getpid()),
            static_cast<long>(syscall(SYS_gettid)), monotonic_ns(),
            g_sequence.fetch_add(1, std::memory_order_relaxed) + 1,
            g_abi_ok ? "true" : "false",
            env_is_one("VLLM_SERVER_DEV_MODE") ? "true" : "false",
            g_trace_enabled ? "true" : "false",
            g_trace_all_apis ? "true" : "false",
            g_capture_tensors ? "true" : "false", g_max_events,
            g_writer_topology_enabled ? "true" : "false",
            g_writer_audit_allowed ? "true" : "false",
            g_writer_audit_abi_ok ? "true" : "false",
            path_resolved ? info.dli_fname : "",
            FAULT26_EXPECTED_LIBRARY_PATH_FRAGMENT);
        if (written > 0 && static_cast<size_t>(written) < sizeof(line)) {
            const bool enabled_before = g_trace_enabled;
            g_trace_enabled = true;
            write_line(std::string(line, static_cast<size_t>(written)));
            g_trace_enabled = enabled_before;
        }

        char symbols_line[2048];
        const int symbols_written = std::snprintf(
            symbols_line, sizeof(symbols_line),
            "{\"event\":\"fault26_trace_symbols\",\"pid\":%ld,"
            "\"tid\":%ld,\"ts_ns\":%" PRIu64
            ",\"sequence\":%" PRIu64 ",\"path_ok\":%s,"
            "\"run\":%s,\"add\":%s,\"add_param_const\":%s,"
            "\"add_param_mutable\":%s,\"get_log_api_info\":%s,"
            "\"get_data_type\":%s,\"get_format\":%s,"
            "\"get_view_offset\":%s,\"get_view_shape\":%s,"
            "\"get_storage_shape\":%s,\"get_view_strides\":%s,"
            "\"get_raw_tensor_addr\":%s,\"data_type_size\":%s,"
            "\"get_device_optional\":%s,\"synchronize_stream_optional\":%s,"
            "\"d2h_memcpy_optional\":%s}",
            static_cast<long>(getpid()),
            static_cast<long>(syscall(SYS_gettid)), monotonic_ns(),
            g_sequence.fetch_add(1, std::memory_order_relaxed) + 1,
            path_ok ? "true" : "false",
            g_real_run != nullptr ? "true" : "false",
            g_real_add != nullptr ? "true" : "false",
            g_real_add_param_const_tensor != nullptr ? "true" : "false",
            g_real_add_param_tensor != nullptr ? "true" : "false",
            g_get_log_api_info != nullptr ? "true" : "false",
            g_get_data_type != nullptr ? "true" : "false",
            g_get_format != nullptr ? "true" : "false",
            g_get_view_offset != nullptr ? "true" : "false",
            g_get_view_shape != nullptr ? "true" : "false",
            g_get_storage_shape != nullptr ? "true" : "false",
            g_get_view_strides != nullptr ? "true" : "false",
            g_get_raw_tensor_addr != nullptr ? "true" : "false",
            g_data_type_size != nullptr ? "true" : "false",
            g_get_device != nullptr ? "true" : "false",
            g_synchronize_stream != nullptr ? "true" : "false",
            g_memcpy != nullptr ? "true" : "false");
        if (symbols_written > 0 &&
            static_cast<size_t>(symbols_written) < sizeof(symbols_line)) {
            const bool enabled_before = g_trace_enabled;
            g_trace_enabled = true;
            write_line(std::string(symbols_line,
                                   static_cast<size_t>(symbols_written)));
            g_trace_enabled = enabled_before;
        }
    }
}

void notify_qmm_after_run(const std::string &api,
                          op::internal::OpExecCache *cache,
                          aclrtStream stream, aclnnStatus status) {
    if (g_qmm_after_run == nullptr ||
        api.rfind("aclnnQuantMatmulWeightNz_", 0) != 0) {
        return;
    }
    try {
        g_qmm_after_run(api.c_str(), cache, stream, status);
    } catch (...) {
        // A diagnostic callback must never change the real operator status.
    }
}

std::string current_api_info() {
    if (g_get_log_api_info == nullptr) {
        return "";
    }
    try {
        return g_get_log_api_info();
    } catch (...) {
        return "";
    }
}

bool api_matches_filter(const std::string &api) {
    if (g_trace_all_apis) {
        return true;
    }
    const char *configured = std::getenv("FAULT26_TRACE_FILTER");
    const std::string filter =
        configured == nullptr || configured[0] == '\0' ? kDefaultFilter
                                                       : configured;
    size_t begin = 0;
    while (begin <= filter.size()) {
        const size_t end = filter.find(',', begin);
        const std::string token =
            filter.substr(begin, end == std::string::npos ? std::string::npos
                                                          : end - begin);
        if (!token.empty() && api.find(token) != std::string::npos) {
            return true;
        }
        if (end == std::string::npos) {
            break;
        }
        begin = end + 1;
    }
    return false;
}

const ExecuteFrame *current_execute_frame() {
    if (g_execute_stack.depth == 0 ||
        g_execute_stack.depth > kMaxExecuteDepth) {
        return nullptr;
    }
    return &g_execute_stack.frames[g_execute_stack.depth - 1];
}

bool range_end(uintptr_t begin, uint64_t size, uintptr_t *end) {
    if (begin == 0 || size == 0 || end == nullptr ||
        size > std::numeric_limits<uintptr_t>::max() - begin) {
        return false;
    }
    *end = begin + static_cast<uintptr_t>(size);
    return true;
}

bool ranges_overlap(uintptr_t first, uint64_t first_size, uintptr_t second,
                    uint64_t second_size) {
    uintptr_t first_end = 0;
    uintptr_t second_end = 0;
    return range_end(first, first_size, &first_end) &&
           range_end(second, second_size, &second_end) && first < second_end &&
           second < first_end;
}

int64_t range_gap(uintptr_t first, uint64_t first_size, uintptr_t second,
                  uint64_t second_size) {
    uintptr_t first_end = 0;
    uintptr_t second_end = 0;
    if (!range_end(first, first_size, &first_end) ||
        !range_end(second, second_size, &second_end) ||
        ranges_overlap(first, first_size, second, second_size)) {
        return 0;
    }
    uint64_t gap = 0;
    bool negative = false;
    if (first_end <= second) {
        gap = static_cast<uint64_t>(second - first_end);
    } else {
        gap = static_cast<uint64_t>(first - second_end);
        negative = true;
    }
    const uint64_t max_value =
        static_cast<uint64_t>(std::numeric_limits<int64_t>::max());
    const int64_t bounded =
        static_cast<int64_t>(std::min<uint64_t>(gap, max_value));
    return negative ? -bounded : bounded;
}

bool qmm_api_matches(const std::string &api) {
    return api.compare(0, std::strlen(kQmmPrefix), kQmmPrefix) == 0;
}

bool same_tensor_storage(const TensorMetadata &first,
                         const TensorMetadata &second) {
    return first.present != 0 && second.present != 0 &&
           first.metadata_ok != 0 && second.metadata_ok != 0 &&
           first.raw_addr != 0 && first.raw_addr == second.raw_addr &&
           first.storage_bytes != 0 &&
           first.storage_bytes == second.storage_bytes &&
           first.dtype == second.dtype && first.format == second.format &&
           first.view_rank == second.view_rank &&
           std::equal(first.view_shape, first.view_shape + first.view_rank,
                       second.view_shape);
}

uint64_t fnv1a64(const std::vector<unsigned char> &bytes) {
    uint64_t value = 1469598103934665603ULL;
    for (const unsigned char byte : bytes) {
        value ^= byte;
        value *= 1099511628211ULL;
    }
    return value;
}

bool writer_audit_enabled(uint64_t *session) {
    if (!g_writer_audit_allowed || !g_writer_audit_abi_ok ||
        !env_is_one("FAULT26_WRITER_AUDIT_ENABLED")) {
        return false;
    }
    const uint64_t current = parse_env_u64(
        "FAULT26_WRITER_AUDIT_SESSION", 0, 1,
        std::numeric_limits<uint64_t>::max());
    if (current == 0) {
        return false;
    }
    if (session != nullptr) {
        *session = current;
    }
    return true;
}

void ensure_writer_session_locked(uint64_t session) {
    if (g_writer_session == session) {
        return;
    }
    g_writer_session = session;
    g_writer_qmm_windows = 0;
    g_writer_residuals.clear();
}

struct WriterReadResult {
    bool ok{false};
    const char *reason{"not_attempted"};
    int32_t sync_status{-1};
    int32_t copy_status{-1};
    uint64_t hash{0};
    std::vector<unsigned char> bytes;
};

WriterReadResult read_residual(const TensorMetadata &metadata,
                               aclrtStream stream) {
    WriterReadResult result{};
    const uint64_t max_bytes = parse_env_u64(
        "FAULT26_WRITER_AUDIT_MAX_BYTES", 4194304ULL, 1,
        1073741824ULL);
    if (metadata.metadata_ok == 0 || metadata.raw_addr == 0 ||
        metadata.storage_bytes == 0) {
        result.reason = "metadata_incomplete";
        return result;
    }
    if (metadata.contiguous == 0 || metadata.view_offset != 0 ||
        metadata.view_bytes != metadata.storage_bytes) {
        result.reason = "storage_layout_not_exact";
        return result;
    }
    if (metadata.storage_bytes > max_bytes) {
        result.reason = "max_bytes_exceeded";
        return result;
    }
    if (stream == nullptr || g_synchronize_stream == nullptr ||
        g_memcpy == nullptr) {
        result.reason = "read_abi_unavailable";
        return result;
    }
    result.sync_status = static_cast<int32_t>(g_synchronize_stream(stream));
    if (result.sync_status != static_cast<int32_t>(ACL_SUCCESS)) {
        result.reason = "stream_sync_failed";
        return result;
    }
    try {
        result.bytes.resize(static_cast<size_t>(metadata.storage_bytes));
    } catch (...) {
        result.reason = "host_allocation_failed";
        return result;
    }
    result.copy_status = static_cast<int32_t>(g_memcpy(
        result.bytes.data(), result.bytes.size(),
        reinterpret_cast<const void *>(metadata.raw_addr), result.bytes.size(),
        ACL_MEMCPY_DEVICE_TO_HOST));
    if (result.copy_status != static_cast<int32_t>(ACL_SUCCESS)) {
        result.bytes.clear();
        result.reason = "d2h_failed";
        return result;
    }
    result.hash = fnv1a64(result.bytes);
    result.ok = true;
    result.reason = "ok";
    return result;
}

struct WriterDiff {
    uint64_t count{0};
    int64_t first{-1};
    int64_t last{-1};
};

WriterDiff compare_bytes(const std::vector<unsigned char> &reference,
                         const std::vector<unsigned char> &current) {
    WriterDiff diff{};
    if (reference.size() != current.size()) {
        diff.count = std::max(reference.size(), current.size());
        diff.first = 0;
        diff.last = diff.count == 0 ? -1 : static_cast<int64_t>(diff.count - 1);
        return diff;
    }
    for (size_t index = 0; index < current.size(); ++index) {
        if (reference[index] == current[index]) {
            continue;
        }
        if (diff.first < 0) {
            diff.first = static_cast<int64_t>(index);
        }
        diff.last = static_cast<int64_t>(index);
        ++diff.count;
    }
    return diff;
}

void log_writer_checkpoint(
    uint64_t session, const char *checkpoint, const char *api,
    int32_t device, uint64_t producer_sequence, uint64_t consumer_sequence,
    op::internal::OpExecCache *qmm_cache, uint64_t qmm_run_index,
    const TensorMetadata &metadata, aclrtStream stream, aclnnStatus op_status,
    const WriterReadResult &read, uint64_t p0_hash, uint64_t previous_hash,
    const WriterDiff &from_p0, const WriterDiff &from_previous,
    uint64_t checkpoint_index) {
    uintptr_t residual_end = 0;
    const bool end_valid =
        range_end(metadata.raw_addr, metadata.storage_bytes, &residual_end);
    const std::string escaped_api = json_escape(api == nullptr ? "" : api);
    char line[6144];
    const int written = std::snprintf(
        line, sizeof(line),
        "{\"event\":\"fault26_residual_checkpoint\","
        "\"pid\":%ld,\"tid\":%ld,\"ts_ns\":%" PRIu64 ","
        "\"sequence\":%" PRIu64 ",\"session\":%" PRIu64 ","
        "\"checkpoint\":\"%s\",\"api\":\"%s\",\"device\":%d,"
        "\"producer_api\":\"aclnnAddRmsNormBias\","
        "\"producer_sequence\":%" PRIu64 ",\"consumer_sequence\":%" PRIu64 ","
        "\"qmm_cache\":\"%p\",\"qmm_run_index\":%" PRIu64 ","
        "\"residual_begin\":\"0x%" PRIxPTR "\","
        "\"residual_end\":\"0x%" PRIxPTR "\","
        "\"residual_bytes\":%" PRIu64 ",\"stream\":\"%p\","
        "\"op_status\":%d,\"read_ok\":%s,\"reason\":\"%s\","
        "\"sync_status\":%d,\"copy_status\":%d,"
        "\"hash\":\"0x%016" PRIx64 "\",\"p0_hash\":\"0x%016" PRIx64 "\","
        "\"previous_hash\":\"0x%016" PRIx64 "\","
        "\"equal_p0\":%s,\"equal_previous\":%s,"
        "\"changed_bytes_from_p0\":%" PRIu64 ","
        "\"first_diff_from_p0\":%" PRId64 ","
        "\"last_diff_from_p0\":%" PRId64 ","
        "\"changed_bytes_from_previous\":%" PRIu64 ","
        "\"first_diff_from_previous\":%" PRId64 ","
        "\"last_diff_from_previous\":%" PRId64 ","
        "\"checkpoint_index\":%" PRIu64 ","
        "\"device_memory_modified\":false}",
        static_cast<long>(getpid()),
        static_cast<long>(syscall(SYS_gettid)), monotonic_ns(),
        g_sequence.fetch_add(1, std::memory_order_relaxed) + 1, session,
        checkpoint, escaped_api.c_str(), device, producer_sequence,
        consumer_sequence, qmm_cache, qmm_run_index, metadata.raw_addr,
        end_valid ? residual_end : 0, metadata.storage_bytes, stream,
        static_cast<int>(op_status), read.ok ? "true" : "false", read.reason,
        read.sync_status, read.copy_status, read.hash, p0_hash, previous_hash,
        read.ok && read.hash == p0_hash ? "true" : "false",
        read.ok && read.hash == previous_hash ? "true" : "false",
        from_p0.count, from_p0.first, from_p0.last, from_previous.count,
        from_previous.first, from_previous.last, checkpoint_index);
    if (written > 0 && static_cast<size_t>(written) < sizeof(line)) {
        write_line(std::string(line, static_cast<size_t>(written)));
    }
}

bool writer_sample_p0(const AddRmsAddressRecord &record, aclrtStream stream,
                      aclnnStatus status) {
    uint64_t session = 0;
    if (status != ACL_SUCCESS || !writer_audit_enabled(&session)) {
        return false;
    }
    std::lock_guard<std::mutex> lock(g_writer_audit_mutex);
    ensure_writer_session_locked(session);
    WriterReadResult read = read_residual(record.x_out, stream);
    WriterResidualState state{};
    state.valid = read.ok;
    state.session = session;
    state.producer_sequence = record.sequence;
    state.metadata = record.x_out;
    if (read.ok) {
        state.p0_hash = read.hash;
        state.last_hash = read.hash;
        state.checkpoint_index = 1;
        state.p0_bytes = read.bytes;
        state.last_bytes = read.bytes;
        g_writer_residuals[record.device] = state;
    } else {
        g_writer_residuals.erase(record.device);
    }
    const WriterDiff unchanged{};
    log_writer_checkpoint(session, "P0", "aclnnAddRmsNormBias", record.device,
                          record.sequence, 0, nullptr, 0, record.x_out, stream,
                          status, read, read.hash, read.hash, unchanged,
                          unchanged, read.ok ? 1 : 0);
    return read.ok;
}

bool writer_sample_existing(
    const char *checkpoint, const char *api, int32_t device,
    uint64_t producer_sequence, uint64_t consumer_sequence,
    op::internal::OpExecCache *qmm_cache, uint64_t qmm_run_index,
    const TensorMetadata &metadata, aclrtStream stream, aclnnStatus status) {
    uint64_t session = 0;
    if (!writer_audit_enabled(&session)) {
        return false;
    }
    std::lock_guard<std::mutex> lock(g_writer_audit_mutex);
    ensure_writer_session_locked(session);
    const auto found = g_writer_residuals.find(device);
    if (found == g_writer_residuals.end() || !found->second.valid ||
        found->second.session != session ||
        found->second.producer_sequence != producer_sequence ||
        !same_tensor_storage(found->second.metadata, metadata)) {
        return false;
    }
    WriterResidualState &state = found->second;
    WriterReadResult read = read_residual(metadata, stream);
    WriterDiff from_p0{};
    WriterDiff from_previous{};
    if (read.ok) {
        from_p0 = compare_bytes(state.p0_bytes, read.bytes);
        from_previous = compare_bytes(state.last_bytes, read.bytes);
    }
    ++state.checkpoint_index;
    log_writer_checkpoint(session, checkpoint, api, device, producer_sequence,
                          consumer_sequence, qmm_cache, qmm_run_index, metadata,
                          stream, status, read, state.p0_hash, state.last_hash,
                          from_p0, from_previous, state.checkpoint_index);
    if (read.ok) {
        state.last_hash = read.hash;
        state.last_bytes = std::move(read.bytes);
    }
    return read.ok;
}

bool peek_next_add_rms_locked(int32_t device, AddRmsAddressRecord *record) {
    bool found = false;
    AddRmsAddressRecord earliest{};
    for (const auto &entry : add_rms_queues()) {
        if (entry.second.empty()) {
            continue;
        }
        const AddRmsAddressRecord &candidate = entry.second.front();
        if (candidate.device != device) {
            continue;
        }
        if (!found || candidate.sequence < earliest.sequence) {
            earliest = candidate;
            found = true;
        }
    }
    if (found && record != nullptr) {
        *record = earliest;
    }
    return found;
}

WriterQmmWindow writer_qmm_begin(
    const RegisteredQmmOperation &registered, const std::string &api,
    op::internal::OpExecCache *cache, uint64_t qmm_run_index,
    aclrtStream stream) {
    WriterQmmWindow window{};
    uint64_t session = 0;
    if (!writer_audit_enabled(&session)) {
        return window;
    }
    AddRmsAddressRecord next{};
    TensorMetadata current{};
    bool has_next = false;
    bool has_current = false;
    {
        std::lock_guard<std::mutex> lock(registry_mutex());
        has_next = peek_next_add_rms_locked(registered.device, &next);
        const auto found = current_residuals().find(registered.device);
        if (found != current_residuals().end()) {
            current = found->second;
            has_current = true;
        }
    }
    if (!has_current) {
        return window;
    }
    {
        std::lock_guard<std::mutex> lock(g_writer_audit_mutex);
        ensure_writer_session_locked(session);
        const auto found = g_writer_residuals.find(registered.device);
        if (found == g_writer_residuals.end() || !found->second.valid ||
            found->second.session != session ||
            !same_tensor_storage(found->second.metadata, current)) {
            return window;
        }
        const uint64_t max_windows = parse_env_u64(
            "FAULT26_WRITER_AUDIT_MAX_QMM_WINDOWS", 4096ULL, 1,
            1000000ULL);
        if (g_writer_qmm_windows >= max_windows) {
            return window;
        }
        ++g_writer_qmm_windows;
        window.active = true;
        window.session = session;
        window.device = registered.device;
        window.producer_sequence = found->second.producer_sequence;
        window.consumer_sequence =
            has_next && same_tensor_storage(current, next.x2) ? next.sequence : 0;
        window.qmm_run_index = qmm_run_index;
        window.cache = cache;
        window.residual = current;
    }
    (void)writer_sample_existing(
        "P1", api.c_str(), window.device, window.producer_sequence,
        window.consumer_sequence, cache, qmm_run_index, window.residual, stream,
        static_cast<aclnnStatus>(ACL_SUCCESS));
    return window;
}

void writer_qmm_end(const WriterQmmWindow &window, const std::string &api,
                    aclrtStream stream, aclnnStatus status) {
    if (!window.active) {
        return;
    }
    (void)writer_sample_existing(
        "P2", api.c_str(), window.device, window.producer_sequence,
        window.consumer_sequence, window.cache, window.qmm_run_index,
        window.residual, stream, status);
}

void writer_add_rms_checkpoint(const char *checkpoint,
                               const AddRmsAddressRecord &record,
                               aclrtStream stream, aclnnStatus status) {
    uint64_t session = 0;
    if (!writer_audit_enabled(&session)) {
        return;
    }
    uint64_t producer_sequence = 0;
    {
        std::lock_guard<std::mutex> lock(g_writer_audit_mutex);
        ensure_writer_session_locked(session);
        const auto found = g_writer_residuals.find(record.device);
        if (found == g_writer_residuals.end() || !found->second.valid ||
            found->second.session != session ||
            !same_tensor_storage(found->second.metadata, record.x2)) {
            return;
        }
        producer_sequence = found->second.producer_sequence;
    }
    (void)writer_sample_existing(
        checkpoint, "aclnnAddRmsNormBias", record.device, producer_sequence,
        record.sequence, nullptr, 0, record.x2, stream, status);
}

struct QmmStorageOrderProof {
    int32_t slots[kQmmFormalCount]{};
    uint32_t compact_storage_count{0};
    bool param_order_exact{false};
    bool address_count_exact{false};
    bool empty_slots_are_null{false};
    bool nonempty_slots_are_nonnull{false};
    bool no_shared_storage{false};
    bool slot_map_proven{false};
    bool output_slot_proven{false};
};

bool qmm_tensor_storage_unique(const Fault20QmmMetadata &metadata,
                               const Fault20TensorMetadata &target) {
    if (target.present == 0 || target.storage_ptr == 0) {
        return false;
    }
    const Fault20TensorMetadata *formal[kQmmFormalCount] = {
        &metadata.x1,        &metadata.x2,       &metadata.x1_scale,
        &metadata.x2_scale,  &metadata.y_scale,  &metadata.x1_offset,
        &metadata.x2_offset, &metadata.y_offset, &metadata.bias,
        &metadata.out,
    };
    for (const Fault20TensorMetadata *other : formal) {
        if (other != &target && other->present != 0 &&
            other->storage_ptr != 0 && other->storage_ptr == target.storage_ptr) {
            return false;
        }
    }
    return true;
}

QmmStorageOrderProof prove_qmm_storage_order(
    const RegisteredQmmOperation &registered,
    const std::vector<void *> &addresses) {
    QmmStorageOrderProof proof{};
    const Fault20QmmMetadata &metadata = registered.metadata;
    const Fault20TensorMetadata *formal[kQmmFormalCount] = {
        &metadata.x1,        &metadata.x2,       &metadata.x1_scale,
        &metadata.x2_scale,  &metadata.y_scale,  &metadata.x1_offset,
        &metadata.x2_offset, &metadata.y_offset, &metadata.bias,
        &metadata.out,
    };
    proof.param_order_exact =
        registered.tensor_param_count == kQmmFormalCount;
    proof.empty_slots_are_null = true;
    proof.nonempty_slots_are_nonnull = true;
    for (size_t i = 0; i < kQmmFormalCount; ++i) {
        proof.slots[i] = -1;
        if (registered.tensor_param_count == kQmmFormalCount &&
            registered.tensor_params[i] != formal[i]->tensor_ptr) {
            proof.param_order_exact = false;
        }
        if (formal[i]->present == 0) {
            continue;
        }
        const size_t slot = proof.compact_storage_count++;
        proof.slots[i] = static_cast<int32_t>(slot);
        if (slot >= addresses.size()) {
            proof.empty_slots_are_null = false;
            proof.nonempty_slots_are_nonnull = false;
        } else if (formal[i]->storage_bytes == 0) {
            proof.empty_slots_are_null =
                proof.empty_slots_are_null && addresses[slot] == nullptr;
        } else {
            proof.nonempty_slots_are_nonnull =
                proof.nonempty_slots_are_nonnull && addresses[slot] != nullptr;
        }
    }
    proof.address_count_exact = proof.compact_storage_count == addresses.size();
    proof.no_shared_storage = metadata.any_shared_storage == 0;
    proof.slot_map_proven =
        proof.param_order_exact && proof.address_count_exact &&
        proof.empty_slots_are_null && proof.nonempty_slots_are_nonnull &&
        proof.no_shared_storage;
    proof.output_slot_proven =
        proof.slot_map_proven && proof.slots[9] >= 0 &&
        static_cast<size_t>(proof.slots[9] + 1) == addresses.size() &&
        metadata.out.present != 0 && metadata.out.storage_bytes != 0 &&
        metadata.out.storage_ptr != 0 &&
        qmm_tensor_storage_unique(metadata, metadata.out);
    return proof;
}

bool build_qmm_underflow_context(
    const RegisteredQmmOperation &registered,
    const QmmStorageOrderProof &proof,
    const std::vector<void *> &addresses, uint64_t run_index,
    op::internal::OpExecCache *cache, Fault26QmmUnderflowContext *context) {
    if (context == nullptr || !proof.output_slot_proven ||
        proof.slots[9] < 0 ||
        static_cast<size_t>(proof.slots[9]) >= addresses.size()) {
        return false;
    }
    AddRmsAddressRecord producer{};
    bool has_producer = false;
    {
        std::lock_guard<std::mutex> lock(registry_mutex());
        const auto found = current_residual_records().find(registered.device);
        if (found != current_residual_records().end()) {
            producer = found->second;
            has_producer = true;
        }
    }
    if (!has_producer || producer.x_out.metadata_ok == 0 ||
        producer.x_out.storage_addr == 0 ||
        producer.x_out.storage_bytes == 0) {
        return false;
    }
    const Fault20TensorMetadata &out = registered.metadata.out;
    if (out.present == 0 || out.contiguous == 0 || out.view_offset != 0 ||
        out.storage_bytes == 0 || out.view_rank != 2 ||
        producer.x_out.contiguous == 0 || producer.x_out.view_offset != 0 ||
        producer.x_out.view_rank != 2) {
        return false;
    }
    *context = Fault26QmmUnderflowContext{};
    context->abi_version = FAULT26_QMM_UNDERFLOW_ABI_VERSION;
    context->device = registered.device;
    context->run_index = run_index;
    context->producer_sequence = producer.sequence;
    // The consumer does not exist yet at this point in the real execution
    // order.  It is identified later by the first successful AddRms whose x2
    // exactly matches this still-live residual storage.
    context->consumer_sequence = 0;
    context->cache = cache;
    context->original_output = addresses[static_cast<size_t>(proof.slots[9])];
    context->output_bytes = out.storage_bytes;
    context->output_m = out.view_shape[0];
    context->output_n = out.view_shape[1];
    context->residual = reinterpret_cast<void *>(producer.x_out.storage_addr);
    context->residual_bytes = producer.x_out.storage_bytes;
    context->residual_m = producer.x_out.view_shape[0];
    context->residual_k = producer.x_out.view_shape[1];
    return context->original_output != nullptr;
}

struct IntervalRelation {
    const char *name{"unknown"};
    int64_t signed_gap_bytes{0};
    bool overlap{false};
    bool valid{false};
};

IntervalRelation classify_interval(uintptr_t qmm_begin, uint64_t qmm_bytes,
                                   uintptr_t residual_begin,
                                   uint64_t residual_bytes) {
    IntervalRelation relation{};
    uintptr_t qmm_end = 0;
    uintptr_t residual_end = 0;
    if (!range_end(qmm_begin, qmm_bytes, &qmm_end) ||
        !range_end(residual_begin, residual_bytes, &residual_end)) {
        return relation;
    }
    relation.valid = true;
    if (qmm_begin < residual_end && residual_begin < qmm_end) {
        relation.name = "overlap";
        relation.overlap = true;
        return relation;
    }
    if (qmm_end == residual_begin) {
        relation.name = "tail_adjacent";
        return relation;
    }
    if (residual_end == qmm_begin) {
        relation.name = "reverse_adjacent";
        return relation;
    }
    if (qmm_end < residual_begin) {
        relation.name = "qmm_before_residual_gap";
        const uint64_t gap = static_cast<uint64_t>(residual_begin - qmm_end);
        relation.signed_gap_bytes = static_cast<int64_t>(std::min<uint64_t>(
            gap, static_cast<uint64_t>(std::numeric_limits<int64_t>::max())));
        return relation;
    }
    relation.name = "residual_before_qmm_gap";
    const uint64_t gap = static_cast<uint64_t>(qmm_begin - residual_end);
    relation.signed_gap_bytes = -static_cast<int64_t>(std::min<uint64_t>(
        gap, static_cast<uint64_t>(std::numeric_limits<int64_t>::max())));
    return relation;
}

bool pop_writer_topology_record(aclOpExecutor *executor, const char *api,
                                WriterTopologyRecord *record) {
    if (executor == nullptr || api == nullptr || record == nullptr) {
        return false;
    }
    std::lock_guard<std::mutex> lock(registry_mutex());
    const auto found = writer_topology_queues().find(executor);
    if (found == writer_topology_queues().end()) {
        return false;
    }
    auto &queue = found->second;
    const auto match = std::find_if(
        queue.begin(), queue.end(),
        [api](const WriterTopologyRecord &candidate) {
            return candidate.api == api;
        });
    if (match == queue.end()) {
        return false;
    }
    *record = *match;
    queue.erase(match);
    if (queue.empty()) {
        writer_topology_queues().erase(found);
    }
    return true;
}

void log_writer_topology_interval(const ExecuteFrame &frame) {
    if (!g_writer_topology_enabled || frame.executor == nullptr ||
        frame.api == nullptr) {
        return;
    }
    WriterTopologyRecord record{};
    const bool registration_match =
        pop_writer_topology_record(frame.executor, frame.api, &record);
    if (!registration_match) {
        return;
    }

    AddRmsAddressRecord next{};
    AddRmsAddressRecord producer{};
    TensorMetadata current{};
    bool has_next = false;
    bool has_current = false;
    {
        std::lock_guard<std::mutex> lock(registry_mutex());
        has_next = peek_next_add_rms_locked(record.device, &next);
        const auto found = current_residuals().find(record.device);
        if (found != current_residuals().end()) {
            current = found->second;
            has_current = true;
        }
        const auto producer_found =
            current_residual_records().find(record.device);
        if (producer_found != current_residual_records().end()) {
            producer = producer_found->second;
        }
    }
    const bool lifecycle_match = has_current && has_next &&
                                 same_tensor_storage(current, next.x2);
    const bool producer_backed_live_state =
        has_current && producer.x_out.metadata_ok != 0 &&
        same_tensor_storage(current, producer.x_out);
    const uintptr_t residual_begin =
        producer_backed_live_state ? current.storage_addr : 0;
    const uint64_t residual_bytes =
        producer_backed_live_state ? current.storage_bytes : 0;

    for (uint32_t region_index = 0; region_index < 2; ++region_index) {
        const bool workspace_region = region_index == 1;
        const uintptr_t span_begin =
            workspace_region ? reinterpret_cast<uintptr_t>(frame.workspace)
                             : record.writable.storage_addr;
        const uint64_t span_bytes =
            workspace_region ? frame.workspace_size
                             : record.writable.storage_bytes;
        uintptr_t span_end = 0;
        uintptr_t residual_end = 0;
        const bool span_known = range_end(span_begin, span_bytes, &span_end);
        const bool residual_known =
            range_end(residual_begin, residual_bytes, &residual_end);
        const IntervalRelation relation =
            span_known && residual_known
                ? classify_interval(span_begin, span_bytes, residual_begin,
                                    residual_bytes)
                : IntervalRelation{};
        uint64_t required_cross_bytes = 0;
        const char *candidate_direction = "unknown";
        if (relation.valid && relation.overlap) {
            candidate_direction = "overlap_or_alias";
        } else if (relation.valid &&
                   (std::strcmp(relation.name, "tail_adjacent") == 0 ||
                    relation.signed_gap_bytes > 0)) {
            candidate_direction = "forward_overrun";
            required_cross_bytes =
                static_cast<uint64_t>(relation.signed_gap_bytes) + 1;
        } else if (relation.valid &&
                   (std::strcmp(relation.name, "reverse_adjacent") == 0 ||
                    relation.signed_gap_bytes < 0)) {
            candidate_direction = "backward_underflow";
            const uint64_t gap = relation.signed_gap_bytes ==
                                         std::numeric_limits<int64_t>::min()
                                     ? static_cast<uint64_t>(
                                           std::numeric_limits<int64_t>::max()) +
                                           1ULL
                                     : static_cast<uint64_t>(
                                           -relation.signed_gap_bytes);
            required_cross_bytes = gap + 1;
        }
        const bool same_storage_as_residual =
            span_known && residual_known && span_begin == residual_begin &&
            span_bytes == residual_bytes;

        char line[6144];
        const int written = std::snprintf(
            line, sizeof(line),
            "{\"event\":\"fault26_writer_topology_interval\","
            "\"pid\":%ld,\"tid\":%ld,\"ts_ns\":%" PRIu64 ","
            "\"sequence\":%" PRIu64 ",\"device\":%d,"
            "\"api\":\"%s\",\"executor\":\"%p\","
            "\"registration_sequence\":%" PRIu64 ","
            "\"region_kind\":\"%s\",\"role\":\"%s\","
            "\"span_known\":%s,\"span_begin\":\"0x%" PRIxPTR "\","
            "\"span_end\":\"0x%" PRIxPTR "\",\"span_bytes\":%" PRIu64 ","
            "\"declared_storage_span_known\":%s,"
            "\"allocator_block_span_known\":false,"
            "\"workspace_size_gws\":%" PRIu64 ","
            "\"workspace_size_execute\":%" PRIu64 ","
            "\"workspace_size_match\":%s,"
            "\"has_current_residual\":%s,\"has_next_add_rms\":%s,"
            "\"producer_backed_live_state\":%s,"
            "\"producer_sequence\":%" PRIu64 ","
            "\"lifecycle_match\":%s,\"consumer_sequence\":%" PRIu64 ","
            "\"residual_begin\":\"0x%" PRIxPTR "\","
            "\"residual_end\":\"0x%" PRIxPTR "\","
            "\"residual_bytes\":%" PRIu64 ","
            "\"relation_valid\":%s,\"relation\":\"%s\","
            "\"signed_gap_bytes\":%" PRId64 ",\"overlap\":%s,"
            "\"same_storage_as_residual\":%s,"
            "\"candidate_direction\":\"%s\","
            "\"required_cross_bytes\":%" PRIu64 ","
            "\"device_memory_modified\":false,"
            "\"allocation_boundary_crossing_proven\":false}",
            static_cast<long>(getpid()),
            static_cast<long>(syscall(SYS_gettid)), monotonic_ns(),
            g_sequence.fetch_add(1, std::memory_order_relaxed) + 1,
            record.device, json_escape(record.api).c_str(), frame.executor,
            record.registration_sequence,
            workspace_region ? "workspace" : "writable_output",
            workspace_region ? "workspace" : json_escape(record.role).c_str(),
            span_known ? "true" : "false", span_begin,
            span_known ? span_end : 0, span_bytes,
            !workspace_region && span_known ? "true" : "false",
            record.workspace_size, frame.workspace_size,
            record.workspace_size == frame.workspace_size ? "true" : "false",
            has_current ? "true" : "false", has_next ? "true" : "false",
            producer_backed_live_state ? "true" : "false",
            producer_backed_live_state ? producer.sequence : 0,
            lifecycle_match ? "true" : "false",
            has_next ? next.sequence : 0, residual_begin,
            residual_known ? residual_end : 0, residual_bytes,
            relation.valid ? "true" : "false", relation.name,
            relation.signed_gap_bytes,
            relation.overlap ? "true" : "false",
            same_storage_as_residual ? "true" : "false",
            candidate_direction, required_cross_bytes);
        if (written > 0 && static_cast<size_t>(written) < sizeof(line)) {
            write_line(std::string(line, static_cast<size_t>(written)));
        }
    }
}

void log_qmm_residual_interval(
    const RegisteredQmmOperation &registered, const std::string &api,
    op::internal::OpExecCache *cache, uint64_t run_index,
    const std::vector<void *> &addresses, void *workspace,
    uint64_t workspace_size, aclnnStatus status) {
    const QmmStorageOrderProof proof =
        prove_qmm_storage_order(registered, addresses);
    AddRmsAddressRecord next{};
    TensorMetadata current{};
    bool has_next = false;
    bool has_current = false;
    {
        std::lock_guard<std::mutex> lock(registry_mutex());
        has_next = peek_next_add_rms_locked(registered.device, &next);
        const auto found = current_residuals().find(registered.device);
        if (found != current_residuals().end()) {
            current = found->second;
            has_current = true;
        }
    }
    const bool lifecycle_match = has_next && has_current &&
                                 same_tensor_storage(current, next.x2);
    const Fault20QmmMetadata &metadata = registered.metadata;
    const Fault20TensorMetadata *formal[kQmmFormalCount] = {
        &metadata.x1,        &metadata.x2,       &metadata.x1_scale,
        &metadata.x2_scale,  &metadata.y_scale,  &metadata.x1_offset,
        &metadata.x2_offset, &metadata.y_offset, &metadata.bias,
        &metadata.out,
    };
    const char *roles[kQmmFormalCount] = {
        "x1",        "x2",       "x1_scale", "x2_scale", "y_scale",
        "x1_offset", "x2_offset", "y_offset", "bias",     "out",
    };
    for (size_t i = 0; i <= kQmmFormalCount; ++i) {
        const bool is_workspace = i == kQmmFormalCount;
        const int32_t slot = is_workspace ? -1 : proof.slots[i];
        const uintptr_t span_begin = is_workspace
            ? reinterpret_cast<uintptr_t>(workspace)
            : (slot >= 0 && static_cast<size_t>(slot) < addresses.size()
                   ? reinterpret_cast<uintptr_t>(addresses[slot])
                   : 0);
        const uint64_t span_bytes =
            is_workspace ? workspace_size : formal[i]->storage_bytes;
        const bool span_known = span_begin != 0 && span_bytes != 0 &&
            (is_workspace || proof.slot_map_proven);
        const uintptr_t residual_begin =
            lifecycle_match ? next.x2.raw_addr : 0;
        const uint64_t residual_bytes =
            lifecycle_match ? next.x2.storage_bytes : 0;
        const IntervalRelation relation =
            span_known && lifecycle_match
                ? classify_interval(span_begin, span_bytes, residual_begin,
                                    residual_bytes)
                : IntervalRelation{};
        uintptr_t span_end = 0;
        uintptr_t residual_end = 0;
        const bool span_end_valid = range_end(span_begin, span_bytes, &span_end);
        const bool residual_end_valid =
            range_end(residual_begin, residual_bytes, &residual_end);
        char line[6144];
        const int written = std::snprintf(
            line, sizeof(line),
            "{\"event\":\"fault26_qmm_residual_interval\","
            "\"pid\":%ld,\"tid\":%ld,\"ts_ns\":%" PRIu64 ","
            "\"sequence\":%" PRIu64 ",\"device\":%d,"
            "\"api_info\":\"%s\",\"cache\":\"%p\","
            "\"run_index\":%" PRIu64 ",\"run_status\":%d,"
            "\"role\":\"%s\",\"formal_ordinal\":%d,\"slot\":%d,"
            "\"slot_map_proven\":%s,\"output_slot_proven\":%s,"
            "\"span_known\":%s,\"span_begin\":\"0x%" PRIxPTR "\","
            "\"span_end\":\"0x%" PRIxPTR "\",\"span_bytes\":%" PRIu64 ","
            "\"workspace_span_known\":%s,\"allocator_span_known\":false,"
            "\"has_current_residual\":%s,\"has_next_add_rms\":%s,"
            "\"lifecycle_match\":%s,\"consumer_sequence\":%" PRIu64 ","
            "\"residual_begin\":\"0x%" PRIxPTR "\","
            "\"residual_end\":\"0x%" PRIxPTR "\","
            "\"residual_bytes\":%" PRIu64 ",\"relation_valid\":%s,"
            "\"relation\":\"%s\",\"overlap\":%s,"
            "\"signed_gap_bytes\":%" PRId64 "}",
            static_cast<long>(getpid()),
            static_cast<long>(syscall(SYS_gettid)), monotonic_ns(),
            g_sequence.fetch_add(1, std::memory_order_relaxed) + 1,
            registered.device, json_escape(api).c_str(), cache, run_index,
            static_cast<int>(status), is_workspace ? "workspace" : roles[i],
            is_workspace ? -1 : static_cast<int32_t>(i), slot,
            proof.slot_map_proven ? "true" : "false",
            proof.output_slot_proven ? "true" : "false",
            span_known && span_end_valid ? "true" : "false", span_begin,
            span_end_valid ? span_end : 0, span_bytes,
            is_workspace && span_known && span_end_valid ? "true" : "false",
            has_current ? "true" : "false", has_next ? "true" : "false",
            lifecycle_match ? "true" : "false",
            has_next ? next.sequence : 0, residual_begin,
            residual_end_valid ? residual_end : 0, residual_bytes,
            relation.valid ? "true" : "false", relation.name,
            relation.overlap ? "true" : "false",
            relation.signed_gap_bytes);
        if (written > 0 && static_cast<size_t>(written) < sizeof(line)) {
            write_line(std::string(line, static_cast<size_t>(written)));
        }
    }
}

void log_execute_event(const char *event, const ExecuteFrame &frame,
                       aclnnStatus status) {
    char line[2048];
    int32_t device = -1;
    if (g_get_device != nullptr) {
        (void)g_get_device(&device);
    }
    const int written = std::snprintf(
        line, sizeof(line),
        "{\"event\":\"%s\",\"pid\":%ld,\"tid\":%ld,"
        "\"ts_ns\":%" PRIu64 ",\"sequence\":%" PRIu64
        ",\"device\":%d,\"kind\":%u,\"api\":\"%s\","
        "\"workspace\":\"%p\",\"workspace_size\":%" PRIu64
        ",\"executor\":\"%p\",\"stream\":\"%p\","
        "\"status\":%d,\"depth\":%zu}",
        event, static_cast<long>(getpid()),
        static_cast<long>(syscall(SYS_gettid)), monotonic_ns(),
        g_sequence.fetch_add(1, std::memory_order_relaxed) + 1, device,
        frame.kind, frame.api == nullptr ? "" : frame.api, frame.workspace,
        frame.workspace_size, frame.executor, frame.stream,
        static_cast<int>(status), g_execute_stack.depth);
    if (written > 0 && static_cast<size_t>(written) < sizeof(line)) {
        write_line(std::string(line, static_cast<size_t>(written)));
    }
}

void capture_param(const aclTensor *tensor) {
    if (!g_trace_enabled || !g_capture_tensors) {
        return;
    }
    if (g_capture.tensors.size() >= kMaxTensors) {
        g_capture.overflow = true;
        return;
    }
    g_capture.tensors.push_back(capture_tensor(tensor));
}

void capture_qmm_param(const aclTensor *tensor) {
    if (!g_qmm_capture.active ||
        g_qmm_capture.tensor_param_count >=
            sizeof(g_qmm_capture.tensor_params) /
                sizeof(g_qmm_capture.tensor_params[0])) {
        return;
    }
    g_qmm_capture.tensor_params[g_qmm_capture.tensor_param_count++] =
        reinterpret_cast<uintptr_t>(tensor);
}

void log_tensor_mapping(
    const std::string &api, const std::string &execute_api,
    op::internal::OpExecCache *cache, uint64_t run_index,
    const TensorMetadata &metadata, size_t formal_ordinal, int32_t slot,
    void *run_address, bool slot_proven, bool address_alias,
    uintptr_t workspace_address, uint64_t workspace_size) {
    const uintptr_t run_begin = reinterpret_cast<uintptr_t>(run_address);
    const bool workspace_known = workspace_address != 0 && workspace_size != 0;
    const bool overlap =
        workspace_known &&
        ranges_overlap(workspace_address, workspace_size, run_begin,
                       metadata.storage_bytes);
    const int64_t gap = workspace_known
                            ? range_gap(workspace_address, workspace_size,
                                        run_begin, metadata.storage_bytes)
                            : 0;
    char line[6144];
    const std::string escaped_api = json_escape(api);
    const std::string escaped_execute = json_escape(execute_api);
    const std::string view_shape =
        dims_json(metadata.view_shape, metadata.view_rank);
    const std::string storage_shape =
        dims_json(metadata.storage_shape, metadata.storage_rank);
    const std::string strides = dims_json(metadata.strides, metadata.stride_rank);
    const int written = std::snprintf(
        line, sizeof(line),
        "{\"event\":\"fault26_tensor_slot\",\"pid\":%ld,"
        "\"tid\":%ld,\"ts_ns\":%" PRIu64
        ",\"sequence\":%" PRIu64 ",\"api_info\":\"%s\","
        "\"execute_api\":\"%s\",\"cache\":\"%p\","
        "\"run_index\":%" PRIu64 ",\"formal_ordinal\":%zu,"
        "\"slot_candidate\":%d,\"slot_proven\":%s,"
        "\"present\":%s,\"metadata_ok\":%s,\"dtype\":%d,"
        "\"format\":%d,\"view_shape\":%s,\"storage_shape\":%s,"
        "\"strides\":%s,\"view_offset\":%" PRId64
        ",\"element_size\":%" PRIu64 ",\"view_bytes\":%" PRIu64
        ",\"storage_bytes\":%" PRIu64 ",\"tensor\":\"0x%" PRIxPTR
        "\",\"storage\":\"0x%" PRIxPTR "\",\"cold_raw\":\"0x%" PRIxPTR
        "\",\"cold_storage_addr\":\"0x%" PRIxPTR
        "\",\"run_address\":\"%p\",\"address_alias\":%s,"
        "\"workspace_known\":%s,\"workspace\":\"0x%" PRIxPTR
        "\",\"workspace_size\":%" PRIu64
        ",\"workspace_overlap\":%s,\"workspace_gap_bytes\":%" PRId64
        "}",
        static_cast<long>(getpid()),
        static_cast<long>(syscall(SYS_gettid)), monotonic_ns(),
        g_sequence.fetch_add(1, std::memory_order_relaxed) + 1,
        escaped_api.c_str(), escaped_execute.c_str(), cache, run_index,
        formal_ordinal, slot, slot_proven ? "true" : "false",
        metadata.present != 0 ? "true" : "false",
        metadata.metadata_ok != 0 ? "true" : "false", metadata.dtype,
        metadata.format, view_shape.c_str(), storage_shape.c_str(),
        strides.c_str(), metadata.view_offset, metadata.element_size,
        metadata.view_bytes, metadata.storage_bytes, metadata.tensor_ptr,
        metadata.storage_ptr, metadata.raw_addr, metadata.storage_addr,
        run_address, address_alias ? "true" : "false",
        workspace_known ? "true" : "false", workspace_address,
        workspace_size, overlap ? "true" : "false", gap);
    if (written > 0 && static_cast<size_t>(written) < sizeof(line)) {
        write_line(std::string(line, static_cast<size_t>(written)));
    }
}

void log_run_event(const char *event, const std::string &api,
                   const std::string &execute_api,
                   op::internal::OpExecCache *cache, void *workspace,
                   uint64_t workspace_size, aclrtStream stream,
                   size_t address_count, bool registered, size_t formal_count,
                   bool capture_overflow, bool slot_proven, uint64_t run_index,
                   aclnnStatus status, uint64_t elapsed_ns) {
    char line[3072];
    int32_t device = -1;
    if (g_get_device != nullptr) {
        (void)g_get_device(&device);
    }
    const std::string escaped_api = json_escape(api);
    const std::string escaped_execute = json_escape(execute_api);
    const int written = std::snprintf(
        line, sizeof(line),
        "{\"event\":\"%s\",\"pid\":%ld,\"tid\":%ld,"
        "\"ts_ns\":%" PRIu64 ",\"sequence\":%" PRIu64
        ",\"device\":%d,\"api_info\":\"%s\","
        "\"execute_api\":\"%s\",\"cache\":\"%p\","
        "\"workspace\":\"%p\",\"workspace_size\":%" PRIu64
        ",\"stream\":\"%p\",\"address_count\":%zu,"
        "\"registered\":%s,\"formal_count\":%zu,"
        "\"capture_overflow\":%s,\"slot_proven\":%s,"
        "\"run_index\":%" PRIu64 ",\"status\":%d,"
        "\"elapsed_ns\":%" PRIu64 "}",
        event, static_cast<long>(getpid()),
        static_cast<long>(syscall(SYS_gettid)), monotonic_ns(),
        g_sequence.fetch_add(1, std::memory_order_relaxed) + 1, device,
        escaped_api.c_str(), escaped_execute.c_str(), cache, workspace,
        workspace_size, stream, address_count,
        registered ? "true" : "false", formal_count,
        capture_overflow ? "true" : "false",
        slot_proven ? "true" : "false", run_index,
        static_cast<int>(status), elapsed_ns);
    if (written > 0 && static_cast<size_t>(written) < sizeof(line)) {
        write_line(std::string(line, static_cast<size_t>(written)));
    }
}

aclnnStatus trace_run(op::internal::OpExecCache *cache, void *workspace,
                      aclrtStream stream,
                      const std::vector<void *> &addresses) {
    pthread_once(&g_resolve_once, resolve_real);
    if (g_real_run == nullptr) {
        return static_cast<aclnnStatus>(1);
    }
    if (g_in_hook) {
        return g_real_run(cache, workspace, stream, addresses);
    }

    g_in_hook = true;
    const std::string api = current_api_info();
    const ExecuteFrame *frame = current_execute_frame();
    const std::string execute_api =
        frame == nullptr || frame->api == nullptr ? "" : frame->api;
    const bool selected =
        g_trace_enabled && (frame != nullptr || api_matches_filter(api));
    if (!selected) {
        g_in_hook = false;
        const aclnnStatus status =
            g_real_run(cache, workspace, stream, addresses);
        g_in_hook = true;
        notify_qmm_after_run(api, cache, stream, status);
        g_in_hook = false;
        return status;
    }

    RegisteredOperation operation{};
    bool registered = false;
    uint64_t run_index = 0;
    {
        std::lock_guard<std::mutex> lock(registry_mutex());
        const auto found = registry().find(cache);
        if (found != registry().end()) {
            registered = true;
            ++found->second.run_count;
            run_index = found->second.run_count;
            operation = found->second;
        }
    }

    RegisteredQmmOperation qmm_operation{};
    bool qmm_registered = false;
    uint64_t qmm_run_index = 0;
    if (qmm_api_matches(api)) {
        std::lock_guard<std::mutex> lock(registry_mutex());
        const auto found = qmm_registry().find(cache);
        if (found != qmm_registry().end()) {
            qmm_registered = true;
            ++found->second.run_count;
            qmm_run_index = found->second.run_count;
            qmm_operation = found->second;
        }
    }

    std::vector<int32_t> slots(operation.tensors.size(), -1);
    std::unordered_set<uintptr_t> storage_ids;
    bool metadata_complete = registered && !operation.tensors.empty();
    bool unique_storage = true;
    size_t compact_count = 0;
    for (size_t i = 0; i < operation.tensors.size(); ++i) {
        const TensorMetadata &metadata = operation.tensors[i];
        if (metadata.present == 0 || metadata.metadata_ok == 0) {
            metadata_complete = false;
            continue;
        }
        if (metadata.storage_bytes == 0 || metadata.storage_addr == 0) {
            continue;
        }
        slots[i] = static_cast<int32_t>(compact_count++);
        if (metadata.storage_ptr == 0 ||
            !storage_ids.insert(metadata.storage_ptr).second) {
            unique_storage = false;
        }
    }
    const bool slot_proven = registered && !operation.overflow &&
                             metadata_complete && unique_storage &&
                             compact_count == addresses.size();
    uint64_t workspace_size = 0;
    uintptr_t workspace_address = reinterpret_cast<uintptr_t>(workspace);
    if (frame != nullptr && frame->workspace == workspace &&
        frame->stream == stream) {
        workspace_size = frame->workspace_size;
    }

    WriterQmmWindow writer_qmm_window{};
    if (qmm_registered) {
        writer_qmm_window = writer_qmm_begin(
            qmm_operation, api, cache, qmm_run_index, stream);
    }
    Fault26QmmUnderflowContext underflow_context{};
    Fault26QmmUnderflowDecision underflow_decision{};
    std::vector<void *> underflow_addresses;
    const std::vector<void *> *run_addresses = &addresses;
    if (qmm_registered && g_qmm_underflow_prepare != nullptr &&
        g_qmm_underflow_complete != nullptr) {
        const QmmStorageOrderProof proof =
            prove_qmm_storage_order(qmm_operation, addresses);
        if (build_qmm_underflow_context(
                qmm_operation, proof, addresses, qmm_run_index, cache,
                &underflow_context)) {
            try {
                underflow_addresses = addresses;
                const bool claimed = g_qmm_underflow_prepare(
                    &underflow_context, stream, &underflow_decision);
                if (claimed && underflow_decision.claimed != 0 &&
                    underflow_decision.abi_version ==
                        FAULT26_QMM_UNDERFLOW_ABI_VERSION &&
                    underflow_decision.replacement_output != nullptr) {
                    underflow_addresses[static_cast<size_t>(proof.slots[9])] =
                        underflow_decision.replacement_output;
                    run_addresses = &underflow_addresses;
                } else {
                    underflow_decision = Fault26QmmUnderflowDecision{};
                }
            } catch (...) {
                underflow_decision = Fault26QmmUnderflowDecision{};
            }
        }
    }
    log_run_event("fault26_run_enter", api, execute_api, cache, workspace,
                  workspace_size, stream, addresses.size(), registered,
                  operation.tensors.size(), operation.overflow, slot_proven,
                  run_index, static_cast<aclnnStatus>(0), 0);
    const uint64_t begin_ns = monotonic_ns();
    g_in_hook = false;
    const aclnnStatus status =
        g_real_run(cache, workspace, stream, *run_addresses);
    g_in_hook = true;
    if (underflow_decision.claimed != 0 &&
        g_qmm_underflow_complete != nullptr) {
        try {
            g_qmm_underflow_complete(&underflow_context, &underflow_decision,
                                     stream, status);
        } catch (...) {
            // The real operator status remains authoritative.
        }
    }
    writer_qmm_end(writer_qmm_window, api, stream, status);
    if (qmm_registered) {
        log_qmm_residual_interval(qmm_operation, api, cache, qmm_run_index,
                                  addresses, workspace, workspace_size,
                                  status);
    }
    notify_qmm_after_run(api, cache, stream, status);
    const uint64_t elapsed_ns = monotonic_ns() - begin_ns;

    for (size_t i = 0; i < operation.tensors.size(); ++i) {
        void *run_address = nullptr;
        const int32_t slot = slots[i];
        if (slot >= 0 && static_cast<size_t>(slot) < addresses.size()) {
            run_address = addresses[static_cast<size_t>(slot)];
        }
        bool alias = false;
        if (run_address != nullptr) {
            alias = std::count(addresses.begin(), addresses.end(), run_address) > 1;
        }
        log_tensor_mapping(api, execute_api, cache, run_index,
                           operation.tensors[i], i, slot, run_address,
                           slot_proven, alias, workspace_address,
                           workspace_size);
    }
    log_run_event("fault26_run_exit", api, execute_api, cache, workspace,
                  workspace_size, stream, addresses.size(), registered,
                  operation.tensors.size(), operation.overflow, slot_proven,
                  run_index, status, elapsed_ns);
    g_in_hook = false;
    return status;
}

void add_param_const(const aclTensor *tensor) {
    pthread_once(&g_resolve_once, resolve_real);
    if (g_real_add_param_const_tensor == nullptr) {
        return;
    }
    const bool outermost = !g_in_hook;
    if (outermost) {
        g_in_hook = true;
        capture_qmm_param(tensor);
        capture_param(tensor);
    }
    g_real_add_param_const_tensor(tensor);
    if (outermost) {
        g_in_hook = false;
    }
}

void add_param_mutable(aclTensor *tensor) {
    pthread_once(&g_resolve_once, resolve_real);
    if (g_real_add_param_tensor == nullptr) {
        return;
    }
    const bool outermost = !g_in_hook;
    if (outermost) {
        g_in_hook = true;
        capture_qmm_param(tensor);
        capture_param(tensor);
    }
    g_real_add_param_tensor(tensor);
    if (outermost) {
        g_in_hook = false;
    }
}

bool add_cache(op::internal::OpExecCache *cache) {
    pthread_once(&g_resolve_once, resolve_real);
    CaptureState snapshot{};
    if (g_trace_enabled && !g_in_hook) {
        snapshot = std::move(g_capture);
        g_capture = CaptureState{};
    }
    if (g_real_add == nullptr) {
        return false;
    }
    const bool result = g_real_add(cache);
    if (g_qmm_capture.active && result && cache != nullptr) {
        ++g_qmm_capture.add_count;
        if (g_qmm_capture.add_count == 1) {
            g_qmm_capture.cache = cache;
        }
    }
    if (g_trace_enabled && result && cache != nullptr &&
        (!snapshot.tensors.empty() || snapshot.overflow)) {
        RegisteredOperation operation{};
        operation.tensors = std::move(snapshot.tensors);
        operation.overflow = snapshot.overflow;
        std::lock_guard<std::mutex> lock(registry_mutex());
        registry()[cache] = std::move(operation);
    }
    return result;
}

}  // namespace

extern "C" void fault20_qmm_gws_begin() {
    g_qmm_capture = QmmCaptureState{};
    g_qmm_capture.active = true;
}

extern "C" bool fault20_qmm_gws_commit(
    const Fault20QmmMetadata *metadata) {
    pthread_once(&g_resolve_once, resolve_real);
    const QmmCaptureState snapshot = g_qmm_capture;
    g_qmm_capture = QmmCaptureState{};
    const bool accepted = metadata != nullptr && snapshot.active &&
                          snapshot.add_count == 1 && snapshot.cache != nullptr &&
                          snapshot.tensor_param_count == kQmmFormalCount;
    int32_t device = -1;
    if (accepted && g_get_device != nullptr) {
        (void)g_get_device(&device);
    }
    if (accepted) {
        RegisteredQmmOperation registered{};
        registered.metadata = *metadata;
        registered.device = device;
        registered.tensor_param_count = snapshot.tensor_param_count;
        std::memcpy(registered.tensor_params, snapshot.tensor_params,
                    sizeof(registered.tensor_params));
        std::lock_guard<std::mutex> lock(registry_mutex());
        qmm_registry()[snapshot.cache] = registered;
    }
    char line[2048];
    const int written = std::snprintf(
        line, sizeof(line),
        "{\"event\":\"fault26_qmm_typed_registration\","
        "\"pid\":%ld,\"tid\":%ld,\"ts_ns\":%" PRIu64 ","
        "\"sequence\":%" PRIu64 ",\"accepted\":%s,\"device\":%d,"
        "\"cache\":\"%p\",\"add_count\":%u,"
        "\"tensor_param_count\":%u,\"present_count\":%u,"
        "\"any_shared_storage\":%s}",
        static_cast<long>(getpid()),
        static_cast<long>(syscall(SYS_gettid)), monotonic_ns(),
        g_sequence.fetch_add(1, std::memory_order_relaxed) + 1,
        accepted ? "true" : "false", device, snapshot.cache,
        snapshot.add_count, snapshot.tensor_param_count,
        metadata == nullptr ? 0U : metadata->present_count,
        metadata != nullptr && metadata->any_shared_storage != 0 ? "true"
                                                                  : "false");
    if (written > 0 && static_cast<size_t>(written) < sizeof(line)) {
        write_line(std::string(line, static_cast<size_t>(written)));
    }
    return accepted;
}

extern "C" void fault20_qmm_gws_abort() {
    g_qmm_capture = QmmCaptureState{};
}

extern "C" void fault26_trace_writer_gws(
    const char *api, const char *role, const aclTensor *writable,
    uint64_t workspace_size, aclOpExecutor *executor, aclnnStatus status) {
    pthread_once(&g_resolve_once, resolve_real);
    if (!g_writer_topology_enabled || api == nullptr || role == nullptr ||
        writable == nullptr || executor == nullptr || status != ACL_SUCCESS) {
        return;
    }
    const bool previous_hook = g_in_hook;
    g_in_hook = true;
    WriterTopologyRecord record{};
    record.api = api;
    record.role = role;
    record.executor = executor;
    record.writable = capture_tensor(writable);
    record.workspace_size = workspace_size;
    if (g_get_device != nullptr) {
        (void)g_get_device(&record.device);
    }
    record.registration_sequence =
        g_sequence.fetch_add(1, std::memory_order_relaxed) + 1;
    g_in_hook = previous_hook;
    if (record.writable.metadata_ok == 0 ||
        record.writable.storage_addr == 0 ||
        record.writable.storage_bytes == 0) {
        return;
    }
    {
        std::lock_guard<std::mutex> lock(registry_mutex());
        writer_topology_queues()[executor].push_back(record);
    }
    char line[4096];
    uintptr_t span_end = 0;
    const bool span_known =
        range_end(record.writable.storage_addr,
                  record.writable.storage_bytes, &span_end);
    const int written = std::snprintf(
        line, sizeof(line),
        "{\"event\":\"fault26_writer_topology_registration\","
        "\"pid\":%ld,\"tid\":%ld,\"ts_ns\":%" PRIu64 ","
        "\"sequence\":%" PRIu64 ",\"device\":%d,"
        "\"api\":\"%s\",\"role\":\"%s\","
        "\"executor\":\"%p\",\"workspace_size\":%" PRIu64 ","
        "\"storage_begin\":\"0x%" PRIxPTR "\","
        "\"storage_end\":\"0x%" PRIxPTR "\","
        "\"storage_bytes\":%" PRIu64 ","
        "\"view_offset\":%" PRId64 ",\"dtype\":%d,\"format\":%d,"
        "\"metadata_ok\":true,\"device_memory_modified\":false}",
        static_cast<long>(getpid()),
        static_cast<long>(syscall(SYS_gettid)), monotonic_ns(),
        record.registration_sequence, record.device,
        json_escape(record.api).c_str(), json_escape(record.role).c_str(),
        executor, workspace_size, record.writable.storage_addr,
        span_known ? span_end : 0, record.writable.storage_bytes,
        record.writable.view_offset, record.writable.dtype,
        record.writable.format);
    if (written > 0 && static_cast<size_t>(written) < sizeof(line)) {
        write_line(std::string(line, static_cast<size_t>(written)));
    }
}

extern "C" void fault26_trace_execute_begin(
    uint32_t kind, const char *api, void *workspace, uint64_t workspace_size,
    aclOpExecutor *executor, aclrtStream stream) {
    pthread_once(&g_resolve_once, resolve_real);
    if (!g_trace_enabled) {
        return;
    }
    if (g_execute_stack.depth >= kMaxExecuteDepth) {
        g_execute_stack.overflow = true;
        return;
    }
    ExecuteFrame &frame = g_execute_stack.frames[g_execute_stack.depth++];
    frame.kind = kind;
    frame.api = api;
    frame.workspace = workspace;
    frame.workspace_size = workspace_size;
    frame.executor = executor;
    frame.stream = stream;
    frame.begin_ns = monotonic_ns();
    log_writer_topology_interval(frame);
    if (kind == FAULT26_EXECUTE_ADD_RMS_NORM_BIAS) {
        g_active_add_rms = ActiveAddRmsAddress{};
        {
            std::lock_guard<std::mutex> lock(registry_mutex());
            const auto found = add_rms_queues().find(executor);
            if (found != add_rms_queues().end() && !found->second.empty()) {
                g_active_add_rms.valid = true;
                g_active_add_rms.record = found->second.front();
                found->second.pop_front();
                if (found->second.empty()) {
                    add_rms_queues().erase(found);
                }
            }
        }
        if (g_active_add_rms.valid) {
            writer_add_rms_checkpoint(
                "P3", g_active_add_rms.record, stream,
                static_cast<aclnnStatus>(ACL_SUCCESS));
        }
    }
    log_execute_event("fault26_execute_enter", frame,
                      static_cast<aclnnStatus>(0));
}

extern "C" void fault26_trace_execute_end(uint32_t kind, const char *api,
                                           aclnnStatus status) {
    pthread_once(&g_resolve_once, resolve_real);
    if (!g_trace_enabled || g_execute_stack.depth == 0) {
        return;
    }
    ExecuteFrame frame = g_execute_stack.frames[g_execute_stack.depth - 1];
    if (frame.kind != kind || frame.api == nullptr || api == nullptr ||
        std::strcmp(frame.api, api) != 0) {
        g_execute_stack.overflow = true;
    }
    log_execute_event("fault26_execute_exit", frame, status);
    if (kind == FAULT26_EXECUTE_ADD_RMS_NORM_BIAS &&
        g_active_add_rms.valid) {
        (void)writer_sample_p0(g_active_add_rms.record, frame.stream, status);
        if (g_qmm_underflow_consumer != nullptr) {
            try {
                g_qmm_underflow_consumer(
                    g_active_add_rms.record.device,
                    g_active_add_rms.record.sequence,
                    reinterpret_cast<void *>(
                        g_active_add_rms.record.x2.storage_addr),
                    g_active_add_rms.record.x2.storage_bytes,
                    reinterpret_cast<void *>(
                        g_active_add_rms.record.x_out.storage_addr),
                    g_active_add_rms.record.x_out.storage_bytes,
                    frame.stream, status);
            } catch (...) {
                // The diagnostic callback is fail-open after real execution.
            }
        }
        std::lock_guard<std::mutex> lock(registry_mutex());
        if (status == ACL_SUCCESS &&
            g_active_add_rms.record.x_out.metadata_ok != 0) {
            current_residuals()[g_active_add_rms.record.device] =
                g_active_add_rms.record.x_out;
            current_residual_records()[g_active_add_rms.record.device] =
                g_active_add_rms.record;
        } else {
            current_residuals().erase(g_active_add_rms.record.device);
            current_residual_records().erase(
                g_active_add_rms.record.device);
        }
        g_active_add_rms = ActiveAddRmsAddress{};
    }
    --g_execute_stack.depth;
    if (g_execute_stack.depth == 0) {
        g_execute_stack.overflow = false;
    }
}

extern "C" void fault26_trace_add_rms_consume(aclrtStream stream) {
    pthread_once(&g_resolve_once, resolve_real);
    if (!g_trace_enabled || !g_active_add_rms.valid) {
        return;
    }
    writer_add_rms_checkpoint(
        "P4", g_active_add_rms.record, stream,
        static_cast<aclnnStatus>(ACL_SUCCESS));
}

extern "C" void fault26_trace_add_rms_norm_bias_gws(
    const aclTensor *x1, const aclTensor *x2, const aclTensor *gamma,
    const aclTensor *beta_optional, double epsilon, const aclTensor *y_out,
    const aclTensor *rstd_out, const aclTensor *x_out,
    uint64_t workspace_size, aclOpExecutor *executor, aclnnStatus status) {
    pthread_once(&g_resolve_once, resolve_real);
    if (!g_trace_enabled) {
        return;
    }
    const char *api = "aclnnAddRmsNormBias";
    log_typed_gws_summary(api, epsilon, workspace_size, executor, status);
    if (status == ACL_SUCCESS && executor != nullptr) {
        const bool previous_hook = g_in_hook;
        g_in_hook = true;
        AddRmsAddressRecord record{};
        record.executor = executor;
        record.x2 = capture_tensor(x2);
        record.x_out = capture_tensor(x_out);
        if (g_get_device != nullptr) {
            (void)g_get_device(&record.device);
        }
        record.sequence =
            g_add_rms_sequence.fetch_add(1, std::memory_order_relaxed) + 1;
        g_in_hook = previous_hook;
        if (record.x2.metadata_ok != 0 && record.x_out.metadata_ok != 0) {
            std::lock_guard<std::mutex> lock(registry_mutex());
            add_rms_queues()[executor].push_back(record);
        }
    }
    if (!g_capture_tensors) {
        return;
    }
    const bool previous_hook = g_in_hook;
    g_in_hook = true;
    log_typed_gws_tensor(api, "input.x1", x1, workspace_size, executor,
                         status);
    log_typed_gws_tensor(api, "input.x2", x2, workspace_size, executor,
                         status);
    log_typed_gws_tensor(api, "input.gamma", gamma, workspace_size, executor,
                         status);
    log_typed_gws_tensor(api, "input.beta_optional", beta_optional,
                         workspace_size, executor, status);
    log_typed_gws_tensor(api, "output.y", y_out, workspace_size, executor,
                         status);
    log_typed_gws_tensor(api, "output.rstd", rstd_out, workspace_size,
                         executor, status);
    log_typed_gws_tensor(api, "output.x", x_out, workspace_size, executor,
                         status);
    g_in_hook = previous_hook;
}

namespace op::internal {

void AddParamToBuf(const aclTensor *tensor) {
    add_param_const(tensor);
}

void AddParamToBuf(aclTensor *tensor) {
    add_param_mutable(tensor);
}

aclnnStatus OpExecCache::Run(void *workspace, const aclrtStream stream,
                             const std::vector<void *> &addresses) {
    return trace_run(this, workspace, stream, addresses);
}

bool AddOpExecCache(OpExecCache *cache) {
    return add_cache(cache);
}

}  // namespace op::internal
