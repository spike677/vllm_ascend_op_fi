#include <acl/acl.h>
#include <aclnn/acl_meta.h>

#include <algorithm>
#include <atomic>
#include <cerrno>
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
#include <utility>
#include <vector>

#if !defined(__aarch64__)
#error "Fault26 FIA writer proxy is version-locked to aarch64"
#endif

#if !defined(_GLIBCXX_USE_CXX11_ABI) || _GLIBCXX_USE_CXX11_ABI != 0
#error "Fault26 FIA writer proxy requires ABI0"
#endif

namespace {

constexpr uint64_t kDefaultMaxBytes = 4ULL * 1024ULL * 1024ULL;
constexpr int32_t kNotCalled = -2147483647;

using GetDeviceFn = aclError (*)(int32_t *);
using GetDataTypeFn = aclnnStatus (*)(const aclTensor *, aclDataType *);
using GetFormatFn = aclnnStatus (*)(const aclTensor *, aclFormat *);
using GetViewOffsetFn = aclnnStatus (*)(const aclTensor *, int64_t *);
using GetShapeFn = aclnnStatus (*)(const aclTensor *, int64_t **, uint64_t *);
using GetRawTensorAddrFn = aclnnStatus (*)(const aclTensor *, void **);
using DataTypeSizeFn = size_t (*)(aclDataType);
using CreateTensorFn = aclTensor *(*)(const int64_t *, uint64_t, aclDataType,
                                      const int64_t *, int64_t, aclFormat,
                                      const int64_t *, uint64_t, void *);
using DestroyTensorFn = aclnnStatus (*)(const aclTensor *);
using MallocFn = aclError (*)(void **, size_t, aclrtMemMallocPolicy);
using FreeFn = aclError (*)(void *);
using MallocHostFn = aclError (*)(void **, size_t);
using FreeHostFn = aclError (*)(void *);
using MemcpyFn = aclError (*)(void *, size_t, const void *, size_t,
                              aclrtMemcpyKind);
using SynchronizeStreamFn = aclError (*)(aclrtStream);

struct TensorMetadata {
    bool ok{false};
    bool contiguous{false};
    aclDataType dtype{};
    aclFormat format{};
    int64_t offset{0};
    uint64_t element_size{0};
    uint64_t bytes{0};
    void *address{nullptr};
    std::vector<int64_t> view_dims;
    std::vector<int64_t> storage_dims;
    std::vector<int64_t> strides;
};

enum class WriterMode : uint32_t {
    kLosslessOutputSubstitution = 1,
    kWrongDestinationResidualStomp = 2,
};

struct Record {
    bool active{false};
    uint64_t session{0};
    uint64_t matching_index{0};
    int32_t device{-1};
    TensorMetadata metadata{};
    const aclTensor *original_tensor{nullptr};
    aclTensor *replacement_tensor{nullptr};
    void *temporary{nullptr};
    int32_t malloc_status{kNotCalled};
    WriterMode mode{WriterMode::kLosslessOutputSubstitution};
    TensorMetadata victim{};
    uint64_t victim_producer_sequence{0};
    uint64_t victim_offset_bytes{0};
    void *victim_backup{nullptr};
    uint64_t victim_backup_bytes{0};
    uint64_t victim_hash_before{0};
    uint64_t victim_hash_after{0};
    uint64_t victim_hash_restored{0};
    uint64_t output_hash_after{0};
    int32_t pre_sync_status{kNotCalled};
    int32_t host_alloc_status{kNotCalled};
    int32_t victim_d2h_status{kNotCalled};
    int32_t post_sync_status{kNotCalled};
    int32_t victim_readback_status{kNotCalled};
    int32_t output_copy_status{kNotCalled};
    int32_t output_readback_status{kNotCalled};
    int32_t destroy_tensor_status{kNotCalled};
    int32_t restore_status{kNotCalled};
    int32_t restore_readback_status{kNotCalled};
    int32_t host_free_status{kNotCalled};
    bool pre_execute_ready{false};
    bool victim_changed{false};
    bool output_copy_equal{false};
    bool consumer_matched{false};
};

struct AddRmsRecord {
    aclOpExecutor *executor{nullptr};
    int32_t device{-1};
    uint64_t sequence{0};
    TensorMetadata x2{};
    TensorMetadata x_out{};
};

struct ResidualRecord {
    TensorMetadata tensor{};
    uint64_t producer_sequence{0};
};

struct ActiveAddRms {
    bool valid{false};
    bool consumes_pending{false};
    AddRmsRecord record{};
};

pthread_once_t g_once = PTHREAD_ONCE_INIT;
GetDeviceFn g_get_device = nullptr;
GetDataTypeFn g_get_data_type = nullptr;
GetFormatFn g_get_format = nullptr;
GetViewOffsetFn g_get_view_offset = nullptr;
GetShapeFn g_get_view_shape = nullptr;
GetShapeFn g_get_storage_shape = nullptr;
GetShapeFn g_get_view_strides = nullptr;
GetRawTensorAddrFn g_get_raw_tensor_addr = nullptr;
DataTypeSizeFn g_data_type_size = nullptr;
CreateTensorFn g_create_tensor = nullptr;
DestroyTensorFn g_destroy_tensor = nullptr;
MallocFn g_malloc = nullptr;
FreeFn g_free = nullptr;
MallocHostFn g_malloc_host = nullptr;
FreeHostFn g_free_host = nullptr;
MemcpyFn g_memcpy = nullptr;
SynchronizeStreamFn g_synchronize_stream = nullptr;
void *g_nnopbase = nullptr;
void *g_ascendcl = nullptr;
bool g_allowed = false;
bool g_runtime_ready = false;
std::mutex g_control_mutex;
uint64_t g_session = 0;
uint64_t g_matching_count = 0;
uint64_t g_hit_count = 0;
bool g_hit_claimed = false;
uint64_t g_min_residual_sequence = 0;
std::mutex g_records_mutex;
std::unordered_map<aclOpExecutor *, std::deque<Record>> g_records;
thread_local Record g_prepared;
std::atomic<uint64_t> g_prepare_calls{0};
std::atomic<uint64_t> g_enabled_prepare_calls{0};
std::atomic<uint64_t> g_add_rms_sequence{0};
std::mutex g_lifecycle_mutex;
std::unordered_map<aclOpExecutor *, std::deque<AddRmsRecord>> g_add_rms_queues;
std::unordered_map<int32_t, ResidualRecord> g_current_residuals;
std::unordered_map<int32_t, Record> g_pending_stomps;
thread_local ActiveAddRms g_active_add_rms;

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

uint64_t env_u64(const char *name, uint64_t fallback, uint64_t minimum,
                 uint64_t maximum) {
    const char *value = std::getenv(name);
    if (value == nullptr || value[0] == '\0') {
        return fallback;
    }
    char *end = nullptr;
    errno = 0;
    const unsigned long long parsed = std::strtoull(value, &end, 10);
    if (errno != 0 || end == value || *end != '\0' || parsed < minimum ||
        parsed > maximum) {
        return fallback;
    }
    return static_cast<uint64_t>(parsed);
}

void write_line(const char *line, size_t length) {
    const char *path = std::getenv("FAULT26_FIA_WRITER_LOG_PATH");
    if (path == nullptr || path[0] == '\0') {
        return;
    }
    const int fd = open(path, O_WRONLY | O_CREAT | O_APPEND | O_CLOEXEC, 0600);
    if (fd < 0) {
        return;
    }
    char newline = '\n';
    iovec parts[2] = {{const_cast<char *>(line), length}, {&newline, 1}};
    const ssize_t written = writev(fd, parts, 2);
    (void)written;
    (void)close(fd);
}

void *resolve_loaded(const char *library, const char *symbol, void **handle) {
    if (*handle == nullptr) {
        *handle = dlopen(library, RTLD_NOW | RTLD_LOCAL | RTLD_NOLOAD);
    }
    return *handle == nullptr ? nullptr : dlsym(*handle, symbol);
}

template <typename T>
T resolve_nnopbase(const char *symbol) {
    return reinterpret_cast<T>(
        resolve_loaded("libnnopbase.so", symbol, &g_nnopbase));
}

template <typename T>
T resolve_ascendcl(const char *symbol) {
    return reinterpret_cast<T>(
        resolve_loaded("libascendcl.so", symbol, &g_ascendcl));
}

void initialize() {
    g_get_device = resolve_ascendcl<GetDeviceFn>("aclrtGetDevice");
    g_get_data_type = resolve_nnopbase<GetDataTypeFn>("aclGetDataType");
    g_get_format = resolve_nnopbase<GetFormatFn>("aclGetFormat");
    g_get_view_offset = resolve_nnopbase<GetViewOffsetFn>("aclGetViewOffset");
    g_get_view_shape = resolve_nnopbase<GetShapeFn>("aclGetViewShape");
    g_get_storage_shape = resolve_nnopbase<GetShapeFn>("aclGetStorageShape");
    g_get_view_strides = resolve_nnopbase<GetShapeFn>("aclGetViewStrides");
    g_get_raw_tensor_addr =
        resolve_nnopbase<GetRawTensorAddrFn>("aclGetRawTensorAddr");
    g_data_type_size = resolve_nnopbase<DataTypeSizeFn>("aclDataTypeSize");
    g_create_tensor = resolve_nnopbase<CreateTensorFn>("aclCreateTensor");
    g_destroy_tensor = resolve_nnopbase<DestroyTensorFn>("aclDestroyTensor");
    g_malloc = resolve_ascendcl<MallocFn>("aclrtMalloc");
    g_free = resolve_ascendcl<FreeFn>("aclrtFree");
    g_malloc_host = resolve_ascendcl<MallocHostFn>("aclrtMallocHost");
    g_free_host = resolve_ascendcl<FreeHostFn>("aclrtFreeHost");
    g_memcpy = resolve_ascendcl<MemcpyFn>("aclrtMemcpy");
    g_synchronize_stream =
        resolve_ascendcl<SynchronizeStreamFn>("aclrtSynchronizeStream");
    g_runtime_ready =
        g_get_device != nullptr && g_get_data_type != nullptr &&
        g_get_format != nullptr && g_get_view_offset != nullptr &&
        g_get_view_shape != nullptr && g_get_storage_shape != nullptr &&
        g_get_view_strides != nullptr && g_get_raw_tensor_addr != nullptr &&
        g_data_type_size != nullptr && g_create_tensor != nullptr &&
        g_destroy_tensor != nullptr && g_malloc != nullptr &&
        g_free != nullptr && g_malloc_host != nullptr &&
        g_free_host != nullptr && g_memcpy != nullptr &&
        g_synchronize_stream != nullptr;
    g_allowed = env_is_one("VLLM_SERVER_DEV_MODE") &&
                env_is_one("FAULT26_FIA_WRITER_PROXY_ALLOWED") &&
                g_runtime_ready;
    char line[1024];
    const int written = std::snprintf(
        line, sizeof(line),
        "{\"event\":\"fault26_fia_writer_initialized\",\"pid\":%ld,"
        "\"tid\":%ld,\"ts_ns\":%" PRIu64 ",\"allowed\":%s,"
        "\"runtime_ready\":%s,\"supported_modes\":"
        "[\"lossless_output_substitution\","
        "\"wrong_destination_residual_stomp\"],"
        "\"residual_write_supported\":true}",
        static_cast<long>(getpid()),
        static_cast<long>(syscall(SYS_gettid)), monotonic_ns(),
        g_allowed ? "true" : "false", g_runtime_ready ? "true" : "false");
    if (written > 0 && static_cast<size_t>(written) < sizeof(line)) {
        write_line(line, static_cast<size_t>(written));
    }
}

bool multiply_u64(uint64_t left, uint64_t right, uint64_t *out) {
    if (out == nullptr ||
        (right != 0 && left > std::numeric_limits<uint64_t>::max() / right)) {
        return false;
    }
    *out = left * right;
    return true;
}

bool copy_dims(GetShapeFn getter, const aclTensor *tensor,
               std::vector<int64_t> *out) {
    int64_t *dims = nullptr;
    uint64_t rank = 0;
    if (getter == nullptr || out == nullptr ||
        getter(tensor, &dims, &rank) != ACL_SUCCESS ||
        (rank != 0 && dims == nullptr) || rank > 32) {
        return false;
    }
    if (rank == 0) {
        out->clear();
    } else {
        out->assign(dims, dims + rank);
    }
    return true;
}

TensorMetadata capture_tensor(const aclTensor *tensor) {
    TensorMetadata result{};
    if (tensor == nullptr ||
        g_get_data_type(tensor, &result.dtype) != ACL_SUCCESS ||
        g_get_format(tensor, &result.format) != ACL_SUCCESS ||
        g_get_view_offset(tensor, &result.offset) != ACL_SUCCESS ||
        !copy_dims(g_get_view_shape, tensor, &result.view_dims) ||
        !copy_dims(g_get_storage_shape, tensor, &result.storage_dims) ||
        !copy_dims(g_get_view_strides, tensor, &result.strides) ||
        g_get_raw_tensor_addr(tensor, &result.address) != ACL_SUCCESS ||
        result.address == nullptr || result.offset != 0 ||
        result.view_dims.size() != result.strides.size()) {
        return result;
    }
    result.element_size = g_data_type_size(result.dtype);
    if (result.element_size == 0) {
        return result;
    }
    uint64_t elements = 1;
    for (const int64_t dim : result.storage_dims) {
        if (dim < 0 || !multiply_u64(elements, static_cast<uint64_t>(dim),
                                     &elements)) {
            return result;
        }
    }
    if (!multiply_u64(elements, result.element_size, &result.bytes) ||
        result.bytes == 0) {
        return result;
    }
    int64_t expected = 1;
    result.contiguous = true;
    for (size_t index = result.view_dims.size(); index > 0; --index) {
        const size_t position = index - 1;
        if (result.view_dims[position] > 1 &&
            result.strides[position] != expected) {
            result.contiguous = false;
            break;
        }
        if (result.view_dims[position] > 0 &&
            expected <= std::numeric_limits<int64_t>::max() /
                            std::max<int64_t>(result.view_dims[position], 1)) {
            expected *= std::max<int64_t>(result.view_dims[position], 1);
        }
    }
    result.ok = result.contiguous;
    return result;
}

uint64_t fnv1a64_bytes(const unsigned char *bytes, size_t size) {
    uint64_t value = 1469598103934665603ULL;
    for (size_t index = 0; index < size; ++index) {
        value ^= bytes[index];
        value *= 1099511628211ULL;
    }
    return value;
}

uint64_t fnv1a64(const std::vector<unsigned char> &bytes) {
    return fnv1a64_bytes(bytes.data(), bytes.size());
}

void cleanup_record(Record *record) {
    if (record == nullptr) {
        return;
    }
    if (record->replacement_tensor != nullptr && g_destroy_tensor != nullptr) {
        (void)g_destroy_tensor(record->replacement_tensor);
    }
    if (record->temporary != nullptr && g_free != nullptr) {
        (void)g_free(record->temporary);
    }
    if (record->victim_backup != nullptr && g_free_host != nullptr) {
        (void)g_free_host(record->victim_backup);
    }
    *record = Record{};
}

bool same_storage(const TensorMetadata &left, const TensorMetadata &right) {
    return left.ok && right.ok && left.address == right.address &&
           left.bytes == right.bytes && left.dtype == right.dtype &&
           left.offset == 0 && right.offset == 0;
}

WriterMode requested_mode() {
    const char *mode = std::getenv("FAULT26_FIA_WRITER_PROXY_MODE");
    if (mode != nullptr &&
        std::strcmp(mode, "wrong_destination_residual_stomp") == 0) {
        return WriterMode::kWrongDestinationResidualStomp;
    }
    return WriterMode::kLosslessOutputSubstitution;
}

const char *mode_name(WriterMode mode) {
    return mode == WriterMode::kWrongDestinationResidualStomp
               ? "wrong_destination_residual_stomp"
               : "lossless_output_substitution";
}

void log_prepare_failure(const char *reason, const TensorMetadata &metadata,
                         int32_t device, int32_t status) {
    char line[1536];
    const int written = std::snprintf(
        line, sizeof(line),
        "{\"event\":\"fault26_fia_lossless_prepare_failed\","
        "\"pid\":%ld,\"tid\":%ld,\"ts_ns\":%" PRIu64 ","
        "\"device\":%d,\"reason\":\"%s\",\"status\":%d,"
        "\"output_bytes\":%" PRIu64 ",\"device_memory_modified\":false}",
        static_cast<long>(getpid()),
        static_cast<long>(syscall(SYS_gettid)), monotonic_ns(), device, reason,
        status, metadata.bytes);
    if (written > 0 && static_cast<size_t>(written) < sizeof(line)) {
        write_line(line, static_cast<size_t>(written));
    }
}

void log_enabled_candidate(uint64_t prepare_index, uint64_t enabled_index,
                           const TensorMetadata &metadata, int32_t device,
                           uint64_t target_device, uint64_t target_bytes,
                           uint64_t target_occurrence, uint64_t max_bytes,
                           WriterMode mode) {
    const uint64_t log_limit = env_u64(
        "FAULT26_FIA_WRITER_PROXY_CANDIDATE_LOG_LIMIT", 64, 1, 1000000);
    if (enabled_index > log_limit) {
        return;
    }
    char line[2048];
    const int written = std::snprintf(
        line, sizeof(line),
        "{\"event\":\"fault26_fia_lossless_candidate\","
        "\"pid\":%ld,\"tid\":%ld,\"ts_ns\":%" PRIu64 ","
        "\"prepare_index\":%" PRIu64 ",\"enabled_index\":%" PRIu64 ","
        "\"device\":%d,\"mode\":\"%s\",\"metadata_ok\":%s,"
        "\"contiguous\":%s,"
        "\"output_address\":\"%p\",\"output_bytes\":%" PRIu64 ","
        "\"dtype\":%d,\"format\":%d,\"view_offset\":%" PRId64 ","
        "\"target_device\":%" PRIu64 ",\"target_bytes\":%" PRIu64 ","
        "\"target_occurrence\":%" PRIu64 ",\"max_bytes\":%" PRIu64 ","
        "\"device_memory_modified\":false}",
        static_cast<long>(getpid()),
        static_cast<long>(syscall(SYS_gettid)), monotonic_ns(), prepare_index,
        enabled_index, device, mode_name(mode),
        metadata.ok ? "true" : "false",
        metadata.contiguous ? "true" : "false", metadata.address,
        metadata.bytes, static_cast<int>(metadata.dtype),
        static_cast<int>(metadata.format), metadata.offset, target_device,
        target_bytes, target_occurrence, max_bytes);
    if (written > 0 && static_cast<size_t>(written) < sizeof(line)) {
        write_line(line, static_cast<size_t>(written));
    }
}

void log_wrong_destination(const char *event, const Record &record,
                           aclnnStatus real_status, bool pass,
                           bool consumer_executed) {
    char line[4096];
    const int written = std::snprintf(
        line, sizeof(line),
        "{\"event\":\"%s\",\"pid\":%ld,\"tid\":%ld,"
        "\"ts_ns\":%" PRIu64 ",\"session\":%" PRIu64 ","
        "\"matching_index\":%" PRIu64 ",\"device\":%d,"
        "\"mode\":\"wrong_destination_residual_stomp\","
        "\"api\":\"aclnnFusedInferAttentionScoreV3\","
        "\"original_output_address\":\"%p\","
        "\"victim_address\":\"%p\",\"victim_offset_bytes\":%" PRIu64 ","
        "\"output_bytes\":%" PRIu64 ",\"victim_bytes\":%" PRIu64 ","
        "\"victim_producer_sequence\":%" PRIu64 ","
        "\"real_execute_status\":%d,\"pre_sync_status\":%d,"
        "\"host_alloc_status\":%d,\"victim_d2h_status\":%d,"
        "\"post_sync_status\":%d,\"victim_readback_status\":%d,"
        "\"output_copy_status\":%d,\"output_readback_status\":%d,"
        "\"destroy_tensor_status\":%d,"
        "\"restore_status\":%d,\"restore_readback_status\":%d,"
        "\"host_free_status\":%d,"
        "\"victim_hash_before\":\"0x%016" PRIx64 "\","
        "\"victim_hash_after\":\"0x%016" PRIx64 "\","
        "\"victim_hash_restored\":\"0x%016" PRIx64 "\","
        "\"output_hash_after\":\"0x%016" PRIx64 "\","
        "\"pre_execute_ready\":%s,\"victim_changed\":%s,"
        "\"output_copy_equal\":%s,\"consumer_matched\":%s,"
        "\"consumer_executed\":%s,\"real_kernel_wrote_victim\":%s,"
        "\"pass\":%s}",
        event, static_cast<long>(getpid()),
        static_cast<long>(syscall(SYS_gettid)), monotonic_ns(), record.session,
        record.matching_index, record.device, record.metadata.address,
        record.victim.address, record.victim_offset_bytes,
        record.metadata.bytes, record.victim.bytes,
        record.victim_producer_sequence, static_cast<int>(real_status),
        record.pre_sync_status, record.host_alloc_status,
        record.victim_d2h_status, record.post_sync_status,
        record.victim_readback_status, record.output_copy_status,
        record.output_readback_status, record.destroy_tensor_status,
        record.restore_status,
        record.restore_readback_status, record.host_free_status,
        record.victim_hash_before, record.victim_hash_after,
        record.victim_hash_restored, record.output_hash_after,
        record.pre_execute_ready ? "true" : "false",
        record.victim_changed ? "true" : "false",
        record.output_copy_equal ? "true" : "false",
        record.consumer_matched ? "true" : "false",
        consumer_executed ? "true" : "false",
        record.pre_execute_ready && real_status == ACL_SUCCESS &&
                record.victim_changed
            ? "true"
            : "false",
        pass ? "true" : "false");
    if (written > 0 && static_cast<size_t>(written) < sizeof(line)) {
        write_line(line, static_cast<size_t>(written));
    }
}

bool restore_wrong_destination(Record *record, aclrtStream stream,
                               aclnnStatus consumer_status,
                               bool consumer_executed) {
    if (record == nullptr || record->victim_backup == nullptr ||
        record->victim.bytes == 0) {
        return false;
    }
    const int32_t sync_status =
        static_cast<int32_t>(g_synchronize_stream(stream));
    if (sync_status == ACL_SUCCESS) {
        record->restore_status = static_cast<int32_t>(g_memcpy(
            record->victim.address, record->victim.bytes,
            record->victim_backup, record->victim.bytes,
            ACL_MEMCPY_HOST_TO_DEVICE));
    }
    if (record->restore_status == ACL_SUCCESS &&
        record->victim_backup_bytes >= record->victim.bytes * 2) {
        auto *scratch = static_cast<unsigned char *>(record->victim_backup) +
                        record->victim.bytes;
        std::memset(scratch, 0, static_cast<size_t>(record->victim.bytes));
        record->restore_readback_status = static_cast<int32_t>(g_memcpy(
            scratch, record->victim.bytes, record->victim.address,
            record->victim.bytes, ACL_MEMCPY_DEVICE_TO_HOST));
        if (record->restore_readback_status == ACL_SUCCESS) {
            record->victim_hash_restored = fnv1a64_bytes(
                scratch, static_cast<size_t>(record->victim.bytes));
        }
    }
    record->host_free_status =
        static_cast<int32_t>(g_free_host(record->victim_backup));
    record->victim_backup = nullptr;
    const bool pass = consumer_executed && consumer_status == ACL_SUCCESS &&
                      record->consumer_matched &&
                      record->restore_status == ACL_SUCCESS &&
                      record->restore_readback_status == ACL_SUCCESS &&
                      record->host_free_status == ACL_SUCCESS &&
                      record->victim_hash_restored == record->victim_hash_before;
    log_wrong_destination("fault26_fia_wrong_destination_restored", *record,
                          consumer_status, pass, consumer_executed);
    return pass;
}

}  // namespace

#ifdef DEMONCAT_NATIVE_PROVIDER_BRIDGE
#define fault26_fia_writer_prepare_output dcat26_fia_prepare_impl
#endif
extern "C" const aclTensor *fault26_fia_writer_prepare_output(
    const aclTensor *attention_out) {
    pthread_once(&g_once, initialize);
    if (g_prepared.active) {
        return attention_out;
    }
    g_prepared = Record{};
    const uint64_t prepare_index =
        g_prepare_calls.fetch_add(1, std::memory_order_relaxed) + 1;
    if (!g_allowed || !env_is_one("FAULT26_FIA_WRITER_PROXY_ENABLED")) {
        return attention_out;
    }
    const uint64_t session = env_u64(
        "FAULT26_FIA_WRITER_PROXY_SESSION", 0, 1,
        std::numeric_limits<uint64_t>::max());
    if (session == 0) {
        return attention_out;
    }
    int32_t device = -1;
    if (g_get_device(&device) != ACL_SUCCESS) {
        return attention_out;
    }
    const TensorMetadata metadata = capture_tensor(attention_out);
    const uint64_t target_device = env_u64(
        "FAULT26_FIA_WRITER_PROXY_TARGET_DEVICE", 0, 0, 65535);
    const uint64_t target_bytes = env_u64(
        "FAULT26_FIA_WRITER_PROXY_TARGET_BYTES", 0, 1,
        std::numeric_limits<uint64_t>::max());
    const uint64_t target_occurrence = env_u64(
        "FAULT26_FIA_WRITER_PROXY_TARGET_OCCURRENCE", 1, 1, 1000000);
    const bool all_hits = env_is_one("FAULT26_FIA_WRITER_PROXY_ALL_HITS");
    const uint64_t max_hits = env_u64(
        "FAULT26_FIA_WRITER_PROXY_MAX_HITS", 1, 1, 1000000);
    const uint64_t max_bytes = env_u64(
        "FAULT26_FIA_WRITER_PROXY_MAX_BYTES", kDefaultMaxBytes, 1,
        1073741824ULL);
    const WriterMode mode = requested_mode();
    const uint64_t enabled_index =
        g_enabled_prepare_calls.fetch_add(1, std::memory_order_relaxed) + 1;
    log_enabled_candidate(prepare_index, enabled_index, metadata, device,
                          target_device, target_bytes, target_occurrence,
                          max_bytes, mode);
    if (!metadata.ok || static_cast<uint64_t>(device) != target_device ||
        metadata.bytes != target_bytes || metadata.bytes > max_bytes) {
        return attention_out;
    }
    Record record{};
    record.active = true;
    record.session = session;
    record.device = device;
    record.metadata = metadata;
    record.original_tensor = attention_out;
    record.mode = mode;
    void *replacement_address = nullptr;
    if (mode == WriterMode::kLosslessOutputSubstitution) {
        std::lock_guard<std::mutex> lock(g_control_mutex);
        if (g_session != session) {
            g_session = session;
            g_matching_count = 0;
            g_hit_count = 0;
            g_hit_claimed = false;
            g_min_residual_sequence = 0;
        }
    }
    if (mode == WriterMode::kWrongDestinationResidualStomp) {
        ResidualRecord residual{};
        bool has_residual = false;
        bool has_pending_stomp = false;
        {
            std::lock_guard<std::mutex> lock(g_lifecycle_mutex);
            const auto found = g_current_residuals.find(device);
            if (found != g_current_residuals.end()) {
                residual = found->second;
                has_residual = true;
            }
            has_pending_stomp =
                g_pending_stomps.find(device) != g_pending_stomps.end();
        }
        const uint64_t target_residual_bytes = env_u64(
            "FAULT26_FIA_WRITER_PROXY_TARGET_RESIDUAL_BYTES", 0, 1,
            1073741824ULL);
        const uint64_t victim_offset = env_u64(
            "FAULT26_FIA_WRITER_PROXY_VICTIM_OFFSET_BYTES", 0, 0,
            1073741824ULL);
        bool new_session = false;
        {
            std::lock_guard<std::mutex> lock(g_control_mutex);
            if (g_session != session) {
                g_session = session;
                g_matching_count = 0;
                g_hit_count = 0;
                g_hit_claimed = false;
                g_min_residual_sequence =
                    has_residual ? residual.producer_sequence : 0;
                new_session = true;
            }
        }
        if (new_session || has_pending_stomp || !has_residual ||
            !residual.tensor.ok ||
            residual.producer_sequence <= g_min_residual_sequence ||
            residual.tensor.bytes != target_residual_bytes ||
            residual.tensor.dtype != metadata.dtype ||
            residual.tensor.address == metadata.address ||
            victim_offset > residual.tensor.bytes ||
            metadata.bytes > residual.tensor.bytes - victim_offset ||
            residual.tensor.bytes > max_bytes) {
            return attention_out;
        }
        record.victim = residual.tensor;
        record.victim_producer_sequence = residual.producer_sequence;
        record.victim_offset_bytes = victim_offset;
        replacement_address =
            static_cast<unsigned char *>(record.victim.address) + victim_offset;
    } else {
        record.malloc_status = static_cast<int32_t>(g_malloc(
            &record.temporary, static_cast<size_t>(metadata.bytes),
            ACL_MEM_MALLOC_HUGE_FIRST));
        if (record.malloc_status != ACL_SUCCESS ||
            record.temporary == nullptr ||
            record.temporary == metadata.address) {
            log_prepare_failure("temporary_allocation_failed_or_aliased",
                                metadata, device, record.malloc_status);
            cleanup_record(&record);
            return attention_out;
        }
        replacement_address = record.temporary;
    }
    {
        std::lock_guard<std::mutex> lock(g_control_mutex);
        if (g_session != session || (!all_hits && g_hit_claimed) ||
            (all_hits && g_hit_count >= max_hits)) {
            cleanup_record(&record);
            return attention_out;
        }
        record.matching_index = ++g_matching_count;
        if (!all_hits && record.matching_index != target_occurrence) {
            cleanup_record(&record);
            return attention_out;
        }
        ++g_hit_count;
        if (!all_hits || g_hit_count >= max_hits) {
            g_hit_claimed = true;
            (void)setenv("FAULT26_FIA_WRITER_PROXY_ENABLED", "0", 1);
        }
    }
    record.replacement_tensor = g_create_tensor(
        metadata.view_dims.data(), metadata.view_dims.size(), metadata.dtype,
        metadata.strides.data(), metadata.offset, metadata.format,
        metadata.storage_dims.data(), metadata.storage_dims.size(),
        replacement_address);
    if (record.replacement_tensor == nullptr) {
        log_prepare_failure("aclCreateTensor_failed", metadata, device,
                            kNotCalled);
        cleanup_record(&record);
        return attention_out;
    }
    g_prepared = std::move(record);
    return g_prepared.replacement_tensor;
}

extern "C" void fault26_fia_writer_gws_complete(
    const aclTensor *original, const aclTensor *effective,
    aclOpExecutor *executor, aclnnStatus status) {
    pthread_once(&g_once, initialize);
    if (!g_prepared.active) {
        return;
    }
    if (original != g_prepared.original_tensor ||
        effective != g_prepared.replacement_tensor) {
        return;
    }
    if (status != ACL_SUCCESS || executor == nullptr) {
        log_prepare_failure("real_gws_failed_or_binding_mismatch",
                            g_prepared.metadata, g_prepared.device,
                            static_cast<int32_t>(status));
        cleanup_record(&g_prepared);
        return;
    }
    try {
        std::lock_guard<std::mutex> lock(g_records_mutex);
        g_records[executor].push_back(std::move(g_prepared));
        g_prepared = Record{};
    } catch (...) {
        log_prepare_failure("executor_registry_failed", g_prepared.metadata,
                            g_prepared.device, kNotCalled);
        cleanup_record(&g_prepared);
    }
}

extern "C" bool fault26_fia_writer_before_execute(
    aclOpExecutor *executor, aclrtStream stream) {
    pthread_once(&g_once, initialize);
    std::lock_guard<std::mutex> lock(g_records_mutex);
    const auto found = g_records.find(executor);
    if (found == g_records.end() || found->second.empty()) {
        return true;
    }
    Record &record = found->second.front();
    if (record.mode != WriterMode::kWrongDestinationResidualStomp) {
        return true;
    }
    if (record.pre_execute_ready) {
        return false;
    }
    if (record.victim.bytes == 0 ||
        record.victim.bytes > std::numeric_limits<uint64_t>::max() / 2) {
        return false;
    }
    record.pre_sync_status =
        static_cast<int32_t>(g_synchronize_stream(stream));
    if (record.pre_sync_status != ACL_SUCCESS) {
        return false;
    }
    record.victim_backup_bytes = record.victim.bytes * 2;
    record.host_alloc_status = static_cast<int32_t>(g_malloc_host(
        &record.victim_backup,
        static_cast<size_t>(record.victim_backup_bytes)));
    if (record.host_alloc_status != ACL_SUCCESS ||
        record.victim_backup == nullptr) {
        return false;
    }
    record.victim_d2h_status = static_cast<int32_t>(g_memcpy(
        record.victim_backup, record.victim.bytes, record.victim.address,
        record.victim.bytes, ACL_MEMCPY_DEVICE_TO_HOST));
    if (record.victim_d2h_status != ACL_SUCCESS) {
        return false;
    }
    record.victim_hash_before = fnv1a64_bytes(
        static_cast<unsigned char *>(record.victim_backup),
        static_cast<size_t>(record.victim.bytes));
    record.pre_execute_ready = true;
    return true;
}

extern "C" void fault26_fia_writer_after_execute(
    aclOpExecutor *executor, aclrtStream stream, aclnnStatus real_status) {
    pthread_once(&g_once, initialize);
    Record record{};
    {
        std::lock_guard<std::mutex> lock(g_records_mutex);
        const auto found = g_records.find(executor);
        if (found == g_records.end() || found->second.empty()) {
            return;
        }
        record = std::move(found->second.front());
        found->second.pop_front();
        if (found->second.empty()) {
            g_records.erase(found);
        }
    }
    if (record.mode == WriterMode::kWrongDestinationResidualStomp) {
        uint64_t victim_output_hash = 0;
        if (record.pre_execute_ready && real_status == ACL_SUCCESS) {
            record.post_sync_status =
                static_cast<int32_t>(g_synchronize_stream(stream));
        }
        auto *scratch = record.victim_backup == nullptr
                            ? nullptr
                            : static_cast<unsigned char *>(record.victim_backup) +
                                  record.victim.bytes;
        if (record.post_sync_status == ACL_SUCCESS && scratch != nullptr) {
            std::memset(scratch, 0,
                        static_cast<size_t>(record.victim.bytes));
            record.victim_readback_status = static_cast<int32_t>(g_memcpy(
                scratch, record.victim.bytes, record.victim.address,
                record.victim.bytes, ACL_MEMCPY_DEVICE_TO_HOST));
        }
        if (record.victim_readback_status == ACL_SUCCESS) {
            record.victim_hash_after = fnv1a64_bytes(
                scratch, static_cast<size_t>(record.victim.bytes));
            record.victim_changed =
                record.victim_hash_after != record.victim_hash_before;
            victim_output_hash = fnv1a64_bytes(
                scratch + record.victim_offset_bytes,
                static_cast<size_t>(record.metadata.bytes));
            record.output_copy_status = static_cast<int32_t>(g_memcpy(
                record.metadata.address, record.metadata.bytes,
                static_cast<unsigned char *>(record.victim.address) +
                    record.victim_offset_bytes,
                record.metadata.bytes, ACL_MEMCPY_DEVICE_TO_DEVICE));
        }
        if (record.output_copy_status == ACL_SUCCESS && scratch != nullptr) {
            std::memset(scratch, 0,
                        static_cast<size_t>(record.metadata.bytes));
            record.output_readback_status = static_cast<int32_t>(g_memcpy(
                scratch, record.metadata.bytes, record.metadata.address,
                record.metadata.bytes, ACL_MEMCPY_DEVICE_TO_HOST));
        }
        if (record.output_readback_status == ACL_SUCCESS) {
            record.output_hash_after = fnv1a64_bytes(
                scratch, static_cast<size_t>(record.metadata.bytes));
            record.output_copy_equal =
                record.output_hash_after == victim_output_hash;
        }
        if (record.replacement_tensor != nullptr) {
            record.destroy_tensor_status = static_cast<int32_t>(
                g_destroy_tensor(record.replacement_tensor));
            record.replacement_tensor = nullptr;
        }
        const bool applied =
            record.pre_execute_ready && real_status == ACL_SUCCESS &&
            record.post_sync_status == ACL_SUCCESS &&
            record.victim_readback_status == ACL_SUCCESS &&
            record.victim_changed &&
            record.output_copy_status == ACL_SUCCESS &&
            record.output_readback_status == ACL_SUCCESS &&
            record.output_copy_equal &&
            record.destroy_tensor_status == ACL_SUCCESS;
        if (!applied) {
            (void)restore_wrong_destination(&record, stream, real_status,
                                            false);
            cleanup_record(&record);
            return;
        }
        bool inserted = false;
        const int32_t pending_device = record.device;
        {
            std::lock_guard<std::mutex> lock(g_lifecycle_mutex);
            if (g_pending_stomps.find(pending_device) ==
                g_pending_stomps.end()) {
                g_pending_stomps.emplace(pending_device, std::move(record));
                inserted = true;
            }
        }
        if (!inserted) {
            (void)restore_wrong_destination(&record, stream, real_status,
                                            false);
            cleanup_record(&record);
            return;
        }
        {
            std::lock_guard<std::mutex> lock(g_lifecycle_mutex);
            const auto pending = g_pending_stomps.find(pending_device);
            if (pending != g_pending_stomps.end()) {
                log_wrong_destination(
                    "fault26_fia_wrong_destination_applied",
                    pending->second, real_status, true, false);
            }
        }
        return;
    }
    const uint64_t begin_ns = monotonic_ns();
    int32_t sync_status = kNotCalled;
    int32_t copy_status = kNotCalled;
    int32_t temp_d2h_status = kNotCalled;
    int32_t original_d2h_status = kNotCalled;
    int32_t destroy_status = kNotCalled;
    int32_t free_status = kNotCalled;
    uint64_t temporary_hash = 0;
    uint64_t original_hash = 0;
    bool verified_equal = false;
    void *const temporary_address = record.temporary;
    std::vector<unsigned char> temporary_host;
    std::vector<unsigned char> original_host;
    if (real_status == ACL_SUCCESS) {
        sync_status = static_cast<int32_t>(g_synchronize_stream(stream));
    }
    if (real_status == ACL_SUCCESS && sync_status == ACL_SUCCESS) {
        copy_status = static_cast<int32_t>(g_memcpy(
            record.metadata.address, record.metadata.bytes, record.temporary,
            record.metadata.bytes, ACL_MEMCPY_DEVICE_TO_DEVICE));
    }
    if (copy_status == ACL_SUCCESS) {
        try {
            temporary_host.resize(static_cast<size_t>(record.metadata.bytes));
            original_host.resize(static_cast<size_t>(record.metadata.bytes));
            temp_d2h_status = static_cast<int32_t>(g_memcpy(
                temporary_host.data(), temporary_host.size(), record.temporary,
                temporary_host.size(), ACL_MEMCPY_DEVICE_TO_HOST));
            original_d2h_status = static_cast<int32_t>(g_memcpy(
                original_host.data(), original_host.size(),
                record.metadata.address, original_host.size(),
                ACL_MEMCPY_DEVICE_TO_HOST));
            if (temp_d2h_status == ACL_SUCCESS &&
                original_d2h_status == ACL_SUCCESS) {
                temporary_hash = fnv1a64(temporary_host);
                original_hash = fnv1a64(original_host);
                verified_equal = temporary_host == original_host;
            }
        } catch (...) {
            temp_d2h_status = kNotCalled;
            original_d2h_status = kNotCalled;
        }
    }
    if (record.replacement_tensor != nullptr) {
        destroy_status =
            static_cast<int32_t>(g_destroy_tensor(record.replacement_tensor));
        record.replacement_tensor = nullptr;
    }
    if (record.temporary != nullptr) {
        free_status = static_cast<int32_t>(g_free(record.temporary));
        record.temporary = nullptr;
    }
    const bool pass =
        real_status == ACL_SUCCESS && sync_status == ACL_SUCCESS &&
        copy_status == ACL_SUCCESS && temp_d2h_status == ACL_SUCCESS &&
        original_d2h_status == ACL_SUCCESS && verified_equal &&
        destroy_status == ACL_SUCCESS && free_status == ACL_SUCCESS;
    char line[4096];
    const int written = std::snprintf(
        line, sizeof(line),
        "{\"event\":\"fault26_fia_lossless_output_substitution\","
        "\"pid\":%ld,\"tid\":%ld,\"ts_ns\":%" PRIu64 ","
        "\"session\":%" PRIu64 ",\"matching_index\":%" PRIu64 ","
        "\"device\":%d,\"mode\":\"lossless_output_substitution\","
        "\"api\":\"aclnnFusedInferAttentionScoreV3\","
        "\"original_address\":\"%p\",\"temporary_address\":\"%p\","
        "\"output_bytes\":%" PRIu64 ",\"dtype\":%d,\"format\":%d,"
        "\"real_execute_status\":%d,\"stream_sync_status\":%d,"
        "\"d2d_copy_status\":%d,\"temporary_d2h_status\":%d,"
        "\"original_d2h_status\":%d,\"destroy_tensor_status\":%d,"
        "\"device_free_status\":%d,"
        "\"temporary_hash\":\"0x%016" PRIx64 "\","
        "\"original_hash\":\"0x%016" PRIx64 "\","
        "\"readback_equal\":%s,\"pass\":%s,"
        "\"kernel_wrote_replacement_binding\":%s,"
        "\"residual_modified\":false,\"device_memory_modified\":true,"
        "\"elapsed_ns\":%" PRIu64 "}",
        static_cast<long>(getpid()),
        static_cast<long>(syscall(SYS_gettid)), monotonic_ns(), record.session,
        record.matching_index, record.device, record.metadata.address,
        temporary_address, record.metadata.bytes,
        static_cast<int>(record.metadata.dtype),
        static_cast<int>(record.metadata.format), static_cast<int>(real_status),
        sync_status, copy_status, temp_d2h_status, original_d2h_status,
        destroy_status, free_status, temporary_hash, original_hash,
        verified_equal ? "true" : "false", pass ? "true" : "false",
        real_status == ACL_SUCCESS ? "true" : "false",
        monotonic_ns() - begin_ns);
    if (written > 0 && static_cast<size_t>(written) < sizeof(line)) {
        write_line(line, static_cast<size_t>(written));
    }
}

extern "C" void fault26_write_gate_add_rms_norm_bias_gws(
    const aclTensor *, const aclTensor *x2, const aclTensor *,
    const aclTensor *, double, const aclTensor *, const aclTensor *,
    const aclTensor *x_out, uint64_t, aclOpExecutor *executor,
    aclnnStatus status) {
    pthread_once(&g_once, initialize);
    if (!g_allowed || status != ACL_SUCCESS || executor == nullptr) {
        return;
    }
    int32_t device = -1;
    if (g_get_device(&device) != ACL_SUCCESS) {
        return;
    }
    AddRmsRecord record{};
    record.executor = executor;
    record.device = device;
    record.sequence =
        g_add_rms_sequence.fetch_add(1, std::memory_order_relaxed) + 1;
    record.x2 = capture_tensor(x2);
    record.x_out = capture_tensor(x_out);
    if (!record.x2.ok || !record.x_out.ok) {
        return;
    }
    try {
        std::lock_guard<std::mutex> lock(g_lifecycle_mutex);
        g_add_rms_queues[executor].push_back(std::move(record));
    } catch (...) {
        (void)setenv("FAULT26_FIA_WRITER_PROXY_ENABLED", "0", 1);
    }
}

extern "C" void fault26_write_gate_before_add_rms_norm_bias(
    void *, uint64_t, aclOpExecutor *executor, aclrtStream stream) {
    pthread_once(&g_once, initialize);
    g_active_add_rms = ActiveAddRms{};
    Record mismatch{};
    bool restore_mismatch = false;
    {
        std::lock_guard<std::mutex> lock(g_lifecycle_mutex);
        const auto queue = g_add_rms_queues.find(executor);
        if (queue == g_add_rms_queues.end() || queue->second.empty()) {
            return;
        }
        g_active_add_rms.valid = true;
        g_active_add_rms.record = std::move(queue->second.front());
        queue->second.pop_front();
        if (queue->second.empty()) {
            g_add_rms_queues.erase(queue);
        }
        const int32_t device = g_active_add_rms.record.device;
        const auto pending = g_pending_stomps.find(device);
        if (pending != g_pending_stomps.end()) {
            const bool victim_is_x2 =
                same_storage(pending->second.victim,
                             g_active_add_rms.record.x2);
            const bool x_out_is_distinct =
                !same_storage(pending->second.victim,
                              g_active_add_rms.record.x_out);
            if (victim_is_x2 && x_out_is_distinct) {
                pending->second.consumer_matched = true;
                g_active_add_rms.consumes_pending = true;
            } else {
                mismatch = std::move(pending->second);
                g_pending_stomps.erase(pending);
                restore_mismatch = true;
            }
        }
    }
    if (restore_mismatch) {
        (void)restore_wrong_destination(&mismatch, stream,
                                        static_cast<aclnnStatus>(1), false);
        cleanup_record(&mismatch);
    }
}

extern "C" void fault26_write_gate_after_add_rms_norm_bias(
    aclnnStatus real_status, aclrtStream stream) {
    pthread_once(&g_once, initialize);
    if (!g_active_add_rms.valid) {
        return;
    }
    Record pending{};
    bool restore = false;
    {
        std::lock_guard<std::mutex> lock(g_lifecycle_mutex);
        const int32_t device = g_active_add_rms.record.device;
        if (g_active_add_rms.consumes_pending) {
            const auto found = g_pending_stomps.find(device);
            if (found != g_pending_stomps.end()) {
                pending = std::move(found->second);
                g_pending_stomps.erase(found);
                restore = true;
            }
        }
        if (real_status == ACL_SUCCESS &&
            g_active_add_rms.record.x_out.ok) {
            ResidualRecord next{};
            next.tensor = g_active_add_rms.record.x_out;
            next.producer_sequence = g_active_add_rms.record.sequence;
            g_current_residuals[device] = std::move(next);
        } else {
            g_current_residuals.erase(device);
        }
    }
    if (restore) {
        (void)restore_wrong_destination(&pending, stream, real_status, true);
        cleanup_record(&pending);
    }
    g_active_add_rms = ActiveAddRms{};
}

#ifdef DEMONCAT_NATIVE_PROVIDER_BRIDGE
#include "../../../fault_injection/native_adapters/legacy26_fia_bridge.inc"
#endif

