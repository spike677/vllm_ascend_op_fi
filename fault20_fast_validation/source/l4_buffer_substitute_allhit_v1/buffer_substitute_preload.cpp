#include "fault20_buffer_api.h"
#include "int_equivalent_transform.h"
#include "mantissa_truncate.h"

#include <acl/acl.h>
#include <aclnn/opdev/op_cache.h>

#include <atomic>
#include <cinttypes>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <fcntl.h>
#include <limits.h>
#include <mutex>
#include <new>
#include <limits>
#include <pthread.h>
#include <string>
#include <sys/file.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/syscall.h>
#include <sys/uio.h>
#include <time.h>
#include <unistd.h>
#include <unordered_map>
#include <vector>

#if !defined(__aarch64__)
#error "Fault20 L4 buffer substitution is version-locked to aarch64"
#endif

#if !defined(_GLIBCXX_USE_CXX11_ABI) || _GLIBCXX_USE_CXX11_ABI != 0
#error "Fault20 L4 buffer substitution requires _GLIBCXX_USE_CXX11_ABI=0"
#endif

namespace {

constexpr const char *kDefaultLogPath =
    "/fault20/runtime/l4_buffer_substitute_logs/buffer_substitute.jsonl";
constexpr const char *kRunSymbol =
    "_ZN2op8internal11OpExecCache3RunEPvS2_RKSt6vectorIS2_SaIS2_EE";
constexpr const char *kAddSymbol =
    "_ZN2op8internal14AddOpExecCacheEPNS0_11OpExecCacheE";
constexpr const char *kAddParamConstTensorSymbol =
    "_ZN2op8internal13AddParamToBufEPK9aclTensor";
constexpr const char *kAddParamTensorSymbol =
    "_ZN2op8internal13AddParamToBufEP9aclTensor";
constexpr const char *kGetLogApiInfoSymbol = "_ZN2op8internal13GetLogApiInfoEv";
constexpr const char *kExpectedLibraryPathFragment = "/8.3.RC2/";

using RunFn = aclnnStatus (*)(op::internal::OpExecCache *, void *, aclrtStream,
                              const std::vector<void *> &);
using AddFn = bool (*)(op::internal::OpExecCache *);
using AddParamConstTensorFn = void (*)(const aclTensor *);
using AddParamTensorFn = void (*)(aclTensor *);
using GetLogApiInfoFn = std::string (*)();
using GetDeviceFn = aclError (*)(int32_t *);
using MallocFn = aclError (*)(void **, size_t, aclrtMemMallocPolicy);
using FreeFn = aclError (*)(void *);
using MallocHostFn = aclError (*)(void **, size_t);
using FreeHostFn = aclError (*)(void *);
using MemcpyFn = aclError (*)(void *, size_t, const void *, size_t, aclrtMemcpyKind);
using MemcpyAsyncFn = aclError (*)(void *, size_t, const void *, size_t, aclrtMemcpyKind,
                                   aclrtStream);
using SynchronizeStreamFn = aclError (*)(aclrtStream);

struct RegisteredMetadata {
    Fault20X1Metadata x1{};
    int32_t device{-1};
};

struct RegisteredMatmulMetadata {
    Fault20MatmulMetadata tensors{};
    int32_t device{-1};
    uint64_t run_count{0};
    uint32_t tensor_param_count{0};
    uintptr_t tensor_params[8]{};
};

enum class CaptureKind : uint32_t {
    kNone = 0,
    kQmm = 1,
    kMatmulWeightNz = 2,
};

struct CaptureState {
    bool active{false};
    CaptureKind kind{CaptureKind::kNone};
    uint32_t add_count{0};
    op::internal::OpExecCache *cache{nullptr};
    uint32_t tensor_param_count{0};
    uintptr_t tensor_params[8]{};
};

struct QuantProducerState {
    bool valid{false};
    void *output{nullptr};
};

constexpr int32_t kStatusSuccess = 1;
constexpr int32_t kStatusVectorCopyFailed = -1;
constexpr int32_t kStatusPreSyncFailed = -2;
constexpr int32_t kStatusHostAllocFailed = -3;
constexpr int32_t kStatusD2HFailed = -4;
constexpr int32_t kStatusNoChangedElements = -5;
constexpr int32_t kStatusDeviceAllocFailed = -6;
constexpr int32_t kStatusH2DFailed = -7;
constexpr int32_t kStatusRunFailed = -8;
constexpr int32_t kStatusPostSyncFailed = -9;
constexpr int32_t kStatusDeviceFreeFailed = -10;
constexpr int32_t kStatusHostFreeFailed = -11;
constexpr int32_t kNotCalled = -2147483647;

struct Int4Attempt {
    uint32_t hit_index{0};
    uint32_t candidate_index{0};
    uint32_t quant_step{16};
    int32_t result_status{0};
    int32_t pre_sync_status{kNotCalled};
    int32_t host_malloc_status{kNotCalled};
    int32_t d2h_status{kNotCalled};
    int32_t device_malloc_status{kNotCalled};
    int32_t h2d_status{kNotCalled};
    int32_t run_status{kNotCalled};
    int32_t post_sync_status{kNotCalled};
    int32_t device_free_status{kNotCalled};
    int32_t host_free_status{kNotCalled};
    void *original{nullptr};
    void *temporary{nullptr};
    void *host_buffer{nullptr};
    bool temporary_used{false};
    bool temporary_freed{false};
    bool host_freed{false};
    bool stats_valid{false};
    uint64_t changed_elements{0};
    uint64_t unchanged_elements{0};
    uint64_t zero_before{0};
    uint64_t zero_after{0};
    uint64_t int2_histogram[4]{};
    int32_t min_before{0};
    int32_t max_before{0};
    int32_t min_after{0};
    int32_t max_after{0};
    uint64_t begin_ns{0};
    uint64_t pre_sync_ns{0};
    uint64_t host_alloc_ns{0};
    uint64_t d2h_ns{0};
    uint64_t cpu_transform_ns{0};
    uint64_t device_alloc_ns{0};
    uint64_t h2d_enqueue_ns{0};
    uint64_t real_run_ns{0};
    uint64_t post_sync_ns{0};
};

struct MantissaAttempt {
    uint32_t hit_index{0};
    uint32_t candidate_index{0};
    uint32_t keep_mantissa_bits{0};
    int32_t result_status{0};
    int32_t pre_sync_status{kNotCalled};
    int32_t host_malloc_status{kNotCalled};
    int32_t d2h_status{kNotCalled};
    int32_t device_malloc_status{kNotCalled};
    int32_t h2d_status{kNotCalled};
    int32_t run_status{kNotCalled};
    int32_t post_sync_status{kNotCalled};
    int32_t device_free_status{kNotCalled};
    int32_t host_free_status{kNotCalled};
    void *original{nullptr};
    void *temporary{nullptr};
    void *host_buffer{nullptr};
    bool temporary_used{false};
    bool temporary_freed{false};
    bool host_freed{false};
    bool stats_valid{false};
    uint64_t changed_elements{0};
    uint64_t unchanged_elements{0};
    uint64_t nonfinite_before{0};
    uint64_t nonfinite_after{0};
    uint64_t zero_before{0};
    uint64_t zero_after{0};
    float min_before{0.0F};
    float max_before{0.0F};
    float min_after{0.0F};
    float max_after{0.0F};
    float max_abs_error{0.0F};
    uint64_t begin_ns{0};
    uint64_t pre_sync_ns{0};
    uint64_t host_alloc_ns{0};
    uint64_t d2h_ns{0};
    uint64_t cpu_transform_ns{0};
    uint64_t device_alloc_ns{0};
    uint64_t h2d_enqueue_ns{0};
    uint64_t real_run_ns{0};
    uint64_t post_sync_ns{0};
};

pthread_once_t g_resolve_once = PTHREAD_ONCE_INIT;
pthread_once_t g_control_once = PTHREAD_ONCE_INIT;
pthread_mutex_t g_runtime_mutex = PTHREAD_MUTEX_INITIALIZER;
RunFn g_real_run = nullptr;
AddFn g_real_add = nullptr;
AddParamConstTensorFn g_real_add_param_const_tensor = nullptr;
AddParamTensorFn g_real_add_param_tensor = nullptr;
GetLogApiInfoFn g_get_log_api_info = nullptr;
GetDeviceFn g_get_device = nullptr;
MallocFn g_malloc = nullptr;
FreeFn g_free = nullptr;
MallocHostFn g_malloc_host = nullptr;
FreeHostFn g_free_host = nullptr;
MemcpyFn g_memcpy = nullptr;
MemcpyAsyncFn g_memcpy_async = nullptr;
SynchronizeStreamFn g_synchronize_stream = nullptr;
void *g_nnopbase_handle = nullptr;
void *g_runtime_handle = nullptr;
Fault20Control *g_control = nullptr;
bool g_abi_ok = false;
bool g_runtime_resolution_logged = false;
thread_local CaptureState g_capture;
thread_local QuantProducerState g_last_quant;
thread_local bool g_add_param_forwarding{false};
thread_local bool g_matmul_substitution_active{false};

std::unordered_map<op::internal::OpExecCache *, RegisteredMetadata> &registry() {
    static auto *value =
        new std::unordered_map<op::internal::OpExecCache *, RegisteredMetadata>();
    return *value;
}

std::unordered_map<op::internal::OpExecCache *, RegisteredMatmulMetadata> &
matmul_registry() {
    static auto *value =
        new std::unordered_map<op::internal::OpExecCache *, RegisteredMatmulMetadata>();
    return *value;
}

std::mutex &registry_mutex() {
    static auto *value = new std::mutex();
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

void write_line(const char *line, size_t size) {
    const char *configured = std::getenv("FAULT20_BUFFER_LOG_PATH");
    const char *path = configured == nullptr || configured[0] == '\0' ? kDefaultLogPath : configured;
    const int fd = open(path, O_WRONLY | O_CREAT | O_APPEND | O_CLOEXEC, 0644);
    if (fd < 0) {
        return;
    }
    iovec parts[2] = {{const_cast<char *>(line), size}, {const_cast<char *>("\n"), 1}};
    const ssize_t written = writev(fd, parts, 2);
    (void)written;
    (void)close(fd);
}

void log_simple(const char *event, const char *reason) {
    char line[768];
    const int written = std::snprintf(
        line, sizeof(line),
        "{\"event\":\"%s\",\"pid\":%ld,\"tid\":%ld,\"ts_ns\":%" PRIu64
        ",\"reason\":\"%s\"}",
        event, static_cast<long>(getpid()), static_cast<long>(syscall(SYS_gettid)),
        monotonic_ns(), reason == nullptr ? "" : reason);
    if (written > 0 && static_cast<size_t>(written) < sizeof(line)) {
        write_line(line, static_cast<size_t>(written));
    }
}

void initialize_control() {
    const int fd = shm_open(FAULT20_CONTROL_SHM, O_RDWR | O_CREAT | O_CLOEXEC, 0600);
    if (fd < 0) {
        return;
    }
    if (flock(fd, LOCK_EX) != 0) {
        (void)close(fd);
        return;
    }
    bool sized = ftruncate(fd, static_cast<off_t>(sizeof(Fault20Control))) == 0;
    void *mapped = MAP_FAILED;
    if (sized) {
        mapped = mmap(nullptr, sizeof(Fault20Control), PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    }
    if (mapped != MAP_FAILED) {
        auto *control = static_cast<Fault20Control *>(mapped);
        if (control->magic != FAULT20_CONTROL_MAGIC ||
            control->version != FAULT20_CONTROL_VERSION) {
            std::memset(control, 0, sizeof(*control));
            control->magic = FAULT20_CONTROL_MAGIC;
            control->version = FAULT20_CONTROL_VERSION;
            control->mode = FAULT20_MODE_DISABLED;
            control->target_device = 0;
            control->target_m = 1;
            control->target_k = 5120;
            control->target_call_index = 1;
            control->max_hits = 1;
            control->quant_step = 16;
            control->keep_mantissa_bits = 3;
            control->target_dtype = ACL_BF16;
            control->max_bytes = 5120;
            control->target_device_mask = 1;
            control->max_hits_per_device = 1;
            control->selection_mode = FAULT20_SELECTION_BOUNDED;
            control->target_api = FAULT20_TARGET_API_NONE;
        }
        g_control = control;
    }
    (void)flock(fd, LOCK_UN);
    (void)close(fd);
}

void resolve_runtime_api() {
    if (g_get_device != nullptr && g_malloc != nullptr && g_free != nullptr &&
        g_malloc_host != nullptr && g_free_host != nullptr && g_memcpy != nullptr &&
        g_memcpy_async != nullptr && g_synchronize_stream != nullptr) {
        return;
    }
    (void)pthread_mutex_lock(&g_runtime_mutex);
    if (g_runtime_handle == nullptr) {
        g_runtime_handle = dlopen("libascendcl.so", RTLD_LAZY | RTLD_NOLOAD);
    }
    void *handle = g_runtime_handle == nullptr ? RTLD_DEFAULT : g_runtime_handle;
    if (g_get_device == nullptr) {
        g_get_device = reinterpret_cast<GetDeviceFn>(dlsym(handle, "aclrtGetDevice"));
    }
    if (g_malloc == nullptr) {
        g_malloc = reinterpret_cast<MallocFn>(dlsym(handle, "aclrtMalloc"));
    }
    if (g_free == nullptr) {
        g_free = reinterpret_cast<FreeFn>(dlsym(handle, "aclrtFree"));
    }
    if (g_malloc_host == nullptr) {
        g_malloc_host = reinterpret_cast<MallocHostFn>(dlsym(handle, "aclrtMallocHost"));
    }
    if (g_free_host == nullptr) {
        g_free_host = reinterpret_cast<FreeHostFn>(dlsym(handle, "aclrtFreeHost"));
    }
    if (g_memcpy == nullptr) {
        g_memcpy = reinterpret_cast<MemcpyFn>(dlsym(handle, "aclrtMemcpy"));
    }
    if (g_memcpy_async == nullptr) {
        g_memcpy_async =
            reinterpret_cast<MemcpyAsyncFn>(dlsym(handle, "aclrtMemcpyAsync"));
    }
    if (g_synchronize_stream == nullptr) {
        g_synchronize_stream = reinterpret_cast<SynchronizeStreamFn>(
            dlsym(handle, "aclrtSynchronizeStream"));
    }
    const bool ready = g_get_device != nullptr && g_malloc != nullptr && g_free != nullptr &&
                       g_malloc_host != nullptr && g_free_host != nullptr &&
                       g_memcpy != nullptr && g_memcpy_async != nullptr &&
                       g_synchronize_stream != nullptr;
    if (ready && !g_runtime_resolution_logged) {
        g_runtime_resolution_logged = true;
        log_simple("runtime_api_resolved", "libascendcl runtime API ready");
    }
    (void)pthread_mutex_unlock(&g_runtime_mutex);
}

void resolve_real() {
    g_real_run = reinterpret_cast<RunFn>(dlsym(RTLD_NEXT, kRunSymbol));
    g_real_add = reinterpret_cast<AddFn>(dlsym(RTLD_NEXT, kAddSymbol));
    g_real_add_param_const_tensor = reinterpret_cast<AddParamConstTensorFn>(
        dlsym(RTLD_NEXT, kAddParamConstTensorSymbol));
    g_real_add_param_tensor = reinterpret_cast<AddParamTensorFn>(
        dlsym(RTLD_NEXT, kAddParamTensorSymbol));
    g_nnopbase_handle = dlopen("libnnopbase.so", RTLD_LAZY | RTLD_NOLOAD);
    if (g_nnopbase_handle != nullptr) {
        if (g_real_run == nullptr) {
            g_real_run = reinterpret_cast<RunFn>(dlsym(g_nnopbase_handle, kRunSymbol));
        }
        if (g_real_add == nullptr) {
            g_real_add = reinterpret_cast<AddFn>(dlsym(g_nnopbase_handle, kAddSymbol));
        }
        if (g_real_add_param_const_tensor == nullptr) {
            g_real_add_param_const_tensor = reinterpret_cast<AddParamConstTensorFn>(
                dlsym(g_nnopbase_handle, kAddParamConstTensorSymbol));
        }
        if (g_real_add_param_tensor == nullptr) {
            g_real_add_param_tensor = reinterpret_cast<AddParamTensorFn>(
                dlsym(g_nnopbase_handle, kAddParamTensorSymbol));
        }
        g_get_log_api_info = reinterpret_cast<GetLogApiInfoFn>(
            dlsym(g_nnopbase_handle, kGetLogApiInfoSymbol));
    }
    resolve_runtime_api();

    Dl_info info{};
    char resolved_library[PATH_MAX]{};
    const bool have_dl_path = g_real_run != nullptr &&
                              dladdr(reinterpret_cast<void *>(g_real_run), &info) != 0 &&
                              info.dli_fname != nullptr;
    const bool have_real_path = have_dl_path && realpath(info.dli_fname, resolved_library) != nullptr;
    const char *validated_path = have_real_path ? resolved_library :
                                 (info.dli_fname == nullptr ? "" : info.dli_fname);
    const bool path_ok = std::strstr(validated_path, kExpectedLibraryPathFragment) != nullptr;
    g_abi_ok = path_ok && g_real_add != nullptr && g_get_log_api_info != nullptr &&
               g_real_add_param_const_tensor != nullptr && g_real_add_param_tensor != nullptr;

    char line[1536];
    const int written = std::snprintf(
        line, sizeof(line),
        "{\"event\":\"abi_validation\",\"pid\":%ld,\"tid\":%ld,\"ts_ns\":%" PRIu64
        ",\"abi_ok\":%s,\"run\":%s,\"add\":%s,\"api_info\":%s,"
        "\"runtime_api\":%s,\"library\":\"%s\"}",
        static_cast<long>(getpid()), static_cast<long>(syscall(SYS_gettid)), monotonic_ns(),
        g_abi_ok ? "true" : "false", g_real_run != nullptr ? "true" : "false",
        g_real_add != nullptr ? "true" : "false", g_get_log_api_info != nullptr ? "true" : "false",
        (g_get_device != nullptr && g_malloc != nullptr && g_free != nullptr &&
         g_malloc_host != nullptr && g_free_host != nullptr && g_memcpy != nullptr &&
         g_memcpy_async != nullptr && g_synchronize_stream != nullptr) ? "true" : "false",
        validated_path);
    if (written > 0 && static_cast<size_t>(written) < sizeof(line)) {
        write_line(line, static_cast<size_t>(written));
    }
}

bool starts_with(const std::string &value, const char *prefix) {
    const size_t length = std::strlen(prefix);
    return value.size() >= length && value.compare(0, length, prefix) == 0;
}

bool reserve_hit(Fault20Control *control, uint32_t *hit_index) {
    uint32_t current = __atomic_load_n(&control->hit_count, __ATOMIC_ACQUIRE);
    const uint32_t limit = __atomic_load_n(&control->max_hits, __ATOMIC_RELAXED);
    while (current < limit) {
        uint32_t desired = current + 1;
        if (__atomic_compare_exchange_n(&control->hit_count, &current, desired, false,
                                        __ATOMIC_ACQ_REL, __ATOMIC_ACQUIRE)) {
            *hit_index = desired;
            return true;
        }
    }
    return false;
}

bool reserve_hit_per_device(Fault20Control *control, int32_t device,
                            uint32_t *hit_index) {
    if (device < 0 || static_cast<size_t>(device) >= FAULT20_MAX_DEVICES) {
        return false;
    }
    uint32_t *counter = &control->hit_count_per_device[device];
    uint32_t current = __atomic_load_n(counter, __ATOMIC_ACQUIRE);
    const uint32_t limit =
        __atomic_load_n(&control->max_hits_per_device, __ATOMIC_RELAXED);
    while (current < limit) {
        const uint32_t desired = current + 1;
        if (__atomic_compare_exchange_n(counter, &current, desired, false,
                                        __ATOMIC_ACQ_REL, __ATOMIC_ACQUIRE)) {
            __atomic_add_fetch(&control->hit_count, 1U, __ATOMIC_ACQ_REL);
            *hit_index = desired;
            return true;
        }
    }
    return false;
}

bool lookup_metadata(op::internal::OpExecCache *cache, RegisteredMetadata *metadata) {
    std::lock_guard<std::mutex> lock(registry_mutex());
    const auto found = registry().find(cache);
    if (found == registry().end()) {
        return false;
    }
    *metadata = found->second;
    return true;
}

bool lookup_matmul_metadata(op::internal::OpExecCache *cache,
                            RegisteredMatmulMetadata *metadata, uint64_t *run_index) {
    std::lock_guard<std::mutex> lock(registry_mutex());
    const auto found = matmul_registry().find(cache);
    if (found == matmul_registry().end()) {
        return false;
    }
    ++found->second.run_count;
    *metadata = found->second;
    *run_index = found->second.run_count;
    return true;
}

void log_registration(const char *event, op::internal::OpExecCache *cache,
                      const Fault20X1Metadata *metadata, int32_t device, uint32_t add_count) {
    char line[1536];
    const int written = std::snprintf(
        line, sizeof(line),
        "{\"event\":\"%s\",\"pid\":%ld,\"tid\":%ld,\"ts_ns\":%" PRIu64
        ",\"cache\":\"%p\",\"device\":%d,\"add_count\":%u,\"dtype\":%d,"
        "\"format\":%d,\"view_rank\":%u,\"storage_rank\":%u,"
        "\"shape\":[%" PRId64 ",%" PRId64 "],\"storage_shape0\":%" PRId64
        ",\"strides\":[%" PRId64 ",%" PRId64 "],\"view_offset\":%" PRId64
        ",\"numel\":%" PRId64 ",\"element_size\":%" PRIu64
        ",\"view_bytes\":%" PRIu64 ",\"storage_bytes\":%" PRIu64
        ",\"out_last_dim\":%" PRId64 "}",
        event, static_cast<long>(getpid()), static_cast<long>(syscall(SYS_gettid)), monotonic_ns(),
        cache, device, add_count, metadata == nullptr ? -1 : metadata->dtype,
        metadata == nullptr ? -1 : metadata->format, metadata == nullptr ? 0 : metadata->view_rank,
        metadata == nullptr ? 0 : metadata->storage_rank,
        metadata == nullptr ? -1 : metadata->view_shape[0],
        metadata == nullptr ? -1 : metadata->view_shape[1],
        metadata == nullptr ? -1 : metadata->storage_shape[0],
        metadata == nullptr ? -1 : metadata->strides[0],
        metadata == nullptr ? -1 : metadata->strides[1],
        metadata == nullptr ? -1 : metadata->view_offset,
        metadata == nullptr ? -1 : metadata->numel,
        metadata == nullptr ? 0 : metadata->element_size,
        metadata == nullptr ? 0 : metadata->view_bytes,
        metadata == nullptr ? 0 : metadata->storage_bytes,
        metadata == nullptr ? -1 : metadata->output_last_dim);
    if (written > 0 && static_cast<size_t>(written) < sizeof(line)) {
        write_line(line, static_cast<size_t>(written));
    }
}

void log_matmul_registration(const char *event, op::internal::OpExecCache *cache,
                             const Fault20MatmulMetadata *metadata, int32_t device,
                             uint32_t add_count) {
    char line[4096];
    const Fault20TensorMetadata empty{};
    const Fault20TensorMetadata &self = metadata == nullptr ? empty : metadata->self;
    const Fault20TensorMetadata &mat2 = metadata == nullptr ? empty : metadata->mat2;
    const Fault20TensorMetadata &out = metadata == nullptr ? empty : metadata->out;
    const int written = std::snprintf(
        line, sizeof(line),
        "{\"event\":\"%s\",\"pid\":%ld,\"tid\":%ld,\"ts_ns\":%" PRIu64
        ",\"cache\":\"%p\",\"device\":%d,\"add_count\":%u,"
        "\"cube_math_type\":%d,\"all_present\":%s,\"any_empty_storage\":%s,"
        "\"any_shared_storage\":%s,"
        "\"self\":{\"dtype\":%d,\"format\":%d,\"view_rank\":%u,"
        "\"storage_rank\":%u,\"stride_rank\":%u,\"contiguous\":%s,"
        "\"shape\":[%" PRId64 ",%" PRId64 "],\"storage_shape\":[%" PRId64
        ",%" PRId64 "],\"strides\":[%" PRId64 ",%" PRId64
        "],\"view_offset\":%" PRId64 ",\"numel\":%" PRId64
        ",\"view_bytes\":%" PRIu64 ",\"storage_bytes\":%" PRIu64
        ",\"raw_addr\":\"0x%" PRIx64 "\",\"storage_addr\":\"0x%" PRIx64
        "\",\"data_addr\":\"0x%" PRIx64 "\"},"
        "\"mat2\":{\"dtype\":%d,\"format\":%d,\"view_rank\":%u,"
        "\"storage_rank\":%u,\"stride_rank\":%u,\"contiguous\":%s,"
        "\"shape\":[%" PRId64 ",%" PRId64 "],\"storage_shape\":[%" PRId64
        ",%" PRId64 "],\"strides\":[%" PRId64 ",%" PRId64
        "],\"view_offset\":%" PRId64 ",\"numel\":%" PRId64
        ",\"view_bytes\":%" PRIu64 ",\"storage_bytes\":%" PRIu64
        ",\"raw_addr\":\"0x%" PRIx64 "\",\"storage_addr\":\"0x%" PRIx64
        "\",\"data_addr\":\"0x%" PRIx64 "\"},"
        "\"out\":{\"dtype\":%d,\"format\":%d,\"view_rank\":%u,"
        "\"storage_rank\":%u,\"stride_rank\":%u,\"contiguous\":%s,"
        "\"shape\":[%" PRId64 ",%" PRId64 "],\"storage_shape\":[%" PRId64
        ",%" PRId64 "],\"strides\":[%" PRId64 ",%" PRId64
        "],\"view_offset\":%" PRId64 ",\"numel\":%" PRId64
        ",\"view_bytes\":%" PRIu64 ",\"storage_bytes\":%" PRIu64
        ",\"raw_addr\":\"0x%" PRIx64 "\",\"storage_addr\":\"0x%" PRIx64
        "\",\"data_addr\":\"0x%" PRIx64 "\"}}",
        event, static_cast<long>(getpid()), static_cast<long>(syscall(SYS_gettid)),
        monotonic_ns(), cache, device, add_count,
        metadata == nullptr ? -1 : metadata->cube_math_type,
        metadata != nullptr && metadata->all_present ? "true" : "false",
        metadata != nullptr && metadata->any_empty_storage ? "true" : "false",
        metadata != nullptr && metadata->any_shared_storage ? "true" : "false",
        self.dtype, self.format, self.view_rank, self.storage_rank, self.stride_rank,
        self.contiguous ? "true" : "false", self.view_shape[0], self.view_shape[1],
        self.storage_shape[0], self.storage_shape[1], self.strides[0], self.strides[1],
        self.view_offset, self.numel, self.view_bytes, self.storage_bytes, self.raw_addr,
        self.storage_addr, self.data_addr, mat2.dtype, mat2.format, mat2.view_rank,
        mat2.storage_rank, mat2.stride_rank, mat2.contiguous ? "true" : "false",
        mat2.view_shape[0], mat2.view_shape[1], mat2.storage_shape[0],
        mat2.storage_shape[1], mat2.strides[0], mat2.strides[1], mat2.view_offset,
        mat2.numel, mat2.view_bytes, mat2.storage_bytes, mat2.raw_addr,
        mat2.storage_addr, mat2.data_addr, out.dtype, out.format, out.view_rank,
        out.storage_rank, out.stride_rank, out.contiguous ? "true" : "false",
        out.view_shape[0], out.view_shape[1], out.storage_shape[0],
        out.storage_shape[1], out.strides[0], out.strides[1], out.view_offset,
        out.numel, out.view_bytes, out.storage_bytes, out.raw_addr, out.storage_addr,
        out.data_addr);
    if (written > 0 && static_cast<size_t>(written) < sizeof(line)) {
        write_line(line, static_cast<size_t>(written));
    }
}

void log_matmul_param_collection(op::internal::OpExecCache *cache,
                                 const Fault20MatmulMetadata *metadata,
                                 uint32_t tensor_param_count,
                                 const uintptr_t tensor_params[8], bool accepted) {
    const uint64_t self_tensor = metadata == nullptr ? 0 : metadata->self.tensor_ptr;
    const uint64_t mat2_tensor = metadata == nullptr ? 0 : metadata->mat2.tensor_ptr;
    const uint64_t out_tensor = metadata == nullptr ? 0 : metadata->out.tensor_ptr;
    const bool count_ok = tensor_param_count == 3;
    const bool self_ok = count_ok && tensor_params[0] == self_tensor;
    const bool mat2_ok = count_ok && tensor_params[1] == mat2_tensor;
    const bool out_ok = count_ok && tensor_params[2] == out_tensor;
    char line[2048];
    const int written = std::snprintf(
        line, sizeof(line),
        "{\"event\":\"matmulweightnz_param_collection\",\"pid\":%ld,\"tid\":%ld,"
        "\"ts_ns\":%" PRIu64 ",\"cache\":\"%p\",\"accepted\":%s,"
        "\"tensor_param_count\":%u,\"tensor_params\":[\"0x%" PRIxPTR
        "\",\"0x%" PRIxPTR "\",\"0x%" PRIxPTR "\"],"
        "\"formal_tensors\":{\"self\":\"0x%" PRIx64
        "\",\"mat2\":\"0x%" PRIx64 "\",\"out\":\"0x%" PRIx64 "\"},"
        "\"order_equality\":{\"count_is_three\":%s,\"param0_self\":%s,"
        "\"param1_mat2\":%s,\"param2_out\":%s},"
        "\"no_empty_storage\":%s,\"no_shared_storage\":%s}",
        static_cast<long>(getpid()), static_cast<long>(syscall(SYS_gettid)),
        monotonic_ns(), cache, accepted ? "true" : "false", tensor_param_count,
        tensor_params[0], tensor_params[1], tensor_params[2], self_tensor, mat2_tensor,
        out_tensor, count_ok ? "true" : "false", self_ok ? "true" : "false",
        mat2_ok ? "true" : "false", out_ok ? "true" : "false",
        metadata != nullptr && metadata->any_empty_storage == 0 ? "true" : "false",
        metadata != nullptr && metadata->any_shared_storage == 0 ? "true" : "false");
    if (written > 0 && static_cast<size_t>(written) < sizeof(line)) {
        write_line(line, static_cast<size_t>(written));
    }
}

void capture_matmul_tensor_param(const aclTensor *tensor, const char *overload) {
    if (!g_capture.active || g_capture.kind != CaptureKind::kMatmulWeightNz) {
        return;
    }
    const uint32_t ordinal = g_capture.tensor_param_count++;
    if (ordinal < 8) {
        g_capture.tensor_params[ordinal] = reinterpret_cast<uintptr_t>(tensor);
    }
    char line[768];
    const int written = std::snprintf(
        line, sizeof(line),
        "{\"event\":\"matmulweightnz_add_param_tensor\",\"pid\":%ld,"
        "\"tid\":%ld,\"ts_ns\":%" PRIu64 ",\"ordinal\":%u,"
        "\"overload\":\"%s\",\"tensor\":\"%p\"}",
        static_cast<long>(getpid()), static_cast<long>(syscall(SYS_gettid)),
        monotonic_ns(), ordinal, overload, tensor);
    if (written > 0 && static_cast<size_t>(written) < sizeof(line)) {
        write_line(line, static_cast<size_t>(written));
    }
}

void add_param_const_and_capture(const aclTensor *tensor) {
    pthread_once(&g_resolve_once, resolve_real);
    if (!g_add_param_forwarding) {
        capture_matmul_tensor_param(tensor, "const");
    }
    if (g_real_add_param_const_tensor != nullptr) {
        const bool previous = g_add_param_forwarding;
        g_add_param_forwarding = true;
        g_real_add_param_const_tensor(tensor);
        g_add_param_forwarding = previous;
    } else {
        log_simple("resolve_add_param_const_tensor_failed", "real AddParamToBuf const unavailable");
    }
}

void add_param_and_capture(aclTensor *tensor) {
    pthread_once(&g_resolve_once, resolve_real);
    if (!g_add_param_forwarding) {
        capture_matmul_tensor_param(tensor, "mutable");
    }
    if (g_real_add_param_tensor != nullptr) {
        const bool previous = g_add_param_forwarding;
        g_add_param_forwarding = true;
        g_real_add_param_tensor(tensor);
        g_add_param_forwarding = previous;
    } else {
        log_simple("resolve_add_param_tensor_failed", "real AddParamToBuf mutable unavailable");
    }
}

bool address_equals(uint64_t expected, void *actual) {
    return expected != 0 && expected == reinterpret_cast<uintptr_t>(actual);
}

void log_matmul_run_trace(op::internal::OpExecCache *cache,
                          const RegisteredMatmulMetadata &registered,
                          const std::string &api_info,
                          const std::vector<void *> &addresses, uint64_t run_index) {
    const void *slot0 = addresses.size() > 0 ? addresses[0] : nullptr;
    const void *slot1 = addresses.size() > 1 ? addresses[1] : nullptr;
    const void *slot2 = addresses.size() > 2 ? addresses[2] : nullptr;
    const auto &self = registered.tensors.self;
    const auto &mat2 = registered.tensors.mat2;
    const auto &out = registered.tensors.out;
    char line[6144];
    const int written = std::snprintf(
        line, sizeof(line),
        "{\"event\":\"matmulweightnz_run_trace\",\"pid\":%ld,\"tid\":%ld,"
        "\"ts_ns\":%" PRIu64 ",\"device\":%d,\"cache\":\"%p\","
        "\"run_index\":%" PRIu64 ",\"api_info\":\"%s\","
        "\"address_count\":%zu,\"addresses\":[\"%p\",\"%p\",\"%p\"],"
        "\"formal_addresses\":{\"self_raw\":\"0x%" PRIx64
        "\",\"self_storage\":\"0x%" PRIx64 "\",\"self_data\":\"0x%" PRIx64
        "\",\"mat2_raw\":\"0x%" PRIx64 "\",\"mat2_storage\":\"0x%" PRIx64
        "\",\"mat2_data\":\"0x%" PRIx64 "\",\"out_raw\":\"0x%" PRIx64
        "\",\"out_storage\":\"0x%" PRIx64 "\",\"out_data\":\"0x%" PRIx64
        "\"},"
        "\"slot_equality\":{\"slot0_self_raw\":%s,\"slot0_self_storage\":%s,"
        "\"slot0_self_data\":%s,\"slot1_mat2_raw\":%s,"
        "\"slot1_mat2_storage\":%s,\"slot1_mat2_data\":%s,"
        "\"slot2_out_raw\":%s,\"slot2_out_storage\":%s,"
        "\"slot2_out_data\":%s},"
        "\"collection_order_proof\":{\"tensor_param_count\":%u,"
        "\"param0_self\":%s,\"param1_mat2\":%s,\"param2_out\":%s,"
        "\"slot_roles\":[\"self\",\"mat2\",\"out\"]},"
        "\"all_present\":%s,\"any_empty_storage\":%s,"
        "\"any_shared_storage\":%s,"
        "\"self\":{\"dtype\":%d,\"format\":%d,\"shape\":[%" PRId64
        ",%" PRId64 "],\"storage_shape\":[%" PRId64 ",%" PRId64
        "],\"strides\":[%" PRId64 ",%" PRId64 "],\"offset\":%" PRId64
        ",\"bytes\":%" PRIu64 ",\"contiguous\":%s},"
        "\"mat2\":{\"dtype\":%d,\"format\":%d,\"shape\":[%" PRId64
        ",%" PRId64 "],\"bytes\":%" PRIu64 "},"
        "\"out\":{\"dtype\":%d,\"format\":%d,\"shape\":[%" PRId64
        ",%" PRId64 "],\"bytes\":%" PRIu64 "}}",
        static_cast<long>(getpid()), static_cast<long>(syscall(SYS_gettid)),
        monotonic_ns(), registered.device, cache, run_index, api_info.c_str(),
        addresses.size(), slot0, slot1, slot2, self.raw_addr, self.storage_addr,
        self.data_addr, mat2.raw_addr, mat2.storage_addr, mat2.data_addr, out.raw_addr,
        out.storage_addr, out.data_addr,
        address_equals(self.raw_addr, addresses.size() > 0 ? addresses[0] : nullptr)
            ? "true" : "false",
        address_equals(self.storage_addr, addresses.size() > 0 ? addresses[0] : nullptr)
            ? "true" : "false",
        address_equals(self.data_addr, addresses.size() > 0 ? addresses[0] : nullptr)
            ? "true" : "false",
        address_equals(mat2.raw_addr, addresses.size() > 1 ? addresses[1] : nullptr)
            ? "true" : "false",
        address_equals(mat2.storage_addr, addresses.size() > 1 ? addresses[1] : nullptr)
            ? "true" : "false",
        address_equals(mat2.data_addr, addresses.size() > 1 ? addresses[1] : nullptr)
            ? "true" : "false",
        address_equals(out.raw_addr, addresses.size() > 2 ? addresses[2] : nullptr)
            ? "true" : "false",
        address_equals(out.storage_addr, addresses.size() > 2 ? addresses[2] : nullptr)
            ? "true" : "false",
        address_equals(out.data_addr, addresses.size() > 2 ? addresses[2] : nullptr)
            ? "true" : "false",
        registered.tensor_param_count,
        registered.tensor_param_count == 3 &&
                registered.tensor_params[0] == self.tensor_ptr
            ? "true" : "false",
        registered.tensor_param_count == 3 &&
                registered.tensor_params[1] == mat2.tensor_ptr
            ? "true" : "false",
        registered.tensor_param_count == 3 &&
                registered.tensor_params[2] == out.tensor_ptr
            ? "true" : "false",
        registered.tensors.all_present ? "true" : "false",
        registered.tensors.any_empty_storage ? "true" : "false",
        registered.tensors.any_shared_storage ? "true" : "false", self.dtype,
        self.format, self.view_shape[0], self.view_shape[1], self.storage_shape[0],
        self.storage_shape[1], self.strides[0], self.strides[1], self.view_offset,
        self.storage_bytes, self.contiguous ? "true" : "false", mat2.dtype,
        mat2.format, mat2.view_shape[0], mat2.view_shape[1], mat2.storage_bytes,
        out.dtype, out.format, out.view_shape[0], out.view_shape[1], out.storage_bytes);
    if (written > 0 && static_cast<size_t>(written) < sizeof(line)) {
        write_line(line, static_cast<size_t>(written));
    }
}

aclnnStatus run_original(op::internal::OpExecCache *cache, void *workspace, aclrtStream stream,
                         const std::vector<void *> &addresses) {
    return g_real_run(cache, workspace, stream, addresses);
}

void set_last_status(Fault20Control *control, int32_t status) {
    __atomic_store_n(&control->last_status, status, __ATOMIC_RELEASE);
}

void set_device_last_status(Fault20Control *control, int32_t device, int32_t status) {
    set_last_status(control, status);
    if (device >= 0 && static_cast<size_t>(device) < FAULT20_MAX_DEVICES) {
        __atomic_store_n(&control->last_status_per_device[device], status,
                         __ATOMIC_RELEASE);
    }
}

void log_numeric_attempt(uint32_t mode, const char *outcome, const std::string &api_info,
                         op::internal::OpExecCache *cache,
                         const RegisteredMetadata &metadata, const Int4Attempt &attempt) {
    const uint64_t total_ns = monotonic_ns() - attempt.begin_ns;
    const char *event_name = mode == FAULT20_MODE_INT2_EQUIVALENT
                                 ? "int2_equivalent_substitution"
                                 : "int4_equivalent_substitution";
    const char *rule = mode == FAULT20_MODE_INT2_EQUIVALENT
                           ? "round_away_from_zero(x/64),clamp[-2,1],x64"
                           : "round_away_from_zero(x/16),clamp[-8,7],x16";
    char line[6144];
    const int written = std::snprintf(
        line, sizeof(line),
        "{\"event\":\"%s\",\"outcome\":\"%s\","
        "\"pid\":%ld,\"tid\":%ld,\"ts_ns\":%" PRIu64 ",\"device\":%d,"
        "\"rank\":%d,\"hit\":%u,\"candidate\":%u,\"cache\":\"%p\","
        "\"api_info\":\"%s\",\"shape\":[%" PRId64 ",%" PRId64 "],"
        "\"strides\":[%" PRId64 ",%" PRId64 "],\"dtype\":%d,\"format\":%d,"
        "\"view_offset\":%" PRId64 ",\"bytes\":%" PRIu64 ","
        "\"original\":\"%p\",\"temporary\":\"%p\",\"host_buffer\":\"%p\","
        "\"producer_match\":true,\"temporary_used\":%s,\"quant_step\":%u,"
        "\"rule\":\"%s\","
        "\"stats_valid\":%s,\"changed_elements\":%" PRIu64 ","
        "\"unchanged_elements\":%" PRIu64 ",\"min_before\":%d,"
        "\"max_before\":%d,\"min_after\":%d,\"max_after\":%d,"
        "\"zero_before\":%" PRIu64 ",\"zero_after\":%" PRIu64 ","
        "\"output_histogram\":{\"-128\":%" PRIu64 ",\"-64\":%" PRIu64
        ",\"0\":%" PRIu64 ",\"64\":%" PRIu64 "},"
        "\"result_status\":%d,\"pre_sync_status\":%d,"
        "\"host_malloc_status\":%d,\"d2h_status\":%d,"
        "\"device_malloc_status\":%d,\"h2d_status\":%d,\"run_status\":%d,"
        "\"post_sync_status\":%d,\"device_free_status\":%d,"
        "\"host_free_status\":%d,\"temporary_freed\":%s,\"host_freed\":%s,"
        "\"source_aligned_64\":%s,\"temporary_aligned_64\":%s,"
        "\"pre_sync_ns\":%" PRIu64 ",\"host_alloc_ns\":%" PRIu64 ","
        "\"d2h_ns\":%" PRIu64 ",\"cpu_transform_ns\":%" PRIu64 ","
        "\"device_alloc_ns\":%" PRIu64 ",\"h2d_enqueue_ns\":%" PRIu64 ","
        "\"real_run_ns\":%" PRIu64 ",\"post_sync_ns\":%" PRIu64 ","
        "\"total_ns\":%" PRIu64 "}",
        event_name, outcome, static_cast<long>(getpid()),
        static_cast<long>(syscall(SYS_gettid)),
        attempt.begin_ns, metadata.device, metadata.device, attempt.hit_index,
        attempt.candidate_index, cache, api_info.c_str(), metadata.x1.view_shape[0],
        metadata.x1.view_shape[1], metadata.x1.strides[0], metadata.x1.strides[1],
        metadata.x1.dtype, metadata.x1.format, metadata.x1.view_offset,
        metadata.x1.storage_bytes, attempt.original, attempt.temporary, attempt.host_buffer,
        attempt.temporary_used ? "true" : "false", attempt.quant_step, rule,
        attempt.stats_valid ? "true" : "false", attempt.changed_elements,
        attempt.unchanged_elements, attempt.min_before, attempt.max_before, attempt.min_after,
        attempt.max_after, attempt.zero_before, attempt.zero_after,
        attempt.int2_histogram[0], attempt.int2_histogram[1],
        attempt.int2_histogram[2], attempt.int2_histogram[3], attempt.result_status,
        attempt.pre_sync_status, attempt.host_malloc_status, attempt.d2h_status,
        attempt.device_malloc_status, attempt.h2d_status, attempt.run_status,
        attempt.post_sync_status, attempt.device_free_status, attempt.host_free_status,
        attempt.temporary_freed ? "true" : "false", attempt.host_freed ? "true" : "false",
        (reinterpret_cast<uintptr_t>(attempt.original) % 64U) == 0 ? "true" : "false",
        (attempt.temporary != nullptr && reinterpret_cast<uintptr_t>(attempt.temporary) % 64U == 0)
            ? "true"
            : "false",
        attempt.pre_sync_ns, attempt.host_alloc_ns, attempt.d2h_ns,
        attempt.cpu_transform_ns, attempt.device_alloc_ns, attempt.h2d_enqueue_ns,
        attempt.real_run_ns, attempt.post_sync_ns, total_ns);
    if (written > 0 && static_cast<size_t>(written) < sizeof(line)) {
        write_line(line, static_cast<size_t>(written));
    }
}

void transform_numeric_equivalent(void *host_buffer, uint64_t bytes, uint32_t mode,
                                  Int4Attempt *attempt) {
    auto *values = static_cast<int8_t *>(host_buffer);
    attempt->min_before = 127;
    attempt->max_before = -128;
    attempt->min_after = 127;
    attempt->max_after = -128;
    for (uint64_t index = 0; index < bytes; ++index) {
        const int before = static_cast<int>(values[index]);
        const int after = static_cast<int>(
            mode == FAULT20_MODE_INT2_EQUIVALENT
                ? fault20_int2_equivalent_value(values[index])
                : fault20_int4_equivalent_value(values[index]));
        values[index] = static_cast<int8_t>(after);
        attempt->min_before = before < attempt->min_before ? before : attempt->min_before;
        attempt->max_before = before > attempt->max_before ? before : attempt->max_before;
        attempt->min_after = after < attempt->min_after ? after : attempt->min_after;
        attempt->max_after = after > attempt->max_after ? after : attempt->max_after;
        attempt->zero_before += before == 0 ? 1U : 0U;
        attempt->zero_after += after == 0 ? 1U : 0U;
        attempt->changed_elements += before != after ? 1U : 0U;
        if (mode == FAULT20_MODE_INT2_EQUIVALENT) {
            if (after == -128) {
                ++attempt->int2_histogram[0];
            } else if (after == -64) {
                ++attempt->int2_histogram[1];
            } else if (after == 0) {
                ++attempt->int2_histogram[2];
            } else if (after == 64) {
                ++attempt->int2_histogram[3];
            }
        }
    }
    attempt->unchanged_elements = bytes - attempt->changed_elements;
    attempt->stats_valid = true;
}

float bf16_bits_to_float(uint16_t bits) {
    const uint32_t wide = static_cast<uint32_t>(bits) << 16U;
    float value = 0.0F;
    std::memcpy(&value, &wide, sizeof(value));
    return value;
}

float fp16_bits_to_float(uint16_t bits) {
    const bool negative = (bits & 0x8000U) != 0;
    const uint32_t exponent = (bits >> 10U) & 0x1FU;
    const uint32_t mantissa = bits & 0x03FFU;
    float value = 0.0F;
    if (exponent == 0) {
        value = std::ldexp(static_cast<float>(mantissa), -24);
    } else if (exponent == 0x1FU) {
        value = mantissa == 0 ? std::numeric_limits<float>::infinity()
                              : std::numeric_limits<float>::quiet_NaN();
    } else {
        value = std::ldexp(1.0F + static_cast<float>(mantissa) / 1024.0F,
                           static_cast<int>(exponent) - 15);
    }
    return negative ? -value : value;
}

bool fp16_or_bf16_nonfinite(uint16_t bits, int32_t dtype) {
    return dtype == ACL_BF16 ? (bits & 0x7F80U) == 0x7F80U
                             : (bits & 0x7C00U) == 0x7C00U;
}

bool fp16_or_bf16_zero(uint16_t bits, int32_t dtype) {
    return dtype == ACL_BF16 ? (bits & 0x7FFFU) == 0
                             : (bits & 0x7FFFU) == 0;
}

void transform_mantissa(void *host_buffer, uint64_t elements, int32_t dtype,
                        uint32_t keep_bits, MantissaAttempt *attempt) {
    auto *values = static_cast<uint16_t *>(host_buffer);
    attempt->min_before = std::numeric_limits<float>::infinity();
    attempt->max_before = -std::numeric_limits<float>::infinity();
    attempt->min_after = std::numeric_limits<float>::infinity();
    attempt->max_after = -std::numeric_limits<float>::infinity();
    uint64_t finite_count = 0;
    for (uint64_t index = 0; index < elements; ++index) {
        const uint16_t before_bits = values[index];
        const uint16_t after_bits =
            dtype == ACL_BF16
                ? fault20_truncate_bf16_mantissa(before_bits, keep_bits)
                : fault20_truncate_fp16_mantissa(before_bits, keep_bits);
        values[index] = after_bits;
        const bool nonfinite_before = fp16_or_bf16_nonfinite(before_bits, dtype);
        const bool nonfinite_after = fp16_or_bf16_nonfinite(after_bits, dtype);
        attempt->nonfinite_before += nonfinite_before ? 1U : 0U;
        attempt->nonfinite_after += nonfinite_after ? 1U : 0U;
        attempt->zero_before += fp16_or_bf16_zero(before_bits, dtype) ? 1U : 0U;
        attempt->zero_after += fp16_or_bf16_zero(after_bits, dtype) ? 1U : 0U;
        attempt->changed_elements += before_bits != after_bits ? 1U : 0U;
        if (!nonfinite_before && !nonfinite_after) {
            const float before = dtype == ACL_BF16 ? bf16_bits_to_float(before_bits)
                                                    : fp16_bits_to_float(before_bits);
            const float after = dtype == ACL_BF16 ? bf16_bits_to_float(after_bits)
                                                   : fp16_bits_to_float(after_bits);
            attempt->min_before = before < attempt->min_before ? before : attempt->min_before;
            attempt->max_before = before > attempt->max_before ? before : attempt->max_before;
            attempt->min_after = after < attempt->min_after ? after : attempt->min_after;
            attempt->max_after = after > attempt->max_after ? after : attempt->max_after;
            const float error = std::fabs(before - after);
            attempt->max_abs_error =
                error > attempt->max_abs_error ? error : attempt->max_abs_error;
            ++finite_count;
        }
    }
    attempt->unchanged_elements = elements - attempt->changed_elements;
    if (finite_count == 0) {
        attempt->min_before = 0.0F;
        attempt->max_before = 0.0F;
        attempt->min_after = 0.0F;
        attempt->max_after = 0.0F;
    }
    attempt->stats_valid = true;
}

void log_mantissa_attempt(const char *outcome, const std::string &api_info,
                          op::internal::OpExecCache *cache,
                          const RegisteredMatmulMetadata &metadata,
                          const MantissaAttempt &attempt) {
    const auto &self = metadata.tensors.self;
    char line[6144];
    const int written = std::snprintf(
        line, sizeof(line),
        "{\"event\":\"matmulweightnz_mantissa_truncate\",\"pid\":%ld,"
        "\"tid\":%ld,\"ts_ns\":%" PRIu64 ",\"outcome\":\"%s\","
        "\"device\":%d,\"hit\":%u,\"candidate\":%u,\"cache\":\"%p\","
        "\"api_info\":\"%s\",\"target_slot\":0,\"slot_role\":\"self\","
        "\"dtype\":%d,\"format\":%d,\"shape\":[%" PRId64 ",%" PRId64
        "],\"strides\":[%" PRId64 ",%" PRId64 "],\"offset\":%" PRId64
        ",\"bytes\":%" PRIu64 ",\"keep_mantissa_bits\":%u,"
        "\"original\":\"%p\",\"temporary\":\"%p\",\"host_buffer\":\"%p\","
        "\"address_changed\":%s,\"temporary_used\":%s,"
        "\"temporary_freed\":%s,\"host_freed\":%s,\"stats_valid\":%s,"
        "\"changed_elements\":%" PRIu64 ",\"unchanged_elements\":%" PRIu64
        ",\"nonfinite_before\":%" PRIu64 ",\"nonfinite_after\":%" PRIu64
        ",\"zero_before\":%" PRIu64 ",\"zero_after\":%" PRIu64
        ",\"min_before\":%.9g,\"max_before\":%.9g,"
        "\"min_after\":%.9g,\"max_after\":%.9g,\"max_abs_error\":%.9g,"
        "\"result_status\":%d,\"pre_sync_status\":%d,"
        "\"host_malloc_status\":%d,\"d2h_status\":%d,"
        "\"device_malloc_status\":%d,\"h2d_status\":%d,"
        "\"run_status\":%d,\"post_sync_status\":%d,"
        "\"device_free_status\":%d,\"host_free_status\":%d,"
        "\"pre_sync_ns\":%" PRIu64 ",\"host_alloc_ns\":%" PRIu64
        ",\"d2h_ns\":%" PRIu64 ",\"cpu_transform_ns\":%" PRIu64
        ",\"device_alloc_ns\":%" PRIu64 ",\"h2d_enqueue_ns\":%" PRIu64
        ",\"real_run_ns\":%" PRIu64 ",\"post_sync_ns\":%" PRIu64
        ",\"total_ns\":%" PRIu64 "}",
        static_cast<long>(getpid()), static_cast<long>(syscall(SYS_gettid)),
        attempt.begin_ns, outcome, metadata.device, attempt.hit_index,
        attempt.candidate_index, cache, api_info.c_str(), self.dtype, self.format,
        self.view_shape[0], self.view_shape[1], self.strides[0], self.strides[1],
        self.view_offset, self.storage_bytes, attempt.keep_mantissa_bits,
        attempt.original, attempt.temporary, attempt.host_buffer,
        attempt.temporary != nullptr && attempt.temporary != attempt.original ? "true" : "false",
        attempt.temporary_used ? "true" : "false",
        attempt.temporary_freed ? "true" : "false",
        attempt.host_freed ? "true" : "false", attempt.stats_valid ? "true" : "false",
        attempt.changed_elements, attempt.unchanged_elements, attempt.nonfinite_before,
        attempt.nonfinite_after, attempt.zero_before, attempt.zero_after,
        static_cast<double>(attempt.min_before), static_cast<double>(attempt.max_before),
        static_cast<double>(attempt.min_after), static_cast<double>(attempt.max_after),
        static_cast<double>(attempt.max_abs_error), attempt.result_status,
        attempt.pre_sync_status, attempt.host_malloc_status, attempt.d2h_status,
        attempt.device_malloc_status, attempt.h2d_status, attempt.run_status,
        attempt.post_sync_status, attempt.device_free_status, attempt.host_free_status,
        attempt.pre_sync_ns, attempt.host_alloc_ns, attempt.d2h_ns,
        attempt.cpu_transform_ns, attempt.device_alloc_ns, attempt.h2d_enqueue_ns,
        attempt.real_run_ns, attempt.post_sync_ns, monotonic_ns() - attempt.begin_ns);
    if (written > 0 && static_cast<size_t>(written) < sizeof(line)) {
        write_line(line, static_cast<size_t>(written));
    }
}

aclnnStatus run_numeric_fallback(op::internal::OpExecCache *cache, void *workspace,
                                 aclrtStream stream,
                                 const std::vector<void *> &addresses,
                                 Fault20Control *control,
                                 const RegisteredMetadata &metadata,
                                 const std::string &api_info, uint32_t mode,
                                 const char *outcome, Int4Attempt *attempt) {
    const uint64_t begin_ns = monotonic_ns();
    const aclnnStatus status = run_original(cache, workspace, stream, addresses);
    attempt->real_run_ns = monotonic_ns() - begin_ns;
    attempt->run_status = static_cast<int32_t>(status);
    set_device_last_status(control, metadata.device, attempt->result_status);
    log_numeric_attempt(mode, outcome, api_info, cache, metadata, *attempt);
    return status;
}

aclnnStatus run_numeric_equivalent(op::internal::OpExecCache *cache, void *workspace,
                                   aclrtStream stream,
                                   const std::vector<void *> &addresses,
                                   Fault20Control *control,
                                   const RegisteredMetadata &metadata,
                                   const std::string &api_info, uint32_t mode,
                                   uint32_t hit_index, uint32_t candidate_index) {
    Int4Attempt attempt{};
    attempt.hit_index = hit_index;
    attempt.candidate_index = candidate_index;
    attempt.quant_step = __atomic_load_n(&control->quant_step, __ATOMIC_RELAXED);
    attempt.original = addresses[0];
    attempt.begin_ns = monotonic_ns();

    std::vector<void *> addresses_copy;
    try {
        addresses_copy = addresses;
    } catch (...) {
        attempt.result_status = kStatusVectorCopyFailed;
        return run_numeric_fallback(cache, workspace, stream, addresses, control,
                                    metadata, api_info, mode,
                                    "address_vector_copy_failed", &attempt);
    }

    uint64_t operation_begin_ns = monotonic_ns();
    attempt.pre_sync_status = static_cast<int32_t>(g_synchronize_stream(stream));
    attempt.pre_sync_ns = monotonic_ns() - operation_begin_ns;
    if (attempt.pre_sync_status != ACL_SUCCESS) {
        attempt.result_status = kStatusPreSyncFailed;
        return run_numeric_fallback(cache, workspace, stream, addresses, control,
                                    metadata, api_info, mode, "pre_sync_failed", &attempt);
    }

    operation_begin_ns = monotonic_ns();
    attempt.host_malloc_status = static_cast<int32_t>(
        g_malloc_host(&attempt.host_buffer, static_cast<size_t>(metadata.x1.storage_bytes)));
    attempt.host_alloc_ns = monotonic_ns() - operation_begin_ns;
    if (attempt.host_malloc_status != ACL_SUCCESS || attempt.host_buffer == nullptr) {
        attempt.result_status = kStatusHostAllocFailed;
        return run_numeric_fallback(cache, workspace, stream, addresses, control,
                                    metadata, api_info, mode, "host_alloc_failed", &attempt);
    }

    operation_begin_ns = monotonic_ns();
    attempt.d2h_status = static_cast<int32_t>(g_memcpy(
        attempt.host_buffer, static_cast<size_t>(metadata.x1.storage_bytes), addresses[0],
        static_cast<size_t>(metadata.x1.storage_bytes), ACL_MEMCPY_DEVICE_TO_HOST));
    attempt.d2h_ns = monotonic_ns() - operation_begin_ns;
    if (attempt.d2h_status != ACL_SUCCESS) {
        attempt.host_free_status = static_cast<int32_t>(g_free_host(attempt.host_buffer));
        attempt.host_freed = attempt.host_free_status == ACL_SUCCESS;
        attempt.result_status = kStatusD2HFailed;
        return run_numeric_fallback(cache, workspace, stream, addresses, control,
                                    metadata, api_info, mode, "d2h_failed", &attempt);
    }

    operation_begin_ns = monotonic_ns();
    transform_numeric_equivalent(attempt.host_buffer, metadata.x1.storage_bytes, mode,
                                 &attempt);
    attempt.cpu_transform_ns = monotonic_ns() - operation_begin_ns;
    if (attempt.changed_elements == 0) {
        attempt.host_free_status = static_cast<int32_t>(g_free_host(attempt.host_buffer));
        attempt.host_freed = attempt.host_free_status == ACL_SUCCESS;
        attempt.result_status = kStatusNoChangedElements;
        return run_numeric_fallback(cache, workspace, stream, addresses, control,
                                    metadata, api_info, mode, "no_changed_elements",
                                    &attempt);
    }

    operation_begin_ns = monotonic_ns();
    attempt.device_malloc_status = static_cast<int32_t>(
        g_malloc(&attempt.temporary, static_cast<size_t>(metadata.x1.storage_bytes),
                 ACL_MEM_MALLOC_HUGE_FIRST));
    attempt.device_alloc_ns = monotonic_ns() - operation_begin_ns;
    if (attempt.device_malloc_status != ACL_SUCCESS || attempt.temporary == nullptr) {
        attempt.host_free_status = static_cast<int32_t>(g_free_host(attempt.host_buffer));
        attempt.host_freed = attempt.host_free_status == ACL_SUCCESS;
        attempt.result_status = kStatusDeviceAllocFailed;
        return run_numeric_fallback(cache, workspace, stream, addresses, control,
                                    metadata, api_info, mode, "device_alloc_failed",
                                    &attempt);
    }

    operation_begin_ns = monotonic_ns();
    attempt.h2d_status = static_cast<int32_t>(g_memcpy_async(
        attempt.temporary, static_cast<size_t>(metadata.x1.storage_bytes), attempt.host_buffer,
        static_cast<size_t>(metadata.x1.storage_bytes), ACL_MEMCPY_HOST_TO_DEVICE, stream));
    attempt.h2d_enqueue_ns = monotonic_ns() - operation_begin_ns;
    if (attempt.h2d_status != ACL_SUCCESS) {
        operation_begin_ns = monotonic_ns();
        attempt.post_sync_status = static_cast<int32_t>(g_synchronize_stream(stream));
        attempt.post_sync_ns = monotonic_ns() - operation_begin_ns;
        if (attempt.post_sync_status == ACL_SUCCESS) {
            attempt.device_free_status = static_cast<int32_t>(g_free(attempt.temporary));
            attempt.temporary_freed = attempt.device_free_status == ACL_SUCCESS;
            attempt.host_free_status = static_cast<int32_t>(g_free_host(attempt.host_buffer));
            attempt.host_freed = attempt.host_free_status == ACL_SUCCESS;
        }
        attempt.result_status = kStatusH2DFailed;
        return run_numeric_fallback(cache, workspace, stream, addresses, control,
                                    metadata, api_info, mode, "h2d_failed", &attempt);
    }

    addresses_copy[0] = attempt.temporary;
    attempt.temporary_used = true;
    operation_begin_ns = monotonic_ns();
    const aclnnStatus run_status = run_original(cache, workspace, stream, addresses_copy);
    attempt.real_run_ns = monotonic_ns() - operation_begin_ns;
    attempt.run_status = static_cast<int32_t>(run_status);

    operation_begin_ns = monotonic_ns();
    attempt.post_sync_status = static_cast<int32_t>(g_synchronize_stream(stream));
    attempt.post_sync_ns = monotonic_ns() - operation_begin_ns;
    if (attempt.post_sync_status == ACL_SUCCESS) {
        attempt.device_free_status = static_cast<int32_t>(g_free(attempt.temporary));
        attempt.temporary_freed = attempt.device_free_status == ACL_SUCCESS;
        attempt.host_free_status = static_cast<int32_t>(g_free_host(attempt.host_buffer));
        attempt.host_freed = attempt.host_free_status == ACL_SUCCESS;
    }

    const char *outcome = "success";
    if (attempt.run_status != ACL_SUCCESS) {
        attempt.result_status = kStatusRunFailed;
        outcome = "real_run_failed";
    } else if (attempt.post_sync_status != ACL_SUCCESS) {
        attempt.result_status = kStatusPostSyncFailed;
        outcome = "post_sync_failed_buffers_retained";
    } else if (attempt.device_free_status != ACL_SUCCESS) {
        attempt.result_status = kStatusDeviceFreeFailed;
        outcome = "device_free_failed";
    } else if (attempt.host_free_status != ACL_SUCCESS) {
        attempt.result_status = kStatusHostFreeFailed;
        outcome = "host_free_failed";
    } else {
        attempt.result_status = kStatusSuccess;
    }
    set_device_last_status(control, metadata.device, attempt.result_status);
    log_numeric_attempt(mode, outcome, api_info, cache, metadata, attempt);
    return run_status;
}

aclnnStatus run_matmul_lossless(op::internal::OpExecCache *cache, void *workspace,
                                aclrtStream stream,
                                const std::vector<void *> &addresses,
                                Fault20Control *control,
                                const RegisteredMatmulMetadata &metadata,
                                const std::string &api_info, uint32_t hit_index,
                                uint32_t candidate_index) {
    const uint64_t begin_ns = monotonic_ns();
    int32_t vector_copy_status = 0;
    int32_t pre_sync_status = kNotCalled;
    int32_t malloc_status = kNotCalled;
    int32_t copy_status = kNotCalled;
    int32_t run_status = kNotCalled;
    int32_t post_sync_status = kNotCalled;
    int32_t free_status = kNotCalled;
    uint64_t pre_sync_ns = 0;
    uint64_t malloc_ns = 0;
    uint64_t d2d_enqueue_ns = 0;
    uint64_t real_run_ns = 0;
    uint64_t post_sync_ns = 0;
    void *temporary = nullptr;
    bool temporary_used = false;
    bool temporary_freed = false;
    const char *outcome = "success";

    std::vector<void *> addresses_copy;
    try {
        addresses_copy = addresses;
    } catch (...) {
        vector_copy_status = kStatusVectorCopyFailed;
        outcome = "address_vector_copy_failed";
    }

    if (vector_copy_status == 0) {
        uint64_t operation_begin_ns = monotonic_ns();
        pre_sync_status = static_cast<int32_t>(g_synchronize_stream(stream));
        pre_sync_ns = monotonic_ns() - operation_begin_ns;
        if (pre_sync_status != ACL_SUCCESS) {
            outcome = "pre_sync_failed";
        }
    }

    if (vector_copy_status == 0 && pre_sync_status == ACL_SUCCESS) {
        uint64_t operation_begin_ns = monotonic_ns();
        malloc_status = static_cast<int32_t>(g_malloc(
            &temporary, static_cast<size_t>(metadata.tensors.self.storage_bytes),
            ACL_MEM_MALLOC_HUGE_FIRST));
        malloc_ns = monotonic_ns() - operation_begin_ns;
        if (malloc_status != ACL_SUCCESS || temporary == nullptr) {
            outcome = "device_alloc_failed";
        }
    }

    if (vector_copy_status == 0 && pre_sync_status == ACL_SUCCESS &&
        malloc_status == ACL_SUCCESS && temporary != nullptr) {
        const uint64_t operation_begin_ns = monotonic_ns();
        copy_status = static_cast<int32_t>(g_memcpy_async(
            temporary, static_cast<size_t>(metadata.tensors.self.storage_bytes), addresses[0],
            static_cast<size_t>(metadata.tensors.self.storage_bytes),
            ACL_MEMCPY_DEVICE_TO_DEVICE, stream));
        d2d_enqueue_ns = monotonic_ns() - operation_begin_ns;
        if (copy_status != ACL_SUCCESS) {
            free_status = static_cast<int32_t>(g_free(temporary));
            temporary_freed = free_status == ACL_SUCCESS;
            outcome = "d2d_failed";
        }
    }

    aclnnStatus result = ACL_SUCCESS;
    if (vector_copy_status == 0 && pre_sync_status == ACL_SUCCESS &&
        malloc_status == ACL_SUCCESS && temporary != nullptr &&
        copy_status == ACL_SUCCESS) {
        addresses_copy[0] = temporary;
        temporary_used = true;
        const uint64_t operation_begin_ns = monotonic_ns();
        result = run_original(cache, workspace, stream, addresses_copy);
        real_run_ns = monotonic_ns() - operation_begin_ns;
        run_status = static_cast<int32_t>(result);

        const uint64_t sync_begin_ns = monotonic_ns();
        post_sync_status = static_cast<int32_t>(g_synchronize_stream(stream));
        post_sync_ns = monotonic_ns() - sync_begin_ns;
        if (post_sync_status == ACL_SUCCESS) {
            free_status = static_cast<int32_t>(g_free(temporary));
            temporary_freed = free_status == ACL_SUCCESS;
        }
        if (run_status != ACL_SUCCESS) {
            outcome = "real_run_failed";
        } else if (post_sync_status != ACL_SUCCESS) {
            outcome = "post_sync_failed_buffers_retained";
        } else if (free_status != ACL_SUCCESS) {
            outcome = "device_free_failed";
        }
    } else {
        const uint64_t operation_begin_ns = monotonic_ns();
        result = run_original(cache, workspace, stream, addresses);
        real_run_ns = monotonic_ns() - operation_begin_ns;
        run_status = static_cast<int32_t>(result);
    }

    const bool success = vector_copy_status == 0 && pre_sync_status == ACL_SUCCESS &&
                         malloc_status == ACL_SUCCESS && copy_status == ACL_SUCCESS &&
                         run_status == ACL_SUCCESS && post_sync_status == ACL_SUCCESS &&
                         free_status == ACL_SUCCESS && temporary_used && temporary_freed &&
                         temporary != nullptr && temporary != addresses[0];
    if (!success && std::strcmp(outcome, "success") == 0) {
        outcome = "incomplete_status";
    }
    set_device_last_status(control, metadata.device,
                           success ? kStatusSuccess : kStatusRunFailed);

    char line[4096];
    const auto &self = metadata.tensors.self;
    const int written = std::snprintf(
        line, sizeof(line),
        "{\"event\":\"matmulweightnz_lossless_substitution\",\"pid\":%ld,"
        "\"tid\":%ld,\"ts_ns\":%" PRIu64 ",\"outcome\":\"%s\","
        "\"device\":%d,\"hit\":%u,\"candidate\":%u,\"cache\":\"%p\","
        "\"api_info\":\"%s\",\"target_slot\":0,\"slot_role\":\"self\","
        "\"dtype\":%d,\"format\":%d,\"shape\":[%" PRId64 ",%" PRId64
        "],\"strides\":[%" PRId64 ",%" PRId64 "],\"offset\":%" PRId64
        ",\"bytes\":%" PRIu64 ",\"original\":\"%p\",\"temporary\":\"%p\","
        "\"address_changed\":%s,\"changed_elements\":0,"
        "\"temporary_used\":%s,\"temporary_freed\":%s,"
        "\"vector_copy_status\":%d,\"pre_sync_status\":%d,"
        "\"malloc_status\":%d,\"copy_status\":%d,\"run_status\":%d,"
        "\"post_sync_status\":%d,\"free_status\":%d,"
        "\"pre_sync_ns\":%" PRIu64 ",\"malloc_ns\":%" PRIu64
        ",\"d2d_enqueue_ns\":%" PRIu64 ",\"real_run_ns\":%" PRIu64
        ",\"post_sync_ns\":%" PRIu64 ",\"total_ns\":%" PRIu64 "}",
        static_cast<long>(getpid()), static_cast<long>(syscall(SYS_gettid)), begin_ns,
        outcome, metadata.device, hit_index, candidate_index, cache, api_info.c_str(),
        self.dtype, self.format, self.view_shape[0], self.view_shape[1], self.strides[0],
        self.strides[1], self.view_offset, self.storage_bytes, addresses[0], temporary,
        temporary != nullptr && temporary != addresses[0] ? "true" : "false",
        temporary_used ? "true" : "false", temporary_freed ? "true" : "false",
        vector_copy_status, pre_sync_status, malloc_status, copy_status, run_status,
        post_sync_status, free_status, pre_sync_ns, malloc_ns, d2d_enqueue_ns,
        real_run_ns, post_sync_ns, monotonic_ns() - begin_ns);
    if (written > 0 && static_cast<size_t>(written) < sizeof(line)) {
        write_line(line, static_cast<size_t>(written));
    }
    return result;
}

aclnnStatus run_matmul_mantissa(op::internal::OpExecCache *cache, void *workspace,
                                aclrtStream stream,
                                const std::vector<void *> &addresses,
                                Fault20Control *control,
                                const RegisteredMatmulMetadata &metadata,
                                const std::string &api_info, uint32_t hit_index,
                                uint32_t candidate_index, uint32_t keep_bits) {
    MantissaAttempt attempt{};
    attempt.hit_index = hit_index;
    attempt.candidate_index = candidate_index;
    attempt.keep_mantissa_bits = keep_bits;
    attempt.original = addresses[0];
    attempt.begin_ns = monotonic_ns();

    auto fallback = [&](const char *outcome) -> aclnnStatus {
        const uint64_t run_begin_ns = monotonic_ns();
        const aclnnStatus status = run_original(cache, workspace, stream, addresses);
        attempt.real_run_ns = monotonic_ns() - run_begin_ns;
        attempt.run_status = static_cast<int32_t>(status);
        set_device_last_status(control, metadata.device, attempt.result_status);
        log_mantissa_attempt(outcome, api_info, cache, metadata, attempt);
        return status;
    };

    std::vector<void *> addresses_copy;
    try {
        addresses_copy = addresses;
    } catch (...) {
        attempt.result_status = kStatusVectorCopyFailed;
        return fallback("address_vector_copy_failed");
    }

    uint64_t operation_begin_ns = monotonic_ns();
    attempt.pre_sync_status = static_cast<int32_t>(g_synchronize_stream(stream));
    attempt.pre_sync_ns = monotonic_ns() - operation_begin_ns;
    if (attempt.pre_sync_status != ACL_SUCCESS) {
        attempt.result_status = kStatusPreSyncFailed;
        return fallback("pre_sync_failed");
    }

    operation_begin_ns = monotonic_ns();
    attempt.host_malloc_status = static_cast<int32_t>(g_malloc_host(
        &attempt.host_buffer, static_cast<size_t>(metadata.tensors.self.storage_bytes)));
    attempt.host_alloc_ns = monotonic_ns() - operation_begin_ns;
    if (attempt.host_malloc_status != ACL_SUCCESS || attempt.host_buffer == nullptr) {
        attempt.result_status = kStatusHostAllocFailed;
        return fallback("host_alloc_failed");
    }

    operation_begin_ns = monotonic_ns();
    attempt.d2h_status = static_cast<int32_t>(g_memcpy(
        attempt.host_buffer, static_cast<size_t>(metadata.tensors.self.storage_bytes),
        addresses[0], static_cast<size_t>(metadata.tensors.self.storage_bytes),
        ACL_MEMCPY_DEVICE_TO_HOST));
    attempt.d2h_ns = monotonic_ns() - operation_begin_ns;
    if (attempt.d2h_status != ACL_SUCCESS) {
        attempt.host_free_status = static_cast<int32_t>(g_free_host(attempt.host_buffer));
        attempt.host_freed = attempt.host_free_status == ACL_SUCCESS;
        attempt.result_status = kStatusD2HFailed;
        return fallback("d2h_failed");
    }

    operation_begin_ns = monotonic_ns();
    transform_mantissa(attempt.host_buffer, metadata.tensors.self.storage_elements,
                       metadata.tensors.self.dtype, keep_bits, &attempt);
    attempt.cpu_transform_ns = monotonic_ns() - operation_begin_ns;
    if (attempt.changed_elements == 0) {
        attempt.host_free_status = static_cast<int32_t>(g_free_host(attempt.host_buffer));
        attempt.host_freed = attempt.host_free_status == ACL_SUCCESS;
        attempt.result_status = kStatusNoChangedElements;
        return fallback("no_changed_elements");
    }

    operation_begin_ns = monotonic_ns();
    attempt.device_malloc_status = static_cast<int32_t>(g_malloc(
        &attempt.temporary, static_cast<size_t>(metadata.tensors.self.storage_bytes),
        ACL_MEM_MALLOC_HUGE_FIRST));
    attempt.device_alloc_ns = monotonic_ns() - operation_begin_ns;
    if (attempt.device_malloc_status != ACL_SUCCESS || attempt.temporary == nullptr) {
        attempt.host_free_status = static_cast<int32_t>(g_free_host(attempt.host_buffer));
        attempt.host_freed = attempt.host_free_status == ACL_SUCCESS;
        attempt.result_status = kStatusDeviceAllocFailed;
        return fallback("device_alloc_failed");
    }

    operation_begin_ns = monotonic_ns();
    attempt.h2d_status = static_cast<int32_t>(g_memcpy_async(
        attempt.temporary, static_cast<size_t>(metadata.tensors.self.storage_bytes),
        attempt.host_buffer, static_cast<size_t>(metadata.tensors.self.storage_bytes),
        ACL_MEMCPY_HOST_TO_DEVICE, stream));
    attempt.h2d_enqueue_ns = monotonic_ns() - operation_begin_ns;
    if (attempt.h2d_status != ACL_SUCCESS) {
        operation_begin_ns = monotonic_ns();
        attempt.post_sync_status = static_cast<int32_t>(g_synchronize_stream(stream));
        attempt.post_sync_ns = monotonic_ns() - operation_begin_ns;
        if (attempt.post_sync_status == ACL_SUCCESS) {
            attempt.device_free_status = static_cast<int32_t>(g_free(attempt.temporary));
            attempt.temporary_freed = attempt.device_free_status == ACL_SUCCESS;
            attempt.host_free_status = static_cast<int32_t>(g_free_host(attempt.host_buffer));
            attempt.host_freed = attempt.host_free_status == ACL_SUCCESS;
        }
        attempt.result_status = kStatusH2DFailed;
        return fallback("h2d_failed");
    }

    addresses_copy[0] = attempt.temporary;
    attempt.temporary_used = true;
    operation_begin_ns = monotonic_ns();
    const aclnnStatus run_status = run_original(cache, workspace, stream, addresses_copy);
    attempt.real_run_ns = monotonic_ns() - operation_begin_ns;
    attempt.run_status = static_cast<int32_t>(run_status);

    operation_begin_ns = monotonic_ns();
    attempt.post_sync_status = static_cast<int32_t>(g_synchronize_stream(stream));
    attempt.post_sync_ns = monotonic_ns() - operation_begin_ns;
    if (attempt.post_sync_status == ACL_SUCCESS) {
        attempt.device_free_status = static_cast<int32_t>(g_free(attempt.temporary));
        attempt.temporary_freed = attempt.device_free_status == ACL_SUCCESS;
        attempt.host_free_status = static_cast<int32_t>(g_free_host(attempt.host_buffer));
        attempt.host_freed = attempt.host_free_status == ACL_SUCCESS;
    }

    const char *outcome = "success";
    if (attempt.run_status != ACL_SUCCESS) {
        attempt.result_status = kStatusRunFailed;
        outcome = "real_run_failed";
    } else if (attempt.post_sync_status != ACL_SUCCESS) {
        attempt.result_status = kStatusPostSyncFailed;
        outcome = "post_sync_failed_buffers_retained";
    } else if (attempt.device_free_status != ACL_SUCCESS) {
        attempt.result_status = kStatusDeviceFreeFailed;
        outcome = "device_free_failed";
    } else if (attempt.host_free_status != ACL_SUCCESS) {
        attempt.result_status = kStatusHostFreeFailed;
        outcome = "host_free_failed";
    } else {
        attempt.result_status = kStatusSuccess;
    }
    set_device_last_status(control, metadata.device, attempt.result_status);
    log_mantissa_attempt(outcome, api_info, cache, metadata, attempt);
    return run_status;
}

aclnnStatus run_and_maybe_substitute(op::internal::OpExecCache *cache, void *workspace,
                                     aclrtStream stream,
                                     const std::vector<void *> &addresses) {
    pthread_once(&g_resolve_once, resolve_real);
    pthread_once(&g_control_once, initialize_control);
    if (g_real_run == nullptr) {
        log_simple("resolve_cache_run_failed", "real Run unavailable");
        return 1;
    }

    std::string api_info;
    if (g_get_log_api_info != nullptr) {
        api_info = g_get_log_api_info();
    }
    if (starts_with(api_info, "aclnnAscendQuantV3_")) {
        g_last_quant.valid = !addresses.empty() && addresses.back() != nullptr;
        g_last_quant.output = g_last_quant.valid ? addresses.back() : nullptr;
        return run_original(cache, workspace, stream, addresses);
    }
    if (starts_with(api_info, "aclnnMatmulWeightNz_")) {
        g_last_quant.valid = false;
        g_last_quant.output = nullptr;
        RegisteredMatmulMetadata metadata{};
        uint64_t run_index = 0;
        if (!g_abi_ok || !lookup_matmul_metadata(cache, &metadata, &run_index)) {
            return run_original(cache, workspace, stream, addresses);
        }
        log_matmul_run_trace(cache, metadata, api_info, addresses, run_index);
        if (g_control == nullptr || g_matmul_substitution_active ||
            __atomic_load_n(&g_control->enabled, __ATOMIC_ACQUIRE) == 0) {
            return run_original(cache, workspace, stream, addresses);
        }

        Fault20Control *control = g_control;
        resolve_runtime_api();
        const uint32_t mode = __atomic_load_n(&control->mode, __ATOMIC_RELAXED);
        if (mode != FAULT20_MODE_LOSSLESS_D2D &&
            mode != FAULT20_MODE_MANTISSA_TRUNCATE) {
            return run_original(cache, workspace, stream, addresses);
        }
        const int32_t target_m =
            __atomic_load_n(&control->target_m, __ATOMIC_RELAXED);
        const int32_t target_k =
            __atomic_load_n(&control->target_k, __ATOMIC_RELAXED);
        const uint32_t target_call_index =
            __atomic_load_n(&control->target_call_index, __ATOMIC_RELAXED);
        const uint64_t target_device_mask =
            __atomic_load_n(&control->target_device_mask, __ATOMIC_RELAXED);
        const uint32_t max_hits_per_device =
            __atomic_load_n(&control->max_hits_per_device, __ATOMIC_RELAXED);
        const uint32_t selection_mode =
            __atomic_load_n(&control->selection_mode, __ATOMIC_RELAXED);
        const uint32_t target_api =
            __atomic_load_n(&control->target_api, __ATOMIC_RELAXED);
        const uint64_t max_bytes =
            __atomic_load_n(&control->max_bytes, __ATOMIC_RELAXED);
        const int32_t target_dtype =
            __atomic_load_n(&control->target_dtype, __ATOMIC_RELAXED);
        const uint32_t keep_bits =
            __atomic_load_n(&control->keep_mantissa_bits, __ATOMIC_RELAXED);
        const int64_t expected_numel = static_cast<int64_t>(target_m) * target_k;
        const uint64_t expected_bytes = expected_numel > 0
                                            ? static_cast<uint64_t>(expected_numel) * 2U
                                            : 0;
        const auto &self = metadata.tensors.self;
        const bool common_runtime_ready = g_malloc != nullptr && g_free != nullptr &&
                                          g_memcpy_async != nullptr &&
                                          g_synchronize_stream != nullptr;
        const bool numeric_runtime_ready = common_runtime_ready &&
                                           g_malloc_host != nullptr &&
                                           g_free_host != nullptr && g_memcpy != nullptr;
        const bool param_order_ok =
            metadata.tensor_param_count == 3 &&
            metadata.tensor_params[0] == self.tensor_ptr &&
            metadata.tensor_params[1] == metadata.tensors.mat2.tensor_ptr &&
            metadata.tensor_params[2] == metadata.tensors.out.tensor_ptr;
        const bool target_device =
            metadata.device >= 0 &&
            static_cast<size_t>(metadata.device) < FAULT20_MAX_DEVICES &&
            (target_device_mask & (1ULL << static_cast<uint32_t>(metadata.device))) != 0;
        if (target_api != FAULT20_TARGET_API_MATMUL_WEIGHT_NZ ||
            target_m <= 0 || target_k <= 0 || !target_device ||
            metadata.tensors.all_present == 0 ||
            metadata.tensors.any_empty_storage != 0 ||
            metadata.tensors.any_shared_storage != 0 || !param_order_ok ||
            self.dtype != target_dtype ||
            (self.dtype != ACL_BF16 && self.dtype != ACL_FLOAT16) ||
            self.format != ACL_FORMAT_ND ||
            self.view_rank != 2 || self.storage_rank != 1 || self.stride_rank != 2 ||
            self.contiguous == 0 || self.view_offset != 0 ||
            self.view_shape[0] != target_m || self.view_shape[1] != target_k ||
            self.storage_shape[0] != expected_numel || self.strides[0] != target_k ||
            self.strides[1] != 1 || self.numel != expected_numel ||
            self.storage_elements != expected_numel || self.element_size != 2 ||
            self.view_bytes != self.storage_bytes || self.storage_bytes != expected_bytes ||
            self.storage_bytes == 0 || self.storage_bytes > max_bytes ||
            self.storage_bytes > 65536 || addresses.size() != 3 ||
            addresses[0] == nullptr || target_call_index == 0 ||
            max_hits_per_device == 0 ||
            (selection_mode != FAULT20_SELECTION_BOUNDED &&
             selection_mode != FAULT20_SELECTION_ALL_HITS) ||
            !common_runtime_ready ||
            (mode == FAULT20_MODE_MANTISSA_TRUNCATE &&
             (!numeric_runtime_ready ||
              (self.dtype == ACL_BF16 && keep_bits > 7) ||
              (self.dtype == ACL_FLOAT16 && keep_bits > 10)))) {
            return run_original(cache, workspace, stream, addresses);
        }

        const uint32_t candidate_index = __atomic_add_fetch(
            &control->candidate_count_per_device[metadata.device], 1U,
            __ATOMIC_ACQ_REL);
        __atomic_add_fetch(&control->candidate_count, 1U, __ATOMIC_ACQ_REL);
        const uint64_t target_end =
            static_cast<uint64_t>(target_call_index) + max_hits_per_device - 1U;
        if (selection_mode == FAULT20_SELECTION_BOUNDED &&
            (candidate_index < target_call_index ||
             static_cast<uint64_t>(candidate_index) > target_end)) {
            return run_original(cache, workspace, stream, addresses);
        }
        uint32_t hit_index = 0;
        if (!reserve_hit_per_device(control, metadata.device, &hit_index)) {
            return run_original(cache, workspace, stream, addresses);
        }
        g_matmul_substitution_active = true;
        const aclnnStatus status = mode == FAULT20_MODE_MANTISSA_TRUNCATE
                                       ? run_matmul_mantissa(
                                             cache, workspace, stream, addresses, control,
                                             metadata, api_info, hit_index, candidate_index,
                                             keep_bits)
                                       : run_matmul_lossless(
                                             cache, workspace, stream, addresses, control,
                                             metadata, api_info, hit_index,
                                             candidate_index);
        g_matmul_substitution_active = false;
        return status;
    }
    if (!starts_with(api_info, "aclnnQuantMatmulWeightNz_")) {
        g_last_quant.valid = false;
        g_last_quant.output = nullptr;
        return run_original(cache, workspace, stream, addresses);
    }

    const bool producer_matches = g_last_quant.valid && !addresses.empty() &&
                                  g_last_quant.output == addresses[0];
    g_last_quant.valid = false;
    g_last_quant.output = nullptr;

    RegisteredMetadata metadata{};
    if (!g_abi_ok || g_control == nullptr || !lookup_metadata(cache, &metadata) ||
        addresses.empty() || addresses[0] == nullptr || !producer_matches) {
        return run_original(cache, workspace, stream, addresses);
    }

    Fault20Control *control = g_control;
    resolve_runtime_api();
    if (__atomic_load_n(&control->enabled, __ATOMIC_ACQUIRE) == 0) {
        return run_original(cache, workspace, stream, addresses);
    }
    const uint32_t mode = __atomic_load_n(&control->mode, __ATOMIC_RELAXED);
    if (mode != FAULT20_MODE_LOSSLESS_D2D && mode != FAULT20_MODE_INT4_EQUIVALENT &&
        mode != FAULT20_MODE_INT2_EQUIVALENT) {
        return run_original(cache, workspace, stream, addresses);
    }
    const int32_t target_m = __atomic_load_n(&control->target_m, __ATOMIC_RELAXED);
    const int32_t target_k = __atomic_load_n(&control->target_k, __ATOMIC_RELAXED);
    const uint32_t target_call_index =
        __atomic_load_n(&control->target_call_index, __ATOMIC_RELAXED);
    const uint64_t target_device_mask =
        __atomic_load_n(&control->target_device_mask, __ATOMIC_RELAXED);
    const uint32_t max_hits_per_device =
        __atomic_load_n(&control->max_hits_per_device, __ATOMIC_RELAXED);
    const uint32_t selection_mode =
        __atomic_load_n(&control->selection_mode, __ATOMIC_RELAXED);
    const uint32_t target_api =
        __atomic_load_n(&control->target_api, __ATOMIC_RELAXED);
    const uint64_t max_bytes = __atomic_load_n(&control->max_bytes, __ATOMIC_RELAXED);
    const int64_t expected_numel = static_cast<int64_t>(target_m) * target_k;
    const bool common_runtime_ready = g_malloc != nullptr && g_free != nullptr &&
                                      g_memcpy_async != nullptr &&
                                      g_synchronize_stream != nullptr;
    const bool numeric_runtime_ready = common_runtime_ready && g_malloc_host != nullptr &&
                                       g_free_host != nullptr && g_memcpy != nullptr;
    const uint32_t quant_step =
        __atomic_load_n(&control->quant_step, __ATOMIC_RELAXED);
    const bool target_device =
        metadata.device >= 0 && static_cast<size_t>(metadata.device) < FAULT20_MAX_DEVICES &&
        (target_device_mask & (1ULL << static_cast<uint32_t>(metadata.device))) != 0;
    if (target_api != FAULT20_TARGET_API_QMM || !target_device ||
        metadata.x1.dtype != ACL_INT8 ||
        metadata.x1.format != ACL_FORMAT_ND || metadata.x1.view_rank != 2 ||
        metadata.x1.storage_rank != 1 || metadata.x1.stride_rank != 2 ||
        metadata.x1.contiguous == 0 || metadata.x1.view_offset != 0 ||
        metadata.x1.view_shape[0] != target_m || metadata.x1.view_shape[1] != target_k ||
        metadata.x1.storage_shape[0] != expected_numel ||
        metadata.x1.strides[0] != target_k || metadata.x1.strides[1] != 1 ||
        metadata.x1.numel != expected_numel || metadata.x1.storage_elements != expected_numel ||
        metadata.x1.element_size != 1 || metadata.x1.view_bytes != metadata.x1.storage_bytes ||
        metadata.x1.storage_bytes != static_cast<uint64_t>(expected_numel) ||
        metadata.x1.storage_bytes == 0 || metadata.x1.storage_bytes > max_bytes ||
        metadata.x1.storage_bytes > 65536 || target_call_index == 0 ||
        max_hits_per_device == 0 ||
        (selection_mode != FAULT20_SELECTION_BOUNDED &&
         selection_mode != FAULT20_SELECTION_ALL_HITS) ||
        !common_runtime_ready ||
        ((mode == FAULT20_MODE_INT4_EQUIVALENT ||
          mode == FAULT20_MODE_INT2_EQUIVALENT) &&
         (!numeric_runtime_ready ||
          (mode == FAULT20_MODE_INT4_EQUIVALENT && quant_step != 16) ||
          (mode == FAULT20_MODE_INT2_EQUIVALENT && quant_step != 64)))) {
        return run_original(cache, workspace, stream, addresses);
    }

    const uint32_t candidate_index = __atomic_add_fetch(
        &control->candidate_count_per_device[metadata.device], 1U, __ATOMIC_ACQ_REL);
    __atomic_add_fetch(&control->candidate_count, 1U, __ATOMIC_ACQ_REL);
    const uint64_t target_end =
        static_cast<uint64_t>(target_call_index) + max_hits_per_device - 1U;
    if (selection_mode == FAULT20_SELECTION_BOUNDED &&
        (candidate_index < target_call_index ||
         static_cast<uint64_t>(candidate_index) > target_end)) {
        return run_original(cache, workspace, stream, addresses);
    }

    uint32_t hit_index = 0;
    if (!reserve_hit_per_device(control, metadata.device, &hit_index)) {
        return run_original(cache, workspace, stream, addresses);
    }

    if (mode == FAULT20_MODE_INT4_EQUIVALENT ||
        mode == FAULT20_MODE_INT2_EQUIVALENT) {
        return run_numeric_equivalent(cache, workspace, stream, addresses, control,
                                      metadata, api_info, mode, hit_index,
                                      candidate_index);
    }

    const uint64_t begin_ns = monotonic_ns();
    std::vector<void *> addresses_copy;
    try {
        addresses_copy = addresses;
    } catch (...) {
        log_simple("lossless_substitute_failed", "address vector copy failed");
        return run_original(cache, workspace, stream, addresses);
    }

    void *temporary = nullptr;
    const aclError malloc_status =
        g_malloc(&temporary, static_cast<size_t>(metadata.x1.storage_bytes),
                 ACL_MEM_MALLOC_HUGE_FIRST);
    if (malloc_status != ACL_SUCCESS || temporary == nullptr) {
        char reason[128];
        std::snprintf(reason, sizeof(reason), "aclrtMalloc=%d", malloc_status);
        log_simple("lossless_substitute_failed", reason);
        return run_original(cache, workspace, stream, addresses);
    }

    const uint64_t copy_begin_ns = monotonic_ns();
    const aclError copy_status = g_memcpy_async(
        temporary, static_cast<size_t>(metadata.x1.storage_bytes), addresses[0],
        static_cast<size_t>(metadata.x1.storage_bytes), ACL_MEMCPY_DEVICE_TO_DEVICE, stream);
    const uint64_t copy_end_ns = monotonic_ns();
    if (copy_status != ACL_SUCCESS) {
        const aclError free_status = g_free(temporary);
        char reason[160];
        std::snprintf(reason, sizeof(reason), "aclrtMemcpyAsync=%d aclrtFree=%d", copy_status,
                      free_status);
        log_simple("lossless_substitute_failed", reason);
        return run_original(cache, workspace, stream, addresses);
    }

    addresses_copy[0] = temporary;
    const uint64_t run_begin_ns = monotonic_ns();
    const aclnnStatus run_status = run_original(cache, workspace, stream, addresses_copy);
    const uint64_t run_end_ns = monotonic_ns();
    const uint64_t sync_begin_ns = monotonic_ns();
    const aclError sync_status = g_synchronize_stream(stream);
    const uint64_t sync_end_ns = monotonic_ns();
    aclError free_status = ACL_SUCCESS;
    bool freed = false;
    if (sync_status == ACL_SUCCESS) {
        free_status = g_free(temporary);
        freed = free_status == ACL_SUCCESS;
    }

    char line[3072];
    const int written = std::snprintf(
        line, sizeof(line),
        "{\"event\":\"lossless_substitution\",\"pid\":%ld,\"tid\":%ld,"
        "\"ts_ns\":%" PRIu64 ",\"device\":%d,\"hit\":%u,\"candidate\":%u,"
        "\"cache\":\"%p\","
        "\"api_info\":\"%s\",\"shape\":[%" PRId64 ",%" PRId64
        "],\"bytes\":%" PRIu64 ",\"original\":\"%p\",\"temporary\":\"%p\","
        "\"producer_match\":true,\"changed_elements\":0,\"malloc_status\":%d,"
        "\"copy_status\":%d,\"run_status\":%d,\"sync_status\":%d,"
        "\"free_status\":%d,\"freed\":%s,\"source_aligned_64\":%s,"
        "\"temporary_aligned_64\":%s,\"d2d_enqueue_ns\":%" PRIu64
        ",\"real_run_ns\":%" PRIu64 ",\"sync_ns\":%" PRIu64
        ",\"total_ns\":%" PRIu64 "}",
        static_cast<long>(getpid()), static_cast<long>(syscall(SYS_gettid)), begin_ns,
        metadata.device, hit_index, candidate_index, cache, api_info.c_str(), metadata.x1.view_shape[0],
        metadata.x1.view_shape[1], metadata.x1.storage_bytes, addresses[0], temporary,
        malloc_status, copy_status, static_cast<int>(run_status), sync_status, free_status,
        freed ? "true" : "false",
        (reinterpret_cast<uintptr_t>(addresses[0]) % 64U) == 0 ? "true" : "false",
        (reinterpret_cast<uintptr_t>(temporary) % 64U) == 0 ? "true" : "false",
        copy_end_ns - copy_begin_ns, run_end_ns - run_begin_ns, sync_end_ns - sync_begin_ns,
        sync_end_ns - begin_ns);
    if (written > 0 && static_cast<size_t>(written) < sizeof(line)) {
        write_line(line, static_cast<size_t>(written));
    }
    set_device_last_status(
        control, metadata.device,
        (run_status == ACL_SUCCESS && sync_status == ACL_SUCCESS && freed)
            ? kStatusSuccess
            : kStatusRunFailed);
    return run_status;
}

bool add_and_capture(op::internal::OpExecCache *cache) {
    pthread_once(&g_resolve_once, resolve_real);
    if (g_capture.active) {
        ++g_capture.add_count;
        if (g_capture.add_count == 1) {
            g_capture.cache = cache;
        }
    }
    if (g_real_add == nullptr) {
        log_simple("resolve_cache_add_failed", "real AddOpExecCache unavailable");
        return false;
    }
    return g_real_add(cache);
}

}  // namespace

extern "C" void fault20_qmm_gws_begin() {
    g_capture = CaptureState{};
    g_capture.active = true;
    g_capture.kind = CaptureKind::kQmm;
    g_capture.add_count = 0;
    g_capture.cache = nullptr;
}

extern "C" bool fault20_qmm_gws_commit(const Fault20X1Metadata *metadata) {
    pthread_once(&g_resolve_once, resolve_real);
    resolve_runtime_api();
    const uint32_t add_count = g_capture.add_count;
    op::internal::OpExecCache *cache = g_capture.cache;
    const CaptureKind kind = g_capture.kind;
    g_capture = CaptureState{};
    if (kind != CaptureKind::kQmm || metadata == nullptr || add_count != 1 ||
        cache == nullptr) {
        log_registration("qmm_cache_registration_rejected", cache, metadata, -1, add_count);
        return false;
    }
    int32_t device = -1;
    if (g_get_device == nullptr || g_get_device(&device) != ACL_SUCCESS) {
        log_registration("qmm_cache_registration_rejected", cache, metadata, device, add_count);
        return false;
    }
    RegisteredMetadata registered{};
    registered.x1 = *metadata;
    registered.device = device;
    {
        std::lock_guard<std::mutex> lock(registry_mutex());
        registry()[cache] = registered;
    }
    log_registration("qmm_cache_registered", cache, metadata, device, add_count);
    return true;
}

extern "C" void fault20_qmm_gws_abort() {
    g_capture = CaptureState{};
}

extern "C" void fault20_matmul_gws_begin() {
    g_capture = CaptureState{};
    g_capture.active = true;
    g_capture.kind = CaptureKind::kMatmulWeightNz;
    g_capture.add_count = 0;
    g_capture.cache = nullptr;
}

extern "C" bool fault20_matmul_gws_commit(const Fault20MatmulMetadata *metadata) {
    pthread_once(&g_resolve_once, resolve_real);
    resolve_runtime_api();
    const uint32_t add_count = g_capture.add_count;
    op::internal::OpExecCache *cache = g_capture.cache;
    const CaptureKind kind = g_capture.kind;
    const uint32_t tensor_param_count = g_capture.tensor_param_count;
    uintptr_t tensor_params[8]{};
    std::memcpy(tensor_params, g_capture.tensor_params, sizeof(tensor_params));
    g_capture = CaptureState{};
    const bool param_order_ok =
        metadata != nullptr && tensor_param_count == 3 &&
        tensor_params[0] == metadata->self.tensor_ptr &&
        tensor_params[1] == metadata->mat2.tensor_ptr &&
        tensor_params[2] == metadata->out.tensor_ptr;
    const bool registration_ok =
        kind == CaptureKind::kMatmulWeightNz && metadata != nullptr && add_count == 1 &&
        cache != nullptr && metadata->all_present != 0 &&
        metadata->any_empty_storage == 0 && metadata->any_shared_storage == 0 &&
        param_order_ok;
    log_matmul_param_collection(cache, metadata, tensor_param_count, tensor_params,
                                registration_ok);
    if (!registration_ok) {
        log_matmul_registration("matmulweightnz_cache_registration_rejected", cache,
                                metadata, -1, add_count);
        return false;
    }
    int32_t device = -1;
    if (g_get_device == nullptr || g_get_device(&device) != ACL_SUCCESS) {
        log_matmul_registration("matmulweightnz_cache_registration_rejected", cache,
                                metadata, device, add_count);
        return false;
    }
    RegisteredMatmulMetadata registered{};
    registered.tensors = *metadata;
    registered.device = device;
    registered.tensor_param_count = tensor_param_count;
    std::memcpy(registered.tensor_params, tensor_params, sizeof(tensor_params));
    {
        std::lock_guard<std::mutex> lock(registry_mutex());
        matmul_registry()[cache] = registered;
    }
    log_matmul_registration("matmulweightnz_cache_registered", cache, metadata, device,
                            add_count);
    return true;
}

extern "C" void fault20_matmul_gws_abort() {
    g_capture = CaptureState{};
}

namespace op::internal {
void AddParamToBuf(const aclTensor *tensor) {
    add_param_const_and_capture(tensor);
}

void AddParamToBuf(aclTensor *tensor) {
    add_param_and_capture(tensor);
}

aclnnStatus OpExecCache::Run(void *workspace, const aclrtStream stream,
                             const std::vector<void *> &addresses) {
    return run_and_maybe_substitute(this, workspace, stream, addresses);
}

bool AddOpExecCache(OpExecCache *cache) {
    return add_and_capture(cache);
}
}  // namespace op::internal
