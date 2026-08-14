#include "fault20_buffer_api.h"
#include "extreme_transform.h"
#include "int_equivalent_transform.h"
#include "mantissa_truncate.h"

#include <acl/acl.h>
#include <aclnn/opdev/op_cache.h>

#include <algorithm>
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
#ifndef FAULT20_EXPECTED_LIBRARY_PATH_FRAGMENT
#define FAULT20_EXPECTED_LIBRARY_PATH_FRAGMENT "/8.3.RC2/"
#endif
constexpr const char *kExpectedLibraryPathFragment =
    FAULT20_EXPECTED_LIBRARY_PATH_FRAGMENT;

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
    Fault20QmmMetadata tensors{};
    int32_t device{-1};
    uint64_t run_count{0};
    uint32_t tensor_param_count{0};
    uintptr_t tensor_params[16]{};
};

struct RegisteredMatmulMetadata {
    Fault20MatmulMetadata tensors{};
    int32_t device{-1};
    uint64_t run_count{0};
    uint32_t tensor_param_count{0};
    uintptr_t tensor_params[16]{};
    uint32_t api_kind{0};
};

enum class MatmulApiKind : uint32_t {
    kUnknown = 0,
    kWeightNz = 1,
    kFloat = 2,
};

enum class CaptureKind : uint32_t {
    kNone = 0,
    kQmm = 1,
    kMatmulWeightNz = 2,
    kFloatMatmul = 3,
};

struct CaptureState {
    bool active{false};
    CaptureKind kind{CaptureKind::kNone};
    uint32_t add_count{0};
    op::internal::OpExecCache *cache{nullptr};
    uint32_t tensor_param_count{0};
    uintptr_t tensor_params[16]{};
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
constexpr int32_t kStatusOutputSlotUnproven = -12;
constexpr int32_t kStatusOutputMetadataRejected = -13;
constexpr int32_t kStatusWeightSlotUnproven = -14;
constexpr int32_t kStatusWeightMetadataRejected = -15;
constexpr int32_t kStatusScaleSlotUnproven = -16;
constexpr int32_t kStatusScaleMetadataRejected = -17;
constexpr int32_t kNotCalled = -2147483647;

struct Int4Attempt {
    uint32_t hit_index{0};
    uint32_t candidate_index{0};
    uint32_t quant_step{16};
    int32_t extreme_value{0};
    uint32_t extreme_ratio_bps{0};
    uint32_t extreme_pattern{0};
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
    uint64_t selected_elements{0};
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

struct QmmOutputObservation {
    uint32_t hit_index{0};
    uint32_t candidate_index{0};
    int32_t out_slot{-1};
    int32_t result_status{0};
    int32_t run_status{kNotCalled};
    int32_t sync_status{kNotCalled};
    int32_t host_malloc_status{kNotCalled};
    int32_t d2h_status{kNotCalled};
    int32_t host_free_status{kNotCalled};
    void *out_address{nullptr};
    void *host_buffer{nullptr};
    bool host_freed{false};
    bool stats_valid{false};
    bool auto_disarmed_on_nonfinite{false};
    uint64_t finite_count{0};
    uint64_t nan_count{0};
    uint64_t positive_inf_count{0};
    uint64_t negative_inf_count{0};
    uint64_t zero_count{0};
    double min_value{0.0};
    double max_value{0.0};
    double max_abs{0.0};
    double mean_abs{0.0};
    uint64_t begin_ns{0};
    uint64_t real_run_ns{0};
    uint64_t sync_ns{0};
    uint64_t host_alloc_ns{0};
    uint64_t d2h_ns{0};
    uint64_t scan_ns{0};
};

struct QmmWeightObservation {
    uint32_t hit_index{0};
    uint32_t candidate_index{0};
    int32_t x2_slot{-1};
    int32_t result_status{0};
    int32_t run_status{kNotCalled};
    void *x2_address{nullptr};
    uint64_t begin_ns{0};
    uint64_t real_run_ns{0};
};

struct QmmScaleObservation {
    uint32_t hit_index{0};
    uint32_t candidate_index{0};
    int32_t scale_slot{-1};
    int32_t result_status{0};
    int32_t run_status{kNotCalled};
    void *scale_address{nullptr};
    uint64_t begin_ns{0};
    uint64_t real_run_ns{0};
};

struct QmmScaleSubstitutionAttempt {
    uint32_t hit_index{0};
    uint32_t candidate_index{0};
    int32_t scale_slot{-1};
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
    void *scale_original{nullptr};
    void *scale_temporary{nullptr};
    void *scale_host{nullptr};
    bool temporary_used{false};
    bool temporary_freed{false};
    bool host_freed{false};
    bool stats_valid{false};
    bool after_stats_valid{false};
    double scale_factor{1.0};
    uint64_t changed_elements{0};
    uint64_t finite_count{0};
    uint64_t nan_count{0};
    uint64_t positive_inf_count{0};
    uint64_t negative_inf_count{0};
    uint64_t zero_count{0};
    double min_value{0.0};
    double max_value{0.0};
    double max_abs{0.0};
    uint64_t after_finite_count{0};
    uint64_t after_nan_count{0};
    uint64_t after_positive_inf_count{0};
    uint64_t after_negative_inf_count{0};
    uint64_t after_zero_count{0};
    double after_min_value{0.0};
    double after_max_value{0.0};
    double after_max_abs{0.0};
    uint64_t begin_ns{0};
    uint64_t pre_sync_ns{0};
    uint64_t host_alloc_ns{0};
    uint64_t d2h_ns{0};
    uint64_t scan_ns{0};
    uint64_t device_alloc_ns{0};
    uint64_t h2d_ns{0};
    uint64_t real_run_ns{0};
    uint64_t post_sync_ns{0};
};

struct DualExtremeAttempt {
    uint32_t hit_index{0};
    uint32_t candidate_index{0};
    int32_t x1_slot{-1};
    int32_t x2_slot{-1};
    int32_t result_status{0};
    int32_t pre_sync_status{kNotCalled};
    int32_t x1_host_malloc_status{kNotCalled};
    int32_t x2_host_malloc_status{kNotCalled};
    int32_t x1_d2h_status{kNotCalled};
    int32_t x2_d2h_status{kNotCalled};
    int32_t x1_device_malloc_status{kNotCalled};
    int32_t x2_device_malloc_status{kNotCalled};
    int32_t x1_h2d_status{kNotCalled};
    int32_t x2_h2d_status{kNotCalled};
    int32_t run_status{kNotCalled};
    int32_t post_sync_status{kNotCalled};
    int32_t x1_device_free_status{kNotCalled};
    int32_t x2_device_free_status{kNotCalled};
    int32_t x1_host_free_status{kNotCalled};
    int32_t x2_host_free_status{kNotCalled};
    void *x1_original{nullptr};
    void *x2_original{nullptr};
    void *x1_temporary{nullptr};
    void *x2_temporary{nullptr};
    void *x1_host{nullptr};
    void *x2_host{nullptr};
    bool x1_temporary_used{false};
    bool x2_temporary_used{false};
    bool x1_temporary_freed{false};
    bool x2_temporary_freed{false};
    bool x1_host_freed{false};
    bool x2_host_freed{false};
    bool extreme{false};
    uint64_t x1_changed_elements{0};
    uint64_t x2_changed_elements{0};
    uint64_t begin_ns{0};
    uint64_t pre_sync_ns{0};
    uint64_t host_alloc_ns{0};
    uint64_t d2h_ns{0};
    uint64_t transform_ns{0};
    uint64_t device_alloc_ns{0};
    uint64_t h2d_ns{0};
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
bool g_developer_mode_ok = false;
bool g_runtime_resolution_logged = false;
bool g_api_inventory_enabled = false;
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

void log_op_exec_api_inventory(op::internal::OpExecCache *cache,
                               const std::string &api_info,
                               const std::vector<void *> &addresses) {
    if (!g_api_inventory_enabled || !g_abi_ok || !g_developer_mode_ok) {
        return;
    }
    int32_t device = -1;
    if (g_get_device != nullptr) {
        (void)g_get_device(&device);
    }
    char line[2048];
    const int written = std::snprintf(
        line, sizeof(line),
        "{\"event\":\"op_exec_api_inventory\",\"pid\":%ld,\"tid\":%ld,"
        "\"ts_ns\":%" PRIu64 ",\"device\":%d,\"cache\":\"%p\","
        "\"api_info\":\"%s\",\"address_count\":%zu}",
        static_cast<long>(getpid()), static_cast<long>(syscall(SYS_gettid)),
        monotonic_ns(), device, cache, api_info.c_str(), addresses.size());
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
            control->extreme_value = 127;
            control->extreme_ratio_bps = 10000;
            control->extreme_pattern = FAULT20_EXTREME_POSITIVE_MAX;
            control->target_dtype = ACL_BF16;
            control->max_bytes = 5120;
            control->max_output_bytes = 65536;
            control->max_weight_bytes = 268435456;
            control->target_x2_n = 0;
            control->max_scale_bytes = 1048576;
            control->target_scale_elements = 0;
            control->scale_factor = 1.0;
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
    const char *developer_mode = std::getenv("VLLM_SERVER_DEV_MODE");
    g_developer_mode_ok = developer_mode != nullptr &&
                          std::strcmp(developer_mode, "1") == 0;
    const char *api_inventory = std::getenv("FAULT20_API_INVENTORY");
    g_api_inventory_enabled = api_inventory != nullptr &&
                              std::strcmp(api_inventory, "1") == 0;
    g_abi_ok = path_ok && g_real_add != nullptr && g_get_log_api_info != nullptr &&
               g_real_add_param_const_tensor != nullptr && g_real_add_param_tensor != nullptr;

    char line[1536];
    const int written = std::snprintf(
        line, sizeof(line),
        "{\"event\":\"abi_validation\",\"pid\":%ld,\"tid\":%ld,\"ts_ns\":%" PRIu64
        ",\"abi_ok\":%s,\"developer_mode_ok\":%s,\"run\":%s,\"add\":%s,"
        "\"api_info\":%s,\"runtime_api\":%s,"
        "\"expected_library_fragment\":\"%s\",\"library\":\"%s\"}",
        static_cast<long>(getpid()), static_cast<long>(syscall(SYS_gettid)), monotonic_ns(),
        g_abi_ok ? "true" : "false", g_developer_mode_ok ? "true" : "false",
        g_real_run != nullptr ? "true" : "false",
        g_real_add != nullptr ? "true" : "false", g_get_log_api_info != nullptr ? "true" : "false",
        (g_get_device != nullptr && g_malloc != nullptr && g_free != nullptr &&
         g_malloc_host != nullptr && g_free_host != nullptr && g_memcpy != nullptr &&
         g_memcpy_async != nullptr && g_synchronize_stream != nullptr) ? "true" : "false",
        kExpectedLibraryPathFragment, validated_path);
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

bool lookup_metadata(op::internal::OpExecCache *cache, RegisteredMetadata *metadata,
                     uint64_t *run_index) {
    std::lock_guard<std::mutex> lock(registry_mutex());
    const auto found = registry().find(cache);
    if (found == registry().end()) {
        return false;
    }
    ++found->second.run_count;
    *metadata = found->second;
    *run_index = found->second.run_count;
    return true;
}

bool lookup_matmul_metadata(op::internal::OpExecCache *cache,
                            MatmulApiKind expected_api_kind,
                            RegisteredMatmulMetadata *metadata, uint64_t *run_index) {
    std::lock_guard<std::mutex> lock(registry_mutex());
    const auto found = matmul_registry().find(cache);
    if (found == matmul_registry().end() ||
        found->second.api_kind != static_cast<uint32_t>(expected_api_kind)) {
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

void log_matmul_param_collection(MatmulApiKind api_kind,
                                 op::internal::OpExecCache *cache,
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
    const char *event = api_kind == MatmulApiKind::kFloat
                            ? "float_matmul_param_collection"
                            : "matmulweightnz_param_collection";
    char line[2048];
    const int written = std::snprintf(
        line, sizeof(line),
        "{\"event\":\"%s\",\"pid\":%ld,\"tid\":%ld,"
        "\"ts_ns\":%" PRIu64 ",\"cache\":\"%p\",\"accepted\":%s,"
        "\"tensor_param_count\":%u,\"tensor_params\":[\"0x%" PRIxPTR
        "\",\"0x%" PRIxPTR "\",\"0x%" PRIxPTR "\"],"
        "\"formal_tensors\":{\"self\":\"0x%" PRIx64
        "\",\"mat2\":\"0x%" PRIx64 "\",\"out\":\"0x%" PRIx64 "\"},"
        "\"order_equality\":{\"count_is_three\":%s,\"param0_self\":%s,"
        "\"param1_mat2\":%s,\"param2_out\":%s},"
        "\"no_empty_storage\":%s,\"no_shared_storage\":%s}",
        event, static_cast<long>(getpid()), static_cast<long>(syscall(SYS_gettid)),
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

void capture_tensor_param(const aclTensor *tensor, const char *overload) {
    if (!g_capture.active || g_capture.kind == CaptureKind::kNone) {
        return;
    }
    const uint32_t ordinal = g_capture.tensor_param_count++;
    if (ordinal < 16) {
        g_capture.tensor_params[ordinal] = reinterpret_cast<uintptr_t>(tensor);
    }
    const char *event = "capture_add_param_tensor";
    if (g_capture.kind == CaptureKind::kQmm) {
        event = "qmm_add_param_tensor";
    } else if (g_capture.kind == CaptureKind::kMatmulWeightNz) {
        event = "matmulweightnz_add_param_tensor";
    } else if (g_capture.kind == CaptureKind::kFloatMatmul) {
        event = "float_matmul_add_param_tensor";
    }
    char line[768];
    const int written = std::snprintf(
        line, sizeof(line),
        "{\"event\":\"%s\",\"pid\":%ld,"
        "\"tid\":%ld,\"ts_ns\":%" PRIu64 ",\"ordinal\":%u,"
        "\"overload\":\"%s\",\"tensor\":\"%p\"}",
        event, static_cast<long>(getpid()), static_cast<long>(syscall(SYS_gettid)),
        monotonic_ns(), ordinal, overload, tensor);
    if (written > 0 && static_cast<size_t>(written) < sizeof(line)) {
        write_line(line, static_cast<size_t>(written));
    }
}

void add_param_const_and_capture(const aclTensor *tensor) {
    pthread_once(&g_resolve_once, resolve_real);
    if (!g_add_param_forwarding) {
        capture_tensor_param(tensor, "const");
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
        capture_tensor_param(tensor, "mutable");
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

int32_t find_unique_tensor_ordinal(const uintptr_t *params, uint32_t count,
                                   uint64_t tensor_ptr) {
    if (tensor_ptr == 0 || params == nullptr || count > 16) {
        return -1;
    }
    int32_t result = -1;
    for (uint32_t i = 0; i < count; ++i) {
        if (params[i] != tensor_ptr) {
            continue;
        }
        if (result >= 0) {
            return -2;
        }
        result = static_cast<int32_t>(i);
    }
    return result;
}

int32_t find_unique_address_slot(const std::vector<void *> &addresses,
                                 const Fault20TensorMetadata &tensor) {
    if (tensor.present == 0) {
        return -1;
    }
    int32_t result = -1;
    for (size_t i = 0; i < addresses.size(); ++i) {
        const bool matches = address_equals(tensor.raw_addr, addresses[i]) ||
                             address_equals(tensor.storage_addr, addresses[i]) ||
                             address_equals(tensor.data_addr, addresses[i]);
        if (!matches) {
            continue;
        }
        if (result >= 0) {
            return -2;
        }
        result = static_cast<int32_t>(i);
    }
    return result;
}

bool qmm_tensor_storage_is_unique(const Fault20QmmMetadata &metadata,
                                  const Fault20TensorMetadata &target) {
    if (target.present == 0 || target.storage_ptr == 0) {
        return false;
    }
    const Fault20TensorMetadata *all[] = {
        &metadata.x1,        &metadata.x2,       &metadata.x1_scale,
        &metadata.x2_scale,  &metadata.y_scale,  &metadata.x1_offset,
        &metadata.x2_offset, &metadata.y_offset, &metadata.bias,
        &metadata.out,
    };
    for (const Fault20TensorMetadata *other : all) {
        if (other == &target) {
            continue;
        }
        if (other->present != 0 && other->storage_ptr != 0 &&
            other->storage_ptr == target.storage_ptr) {
            return false;
        }
    }
    return true;
}

bool qmm_output_storage_is_unique(const Fault20QmmMetadata &metadata) {
    return qmm_tensor_storage_is_unique(metadata, metadata.out);
}

struct QmmStorageOrderProof {
    int32_t slots[10]{};
    uint32_t compact_storage_count{0};
    bool param_order_exact{false};
    bool address_count_exact{false};
    bool empty_slots_are_null{false};
    bool nonempty_slots_are_nonnull{false};
    bool no_shared_storage{false};
    bool out_is_last_storage{false};
    bool x2_storage_unique{false};
    bool x2_scale_storage_unique{false};
    bool weight_slot_proven{false};
    bool x2_scale_slot_proven{false};
    bool output_slot_proven{false};
};

QmmStorageOrderProof prove_qmm_storage_order(
    const RegisteredMetadata &registered, const std::vector<void *> &addresses) {
    QmmStorageOrderProof proof{};
    const Fault20QmmMetadata &m = registered.tensors;
    const Fault20TensorMetadata *formal[] = {
        &m.x1,       &m.x2,        &m.x1_scale, &m.x2_scale, &m.y_scale,
        &m.x1_offset, &m.x2_offset, &m.y_offset, &m.bias,     &m.out,
    };
    proof.param_order_exact = registered.tensor_param_count == 10;
    proof.empty_slots_are_null = true;
    proof.nonempty_slots_are_nonnull = true;
    for (size_t i = 0; i < 10; ++i) {
        proof.slots[i] = -1;
        if (registered.tensor_param_count == 10 &&
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
            continue;
        }
        if (formal[i]->storage_bytes == 0) {
            proof.empty_slots_are_null =
                proof.empty_slots_are_null && addresses[slot] == nullptr;
        } else {
            proof.nonempty_slots_are_nonnull =
                proof.nonempty_slots_are_nonnull && addresses[slot] != nullptr;
        }
    }
    proof.address_count_exact = proof.compact_storage_count == addresses.size();
    proof.no_shared_storage = m.any_shared_storage == 0;
    proof.out_is_last_storage =
        proof.slots[9] >= 0 &&
        static_cast<size_t>(proof.slots[9] + 1) == addresses.size();
    const int32_t out_param_ordinal = find_unique_tensor_ordinal(
        registered.tensor_params, registered.tensor_param_count, m.out.tensor_ptr);
    const int32_t x2_param_ordinal = find_unique_tensor_ordinal(
        registered.tensor_params, registered.tensor_param_count, m.x2.tensor_ptr);
    const int32_t x2_scale_param_ordinal = find_unique_tensor_ordinal(
        registered.tensor_params, registered.tensor_param_count,
        m.x2_scale.tensor_ptr);
    proof.x2_storage_unique = qmm_tensor_storage_is_unique(m, m.x2);
    proof.x2_scale_storage_unique =
        qmm_tensor_storage_is_unique(m, m.x2_scale);
    proof.weight_slot_proven =
        proof.param_order_exact && proof.address_count_exact &&
        proof.empty_slots_are_null && proof.nonempty_slots_are_nonnull &&
        proof.no_shared_storage && x2_param_ordinal == 1 && proof.slots[1] == 1 &&
        m.x2.present != 0 && m.x2.storage_bytes != 0 &&
        m.x2.storage_ptr != 0 && proof.x2_storage_unique;
    proof.x2_scale_slot_proven =
        proof.param_order_exact && proof.address_count_exact &&
        proof.empty_slots_are_null && proof.nonempty_slots_are_nonnull &&
        proof.no_shared_storage && x2_scale_param_ordinal == 3 &&
        proof.slots[3] >= 0 &&
        static_cast<size_t>(proof.slots[3]) < addresses.size() &&
        addresses[static_cast<size_t>(proof.slots[3])] != nullptr &&
        m.x2_scale.present != 0 && m.x2_scale.storage_bytes != 0 &&
        m.x2_scale.storage_ptr != 0 && proof.x2_scale_storage_unique;
    proof.output_slot_proven =
        proof.param_order_exact && proof.address_count_exact &&
        proof.empty_slots_are_null && proof.nonempty_slots_are_nonnull &&
        proof.no_shared_storage && out_param_ordinal == 9 &&
        proof.out_is_last_storage && m.out.present != 0 &&
        m.out.storage_bytes != 0 && m.out.storage_ptr != 0 &&
        qmm_output_storage_is_unique(m);
    return proof;
}

void log_qmm_param_collection(op::internal::OpExecCache *cache,
                              const Fault20QmmMetadata *metadata,
                              uint32_t tensor_param_count,
                              const uintptr_t tensor_params[16], bool accepted) {
    const Fault20TensorMetadata empty{};
    const Fault20QmmMetadata empty_qmm{};
    const Fault20QmmMetadata &m = metadata == nullptr ? empty_qmm : *metadata;
    const Fault20TensorMetadata *formal[] = {
        &m.x1,       &m.x2,        &m.x1_scale, &m.x2_scale, &m.y_scale,
        &m.x1_offset, &m.x2_offset, &m.y_offset, &m.bias,     &m.out,
    };
    int32_t ordinals[10]{};
    for (size_t i = 0; i < 10; ++i) {
        const Fault20TensorMetadata &tensor = metadata == nullptr ? empty : *formal[i];
        ordinals[i] = find_unique_tensor_ordinal(
            tensor_params, tensor_param_count, tensor.tensor_ptr);
    }
    char line[4096];
    const int written = std::snprintf(
        line, sizeof(line),
        "{\"event\":\"qmm_param_collection\",\"pid\":%ld,\"tid\":%ld,"
        "\"ts_ns\":%" PRIu64 ",\"cache\":\"%p\",\"accepted\":%s,"
        "\"tensor_param_count\":%u,\"present_count\":%u,"
        "\"formal_ordinals\":{\"x1\":%d,\"x2\":%d,\"x1_scale\":%d,"
        "\"x2_scale\":%d,\"y_scale\":%d,\"x1_offset\":%d,"
        "\"x2_offset\":%d,\"y_offset\":%d,\"bias\":%d,\"out\":%d},"
        "\"out_param_unique\":%s,\"out_storage_unique\":%s,"
        "\"any_empty_storage\":%s,\"any_shared_storage\":%s}",
        static_cast<long>(getpid()), static_cast<long>(syscall(SYS_gettid)),
        monotonic_ns(), cache, accepted ? "true" : "false", tensor_param_count,
        metadata == nullptr ? 0U : metadata->present_count, ordinals[0], ordinals[1],
        ordinals[2], ordinals[3], ordinals[4], ordinals[5], ordinals[6], ordinals[7],
        ordinals[8], ordinals[9], ordinals[9] >= 0 ? "true" : "false",
        metadata != nullptr && qmm_output_storage_is_unique(*metadata) ? "true" : "false",
        metadata != nullptr && metadata->any_empty_storage == 0 ? "false" : "true",
        metadata != nullptr && metadata->any_shared_storage == 0 ? "false" : "true");
    if (written > 0 && static_cast<size_t>(written) < sizeof(line)) {
        write_line(line, static_cast<size_t>(written));
    }
}

bool log_qmm_run_trace(op::internal::OpExecCache *cache,
                       const RegisteredMetadata &registered,
                       const std::string &api_info,
                       const std::vector<void *> &addresses, uint64_t run_index) {
    const Fault20QmmMetadata &m = registered.tensors;
    const QmmStorageOrderProof proof = prove_qmm_storage_order(registered, addresses);
    const int32_t direct_out_address_match = find_unique_address_slot(addresses, m.out);
    const int32_t out_slot = proof.slots[9];
    const int32_t out_param_ordinal = find_unique_tensor_ordinal(
        registered.tensor_params, registered.tensor_param_count, m.out.tensor_ptr);
    const bool out_storage_unique = qmm_output_storage_is_unique(m);
    const void *out_runtime_address =
        out_slot >= 0 && static_cast<size_t>(out_slot) < addresses.size()
            ? addresses[static_cast<size_t>(out_slot)]
            : nullptr;
    char line[8192];
    const int written = std::snprintf(
        line, sizeof(line),
        "{\"event\":\"qmm_run_trace\",\"pid\":%ld,\"tid\":%ld,"
        "\"ts_ns\":%" PRIu64 ",\"device\":%d,\"cache\":\"%p\","
        "\"run_index\":%" PRIu64 ",\"api_info\":\"%s\","
        "\"address_count\":%zu,\"compact_storage_count\":%u,"
        "\"storage_order_slots\":{\"x1\":%d,\"x2\":%d,"
        "\"x1_scale\":%d,\"x2_scale\":%d,\"y_scale\":%d,"
        "\"x1_offset\":%d,\"x2_offset\":%d,\"y_offset\":%d,"
        "\"bias\":%d,\"out\":%d},\"out_param_ordinal\":%d,"
        "\"proof\":{\"param_order_exact\":%s,\"address_count_exact\":%s,"
        "\"empty_slots_are_null\":%s,\"nonempty_slots_are_nonnull\":%s,"
        "\"no_shared_storage\":%s,\"out_is_last_storage\":%s,"
        "\"direct_gws_address_match\":%d},"
        "\"out_runtime_address\":\"%p\",\"out_raw_addr\":\"0x%" PRIx64
        "\",\"out_storage_addr\":\"0x%" PRIx64
        "\",\"out_data_addr\":\"0x%" PRIx64 "\","
        "\"out_storage_unique\":%s,\"output_slot_proven\":%s,"
        "\"out\":{\"dtype\":%d,\"format\":%d,\"view_rank\":%u,"
        "\"storage_rank\":%u,\"stride_rank\":%u,\"contiguous\":%s,"
        "\"shape\":[%" PRId64 ",%" PRId64 "],\"storage_shape\":[%" PRId64
        ",%" PRId64 "],\"strides\":[%" PRId64 ",%" PRId64 "],"
        "\"view_offset\":%" PRId64 ",\"numel\":%" PRId64
        ",\"element_size\":%" PRIu64 ",\"view_bytes\":%" PRIu64
        ",\"storage_bytes\":%" PRIu64 "}}",
        static_cast<long>(getpid()), static_cast<long>(syscall(SYS_gettid)),
        monotonic_ns(), registered.device, cache, run_index, api_info.c_str(),
        addresses.size(), proof.compact_storage_count, proof.slots[0],
        proof.slots[1], proof.slots[2], proof.slots[3], proof.slots[4],
        proof.slots[5], proof.slots[6], proof.slots[7], proof.slots[8],
        proof.slots[9], out_param_ordinal,
        proof.param_order_exact ? "true" : "false",
        proof.address_count_exact ? "true" : "false",
        proof.empty_slots_are_null ? "true" : "false",
        proof.nonempty_slots_are_nonnull ? "true" : "false",
        proof.no_shared_storage ? "true" : "false",
        proof.out_is_last_storage ? "true" : "false", direct_out_address_match,
        out_runtime_address, m.out.raw_addr,
        m.out.storage_addr, m.out.data_addr, out_storage_unique ? "true" : "false",
        proof.output_slot_proven ? "true" : "false", m.out.dtype, m.out.format,
        m.out.view_rank, m.out.storage_rank, m.out.stride_rank,
        m.out.contiguous ? "true" : "false", m.out.view_shape[0],
        m.out.view_shape[1], m.out.storage_shape[0], m.out.storage_shape[1],
        m.out.strides[0], m.out.strides[1], m.out.view_offset, m.out.numel,
        m.out.element_size, m.out.view_bytes, m.out.storage_bytes);
    if (written > 0 && static_cast<size_t>(written) < sizeof(line)) {
        write_line(line, static_cast<size_t>(written));
    }
    return proof.output_slot_proven;
}

bool matmul_tensor_storage_is_unique(const Fault20MatmulMetadata &metadata,
                                     const Fault20TensorMetadata &target) {
    if (target.present == 0 || target.storage_ptr == 0) {
        return false;
    }
    const Fault20TensorMetadata *all[] = {
        &metadata.self,
        &metadata.mat2,
        &metadata.out,
    };
    for (const Fault20TensorMetadata *other : all) {
        if (other == &target) {
            continue;
        }
        if (other->present != 0 && other->storage_ptr != 0 &&
            other->storage_ptr == target.storage_ptr) {
            return false;
        }
    }
    return true;
}

struct MatmulStorageOrderProof {
    int32_t slots[3]{-1, -1, -1};
    int32_t param_ordinals[3]{-1, -1, -1};
    uint32_t compact_storage_count{0};
    bool param_order_exact{false};
    bool address_count_exact{false};
    bool all_present{false};
    bool no_empty_storage{false};
    bool nonempty_slots_are_nonnull{false};
    bool no_shared_storage{false};
    bool self_storage_unique{false};
    bool mat2_storage_unique{false};
    bool out_storage_unique{false};
    bool self_slot_proven{false};
    bool mat2_slot_proven{false};
    bool output_slot_proven{false};
};

MatmulStorageOrderProof prove_matmul_storage_order(
    const RegisteredMatmulMetadata &registered,
    const std::vector<void *> &addresses) {
    MatmulStorageOrderProof proof{};
    const Fault20MatmulMetadata &m = registered.tensors;
    const Fault20TensorMetadata *formal[] = {&m.self, &m.mat2, &m.out};
    proof.param_order_exact = registered.tensor_param_count == 3;
    proof.all_present = m.all_present != 0;
    proof.no_empty_storage = m.any_empty_storage == 0;
    proof.nonempty_slots_are_nonnull = true;
    proof.no_shared_storage = m.any_shared_storage == 0;
    for (size_t i = 0; i < 3; ++i) {
        proof.param_ordinals[i] = find_unique_tensor_ordinal(
            registered.tensor_params, registered.tensor_param_count,
            formal[i]->tensor_ptr);
        if (registered.tensor_param_count != 3 ||
            registered.tensor_params[i] != formal[i]->tensor_ptr) {
            proof.param_order_exact = false;
        }
        if (formal[i]->present == 0) {
            continue;
        }
        const size_t slot = proof.compact_storage_count++;
        proof.slots[i] = static_cast<int32_t>(slot);
        if (slot >= addresses.size() || addresses[slot] == nullptr ||
            formal[i]->storage_ptr == 0 || formal[i]->storage_bytes == 0) {
            proof.nonempty_slots_are_nonnull = false;
        }
    }
    proof.address_count_exact = proof.compact_storage_count == addresses.size();
    proof.self_storage_unique = matmul_tensor_storage_is_unique(m, m.self);
    proof.mat2_storage_unique = matmul_tensor_storage_is_unique(m, m.mat2);
    proof.out_storage_unique = matmul_tensor_storage_is_unique(m, m.out);
    const bool common = proof.param_order_exact && proof.address_count_exact &&
                        proof.all_present && proof.no_empty_storage &&
                        proof.nonempty_slots_are_nonnull &&
                        proof.no_shared_storage;
    proof.self_slot_proven = common && proof.param_ordinals[0] == 0 &&
                             proof.slots[0] == 0 &&
                             proof.self_storage_unique;
    proof.mat2_slot_proven = common && proof.param_ordinals[1] == 1 &&
                             proof.slots[1] == 1 &&
                             proof.mat2_storage_unique;
    proof.output_slot_proven = common && proof.param_ordinals[2] == 2 &&
                               proof.slots[2] == 2 &&
                               proof.out_storage_unique;
    return proof;
}

bool is_target_float_matmul_slot_proof(
    const RegisteredMatmulMetadata &registered) {
    const auto &m = registered.tensors;
    return registered.api_kind == static_cast<uint32_t>(MatmulApiKind::kFloat) &&
           registered.device == 0 && m.self.dtype == ACL_BF16 &&
           m.mat2.dtype == ACL_BF16 && m.out.dtype == ACL_BF16 &&
           m.self.view_rank == 2 && m.mat2.view_rank == 2 &&
           m.out.view_rank == 2 && m.self.view_shape[0] == 1 &&
           m.self.view_shape[1] == 12800 &&
           m.mat2.view_shape[0] == 12800 &&
           m.mat2.view_shape[1] == 5120 && m.out.view_shape[0] == 1 &&
           m.out.view_shape[1] == 5120;
}

void log_float_matmul_run_slot_proof(
    op::internal::OpExecCache *cache,
    const RegisteredMatmulMetadata &registered, const std::string &api_info,
    const std::vector<void *> &addresses, uint64_t run_index) {
    const auto &m = registered.tensors;
    const MatmulStorageOrderProof proof =
        prove_matmul_storage_order(registered, addresses);
    const int32_t direct_self_match = find_unique_address_slot(addresses, m.self);
    const int32_t direct_mat2_match = find_unique_address_slot(addresses, m.mat2);
    const int32_t direct_out_match = find_unique_address_slot(addresses, m.out);
    const void *self_runtime = proof.slots[0] >= 0 &&
                                       static_cast<size_t>(proof.slots[0]) < addresses.size()
                                   ? addresses[static_cast<size_t>(proof.slots[0])]
                                   : nullptr;
    const void *mat2_runtime = proof.slots[1] >= 0 &&
                                       static_cast<size_t>(proof.slots[1]) < addresses.size()
                                   ? addresses[static_cast<size_t>(proof.slots[1])]
                                   : nullptr;
    const void *out_runtime = proof.slots[2] >= 0 &&
                                      static_cast<size_t>(proof.slots[2]) < addresses.size()
                                  ? addresses[static_cast<size_t>(proof.slots[2])]
                                  : nullptr;
    char line[8192];
    const int written = std::snprintf(
        line, sizeof(line),
        "{\"event\":\"float_matmul_run_slot_proof\",\"pid\":%ld,"
        "\"tid\":%ld,\"ts_ns\":%" PRIu64 ",\"device\":%d,"
        "\"cache\":\"%p\",\"run_index\":%" PRIu64
        ",\"api_info\":\"%s\",\"address_count\":%zu,"
        "\"compact_storage_count\":%u,"
        "\"formal_ordinals\":{\"self\":%d,\"mat2\":%d,\"out\":%d},"
        "\"storage_order_slots\":{\"self\":%d,\"mat2\":%d,\"out\":%d},"
        "\"runtime_addresses\":{\"self\":\"%p\",\"mat2\":\"%p\","
        "\"out\":\"%p\"},"
        "\"proof\":{\"param_order_exact\":%s,\"address_count_exact\":%s,"
        "\"all_present\":%s,\"no_empty_storage\":%s,"
        "\"nonempty_slots_are_nonnull\":%s,\"no_shared_storage\":%s,"
        "\"self_storage_unique\":%s,\"mat2_storage_unique\":%s,"
        "\"out_storage_unique\":%s,\"self_slot_proven\":%s,"
        "\"mat2_slot_proven\":%s,\"output_slot_proven\":%s},"
        "\"supplemental_gws_address_matches\":{\"self\":%d,\"mat2\":%d,"
        "\"out\":%d},"
        "\"self\":{\"dtype\":%d,\"format\":%d,\"shape\":[%" PRId64
        ",%" PRId64 "],\"storage_shape\":[%" PRId64 ",%" PRId64
        "],\"strides\":[%" PRId64 ",%" PRId64 "],\"view_offset\":%" PRId64
        ",\"storage_bytes\":%" PRIu64 "},"
        "\"mat2\":{\"dtype\":%d,\"format\":%d,\"shape\":[%" PRId64
        ",%" PRId64 "],\"storage_shape\":[%" PRId64 ",%" PRId64
        "],\"strides\":[%" PRId64 ",%" PRId64 "],\"view_offset\":%" PRId64
        ",\"storage_bytes\":%" PRIu64 "},"
        "\"out\":{\"dtype\":%d,\"format\":%d,\"shape\":[%" PRId64
        ",%" PRId64 "],\"storage_shape\":[%" PRId64 ",%" PRId64
        "],\"strides\":[%" PRId64 ",%" PRId64 "],\"view_offset\":%" PRId64
        ",\"storage_bytes\":%" PRIu64 "}}",
        static_cast<long>(getpid()), static_cast<long>(syscall(SYS_gettid)),
        monotonic_ns(), registered.device, cache, run_index, api_info.c_str(),
        addresses.size(), proof.compact_storage_count, proof.param_ordinals[0],
        proof.param_ordinals[1], proof.param_ordinals[2], proof.slots[0],
        proof.slots[1], proof.slots[2], self_runtime, mat2_runtime, out_runtime,
        proof.param_order_exact ? "true" : "false",
        proof.address_count_exact ? "true" : "false",
        proof.all_present ? "true" : "false",
        proof.no_empty_storage ? "true" : "false",
        proof.nonempty_slots_are_nonnull ? "true" : "false",
        proof.no_shared_storage ? "true" : "false",
        proof.self_storage_unique ? "true" : "false",
        proof.mat2_storage_unique ? "true" : "false",
        proof.out_storage_unique ? "true" : "false",
        proof.self_slot_proven ? "true" : "false",
        proof.mat2_slot_proven ? "true" : "false",
        proof.output_slot_proven ? "true" : "false", direct_self_match,
        direct_mat2_match, direct_out_match, m.self.dtype, m.self.format,
        m.self.view_shape[0], m.self.view_shape[1], m.self.storage_shape[0],
        m.self.storage_shape[1], m.self.strides[0], m.self.strides[1],
        m.self.view_offset, m.self.storage_bytes, m.mat2.dtype, m.mat2.format,
        m.mat2.view_shape[0], m.mat2.view_shape[1], m.mat2.storage_shape[0],
        m.mat2.storage_shape[1], m.mat2.strides[0], m.mat2.strides[1],
        m.mat2.view_offset, m.mat2.storage_bytes, m.out.dtype, m.out.format,
        m.out.view_shape[0], m.out.view_shape[1], m.out.storage_shape[0],
        m.out.storage_shape[1], m.out.strides[0], m.out.strides[1],
        m.out.view_offset, m.out.storage_bytes);
    if (written > 0 && static_cast<size_t>(written) < sizeof(line)) {
        write_line(line, static_cast<size_t>(written));
    }
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

struct FloatMatmulLosslessAttempt {
    uint32_t mode{0};
    uint32_t hit_index{0};
    uint32_t candidate_index{0};
    int32_t result_status{0};
    int32_t pre_sync_status{kNotCalled};
    int32_t self_malloc_status{kNotCalled};
    int32_t mat2_malloc_status{kNotCalled};
    int32_t self_copy_status{kNotCalled};
    int32_t mat2_copy_status{kNotCalled};
    int32_t run_status{kNotCalled};
    int32_t post_sync_status{kNotCalled};
    int32_t self_free_status{kNotCalled};
    int32_t mat2_free_status{kNotCalled};
    void *original_self{nullptr};
    void *original_mat2{nullptr};
    void *temporary_self{nullptr};
    void *temporary_mat2{nullptr};
    bool self_used{false};
    bool mat2_used{false};
    bool self_freed{false};
    bool mat2_freed{false};
    uint64_t begin_ns{0};
    uint64_t pre_sync_ns{0};
    uint64_t copy_ns{0};
    uint64_t run_ns{0};
    uint64_t post_sync_ns{0};
};

struct FloatMatmulFiniteAttempt {
    uint32_t mode{0};
    uint32_t hit_index{0};
    uint32_t candidate_index{0};
    uint32_t cancellation_pattern{0};
    uint32_t cancellation_block_size{0};
    int32_t exponent{0};
    uint16_t bf16_bits{0};
    int32_t result_status{0};
    int32_t pre_sync_status{kNotCalled};
    int32_t self_host_malloc_status{kNotCalled};
    int32_t mat2_host_malloc_status{kNotCalled};
    int32_t self_d2h_status{kNotCalled};
    int32_t mat2_d2h_status{kNotCalled};
    int32_t self_device_malloc_status{kNotCalled};
    int32_t mat2_device_malloc_status{kNotCalled};
    int32_t self_h2d_status{kNotCalled};
    int32_t mat2_h2d_status{kNotCalled};
    int32_t run_status{kNotCalled};
    int32_t post_sync_status{kNotCalled};
    int32_t self_device_free_status{kNotCalled};
    int32_t mat2_device_free_status{kNotCalled};
    int32_t self_host_free_status{kNotCalled};
    int32_t mat2_host_free_status{kNotCalled};
    void *self_original{nullptr};
    void *mat2_original{nullptr};
    void *self_temporary{nullptr};
    void *mat2_temporary{nullptr};
    void *self_host{nullptr};
    void *mat2_host{nullptr};
    bool self_temporary_used{false};
    bool mat2_temporary_used{false};
    bool self_temporary_freed{false};
    bool mat2_temporary_freed{false};
    bool self_host_freed{false};
    bool mat2_host_freed{false};
    uint64_t self_changed_elements{0};
    uint64_t mat2_changed_elements{0};
    uint64_t self_unchanged_elements{0};
    uint64_t mat2_unchanged_elements{0};
    uint64_t mat2_positive_elements{0};
    uint64_t mat2_negative_elements{0};
    bool theoretical_output_exact_zero{false};
    uint64_t begin_ns{0};
    uint64_t pre_sync_ns{0};
    uint64_t host_alloc_ns{0};
    uint64_t d2h_ns{0};
    uint64_t transform_ns{0};
    uint64_t device_alloc_ns{0};
    uint64_t h2d_ns{0};
    uint64_t real_run_ns{0};
    uint64_t post_sync_ns{0};
};

struct FloatMatmulOutputProxyAttempt {
    uint32_t hit_index{0};
    uint32_t candidate_index{0};
    uint32_t proxy_kind{0};
    uint32_t requested_elements{0};
    uint16_t proxy_bits{0};
    int32_t result_status{0};
    int32_t run_status{kNotCalled};
    int32_t pre_sync_status{kNotCalled};
    int32_t host_malloc_status{kNotCalled};
    int32_t pre_d2h_status{kNotCalled};
    int32_t h2d_status{kNotCalled};
    int32_t post_sync_status{kNotCalled};
    int32_t readback_status{kNotCalled};
    int32_t host_free_status{kNotCalled};
    void *out_address{nullptr};
    void *host_buffer{nullptr};
    bool host_freed{false};
    bool readback_exact{false};
    uint64_t changed_elements{0};
    QmmOutputObservation before{};
    QmmOutputObservation after{};
    uint64_t begin_ns{0};
    uint64_t real_run_ns{0};
    uint64_t pre_sync_ns{0};
    uint64_t host_alloc_ns{0};
    uint64_t pre_d2h_ns{0};
    uint64_t mutate_ns{0};
    uint64_t h2d_ns{0};
    uint64_t post_sync_ns{0};
    uint64_t readback_ns{0};
    uint64_t scan_ns{0};
};

const char *float_matmul_lossless_role(uint32_t mode) {
    if (mode == FAULT20_MODE_FLOAT_MATMUL_LOSSLESS_SELF) {
        return "self";
    }
    if (mode == FAULT20_MODE_FLOAT_MATMUL_LOSSLESS_MAT2) {
        return "mat2";
    }
    if (mode == FAULT20_MODE_FLOAT_MATMUL_LOSSLESS_DUAL) {
        return "dual";
    }
    return "unknown";
}

void log_float_matmul_lossless(
    const char *outcome, op::internal::OpExecCache *cache,
    const RegisteredMatmulMetadata &metadata, const std::string &api_info,
    const FloatMatmulLosslessAttempt &attempt) {
    char line[6144];
    const int written = std::snprintf(
        line, sizeof(line),
        "{\"event\":\"float_matmul_lossless_substitution\","
        "\"outcome\":\"%s\",\"role\":\"%s\",\"pid\":%ld,"
        "\"tid\":%ld,\"ts_ns\":%" PRIu64 ",\"device\":%d,"
        "\"cache\":\"%p\",\"api_info\":\"%s\",\"hit\":%u,"
        "\"candidate\":%u,\"shape\":{\"self\":[1,12800],"
        "\"mat2\":[12800,5120],\"out\":[1,5120]},"
        "\"bytes\":{\"self\":%" PRIu64 ",\"mat2\":%" PRIu64
        ",\"out\":%" PRIu64 "},"
        "\"addresses\":{\"original_self\":\"%p\","
        "\"temporary_self\":\"%p\",\"original_mat2\":\"%p\","
        "\"temporary_mat2\":\"%p\"},"
        "\"address_distinct\":{\"self\":%s,\"mat2\":%s},"
        "\"temporary_used\":{\"self\":%s,\"mat2\":%s},"
        "\"changed_elements\":{\"self\":0,\"mat2\":0,\"total\":0},"
        "\"copy_rule\":\"D2D byte-for-byte, no numeric transform\","
        "\"status\":{\"result\":%d,\"pre_sync\":%d,"
        "\"self_malloc\":%d,\"mat2_malloc\":%d,"
        "\"self_d2d\":%d,\"mat2_d2d\":%d,\"run\":%d,"
        "\"post_sync\":%d,\"self_free\":%d,\"mat2_free\":%d},"
        "\"freed\":{\"self\":%s,\"mat2\":%s},"
        "\"timing_ns\":{\"pre_sync\":%" PRIu64
        ",\"d2d_enqueue\":%" PRIu64 ",\"real_run\":%" PRIu64
        ",\"post_sync\":%" PRIu64 ",\"total\":%" PRIu64 "}}",
        outcome, float_matmul_lossless_role(attempt.mode),
        static_cast<long>(getpid()), static_cast<long>(syscall(SYS_gettid)),
        monotonic_ns(), metadata.device, cache, api_info.c_str(),
        attempt.hit_index, attempt.candidate_index,
        metadata.tensors.self.storage_bytes, metadata.tensors.mat2.storage_bytes,
        metadata.tensors.out.storage_bytes, attempt.original_self,
        attempt.temporary_self, attempt.original_mat2, attempt.temporary_mat2,
        !attempt.self_used || attempt.original_self != attempt.temporary_self
            ? "true"
            : "false",
        !attempt.mat2_used || attempt.original_mat2 != attempt.temporary_mat2
            ? "true"
            : "false",
        attempt.self_used ? "true" : "false",
        attempt.mat2_used ? "true" : "false", attempt.result_status,
        attempt.pre_sync_status, attempt.self_malloc_status,
        attempt.mat2_malloc_status, attempt.self_copy_status,
        attempt.mat2_copy_status, attempt.run_status, attempt.post_sync_status,
        attempt.self_free_status, attempt.mat2_free_status,
        attempt.self_freed ? "true" : "false",
        attempt.mat2_freed ? "true" : "false", attempt.pre_sync_ns,
        attempt.copy_ns, attempt.run_ns, attempt.post_sync_ns,
        monotonic_ns() - attempt.begin_ns);
    if (written > 0 && static_cast<size_t>(written) < sizeof(line)) {
        write_line(line, static_cast<size_t>(written));
    }
}

aclnnStatus run_float_matmul_lossless(
    op::internal::OpExecCache *cache, void *workspace, aclrtStream stream,
    const std::vector<void *> &addresses, Fault20Control *control,
    const RegisteredMatmulMetadata &metadata, const std::string &api_info,
    uint32_t mode, uint32_t hit_index, uint32_t candidate_index) {
    FloatMatmulLosslessAttempt attempt{};
    attempt.mode = mode;
    attempt.hit_index = hit_index;
    attempt.candidate_index = candidate_index;
    attempt.begin_ns = monotonic_ns();
    attempt.original_self = addresses[0];
    attempt.original_mat2 = addresses[1];
    const bool copy_self =
        mode == FAULT20_MODE_FLOAT_MATMUL_LOSSLESS_SELF ||
        mode == FAULT20_MODE_FLOAT_MATMUL_LOSSLESS_DUAL;
    const bool copy_mat2 =
        mode == FAULT20_MODE_FLOAT_MATMUL_LOSSLESS_MAT2 ||
        mode == FAULT20_MODE_FLOAT_MATMUL_LOSSLESS_DUAL;
    std::vector<void *> addresses_copy;
    try {
        addresses_copy = addresses;
    } catch (...) {
        attempt.result_status = kStatusVectorCopyFailed;
        log_float_matmul_lossless("address_vector_copy_failed", cache, metadata,
                                  api_info, attempt);
        set_device_last_status(control, metadata.device, attempt.result_status);
        return run_original(cache, workspace, stream, addresses);
    }

    uint64_t operation_begin_ns = monotonic_ns();
    attempt.pre_sync_status =
        static_cast<int32_t>(g_synchronize_stream(stream));
    attempt.pre_sync_ns = monotonic_ns() - operation_begin_ns;
    if (attempt.pre_sync_status != ACL_SUCCESS) {
        attempt.result_status = kStatusPreSyncFailed;
        log_float_matmul_lossless("pre_sync_failed", cache, metadata, api_info,
                                  attempt);
        set_device_last_status(control, metadata.device, attempt.result_status);
        return run_original(cache, workspace, stream, addresses);
    }

    if (copy_self) {
        attempt.self_malloc_status = static_cast<int32_t>(g_malloc(
            &attempt.temporary_self,
            static_cast<size_t>(metadata.tensors.self.storage_bytes),
            ACL_MEM_MALLOC_HUGE_FIRST));
    }
    if (copy_mat2) {
        attempt.mat2_malloc_status = static_cast<int32_t>(g_malloc(
            &attempt.temporary_mat2,
            static_cast<size_t>(metadata.tensors.mat2.storage_bytes),
            ACL_MEM_MALLOC_HUGE_FIRST));
    }
    const bool allocation_ok =
        (!copy_self || (attempt.self_malloc_status == ACL_SUCCESS &&
                       attempt.temporary_self != nullptr &&
                       attempt.temporary_self != attempt.original_self)) &&
        (!copy_mat2 || (attempt.mat2_malloc_status == ACL_SUCCESS &&
                       attempt.temporary_mat2 != nullptr &&
                       attempt.temporary_mat2 != attempt.original_mat2));
    if (!allocation_ok) {
        if (attempt.temporary_self != nullptr) {
            attempt.self_free_status =
                static_cast<int32_t>(g_free(attempt.temporary_self));
            attempt.self_freed = attempt.self_free_status == ACL_SUCCESS;
        }
        if (attempt.temporary_mat2 != nullptr) {
            attempt.mat2_free_status =
                static_cast<int32_t>(g_free(attempt.temporary_mat2));
            attempt.mat2_freed = attempt.mat2_free_status == ACL_SUCCESS;
        }
        attempt.result_status = kStatusDeviceAllocFailed;
        log_float_matmul_lossless("allocation_or_address_distinct_failed", cache,
                                  metadata, api_info, attempt);
        set_device_last_status(control, metadata.device, attempt.result_status);
        return run_original(cache, workspace, stream, addresses);
    }

    operation_begin_ns = monotonic_ns();
    if (copy_self) {
        attempt.self_copy_status = static_cast<int32_t>(g_memcpy_async(
            attempt.temporary_self,
            static_cast<size_t>(metadata.tensors.self.storage_bytes),
            attempt.original_self,
            static_cast<size_t>(metadata.tensors.self.storage_bytes),
            ACL_MEMCPY_DEVICE_TO_DEVICE, stream));
    }
    if (copy_mat2) {
        attempt.mat2_copy_status = static_cast<int32_t>(g_memcpy_async(
            attempt.temporary_mat2,
            static_cast<size_t>(metadata.tensors.mat2.storage_bytes),
            attempt.original_mat2,
            static_cast<size_t>(metadata.tensors.mat2.storage_bytes),
            ACL_MEMCPY_DEVICE_TO_DEVICE, stream));
    }
    attempt.copy_ns = monotonic_ns() - operation_begin_ns;
    const bool copy_ok =
        (!copy_self || attempt.self_copy_status == ACL_SUCCESS) &&
        (!copy_mat2 || attempt.mat2_copy_status == ACL_SUCCESS);
    if (!copy_ok) {
        attempt.post_sync_status =
            static_cast<int32_t>(g_synchronize_stream(stream));
        if (attempt.post_sync_status == ACL_SUCCESS) {
            if (attempt.temporary_self != nullptr) {
                attempt.self_free_status =
                    static_cast<int32_t>(g_free(attempt.temporary_self));
                attempt.self_freed = attempt.self_free_status == ACL_SUCCESS;
            }
            if (attempt.temporary_mat2 != nullptr) {
                attempt.mat2_free_status =
                    static_cast<int32_t>(g_free(attempt.temporary_mat2));
                attempt.mat2_freed = attempt.mat2_free_status == ACL_SUCCESS;
            }
        }
        attempt.result_status = kStatusH2DFailed;
        log_float_matmul_lossless("d2d_copy_failed", cache, metadata, api_info,
                                  attempt);
        set_device_last_status(control, metadata.device, attempt.result_status);
        return run_original(cache, workspace, stream, addresses);
    }

    if (copy_self) {
        addresses_copy[0] = attempt.temporary_self;
        attempt.self_used = true;
    }
    if (copy_mat2) {
        addresses_copy[1] = attempt.temporary_mat2;
        attempt.mat2_used = true;
    }
    operation_begin_ns = monotonic_ns();
    const aclnnStatus run_status =
        run_original(cache, workspace, stream, addresses_copy);
    attempt.run_ns = monotonic_ns() - operation_begin_ns;
    attempt.run_status = static_cast<int32_t>(run_status);

    operation_begin_ns = monotonic_ns();
    attempt.post_sync_status =
        static_cast<int32_t>(g_synchronize_stream(stream));
    attempt.post_sync_ns = monotonic_ns() - operation_begin_ns;
    if (attempt.post_sync_status == ACL_SUCCESS) {
        if (attempt.temporary_self != nullptr) {
            attempt.self_free_status =
                static_cast<int32_t>(g_free(attempt.temporary_self));
            attempt.self_freed = attempt.self_free_status == ACL_SUCCESS;
        }
        if (attempt.temporary_mat2 != nullptr) {
            attempt.mat2_free_status =
                static_cast<int32_t>(g_free(attempt.temporary_mat2));
            attempt.mat2_freed = attempt.mat2_free_status == ACL_SUCCESS;
        }
    }

    const bool free_ok =
        (!copy_self || attempt.self_freed) &&
        (!copy_mat2 || attempt.mat2_freed);
    const char *outcome = "success";
    if (attempt.run_status != ACL_SUCCESS) {
        attempt.result_status = kStatusRunFailed;
        outcome = "real_run_failed";
    } else if (attempt.post_sync_status != ACL_SUCCESS) {
        attempt.result_status = kStatusPostSyncFailed;
        outcome = "post_sync_failed_buffers_retained";
    } else if (!free_ok) {
        attempt.result_status = kStatusDeviceFreeFailed;
        outcome = "device_free_failed";
    } else {
        attempt.result_status = kStatusSuccess;
    }
    set_device_last_status(control, metadata.device, attempt.result_status);
    log_float_matmul_lossless(outcome, cache, metadata, api_info, attempt);
    return run_status;
}

void log_numeric_attempt(uint32_t mode, const char *outcome, const std::string &api_info,
                         op::internal::OpExecCache *cache,
                         const RegisteredMetadata &metadata, const Int4Attempt &attempt) {
    const uint64_t total_ns = monotonic_ns() - attempt.begin_ns;
    const bool extreme = mode == FAULT20_MODE_QMM_EXTREME_X1;
    const char *event_name = extreme
                                 ? "qmm_extreme_x1_substitution"
                                 : (mode == FAULT20_MODE_INT2_EQUIVALENT
                                        ? "int2_equivalent_substitution"
                                        : "int4_equivalent_substitution");
    const char *rule = extreme
                           ? "deterministic_uniform_selection"
                           : (mode == FAULT20_MODE_INT2_EQUIVALENT
                                  ? "round_away_from_zero(x/64),clamp[-2,1],x64"
                                  : "round_away_from_zero(x/16),clamp[-8,7],x16");
    const char *pattern = attempt.extreme_pattern == FAULT20_EXTREME_POSITIVE_MAX
                              ? "positive_max"
                              : (attempt.extreme_pattern == FAULT20_EXTREME_NEGATIVE_MAX
                                     ? "negative_max"
                                     : (attempt.extreme_pattern == FAULT20_EXTREME_ALTERNATING
                                            ? "alternating"
                                            : "none"));
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
        "\"rule\":\"%s\",\"extreme_value\":%d,"
        "\"extreme_ratio_bps\":%u,\"extreme_pattern\":\"%s\","
        "\"selected_elements\":%" PRIu64 ","
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
        attempt.extreme_value, attempt.extreme_ratio_bps, pattern,
        attempt.selected_elements, attempt.stats_valid ? "true" : "false",
        attempt.changed_elements,
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
    const bool extreme = mode == FAULT20_MODE_QMM_EXTREME_X1;
    attempt->min_before = 127;
    attempt->max_before = -128;
    attempt->min_after = 127;
    attempt->max_after = -128;
    if (extreme) {
        for (uint64_t index = 0; index < bytes; ++index) {
            const int before = static_cast<int>(values[index]);
            attempt->min_before =
                before < attempt->min_before ? before : attempt->min_before;
            attempt->max_before =
                before > attempt->max_before ? before : attempt->max_before;
            attempt->zero_before += before == 0 ? 1U : 0U;
        }
        const Fault20ExtremeTransformStats stats = fault20_apply_extreme_transform(
            values, bytes, attempt->extreme_value, attempt->extreme_ratio_bps,
            attempt->extreme_pattern);
        attempt->selected_elements = stats.selected_elements;
        attempt->changed_elements = stats.changed_elements;
        for (uint64_t index = 0; index < bytes; ++index) {
            const int after = static_cast<int>(values[index]);
            attempt->min_after =
                after < attempt->min_after ? after : attempt->min_after;
            attempt->max_after =
                after > attempt->max_after ? after : attempt->max_after;
            attempt->zero_after += after == 0 ? 1U : 0U;
        }
        attempt->unchanged_elements = bytes - attempt->changed_elements;
        attempt->stats_valid = true;
        return;
    }
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

bool qmm_output_metadata_valid(const Fault20TensorMetadata &out,
                               uint64_t max_output_bytes) {
    const bool dtype_ok = out.dtype == ACL_BF16 || out.dtype == ACL_FLOAT16 ||
                          out.dtype == ACL_FLOAT;
    const uint64_t expected_element_size = out.dtype == ACL_FLOAT ? 4U : 2U;
    if (!dtype_ok || out.present == 0 || out.format != ACL_FORMAT_ND ||
        out.view_rank != 2 || out.storage_rank != 1 || out.stride_rank != 2 ||
        out.contiguous == 0 || out.view_offset != 0 || out.view_shape[0] <= 0 ||
        out.view_shape[1] <= 0 || out.storage_shape[0] != out.numel ||
        out.strides[0] != out.view_shape[1] || out.strides[1] != 1 ||
        out.numel <= 0 || out.storage_elements != out.numel ||
        out.element_size != expected_element_size ||
        out.view_bytes != out.storage_bytes || out.storage_bytes == 0 ||
        out.storage_bytes > max_output_bytes || out.storage_bytes > 1048576U) {
        return false;
    }
    const uint64_t elements = static_cast<uint64_t>(out.numel);
    return elements <= std::numeric_limits<uint64_t>::max() / out.element_size &&
           elements * out.element_size == out.storage_bytes;
}

bool qmm_weight_metadata_valid(const Fault20TensorMetadata &x2, int32_t target_k,
                               uint64_t max_weight_bytes) {
    // CANN 8.5.1 reports QuantMatmulWeightNz INT8 weight storage as format 29
    // (FRACTAL_NZ), with a logical rank-2 view and physical rank-4 storage.
    constexpr int32_t kC859FractalNzFormat = 29;
    if (x2.present == 0 || x2.dtype != ACL_INT8 ||
        x2.format != kC859FractalNzFormat || x2.view_rank != 2 ||
        x2.storage_rank != 4 || x2.stride_rank != 2 || x2.contiguous == 0 ||
        x2.view_offset != 0 || target_k <= 0 || x2.view_shape[0] != target_k ||
        x2.view_shape[1] <= 0 || x2.strides[0] != x2.view_shape[1] ||
        x2.strides[1] != 1 || x2.element_size != 1 || x2.numel <= 0 ||
        x2.storage_elements != x2.numel || x2.view_bytes != x2.storage_bytes ||
        x2.storage_bytes == 0 || x2.storage_bytes > max_weight_bytes ||
        x2.storage_bytes > 1073741824ULL) {
        return false;
    }
    uint64_t storage_product = 1;
    for (size_t dim = 0; dim < 4; ++dim) {
        if (x2.storage_shape[dim] <= 0 ||
            storage_product > std::numeric_limits<uint64_t>::max() /
                                  static_cast<uint64_t>(x2.storage_shape[dim])) {
            return false;
        }
        storage_product *= static_cast<uint64_t>(x2.storage_shape[dim]);
    }
    return static_cast<uint64_t>(x2.numel) == x2.storage_bytes &&
           storage_product == static_cast<uint64_t>(x2.storage_elements);
}

bool qmm_x2_scale_metadata_valid(const Fault20TensorMetadata &scale,
                                 int32_t target_scale_elements,
                                 uint64_t max_scale_bytes) {
    if (target_scale_elements <= 0 || scale.present == 0 ||
        scale.dtype != ACL_FLOAT || scale.format != ACL_FORMAT_ND ||
        scale.view_rank != 1 || scale.storage_rank != 1 ||
        scale.stride_rank != 1 || scale.contiguous == 0 ||
        scale.view_offset != 0 ||
        scale.view_shape[0] != target_scale_elements ||
        scale.storage_shape[0] != target_scale_elements ||
        scale.strides[0] != 1 || scale.numel != target_scale_elements ||
        scale.storage_elements != target_scale_elements ||
        scale.element_size != sizeof(float) ||
        scale.view_bytes != scale.storage_bytes || scale.storage_bytes == 0 ||
        scale.storage_bytes > max_scale_bytes ||
        scale.storage_bytes > 4194304ULL) {
        return false;
    }
    const uint64_t elements = static_cast<uint64_t>(target_scale_elements);
    return elements <= std::numeric_limits<uint64_t>::max() / sizeof(float) &&
           elements * sizeof(float) == scale.storage_bytes;
}

void scan_qmm_output(const void *host_buffer, const Fault20TensorMetadata &out,
                     QmmOutputObservation *observation) {
    double min_value = std::numeric_limits<double>::infinity();
    double max_value = -std::numeric_limits<double>::infinity();
    long double sum_abs = 0.0L;
    const uint64_t elements = static_cast<uint64_t>(out.numel);
    for (uint64_t index = 0; index < elements; ++index) {
        double value = 0.0;
        if (out.dtype == ACL_BF16 || out.dtype == ACL_FLOAT16) {
            const auto *bits = static_cast<const uint16_t *>(host_buffer);
            value = out.dtype == ACL_BF16
                        ? static_cast<double>(bf16_bits_to_float(bits[index]))
                        : static_cast<double>(fp16_bits_to_float(bits[index]));
        } else {
            const auto *values = static_cast<const float *>(host_buffer);
            value = static_cast<double>(values[index]);
        }
        if (std::isnan(value)) {
            ++observation->nan_count;
            continue;
        }
        if (std::isinf(value)) {
            if (std::signbit(value)) {
                ++observation->negative_inf_count;
            } else {
                ++observation->positive_inf_count;
            }
            continue;
        }
        ++observation->finite_count;
        observation->zero_count += value == 0.0 ? 1U : 0U;
        min_value = value < min_value ? value : min_value;
        max_value = value > max_value ? value : max_value;
        const double absolute = std::fabs(value);
        observation->max_abs = absolute > observation->max_abs
                                   ? absolute
                                   : observation->max_abs;
        sum_abs += static_cast<long double>(absolute);
    }
    if (observation->finite_count != 0) {
        observation->min_value = min_value;
        observation->max_value = max_value;
        observation->mean_abs =
            static_cast<double>(sum_abs / observation->finite_count);
    }
    observation->stats_valid =
        observation->finite_count + observation->nan_count +
            observation->positive_inf_count + observation->negative_inf_count ==
        elements;
}

void log_float_matmul_output_observation(
    op::internal::OpExecCache *cache,
    const RegisteredMatmulMetadata &metadata, const std::string &api_info,
    const MatmulStorageOrderProof &proof,
    const QmmOutputObservation &observation) {
    const auto &out = metadata.tensors.out;
    const char *outcome = observation.result_status == kStatusSuccess
                              ? "success"
                              : "failed_closed";
    char line[6144];
    const int written = std::snprintf(
        line, sizeof(line),
        "{\"event\":\"float_matmul_output_observation\","
        "\"outcome\":\"%s\",\"pid\":%ld,\"tid\":%ld,"
        "\"ts_ns\":%" PRIu64 ",\"device\":%d,\"hit\":%u,"
        "\"candidate\":%u,\"cache\":\"%p\",\"api_info\":\"%s\","
        "\"out_slot\":%d,\"out_address\":\"%p\","
        "\"output_slot_proven\":%s,\"dtype\":%d,\"format\":%d,"
        "\"shape\":[%" PRId64 ",%" PRId64 "],\"strides\":[%" PRId64
        ",%" PRId64 "],\"view_offset\":%" PRId64 ",\"bytes\":%" PRIu64
        ",\"stats_valid\":%s,\"finite_count\":%" PRIu64
        ",\"nan_count\":%" PRIu64 ",\"positive_inf_count\":%" PRIu64
        ",\"negative_inf_count\":%" PRIu64 ",\"zero_count\":%" PRIu64
        ",\"min\":%.17g,\"max\":%.17g,\"max_abs\":%.17g,"
        "\"mean_abs\":%.17g,\"result_status\":%d,\"run_status\":%d,"
        "\"sync_status\":%d,\"host_malloc_status\":%d,"
        "\"d2h_status\":%d,\"host_free_status\":%d,\"host_freed\":%s,"
        "\"auto_disarmed_on_nonfinite\":%s,\"real_run_ns\":%" PRIu64
        ",\"sync_ns\":%" PRIu64 ",\"host_alloc_ns\":%" PRIu64
        ",\"d2h_ns\":%" PRIu64 ",\"scan_ns\":%" PRIu64
        ",\"total_ns\":%" PRIu64 "}",
        outcome, static_cast<long>(getpid()),
        static_cast<long>(syscall(SYS_gettid)), observation.begin_ns,
        metadata.device, observation.hit_index, observation.candidate_index,
        cache, api_info.c_str(), observation.out_slot, observation.out_address,
        proof.output_slot_proven ? "true" : "false", out.dtype, out.format,
        out.view_shape[0], out.view_shape[1], out.strides[0], out.strides[1],
        out.view_offset, out.storage_bytes,
        observation.stats_valid ? "true" : "false", observation.finite_count,
        observation.nan_count, observation.positive_inf_count,
        observation.negative_inf_count, observation.zero_count,
        observation.min_value, observation.max_value, observation.max_abs,
        observation.mean_abs, observation.result_status, observation.run_status,
        observation.sync_status, observation.host_malloc_status,
        observation.d2h_status, observation.host_free_status,
        observation.host_freed ? "true" : "false",
        observation.auto_disarmed_on_nonfinite ? "true" : "false",
        observation.real_run_ns, observation.sync_ns,
        observation.host_alloc_ns, observation.d2h_ns, observation.scan_ns,
        monotonic_ns() - observation.begin_ns);
    if (written > 0 && static_cast<size_t>(written) < sizeof(line)) {
        write_line(line, static_cast<size_t>(written));
    }
}

void observe_float_matmul_output_after_run(
    op::internal::OpExecCache *cache, aclrtStream stream,
    const std::vector<void *> &addresses, Fault20Control *control,
    const RegisteredMatmulMetadata &metadata, const std::string &api_info,
    uint32_t hit_index, uint32_t candidate_index, aclnnStatus run_status,
    uint64_t real_run_ns) {
    QmmOutputObservation observation{};
    observation.begin_ns = monotonic_ns();
    observation.hit_index = hit_index;
    observation.candidate_index = candidate_index;
    observation.run_status = static_cast<int32_t>(run_status);
    observation.real_run_ns = real_run_ns;
    const MatmulStorageOrderProof proof =
        prove_matmul_storage_order(metadata, addresses);
    observation.out_slot = proof.slots[2];
    if (!proof.output_slot_proven || observation.out_slot < 0 ||
        static_cast<size_t>(observation.out_slot) >= addresses.size()) {
        observation.result_status = kStatusOutputSlotUnproven;
        set_device_last_status(control, metadata.device,
                               observation.result_status);
        log_float_matmul_output_observation(cache, metadata, api_info, proof,
                                            observation);
        return;
    }
    observation.out_address =
        addresses[static_cast<size_t>(observation.out_slot)];
    if (run_status != ACL_SUCCESS || observation.out_address == nullptr ||
        !qmm_output_metadata_valid(metadata.tensors.out,
                                   control->max_output_bytes)) {
        observation.result_status = run_status == ACL_SUCCESS
                                        ? kStatusOutputMetadataRejected
                                        : kStatusRunFailed;
        set_device_last_status(control, metadata.device,
                               observation.result_status);
        log_float_matmul_output_observation(cache, metadata, api_info, proof,
                                            observation);
        return;
    }
    uint64_t begin = monotonic_ns();
    observation.sync_status = g_synchronize_stream(stream);
    observation.sync_ns = monotonic_ns() - begin;
    if (observation.sync_status != ACL_SUCCESS) {
        observation.result_status = kStatusPostSyncFailed;
        set_device_last_status(control, metadata.device,
                               observation.result_status);
        log_float_matmul_output_observation(cache, metadata, api_info, proof,
                                            observation);
        return;
    }
    const size_t bytes =
        static_cast<size_t>(metadata.tensors.out.storage_bytes);
    begin = monotonic_ns();
    observation.host_malloc_status =
        g_malloc_host(&observation.host_buffer, bytes);
    observation.host_alloc_ns = monotonic_ns() - begin;
    if (observation.host_malloc_status != ACL_SUCCESS ||
        observation.host_buffer == nullptr) {
        observation.result_status = kStatusHostAllocFailed;
        set_device_last_status(control, metadata.device,
                               observation.result_status);
        log_float_matmul_output_observation(cache, metadata, api_info, proof,
                                            observation);
        return;
    }
    begin = monotonic_ns();
    observation.d2h_status = g_memcpy(
        observation.host_buffer, bytes, observation.out_address, bytes,
        ACL_MEMCPY_DEVICE_TO_HOST);
    observation.d2h_ns = monotonic_ns() - begin;
    if (observation.d2h_status == ACL_SUCCESS) {
        begin = monotonic_ns();
        scan_qmm_output(observation.host_buffer, metadata.tensors.out,
                        &observation);
        observation.scan_ns = monotonic_ns() - begin;
        const uint64_t nonfinite =
            observation.nan_count + observation.positive_inf_count +
            observation.negative_inf_count;
        if (nonfinite != 0) {
            __atomic_store_n(&control->enabled, 0U, __ATOMIC_RELEASE);
            observation.auto_disarmed_on_nonfinite = true;
        }
    }
    observation.host_free_status = g_free_host(observation.host_buffer);
    observation.host_freed = observation.host_free_status == ACL_SUCCESS;
    if (observation.d2h_status != ACL_SUCCESS) {
        observation.result_status = kStatusD2HFailed;
    } else if (!observation.stats_valid) {
        observation.result_status = kStatusOutputMetadataRejected;
    } else if (!observation.host_freed) {
        observation.result_status = kStatusHostFreeFailed;
    } else {
        observation.result_status = kStatusSuccess;
    }
    if (observation.result_status != kStatusSuccess) {
        __atomic_store_n(&control->enabled, 0U, __ATOMIC_RELEASE);
    }
    set_device_last_status(control, metadata.device, observation.result_status);
    log_float_matmul_output_observation(cache, metadata, api_info, proof,
                                        observation);
}

bool valid_float_matmul_exponent(int32_t exponent) {
    return exponent == 48 || exponent == 56 || exponent == 57 ||
           exponent == 58 || exponent == 59;
}

const char *float_matmul_cancellation_pattern_name(uint32_t pattern,
                                                    uint32_t block_size) {
    if (pattern == FAULT20_CANCELLATION_HALF && block_size == 0) {
        return "half";
    }
    if (pattern == FAULT20_CANCELLATION_BLOCK && block_size == 256) {
        return "block256_mirrored";
    }
    if (pattern == FAULT20_CANCELLATION_BLOCK && block_size == 512) {
        return "block512_mirrored";
    }
    if (pattern == FAULT20_CANCELLATION_BLOCK && block_size == 1024) {
        return "block1024_mirrored";
    }
    if (pattern == FAULT20_CANCELLATION_ALTERNATING && block_size == 1) {
        return "alternating";
    }
    return "invalid";
}

bool valid_float_matmul_cancellation_config(uint32_t pattern,
                                            uint32_t block_size) {
    return std::strcmp(
               float_matmul_cancellation_pattern_name(pattern, block_size),
               "invalid") != 0;
}

bool float_matmul_cancellation_metadata_valid(
    const Fault20MatmulMetadata &metadata) {
    const auto &self = metadata.self;
    const auto &mat2 = metadata.mat2;
    return self.dtype == ACL_BF16 && self.view_rank == 2 &&
           self.storage_rank == 1 && self.stride_rank == 2 &&
           self.view_offset == 0 && self.view_shape[0] == 1 &&
           self.view_shape[1] == 12800 && self.strides[0] == 12800 &&
           self.strides[1] == 1 && self.numel == 12800 &&
           self.storage_shape[0] == 12800 && self.storage_elements == 12800 &&
           self.element_size == sizeof(uint16_t) &&
           self.storage_bytes == 25600 && mat2.dtype == ACL_BF16 &&
           mat2.view_rank == 2 && mat2.storage_rank == 1 &&
           mat2.stride_rank == 2 && mat2.view_offset == 0 &&
           mat2.view_shape[0] == 12800 && mat2.view_shape[1] == 5120 &&
           mat2.strides[0] == 1 && mat2.strides[1] == 12800 &&
           mat2.numel == 65536000 &&
           mat2.storage_shape[0] == 65536000 &&
           mat2.storage_elements == 65536000 &&
           mat2.element_size == sizeof(uint16_t) &&
           mat2.storage_bytes == 131072000;
}

uint16_t bf16_power_of_two_bits(int32_t exponent) {
    return static_cast<uint16_t>(
        static_cast<uint16_t>(exponent + 127) << 7U);
}

void fill_bf16_power_of_two(void *host_buffer, uint64_t elements,
                            uint16_t target_bits, uint64_t *changed,
                            uint64_t *unchanged) {
    auto *values = static_cast<uint16_t *>(host_buffer);
    uint64_t changed_count = 0;
    for (uint64_t index = 0; index < elements; ++index) {
        if (values[index] != target_bits) {
            ++changed_count;
        }
        values[index] = target_bits;
    }
    *changed = changed_count;
    *unchanged = elements - changed_count;
}

bool fill_bf16_cancellation_mat2(
    void *host_buffer, const Fault20TensorMetadata &mat2,
    uint16_t positive_bits, uint32_t pattern, uint32_t block_size,
    uint64_t *changed, uint64_t *unchanged, uint64_t *positive_elements,
    uint64_t *negative_elements) {
    if (host_buffer == nullptr || changed == nullptr || unchanged == nullptr ||
        positive_elements == nullptr || negative_elements == nullptr ||
        !valid_float_matmul_cancellation_config(pattern, block_size)) {
        return false;
    }
    auto *values = static_cast<uint16_t *>(host_buffer);
    const uint16_t negative_bits =
        static_cast<uint16_t>(positive_bits | static_cast<uint16_t>(0x8000U));
    uint64_t changed_count = 0;
    uint64_t positive_count = 0;
    uint64_t negative_count = 0;
    constexpr int64_t kExtent = 12800;
    constexpr int64_t nExtent = 5120;
    constexpr int64_t kHalf = kExtent / 2;
    for (int64_t n = 0; n < nExtent; ++n) {
        for (int64_t k = 0; k < kExtent; ++k) {
            bool positive = false;
            if (pattern == FAULT20_CANCELLATION_HALF) {
                positive = k < kHalf;
            } else if (pattern == FAULT20_CANCELLATION_ALTERNATING) {
                positive = (k & 1LL) == 0;
            } else {
                const bool first_half = k < kHalf;
                const int64_t paired_k = first_half ? k : k - kHalf;
                const bool base_positive =
                    ((paired_k / static_cast<int64_t>(block_size)) & 1LL) == 0;
                positive = first_half ? base_positive : !base_positive;
            }
            const int64_t logical_offset =
                mat2.view_offset + k * mat2.strides[0] +
                n * mat2.strides[1];
            if (logical_offset < 0 ||
                logical_offset >= mat2.storage_elements) {
                return false;
            }
            const uint16_t target = positive ? positive_bits : negative_bits;
            if (values[logical_offset] != target) {
                ++changed_count;
            }
            values[logical_offset] = target;
            positive_count += positive ? 1U : 0U;
            negative_count += positive ? 0U : 1U;
        }
    }
    *changed = changed_count;
    *unchanged = static_cast<uint64_t>(mat2.storage_elements) - changed_count;
    *positive_elements = positive_count;
    *negative_elements = negative_count;
    return positive_count == negative_count &&
           positive_count + negative_count ==
               static_cast<uint64_t>(mat2.storage_elements);
}

void free_float_matmul_finite_buffers(FloatMatmulFiniteAttempt *attempt) {
    if (attempt->self_temporary != nullptr) {
        attempt->self_device_free_status =
            static_cast<int32_t>(g_free(attempt->self_temporary));
        attempt->self_temporary_freed =
            attempt->self_device_free_status == ACL_SUCCESS;
    }
    if (attempt->mat2_temporary != nullptr) {
        attempt->mat2_device_free_status =
            static_cast<int32_t>(g_free(attempt->mat2_temporary));
        attempt->mat2_temporary_freed =
            attempt->mat2_device_free_status == ACL_SUCCESS;
    }
    if (attempt->self_host != nullptr) {
        attempt->self_host_free_status =
            static_cast<int32_t>(g_free_host(attempt->self_host));
        attempt->self_host_freed =
            attempt->self_host_free_status == ACL_SUCCESS;
    }
    if (attempt->mat2_host != nullptr) {
        attempt->mat2_host_free_status =
            static_cast<int32_t>(g_free_host(attempt->mat2_host));
        attempt->mat2_host_freed =
            attempt->mat2_host_free_status == ACL_SUCCESS;
    }
}

void log_float_matmul_finite_substitution(
    const char *outcome, op::internal::OpExecCache *cache,
    const RegisteredMatmulMetadata &metadata, const std::string &api_info,
    const FloatMatmulFiniteAttempt &attempt) {
    const uint64_t self_elements =
        static_cast<uint64_t>(metadata.tensors.self.storage_elements);
    const uint64_t mat2_elements =
        static_cast<uint64_t>(metadata.tensors.mat2.storage_elements);
    const bool cancellation_mode =
        attempt.mode == FAULT20_MODE_FLOAT_MATMUL_CANCELLATION;
    const char *event = cancellation_mode
                            ? "float_matmul_cancellation_substitution"
                            : "float_matmul_finite_dual_substitution";
    const char *pattern = cancellation_mode
                              ? float_matmul_cancellation_pattern_name(
                                    attempt.cancellation_pattern,
                                    attempt.cancellation_block_size)
                              : "all_positive";
    char line[8192];
    const int written = std::snprintf(
        line, sizeof(line),
        "{\"event\":\"%s\","
        "\"outcome\":\"%s\",\"pid\":%ld,\"tid\":%ld,"
        "\"ts_ns\":%" PRIu64 ",\"device\":%d,\"hit\":%u,"
        "\"candidate\":%u,\"cache\":\"%p\",\"api_info\":\"%s\","
        "\"shape\":{\"self\":[1,12800],\"mat2\":[12800,5120],"
        "\"out\":[1,5120]},\"mode\":%u,\"pattern\":\"%s\","
        "\"block_size\":%u,\"exponent\":%d,\"finite_value\":%.17g,"
        "\"bf16_bits\":\"0x%04x\",\"input_dtype\":27,"
        "\"mat2_sign_elements\":{\"positive\":%" PRIu64
        ",\"negative\":%" PRIu64 "},"
        "\"per_output_terms\":{\"positive\":%" PRIu64
        ",\"negative\":%" PRIu64 "},"
        "\"theoretical_output_exact_zero\":%s,"
        "\"mat2_layout_rule\":\"offset=k*stride0+n*stride1\","
        "\"input_finite_after\":{\"self\":%" PRIu64
        ",\"mat2\":%" PRIu64 "},"
        "\"changed_elements\":{\"self\":%" PRIu64
        ",\"mat2\":%" PRIu64 ",\"total\":%" PRIu64 "},"
        "\"unchanged_elements\":{\"self\":%" PRIu64
        ",\"mat2\":%" PRIu64 "},"
        "\"bytes\":{\"self\":%" PRIu64 ",\"mat2\":%" PRIu64
        ",\"out\":%" PRIu64 "},"
        "\"addresses\":{\"self_original\":\"%p\","
        "\"self_temporary\":\"%p\",\"mat2_original\":\"%p\","
        "\"mat2_temporary\":\"%p\"},"
        "\"address_distinct\":{\"self\":%s,\"mat2\":%s},"
        "\"temporary_used\":{\"self\":%s,\"mat2\":%s},"
        "\"status\":{\"result\":%d,\"pre_sync\":%d,"
        "\"self_host_malloc\":%d,\"mat2_host_malloc\":%d,"
        "\"self_d2h\":%d,\"mat2_d2h\":%d,"
        "\"self_device_malloc\":%d,\"mat2_device_malloc\":%d,"
        "\"self_h2d\":%d,\"mat2_h2d\":%d,\"run\":%d,"
        "\"post_sync\":%d,\"self_device_free\":%d,"
        "\"mat2_device_free\":%d,\"self_host_free\":%d,"
        "\"mat2_host_free\":%d},"
        "\"freed\":{\"self_device\":%s,\"mat2_device\":%s,"
        "\"self_host\":%s,\"mat2_host\":%s},"
        "\"timing_ns\":{\"pre_sync\":%" PRIu64
        ",\"host_alloc\":%" PRIu64 ",\"d2h\":%" PRIu64
        ",\"transform\":%" PRIu64 ",\"device_alloc\":%" PRIu64
        ",\"h2d_enqueue\":%" PRIu64 ",\"real_run\":%" PRIu64
        ",\"post_sync\":%" PRIu64 ",\"total\":%" PRIu64 "}}",
        event, outcome, static_cast<long>(getpid()),
        static_cast<long>(syscall(SYS_gettid)), attempt.begin_ns,
        metadata.device, attempt.hit_index, attempt.candidate_index, cache,
        api_info.c_str(), attempt.mode, pattern,
        attempt.cancellation_block_size, attempt.exponent,
        std::ldexp(1.0, attempt.exponent),
        static_cast<unsigned int>(attempt.bf16_bits),
        attempt.mat2_positive_elements, attempt.mat2_negative_elements,
        attempt.mat2_positive_elements / 5120U,
        attempt.mat2_negative_elements / 5120U,
        attempt.theoretical_output_exact_zero ? "true" : "false",
        self_elements,
        mat2_elements, attempt.self_changed_elements,
        attempt.mat2_changed_elements,
        attempt.self_changed_elements + attempt.mat2_changed_elements,
        attempt.self_unchanged_elements, attempt.mat2_unchanged_elements,
        metadata.tensors.self.storage_bytes,
        metadata.tensors.mat2.storage_bytes, metadata.tensors.out.storage_bytes,
        attempt.self_original, attempt.self_temporary, attempt.mat2_original,
        attempt.mat2_temporary,
        attempt.self_temporary != nullptr &&
                attempt.self_temporary != attempt.self_original
            ? "true"
            : "false",
        attempt.mat2_temporary != nullptr &&
                attempt.mat2_temporary != attempt.mat2_original
            ? "true"
            : "false",
        attempt.self_temporary_used ? "true" : "false",
        attempt.mat2_temporary_used ? "true" : "false",
        attempt.result_status, attempt.pre_sync_status,
        attempt.self_host_malloc_status, attempt.mat2_host_malloc_status,
        attempt.self_d2h_status, attempt.mat2_d2h_status,
        attempt.self_device_malloc_status, attempt.mat2_device_malloc_status,
        attempt.self_h2d_status, attempt.mat2_h2d_status, attempt.run_status,
        attempt.post_sync_status, attempt.self_device_free_status,
        attempt.mat2_device_free_status, attempt.self_host_free_status,
        attempt.mat2_host_free_status,
        attempt.self_temporary_freed ? "true" : "false",
        attempt.mat2_temporary_freed ? "true" : "false",
        attempt.self_host_freed ? "true" : "false",
        attempt.mat2_host_freed ? "true" : "false", attempt.pre_sync_ns,
        attempt.host_alloc_ns, attempt.d2h_ns, attempt.transform_ns,
        attempt.device_alloc_ns, attempt.h2d_ns, attempt.real_run_ns,
        attempt.post_sync_ns, monotonic_ns() - attempt.begin_ns);
    if (written > 0 && static_cast<size_t>(written) < sizeof(line)) {
        write_line(line, static_cast<size_t>(written));
    }
}

aclnnStatus run_float_matmul_finite_dual(
    op::internal::OpExecCache *cache, void *workspace, aclrtStream stream,
    const std::vector<void *> &addresses, Fault20Control *control,
    const RegisteredMatmulMetadata &metadata, const std::string &api_info,
    uint32_t hit_index, uint32_t candidate_index, uint32_t mode,
    int32_t exponent, uint32_t cancellation_pattern,
    uint32_t cancellation_block_size) {
    FloatMatmulFiniteAttempt attempt{};
    attempt.begin_ns = monotonic_ns();
    attempt.mode = mode;
    attempt.hit_index = hit_index;
    attempt.candidate_index = candidate_index;
    attempt.cancellation_pattern = cancellation_pattern;
    attempt.cancellation_block_size = cancellation_block_size;
    attempt.exponent = exponent;
    attempt.bf16_bits = bf16_power_of_two_bits(exponent);
    attempt.self_original = addresses[0];
    attempt.mat2_original = addresses[1];

    const bool cancellation_mode =
        mode == FAULT20_MODE_FLOAT_MATMUL_CANCELLATION;

    auto fallback = [&](const char *outcome) -> aclnnStatus {
        const uint64_t begin = monotonic_ns();
        const aclnnStatus status =
            run_original(cache, workspace, stream, addresses);
        attempt.real_run_ns = monotonic_ns() - begin;
        attempt.run_status = static_cast<int32_t>(status);
        __atomic_store_n(&control->enabled, 0U, __ATOMIC_RELEASE);
        set_device_last_status(control, metadata.device,
                               attempt.result_status);
        log_float_matmul_finite_substitution(outcome, cache, metadata,
                                             api_info, attempt);
        return status;
    };

    std::vector<void *> addresses_copy;
    try {
        addresses_copy = addresses;
    } catch (...) {
        attempt.result_status = kStatusVectorCopyFailed;
        return fallback("address_vector_copy_failed");
    }

    uint64_t begin = monotonic_ns();
    attempt.pre_sync_status =
        static_cast<int32_t>(g_synchronize_stream(stream));
    attempt.pre_sync_ns = monotonic_ns() - begin;
    if (attempt.pre_sync_status != ACL_SUCCESS) {
        attempt.result_status = kStatusPreSyncFailed;
        return fallback("pre_sync_failed");
    }

    begin = monotonic_ns();
    attempt.self_host_malloc_status = static_cast<int32_t>(g_malloc_host(
        &attempt.self_host,
        static_cast<size_t>(metadata.tensors.self.storage_bytes)));
    if (attempt.self_host_malloc_status == ACL_SUCCESS) {
        attempt.mat2_host_malloc_status = static_cast<int32_t>(g_malloc_host(
            &attempt.mat2_host,
            static_cast<size_t>(metadata.tensors.mat2.storage_bytes)));
    }
    attempt.host_alloc_ns = monotonic_ns() - begin;
    if (attempt.self_host_malloc_status != ACL_SUCCESS ||
        attempt.mat2_host_malloc_status != ACL_SUCCESS ||
        attempt.self_host == nullptr || attempt.mat2_host == nullptr) {
        free_float_matmul_finite_buffers(&attempt);
        attempt.result_status = kStatusHostAllocFailed;
        return fallback("host_alloc_failed");
    }

    begin = monotonic_ns();
    attempt.self_d2h_status = static_cast<int32_t>(g_memcpy(
        attempt.self_host,
        static_cast<size_t>(metadata.tensors.self.storage_bytes),
        attempt.self_original,
        static_cast<size_t>(metadata.tensors.self.storage_bytes),
        ACL_MEMCPY_DEVICE_TO_HOST));
    if (attempt.self_d2h_status == ACL_SUCCESS) {
        attempt.mat2_d2h_status = static_cast<int32_t>(g_memcpy(
            attempt.mat2_host,
            static_cast<size_t>(metadata.tensors.mat2.storage_bytes),
            attempt.mat2_original,
            static_cast<size_t>(metadata.tensors.mat2.storage_bytes),
            ACL_MEMCPY_DEVICE_TO_HOST));
    }
    attempt.d2h_ns = monotonic_ns() - begin;
    if (attempt.self_d2h_status != ACL_SUCCESS ||
        attempt.mat2_d2h_status != ACL_SUCCESS) {
        free_float_matmul_finite_buffers(&attempt);
        attempt.result_status = kStatusD2HFailed;
        return fallback("d2h_failed");
    }

    begin = monotonic_ns();
    fill_bf16_power_of_two(
        attempt.self_host,
        static_cast<uint64_t>(metadata.tensors.self.storage_elements),
        attempt.bf16_bits, &attempt.self_changed_elements,
        &attempt.self_unchanged_elements);
    bool transform_valid = true;
    if (cancellation_mode) {
        transform_valid = float_matmul_cancellation_metadata_valid(
                              metadata.tensors) &&
                          fill_bf16_cancellation_mat2(
                              attempt.mat2_host, metadata.tensors.mat2,
                              attempt.bf16_bits, cancellation_pattern,
                              cancellation_block_size,
                              &attempt.mat2_changed_elements,
                              &attempt.mat2_unchanged_elements,
                              &attempt.mat2_positive_elements,
                              &attempt.mat2_negative_elements);
        attempt.theoretical_output_exact_zero =
            transform_valid && attempt.mat2_positive_elements == 32768000U &&
            attempt.mat2_negative_elements == 32768000U;
    } else {
        fill_bf16_power_of_two(
            attempt.mat2_host,
            static_cast<uint64_t>(metadata.tensors.mat2.storage_elements),
            attempt.bf16_bits, &attempt.mat2_changed_elements,
            &attempt.mat2_unchanged_elements);
        attempt.mat2_positive_elements =
            static_cast<uint64_t>(metadata.tensors.mat2.storage_elements);
        attempt.mat2_negative_elements = 0;
    }
    attempt.transform_ns = monotonic_ns() - begin;
    if (!transform_valid ||
        (cancellation_mode && !attempt.theoretical_output_exact_zero)) {
        free_float_matmul_finite_buffers(&attempt);
        attempt.result_status = kStatusOutputMetadataRejected;
        return fallback("cancellation_transform_or_layout_rejected");
    }
    if (attempt.self_changed_elements == 0 ||
        attempt.mat2_changed_elements == 0) {
        free_float_matmul_finite_buffers(&attempt);
        attempt.result_status = kStatusNoChangedElements;
        return fallback("no_changed_elements");
    }

    begin = monotonic_ns();
    attempt.self_device_malloc_status = static_cast<int32_t>(g_malloc(
        &attempt.self_temporary,
        static_cast<size_t>(metadata.tensors.self.storage_bytes),
        ACL_MEM_MALLOC_HUGE_FIRST));
    if (attempt.self_device_malloc_status == ACL_SUCCESS) {
        attempt.mat2_device_malloc_status = static_cast<int32_t>(g_malloc(
            &attempt.mat2_temporary,
            static_cast<size_t>(metadata.tensors.mat2.storage_bytes),
            ACL_MEM_MALLOC_HUGE_FIRST));
    }
    attempt.device_alloc_ns = monotonic_ns() - begin;
    if (attempt.self_device_malloc_status != ACL_SUCCESS ||
        attempt.mat2_device_malloc_status != ACL_SUCCESS ||
        attempt.self_temporary == nullptr || attempt.mat2_temporary == nullptr ||
        attempt.self_temporary == attempt.self_original ||
        attempt.mat2_temporary == attempt.mat2_original) {
        free_float_matmul_finite_buffers(&attempt);
        attempt.result_status = kStatusDeviceAllocFailed;
        return fallback("device_alloc_or_address_distinct_failed");
    }

    begin = monotonic_ns();
    attempt.self_h2d_status = static_cast<int32_t>(g_memcpy_async(
        attempt.self_temporary,
        static_cast<size_t>(metadata.tensors.self.storage_bytes),
        attempt.self_host,
        static_cast<size_t>(metadata.tensors.self.storage_bytes),
        ACL_MEMCPY_HOST_TO_DEVICE, stream));
    if (attempt.self_h2d_status == ACL_SUCCESS) {
        attempt.mat2_h2d_status = static_cast<int32_t>(g_memcpy_async(
            attempt.mat2_temporary,
            static_cast<size_t>(metadata.tensors.mat2.storage_bytes),
            attempt.mat2_host,
            static_cast<size_t>(metadata.tensors.mat2.storage_bytes),
            ACL_MEMCPY_HOST_TO_DEVICE, stream));
    }
    attempt.h2d_ns = monotonic_ns() - begin;
    if (attempt.self_h2d_status != ACL_SUCCESS ||
        attempt.mat2_h2d_status != ACL_SUCCESS) {
        begin = monotonic_ns();
        attempt.post_sync_status =
            static_cast<int32_t>(g_synchronize_stream(stream));
        attempt.post_sync_ns = monotonic_ns() - begin;
        if (attempt.post_sync_status == ACL_SUCCESS) {
            free_float_matmul_finite_buffers(&attempt);
        }
        attempt.result_status = kStatusH2DFailed;
        return fallback("h2d_failed");
    }

    addresses_copy[0] = attempt.self_temporary;
    addresses_copy[1] = attempt.mat2_temporary;
    attempt.self_temporary_used = true;
    attempt.mat2_temporary_used = true;
    begin = monotonic_ns();
    const aclnnStatus run_status =
        run_original(cache, workspace, stream, addresses_copy);
    attempt.real_run_ns = monotonic_ns() - begin;
    attempt.run_status = static_cast<int32_t>(run_status);

    begin = monotonic_ns();
    attempt.post_sync_status =
        static_cast<int32_t>(g_synchronize_stream(stream));
    attempt.post_sync_ns = monotonic_ns() - begin;
    if (attempt.post_sync_status == ACL_SUCCESS) {
        free_float_matmul_finite_buffers(&attempt);
    }

    const char *outcome = "success";
    if (attempt.run_status != ACL_SUCCESS) {
        attempt.result_status = kStatusRunFailed;
        outcome = "real_run_failed";
    } else if (attempt.post_sync_status != ACL_SUCCESS) {
        attempt.result_status = kStatusPostSyncFailed;
        outcome = "post_sync_failed_buffers_retained";
    } else if (!attempt.self_temporary_freed ||
               !attempt.mat2_temporary_freed) {
        attempt.result_status = kStatusDeviceFreeFailed;
        outcome = "device_free_failed";
    } else if (!attempt.self_host_freed || !attempt.mat2_host_freed) {
        attempt.result_status = kStatusHostFreeFailed;
        outcome = "host_free_failed";
    } else {
        attempt.result_status = kStatusSuccess;
    }
    if (attempt.result_status != kStatusSuccess) {
        __atomic_store_n(&control->enabled, 0U, __ATOMIC_RELEASE);
    }
    set_device_last_status(control, metadata.device, attempt.result_status);
    log_float_matmul_finite_substitution(outcome, cache, metadata, api_info,
                                         attempt);
    if (attempt.result_status == kStatusSuccess) {
        observe_float_matmul_output_after_run(
            cache, stream, addresses, control, metadata, api_info, hit_index,
            candidate_index, run_status, attempt.real_run_ns);
    }
    return run_status;
}

const char *float_matmul_output_proxy_name(uint32_t proxy_kind) {
    if (proxy_kind == FAULT20_OUTPUT_PROXY_POS_INF) {
        return "assign_pos_inf";
    }
    if (proxy_kind == FAULT20_OUTPUT_PROXY_NEG_INF) {
        return "assign_neg_inf";
    }
    if (proxy_kind == FAULT20_OUTPUT_PROXY_QNAN) {
        return "assign_qnan";
    }
    return "invalid";
}

uint16_t float_matmul_output_proxy_bits(uint32_t proxy_kind) {
    if (proxy_kind == FAULT20_OUTPUT_PROXY_POS_INF) {
        return 0x7F80U;
    }
    if (proxy_kind == FAULT20_OUTPUT_PROXY_NEG_INF) {
        return 0xFF80U;
    }
    if (proxy_kind == FAULT20_OUTPUT_PROXY_QNAN) {
        return 0x7FC0U;
    }
    return 0;
}

bool valid_float_matmul_output_proxy_config(uint32_t proxy_kind,
                                            uint32_t element_count) {
    const bool kind_valid =
        proxy_kind == FAULT20_OUTPUT_PROXY_POS_INF ||
        proxy_kind == FAULT20_OUTPUT_PROXY_NEG_INF ||
        proxy_kind == FAULT20_OUTPUT_PROXY_QNAN;
    const bool count_valid = element_count == 1 || element_count == 5 ||
                             element_count == 51 || element_count == 512;
    return kind_valid && count_valid;
}

void log_float_matmul_output_proxy(
    const char *outcome, op::internal::OpExecCache *cache,
    const RegisteredMatmulMetadata &metadata, const std::string &api_info,
    const FloatMatmulOutputProxyAttempt &attempt) {
    char line[8192];
    const int written = std::snprintf(
        line, sizeof(line),
        "{\"event\":\"float_matmul_output_nonfinite_proxy\","
        "\"outcome\":\"%s\",\"pid\":%ld,\"tid\":%ld,"
        "\"ts_ns\":%" PRIu64 ",\"device\":%d,\"hit\":%u,"
        "\"candidate\":%u,\"cache\":\"%p\",\"api_info\":\"%s\","
        "\"shape\":[1,5120],\"dtype\":27,\"out_slot\":2,"
        "\"out_address\":\"%p\",\"proxy\":\"%s\","
        "\"bf16_bits\":\"0x%04x\",\"selection\":\"prefix\","
        "\"requested_elements\":%u,\"changed_elements\":%" PRIu64
        ",\"actual_ratio_bps\":%.9g,\"readback_exact\":%s,"
        "\"before\":{\"finite\":%" PRIu64 ",\"zero\":%" PRIu64
        ",\"nan\":%" PRIu64 ",\"positive_inf\":%" PRIu64
        ",\"negative_inf\":%" PRIu64 ",\"min\":%.17g,"
        "\"max\":%.17g,\"max_abs\":%.17g,\"mean_abs\":%.17g},"
        "\"after\":{\"finite\":%" PRIu64 ",\"zero\":%" PRIu64
        ",\"nan\":%" PRIu64 ",\"positive_inf\":%" PRIu64
        ",\"negative_inf\":%" PRIu64 ",\"min\":%.17g,"
        "\"max\":%.17g,\"max_abs\":%.17g,\"mean_abs\":%.17g},"
        "\"status\":{\"result\":%d,\"run\":%d,\"pre_sync\":%d,"
        "\"host_malloc\":%d,\"pre_d2h\":%d,\"h2d\":%d,"
        "\"post_sync\":%d,\"readback\":%d,\"host_free\":%d},"
        "\"host_freed\":%s,\"auto_disarmed\":true,"
        "\"timing_ns\":{\"real_run\":%" PRIu64
        ",\"pre_sync\":%" PRIu64 ",\"host_alloc\":%" PRIu64
        ",\"pre_d2h\":%" PRIu64 ",\"mutate\":%" PRIu64
        ",\"h2d\":%" PRIu64 ",\"post_sync\":%" PRIu64
        ",\"readback\":%" PRIu64 ",\"scan\":%" PRIu64
        ",\"total\":%" PRIu64 "}}",
        outcome, static_cast<long>(getpid()),
        static_cast<long>(syscall(SYS_gettid)), attempt.begin_ns,
        metadata.device, attempt.hit_index, attempt.candidate_index, cache,
        api_info.c_str(), attempt.out_address,
        float_matmul_output_proxy_name(attempt.proxy_kind),
        static_cast<unsigned int>(attempt.proxy_bits),
        attempt.requested_elements, attempt.changed_elements,
        static_cast<double>(attempt.requested_elements) * 10000.0 / 5120.0,
        attempt.readback_exact ? "true" : "false",
        attempt.before.finite_count, attempt.before.zero_count,
        attempt.before.nan_count, attempt.before.positive_inf_count,
        attempt.before.negative_inf_count, attempt.before.min_value,
        attempt.before.max_value, attempt.before.max_abs,
        attempt.before.mean_abs, attempt.after.finite_count,
        attempt.after.zero_count, attempt.after.nan_count,
        attempt.after.positive_inf_count, attempt.after.negative_inf_count,
        attempt.after.min_value, attempt.after.max_value,
        attempt.after.max_abs, attempt.after.mean_abs, attempt.result_status,
        attempt.run_status, attempt.pre_sync_status,
        attempt.host_malloc_status, attempt.pre_d2h_status,
        attempt.h2d_status, attempt.post_sync_status,
        attempt.readback_status, attempt.host_free_status,
        attempt.host_freed ? "true" : "false", attempt.real_run_ns,
        attempt.pre_sync_ns, attempt.host_alloc_ns, attempt.pre_d2h_ns,
        attempt.mutate_ns, attempt.h2d_ns, attempt.post_sync_ns,
        attempt.readback_ns, attempt.scan_ns,
        monotonic_ns() - attempt.begin_ns);
    if (written > 0 && static_cast<size_t>(written) < sizeof(line)) {
        write_line(line, static_cast<size_t>(written));
    }
}

aclnnStatus run_float_matmul_output_proxy(
    op::internal::OpExecCache *cache, void *workspace, aclrtStream stream,
    const std::vector<void *> &addresses, Fault20Control *control,
    const RegisteredMatmulMetadata &metadata, const std::string &api_info,
    uint32_t hit_index, uint32_t candidate_index, uint32_t proxy_kind,
    uint32_t element_count) {
    FloatMatmulOutputProxyAttempt attempt{};
    attempt.begin_ns = monotonic_ns();
    attempt.hit_index = hit_index;
    attempt.candidate_index = candidate_index;
    attempt.proxy_kind = proxy_kind;
    attempt.requested_elements = element_count;
    attempt.proxy_bits = float_matmul_output_proxy_bits(proxy_kind);
    attempt.out_address = addresses[2];

    auto finish = [&](const char *outcome) -> aclnnStatus {
        __atomic_store_n(&control->enabled, 0U, __ATOMIC_RELEASE);
        set_device_last_status(control, metadata.device, attempt.result_status);
        log_float_matmul_output_proxy(outcome, cache, metadata, api_info,
                                      attempt);
        return static_cast<aclnnStatus>(attempt.run_status);
    };

    uint64_t begin = monotonic_ns();
    const aclnnStatus run_status =
        run_original(cache, workspace, stream, addresses);
    attempt.real_run_ns = monotonic_ns() - begin;
    attempt.run_status = static_cast<int32_t>(run_status);
    if (run_status != ACL_SUCCESS) {
        attempt.result_status = kStatusRunFailed;
        return finish("real_run_failed_no_mutation");
    }

    begin = monotonic_ns();
    attempt.pre_sync_status =
        static_cast<int32_t>(g_synchronize_stream(stream));
    attempt.pre_sync_ns = monotonic_ns() - begin;
    if (attempt.pre_sync_status != ACL_SUCCESS) {
        attempt.result_status = kStatusPreSyncFailed;
        return finish("pre_sync_failed_no_mutation");
    }

    const size_t bytes =
        static_cast<size_t>(metadata.tensors.out.storage_bytes);
    begin = monotonic_ns();
    attempt.host_malloc_status =
        static_cast<int32_t>(g_malloc_host(&attempt.host_buffer, bytes));
    attempt.host_alloc_ns = monotonic_ns() - begin;
    if (attempt.host_malloc_status != ACL_SUCCESS ||
        attempt.host_buffer == nullptr) {
        attempt.result_status = kStatusHostAllocFailed;
        return finish("host_alloc_failed_no_mutation");
    }

    begin = monotonic_ns();
    attempt.pre_d2h_status = static_cast<int32_t>(g_memcpy(
        attempt.host_buffer, bytes, attempt.out_address, bytes,
        ACL_MEMCPY_DEVICE_TO_HOST));
    attempt.pre_d2h_ns = monotonic_ns() - begin;
    if (attempt.pre_d2h_status != ACL_SUCCESS) {
        attempt.host_free_status =
            static_cast<int32_t>(g_free_host(attempt.host_buffer));
        attempt.host_freed = attempt.host_free_status == ACL_SUCCESS;
        attempt.result_status = kStatusD2HFailed;
        return finish("pre_d2h_failed_no_mutation");
    }

    begin = monotonic_ns();
    scan_qmm_output(attempt.host_buffer, metadata.tensors.out,
                    &attempt.before);
    if (!attempt.before.stats_valid || attempt.before.nan_count != 0 ||
        attempt.before.positive_inf_count != 0 ||
        attempt.before.negative_inf_count != 0) {
        attempt.host_free_status =
            static_cast<int32_t>(g_free_host(attempt.host_buffer));
        attempt.host_freed = attempt.host_free_status == ACL_SUCCESS;
        attempt.result_status = kStatusOutputMetadataRejected;
        return finish("pre_output_nonfinite_or_invalid_no_mutation");
    }
    auto *values = static_cast<uint16_t *>(attempt.host_buffer);
    for (uint32_t index = 0; index < element_count; ++index) {
        if (values[index] != attempt.proxy_bits) {
            ++attempt.changed_elements;
        }
        values[index] = attempt.proxy_bits;
    }
    attempt.mutate_ns = monotonic_ns() - begin;
    if (attempt.changed_elements != element_count) {
        attempt.host_free_status =
            static_cast<int32_t>(g_free_host(attempt.host_buffer));
        attempt.host_freed = attempt.host_free_status == ACL_SUCCESS;
        attempt.result_status = kStatusNoChangedElements;
        return finish("changed_count_mismatch_no_writeback");
    }

    begin = monotonic_ns();
    attempt.h2d_status = static_cast<int32_t>(g_memcpy(
        attempt.out_address, bytes, attempt.host_buffer, bytes,
        ACL_MEMCPY_HOST_TO_DEVICE));
    attempt.h2d_ns = monotonic_ns() - begin;
    if (attempt.h2d_status != ACL_SUCCESS) {
        attempt.host_free_status =
            static_cast<int32_t>(g_free_host(attempt.host_buffer));
        attempt.host_freed = attempt.host_free_status == ACL_SUCCESS;
        attempt.result_status = kStatusH2DFailed;
        return finish("h2d_failed");
    }

    begin = monotonic_ns();
    attempt.post_sync_status =
        static_cast<int32_t>(g_synchronize_stream(stream));
    attempt.post_sync_ns = monotonic_ns() - begin;
    if (attempt.post_sync_status != ACL_SUCCESS) {
        attempt.result_status = kStatusPostSyncFailed;
        return finish("post_sync_failed_host_retained");
    }

    begin = monotonic_ns();
    attempt.readback_status = static_cast<int32_t>(g_memcpy(
        attempt.host_buffer, bytes, attempt.out_address, bytes,
        ACL_MEMCPY_DEVICE_TO_HOST));
    attempt.readback_ns = monotonic_ns() - begin;
    if (attempt.readback_status == ACL_SUCCESS) {
        begin = monotonic_ns();
        scan_qmm_output(attempt.host_buffer, metadata.tensors.out,
                        &attempt.after);
        attempt.readback_exact = true;
        const auto *readback =
            static_cast<const uint16_t *>(attempt.host_buffer);
        for (uint32_t index = 0; index < element_count; ++index) {
            if (readback[index] != attempt.proxy_bits) {
                attempt.readback_exact = false;
                break;
            }
        }
        attempt.scan_ns = monotonic_ns() - begin;
    }
    attempt.host_free_status =
        static_cast<int32_t>(g_free_host(attempt.host_buffer));
    attempt.host_freed = attempt.host_free_status == ACL_SUCCESS;

    const uint64_t expected_finite = 5120U - element_count;
    const uint64_t expected_nan =
        proxy_kind == FAULT20_OUTPUT_PROXY_QNAN ? element_count : 0U;
    const uint64_t expected_pos_inf =
        proxy_kind == FAULT20_OUTPUT_PROXY_POS_INF ? element_count : 0U;
    const uint64_t expected_neg_inf =
        proxy_kind == FAULT20_OUTPUT_PROXY_NEG_INF ? element_count : 0U;
    if (attempt.readback_status != ACL_SUCCESS) {
        attempt.result_status = kStatusD2HFailed;
        return finish("readback_failed");
    }
    if (!attempt.after.stats_valid || !attempt.readback_exact ||
        attempt.after.finite_count != expected_finite ||
        attempt.after.nan_count != expected_nan ||
        attempt.after.positive_inf_count != expected_pos_inf ||
        attempt.after.negative_inf_count != expected_neg_inf) {
        attempt.result_status = kStatusOutputMetadataRejected;
        return finish("readback_count_mismatch");
    }
    if (!attempt.host_freed) {
        attempt.result_status = kStatusHostFreeFailed;
        return finish("host_free_failed");
    }
    attempt.result_status = kStatusSuccess;
    return finish("success");
}

void log_qmm_output_observation(const char *source_mode,
                                op::internal::OpExecCache *cache,
                                const RegisteredMetadata &metadata,
                                const std::string &api_info,
                                const QmmOutputObservation &observation) {
    const auto &out = metadata.tensors.out;
    const char *outcome = observation.result_status == kStatusSuccess
                              ? "success"
                              : "failed_closed";
    char line[6144];
    const int written = std::snprintf(
        line, sizeof(line),
        "{\"event\":\"qmm_output_observation\",\"pid\":%ld,\"tid\":%ld,"
        "\"ts_ns\":%" PRIu64 ",\"outcome\":\"%s\","
        "\"source_mode\":\"%s\",\"device\":%d,\"hit\":%u,"
        "\"candidate\":%u,\"cache\":\"%p\",\"api_info\":\"%s\","
        "\"out_slot\":%d,\"out_address\":\"%p\",\"dtype\":%d,"
        "\"format\":%d,\"shape\":[%" PRId64 ",%" PRId64 "],"
        "\"strides\":[%" PRId64 ",%" PRId64 "],\"offset\":%" PRId64
        ",\"bytes\":%" PRIu64 ",\"stats_valid\":%s,"
        "\"finite_count\":%" PRIu64 ",\"nan_count\":%" PRIu64
        ",\"positive_inf_count\":%" PRIu64
        ",\"negative_inf_count\":%" PRIu64 ",\"zero_count\":%" PRIu64
        ",\"min\":%.17g,\"max\":%.17g,\"max_abs\":%.17g,"
        "\"mean_abs\":%.17g,\"result_status\":%d,\"run_status\":%d,"
        "\"sync_status\":%d,\"host_malloc_status\":%d,"
        "\"d2h_status\":%d,\"host_free_status\":%d,\"host_freed\":%s,"
        "\"auto_disarmed_on_nonfinite\":%s,"
        "\"real_run_ns\":%" PRIu64 ",\"sync_ns\":%" PRIu64
        ",\"host_alloc_ns\":%" PRIu64 ",\"d2h_ns\":%" PRIu64
        ",\"scan_ns\":%" PRIu64 ",\"total_ns\":%" PRIu64 "}",
        static_cast<long>(getpid()), static_cast<long>(syscall(SYS_gettid)),
        observation.begin_ns, outcome, source_mode, metadata.device,
        observation.hit_index, observation.candidate_index, cache, api_info.c_str(),
        observation.out_slot, observation.out_address, out.dtype, out.format,
        out.view_shape[0], out.view_shape[1], out.strides[0], out.strides[1],
        out.view_offset, out.storage_bytes,
        observation.stats_valid ? "true" : "false", observation.finite_count,
        observation.nan_count, observation.positive_inf_count,
        observation.negative_inf_count, observation.zero_count,
        observation.min_value, observation.max_value, observation.max_abs,
        observation.mean_abs, observation.result_status, observation.run_status,
        observation.sync_status, observation.host_malloc_status,
        observation.d2h_status, observation.host_free_status,
        observation.host_freed ? "true" : "false",
        observation.auto_disarmed_on_nonfinite ? "true" : "false",
        observation.real_run_ns,
        observation.sync_ns, observation.host_alloc_ns, observation.d2h_ns,
        observation.scan_ns, monotonic_ns() - observation.begin_ns);
    if (written > 0 && static_cast<size_t>(written) < sizeof(line)) {
        write_line(line, static_cast<size_t>(written));
    }
}

void log_qmm_weight_observation(op::internal::OpExecCache *cache,
                                const RegisteredMetadata &metadata,
                                const std::string &api_info,
                                const QmmStorageOrderProof &proof,
                                const QmmWeightObservation &observation) {
    const auto &x2 = metadata.tensors.x2;
    const char *outcome = observation.result_status == kStatusSuccess
                              ? "success"
                              : "failed_closed";
    char line[6144];
    const int written = std::snprintf(
        line, sizeof(line),
        "{\"event\":\"qmm_weight_observation\",\"pid\":%ld,\"tid\":%ld,"
        "\"ts_ns\":%" PRIu64 ",\"outcome\":\"%s\",\"device\":%d,"
        "\"hit\":%u,\"candidate\":%u,\"cache\":\"%p\","
        "\"api_info\":\"%s\",\"x2_slot\":%d,\"x2_address\":\"%p\","
        "\"weight_slot_proven\":%s,\"x2_storage_unique\":%s,"
        "\"dtype\":%d,\"format\":%d,\"view_shape\":[%" PRId64
        ",%" PRId64 "],\"storage_shape\":[%" PRId64 ",%" PRId64
        ",%" PRId64 ",%" PRId64 "],\"strides\":[%" PRId64
        ",%" PRId64 "],\"view_offset\":%" PRId64
        ",\"numel\":%" PRId64 ",\"storage_elements\":%" PRId64
        ",\"element_size\":%" PRIu64 ",\"view_bytes\":%" PRIu64
        ",\"storage_bytes\":%" PRIu64 ",\"transpose_x2\":%u,"
        "\"result_status\":%d,\"run_status\":%d,\"real_run_ns\":%" PRIu64
        ",\"total_ns\":%" PRIu64 "}",
        static_cast<long>(getpid()), static_cast<long>(syscall(SYS_gettid)),
        observation.begin_ns, outcome, metadata.device, observation.hit_index,
        observation.candidate_index, cache, api_info.c_str(), observation.x2_slot,
        observation.x2_address, proof.weight_slot_proven ? "true" : "false",
        proof.x2_storage_unique ? "true" : "false", x2.dtype, x2.format,
        x2.view_shape[0], x2.view_shape[1], x2.storage_shape[0],
        x2.storage_shape[1], x2.storage_shape[2], x2.storage_shape[3],
        x2.strides[0], x2.strides[1], x2.view_offset, x2.numel,
        x2.storage_elements, x2.element_size, x2.view_bytes, x2.storage_bytes,
        metadata.tensors.transpose_x2, observation.result_status,
        observation.run_status, observation.real_run_ns,
        monotonic_ns() - observation.begin_ns);
    if (written > 0 && static_cast<size_t>(written) < sizeof(line)) {
        write_line(line, static_cast<size_t>(written));
    }
}

aclnnStatus observe_qmm_weight_and_run(
    op::internal::OpExecCache *cache, void *workspace, aclrtStream stream,
    const std::vector<void *> &addresses, Fault20Control *control,
    const RegisteredMetadata &metadata, const std::string &api_info,
    uint32_t hit_index, uint32_t candidate_index) {
    QmmWeightObservation observation{};
    observation.begin_ns = monotonic_ns();
    observation.hit_index = hit_index;
    observation.candidate_index = candidate_index;
    const QmmStorageOrderProof proof = prove_qmm_storage_order(metadata, addresses);
    observation.x2_slot = proof.slots[1];
    const uint64_t max_weight_bytes =
        __atomic_load_n(&control->max_weight_bytes, __ATOMIC_RELAXED);
    if (!proof.weight_slot_proven || observation.x2_slot < 0 ||
        static_cast<size_t>(observation.x2_slot) >= addresses.size()) {
        observation.result_status = kStatusWeightSlotUnproven;
    } else {
        observation.x2_address = addresses[static_cast<size_t>(observation.x2_slot)];
        if (observation.x2_address == nullptr ||
            !qmm_weight_metadata_valid(metadata.tensors.x2,
                                       metadata.x1.view_shape[1],
                                       max_weight_bytes)) {
            observation.result_status = kStatusWeightMetadataRejected;
        }
    }
    const uint64_t begin = monotonic_ns();
    const aclnnStatus run_status = run_original(cache, workspace, stream, addresses);
    observation.real_run_ns = monotonic_ns() - begin;
    observation.run_status = static_cast<int32_t>(run_status);
    if (observation.result_status == 0) {
        observation.result_status = run_status == ACL_SUCCESS ? kStatusSuccess
                                                              : kStatusRunFailed;
    }
    if (observation.result_status != kStatusSuccess) {
        __atomic_store_n(&control->enabled, 0U, __ATOMIC_RELEASE);
    }
    set_device_last_status(control, metadata.device, observation.result_status);
    log_qmm_weight_observation(cache, metadata, api_info, proof, observation);
    return run_status;
}

void log_qmm_x2_scale_observation(
    op::internal::OpExecCache *cache, const RegisteredMetadata &metadata,
    const std::string &api_info, const QmmStorageOrderProof &proof,
    const QmmScaleObservation &observation) {
    const auto &scale = metadata.tensors.x2_scale;
    const char *outcome = observation.result_status == kStatusSuccess
                              ? "success"
                              : "failed_closed";
    char line[6144];
    const int written = std::snprintf(
        line, sizeof(line),
        "{\"event\":\"qmm_x2_scale_observation\",\"pid\":%ld,\"tid\":%ld,"
        "\"ts_ns\":%" PRIu64 ",\"outcome\":\"%s\",\"device\":%d,"
        "\"hit\":%u,\"candidate\":%u,\"cache\":\"%p\","
        "\"api_info\":\"%s\",\"scale_role\":\"x2_scale\","
        "\"scale_formal_ordinal\":3,\"scale_slot\":%d,"
        "\"scale_address\":\"%p\",\"scale_slot_proven\":%s,"
        "\"scale_storage_unique\":%s,\"x1_scale_present\":%s,"
        "\"y_scale_present\":%s,\"y_scale_storage_bytes\":%" PRIu64 ","
        "\"dtype\":%d,\"format\":%d,\"view_shape\":[%" PRId64
        "],\"storage_shape\":[%" PRId64 "],\"strides\":[%" PRId64
        "],\"view_offset\":%" PRId64 ",\"numel\":%" PRId64
        ",\"storage_elements\":%" PRId64 ",\"element_size\":%" PRIu64
        ",\"view_bytes\":%" PRIu64 ",\"storage_bytes\":%" PRIu64
        ",\"result_status\":%d,\"run_status\":%d,"
        "\"real_run_ns\":%" PRIu64 ",\"total_ns\":%" PRIu64 "}",
        static_cast<long>(getpid()), static_cast<long>(syscall(SYS_gettid)),
        observation.begin_ns, outcome, metadata.device, observation.hit_index,
        observation.candidate_index, cache, api_info.c_str(),
        observation.scale_slot, observation.scale_address,
        proof.x2_scale_slot_proven ? "true" : "false",
        proof.x2_scale_storage_unique ? "true" : "false",
        metadata.tensors.x1_scale.present != 0 ? "true" : "false",
        metadata.tensors.y_scale.present != 0 ? "true" : "false",
        metadata.tensors.y_scale.storage_bytes, scale.dtype, scale.format,
        scale.view_shape[0], scale.storage_shape[0], scale.strides[0],
        scale.view_offset, scale.numel, scale.storage_elements,
        scale.element_size, scale.view_bytes, scale.storage_bytes,
        observation.result_status, observation.run_status,
        observation.real_run_ns, monotonic_ns() - observation.begin_ns);
    if (written > 0 && static_cast<size_t>(written) < sizeof(line)) {
        write_line(line, static_cast<size_t>(written));
    }
}

aclnnStatus observe_qmm_x2_scale_and_run(
    op::internal::OpExecCache *cache, void *workspace, aclrtStream stream,
    const std::vector<void *> &addresses, Fault20Control *control,
    const RegisteredMetadata &metadata, const std::string &api_info,
    uint32_t hit_index, uint32_t candidate_index) {
    QmmScaleObservation observation{};
    observation.begin_ns = monotonic_ns();
    observation.hit_index = hit_index;
    observation.candidate_index = candidate_index;
    const QmmStorageOrderProof proof = prove_qmm_storage_order(metadata, addresses);
    observation.scale_slot = proof.slots[3];
    const uint64_t max_scale_bytes =
        __atomic_load_n(&control->max_scale_bytes, __ATOMIC_RELAXED);
    const int32_t target_scale_elements =
        __atomic_load_n(&control->target_scale_elements, __ATOMIC_RELAXED);
    if (!proof.x2_scale_slot_proven || observation.scale_slot < 0 ||
        static_cast<size_t>(observation.scale_slot) >= addresses.size()) {
        observation.result_status = kStatusScaleSlotUnproven;
    } else {
        observation.scale_address =
            addresses[static_cast<size_t>(observation.scale_slot)];
        if (observation.scale_address == nullptr ||
            !qmm_x2_scale_metadata_valid(metadata.tensors.x2_scale,
                                         target_scale_elements,
                                         max_scale_bytes)) {
            observation.result_status = kStatusScaleMetadataRejected;
        }
    }
    const uint64_t begin = monotonic_ns();
    const aclnnStatus run_status = run_original(cache, workspace, stream, addresses);
    observation.real_run_ns = monotonic_ns() - begin;
    observation.run_status = static_cast<int32_t>(run_status);
    if (observation.result_status == 0) {
        observation.result_status = run_status == ACL_SUCCESS ? kStatusSuccess
                                                              : kStatusRunFailed;
    }
    if (observation.result_status != kStatusSuccess) {
        __atomic_store_n(&control->enabled, 0U, __ATOMIC_RELEASE);
    }
    set_device_last_status(control, metadata.device, observation.result_status);
    log_qmm_x2_scale_observation(cache, metadata, api_info, proof, observation);
    return run_status;
}

void scan_fp32_scale(const void *host_buffer, uint64_t elements,
                     QmmScaleSubstitutionAttempt *attempt) {
    const auto *values = static_cast<const float *>(host_buffer);
    double min_value = std::numeric_limits<double>::infinity();
    double max_value = -std::numeric_limits<double>::infinity();
    double max_abs = 0.0;
    for (uint64_t index = 0; index < elements; ++index) {
        const double value = static_cast<double>(values[index]);
        if (std::isnan(value)) {
            ++attempt->nan_count;
        } else if (std::isinf(value)) {
            if (std::signbit(value)) {
                ++attempt->negative_inf_count;
            } else {
                ++attempt->positive_inf_count;
            }
        } else {
            ++attempt->finite_count;
            attempt->zero_count += value == 0.0 ? 1U : 0U;
            min_value = std::min(min_value, value);
            max_value = std::max(max_value, value);
            max_abs = std::max(max_abs, std::fabs(value));
        }
    }
    if (attempt->finite_count != 0) {
        attempt->min_value = min_value;
        attempt->max_value = max_value;
        attempt->max_abs = max_abs;
    }
    attempt->stats_valid = true;
}

void scan_fp32_scale_after(const void *host_buffer, uint64_t elements,
                           QmmScaleSubstitutionAttempt *attempt) {
    const auto *values = static_cast<const float *>(host_buffer);
    double min_value = std::numeric_limits<double>::infinity();
    double max_value = -std::numeric_limits<double>::infinity();
    double max_abs = 0.0;
    for (uint64_t index = 0; index < elements; ++index) {
        const double value = static_cast<double>(values[index]);
        if (std::isnan(value)) {
            ++attempt->after_nan_count;
        } else if (std::isinf(value)) {
            if (std::signbit(value)) {
                ++attempt->after_negative_inf_count;
            } else {
                ++attempt->after_positive_inf_count;
            }
        } else {
            ++attempt->after_finite_count;
            attempt->after_zero_count += value == 0.0 ? 1U : 0U;
            min_value = std::min(min_value, value);
            max_value = std::max(max_value, value);
            max_abs = std::max(max_abs, std::fabs(value));
        }
    }
    if (attempt->after_finite_count != 0) {
        attempt->after_min_value = min_value;
        attempt->after_max_value = max_value;
        attempt->after_max_abs = max_abs;
    }
    attempt->after_stats_valid = true;
}

void amplify_fp32_scale(void *host_buffer, uint64_t elements, double factor,
                        QmmScaleSubstitutionAttempt *attempt) {
    auto *values = static_cast<float *>(host_buffer);
    const float float_factor = static_cast<float>(factor);
    for (uint64_t index = 0; index < elements; ++index) {
        const float before = values[index];
        const float after = before * float_factor;
        attempt->changed_elements += before != after ? 1U : 0U;
        values[index] = after;
    }
}

void free_scale_substitution_buffers(QmmScaleSubstitutionAttempt *attempt) {
    if (attempt->scale_temporary != nullptr) {
        attempt->device_free_status = g_free(attempt->scale_temporary);
        attempt->temporary_freed = attempt->device_free_status == ACL_SUCCESS;
    }
    if (attempt->scale_host != nullptr) {
        attempt->host_free_status = g_free_host(attempt->scale_host);
        attempt->host_freed = attempt->host_free_status == ACL_SUCCESS;
    }
}

void log_qmm_x2_scale_substitution(
    const char *outcome, op::internal::OpExecCache *cache,
    const RegisteredMetadata &metadata, const std::string &api_info,
    const QmmScaleSubstitutionAttempt &attempt) {
    const auto &scale = metadata.tensors.x2_scale;
    char line[8192];
    const int written = std::snprintf(
        line, sizeof(line),
        "{\"event\":\"%s\","
        "\"pid\":%ld,\"tid\":%ld,\"ts_ns\":%" PRIu64
        ",\"outcome\":\"%s\",\"device\":%d,\"hit\":%u,"
        "\"candidate\":%u,\"cache\":\"%p\",\"api_info\":\"%s\","
        "\"scale_role\":\"x2_scale\",\"scale_formal_ordinal\":3,"
        "\"scale_slot\":%d,\"scale_bytes\":%" PRIu64
        ",\"scale_elements\":%" PRId64 ",\"scale_original\":\"%p\","
        "\"scale_temporary\":\"%p\",\"scale_host\":\"%p\","
        "\"temporary_used\":%s,\"temporary_freed\":%s,"
        "\"host_freed\":%s,\"scale_factor\":%.17g,"
        "\"changed_elements\":%" PRIu64
        ",\"before_stats_valid\":%s,\"before_finite_count\":%" PRIu64
        ",\"before_nan_count\":%" PRIu64
        ",\"before_positive_inf_count\":%" PRIu64
        ",\"before_negative_inf_count\":%" PRIu64
        ",\"before_zero_count\":%" PRIu64
        ",\"before_min\":%.17g,\"before_max\":%.17g,"
        "\"before_max_abs\":%.17g,\"after_stats_valid\":%s,"
        "\"after_finite_count\":%" PRIu64
        ",\"after_nan_count\":%" PRIu64
        ",\"after_positive_inf_count\":%" PRIu64
        ",\"after_negative_inf_count\":%" PRIu64
        ",\"after_zero_count\":%" PRIu64
        ",\"after_min\":%.17g,\"after_max\":%.17g,"
        "\"after_max_abs\":%.17g,"
        "\"result_status\":%d,\"pre_sync_status\":%d,"
        "\"host_malloc_status\":%d,\"d2h_status\":%d,"
        "\"device_malloc_status\":%d,\"h2d_status\":%d,"
        "\"run_status\":%d,\"post_sync_status\":%d,"
        "\"device_free_status\":%d,\"host_free_status\":%d,"
        "\"pre_sync_ns\":%" PRIu64 ",\"host_alloc_ns\":%" PRIu64
        ",\"d2h_ns\":%" PRIu64 ",\"scan_ns\":%" PRIu64
        ",\"device_alloc_ns\":%" PRIu64 ",\"h2d_ns\":%" PRIu64
        ",\"real_run_ns\":%" PRIu64 ",\"post_sync_ns\":%" PRIu64
        ",\"total_ns\":%" PRIu64 "}",
        attempt.scale_factor == 1.0
            ? "qmm_lossless_x2_scale_substitution"
            : "qmm_amplify_x2_scale_substitution",
        static_cast<long>(getpid()), static_cast<long>(syscall(SYS_gettid)),
        attempt.begin_ns, outcome, metadata.device, attempt.hit_index,
        attempt.candidate_index, cache, api_info.c_str(), attempt.scale_slot,
        scale.storage_bytes, scale.numel, attempt.scale_original,
        attempt.scale_temporary, attempt.scale_host,
        attempt.temporary_used ? "true" : "false",
        attempt.temporary_freed ? "true" : "false",
        attempt.host_freed ? "true" : "false", attempt.scale_factor,
        attempt.changed_elements, attempt.stats_valid ? "true" : "false",
        attempt.finite_count,
        attempt.nan_count, attempt.positive_inf_count,
        attempt.negative_inf_count, attempt.zero_count, attempt.min_value,
        attempt.max_value, attempt.max_abs,
        attempt.after_stats_valid ? "true" : "false",
        attempt.after_finite_count, attempt.after_nan_count,
        attempt.after_positive_inf_count, attempt.after_negative_inf_count,
        attempt.after_zero_count, attempt.after_min_value,
        attempt.after_max_value, attempt.after_max_abs, attempt.result_status,
        attempt.pre_sync_status, attempt.host_malloc_status,
        attempt.d2h_status, attempt.device_malloc_status, attempt.h2d_status,
        attempt.run_status, attempt.post_sync_status,
        attempt.device_free_status, attempt.host_free_status,
        attempt.pre_sync_ns, attempt.host_alloc_ns, attempt.d2h_ns,
        attempt.scan_ns, attempt.device_alloc_ns, attempt.h2d_ns,
        attempt.real_run_ns, attempt.post_sync_ns,
        monotonic_ns() - attempt.begin_ns);
    if (written > 0 && static_cast<size_t>(written) < sizeof(line)) {
        write_line(line, static_cast<size_t>(written));
    }
}

aclnnStatus run_scale_substitution_fallback(
    const char *outcome, op::internal::OpExecCache *cache, void *workspace,
    aclrtStream stream, const std::vector<void *> &addresses,
    Fault20Control *control, const RegisteredMetadata &metadata,
    const std::string &api_info, QmmScaleSubstitutionAttempt *attempt) {
    const uint64_t begin = monotonic_ns();
    const aclnnStatus status = run_original(cache, workspace, stream, addresses);
    attempt->real_run_ns = monotonic_ns() - begin;
    attempt->run_status = static_cast<int32_t>(status);
    __atomic_store_n(&control->enabled, 0U, __ATOMIC_RELEASE);
    set_device_last_status(control, metadata.device, attempt->result_status);
    log_qmm_x2_scale_substitution(outcome, cache, metadata, api_info, *attempt);
    return status;
}

aclnnStatus run_x2_scale_substitution(
    op::internal::OpExecCache *cache, void *workspace, aclrtStream stream,
    const std::vector<void *> &addresses, Fault20Control *control,
    const RegisteredMetadata &metadata, const std::string &api_info,
    uint32_t hit_index, uint32_t candidate_index, double scale_factor) {
    QmmScaleSubstitutionAttempt attempt{};
    attempt.begin_ns = monotonic_ns();
    attempt.hit_index = hit_index;
    attempt.candidate_index = candidate_index;
    attempt.scale_factor = scale_factor;
    const QmmStorageOrderProof proof = prove_qmm_storage_order(metadata, addresses);
    attempt.scale_slot = proof.slots[3];
    if (!proof.x2_scale_slot_proven || attempt.scale_slot < 0 ||
        static_cast<size_t>(attempt.scale_slot) >= addresses.size()) {
        attempt.result_status = kStatusScaleSlotUnproven;
        return run_scale_substitution_fallback(
            "scale_slot_unproven", cache, workspace, stream, addresses,
            control, metadata, api_info, &attempt);
    }
    attempt.scale_original = addresses[static_cast<size_t>(attempt.scale_slot)];

    std::vector<void *> addresses_copy;
    try {
        addresses_copy = addresses;
    } catch (...) {
        attempt.result_status = kStatusVectorCopyFailed;
        return run_scale_substitution_fallback(
            "address_vector_copy_failed", cache, workspace, stream, addresses,
            control, metadata, api_info, &attempt);
    }

    const size_t bytes =
        static_cast<size_t>(metadata.tensors.x2_scale.storage_bytes);
    uint64_t begin = monotonic_ns();
    attempt.pre_sync_status = g_synchronize_stream(stream);
    attempt.pre_sync_ns = monotonic_ns() - begin;
    if (attempt.pre_sync_status != ACL_SUCCESS) {
        attempt.result_status = kStatusPreSyncFailed;
        return run_scale_substitution_fallback(
            "pre_sync_failed", cache, workspace, stream, addresses, control,
            metadata, api_info, &attempt);
    }

    begin = monotonic_ns();
    attempt.host_malloc_status = g_malloc_host(&attempt.scale_host, bytes);
    attempt.host_alloc_ns = monotonic_ns() - begin;
    if (attempt.host_malloc_status != ACL_SUCCESS ||
        attempt.scale_host == nullptr) {
        free_scale_substitution_buffers(&attempt);
        attempt.result_status = kStatusHostAllocFailed;
        return run_scale_substitution_fallback(
            "host_alloc_failed", cache, workspace, stream, addresses, control,
            metadata, api_info, &attempt);
    }

    begin = monotonic_ns();
    attempt.d2h_status = g_memcpy(attempt.scale_host, bytes,
                                  attempt.scale_original, bytes,
                                  ACL_MEMCPY_DEVICE_TO_HOST);
    attempt.d2h_ns = monotonic_ns() - begin;
    if (attempt.d2h_status != ACL_SUCCESS) {
        free_scale_substitution_buffers(&attempt);
        attempt.result_status = kStatusD2HFailed;
        return run_scale_substitution_fallback(
            "d2h_failed", cache, workspace, stream, addresses, control,
            metadata, api_info, &attempt);
    }

    begin = monotonic_ns();
    scan_fp32_scale(attempt.scale_host,
                    static_cast<uint64_t>(metadata.tensors.x2_scale.numel),
                    &attempt);
    attempt.scan_ns = monotonic_ns() - begin;
    if (attempt.finite_count !=
            static_cast<uint64_t>(metadata.tensors.x2_scale.numel) ||
        attempt.nan_count != 0 || attempt.positive_inf_count != 0 ||
        attempt.negative_inf_count != 0) {
        free_scale_substitution_buffers(&attempt);
        attempt.result_status = kStatusScaleMetadataRejected;
        return run_scale_substitution_fallback(
            "preexisting_nonfinite_scale", cache, workspace, stream,
            addresses, control, metadata, api_info, &attempt);
    }

    if (scale_factor != 1.0) {
        amplify_fp32_scale(
            attempt.scale_host,
            static_cast<uint64_t>(metadata.tensors.x2_scale.numel),
            scale_factor, &attempt);
    }
    scan_fp32_scale_after(
        attempt.scale_host,
        static_cast<uint64_t>(metadata.tensors.x2_scale.numel), &attempt);
    attempt.scan_ns = monotonic_ns() - begin;
    if (attempt.after_finite_count !=
            static_cast<uint64_t>(metadata.tensors.x2_scale.numel) ||
        attempt.after_nan_count != 0 ||
        attempt.after_positive_inf_count != 0 ||
        attempt.after_negative_inf_count != 0) {
        free_scale_substitution_buffers(&attempt);
        attempt.result_status = kStatusScaleMetadataRejected;
        return run_scale_substitution_fallback(
            "amplified_scale_nonfinite", cache, workspace, stream, addresses,
            control, metadata, api_info, &attempt);
    }
    if (scale_factor != 1.0 && attempt.changed_elements == 0) {
        free_scale_substitution_buffers(&attempt);
        attempt.result_status = kStatusNoChangedElements;
        return run_scale_substitution_fallback(
            "no_changed_scale_elements", cache, workspace, stream, addresses,
            control, metadata, api_info, &attempt);
    }

    begin = monotonic_ns();
    attempt.device_malloc_status =
        g_malloc(&attempt.scale_temporary, bytes, ACL_MEM_MALLOC_HUGE_FIRST);
    attempt.device_alloc_ns = monotonic_ns() - begin;
    if (attempt.device_malloc_status != ACL_SUCCESS ||
        attempt.scale_temporary == nullptr) {
        free_scale_substitution_buffers(&attempt);
        attempt.result_status = kStatusDeviceAllocFailed;
        return run_scale_substitution_fallback(
            "device_alloc_failed", cache, workspace, stream, addresses,
            control, metadata, api_info, &attempt);
    }

    begin = monotonic_ns();
    attempt.h2d_status = g_memcpy_async(
        attempt.scale_temporary, bytes, attempt.scale_host, bytes,
        ACL_MEMCPY_HOST_TO_DEVICE, stream);
    attempt.h2d_ns = monotonic_ns() - begin;
    if (attempt.h2d_status != ACL_SUCCESS) {
        begin = monotonic_ns();
        attempt.post_sync_status = g_synchronize_stream(stream);
        attempt.post_sync_ns = monotonic_ns() - begin;
        if (attempt.post_sync_status == ACL_SUCCESS) {
            free_scale_substitution_buffers(&attempt);
        }
        attempt.result_status = kStatusH2DFailed;
        return run_scale_substitution_fallback(
            "h2d_failed", cache, workspace, stream, addresses, control,
            metadata, api_info, &attempt);
    }

    addresses_copy[static_cast<size_t>(attempt.scale_slot)] =
        attempt.scale_temporary;
    attempt.temporary_used = true;
    begin = monotonic_ns();
    const aclnnStatus run_status =
        run_original(cache, workspace, stream, addresses_copy);
    attempt.real_run_ns = monotonic_ns() - begin;
    attempt.run_status = static_cast<int32_t>(run_status);

    begin = monotonic_ns();
    attempt.post_sync_status = g_synchronize_stream(stream);
    attempt.post_sync_ns = monotonic_ns() - begin;
    if (attempt.post_sync_status == ACL_SUCCESS) {
        free_scale_substitution_buffers(&attempt);
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
    if (attempt.result_status != kStatusSuccess) {
        __atomic_store_n(&control->enabled, 0U, __ATOMIC_RELEASE);
    }
    set_device_last_status(control, metadata.device, attempt.result_status);
    log_qmm_x2_scale_substitution(outcome, cache, metadata, api_info, attempt);
    return run_status;
}

void observe_qmm_output_after_run(
    const char *source_mode, op::internal::OpExecCache *cache, aclrtStream stream,
    const std::vector<void *> &addresses, Fault20Control *control,
    const RegisteredMetadata &metadata, const std::string &api_info,
    uint32_t hit_index, uint32_t candidate_index, aclnnStatus run_status,
    bool stream_already_synchronized, uint64_t real_run_ns) {
    QmmOutputObservation observation{};
    observation.begin_ns = monotonic_ns();
    observation.hit_index = hit_index;
    observation.candidate_index = candidate_index;
    observation.run_status = static_cast<int32_t>(run_status);
    observation.real_run_ns = real_run_ns;
    const QmmStorageOrderProof proof = prove_qmm_storage_order(metadata, addresses);
    observation.out_slot = proof.slots[9];
    if (!proof.output_slot_proven || observation.out_slot < 0 ||
        static_cast<size_t>(observation.out_slot) >= addresses.size()) {
        observation.result_status = kStatusOutputSlotUnproven;
        set_device_last_status(control, metadata.device, observation.result_status);
        log_qmm_output_observation(source_mode, cache, metadata, api_info, observation);
        return;
    }
    observation.out_address = addresses[static_cast<size_t>(observation.out_slot)];
    const uint64_t max_output_bytes =
        __atomic_load_n(&control->max_output_bytes, __ATOMIC_RELAXED);
    if (run_status != ACL_SUCCESS || observation.out_address == nullptr ||
        !qmm_output_metadata_valid(metadata.tensors.out, max_output_bytes)) {
        observation.result_status = run_status == ACL_SUCCESS
                                        ? kStatusOutputMetadataRejected
                                        : kStatusRunFailed;
        set_device_last_status(control, metadata.device, observation.result_status);
        log_qmm_output_observation(source_mode, cache, metadata, api_info, observation);
        return;
    }
    if (stream_already_synchronized) {
        observation.sync_status = ACL_SUCCESS;
    } else {
        const uint64_t begin = monotonic_ns();
        observation.sync_status = g_synchronize_stream(stream);
        observation.sync_ns = monotonic_ns() - begin;
    }
    if (observation.sync_status != ACL_SUCCESS) {
        observation.result_status = kStatusPostSyncFailed;
        set_device_last_status(control, metadata.device, observation.result_status);
        log_qmm_output_observation(source_mode, cache, metadata, api_info, observation);
        return;
    }
    const size_t bytes = static_cast<size_t>(metadata.tensors.out.storage_bytes);
    uint64_t begin = monotonic_ns();
    observation.host_malloc_status = g_malloc_host(&observation.host_buffer, bytes);
    observation.host_alloc_ns = monotonic_ns() - begin;
    if (observation.host_malloc_status != ACL_SUCCESS ||
        observation.host_buffer == nullptr) {
        observation.result_status = kStatusHostAllocFailed;
        set_device_last_status(control, metadata.device, observation.result_status);
        log_qmm_output_observation(source_mode, cache, metadata, api_info, observation);
        return;
    }
    begin = monotonic_ns();
    observation.d2h_status =
        g_memcpy(observation.host_buffer, bytes, observation.out_address, bytes,
                 ACL_MEMCPY_DEVICE_TO_HOST);
    observation.d2h_ns = monotonic_ns() - begin;
    if (observation.d2h_status == ACL_SUCCESS) {
        begin = monotonic_ns();
        scan_qmm_output(observation.host_buffer, metadata.tensors.out, &observation);
        observation.scan_ns = monotonic_ns() - begin;
        const uint64_t nonfinite_count =
            observation.nan_count + observation.positive_inf_count +
            observation.negative_inf_count;
        if (nonfinite_count != 0) {
            __atomic_store_n(&control->enabled, 0U, __ATOMIC_RELEASE);
            observation.auto_disarmed_on_nonfinite = true;
        }
    }
    observation.host_free_status = g_free_host(observation.host_buffer);
    observation.host_freed = observation.host_free_status == ACL_SUCCESS;
    if (observation.d2h_status != ACL_SUCCESS) {
        observation.result_status = kStatusD2HFailed;
    } else if (!observation.stats_valid) {
        observation.result_status = kStatusOutputMetadataRejected;
    } else if (!observation.host_freed) {
        observation.result_status = kStatusHostFreeFailed;
    } else {
        observation.result_status = kStatusSuccess;
    }
    if (observation.result_status != kStatusSuccess) {
        __atomic_store_n(&control->enabled, 0U, __ATOMIC_RELEASE);
    }
    set_device_last_status(control, metadata.device, observation.result_status);
    log_qmm_output_observation(source_mode, cache, metadata, api_info, observation);
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
    attempt.extreme_value = __atomic_load_n(&control->extreme_value, __ATOMIC_RELAXED);
    attempt.extreme_ratio_bps =
        __atomic_load_n(&control->extreme_ratio_bps, __ATOMIC_RELAXED);
    attempt.extreme_pattern =
        __atomic_load_n(&control->extreme_pattern, __ATOMIC_RELAXED);
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

uint64_t fill_int8_positive_max(void *buffer, uint64_t bytes) {
    auto *values = static_cast<int8_t *>(buffer);
    constexpr int8_t kPositiveMax = std::numeric_limits<int8_t>::max();
    uint64_t changed = 0;
    for (uint64_t index = 0; index < bytes; ++index) {
        if (values[index] != kPositiveMax) {
            ++changed;
        }
        values[index] = kPositiveMax;
    }
    return changed;
}

void log_dual_extreme_attempt(const char *outcome,
                              op::internal::OpExecCache *cache,
                              const RegisteredMetadata &metadata,
                              const std::string &api_info,
                              const DualExtremeAttempt &attempt) {
    const auto &x2 = metadata.tensors.x2;
    char line[8192];
    const int written = std::snprintf(
        line, sizeof(line),
        "{\"event\":\"%s\",\"pid\":%ld,"
        "\"tid\":%ld,\"ts_ns\":%" PRIu64 ",\"outcome\":\"%s\","
        "\"device\":%d,\"hit\":%u,\"candidate\":%u,\"cache\":\"%p\","
        "\"api_info\":\"%s\",\"x1_slot\":%d,\"x2_slot\":%d,"
        "\"x1_bytes\":%" PRIu64 ",\"x2_bytes\":%" PRIu64
        ",\"x2_format\":%d,\"x2_view_shape\":[%" PRId64 ",%" PRId64
        "],\"x2_storage_shape\":[%" PRId64 ",%" PRId64 ",%" PRId64
        ",%" PRId64 "],\"x1_original\":\"%p\",\"x2_original\":\"%p\","
        "\"x1_temporary\":\"%p\",\"x2_temporary\":\"%p\","
        "\"x1_host\":\"%p\",\"x2_host\":\"%p\","
        "\"x1_temporary_used\":%s,\"x2_temporary_used\":%s,"
        "\"x1_temporary_freed\":%s,\"x2_temporary_freed\":%s,"
        "\"x1_host_freed\":%s,\"x2_host_freed\":%s,"
        "\"x1_changed_elements\":%" PRIu64 ",\"x2_changed_elements\":%" PRIu64
        ",\"result_status\":%d,\"pre_sync_status\":%d,"
        "\"x1_host_malloc_status\":%d,\"x2_host_malloc_status\":%d,"
        "\"x1_d2h_status\":%d,\"x2_d2h_status\":%d,"
        "\"x1_device_malloc_status\":%d,\"x2_device_malloc_status\":%d,"
        "\"x1_h2d_status\":%d,\"x2_h2d_status\":%d,\"run_status\":%d,"
        "\"post_sync_status\":%d,\"x1_device_free_status\":%d,"
        "\"x2_device_free_status\":%d,\"x1_host_free_status\":%d,"
        "\"x2_host_free_status\":%d,\"pre_sync_ns\":%" PRIu64
        ",\"host_alloc_ns\":%" PRIu64 ",\"d2h_ns\":%" PRIu64
        ",\"transform_ns\":%" PRIu64 ",\"device_alloc_ns\":%" PRIu64
        ",\"h2d_ns\":%" PRIu64 ",\"real_run_ns\":%" PRIu64
        ",\"post_sync_ns\":%" PRIu64 ",\"total_ns\":%" PRIu64 "}",
        attempt.extreme ? "qmm_extreme_x1_x2_substitution"
                        : "qmm_lossless_x1_x2_substitution",
        static_cast<long>(getpid()), static_cast<long>(syscall(SYS_gettid)),
        attempt.begin_ns, outcome, metadata.device, attempt.hit_index,
        attempt.candidate_index, cache, api_info.c_str(), attempt.x1_slot,
        attempt.x2_slot, metadata.x1.storage_bytes, x2.storage_bytes, x2.format,
        x2.view_shape[0], x2.view_shape[1], x2.storage_shape[0],
        x2.storage_shape[1], x2.storage_shape[2], x2.storage_shape[3],
        attempt.x1_original, attempt.x2_original, attempt.x1_temporary,
        attempt.x2_temporary, attempt.x1_host, attempt.x2_host,
        attempt.x1_temporary_used ? "true" : "false",
        attempt.x2_temporary_used ? "true" : "false",
        attempt.x1_temporary_freed ? "true" : "false",
        attempt.x2_temporary_freed ? "true" : "false",
        attempt.x1_host_freed ? "true" : "false",
        attempt.x2_host_freed ? "true" : "false", attempt.x1_changed_elements,
        attempt.x2_changed_elements, attempt.result_status,
        attempt.pre_sync_status, attempt.x1_host_malloc_status,
        attempt.x2_host_malloc_status, attempt.x1_d2h_status,
        attempt.x2_d2h_status, attempt.x1_device_malloc_status,
        attempt.x2_device_malloc_status, attempt.x1_h2d_status,
        attempt.x2_h2d_status, attempt.run_status, attempt.post_sync_status,
        attempt.x1_device_free_status, attempt.x2_device_free_status,
        attempt.x1_host_free_status, attempt.x2_host_free_status,
        attempt.pre_sync_ns, attempt.host_alloc_ns, attempt.d2h_ns,
        attempt.transform_ns, attempt.device_alloc_ns, attempt.h2d_ns,
        attempt.real_run_ns, attempt.post_sync_ns,
        monotonic_ns() - attempt.begin_ns);
    if (written > 0 && static_cast<size_t>(written) < sizeof(line)) {
        write_line(line, static_cast<size_t>(written));
    }
}

void free_dual_extreme_buffers(DualExtremeAttempt *attempt) {
    if (attempt->x1_temporary != nullptr) {
        attempt->x1_device_free_status = g_free(attempt->x1_temporary);
        attempt->x1_temporary_freed = attempt->x1_device_free_status == ACL_SUCCESS;
    }
    if (attempt->x2_temporary != nullptr) {
        attempt->x2_device_free_status = g_free(attempt->x2_temporary);
        attempt->x2_temporary_freed = attempt->x2_device_free_status == ACL_SUCCESS;
    }
    if (attempt->x1_host != nullptr) {
        attempt->x1_host_free_status = g_free_host(attempt->x1_host);
        attempt->x1_host_freed = attempt->x1_host_free_status == ACL_SUCCESS;
    }
    if (attempt->x2_host != nullptr) {
        attempt->x2_host_free_status = g_free_host(attempt->x2_host);
        attempt->x2_host_freed = attempt->x2_host_free_status == ACL_SUCCESS;
    }
}

aclnnStatus run_dual_extreme_fallback(
    const char *outcome, op::internal::OpExecCache *cache, void *workspace,
    aclrtStream stream, const std::vector<void *> &addresses,
    Fault20Control *control, const RegisteredMetadata &metadata,
    const std::string &api_info, DualExtremeAttempt *attempt) {
    const uint64_t begin = monotonic_ns();
    const aclnnStatus status = run_original(cache, workspace, stream, addresses);
    attempt->real_run_ns = monotonic_ns() - begin;
    attempt->run_status = static_cast<int32_t>(status);
    __atomic_store_n(&control->enabled, 0U, __ATOMIC_RELEASE);
    set_device_last_status(control, metadata.device, attempt->result_status);
    log_dual_extreme_attempt(outcome, cache, metadata, api_info, *attempt);
    return status;
}

aclnnStatus run_dual_x1_x2(
    op::internal::OpExecCache *cache, void *workspace, aclrtStream stream,
    const std::vector<void *> &addresses, Fault20Control *control,
    const RegisteredMetadata &metadata, const std::string &api_info,
    uint32_t hit_index, uint32_t candidate_index, bool extreme) {
    DualExtremeAttempt attempt{};
    attempt.begin_ns = monotonic_ns();
    attempt.hit_index = hit_index;
    attempt.candidate_index = candidate_index;
    attempt.extreme = extreme;
    const QmmStorageOrderProof proof = prove_qmm_storage_order(metadata, addresses);
    attempt.x1_slot = proof.slots[0];
    attempt.x2_slot = proof.slots[1];
    if (attempt.x1_slot != 0 || !proof.weight_slot_proven ||
        attempt.x2_slot < 0 ||
        static_cast<size_t>(attempt.x2_slot) >= addresses.size()) {
        attempt.result_status = kStatusWeightSlotUnproven;
        return run_dual_extreme_fallback(
            "operand_slot_unproven", cache, workspace, stream, addresses,
            control, metadata, api_info, &attempt);
    }
    attempt.x1_original = addresses[static_cast<size_t>(attempt.x1_slot)];
    attempt.x2_original = addresses[static_cast<size_t>(attempt.x2_slot)];

    std::vector<void *> addresses_copy;
    try {
        addresses_copy = addresses;
    } catch (...) {
        attempt.result_status = kStatusVectorCopyFailed;
        return run_dual_extreme_fallback(
            "address_vector_copy_failed", cache, workspace, stream, addresses,
            control, metadata, api_info, &attempt);
    }

    uint64_t begin = monotonic_ns();
    attempt.pre_sync_status = g_synchronize_stream(stream);
    attempt.pre_sync_ns = monotonic_ns() - begin;
    if (attempt.pre_sync_status != ACL_SUCCESS) {
        attempt.result_status = kStatusPreSyncFailed;
        return run_dual_extreme_fallback(
            "pre_sync_failed", cache, workspace, stream, addresses, control,
            metadata, api_info, &attempt);
    }

    begin = monotonic_ns();
    attempt.x1_host_malloc_status =
        g_malloc_host(&attempt.x1_host,
                      static_cast<size_t>(metadata.x1.storage_bytes));
    if (attempt.x1_host_malloc_status == ACL_SUCCESS) {
        attempt.x2_host_malloc_status =
            g_malloc_host(&attempt.x2_host,
                          static_cast<size_t>(metadata.tensors.x2.storage_bytes));
    }
    attempt.host_alloc_ns = monotonic_ns() - begin;
    if (attempt.x1_host_malloc_status != ACL_SUCCESS ||
        attempt.x2_host_malloc_status != ACL_SUCCESS || attempt.x1_host == nullptr ||
        attempt.x2_host == nullptr) {
        free_dual_extreme_buffers(&attempt);
        attempt.result_status = kStatusHostAllocFailed;
        return run_dual_extreme_fallback(
            "host_alloc_failed", cache, workspace, stream, addresses, control,
            metadata, api_info, &attempt);
    }

    begin = monotonic_ns();
    attempt.x1_d2h_status = g_memcpy(
        attempt.x1_host, static_cast<size_t>(metadata.x1.storage_bytes),
        attempt.x1_original, static_cast<size_t>(metadata.x1.storage_bytes),
        ACL_MEMCPY_DEVICE_TO_HOST);
    if (attempt.x1_d2h_status == ACL_SUCCESS) {
        attempt.x2_d2h_status = g_memcpy(
            attempt.x2_host,
            static_cast<size_t>(metadata.tensors.x2.storage_bytes),
            attempt.x2_original,
            static_cast<size_t>(metadata.tensors.x2.storage_bytes),
            ACL_MEMCPY_DEVICE_TO_HOST);
    }
    attempt.d2h_ns = monotonic_ns() - begin;
    if (attempt.x1_d2h_status != ACL_SUCCESS ||
        attempt.x2_d2h_status != ACL_SUCCESS) {
        free_dual_extreme_buffers(&attempt);
        attempt.result_status = kStatusD2HFailed;
        return run_dual_extreme_fallback(
            "d2h_failed", cache, workspace, stream, addresses, control,
            metadata, api_info, &attempt);
    }

    if (extreme) {
        begin = monotonic_ns();
        attempt.x1_changed_elements =
            fill_int8_positive_max(attempt.x1_host, metadata.x1.storage_bytes);
        attempt.x2_changed_elements = fill_int8_positive_max(
            attempt.x2_host, metadata.tensors.x2.storage_bytes);
        attempt.transform_ns = monotonic_ns() - begin;
        if (attempt.x1_changed_elements == 0 || attempt.x2_changed_elements == 0) {
            free_dual_extreme_buffers(&attempt);
            attempt.result_status = kStatusNoChangedElements;
            return run_dual_extreme_fallback(
                "no_changed_elements", cache, workspace, stream, addresses,
                control, metadata, api_info, &attempt);
        }
    }

    begin = monotonic_ns();
    attempt.x1_device_malloc_status = g_malloc(
        &attempt.x1_temporary, static_cast<size_t>(metadata.x1.storage_bytes),
        ACL_MEM_MALLOC_HUGE_FIRST);
    if (attempt.x1_device_malloc_status == ACL_SUCCESS) {
        attempt.x2_device_malloc_status = g_malloc(
            &attempt.x2_temporary,
            static_cast<size_t>(metadata.tensors.x2.storage_bytes),
            ACL_MEM_MALLOC_HUGE_FIRST);
    }
    attempt.device_alloc_ns = monotonic_ns() - begin;
    if (attempt.x1_device_malloc_status != ACL_SUCCESS ||
        attempt.x2_device_malloc_status != ACL_SUCCESS ||
        attempt.x1_temporary == nullptr || attempt.x2_temporary == nullptr) {
        free_dual_extreme_buffers(&attempt);
        attempt.result_status = kStatusDeviceAllocFailed;
        return run_dual_extreme_fallback(
            "device_alloc_failed", cache, workspace, stream, addresses, control,
            metadata, api_info, &attempt);
    }

    begin = monotonic_ns();
    attempt.x1_h2d_status = g_memcpy_async(
        attempt.x1_temporary, static_cast<size_t>(metadata.x1.storage_bytes),
        attempt.x1_host, static_cast<size_t>(metadata.x1.storage_bytes),
        ACL_MEMCPY_HOST_TO_DEVICE, stream);
    if (attempt.x1_h2d_status == ACL_SUCCESS) {
        attempt.x2_h2d_status = g_memcpy_async(
            attempt.x2_temporary,
            static_cast<size_t>(metadata.tensors.x2.storage_bytes),
            attempt.x2_host,
            static_cast<size_t>(metadata.tensors.x2.storage_bytes),
            ACL_MEMCPY_HOST_TO_DEVICE, stream);
    }
    attempt.h2d_ns = monotonic_ns() - begin;
    if (attempt.x1_h2d_status != ACL_SUCCESS ||
        attempt.x2_h2d_status != ACL_SUCCESS) {
        begin = monotonic_ns();
        attempt.post_sync_status = g_synchronize_stream(stream);
        attempt.post_sync_ns = monotonic_ns() - begin;
        if (attempt.post_sync_status == ACL_SUCCESS) {
            free_dual_extreme_buffers(&attempt);
        }
        attempt.result_status = kStatusH2DFailed;
        return run_dual_extreme_fallback(
            "h2d_failed", cache, workspace, stream, addresses, control,
            metadata, api_info, &attempt);
    }

    addresses_copy[static_cast<size_t>(attempt.x1_slot)] = attempt.x1_temporary;
    addresses_copy[static_cast<size_t>(attempt.x2_slot)] = attempt.x2_temporary;
    attempt.x1_temporary_used = true;
    attempt.x2_temporary_used = true;
    begin = monotonic_ns();
    const aclnnStatus run_status =
        run_original(cache, workspace, stream, addresses_copy);
    attempt.real_run_ns = monotonic_ns() - begin;
    attempt.run_status = static_cast<int32_t>(run_status);

    begin = monotonic_ns();
    attempt.post_sync_status = g_synchronize_stream(stream);
    attempt.post_sync_ns = monotonic_ns() - begin;
    if (attempt.post_sync_status == ACL_SUCCESS) {
        free_dual_extreme_buffers(&attempt);
    }

    const char *outcome = "success";
    if (attempt.run_status != ACL_SUCCESS) {
        attempt.result_status = kStatusRunFailed;
        outcome = "real_run_failed";
    } else if (attempt.post_sync_status != ACL_SUCCESS) {
        attempt.result_status = kStatusPostSyncFailed;
        outcome = "post_sync_failed_buffers_retained";
    } else if (attempt.x1_device_free_status != ACL_SUCCESS ||
               attempt.x2_device_free_status != ACL_SUCCESS) {
        attempt.result_status = kStatusDeviceFreeFailed;
        outcome = "device_free_failed";
    } else if (attempt.x1_host_free_status != ACL_SUCCESS ||
               attempt.x2_host_free_status != ACL_SUCCESS) {
        attempt.result_status = kStatusHostFreeFailed;
        outcome = "host_free_failed";
    } else {
        attempt.result_status = kStatusSuccess;
    }
    if (attempt.result_status != kStatusSuccess) {
        __atomic_store_n(&control->enabled, 0U, __ATOMIC_RELEASE);
    }
    set_device_last_status(control, metadata.device, attempt.result_status);
    log_dual_extreme_attempt(outcome, cache, metadata, api_info, attempt);
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
    log_op_exec_api_inventory(cache, api_info, addresses);
    if (starts_with(api_info, "aclnnAscendQuantV3_")) {
        g_last_quant.valid = !addresses.empty() && addresses.back() != nullptr;
        g_last_quant.output = g_last_quant.valid ? addresses.back() : nullptr;
        return run_original(cache, workspace, stream, addresses);
    }
    if (starts_with(api_info, "aclnnMatmul_")) {
        g_last_quant.valid = false;
        g_last_quant.output = nullptr;
        RegisteredMatmulMetadata metadata{};
        uint64_t run_index = 0;
        if (!g_abi_ok || !g_developer_mode_ok ||
            !lookup_matmul_metadata(cache, MatmulApiKind::kFloat, &metadata,
                                    &run_index)) {
            return run_original(cache, workspace, stream, addresses);
        }
        const bool exact_target = is_target_float_matmul_slot_proof(metadata);
        if (run_index == 1 && exact_target) {
            log_float_matmul_run_slot_proof(cache, metadata, api_info, addresses,
                                            run_index);
        }
        if (!exact_target || g_control == nullptr ||
            g_matmul_substitution_active ||
            __atomic_load_n(&g_control->enabled, __ATOMIC_ACQUIRE) == 0) {
            return run_original(cache, workspace, stream, addresses);
        }
        Fault20Control *control = g_control;
        resolve_runtime_api();
        const uint32_t mode =
            __atomic_load_n(&control->mode, __ATOMIC_RELAXED);
        const bool lossless_mode =
            mode == FAULT20_MODE_FLOAT_MATMUL_LOSSLESS_SELF ||
            mode == FAULT20_MODE_FLOAT_MATMUL_LOSSLESS_MAT2 ||
            mode == FAULT20_MODE_FLOAT_MATMUL_LOSSLESS_DUAL;
        const bool observer_mode =
            mode == FAULT20_MODE_FLOAT_MATMUL_OUTPUT_OBSERVE;
        const bool finite_mode =
            mode == FAULT20_MODE_FLOAT_MATMUL_FINITE_DUAL;
        const bool cancellation_mode =
            mode == FAULT20_MODE_FLOAT_MATMUL_CANCELLATION;
        const bool output_proxy_mode =
            mode == FAULT20_MODE_FLOAT_MATMUL_OUTPUT_PROXY;
        const int32_t finite_exponent =
            __atomic_load_n(&control->extreme_value, __ATOMIC_RELAXED);
        const uint32_t cancellation_pattern =
            __atomic_load_n(&control->extreme_pattern, __ATOMIC_RELAXED);
        const uint32_t cancellation_block_size =
            __atomic_load_n(&control->quant_step, __ATOMIC_RELAXED);
        const uint32_t output_proxy_kind = cancellation_pattern;
        const uint32_t output_proxy_elements = cancellation_block_size;
        const MatmulStorageOrderProof proof =
            prove_matmul_storage_order(metadata, addresses);
        const uint32_t target_call_index = __atomic_load_n(
            &control->target_call_index, __ATOMIC_RELAXED);
        const bool runtime_ready =
            g_malloc != nullptr && g_free != nullptr &&
            g_memcpy_async != nullptr && g_synchronize_stream != nullptr;
        const bool observer_runtime_ready =
            g_malloc_host != nullptr && g_free_host != nullptr &&
            g_memcpy != nullptr && g_synchronize_stream != nullptr;
        if ((!lossless_mode && !observer_mode && !finite_mode &&
             !cancellation_mode && !output_proxy_mode) ||
            (lossless_mode && !runtime_ready) ||
            (observer_mode && !observer_runtime_ready) ||
            (finite_mode && (!runtime_ready || !observer_runtime_ready ||
                             !valid_float_matmul_exponent(finite_exponent))) ||
            (cancellation_mode &&
             (!runtime_ready || !observer_runtime_ready ||
              finite_exponent != 58 ||
              !valid_float_matmul_cancellation_config(
                  cancellation_pattern, cancellation_block_size) ||
              !float_matmul_cancellation_metadata_valid(metadata.tensors))) ||
            (output_proxy_mode &&
             (!observer_runtime_ready ||
              !valid_float_matmul_output_proxy_config(
                  output_proxy_kind, output_proxy_elements) ||
              !qmm_output_metadata_valid(metadata.tensors.out,
                                         control->max_output_bytes))) ||
            control->target_api != FAULT20_TARGET_API_FLOAT_MATMUL ||
            control->target_device != 0 || control->target_m != 1 ||
            control->target_k != 12800 || control->target_x2_n != 5120 ||
            control->target_dtype != ACL_BF16 || control->max_bytes != 25600 ||
            control->max_weight_bytes != 131072000 ||
            control->max_output_bytes != 10240 ||
            control->target_device_mask != 1 ||
            control->max_hits_per_device != 1 ||
            control->selection_mode != FAULT20_SELECTION_BOUNDED ||
            target_call_index == 0 || !proof.self_slot_proven ||
            !proof.mat2_slot_proven || !proof.output_slot_proven ||
            addresses.size() != 3 || addresses[0] == nullptr ||
            addresses[1] == nullptr || addresses[2] == nullptr) {
            return run_original(cache, workspace, stream, addresses);
        }
        const uint32_t candidate_index = __atomic_add_fetch(
            &control->candidate_count_per_device[metadata.device], 1U,
            __ATOMIC_ACQ_REL);
        __atomic_add_fetch(&control->candidate_count, 1U, __ATOMIC_ACQ_REL);
        if (candidate_index != target_call_index) {
            return run_original(cache, workspace, stream, addresses);
        }
        uint32_t hit_index = 0;
        if (!reserve_hit_per_device(control, metadata.device, &hit_index)) {
            return run_original(cache, workspace, stream, addresses);
        }
        if (observer_mode) {
            const uint64_t run_begin_ns = monotonic_ns();
            const aclnnStatus run_status =
                run_original(cache, workspace, stream, addresses);
            const uint64_t real_run_ns = monotonic_ns() - run_begin_ns;
            observe_float_matmul_output_after_run(
                cache, stream, addresses, control, metadata, api_info,
                hit_index, candidate_index, run_status, real_run_ns);
            return run_status;
        }
        if (output_proxy_mode) {
            g_matmul_substitution_active = true;
            const aclnnStatus status = run_float_matmul_output_proxy(
                cache, workspace, stream, addresses, control, metadata,
                api_info, hit_index, candidate_index, output_proxy_kind,
                output_proxy_elements);
            g_matmul_substitution_active = false;
            return status;
        }
        if (finite_mode || cancellation_mode) {
            g_matmul_substitution_active = true;
            const aclnnStatus status = run_float_matmul_finite_dual(
                cache, workspace, stream, addresses, control, metadata,
                api_info, hit_index, candidate_index, mode, finite_exponent,
                cancellation_pattern, cancellation_block_size);
            g_matmul_substitution_active = false;
            return status;
        }
        g_matmul_substitution_active = true;
        const aclnnStatus status = run_float_matmul_lossless(
            cache, workspace, stream, addresses, control, metadata, api_info,
            mode, hit_index, candidate_index);
        g_matmul_substitution_active = false;
        return status;
    }
    if (starts_with(api_info, "aclnnMatmulWeightNz_")) {
        g_last_quant.valid = false;
        g_last_quant.output = nullptr;
        RegisteredMatmulMetadata metadata{};
        uint64_t run_index = 0;
        if (!g_abi_ok || !g_developer_mode_ok ||
            !lookup_matmul_metadata(cache, MatmulApiKind::kWeightNz, &metadata,
                                    &run_index)) {
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
    uint64_t qmm_run_index = 0;
    if (!g_abi_ok || !g_developer_mode_ok || g_control == nullptr ||
        !lookup_metadata(cache, &metadata, &qmm_run_index) ||
        addresses.empty() || addresses[0] == nullptr || !producer_matches) {
        return run_original(cache, workspace, stream, addresses);
    }
    (void)log_qmm_run_trace(cache, metadata, api_info, addresses, qmm_run_index);

    Fault20Control *control = g_control;
    resolve_runtime_api();
    if (__atomic_load_n(&control->enabled, __ATOMIC_ACQUIRE) == 0) {
        return run_original(cache, workspace, stream, addresses);
    }
    const uint32_t mode = __atomic_load_n(&control->mode, __ATOMIC_RELAXED);
    if (mode != FAULT20_MODE_LOSSLESS_D2D && mode != FAULT20_MODE_INT4_EQUIVALENT &&
        mode != FAULT20_MODE_INT2_EQUIVALENT &&
        mode != FAULT20_MODE_QMM_EXTREME_X1 &&
        mode != FAULT20_MODE_QMM_OUTPUT_OBSERVE &&
        mode != FAULT20_MODE_QMM_WEIGHT_OBSERVE &&
        mode != FAULT20_MODE_QMM_X2_SCALE_OBSERVE &&
        mode != FAULT20_MODE_QMM_X2_SCALE_LOSSLESS &&
        mode != FAULT20_MODE_QMM_X2_SCALE_AMPLIFY &&
        mode != FAULT20_MODE_QMM_EXTREME_X1_X2 &&
        mode != FAULT20_MODE_QMM_LOSSLESS_X1_X2) {
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
    const uint64_t max_output_bytes =
        __atomic_load_n(&control->max_output_bytes, __ATOMIC_RELAXED);
    const uint64_t max_weight_bytes =
        __atomic_load_n(&control->max_weight_bytes, __ATOMIC_RELAXED);
    const int32_t target_x2_n =
        __atomic_load_n(&control->target_x2_n, __ATOMIC_RELAXED);
    const uint64_t max_scale_bytes =
        __atomic_load_n(&control->max_scale_bytes, __ATOMIC_RELAXED);
    const int32_t target_scale_elements =
        __atomic_load_n(&control->target_scale_elements, __ATOMIC_RELAXED);
    // The controller writes all configuration fields before the release-store
    // that enables the control block. This Run path has already observed that
    // flag with an acquire-load, so the immutable per-generation factor can be
    // read normally here. GCC's scalar __atomic_load_n does not accept double
    // on this aarch64 toolchain.
    const double scale_factor = control->scale_factor;
    const int64_t expected_numel = static_cast<int64_t>(target_m) * target_k;
    const bool common_runtime_ready = g_malloc != nullptr && g_free != nullptr &&
                                      g_memcpy_async != nullptr &&
                                      g_synchronize_stream != nullptr;
    const bool numeric_runtime_ready = common_runtime_ready && g_malloc_host != nullptr &&
                                       g_free_host != nullptr && g_memcpy != nullptr;
    const uint32_t quant_step =
        __atomic_load_n(&control->quant_step, __ATOMIC_RELAXED);
    const int32_t extreme_value =
        __atomic_load_n(&control->extreme_value, __ATOMIC_RELAXED);
    const uint32_t extreme_ratio_bps =
        __atomic_load_n(&control->extreme_ratio_bps, __ATOMIC_RELAXED);
    const uint32_t extreme_pattern =
        __atomic_load_n(&control->extreme_pattern, __ATOMIC_RELAXED);
    const bool target_device =
        metadata.device >= 0 && static_cast<size_t>(metadata.device) < FAULT20_MAX_DEVICES &&
        (target_device_mask & (1ULL << static_cast<uint32_t>(metadata.device))) != 0;
    const QmmStorageOrderProof qmm_proof =
        prove_qmm_storage_order(metadata, addresses);
    const bool output_observer_ready =
        qmm_proof.output_slot_proven && numeric_runtime_ready &&
        qmm_output_metadata_valid(metadata.tensors.out, max_output_bytes);
    const bool weight_observer_ready =
        qmm_proof.weight_slot_proven &&
        qmm_weight_metadata_valid(metadata.tensors.x2, target_k,
                                  max_weight_bytes);
    const bool scale_observer_ready =
        qmm_proof.x2_scale_slot_proven &&
        qmm_x2_scale_metadata_valid(metadata.tensors.x2_scale,
                                    target_scale_elements,
                                    max_scale_bytes);
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
          mode == FAULT20_MODE_INT2_EQUIVALENT ||
          mode == FAULT20_MODE_QMM_EXTREME_X1 ||
          mode == FAULT20_MODE_QMM_OUTPUT_OBSERVE ||
          mode == FAULT20_MODE_QMM_X2_SCALE_LOSSLESS ||
          mode == FAULT20_MODE_QMM_X2_SCALE_AMPLIFY ||
          mode == FAULT20_MODE_QMM_EXTREME_X1_X2 ||
          mode == FAULT20_MODE_QMM_LOSSLESS_X1_X2) &&
         (!numeric_runtime_ready ||
          (mode == FAULT20_MODE_INT4_EQUIVALENT && quant_step != 16) ||
          (mode == FAULT20_MODE_INT2_EQUIVALENT && quant_step != 64) ||
          (mode == FAULT20_MODE_QMM_EXTREME_X1 &&
           (!fault20_valid_extreme_configuration(extreme_value, extreme_ratio_bps,
                                                 extreme_pattern) ||
            (selection_mode == FAULT20_SELECTION_BOUNDED &&
             max_hits_per_device > 64) ||
            (selection_mode == FAULT20_SELECTION_ALL_HITS &&
             max_hits_per_device != FAULT20_ALL_HITS_LIMIT))) ||
          (mode == FAULT20_MODE_QMM_OUTPUT_OBSERVE &&
           (selection_mode != FAULT20_SELECTION_BOUNDED ||
            max_hits_per_device != 1)) ||
           ((mode == FAULT20_MODE_QMM_EXTREME_X1 ||
             mode == FAULT20_MODE_QMM_OUTPUT_OBSERVE ||
             mode == FAULT20_MODE_QMM_X2_SCALE_LOSSLESS ||
             mode == FAULT20_MODE_QMM_X2_SCALE_AMPLIFY ||
             mode == FAULT20_MODE_QMM_EXTREME_X1_X2 ||
             mode == FAULT20_MODE_QMM_LOSSLESS_X1_X2) &&
            !output_observer_ready))) ||
        (mode == FAULT20_MODE_QMM_WEIGHT_OBSERVE &&
         (selection_mode != FAULT20_SELECTION_BOUNDED ||
          max_hits_per_device != 1 || !weight_observer_ready)) ||
        ((mode == FAULT20_MODE_QMM_X2_SCALE_OBSERVE ||
          mode == FAULT20_MODE_QMM_X2_SCALE_LOSSLESS ||
          mode == FAULT20_MODE_QMM_X2_SCALE_AMPLIFY) &&
         (selection_mode != FAULT20_SELECTION_BOUNDED ||
          max_hits_per_device != 1 || !scale_observer_ready)) ||
        (mode == FAULT20_MODE_QMM_X2_SCALE_LOSSLESS &&
         scale_factor != 1.0) ||
        (mode == FAULT20_MODE_QMM_X2_SCALE_AMPLIFY &&
         (scale_factor != 2.0 && scale_factor != 10.0 &&
          scale_factor != 100.0 && scale_factor != 1000.0)) ||
        ((mode == FAULT20_MODE_QMM_EXTREME_X1_X2 ||
          mode == FAULT20_MODE_QMM_LOSSLESS_X1_X2) &&
         (selection_mode != FAULT20_SELECTION_BOUNDED ||
          max_hits_per_device != 1 || !weight_observer_ready ||
          target_x2_n <= 0 || metadata.tensors.x2.view_shape[1] != target_x2_n ||
          metadata.tensors.x2.storage_bytes != max_weight_bytes ||
          metadata.x1.storage_bytes != max_bytes)) ||
        (mode == FAULT20_MODE_QMM_EXTREME_X1_X2 &&
         (extreme_value != 127 || extreme_ratio_bps != 10000 ||
          extreme_pattern != FAULT20_EXTREME_POSITIVE_MAX))) {
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

    if (mode == FAULT20_MODE_QMM_OUTPUT_OBSERVE) {
        const uint64_t run_begin_ns = monotonic_ns();
        const aclnnStatus run_status = run_original(cache, workspace, stream, addresses);
        const uint64_t real_run_ns = monotonic_ns() - run_begin_ns;
        observe_qmm_output_after_run(
            "baseline_observer", cache, stream, addresses, control, metadata,
            api_info, hit_index, candidate_index, run_status, false, real_run_ns);
        return run_status;
    }

    if (mode == FAULT20_MODE_QMM_WEIGHT_OBSERVE) {
        return observe_qmm_weight_and_run(
            cache, workspace, stream, addresses, control, metadata, api_info,
            hit_index, candidate_index);
    }

    if (mode == FAULT20_MODE_QMM_X2_SCALE_OBSERVE) {
        return observe_qmm_x2_scale_and_run(
            cache, workspace, stream, addresses, control, metadata, api_info,
            hit_index, candidate_index);
    }

    if (mode == FAULT20_MODE_QMM_X2_SCALE_LOSSLESS) {
        const aclnnStatus status = run_x2_scale_substitution(
            cache, workspace, stream, addresses, control, metadata, api_info,
            hit_index, candidate_index, 1.0);
        const int32_t substitution_status =
            __atomic_load_n(&control->last_status_per_device[metadata.device],
                            __ATOMIC_ACQUIRE);
        if (substitution_status == kStatusSuccess) {
            observe_qmm_output_after_run(
                "lossless_x2_scale", cache, stream, addresses, control,
                metadata, api_info, hit_index, candidate_index, status, true, 0);
        } else {
            __atomic_store_n(&control->enabled, 0U, __ATOMIC_RELEASE);
        }
        return status;
    }

    if (mode == FAULT20_MODE_QMM_X2_SCALE_AMPLIFY) {
        const aclnnStatus status = run_x2_scale_substitution(
            cache, workspace, stream, addresses, control, metadata, api_info,
            hit_index, candidate_index, scale_factor);
        const int32_t substitution_status =
            __atomic_load_n(&control->last_status_per_device[metadata.device],
                            __ATOMIC_ACQUIRE);
        if (substitution_status == kStatusSuccess) {
            observe_qmm_output_after_run(
                "amplify_x2_scale", cache, stream, addresses, control,
                metadata, api_info, hit_index, candidate_index, status, true, 0);
        } else {
            __atomic_store_n(&control->enabled, 0U, __ATOMIC_RELEASE);
        }
        return status;
    }

    if (mode == FAULT20_MODE_QMM_EXTREME_X1_X2 ||
        mode == FAULT20_MODE_QMM_LOSSLESS_X1_X2) {
        const bool extreme = mode == FAULT20_MODE_QMM_EXTREME_X1_X2;
        const aclnnStatus status = run_dual_x1_x2(
            cache, workspace, stream, addresses, control, metadata, api_info,
            hit_index, candidate_index, extreme);
        const int32_t injection_status =
            __atomic_load_n(&control->last_status_per_device[metadata.device],
                            __ATOMIC_ACQUIRE);
        if (injection_status == kStatusSuccess) {
            observe_qmm_output_after_run(
                extreme ? "extreme_x1_x2" : "lossless_x1_x2", cache, stream,
                addresses, control, metadata,
                api_info, hit_index, candidate_index, status, true, 0);
        } else {
            __atomic_store_n(&control->enabled, 0U, __ATOMIC_RELEASE);
        }
        return status;
    }

    if (mode == FAULT20_MODE_INT4_EQUIVALENT ||
        mode == FAULT20_MODE_INT2_EQUIVALENT ||
        mode == FAULT20_MODE_QMM_EXTREME_X1) {
        const aclnnStatus status = run_numeric_equivalent(
            cache, workspace, stream, addresses, control, metadata, api_info, mode,
            hit_index, candidate_index);
        const int32_t injection_status =
            __atomic_load_n(&control->last_status_per_device[metadata.device],
                            __ATOMIC_ACQUIRE);
        if (mode == FAULT20_MODE_QMM_EXTREME_X1 &&
            injection_status == kStatusSuccess) {
            const char *source_mode =
                selection_mode == FAULT20_SELECTION_ALL_HITS
                    ? "all_hit_extreme_x1"
                    : (max_hits_per_device > 1 ? "progressive_extreme_x1"
                                               : "extreme_x1");
            observe_qmm_output_after_run(
                source_mode, cache, stream, addresses, control, metadata,
                api_info, hit_index, candidate_index, status, true, 0);
        } else if (mode == FAULT20_MODE_QMM_EXTREME_X1) {
            __atomic_store_n(&control->enabled, 0U, __ATOMIC_RELEASE);
        }
        return status;
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

extern "C" bool fault20_qmm_gws_commit(const Fault20QmmMetadata *metadata) {
    pthread_once(&g_resolve_once, resolve_real);
    resolve_runtime_api();
    const uint32_t add_count = g_capture.add_count;
    op::internal::OpExecCache *cache = g_capture.cache;
    const CaptureKind kind = g_capture.kind;
    const uint32_t tensor_param_count = g_capture.tensor_param_count;
    uintptr_t tensor_params[16]{};
    std::memcpy(tensor_params, g_capture.tensor_params, sizeof(tensor_params));
    g_capture = CaptureState{};
    const bool registration_ok =
        kind == CaptureKind::kQmm && metadata != nullptr && add_count == 1 &&
        cache != nullptr && metadata->x1.present != 0 && metadata->out.present != 0 &&
        metadata->x1_legacy.storage_bytes != 0;
    log_qmm_param_collection(cache, metadata, tensor_param_count, tensor_params,
                             registration_ok);
    if (!registration_ok) {
        log_registration("qmm_cache_registration_rejected", cache,
                         metadata == nullptr ? nullptr : &metadata->x1_legacy, -1,
                         add_count);
        return false;
    }
    int32_t device = -1;
    if (g_get_device == nullptr || g_get_device(&device) != ACL_SUCCESS) {
        log_registration("qmm_cache_registration_rejected", cache,
                         &metadata->x1_legacy, device, add_count);
        return false;
    }
    RegisteredMetadata registered{};
    registered.x1 = metadata->x1_legacy;
    registered.tensors = *metadata;
    registered.device = device;
    registered.tensor_param_count = tensor_param_count;
    std::memcpy(registered.tensor_params, tensor_params, sizeof(tensor_params));
    {
        std::lock_guard<std::mutex> lock(registry_mutex());
        registry()[cache] = registered;
    }
    log_registration("qmm_cache_registered", cache, &metadata->x1_legacy, device,
                     add_count);
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
    log_matmul_param_collection(MatmulApiKind::kWeightNz, cache, metadata,
                                tensor_param_count, tensor_params, registration_ok);
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
    registered.api_kind = static_cast<uint32_t>(MatmulApiKind::kWeightNz);
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

extern "C" void fault20_float_matmul_gws_begin() {
    g_capture = CaptureState{};
    g_capture.active = true;
    g_capture.kind = CaptureKind::kFloatMatmul;
    g_capture.add_count = 0;
    g_capture.cache = nullptr;
}

extern "C" bool fault20_float_matmul_gws_commit(
    const Fault20MatmulMetadata *metadata) {
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
        kind == CaptureKind::kFloatMatmul && metadata != nullptr &&
        add_count == 1 && cache != nullptr && metadata->all_present != 0 &&
        metadata->any_empty_storage == 0 &&
        metadata->any_shared_storage == 0 && param_order_ok;
    log_matmul_param_collection(MatmulApiKind::kFloat, cache, metadata,
                                tensor_param_count, tensor_params, registration_ok);
    if (!registration_ok) {
        log_matmul_registration("float_matmul_cache_registration_rejected", cache,
                                metadata, -1, add_count);
        return false;
    }
    int32_t device = -1;
    if (g_get_device == nullptr || g_get_device(&device) != ACL_SUCCESS) {
        log_matmul_registration("float_matmul_cache_registration_rejected", cache,
                                metadata, device, add_count);
        return false;
    }
    RegisteredMatmulMetadata registered{};
    registered.tensors = *metadata;
    registered.device = device;
    registered.tensor_param_count = tensor_param_count;
    registered.api_kind = static_cast<uint32_t>(MatmulApiKind::kFloat);
    std::memcpy(registered.tensor_params, tensor_params, sizeof(tensor_params));
    {
        std::lock_guard<std::mutex> lock(registry_mutex());
        matmul_registry()[cache] = registered;
    }
    log_matmul_registration("float_matmul_cache_registered", cache, metadata,
                            device, add_count);
    return true;
}

extern "C" void fault20_float_matmul_gws_abort() {
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
