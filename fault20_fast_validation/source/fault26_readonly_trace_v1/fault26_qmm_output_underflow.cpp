#include "fault26_qmm_underflow_api.h"

#include <algorithm>
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <fcntl.h>
#include <limits>
#include <mutex>
#include <pthread.h>
#include <sys/syscall.h>
#include <sys/uio.h>
#include <time.h>
#include <unistd.h>

#if !defined(__aarch64__)
#error "Fault26 QMM output underflow proxy is version-locked to aarch64"
#endif

#if !defined(_GLIBCXX_USE_CXX11_ABI) || _GLIBCXX_USE_CXX11_ABI != 0
#error "Fault26 QMM output underflow proxy requires ABI0"
#endif

namespace {

using SynchronizeStreamFn = aclError (*)(aclrtStream);
using MallocHostFn = aclError (*)(void **, size_t);
using FreeHostFn = aclError (*)(void *);
using MemcpyFn = aclError (*)(void *, size_t, const void *, size_t,
                              aclrtMemcpyKind);

constexpr int32_t kNotCalled = std::numeric_limits<int32_t>::min() + 1;

struct Pending {
    bool active{false};
    bool ready_for_consumer{false};
    uint64_t token{0};
    uint64_t hit_number{0};
    uint64_t max_hits{1};
    bool all_hits{false};
    Fault26QmmUnderflowContext context{};
    void *replacement_output{nullptr};
    uint64_t shift_bytes{0};
    uint64_t padding_bytes{0};
    uint64_t residual_overlap_bytes{0};
    void *host_block{nullptr};
    uint64_t host_bytes{0};
    uint64_t residual_hash_before{0};
    uint64_t residual_hash_after{0};
    uint64_t residual_hash_restored{0};
    uint64_t shifted_output_hash{0};
    uint64_t original_output_hash{0};
    uint64_t prefix_hash_before{0};
    uint64_t padding_hash_before{0};
    uint64_t padding_hash_restored{0};
    uint64_t changed_residual_bytes{0};
    int32_t pre_sync_status{kNotCalled};
    int32_t host_alloc_status{kNotCalled};
    int32_t residual_d2h_status{kNotCalled};
    int32_t prefix_d2h_status{kNotCalled};
    int32_t post_sync_status{kNotCalled};
    int32_t residual_after_d2h_status{kNotCalled};
    int32_t shifted_output_d2h_status{kNotCalled};
    int32_t output_h2d_status{kNotCalled};
    int32_t output_sync_status{kNotCalled};
    int32_t output_readback_status{kNotCalled};
    int32_t padding_restore_h2d_status{kNotCalled};
    int32_t padding_restore_sync_status{kNotCalled};
    int32_t padding_readback_status{kNotCalled};
    int32_t consumer_status{kNotCalled};
    int32_t restore_h2d_status{kNotCalled};
    int32_t restore_sync_status{kNotCalled};
    int32_t restore_readback_status{kNotCalled};
    int32_t host_free_status{kNotCalled};
    bool output_restored_exact{false};
    bool padding_restored_exact{false};
    bool residual_restored_exact{false};
};

pthread_once_t g_once = PTHREAD_ONCE_INIT;
SynchronizeStreamFn g_sync = nullptr;
MallocHostFn g_malloc_host = nullptr;
FreeHostFn g_free_host = nullptr;
MemcpyFn g_memcpy = nullptr;
bool g_allowed = false;
bool g_ready = false;
std::mutex g_mutex;
Pending g_pending;
uint64_t g_candidates = 0;
uint64_t g_hits = 0;
uint64_t g_next_token = 1;

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
                 uint64_t maximum, bool *ok = nullptr) {
    const char *value = std::getenv(name);
    if (value == nullptr || value[0] == '\0') {
        if (ok != nullptr) {
            *ok = false;
        }
        return fallback;
    }
    char *end = nullptr;
    const unsigned long long parsed = std::strtoull(value, &end, 0);
    const bool valid = end != value && *end == '\0' && parsed >= minimum &&
                       parsed <= maximum;
    if (ok != nullptr) {
        *ok = valid;
    }
    return valid ? static_cast<uint64_t>(parsed) : fallback;
}

int32_t env_i32(const char *name, int32_t fallback) {
    bool ok = false;
    const uint64_t parsed = env_u64(name, static_cast<uint64_t>(fallback), 0,
                                    65535, &ok);
    return ok ? static_cast<int32_t>(parsed) : fallback;
}

uint64_t fnv1a64(const unsigned char *data, size_t size) {
    uint64_t value = 1469598103934665603ULL;
    for (size_t index = 0; index < size; ++index) {
        value ^= data[index];
        value *= 1099511628211ULL;
    }
    return value;
}

void write_line(const char *line, size_t length) {
    const char *path = std::getenv("FAULT26_QMM_UNDERFLOW_LOG_PATH");
    if (path == nullptr || path[0] == '\0') {
        return;
    }
    const int fd = open(path, O_WRONLY | O_CREAT | O_APPEND | O_CLOEXEC, 0600);
    if (fd < 0) {
        return;
    }
    char newline = '\n';
    iovec parts[2] = {{const_cast<char *>(line), length}, {&newline, 1}};
    const ssize_t bytes_written = writev(fd, parts, 2);
    if (bytes_written < 0) {
        // Logging is best-effort and must never perturb the executor path.
    }
    (void)close(fd);
}

void log_pending(const char *event, const Pending &p, aclnnStatus run_status,
                 const char *outcome) {
    char line[4096];
    const int written = std::snprintf(
        line, sizeof(line),
        "{\"event\":\"%s\",\"pid\":%ld,\"tid\":%ld,"
        "\"ts_ns\":%" PRIu64 ",\"outcome\":\"%s\","
        "\"device\":%d,\"token\":%" PRIu64
        ",\"hit_number\":%" PRIu64 ",\"max_hits\":%" PRIu64
        ",\"all_hits\":%s"
        ",\"run_index\":%" PRIu64 ",\"producer_sequence\":%" PRIu64
        ",\"consumer_sequence\":%" PRIu64 ",\"cache\":\"%p\","
        "\"original_output\":\"%p\",\"replacement_output\":\"%p\","
        "\"output_bytes\":%" PRIu64 ",\"output_shape\":[%" PRId64
        ",%" PRId64 "],\"residual\":\"%p\",\"residual_bytes\":%" PRIu64
        ",\"residual_shape\":[%" PRId64 ",%" PRId64 "],"
        "\"shift_bytes\":%" PRIu64 ",\"allocator_padding_bytes\":%" PRIu64
        ",\"residual_overlap_bytes\":%" PRIu64 ","
        "\"crosses_writer_allocation_boundary\":true,"
        "\"run_status\":%d,\"pre_sync_status\":%d,"
        "\"host_alloc_status\":%d,\"residual_d2h_status\":%d,"
        "\"prefix_d2h_status\":%d,"
        "\"post_sync_status\":%d,\"residual_after_d2h_status\":%d,"
        "\"shifted_output_d2h_status\":%d,\"output_h2d_status\":%d,"
        "\"output_sync_status\":%d,\"output_readback_status\":%d,"
        "\"padding_restore_h2d_status\":%d,"
        "\"padding_restore_sync_status\":%d,"
        "\"padding_readback_status\":%d,"
        "\"consumer_status\":%d,\"restore_h2d_status\":%d,"
        "\"restore_sync_status\":%d,\"restore_readback_status\":%d,"
        "\"host_free_status\":%d,\"changed_residual_bytes\":%" PRIu64
        ",\"residual_hash_before\":\"0x%016" PRIx64 "\","
        "\"residual_hash_after\":\"0x%016" PRIx64 "\","
        "\"residual_hash_restored\":\"0x%016" PRIx64 "\","
        "\"shifted_output_hash\":\"0x%016" PRIx64 "\","
        "\"original_output_hash\":\"0x%016" PRIx64 "\","
        "\"prefix_hash_before\":\"0x%016" PRIx64 "\","
        "\"padding_hash_before\":\"0x%016" PRIx64 "\","
        "\"padding_hash_restored\":\"0x%016" PRIx64 "\","
        "\"output_restored_exact\":%s,\"padding_restored_exact\":%s,"
        "\"residual_restored_exact\":%s}",
        event, static_cast<long>(getpid()),
        static_cast<long>(syscall(SYS_gettid)), monotonic_ns(), outcome,
        p.context.device, p.token, p.hit_number, p.max_hits,
        p.all_hits ? "true" : "false",
        p.context.run_index,
        p.context.producer_sequence, p.context.consumer_sequence,
        p.context.cache, p.context.original_output, p.replacement_output,
        p.context.output_bytes, p.context.output_m, p.context.output_n,
        p.context.residual, p.context.residual_bytes, p.context.residual_m,
        p.context.residual_k, p.shift_bytes, p.padding_bytes,
        p.residual_overlap_bytes, static_cast<int>(run_status),
        p.pre_sync_status, p.host_alloc_status, p.residual_d2h_status,
        p.prefix_d2h_status,
        p.post_sync_status, p.residual_after_d2h_status,
        p.shifted_output_d2h_status, p.output_h2d_status,
        p.output_sync_status, p.output_readback_status,
        p.padding_restore_h2d_status, p.padding_restore_sync_status,
        p.padding_readback_status, p.consumer_status,
        p.restore_h2d_status, p.restore_sync_status,
        p.restore_readback_status, p.host_free_status,
        p.changed_residual_bytes, p.residual_hash_before,
        p.residual_hash_after, p.residual_hash_restored,
        p.shifted_output_hash, p.original_output_hash, p.prefix_hash_before,
        p.padding_hash_before, p.padding_hash_restored,
        p.output_restored_exact ? "true" : "false",
        p.padding_restored_exact ? "true" : "false",
        p.residual_restored_exact ? "true" : "false");
    if (written > 0 && static_cast<size_t>(written) < sizeof(line)) {
        write_line(line, static_cast<size_t>(written));
    }
}

void initialize() {
    g_sync = reinterpret_cast<SynchronizeStreamFn>(
        dlsym(RTLD_DEFAULT, "aclrtSynchronizeStream"));
    g_malloc_host = reinterpret_cast<MallocHostFn>(
        dlsym(RTLD_DEFAULT, "aclrtMallocHost"));
    g_free_host =
        reinterpret_cast<FreeHostFn>(dlsym(RTLD_DEFAULT, "aclrtFreeHost"));
    g_memcpy =
        reinterpret_cast<MemcpyFn>(dlsym(RTLD_DEFAULT, "aclrtMemcpy"));
    g_allowed = env_is_one("VLLM_SERVER_DEV_MODE") &&
                env_is_one("FAULT26_QMM_UNDERFLOW_ALLOWED");
    g_ready = g_allowed && g_sync != nullptr && g_malloc_host != nullptr &&
              g_free_host != nullptr && g_memcpy != nullptr;
}

unsigned char *residual_original(Pending &p) {
    return static_cast<unsigned char *>(p.host_block);
}
unsigned char *residual_after(Pending &p) {
    return residual_original(p) + p.context.residual_bytes;
}
unsigned char *shifted_output(Pending &p) {
    return residual_after(p) + p.context.residual_bytes;
}
unsigned char *output_readback(Pending &p) {
    return shifted_output(p) + p.context.output_bytes;
}
unsigned char *prefix_original(Pending &p) {
    return output_readback(p) + p.context.output_bytes;
}
unsigned char *prefix_readback(Pending &p) {
    return prefix_original(p) + p.shift_bytes;
}

void *padding_device_address(const Pending &p) {
    return reinterpret_cast<void *>(
        reinterpret_cast<uintptr_t>(p.context.residual) +
        p.context.residual_bytes);
}

void release_pending(Pending *p) {
    if (p == nullptr || p->host_block == nullptr || g_free_host == nullptr) {
        return;
    }
    p->host_free_status = static_cast<int32_t>(g_free_host(p->host_block));
    p->host_block = nullptr;
}

bool ranges_overlap(uintptr_t a, uint64_t a_bytes, uintptr_t b,
                    uint64_t b_bytes) {
    if (a_bytes == 0 || b_bytes == 0 ||
        a > std::numeric_limits<uintptr_t>::max() - a_bytes ||
        b > std::numeric_limits<uintptr_t>::max() - b_bytes) {
        return true;
    }
    return a < b + b_bytes && b < a + a_bytes;
}

bool match_target_pair_spec(const char *spec, uint64_t output,
                            uint64_t residual,
                            uint64_t *residual_allocation_end,
                            uint64_t *output_allocation_begin) {
    if (spec == nullptr || spec[0] == '\0' ||
        residual_allocation_end == nullptr ||
        output_allocation_begin == nullptr) {
        return false;
    }
    const char *cursor = spec;
    bool matched = false;
    uint64_t matched_residual_end = 0;
    uint64_t matched_output_begin = 0;
    size_t count = 0;
    while (*cursor != '\0') {
        if (++count > 8) {
            return false;
        }
        uint64_t values[4]{};
        for (size_t index = 0; index < 4; ++index) {
            char *end = nullptr;
            const unsigned long long parsed = std::strtoull(cursor, &end, 0);
            if (end == cursor) {
                return false;
            }
            values[index] = static_cast<uint64_t>(parsed);
            const char expected = index == 3 ? ';' : ',';
            if (index < 3) {
                if (*end != expected) {
                    return false;
                }
                cursor = end + 1;
            } else if (*end == ';') {
                cursor = end + 1;
                if (*cursor == '\0') {
                    return false;
                }
            } else if (*end == '\0') {
                cursor = end;
            } else {
                return false;
            }
        }
        if (values[0] == output && values[1] == residual) {
            if (matched) {
                return false;
            }
            matched = true;
            matched_residual_end = values[2];
            matched_output_begin = values[3];
        }
    }
    if (!matched) {
        return false;
    }
    *residual_allocation_end = matched_residual_end;
    *output_allocation_begin = matched_output_begin;
    return true;
}

}  // namespace

extern "C" bool fault26_qmm_underflow_prepare(
    const Fault26QmmUnderflowContext *context, aclrtStream stream,
    Fault26QmmUnderflowDecision *decision) {
    pthread_once(&g_once, initialize);
    if (decision != nullptr) {
        *decision = Fault26QmmUnderflowDecision{};
    }
    if (!g_ready || context == nullptr || decision == nullptr ||
        context->abi_version != FAULT26_QMM_UNDERFLOW_ABI_VERSION ||
        !env_is_one("FAULT26_QMM_UNDERFLOW_ENABLED")) {
        return false;
    }
    const int32_t target_device =
        env_i32("FAULT26_QMM_UNDERFLOW_TARGET_DEVICE", -1);
    bool values_ok = false;
    const uint64_t target_output_bytes = env_u64(
        "FAULT26_QMM_UNDERFLOW_TARGET_OUTPUT_BYTES", 0, 1, 8388608, &values_ok);
    if (!values_ok) {
        return false;
    }
    const uint64_t target_residual_bytes = env_u64(
        "FAULT26_QMM_UNDERFLOW_TARGET_RESIDUAL_BYTES", 0, 1, 8388608,
        &values_ok);
    if (!values_ok) {
        return false;
    }
    uint64_t target_output =
        reinterpret_cast<uint64_t>(context->original_output);
    uint64_t target_residual = reinterpret_cast<uint64_t>(context->residual);
    uint64_t residual_allocation_end = 0;
    uint64_t output_allocation_begin = 0;
    const char *pair_spec =
        std::getenv("FAULT26_QMM_UNDERFLOW_TARGET_PAIRS");
    if (pair_spec != nullptr && pair_spec[0] != '\0') {
        if (!match_target_pair_spec(pair_spec, target_output, target_residual,
                                    &residual_allocation_end,
                                    &output_allocation_begin)) {
            return false;
        }
    } else {
        target_output = env_u64(
            "FAULT26_QMM_UNDERFLOW_TARGET_OUTPUT_ADDRESS", 0, 1,
            std::numeric_limits<uint64_t>::max(), &values_ok);
        if (!values_ok) {
            return false;
        }
        target_residual = env_u64(
            "FAULT26_QMM_UNDERFLOW_TARGET_RESIDUAL_ADDRESS", 0, 1,
            std::numeric_limits<uint64_t>::max(), &values_ok);
        if (!values_ok) {
            return false;
        }
        residual_allocation_end = env_u64(
            "FAULT26_QMM_UNDERFLOW_RESIDUAL_ALLOCATION_END", 0, 1,
            std::numeric_limits<uint64_t>::max(), &values_ok);
        if (!values_ok) {
            return false;
        }
        output_allocation_begin = env_u64(
            "FAULT26_QMM_UNDERFLOW_OUTPUT_ALLOCATION_BEGIN", 0, 1,
            std::numeric_limits<uint64_t>::max(), &values_ok);
        if (!values_ok) {
            return false;
        }
    }
    const uint64_t shift = env_u64(
        "FAULT26_QMM_UNDERFLOW_SHIFT_BYTES", 0, 1, 4096, &values_ok);
    if (!values_ok) {
        return false;
    }
    const uint64_t max_hits = env_u64(
        "FAULT26_QMM_UNDERFLOW_MAX_HITS", 1, 1, 4096, &values_ok);
    if (!values_ok || context->device != target_device ||
        reinterpret_cast<uint64_t>(context->original_output) != target_output ||
        context->output_bytes != target_output_bytes ||
        reinterpret_cast<uint64_t>(context->residual) != target_residual ||
        context->residual_bytes != target_residual_bytes ||
        target_output != output_allocation_begin ||
        residual_allocation_end != output_allocation_begin ||
        target_residual >
            std::numeric_limits<uint64_t>::max() - target_residual_bytes) {
        return false;
    }
    const uint64_t residual_end = target_residual + target_residual_bytes;
    if (residual_end > residual_allocation_end) {
        return false;
    }
    const uint64_t padding = residual_allocation_end - residual_end;
    if (shift <= padding || shift > target_output - target_residual) {
        return false;
    }
    const uint64_t replacement = target_output - shift;
    const uint64_t overlap = shift - padding;

    std::lock_guard<std::mutex> lock(g_mutex);
    ++g_candidates;
    if (g_pending.active || g_hits >= max_hits) {
        return false;
    }
    Pending pending{};
    pending.active = true;
    pending.token = g_next_token++;
    pending.hit_number = g_hits + 1;
    pending.max_hits = max_hits;
    pending.all_hits = env_is_one("FAULT26_QMM_UNDERFLOW_ALL_HITS");
    pending.context = *context;
    pending.replacement_output = reinterpret_cast<void *>(replacement);
    pending.shift_bytes = shift;
    pending.padding_bytes = padding;
    pending.residual_overlap_bytes = overlap;
    pending.pre_sync_status = static_cast<int32_t>(g_sync(stream));
    if (pending.pre_sync_status != ACL_SUCCESS) {
        log_pending("fault26_qmm_output_underflow_prepare_failed", pending,
                    static_cast<aclnnStatus>(pending.pre_sync_status),
                    "pre_sync_failed");
        return false;
    }
    if (context->residual_bytes >
            std::numeric_limits<uint64_t>::max() / 2 ||
        context->output_bytes >
            (std::numeric_limits<uint64_t>::max() -
             context->residual_bytes * 2) /
                2 ||
        shift > (std::numeric_limits<uint64_t>::max() -
                 context->residual_bytes * 2 - context->output_bytes * 2) /
                    2) {
        return false;
    }
    pending.host_bytes =
        context->residual_bytes * 2 + context->output_bytes * 2 + shift * 2;
    pending.host_alloc_status = static_cast<int32_t>(
        g_malloc_host(&pending.host_block, static_cast<size_t>(pending.host_bytes)));
    if (pending.host_alloc_status != ACL_SUCCESS ||
        pending.host_block == nullptr) {
        log_pending("fault26_qmm_output_underflow_prepare_failed", pending,
                    static_cast<aclnnStatus>(pending.host_alloc_status),
                    "host_alloc_failed");
        return false;
    }
    pending.residual_d2h_status = static_cast<int32_t>(g_memcpy(
        residual_original(pending), context->residual_bytes, context->residual,
        context->residual_bytes, ACL_MEMCPY_DEVICE_TO_HOST));
    if (pending.residual_d2h_status != ACL_SUCCESS) {
        release_pending(&pending);
        log_pending("fault26_qmm_output_underflow_prepare_failed", pending,
                    static_cast<aclnnStatus>(pending.residual_d2h_status),
                    "residual_backup_failed");
        return false;
    }
    pending.prefix_d2h_status = static_cast<int32_t>(g_memcpy(
        prefix_original(pending), pending.shift_bytes,
        pending.replacement_output, pending.shift_bytes,
        ACL_MEMCPY_DEVICE_TO_HOST));
    if (pending.prefix_d2h_status != ACL_SUCCESS) {
        release_pending(&pending);
        log_pending("fault26_qmm_output_underflow_prepare_failed", pending,
                    static_cast<aclnnStatus>(pending.prefix_d2h_status),
                    "prefix_backup_failed");
        return false;
    }
    pending.residual_hash_before = fnv1a64(
        residual_original(pending), static_cast<size_t>(context->residual_bytes));
    pending.prefix_hash_before = fnv1a64(
        prefix_original(pending), static_cast<size_t>(pending.shift_bytes));
    pending.padding_hash_before = fnv1a64(
        prefix_original(pending) + pending.residual_overlap_bytes,
        static_cast<size_t>(pending.padding_bytes));
    g_pending = pending;
    ++g_hits;
    if (g_hits >= max_hits) {
        (void)setenv("FAULT26_QMM_UNDERFLOW_ENABLED", "0", 1);
    }
    decision->abi_version = FAULT26_QMM_UNDERFLOW_ABI_VERSION;
    decision->claimed = 1;
    decision->token = pending.token;
    decision->replacement_output = pending.replacement_output;
    log_pending("fault26_qmm_output_underflow_prepared", g_pending,
                static_cast<aclnnStatus>(ACL_SUCCESS), "prepared");
    return true;
}

extern "C" void fault26_qmm_underflow_complete(
    const Fault26QmmUnderflowContext *context,
    const Fault26QmmUnderflowDecision *decision, aclrtStream stream,
    aclnnStatus run_status) {
    pthread_once(&g_once, initialize);
    if (!g_ready || context == nullptr || decision == nullptr) {
        return;
    }
    std::lock_guard<std::mutex> lock(g_mutex);
    if (!g_pending.active || decision->claimed == 0 ||
        g_pending.token != decision->token) {
        return;
    }
    Pending &p = g_pending;
    p.post_sync_status = static_cast<int32_t>(g_sync(stream));
    if (run_status == ACL_SUCCESS && p.post_sync_status == ACL_SUCCESS) {
        p.residual_after_d2h_status = static_cast<int32_t>(g_memcpy(
            residual_after(p), p.context.residual_bytes, p.context.residual,
            p.context.residual_bytes, ACL_MEMCPY_DEVICE_TO_HOST));
        p.shifted_output_d2h_status = static_cast<int32_t>(g_memcpy(
            shifted_output(p), p.context.output_bytes, p.replacement_output,
            p.context.output_bytes, ACL_MEMCPY_DEVICE_TO_HOST));
    }
    if (p.residual_after_d2h_status == ACL_SUCCESS) {
        p.residual_hash_after = fnv1a64(
            residual_after(p), static_cast<size_t>(p.context.residual_bytes));
        for (uint64_t index = 0; index < p.context.residual_bytes; ++index) {
            if (residual_original(p)[index] != residual_after(p)[index]) {
                ++p.changed_residual_bytes;
            }
        }
    }
    if (p.shifted_output_d2h_status == ACL_SUCCESS) {
        p.shifted_output_hash = fnv1a64(
            shifted_output(p), static_cast<size_t>(p.context.output_bytes));
        p.output_h2d_status = static_cast<int32_t>(g_memcpy(
            p.context.original_output, p.context.output_bytes,
            shifted_output(p), p.context.output_bytes,
            ACL_MEMCPY_HOST_TO_DEVICE));
        if (p.output_h2d_status == ACL_SUCCESS) {
            if (p.padding_bytes == 0) {
                p.padding_restore_h2d_status = ACL_SUCCESS;
            } else {
                p.padding_restore_h2d_status = static_cast<int32_t>(g_memcpy(
                    padding_device_address(p), p.padding_bytes,
                    prefix_original(p) + p.residual_overlap_bytes,
                    p.padding_bytes, ACL_MEMCPY_HOST_TO_DEVICE));
            }
        }
        if (p.output_h2d_status == ACL_SUCCESS &&
            p.padding_restore_h2d_status == ACL_SUCCESS) {
            p.output_sync_status = static_cast<int32_t>(g_sync(stream));
            p.padding_restore_sync_status = p.output_sync_status;
        }
        if (p.output_sync_status == ACL_SUCCESS) {
            p.output_readback_status = static_cast<int32_t>(g_memcpy(
                output_readback(p), p.context.output_bytes,
                p.context.original_output, p.context.output_bytes,
                ACL_MEMCPY_DEVICE_TO_HOST));
            if (p.padding_bytes == 0) {
                p.padding_readback_status = ACL_SUCCESS;
                p.padding_restored_exact = true;
            } else {
                p.padding_readback_status = static_cast<int32_t>(g_memcpy(
                    prefix_readback(p), p.padding_bytes,
                    padding_device_address(p), p.padding_bytes,
                    ACL_MEMCPY_DEVICE_TO_HOST));
            }
        }
    }
    if (p.output_readback_status == ACL_SUCCESS) {
        p.original_output_hash = fnv1a64(
            output_readback(p), static_cast<size_t>(p.context.output_bytes));
        p.output_restored_exact =
            p.original_output_hash == p.shifted_output_hash &&
            std::memcmp(output_readback(p), shifted_output(p),
                        static_cast<size_t>(p.context.output_bytes)) == 0;
    }
    if (p.padding_bytes > 0 && p.padding_readback_status == ACL_SUCCESS) {
        p.padding_hash_restored = fnv1a64(
            prefix_readback(p), static_cast<size_t>(p.padding_bytes));
        p.padding_restored_exact =
            p.padding_hash_restored == p.padding_hash_before &&
            std::memcmp(prefix_readback(p),
                        prefix_original(p) + p.residual_overlap_bytes,
                        static_cast<size_t>(p.padding_bytes)) == 0;
    }
    p.ready_for_consumer = run_status == ACL_SUCCESS &&
                           p.post_sync_status == ACL_SUCCESS &&
                           p.residual_after_d2h_status == ACL_SUCCESS &&
                           p.shifted_output_d2h_status == ACL_SUCCESS &&
                           p.output_h2d_status == ACL_SUCCESS &&
                           p.output_sync_status == ACL_SUCCESS &&
                           p.output_readback_status == ACL_SUCCESS &&
                           p.padding_restore_h2d_status == ACL_SUCCESS &&
                           p.padding_restore_sync_status == ACL_SUCCESS &&
                           p.padding_readback_status == ACL_SUCCESS &&
                           p.output_restored_exact &&
                           p.padding_restored_exact &&
                           p.changed_residual_bytes > 0;
    log_pending("fault26_qmm_output_underflow_applied", p, run_status,
                p.ready_for_consumer ? "pending_consumer" : "post_run_failed");
    if (!p.ready_for_consumer) {
        p.restore_h2d_status = static_cast<int32_t>(g_memcpy(
            p.context.residual, p.context.residual_bytes, residual_original(p),
            p.context.residual_bytes, ACL_MEMCPY_HOST_TO_DEVICE));
        if (p.restore_h2d_status == ACL_SUCCESS) {
            if (p.padding_bytes == 0 ||
                g_memcpy(padding_device_address(p), p.padding_bytes,
                         prefix_original(p) + p.residual_overlap_bytes,
                         p.padding_bytes,
                         ACL_MEMCPY_HOST_TO_DEVICE) == ACL_SUCCESS) {
                p.restore_sync_status = static_cast<int32_t>(g_sync(stream));
            }
        }
        release_pending(&p);
        log_pending("fault26_qmm_output_underflow_aborted", p, run_status,
                    "restored_before_consumer");
        g_pending = Pending{};
    }
}

extern "C" void fault26_qmm_underflow_consumer_complete(
    int32_t device, uint64_t consumer_sequence, void *x2, uint64_t x2_bytes,
    void *x_out, uint64_t x_out_bytes, aclrtStream stream,
    aclnnStatus consumer_status) {
    pthread_once(&g_once, initialize);
    if (!g_ready) {
        return;
    }
    std::lock_guard<std::mutex> lock(g_mutex);
    if (!g_pending.active || !g_pending.ready_for_consumer ||
        g_pending.context.device != device || consumer_sequence == 0 ||
        consumer_status != ACL_SUCCESS || x_out == nullptr || x_out_bytes == 0 ||
        ranges_overlap(reinterpret_cast<uintptr_t>(x2), x2_bytes,
                       reinterpret_cast<uintptr_t>(x_out), x_out_bytes)) {
        return;
    }
    Pending &p = g_pending;
    const uint64_t expected_consumer = p.context.producer_sequence + 1;
    if (consumer_sequence < expected_consumer) {
        return;
    }
    if (consumer_sequence != expected_consumer ||
        p.context.residual != x2 || p.context.residual_bytes != x2_bytes) {
        p.context.consumer_sequence = consumer_sequence;
        p.consumer_status = static_cast<int32_t>(consumer_status);
        p.restore_h2d_status = static_cast<int32_t>(g_memcpy(
            p.context.residual, p.context.residual_bytes, residual_original(p),
            p.context.residual_bytes, ACL_MEMCPY_HOST_TO_DEVICE));
        if (p.restore_h2d_status == ACL_SUCCESS) {
            p.restore_sync_status = static_cast<int32_t>(g_sync(stream));
        }
        release_pending(&p);
        log_pending("fault26_qmm_output_underflow_aborted", p,
                    static_cast<aclnnStatus>(consumer_status),
                    "unexpected_consumer_restored");
        g_pending = Pending{};
        return;
    }
    p.context.consumer_sequence = consumer_sequence;
    p.consumer_status = static_cast<int32_t>(consumer_status);
    p.restore_h2d_status = static_cast<int32_t>(g_memcpy(
        p.context.residual, p.context.residual_bytes, residual_original(p),
        p.context.residual_bytes, ACL_MEMCPY_HOST_TO_DEVICE));
    if (p.restore_h2d_status == ACL_SUCCESS) {
        p.restore_sync_status = static_cast<int32_t>(g_sync(stream));
    }
    if (p.restore_sync_status == ACL_SUCCESS) {
        p.restore_readback_status = static_cast<int32_t>(g_memcpy(
            residual_after(p), p.context.residual_bytes, p.context.residual,
            p.context.residual_bytes, ACL_MEMCPY_DEVICE_TO_HOST));
    }
    if (p.restore_readback_status == ACL_SUCCESS) {
        p.residual_hash_restored = fnv1a64(
            residual_after(p), static_cast<size_t>(p.context.residual_bytes));
        p.residual_restored_exact =
            p.residual_hash_restored == p.residual_hash_before &&
            std::memcmp(residual_after(p), residual_original(p),
                        static_cast<size_t>(p.context.residual_bytes)) == 0;
    }
    release_pending(&p);
    const bool pass = p.output_restored_exact && p.padding_restored_exact &&
                      p.residual_restored_exact &&
                      p.changed_residual_bytes > 0 &&
                      p.consumer_status == ACL_SUCCESS;
    log_pending("fault26_qmm_output_underflow_consumer_restored", p,
                static_cast<aclnnStatus>(ACL_SUCCESS),
                pass ? "QMM_OUTPUT_BASE_UNDERFLOW_RESIDUAL_STOMP_PROXY_PROVEN"
                     : "consumer_restore_failed");
    g_pending = Pending{};
}
