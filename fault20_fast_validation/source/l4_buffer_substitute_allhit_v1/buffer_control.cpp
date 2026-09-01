#include "fault20_buffer_api.h"

#include <cerrno>
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <sched.h>
#include <sys/file.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

namespace {

Fault20Control *map_control(int *fd_out) {
    const int fd = shm_open(FAULT20_CONTROL_SHM, O_RDWR | O_CREAT | O_CLOEXEC, 0600);
    if (fd < 0) {
        std::fprintf(stderr, "shm_open failed: %s\n", std::strerror(errno));
        return nullptr;
    }
    if (flock(fd, LOCK_EX) != 0 ||
        ftruncate(fd, static_cast<off_t>(sizeof(Fault20Control))) != 0) {
        std::fprintf(stderr, "prepare shared memory failed: %s\n", std::strerror(errno));
        (void)close(fd);
        return nullptr;
    }
    void *mapped = mmap(nullptr, sizeof(Fault20Control), PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    if (mapped == MAP_FAILED) {
        std::fprintf(stderr, "mmap failed: %s\n", std::strerror(errno));
        (void)flock(fd, LOCK_UN);
        (void)close(fd);
        return nullptr;
    }
    auto *control = static_cast<Fault20Control *>(mapped);
    if (control->magic != FAULT20_CONTROL_MAGIC || control->version != FAULT20_CONTROL_VERSION) {
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
        control->target_dtype = 27;
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
        control->proof_mode = FAULT20_PROOF_MODE_STRICT_PRODUCER;
        control->proof_reserved = 0;
        control->hit_scope = FAULT20_HIT_SCOPE_ARM_LIFETIME;
        control->max_hits_total = 0;
    }
    *fd_out = fd;
    return control;
}

const char *proof_mode_name(uint32_t proof_mode) {
    if (proof_mode == FAULT20_PROOF_MODE_STRICT_PRODUCER) {
        return "STRICT_PRODUCER";
    }
    if (proof_mode == FAULT20_PROOF_MODE_SEMANTIC_SLOT) {
        return "SEMANTIC_SLOT";
    }
    return "UNKNOWN";
}

void print_status(const Fault20Control *control) {
    std::printf(
        "{\"magic\":\"0x%016" PRIx64 "\",\"version\":%u,\"enabled\":%u,"
        "\"mode\":%u,\"target_device\":%d,\"target_m\":%d,\"target_k\":%d,"
        "\"target_call_index\":%u,\"max_hits\":%u,\"hit_count\":%u,"
        "\"candidate_count\":%u,\"quant_step\":%u,"
        "\"keep_mantissa_bits\":%u,\"extreme_value\":%d,"
        "\"extreme_ratio_bps\":%u,\"extreme_pattern\":%u,"
        "\"target_dtype\":%d,\"last_status\":%d,"
        "\"selection_mode\":%u,\"hit_scope\":%u,\"target_api\":%u,"
        "\"proof_mode\":%u,\"proof_mode_name\":\"%s\","
        "\"max_bytes\":%" PRIu64 ",\"max_output_bytes\":%" PRIu64
        ",\"max_weight_bytes\":%" PRIu64 ",\"target_x2_n\":%d"
        ",\"max_scale_bytes\":%" PRIu64 ",\"target_scale_elements\":%d"
        ",\"scale_factor\":%.17g"
        ",\"generation\":%" PRIu64 ",\"request_generation\":%" PRIu64
        ",\"target_device_mask\":\"0x%016" PRIx64
        "\",\"max_hits_per_device\":%u,\"max_hits_total\":%u,"
        "\"total_hit_count\":%u,"
        "\"candidate_count_per_device\":[%u,%u],"
        "\"hit_count_per_device\":[%u,%u],"
        "\"request_candidate_count_per_device\":[%u,%u],"
        "\"request_hit_count_per_device\":[%u,%u],"
        "\"last_status_per_device\":[%d,%d]}\n",
        control->magic, control->version,
        __atomic_load_n(&control->enabled, __ATOMIC_ACQUIRE), control->mode,
        control->target_device, control->target_m, control->target_k,
        control->target_call_index, control->max_hits,
        __atomic_load_n(&control->hit_count, __ATOMIC_ACQUIRE),
        __atomic_load_n(&control->candidate_count, __ATOMIC_ACQUIRE), control->quant_step,
        control->keep_mantissa_bits, control->extreme_value,
        control->extreme_ratio_bps, control->extreme_pattern, control->target_dtype,
        __atomic_load_n(&control->last_status, __ATOMIC_ACQUIRE),
        control->selection_mode, control->hit_scope, control->target_api,
        control->proof_mode, proof_mode_name(control->proof_mode), control->max_bytes,
        control->max_output_bytes, control->max_weight_bytes, control->target_x2_n,
        control->max_scale_bytes, control->target_scale_elements,
        control->scale_factor,
        control->generation,
        __atomic_load_n(&control->request_generation, __ATOMIC_ACQUIRE),
        control->target_device_mask, control->max_hits_per_device,
        control->max_hits_total,
        __atomic_load_n(&control->hit_count, __ATOMIC_ACQUIRE),
        __atomic_load_n(&control->candidate_count_per_device[0], __ATOMIC_ACQUIRE),
        __atomic_load_n(&control->candidate_count_per_device[1], __ATOMIC_ACQUIRE),
        __atomic_load_n(&control->hit_count_per_device[0], __ATOMIC_ACQUIRE),
        __atomic_load_n(&control->hit_count_per_device[1], __ATOMIC_ACQUIRE),
        __atomic_load_n(&control->request_candidate_count_per_device[0],
                        __ATOMIC_ACQUIRE),
        __atomic_load_n(&control->request_candidate_count_per_device[1],
                        __ATOMIC_ACQUIRE),
        __atomic_load_n(&control->request_hit_count_per_device[0],
                        __ATOMIC_ACQUIRE),
        __atomic_load_n(&control->request_hit_count_per_device[1],
                        __ATOMIC_ACQUIRE),
        __atomic_load_n(&control->last_status_per_device[0], __ATOMIC_ACQUIRE),
        __atomic_load_n(&control->last_status_per_device[1], __ATOMIC_ACQUIRE));
}

bool parse_int(const char *text, long long min_value, long long max_value, long long *value) {
    char *end = nullptr;
    errno = 0;
    const long long parsed = std::strtoll(text, &end, 10);
    if (errno != 0 || end == text || *end != '\0' || parsed < min_value || parsed > max_value) {
        return false;
    }
    *value = parsed;
    return true;
}

void reset_request_counters(Fault20Control *control) {
    for (size_t device = 0; device < FAULT20_MAX_DEVICES; ++device) {
        __atomic_store_n(&control->request_candidate_count_per_device[device], 0U,
                         __ATOMIC_RELEASE);
        __atomic_store_n(&control->request_hit_count_per_device[device], 0U,
                         __ATOMIC_RELEASE);
    }
}

void reset_device_counters(Fault20Control *control) {
    for (size_t device = 0; device < FAULT20_MAX_DEVICES; ++device) {
        __atomic_store_n(&control->candidate_count_per_device[device], 0U,
                         __ATOMIC_RELEASE);
        __atomic_store_n(&control->hit_count_per_device[device], 0U,
                         __ATOMIC_RELEASE);
        __atomic_store_n(&control->last_status_per_device[device], 0,
                         __ATOMIC_RELEASE);
    }
    reset_request_counters(control);
    __atomic_store_n(&control->request_generation, 0ULL, __ATOMIC_RELEASE);
    __atomic_store_n(&control->request_reservation_lock, 0U, __ATOMIC_RELEASE);
    control->hit_scope = FAULT20_HIT_SCOPE_ARM_LIFETIME;
    control->max_hits_total = 0;
    control->proof_mode = FAULT20_PROOF_MODE_STRICT_PRODUCER;
    control->proof_reserved = 0;
}

void lock_request_reservation(Fault20Control *control) {
    uint32_t expected = 0;
    while (!__atomic_compare_exchange_n(&control->request_reservation_lock, &expected,
                                        1U, false, __ATOMIC_ACQUIRE,
                                        __ATOMIC_RELAXED)) {
        expected = 0;
        sched_yield();
    }
}

void unlock_request_reservation(Fault20Control *control) {
    __atomic_store_n(&control->request_reservation_lock, 0U, __ATOMIC_RELEASE);
}

bool developer_mode_enabled() {
    const char *value = std::getenv("VLLM_SERVER_DEV_MODE");
    return value != nullptr && std::strcmp(value, "1") == 0;
}

uint32_t parse_extreme_pattern(const char *text) {
    if (std::strcmp(text, "positive_max") == 0) {
        return FAULT20_EXTREME_POSITIVE_MAX;
    }
    if (std::strcmp(text, "negative_max") == 0) {
        return FAULT20_EXTREME_NEGATIVE_MAX;
    }
    if (std::strcmp(text, "alternating") == 0) {
        return FAULT20_EXTREME_ALTERNATING;
    }
    return 0;
}

}  // namespace

int main(int argc, char **argv) {
    int fd = -1;
    Fault20Control *control = map_control(&fd);
    if (control == nullptr) {
        return 1;
    }
    int result = 0;
    const bool developer_operation =
        argc >= 2 && (std::strncmp(argv[1], "arm-", 4) == 0 ||
                      std::strcmp(argv[1], "request-begin") == 0);
    if (developer_operation && !developer_mode_enabled()) {
        std::fprintf(stderr,
                     "refusing developer control operation: VLLM_SERVER_DEV_MODE must equal 1\n");
        result = 3;
    } else if (argc == 2 && std::strcmp(argv[1], "status") == 0) {
        print_status(control);
    } else if (argc == 3 && std::strcmp(argv[1], "request-begin") == 0) {
        long long request_id = 0;
        if (!parse_int(argv[2], 1, 0x7FFFFFFFFFFFFFFFLL, &request_id) ||
            control->hit_scope != FAULT20_HIT_SCOPE_REQUEST ||
            __atomic_load_n(&control->enabled, __ATOMIC_ACQUIRE) == 0) {
            std::fprintf(stderr,
                         "request-begin requires an armed request-scoped control and a positive monotonic request id\n");
            result = 2;
        } else {
            const uint64_t requested = static_cast<uint64_t>(request_id);
            const uint64_t current =
                __atomic_load_n(&control->request_generation, __ATOMIC_ACQUIRE);
            if (requested < current) {
                std::fprintf(stderr, "request-begin request id is stale\n");
                result = 2;
            } else if (requested == current) {
                print_status(control);
            } else {
                // A controller must advance this only at a completed request
                // boundary. Disabling first prevents a concurrent reservation
                // from being attributed to the new request while counters reset.
                __atomic_store_n(&control->enabled, 0U, __ATOMIC_RELEASE);
                lock_request_reservation(control);
                reset_request_counters(control);
                __atomic_store_n(&control->request_generation, requested,
                                 __ATOMIC_RELEASE);
                unlock_request_reservation(control);
                __atomic_store_n(&control->enabled, 1U, __ATOMIC_RELEASE);
                print_status(control);
            }
        }
    } else if (argc == 2 && std::strcmp(argv[1], "disarm") == 0) {
        __atomic_store_n(&control->enabled, 0U, __ATOMIC_RELEASE);
        ++control->generation;
        print_status(control);
    } else if (argc == 2 && std::strcmp(argv[1], "reset") == 0) {
        __atomic_store_n(&control->enabled, 0U, __ATOMIC_RELEASE);
        __atomic_store_n(&control->hit_count, 0U, __ATOMIC_RELEASE);
        __atomic_store_n(&control->candidate_count, 0U, __ATOMIC_RELEASE);
        __atomic_store_n(&control->last_status, 0, __ATOMIC_RELEASE);
        reset_device_counters(control);
        control->mode = FAULT20_MODE_DISABLED;
        control->selection_mode = FAULT20_SELECTION_BOUNDED;
        control->target_api = FAULT20_TARGET_API_NONE;
        control->proof_mode = FAULT20_PROOF_MODE_STRICT_PRODUCER;
        control->proof_reserved = 0;
        control->extreme_value = 127;
        control->extreme_ratio_bps = 10000;
        control->extreme_pattern = FAULT20_EXTREME_POSITIVE_MAX;
        control->max_output_bytes = 65536;
        control->max_weight_bytes = 268435456;
        control->target_x2_n = 0;
        control->max_scale_bytes = 1048576;
        control->target_scale_elements = 0;
        control->scale_factor = 1.0;
        ++control->generation;
        print_status(control);
    } else if (argc == 5 &&
               std::strcmp(argv[1], "arm-float-matmul-finite") == 0) {
        long long device = 0;
        long long call_index = 0;
        long long exponent = 0;
        if (!parse_int(argv[2], 0, 63, &device) ||
            !parse_int(argv[3], 1, 1048576, &call_index) ||
            !parse_int(argv[4], 48, 59, &exponent) ||
            (exponent != 48 && exponent != 56 && exponent != 57 &&
             exponent != 58 && exponent != 59)) {
            std::fprintf(stderr,
                         "invalid arm-float-matmul-finite arguments\n");
            result = 2;
        } else {
            __atomic_store_n(&control->enabled, 0U, __ATOMIC_RELEASE);
            control->mode = FAULT20_MODE_FLOAT_MATMUL_FINITE_DUAL;
            control->target_device = static_cast<int32_t>(device);
            control->target_m = 1;
            control->target_k = 12800;
            control->target_x2_n = 5120;
            control->target_call_index = static_cast<uint32_t>(call_index);
            control->max_hits = 1;
            control->quant_step = 0;
            control->keep_mantissa_bits = 0;
            control->extreme_value = static_cast<int32_t>(exponent);
            control->target_dtype = 27;
            control->max_bytes = 25600;
            control->max_weight_bytes = 131072000;
            control->max_output_bytes = 10240;
            control->target_device_mask =
                1ULL << static_cast<uint32_t>(device);
            control->max_hits_per_device = 1;
            control->selection_mode = FAULT20_SELECTION_BOUNDED;
            control->target_api = FAULT20_TARGET_API_FLOAT_MATMUL;
            __atomic_store_n(&control->hit_count, 0U, __ATOMIC_RELEASE);
            __atomic_store_n(&control->candidate_count, 0U,
                             __ATOMIC_RELEASE);
            __atomic_store_n(&control->last_status, 0, __ATOMIC_RELEASE);
            reset_device_counters(control);
            ++control->generation;
            __atomic_store_n(&control->enabled, 1U, __ATOMIC_RELEASE);
            print_status(control);
        }
    } else if (argc == 5 &&
               std::strcmp(argv[1], "arm-float-matmul-cancel") == 0) {
        long long device = 0;
        long long call_index = 0;
        uint32_t pattern = 0;
        uint32_t block_size = 0;
        if (std::strcmp(argv[2], "half") == 0) {
            pattern = FAULT20_CANCELLATION_HALF;
        } else if (std::strcmp(argv[2], "block256") == 0) {
            pattern = FAULT20_CANCELLATION_BLOCK;
            block_size = 256;
        } else if (std::strcmp(argv[2], "block512") == 0) {
            pattern = FAULT20_CANCELLATION_BLOCK;
            block_size = 512;
        } else if (std::strcmp(argv[2], "block1024") == 0) {
            pattern = FAULT20_CANCELLATION_BLOCK;
            block_size = 1024;
        } else if (std::strcmp(argv[2], "alternating") == 0) {
            pattern = FAULT20_CANCELLATION_ALTERNATING;
            block_size = 1;
        }
        if (pattern == 0 || !parse_int(argv[3], 0, 63, &device) ||
            !parse_int(argv[4], 1, 1048576, &call_index)) {
            std::fprintf(stderr,
                         "invalid arm-float-matmul-cancel arguments\n");
            result = 2;
        } else {
            __atomic_store_n(&control->enabled, 0U, __ATOMIC_RELEASE);
            control->mode = FAULT20_MODE_FLOAT_MATMUL_CANCELLATION;
            control->target_device = static_cast<int32_t>(device);
            control->target_m = 1;
            control->target_k = 12800;
            control->target_x2_n = 5120;
            control->target_call_index = static_cast<uint32_t>(call_index);
            control->max_hits = 1;
            control->quant_step = block_size;
            control->keep_mantissa_bits = 0;
            control->extreme_value = 58;
            control->extreme_ratio_bps = 10000;
            control->extreme_pattern = pattern;
            control->target_dtype = 27;
            control->max_bytes = 25600;
            control->max_weight_bytes = 131072000;
            control->max_output_bytes = 10240;
            control->target_device_mask =
                1ULL << static_cast<uint32_t>(device);
            control->max_hits_per_device = 1;
            control->selection_mode = FAULT20_SELECTION_BOUNDED;
            control->target_api = FAULT20_TARGET_API_FLOAT_MATMUL;
            __atomic_store_n(&control->hit_count, 0U, __ATOMIC_RELEASE);
            __atomic_store_n(&control->candidate_count, 0U,
                             __ATOMIC_RELEASE);
            __atomic_store_n(&control->last_status, 0, __ATOMIC_RELEASE);
            reset_device_counters(control);
            ++control->generation;
            __atomic_store_n(&control->enabled, 1U, __ATOMIC_RELEASE);
            print_status(control);
        }
    } else if (argc == 6 &&
               std::strcmp(argv[1], "arm-float-matmul-output-proxy") == 0) {
        long long device = 0;
        long long call_index = 0;
        long long element_count = 0;
        uint32_t proxy_kind = 0;
        if (std::strcmp(argv[2], "pos_inf") == 0) {
            proxy_kind = FAULT20_OUTPUT_PROXY_POS_INF;
        } else if (std::strcmp(argv[2], "neg_inf") == 0) {
            proxy_kind = FAULT20_OUTPUT_PROXY_NEG_INF;
        } else if (std::strcmp(argv[2], "qnan") == 0) {
            proxy_kind = FAULT20_OUTPUT_PROXY_QNAN;
        }
        if (proxy_kind == 0 || !parse_int(argv[3], 0, 63, &device) ||
            !parse_int(argv[4], 1, 1048576, &call_index) ||
            !parse_int(argv[5], 1, 512, &element_count) ||
            (element_count != 1 && element_count != 5 &&
             element_count != 51 && element_count != 512)) {
            std::fprintf(stderr,
                         "invalid arm-float-matmul-output-proxy arguments\n");
            result = 2;
        } else {
            __atomic_store_n(&control->enabled, 0U, __ATOMIC_RELEASE);
            control->mode = FAULT20_MODE_FLOAT_MATMUL_OUTPUT_PROXY;
            control->target_device = static_cast<int32_t>(device);
            control->target_m = 1;
            control->target_k = 12800;
            control->target_x2_n = 5120;
            control->target_call_index = static_cast<uint32_t>(call_index);
            control->max_hits = 1;
            control->quant_step = static_cast<uint32_t>(element_count);
            control->keep_mantissa_bits = 0;
            control->extreme_value = 0;
            control->extreme_ratio_bps = static_cast<uint32_t>(
                (element_count * 10000 + 2560) / 5120);
            control->extreme_pattern = proxy_kind;
            control->target_dtype = 27;
            control->max_bytes = 25600;
            control->max_weight_bytes = 131072000;
            control->max_output_bytes = 10240;
            control->target_device_mask =
                1ULL << static_cast<uint32_t>(device);
            control->max_hits_per_device = 1;
            control->selection_mode = FAULT20_SELECTION_BOUNDED;
            control->target_api = FAULT20_TARGET_API_FLOAT_MATMUL;
            __atomic_store_n(&control->hit_count, 0U, __ATOMIC_RELEASE);
            __atomic_store_n(&control->candidate_count, 0U,
                             __ATOMIC_RELEASE);
            __atomic_store_n(&control->last_status, 0, __ATOMIC_RELEASE);
            reset_device_counters(control);
            ++control->generation;
            __atomic_store_n(&control->enabled, 1U, __ATOMIC_RELEASE);
            print_status(control);
        }
    } else if (argc == 4 &&
               std::strcmp(argv[1], "arm-float-matmul-observe") == 0) {
        long long device = 0;
        long long call_index = 0;
        if (!parse_int(argv[2], 0, 63, &device) ||
            !parse_int(argv[3], 1, 1048576, &call_index)) {
            std::fprintf(stderr,
                         "invalid arm-float-matmul-observe arguments\n");
            result = 2;
        } else {
            __atomic_store_n(&control->enabled, 0U, __ATOMIC_RELEASE);
            control->mode = FAULT20_MODE_FLOAT_MATMUL_OUTPUT_OBSERVE;
            control->target_device = static_cast<int32_t>(device);
            control->target_m = 1;
            control->target_k = 12800;
            control->target_x2_n = 5120;
            control->target_call_index = static_cast<uint32_t>(call_index);
            control->max_hits = 1;
            control->quant_step = 0;
            control->keep_mantissa_bits = 0;
            control->target_dtype = 27;
            control->max_bytes = 25600;
            control->max_weight_bytes = 131072000;
            control->max_output_bytes = 10240;
            control->target_device_mask =
                1ULL << static_cast<uint32_t>(device);
            control->max_hits_per_device = 1;
            control->selection_mode = FAULT20_SELECTION_BOUNDED;
            control->target_api = FAULT20_TARGET_API_FLOAT_MATMUL;
            __atomic_store_n(&control->hit_count, 0U, __ATOMIC_RELEASE);
            __atomic_store_n(&control->candidate_count, 0U,
                             __ATOMIC_RELEASE);
            __atomic_store_n(&control->last_status, 0, __ATOMIC_RELEASE);
            reset_device_counters(control);
            ++control->generation;
            __atomic_store_n(&control->enabled, 1U, __ATOMIC_RELEASE);
            print_status(control);
        }
    } else if (argc == 5 &&
               std::strcmp(argv[1], "arm-float-matmul-lossless") == 0) {
        long long device = 0;
        long long call_index = 0;
        uint32_t mode = FAULT20_MODE_DISABLED;
        if (std::strcmp(argv[2], "self") == 0) {
            mode = FAULT20_MODE_FLOAT_MATMUL_LOSSLESS_SELF;
        } else if (std::strcmp(argv[2], "mat2") == 0) {
            mode = FAULT20_MODE_FLOAT_MATMUL_LOSSLESS_MAT2;
        } else if (std::strcmp(argv[2], "dual") == 0) {
            mode = FAULT20_MODE_FLOAT_MATMUL_LOSSLESS_DUAL;
        }
        if (mode == FAULT20_MODE_DISABLED ||
            !parse_int(argv[3], 0, 63, &device) ||
            !parse_int(argv[4], 1, 1048576, &call_index)) {
            std::fprintf(stderr,
                         "invalid arm-float-matmul-lossless arguments\n");
            result = 2;
        } else {
            __atomic_store_n(&control->enabled, 0U, __ATOMIC_RELEASE);
            control->mode = mode;
            control->target_device = static_cast<int32_t>(device);
            control->target_m = 1;
            control->target_k = 12800;
            control->target_x2_n = 5120;
            control->target_call_index = static_cast<uint32_t>(call_index);
            control->max_hits = 1;
            control->quant_step = 0;
            control->keep_mantissa_bits = 0;
            control->target_dtype = 27;
            control->max_bytes = 25600;
            control->max_weight_bytes = 131072000;
            control->max_output_bytes = 10240;
            control->target_device_mask =
                1ULL << static_cast<uint32_t>(device);
            control->max_hits_per_device = 1;
            control->selection_mode = FAULT20_SELECTION_BOUNDED;
            control->target_api = FAULT20_TARGET_API_FLOAT_MATMUL;
            __atomic_store_n(&control->hit_count, 0U, __ATOMIC_RELEASE);
            __atomic_store_n(&control->candidate_count, 0U,
                             __ATOMIC_RELEASE);
            __atomic_store_n(&control->last_status, 0, __ATOMIC_RELEASE);
            reset_device_counters(control);
            ++control->generation;
            __atomic_store_n(&control->enabled, 1U, __ATOMIC_RELEASE);
            print_status(control);
        }
    } else if (argc == 7 && std::strcmp(argv[1], "arm-observe") == 0) {
        long long device = 0;
        long long m = 0;
        long long k = 0;
        long long call_index = 0;
        long long output_bytes = 0;
        if (!parse_int(argv[2], 0, 63, &device) ||
            !parse_int(argv[3], 1, 1048576, &m) ||
            !parse_int(argv[4], 1, 1048576, &k) ||
            !parse_int(argv[5], 1, 1048576, &call_index) ||
            !parse_int(argv[6], 2, 1048576, &output_bytes) ||
            static_cast<unsigned long long>(m) *
                    static_cast<unsigned long long>(k) >
                65536ULL) {
            std::fprintf(stderr, "invalid arm-observe arguments\n");
            result = 2;
        } else {
            __atomic_store_n(&control->enabled, 0U, __ATOMIC_RELEASE);
            control->mode = FAULT20_MODE_QMM_OUTPUT_OBSERVE;
            control->target_device = static_cast<int32_t>(device);
            control->target_m = static_cast<int32_t>(m);
            control->target_k = static_cast<int32_t>(k);
            control->target_call_index = static_cast<uint32_t>(call_index);
            control->max_hits = 1;
            control->quant_step = 0;
            control->keep_mantissa_bits = 0;
            control->target_dtype = 2;
            control->max_bytes = static_cast<uint64_t>(m * k);
            control->max_output_bytes = static_cast<uint64_t>(output_bytes);
            control->target_device_mask = 1ULL << static_cast<uint32_t>(device);
            control->max_hits_per_device = 1;
            control->selection_mode = FAULT20_SELECTION_BOUNDED;
            control->target_api = FAULT20_TARGET_API_QMM;
            __atomic_store_n(&control->hit_count, 0U, __ATOMIC_RELEASE);
            __atomic_store_n(&control->candidate_count, 0U, __ATOMIC_RELEASE);
            __atomic_store_n(&control->last_status, 0, __ATOMIC_RELEASE);
            reset_device_counters(control);
            ++control->generation;
            __atomic_store_n(&control->enabled, 1U, __ATOMIC_RELEASE);
            print_status(control);
        }
    } else if (argc == 7 && std::strcmp(argv[1], "arm-observe-x2") == 0) {
        long long device = 0;
        long long m = 0;
        long long k = 0;
        long long call_index = 0;
        long long weight_bytes = 0;
        if (!parse_int(argv[2], 0, 63, &device) ||
            !parse_int(argv[3], 1, 1048576, &m) ||
            !parse_int(argv[4], 1, 1048576, &k) ||
            !parse_int(argv[5], 1, 1048576, &call_index) ||
            !parse_int(argv[6], 1, 1073741824, &weight_bytes) ||
            static_cast<unsigned long long>(m) *
                    static_cast<unsigned long long>(k) >
                65536ULL) {
            std::fprintf(stderr, "invalid arm-observe-x2 arguments\n");
            result = 2;
        } else {
            __atomic_store_n(&control->enabled, 0U, __ATOMIC_RELEASE);
            control->mode = FAULT20_MODE_QMM_WEIGHT_OBSERVE;
            control->target_device = static_cast<int32_t>(device);
            control->target_m = static_cast<int32_t>(m);
            control->target_k = static_cast<int32_t>(k);
            control->target_call_index = static_cast<uint32_t>(call_index);
            control->max_hits = 1;
            control->quant_step = 0;
            control->keep_mantissa_bits = 0;
            control->target_dtype = 2;
            control->max_bytes = static_cast<uint64_t>(m * k);
            control->max_output_bytes = 65536;
            control->max_weight_bytes = static_cast<uint64_t>(weight_bytes);
            control->target_x2_n = 0;
            control->target_device_mask = 1ULL << static_cast<uint32_t>(device);
            control->max_hits_per_device = 1;
            control->selection_mode = FAULT20_SELECTION_BOUNDED;
            control->target_api = FAULT20_TARGET_API_QMM;
            __atomic_store_n(&control->hit_count, 0U, __ATOMIC_RELEASE);
            __atomic_store_n(&control->candidate_count, 0U, __ATOMIC_RELEASE);
            __atomic_store_n(&control->last_status, 0, __ATOMIC_RELEASE);
            reset_device_counters(control);
            ++control->generation;
            __atomic_store_n(&control->enabled, 1U, __ATOMIC_RELEASE);
            print_status(control);
        }
    } else if (argc == 8 &&
               std::strcmp(argv[1], "arm-observe-x2-scale") == 0) {
        long long device = 0;
        long long m = 0;
        long long k = 0;
        long long scale_elements = 0;
        long long call_index = 0;
        long long scale_bytes = 0;
        if (!parse_int(argv[2], 0, 63, &device) ||
            !parse_int(argv[3], 1, 1048576, &m) ||
            !parse_int(argv[4], 1, 1048576, &k) ||
            !parse_int(argv[5], 1, 1048576, &scale_elements) ||
            !parse_int(argv[6], 1, 1048576, &call_index) ||
            !parse_int(argv[7], 4, 4194304, &scale_bytes) ||
            static_cast<unsigned long long>(m) *
                    static_cast<unsigned long long>(k) >
                65536ULL ||
            static_cast<unsigned long long>(scale_elements) * 4ULL !=
                static_cast<unsigned long long>(scale_bytes)) {
            std::fprintf(stderr, "invalid arm-observe-x2-scale arguments\n");
            result = 2;
        } else {
            __atomic_store_n(&control->enabled, 0U, __ATOMIC_RELEASE);
            control->mode = FAULT20_MODE_QMM_X2_SCALE_OBSERVE;
            control->target_device = static_cast<int32_t>(device);
            control->target_m = static_cast<int32_t>(m);
            control->target_k = static_cast<int32_t>(k);
            control->target_scale_elements = static_cast<int32_t>(scale_elements);
            control->target_call_index = static_cast<uint32_t>(call_index);
            control->max_hits = 1;
            control->quant_step = 0;
            control->keep_mantissa_bits = 0;
            control->target_dtype = 0;
            control->max_bytes = static_cast<uint64_t>(m * k);
            control->max_output_bytes = 65536;
            control->max_scale_bytes = static_cast<uint64_t>(scale_bytes);
            control->scale_factor = 1.0;
            control->target_device_mask = 1ULL << static_cast<uint32_t>(device);
            control->max_hits_per_device = 1;
            control->selection_mode = FAULT20_SELECTION_BOUNDED;
            control->target_api = FAULT20_TARGET_API_QMM;
            __atomic_store_n(&control->hit_count, 0U, __ATOMIC_RELEASE);
            __atomic_store_n(&control->candidate_count, 0U, __ATOMIC_RELEASE);
            __atomic_store_n(&control->last_status, 0, __ATOMIC_RELEASE);
            reset_device_counters(control);
            ++control->generation;
            __atomic_store_n(&control->enabled, 1U, __ATOMIC_RELEASE);
            print_status(control);
        }
    } else if (argc == 8 &&
               std::strcmp(argv[1], "arm-lossless-x2-scale") == 0) {
        long long device = 0;
        long long m = 0;
        long long k = 0;
        long long scale_elements = 0;
        long long call_index = 0;
        long long scale_bytes = 0;
        if (!parse_int(argv[2], 0, 63, &device) ||
            !parse_int(argv[3], 1, 1048576, &m) ||
            !parse_int(argv[4], 1, 1048576, &k) ||
            !parse_int(argv[5], 1, 1048576, &scale_elements) ||
            !parse_int(argv[6], 1, 1048576, &call_index) ||
            !parse_int(argv[7], 4, 4194304, &scale_bytes) ||
            static_cast<unsigned long long>(m) *
                    static_cast<unsigned long long>(k) >
                65536ULL ||
            static_cast<unsigned long long>(scale_elements) * 4ULL !=
                static_cast<unsigned long long>(scale_bytes)) {
            std::fprintf(stderr, "invalid arm-lossless-x2-scale arguments\n");
            result = 2;
        } else {
            __atomic_store_n(&control->enabled, 0U, __ATOMIC_RELEASE);
            control->mode = FAULT20_MODE_QMM_X2_SCALE_LOSSLESS;
            control->target_device = static_cast<int32_t>(device);
            control->target_m = static_cast<int32_t>(m);
            control->target_k = static_cast<int32_t>(k);
            control->target_scale_elements = static_cast<int32_t>(scale_elements);
            control->target_call_index = static_cast<uint32_t>(call_index);
            control->max_hits = 1;
            control->quant_step = 0;
            control->keep_mantissa_bits = 0;
            control->target_dtype = 0;
            control->max_bytes = static_cast<uint64_t>(m * k);
            control->max_output_bytes = 65536;
            control->max_scale_bytes = static_cast<uint64_t>(scale_bytes);
            control->scale_factor = 1.0;
            control->target_device_mask = 1ULL << static_cast<uint32_t>(device);
            control->max_hits_per_device = 1;
            control->selection_mode = FAULT20_SELECTION_BOUNDED;
            control->target_api = FAULT20_TARGET_API_QMM;
            __atomic_store_n(&control->hit_count, 0U, __ATOMIC_RELEASE);
            __atomic_store_n(&control->candidate_count, 0U, __ATOMIC_RELEASE);
            __atomic_store_n(&control->last_status, 0, __ATOMIC_RELEASE);
            reset_device_counters(control);
            ++control->generation;
            __atomic_store_n(&control->enabled, 1U, __ATOMIC_RELEASE);
            print_status(control);
        }
    } else if (argc == 9 &&
               std::strcmp(argv[1], "arm-amplify-x2-scale") == 0) {
        long long device = 0;
        long long m = 0;
        long long k = 0;
        long long scale_elements = 0;
        long long call_index = 0;
        long long scale_bytes = 0;
        long long factor = 0;
        if (!parse_int(argv[2], 0, 63, &device) ||
            !parse_int(argv[3], 1, 1048576, &m) ||
            !parse_int(argv[4], 1, 1048576, &k) ||
            !parse_int(argv[5], 1, 1048576, &scale_elements) ||
            !parse_int(argv[6], 1, 1048576, &call_index) ||
            !parse_int(argv[7], 4, 4194304, &scale_bytes) ||
            !parse_int(argv[8], 2, 1000, &factor) ||
            (factor != 2 && factor != 10 && factor != 100 && factor != 1000) ||
            static_cast<unsigned long long>(m) *
                    static_cast<unsigned long long>(k) >
                65536ULL ||
            static_cast<unsigned long long>(scale_elements) * 4ULL !=
                static_cast<unsigned long long>(scale_bytes)) {
            std::fprintf(stderr, "invalid arm-amplify-x2-scale arguments\n");
            result = 2;
        } else {
            __atomic_store_n(&control->enabled, 0U, __ATOMIC_RELEASE);
            control->mode = FAULT20_MODE_QMM_X2_SCALE_AMPLIFY;
            control->target_device = static_cast<int32_t>(device);
            control->target_m = static_cast<int32_t>(m);
            control->target_k = static_cast<int32_t>(k);
            control->target_scale_elements = static_cast<int32_t>(scale_elements);
            control->target_call_index = static_cast<uint32_t>(call_index);
            control->max_hits = 1;
            control->quant_step = 0;
            control->keep_mantissa_bits = 0;
            control->target_dtype = 0;
            control->max_bytes = static_cast<uint64_t>(m * k);
            control->max_output_bytes = 65536;
            control->max_scale_bytes = static_cast<uint64_t>(scale_bytes);
            control->scale_factor = static_cast<double>(factor);
            control->target_device_mask = 1ULL << static_cast<uint32_t>(device);
            control->max_hits_per_device = 1;
            control->selection_mode = FAULT20_SELECTION_BOUNDED;
            control->target_api = FAULT20_TARGET_API_QMM;
            __atomic_store_n(&control->hit_count, 0U, __ATOMIC_RELEASE);
            __atomic_store_n(&control->candidate_count, 0U, __ATOMIC_RELEASE);
            __atomic_store_n(&control->last_status, 0, __ATOMIC_RELEASE);
            reset_device_counters(control);
            ++control->generation;
            __atomic_store_n(&control->enabled, 1U, __ATOMIC_RELEASE);
            print_status(control);
        }
    } else if (argc == 9 &&
               std::strcmp(argv[1], "arm-lossless-x1-x2") == 0) {
        long long device = 0;
        long long m = 0;
        long long k = 0;
        long long x2_n = 0;
        long long call_index = 0;
        long long x1_bytes = 0;
        long long weight_bytes = 0;
        if (!parse_int(argv[2], 0, 63, &device) ||
            !parse_int(argv[3], 1, 1048576, &m) ||
            !parse_int(argv[4], 1, 1048576, &k) ||
            !parse_int(argv[5], 1, 1048576, &x2_n) ||
            !parse_int(argv[6], 1, 1048576, &call_index) ||
            !parse_int(argv[7], 1, 65536, &x1_bytes) ||
            !parse_int(argv[8], 1, 1073741824, &weight_bytes) ||
            static_cast<unsigned long long>(m) *
                    static_cast<unsigned long long>(k) !=
                static_cast<unsigned long long>(x1_bytes) ||
            static_cast<unsigned long long>(k) *
                    static_cast<unsigned long long>(x2_n) !=
                static_cast<unsigned long long>(weight_bytes)) {
            std::fprintf(stderr, "invalid arm-lossless-x1-x2 arguments\n");
            result = 2;
        } else {
            __atomic_store_n(&control->enabled, 0U, __ATOMIC_RELEASE);
            control->mode = FAULT20_MODE_QMM_LOSSLESS_X1_X2;
            control->target_device = static_cast<int32_t>(device);
            control->target_m = static_cast<int32_t>(m);
            control->target_k = static_cast<int32_t>(k);
            control->target_x2_n = static_cast<int32_t>(x2_n);
            control->target_call_index = static_cast<uint32_t>(call_index);
            control->max_hits = 1;
            control->quant_step = 0;
            control->keep_mantissa_bits = 0;
            control->target_dtype = 2;
            control->max_bytes = static_cast<uint64_t>(x1_bytes);
            control->max_output_bytes = 65536;
            control->max_weight_bytes = static_cast<uint64_t>(weight_bytes);
            control->target_device_mask = 1ULL << static_cast<uint32_t>(device);
            control->max_hits_per_device = 1;
            control->selection_mode = FAULT20_SELECTION_BOUNDED;
            control->target_api = FAULT20_TARGET_API_QMM;
            __atomic_store_n(&control->hit_count, 0U, __ATOMIC_RELEASE);
            __atomic_store_n(&control->candidate_count, 0U, __ATOMIC_RELEASE);
            __atomic_store_n(&control->last_status, 0, __ATOMIC_RELEASE);
            reset_device_counters(control);
            ++control->generation;
            __atomic_store_n(&control->enabled, 1U, __ATOMIC_RELEASE);
            print_status(control);
        }
    } else if (argc == 11 &&
               std::strcmp(argv[1], "arm-extreme-x1-x2") == 0) {
        long long device = 0;
        long long m = 0;
        long long k = 0;
        long long x2_n = 0;
        long long call_index = 0;
        long long x1_bytes = 0;
        long long weight_bytes = 0;
        long long value = 0;
        const uint32_t pattern = parse_extreme_pattern(argv[10]);
        if (!parse_int(argv[2], 0, 63, &device) ||
            !parse_int(argv[3], 1, 1048576, &m) ||
            !parse_int(argv[4], 1, 1048576, &k) ||
            !parse_int(argv[5], 1, 1048576, &x2_n) ||
            !parse_int(argv[6], 1, 1048576, &call_index) ||
            !parse_int(argv[7], 1, 65536, &x1_bytes) ||
            !parse_int(argv[8], 1, 1073741824, &weight_bytes) ||
            !parse_int(argv[9], -128, 127, &value) ||
            pattern != FAULT20_EXTREME_POSITIVE_MAX || value != 127 ||
            static_cast<unsigned long long>(m) *
                    static_cast<unsigned long long>(k) !=
                static_cast<unsigned long long>(x1_bytes) ||
            static_cast<unsigned long long>(k) *
                    static_cast<unsigned long long>(x2_n) !=
                static_cast<unsigned long long>(weight_bytes)) {
            std::fprintf(stderr, "invalid arm-extreme-x1-x2 arguments\n");
            result = 2;
        } else {
            __atomic_store_n(&control->enabled, 0U, __ATOMIC_RELEASE);
            control->mode = FAULT20_MODE_QMM_EXTREME_X1_X2;
            control->target_device = static_cast<int32_t>(device);
            control->target_m = static_cast<int32_t>(m);
            control->target_k = static_cast<int32_t>(k);
            control->target_x2_n = static_cast<int32_t>(x2_n);
            control->target_call_index = static_cast<uint32_t>(call_index);
            control->max_hits = 1;
            control->quant_step = 0;
            control->keep_mantissa_bits = 0;
            control->extreme_value = static_cast<int32_t>(value);
            control->extreme_ratio_bps = 10000;
            control->extreme_pattern = pattern;
            control->target_dtype = 2;
            control->max_bytes = static_cast<uint64_t>(x1_bytes);
            control->max_output_bytes = 65536;
            control->max_weight_bytes = static_cast<uint64_t>(weight_bytes);
            control->target_device_mask = 1ULL << static_cast<uint32_t>(device);
            control->max_hits_per_device = 1;
            control->selection_mode = FAULT20_SELECTION_BOUNDED;
            control->target_api = FAULT20_TARGET_API_QMM;
            __atomic_store_n(&control->hit_count, 0U, __ATOMIC_RELEASE);
            __atomic_store_n(&control->candidate_count, 0U, __ATOMIC_RELEASE);
            __atomic_store_n(&control->last_status, 0, __ATOMIC_RELEASE);
            reset_device_counters(control);
            ++control->generation;
            __atomic_store_n(&control->enabled, 1U, __ATOMIC_RELEASE);
            print_status(control);
        }
    } else if (argc == 6 && std::strcmp(argv[1], "arm-lossless") == 0) {
        long long device = 0;
        long long k = 0;
        long long hits = 0;
        long long bytes = 0;
        if (!parse_int(argv[2], 0, 63, &device) || !parse_int(argv[3], 1, 1048576, &k) ||
            !parse_int(argv[4], 1, 1024, &hits) || !parse_int(argv[5], 1, 65536, &bytes)) {
            std::fprintf(stderr, "invalid arm-lossless arguments\n");
            result = 2;
        } else {
            __atomic_store_n(&control->enabled, 0U, __ATOMIC_RELEASE);
            control->mode = FAULT20_MODE_LOSSLESS_D2D;
            control->target_device = static_cast<int32_t>(device);
            control->target_m = 1;
            control->target_k = static_cast<int32_t>(k);
            control->target_call_index = 1;
            control->max_hits = static_cast<uint32_t>(hits);
            control->quant_step = 16;
            control->keep_mantissa_bits = 3;
            control->target_dtype = 27;
            control->max_bytes = static_cast<uint64_t>(bytes);
            control->target_device_mask = 1ULL << static_cast<uint32_t>(device);
            control->max_hits_per_device = static_cast<uint32_t>(hits);
            control->selection_mode = FAULT20_SELECTION_BOUNDED;
            control->target_api = FAULT20_TARGET_API_QMM;
            __atomic_store_n(&control->hit_count, 0U, __ATOMIC_RELEASE);
            __atomic_store_n(&control->candidate_count, 0U, __ATOMIC_RELEASE);
            __atomic_store_n(&control->last_status, 0, __ATOMIC_RELEASE);
            reset_device_counters(control);
            ++control->generation;
            __atomic_store_n(&control->enabled, 1U, __ATOMIC_RELEASE);
            print_status(control);
        }
    } else if ((argc == 8 || argc == 9) &&
               (std::strcmp(argv[1], "arm-int4") == 0 ||
                std::strcmp(argv[1], "arm-int4-observe") == 0 ||
                std::strcmp(argv[1], "arm-int4-slotproof") == 0)) {
        long long device = 0;
        long long m = 0;
        long long k = 0;
        long long call_index = 0;
        long long hits = 0;
        long long bytes = 0;
        long long output_bytes = 65536;
        const bool observe =
            std::strcmp(argv[1], "arm-int4-observe") == 0;
        const bool slotproof =
            std::strcmp(argv[1], "arm-int4-slotproof") == 0;
        if (!parse_int(argv[2], 0, 63, &device) ||
            !parse_int(argv[3], 1, 1048576, &m) ||
            !parse_int(argv[4], 1, 1048576, &k) ||
            !parse_int(argv[5], 1, 1048576, &call_index) ||
            !parse_int(argv[6], 1, 1024, &hits) ||
            !parse_int(argv[7], 1, 65536, &bytes) ||
            (!observe && argc != 8) ||
            (observe && argc == 9 &&
             !parse_int(argv[8], 1, 268435456, &output_bytes))) {
            std::fprintf(stderr, "invalid arm-int4 numeric arguments\n");
            result = 2;
        } else {
            __atomic_store_n(&control->enabled, 0U, __ATOMIC_RELEASE);
            control->mode = observe ? FAULT20_MODE_INT4_EQUIVALENT_OBSERVE
                                    : FAULT20_MODE_INT4_EQUIVALENT;
            control->target_device = static_cast<int32_t>(device);
            control->target_m = static_cast<int32_t>(m);
            control->target_k = static_cast<int32_t>(k);
            control->target_call_index = static_cast<uint32_t>(call_index);
            control->max_hits = static_cast<uint32_t>(hits);
            control->quant_step = 16;
            control->keep_mantissa_bits = 0;
            control->target_dtype = 2;
            control->max_bytes = static_cast<uint64_t>(bytes);
            control->max_output_bytes = static_cast<uint64_t>(output_bytes);
            control->target_device_mask = 1ULL << static_cast<uint32_t>(device);
            control->max_hits_per_device = static_cast<uint32_t>(hits);
            control->selection_mode = FAULT20_SELECTION_BOUNDED;
            control->target_api = FAULT20_TARGET_API_QMM;
            __atomic_store_n(&control->hit_count, 0U, __ATOMIC_RELEASE);
            __atomic_store_n(&control->candidate_count, 0U, __ATOMIC_RELEASE);
            __atomic_store_n(&control->last_status, 0, __ATOMIC_RELEASE);
            reset_device_counters(control);
            control->proof_mode = slotproof ? FAULT20_PROOF_MODE_SEMANTIC_SLOT
                                            : FAULT20_PROOF_MODE_STRICT_PRODUCER;
            ++control->generation;
            __atomic_store_n(&control->enabled, 1U, __ATOMIC_RELEASE);
            print_status(control);
        }
    } else if ((argc == 8 || argc == 9) &&
               (std::strcmp(argv[1], "arm-int2") == 0 ||
                std::strcmp(argv[1], "arm-int2-observe") == 0)) {
        long long device = 0;
        long long m = 0;
        long long k = 0;
        long long call_index = 0;
        long long hits = 0;
        long long bytes = 0;
        long long output_bytes = 65536;
        const bool observe =
            std::strcmp(argv[1], "arm-int2-observe") == 0;
        if (!parse_int(argv[2], 0, 63, &device) ||
            !parse_int(argv[3], 1, 1048576, &m) ||
            !parse_int(argv[4], 1, 1048576, &k) ||
            !parse_int(argv[5], 1, 1048576, &call_index) ||
            !parse_int(argv[6], 1, 1024, &hits) ||
            !parse_int(argv[7], 1, 65536, &bytes) ||
            (!observe && argc != 8) ||
            (observe && argc == 9 &&
             !parse_int(argv[8], 1, 268435456, &output_bytes))) {
            std::fprintf(stderr, "invalid arm-int2 numeric arguments\n");
            result = 2;
        } else {
            __atomic_store_n(&control->enabled, 0U, __ATOMIC_RELEASE);
            control->mode = observe ? FAULT20_MODE_INT2_EQUIVALENT_OBSERVE
                                    : FAULT20_MODE_INT2_EQUIVALENT;
            control->target_device = static_cast<int32_t>(device);
            control->target_m = static_cast<int32_t>(m);
            control->target_k = static_cast<int32_t>(k);
            control->target_call_index = static_cast<uint32_t>(call_index);
            control->max_hits = static_cast<uint32_t>(hits);
            control->quant_step = 64;
            control->keep_mantissa_bits = 0;
            control->target_dtype = 2;
            control->max_bytes = static_cast<uint64_t>(bytes);
            control->max_output_bytes = static_cast<uint64_t>(output_bytes);
            control->target_device_mask = 1ULL << static_cast<uint32_t>(device);
            control->max_hits_per_device = static_cast<uint32_t>(hits);
            control->selection_mode = FAULT20_SELECTION_BOUNDED;
            control->target_api = FAULT20_TARGET_API_QMM;
            __atomic_store_n(&control->hit_count, 0U, __ATOMIC_RELEASE);
            __atomic_store_n(&control->candidate_count, 0U, __ATOMIC_RELEASE);
            __atomic_store_n(&control->last_status, 0, __ATOMIC_RELEASE);
            reset_device_counters(control);
            ++control->generation;
            __atomic_store_n(&control->enabled, 1U, __ATOMIC_RELEASE);
            print_status(control);
        }
    } else if ((argc == 9 || argc == 10) &&
               (std::strcmp(argv[1], "arm-int4-request") == 0 ||
                std::strcmp(argv[1], "arm-int2-request") == 0 ||
                std::strcmp(argv[1], "arm-int4-observe-request") == 0 ||
                std::strcmp(argv[1], "arm-int2-observe-request") == 0)) {
        long long device = 0;
        long long m = 0;
        long long k = 0;
        long long call_index = 0;
        long long hits_per_request = 0;
        long long max_hits_total = 0;
        long long bytes = 0;
        long long output_bytes = 65536;
        const bool observe = std::strstr(argv[1], "-observe-") != nullptr;
        if (!parse_int(argv[2], 0, 63, &device) ||
            !parse_int(argv[3], 1, 1048576, &m) ||
            !parse_int(argv[4], 1, 1048576, &k) ||
            !parse_int(argv[5], 1, 1048576, &call_index) ||
            !parse_int(argv[6], 1, 1024, &hits_per_request) ||
            !parse_int(argv[7], hits_per_request, 1048576, &max_hits_total) ||
            !parse_int(argv[8], 1, 65536, &bytes) ||
            (!observe && argc != 9) ||
            (observe && argc == 10 &&
             !parse_int(argv[9], 1, 268435456, &output_bytes))) {
            std::fprintf(stderr, "invalid request-scoped numeric arguments\n");
            result = 2;
        } else {
            const bool int4 = std::strncmp(argv[1], "arm-int4", 8) == 0;
            __atomic_store_n(&control->enabled, 0U, __ATOMIC_RELEASE);
            control->mode = int4 ? (observe ? FAULT20_MODE_INT4_EQUIVALENT_OBSERVE
                                            : FAULT20_MODE_INT4_EQUIVALENT)
                                 : (observe ? FAULT20_MODE_INT2_EQUIVALENT_OBSERVE
                                            : FAULT20_MODE_INT2_EQUIVALENT);
            control->target_device = static_cast<int32_t>(device);
            control->target_m = static_cast<int32_t>(m);
            control->target_k = static_cast<int32_t>(k);
            control->target_call_index = static_cast<uint32_t>(call_index);
            control->max_hits = static_cast<uint32_t>(hits_per_request);
            control->quant_step = int4 ? 16U : 64U;
            control->keep_mantissa_bits = 0;
            control->target_dtype = 2;
            control->max_bytes = static_cast<uint64_t>(bytes);
            control->max_output_bytes = static_cast<uint64_t>(output_bytes);
            control->target_device_mask = 1ULL << static_cast<uint32_t>(device);
            control->max_hits_per_device = static_cast<uint32_t>(hits_per_request);
            control->selection_mode = FAULT20_SELECTION_BOUNDED;
            control->target_api = FAULT20_TARGET_API_QMM;
            __atomic_store_n(&control->hit_count, 0U, __ATOMIC_RELEASE);
            __atomic_store_n(&control->candidate_count, 0U, __ATOMIC_RELEASE);
            __atomic_store_n(&control->last_status, 0, __ATOMIC_RELEASE);
            reset_device_counters(control);
            control->hit_scope = FAULT20_HIT_SCOPE_REQUEST;
            control->max_hits_total = static_cast<uint32_t>(max_hits_total);
            ++control->generation;
            __atomic_store_n(&control->enabled, 1U, __ATOMIC_RELEASE);
            print_status(control);
        }
    } else if (argc == 10 && std::strcmp(argv[1], "arm-extreme") == 0) {
        long long device = 0;
        long long m = 0;
        long long k = 0;
        long long call_index = 0;
        long long bytes = 0;
        long long value = 0;
        long long ratio_bps = 0;
        const uint32_t pattern = parse_extreme_pattern(argv[9]);
        if (!parse_int(argv[2], 0, 63, &device) ||
            !parse_int(argv[3], 1, 1048576, &m) ||
            !parse_int(argv[4], 1, 1048576, &k) ||
            !parse_int(argv[5], 1, 1048576, &call_index) ||
            !parse_int(argv[6], 1, 65536, &bytes) ||
            !parse_int(argv[7], -128, 127, &value) ||
            !parse_int(argv[8], 1, 10000, &ratio_bps) || pattern == 0 ||
            (pattern == FAULT20_EXTREME_POSITIVE_MAX && value != 127) ||
            (pattern == FAULT20_EXTREME_NEGATIVE_MAX && value != -128) ||
            (pattern == FAULT20_EXTREME_ALTERNATING && value != 127 && value != -128)) {
            std::fprintf(stderr, "invalid arm-extreme arguments\n");
            result = 2;
        } else {
            __atomic_store_n(&control->enabled, 0U, __ATOMIC_RELEASE);
            control->mode = FAULT20_MODE_QMM_EXTREME_X1;
            control->target_device = static_cast<int32_t>(device);
            control->target_m = static_cast<int32_t>(m);
            control->target_k = static_cast<int32_t>(k);
            control->target_call_index = static_cast<uint32_t>(call_index);
            control->max_hits = 1;
            control->quant_step = 0;
            control->keep_mantissa_bits = 0;
            control->extreme_value = static_cast<int32_t>(value);
            control->extreme_ratio_bps = static_cast<uint32_t>(ratio_bps);
            control->extreme_pattern = pattern;
            control->target_dtype = 2;
            control->max_bytes = static_cast<uint64_t>(bytes);
            control->max_output_bytes = 65536;
            control->target_device_mask = 1ULL << static_cast<uint32_t>(device);
            control->max_hits_per_device = 1;
            control->selection_mode = FAULT20_SELECTION_BOUNDED;
            control->target_api = FAULT20_TARGET_API_QMM;
            __atomic_store_n(&control->hit_count, 0U, __ATOMIC_RELEASE);
            __atomic_store_n(&control->candidate_count, 0U, __ATOMIC_RELEASE);
            __atomic_store_n(&control->last_status, 0, __ATOMIC_RELEASE);
            reset_device_counters(control);
            ++control->generation;
            __atomic_store_n(&control->enabled, 1U, __ATOMIC_RELEASE);
            print_status(control);
        }
    } else if (argc == 11 &&
               std::strcmp(argv[1], "arm-progressive-extreme") == 0) {
        long long device = 0;
        long long m = 0;
        long long k = 0;
        long long call_index = 0;
        long long hits = 0;
        long long bytes = 0;
        long long value = 0;
        long long ratio_bps = 0;
        const uint32_t pattern = parse_extreme_pattern(argv[10]);
        if (!parse_int(argv[2], 0, 63, &device) ||
            !parse_int(argv[3], 1, 1048576, &m) ||
            !parse_int(argv[4], 1, 1048576, &k) ||
            !parse_int(argv[5], 1, 1048576, &call_index) ||
            !parse_int(argv[6], 2, 64, &hits) ||
            !parse_int(argv[7], 1, 65536, &bytes) ||
            !parse_int(argv[8], -128, 127, &value) ||
            !parse_int(argv[9], 1, 10000, &ratio_bps) || pattern == 0 ||
            (pattern == FAULT20_EXTREME_POSITIVE_MAX && value != 127) ||
            (pattern == FAULT20_EXTREME_NEGATIVE_MAX && value != -128) ||
            (pattern == FAULT20_EXTREME_ALTERNATING && value != 127 &&
             value != -128)) {
            std::fprintf(stderr, "invalid arm-progressive-extreme arguments\n");
            result = 2;
        } else {
            __atomic_store_n(&control->enabled, 0U, __ATOMIC_RELEASE);
            control->mode = FAULT20_MODE_QMM_EXTREME_X1;
            control->target_device = static_cast<int32_t>(device);
            control->target_m = static_cast<int32_t>(m);
            control->target_k = static_cast<int32_t>(k);
            control->target_call_index = static_cast<uint32_t>(call_index);
            control->max_hits = static_cast<uint32_t>(hits);
            control->quant_step = 0;
            control->keep_mantissa_bits = 0;
            control->extreme_value = static_cast<int32_t>(value);
            control->extreme_ratio_bps = static_cast<uint32_t>(ratio_bps);
            control->extreme_pattern = pattern;
            control->target_dtype = 2;
            control->max_bytes = static_cast<uint64_t>(bytes);
            control->max_output_bytes = 65536;
            control->target_device_mask = 1ULL << static_cast<uint32_t>(device);
            control->max_hits_per_device = static_cast<uint32_t>(hits);
            control->selection_mode = FAULT20_SELECTION_BOUNDED;
            control->target_api = FAULT20_TARGET_API_QMM;
            __atomic_store_n(&control->hit_count, 0U, __ATOMIC_RELEASE);
            __atomic_store_n(&control->candidate_count, 0U, __ATOMIC_RELEASE);
            __atomic_store_n(&control->last_status, 0, __ATOMIC_RELEASE);
            reset_device_counters(control);
            ++control->generation;
            __atomic_store_n(&control->enabled, 1U, __ATOMIC_RELEASE);
            print_status(control);
        }
    } else if (argc == 9 &&
               std::strcmp(argv[1], "arm-all-qmm-extreme") == 0) {
        long long device_mask = 0;
        long long m = 0;
        long long k = 0;
        long long bytes = 0;
        long long value = 0;
        long long ratio_bps = 0;
        const uint32_t pattern = parse_extreme_pattern(argv[8]);
        if (!parse_int(argv[2], 1, 0x7FFFFFFFFFFFFFFFLL, &device_mask) ||
            !parse_int(argv[3], 1, 1048576, &m) ||
            !parse_int(argv[4], 1, 1048576, &k) ||
            !parse_int(argv[5], 1, 65536, &bytes) ||
            !parse_int(argv[6], -128, 127, &value) ||
            !parse_int(argv[7], 1, 10000, &ratio_bps) || pattern == 0 ||
            (pattern == FAULT20_EXTREME_POSITIVE_MAX && value != 127) ||
            (pattern == FAULT20_EXTREME_NEGATIVE_MAX && value != -128) ||
            (pattern == FAULT20_EXTREME_ALTERNATING && value != 127 &&
             value != -128)) {
            std::fprintf(stderr, "invalid arm-all-qmm-extreme arguments\n");
            result = 2;
        } else {
            const uint64_t mask = static_cast<uint64_t>(device_mask);
            __atomic_store_n(&control->enabled, 0U, __ATOMIC_RELEASE);
            control->mode = FAULT20_MODE_QMM_EXTREME_X1;
            control->target_device = -1;
            control->target_m = static_cast<int32_t>(m);
            control->target_k = static_cast<int32_t>(k);
            control->target_call_index = 1;
            control->max_hits = FAULT20_ALL_HITS_LIMIT;
            control->quant_step = 0;
            control->keep_mantissa_bits = 0;
            control->extreme_value = static_cast<int32_t>(value);
            control->extreme_ratio_bps = static_cast<uint32_t>(ratio_bps);
            control->extreme_pattern = pattern;
            control->target_dtype = 2;
            control->max_bytes = static_cast<uint64_t>(bytes);
            control->max_output_bytes = 65536;
            control->target_device_mask = mask;
            control->max_hits_per_device = FAULT20_ALL_HITS_LIMIT;
            control->selection_mode = FAULT20_SELECTION_ALL_HITS;
            control->target_api = FAULT20_TARGET_API_QMM;
            __atomic_store_n(&control->hit_count, 0U, __ATOMIC_RELEASE);
            __atomic_store_n(&control->candidate_count, 0U, __ATOMIC_RELEASE);
            __atomic_store_n(&control->last_status, 0, __ATOMIC_RELEASE);
            reset_device_counters(control);
            ++control->generation;
            __atomic_store_n(&control->enabled, 1U, __ATOMIC_RELEASE);
            print_status(control);
        }
    } else if (argc == 10 && std::strcmp(argv[1], "arm-mantissa") == 0) {
        long long device = 0;
        long long m = 0;
        long long k = 0;
        long long call_index = 0;
        long long hits = 0;
        long long bytes = 0;
        long long keep_bits = 0;
        const bool is_bf16 = std::strcmp(argv[8], "bf16") == 0;
        const bool is_fp16 = std::strcmp(argv[8], "fp16") == 0;
        if (!parse_int(argv[2], 0, 63, &device) ||
            !parse_int(argv[3], 1, 1048576, &m) ||
            !parse_int(argv[4], 1, 1048576, &k) ||
            !parse_int(argv[5], 1, 1048576, &call_index) ||
            !parse_int(argv[6], 1, 1024, &hits) ||
            !parse_int(argv[7], 1, 65536, &bytes) || (!is_bf16 && !is_fp16) ||
            !parse_int(argv[9], 0, is_bf16 ? 7 : 10, &keep_bits)) {
            std::fprintf(stderr, "invalid arm-mantissa arguments\n");
            result = 2;
        } else {
            __atomic_store_n(&control->enabled, 0U, __ATOMIC_RELEASE);
            control->mode = FAULT20_MODE_MANTISSA_TRUNCATE;
            control->target_device = static_cast<int32_t>(device);
            control->target_m = static_cast<int32_t>(m);
            control->target_k = static_cast<int32_t>(k);
            control->target_call_index = static_cast<uint32_t>(call_index);
            control->max_hits = static_cast<uint32_t>(hits);
            control->quant_step = 0;
            control->keep_mantissa_bits = static_cast<uint32_t>(keep_bits);
            control->target_dtype = is_bf16 ? 27 : 1;
            control->max_bytes = static_cast<uint64_t>(bytes);
            control->target_device_mask = 1ULL << static_cast<uint32_t>(device);
            control->max_hits_per_device = static_cast<uint32_t>(hits);
            control->selection_mode = FAULT20_SELECTION_BOUNDED;
            control->target_api = FAULT20_TARGET_API_MATMUL_WEIGHT_NZ;
            __atomic_store_n(&control->hit_count, 0U, __ATOMIC_RELEASE);
            __atomic_store_n(&control->candidate_count, 0U, __ATOMIC_RELEASE);
            __atomic_store_n(&control->last_status, 0, __ATOMIC_RELEASE);
            reset_device_counters(control);
            ++control->generation;
            __atomic_store_n(&control->enabled, 1U, __ATOMIC_RELEASE);
            print_status(control);
        }
    } else if (argc == 8 &&
               (std::strcmp(argv[1], "arm-multirank-lossless") == 0 ||
                std::strcmp(argv[1], "arm-multirank-int4") == 0)) {
        long long device_mask = 0;
        long long m = 0;
        long long k = 0;
        long long call_index = 0;
        long long hits_per_device = 0;
        long long bytes = 0;
        if (!parse_int(argv[2], 1, 0x7FFFFFFFFFFFFFFFLL, &device_mask) ||
            !parse_int(argv[3], 1, 1048576, &m) ||
            !parse_int(argv[4], 1, 1048576, &k) ||
            !parse_int(argv[5], 1, 1048576, &call_index) ||
            !parse_int(argv[6], 1, 1024, &hits_per_device) ||
            !parse_int(argv[7], 1, 65536, &bytes)) {
            std::fprintf(stderr, "invalid arm-multirank arguments\n");
            result = 2;
        } else {
            const bool numeric = std::strcmp(argv[1], "arm-multirank-int4") == 0;
            const uint64_t mask = static_cast<uint64_t>(device_mask);
            __atomic_store_n(&control->enabled, 0U, __ATOMIC_RELEASE);
            control->mode = numeric ? FAULT20_MODE_INT4_EQUIVALENT
                                    : FAULT20_MODE_LOSSLESS_D2D;
            control->target_device = -1;
            control->target_m = static_cast<int32_t>(m);
            control->target_k = static_cast<int32_t>(k);
            control->target_call_index = static_cast<uint32_t>(call_index);
            control->max_hits = static_cast<uint32_t>(
                hits_per_device * static_cast<long long>(__builtin_popcountll(mask)));
            control->quant_step = numeric ? 16U : 0U;
            control->keep_mantissa_bits = 0;
            control->target_dtype = 2;
            control->max_bytes = static_cast<uint64_t>(bytes);
            control->target_device_mask = mask;
            control->max_hits_per_device = static_cast<uint32_t>(hits_per_device);
            control->selection_mode = FAULT20_SELECTION_BOUNDED;
            control->target_api = FAULT20_TARGET_API_QMM;
            __atomic_store_n(&control->hit_count, 0U, __ATOMIC_RELEASE);
            __atomic_store_n(&control->candidate_count, 0U, __ATOMIC_RELEASE);
            __atomic_store_n(&control->last_status, 0, __ATOMIC_RELEASE);
            reset_device_counters(control);
            ++control->generation;
            __atomic_store_n(&control->enabled, 1U, __ATOMIC_RELEASE);
            print_status(control);
        }
    } else if (argc == 6 &&
               (std::strcmp(argv[1], "arm-all-qmm-lossless") == 0 ||
                std::strcmp(argv[1], "arm-all-qmm-int4") == 0 ||
                std::strcmp(argv[1], "arm-all-qmm-int2") == 0 ||
                std::strcmp(argv[1], "arm-all-matmul-lossless") == 0)) {
        long long device_mask = 0;
        long long m = 0;
        long long k = 0;
        long long bytes = 0;
        if (!parse_int(argv[2], 1, 0x7FFFFFFFFFFFFFFFLL, &device_mask) ||
            !parse_int(argv[3], 1, 1048576, &m) ||
            !parse_int(argv[4], 1, 1048576, &k) ||
            !parse_int(argv[5], 1, 65536, &bytes)) {
            std::fprintf(stderr, "invalid all-hit arguments\n");
            result = 2;
        } else {
            const bool qmm_int4 = std::strcmp(argv[1], "arm-all-qmm-int4") == 0;
            const bool qmm_int2 = std::strcmp(argv[1], "arm-all-qmm-int2") == 0;
            const bool matmul = std::strcmp(argv[1], "arm-all-matmul-lossless") == 0;
            const uint64_t mask = static_cast<uint64_t>(device_mask);
            __atomic_store_n(&control->enabled, 0U, __ATOMIC_RELEASE);
            control->mode = qmm_int4 ? FAULT20_MODE_INT4_EQUIVALENT
                                     : (qmm_int2 ? FAULT20_MODE_INT2_EQUIVALENT
                                                 : FAULT20_MODE_LOSSLESS_D2D);
            control->target_device = -1;
            control->target_m = static_cast<int32_t>(m);
            control->target_k = static_cast<int32_t>(k);
            control->target_call_index = 1;
            control->max_hits = FAULT20_ALL_HITS_LIMIT;
            control->quant_step = qmm_int4 ? 16U : (qmm_int2 ? 64U : 0U);
            control->keep_mantissa_bits = 0;
            control->target_dtype = matmul ? 27 : 2;
            control->max_bytes = static_cast<uint64_t>(bytes);
            control->target_device_mask = mask;
            control->max_hits_per_device = FAULT20_ALL_HITS_LIMIT;
            control->selection_mode = FAULT20_SELECTION_ALL_HITS;
            control->target_api = matmul ? FAULT20_TARGET_API_MATMUL_WEIGHT_NZ
                                         : FAULT20_TARGET_API_QMM;
            __atomic_store_n(&control->hit_count, 0U, __ATOMIC_RELEASE);
            __atomic_store_n(&control->candidate_count, 0U, __ATOMIC_RELEASE);
            __atomic_store_n(&control->last_status, 0, __ATOMIC_RELEASE);
            reset_device_counters(control);
            control->proof_mode = matmul ? FAULT20_PROOF_MODE_STRICT_PRODUCER
                                         : FAULT20_PROOF_MODE_SEMANTIC_SLOT;
            ++control->generation;
            __atomic_store_n(&control->enabled, 1U, __ATOMIC_RELEASE);
            print_status(control);
        }
    } else if (argc == 8 &&
               std::strcmp(argv[1], "arm-all-matmul-mantissa") == 0) {
        long long device_mask = 0;
        long long m = 0;
        long long k = 0;
        long long bytes = 0;
        long long keep_bits = 0;
        const bool is_bf16 = std::strcmp(argv[6], "bf16") == 0;
        const bool is_fp16 = std::strcmp(argv[6], "fp16") == 0;
        if (!parse_int(argv[2], 1, 0x7FFFFFFFFFFFFFFFLL, &device_mask) ||
            !parse_int(argv[3], 1, 1048576, &m) ||
            !parse_int(argv[4], 1, 1048576, &k) ||
            !parse_int(argv[5], 1, 65536, &bytes) || (!is_bf16 && !is_fp16) ||
            !parse_int(argv[7], 0, is_bf16 ? 7 : 10, &keep_bits)) {
            std::fprintf(stderr, "invalid all-hit matmul mantissa arguments\n");
            result = 2;
        } else {
            __atomic_store_n(&control->enabled, 0U, __ATOMIC_RELEASE);
            control->mode = FAULT20_MODE_MANTISSA_TRUNCATE;
            control->target_device = -1;
            control->target_m = static_cast<int32_t>(m);
            control->target_k = static_cast<int32_t>(k);
            control->target_call_index = 1;
            control->max_hits = FAULT20_ALL_HITS_LIMIT;
            control->quant_step = 0;
            control->keep_mantissa_bits = static_cast<uint32_t>(keep_bits);
            control->target_dtype = is_bf16 ? 27 : 1;
            control->max_bytes = static_cast<uint64_t>(bytes);
            control->target_device_mask = static_cast<uint64_t>(device_mask);
            control->max_hits_per_device = FAULT20_ALL_HITS_LIMIT;
            control->selection_mode = FAULT20_SELECTION_ALL_HITS;
            control->target_api = FAULT20_TARGET_API_MATMUL_WEIGHT_NZ;
            __atomic_store_n(&control->hit_count, 0U, __ATOMIC_RELEASE);
            __atomic_store_n(&control->candidate_count, 0U, __ATOMIC_RELEASE);
            __atomic_store_n(&control->last_status, 0, __ATOMIC_RELEASE);
            reset_device_counters(control);
            ++control->generation;
            __atomic_store_n(&control->enabled, 1U, __ATOMIC_RELEASE);
            print_status(control);
        }
    } else {
        std::fprintf(stderr,
                     "usage:\n"
                     "  fault20_buffer_ctl status\n"
                     "  fault20_buffer_ctl arm-float-matmul-output-proxy pos_inf|neg_inf|qnan DEVICE CALL_INDEX ELEMENT_COUNT(1|5|51|512)\n"
                     "  fault20_buffer_ctl arm-float-matmul-cancel half|block256|block512|block1024|alternating DEVICE CALL_INDEX\n"
                     "  fault20_buffer_ctl arm-float-matmul-finite DEVICE CALL_INDEX EXPONENT(48|56|57|58|59)\n"
                     "  fault20_buffer_ctl arm-float-matmul-observe DEVICE CALL_INDEX\n"
                     "  fault20_buffer_ctl arm-float-matmul-lossless self|mat2|dual DEVICE CALL_INDEX\n"
                     "  fault20_buffer_ctl arm-observe DEVICE M K CALL_INDEX MAX_OUTPUT_BYTES\n"
                     "  fault20_buffer_ctl arm-observe-x2 DEVICE M K CALL_INDEX MAX_WEIGHT_BYTES\n"
                     "  fault20_buffer_ctl arm-observe-x2-scale DEVICE M K SCALE_ELEMENTS CALL_INDEX MAX_SCALE_BYTES\n"
                     "  fault20_buffer_ctl arm-lossless-x2-scale DEVICE M K SCALE_ELEMENTS CALL_INDEX MAX_SCALE_BYTES\n"
                     "  fault20_buffer_ctl arm-amplify-x2-scale DEVICE M K SCALE_ELEMENTS CALL_INDEX MAX_SCALE_BYTES FACTOR(2|10|100|1000)\n"
                     "  fault20_buffer_ctl arm-lossless-x1-x2 DEVICE M K X2_N CALL_INDEX X1_BYTES WEIGHT_BYTES\n"
                     "  fault20_buffer_ctl arm-extreme-x1-x2 DEVICE M K X2_N CALL_INDEX X1_BYTES WEIGHT_BYTES VALUE positive_max\n"
                      "  fault20_buffer_ctl arm-lossless DEVICE K MAX_HITS MAX_BYTES\n"
                      "  fault20_buffer_ctl arm-int4 DEVICE M K CALL_INDEX MAX_HITS MAX_BYTES\n"
                      "  fault20_buffer_ctl arm-int4-slotproof DEVICE M K CALL_INDEX MAX_HITS MAX_BYTES\n"
                      "  fault20_buffer_ctl arm-int2 DEVICE M K CALL_INDEX MAX_HITS MAX_BYTES\n"
                      "  fault20_buffer_ctl arm-int4-observe DEVICE M K CALL_INDEX MAX_HITS MAX_BYTES [MAX_OUTPUT_BYTES]\n"
                      "  fault20_buffer_ctl arm-int2-observe DEVICE M K CALL_INDEX MAX_HITS MAX_BYTES [MAX_OUTPUT_BYTES]\n"
                      "  fault20_buffer_ctl arm-int4-request DEVICE M K CALL_INDEX MAX_HITS_PER_REQUEST MAX_HITS_TOTAL MAX_BYTES\n"
                      "  fault20_buffer_ctl arm-int2-request DEVICE M K CALL_INDEX MAX_HITS_PER_REQUEST MAX_HITS_TOTAL MAX_BYTES\n"
                      "  fault20_buffer_ctl arm-int4-observe-request DEVICE M K CALL_INDEX MAX_HITS_PER_REQUEST MAX_HITS_TOTAL MAX_BYTES [MAX_OUTPUT_BYTES]\n"
                      "  fault20_buffer_ctl arm-int2-observe-request DEVICE M K CALL_INDEX MAX_HITS_PER_REQUEST MAX_HITS_TOTAL MAX_BYTES [MAX_OUTPUT_BYTES]\n"
                      "  fault20_buffer_ctl request-begin MONOTONIC_REQUEST_ID\n"
                     "  fault20_buffer_ctl arm-extreme DEVICE M K CALL_INDEX MAX_BYTES VALUE RATIO_BPS positive_max|negative_max|alternating\n"
                     "  fault20_buffer_ctl arm-progressive-extreme DEVICE M K CALL_INDEX MAX_HITS MAX_BYTES VALUE RATIO_BPS positive_max|negative_max|alternating\n"
                     "  fault20_buffer_ctl arm-all-qmm-extreme DEVICE_MASK M K MAX_BYTES VALUE RATIO_BPS positive_max|negative_max|alternating\n"
                     "  fault20_buffer_ctl arm-mantissa DEVICE M K CALL_INDEX MAX_HITS MAX_BYTES bf16|fp16 KEEP_BITS\n"
                     "  fault20_buffer_ctl arm-multirank-lossless DEVICE_MASK M K CALL_INDEX MAX_HITS_PER_DEVICE MAX_BYTES\n"
                     "  fault20_buffer_ctl arm-multirank-int4 DEVICE_MASK M K CALL_INDEX MAX_HITS_PER_DEVICE MAX_BYTES\n"
                     "  fault20_buffer_ctl arm-all-qmm-lossless DEVICE_MASK M K MAX_BYTES\n"
                     "  fault20_buffer_ctl arm-all-qmm-int4 DEVICE_MASK M K MAX_BYTES\n"
                     "  fault20_buffer_ctl arm-all-qmm-int2 DEVICE_MASK M K MAX_BYTES\n"
                     "  fault20_buffer_ctl arm-all-matmul-lossless DEVICE_MASK M K MAX_BYTES\n"
                     "  fault20_buffer_ctl arm-all-matmul-mantissa DEVICE_MASK M K MAX_BYTES bf16|fp16 KEEP_BITS\n"
                     "  fault20_buffer_ctl disarm\n"
                     "  fault20_buffer_ctl reset\n");
        result = 2;
    }
    (void)munmap(control, sizeof(*control));
    (void)flock(fd, LOCK_UN);
    (void)close(fd);
    return result;
}
