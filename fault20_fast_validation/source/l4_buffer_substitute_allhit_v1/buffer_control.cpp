#include "fault20_buffer_api.h"

#include <cerrno>
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
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
        control->target_dtype = 27;
        control->max_bytes = 5120;
        control->target_device_mask = 1;
        control->max_hits_per_device = 1;
        control->selection_mode = FAULT20_SELECTION_BOUNDED;
        control->target_api = FAULT20_TARGET_API_NONE;
    }
    *fd_out = fd;
    return control;
}

void print_status(const Fault20Control *control) {
    std::printf(
        "{\"magic\":\"0x%016" PRIx64 "\",\"version\":%u,\"enabled\":%u,"
        "\"mode\":%u,\"target_device\":%d,\"target_m\":%d,\"target_k\":%d,"
        "\"target_call_index\":%u,\"max_hits\":%u,\"hit_count\":%u,"
        "\"candidate_count\":%u,\"quant_step\":%u,"
        "\"keep_mantissa_bits\":%u,\"target_dtype\":%d,\"last_status\":%d,"
        "\"selection_mode\":%u,\"target_api\":%u,"
        "\"max_bytes\":%" PRIu64 ",\"generation\":%" PRIu64
        ",\"target_device_mask\":\"0x%016" PRIx64
        "\",\"max_hits_per_device\":%u,"
        "\"candidate_count_per_device\":[%u,%u],"
        "\"hit_count_per_device\":[%u,%u],"
        "\"last_status_per_device\":[%d,%d]}\n",
        control->magic, control->version,
        __atomic_load_n(&control->enabled, __ATOMIC_ACQUIRE), control->mode,
        control->target_device, control->target_m, control->target_k,
        control->target_call_index, control->max_hits,
        __atomic_load_n(&control->hit_count, __ATOMIC_ACQUIRE),
        __atomic_load_n(&control->candidate_count, __ATOMIC_ACQUIRE), control->quant_step,
        control->keep_mantissa_bits, control->target_dtype,
        __atomic_load_n(&control->last_status, __ATOMIC_ACQUIRE),
        control->selection_mode, control->target_api, control->max_bytes,
        control->generation, control->target_device_mask, control->max_hits_per_device,
        __atomic_load_n(&control->candidate_count_per_device[0], __ATOMIC_ACQUIRE),
        __atomic_load_n(&control->candidate_count_per_device[1], __ATOMIC_ACQUIRE),
        __atomic_load_n(&control->hit_count_per_device[0], __ATOMIC_ACQUIRE),
        __atomic_load_n(&control->hit_count_per_device[1], __ATOMIC_ACQUIRE),
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

void reset_device_counters(Fault20Control *control) {
    for (size_t device = 0; device < FAULT20_MAX_DEVICES; ++device) {
        __atomic_store_n(&control->candidate_count_per_device[device], 0U,
                         __ATOMIC_RELEASE);
        __atomic_store_n(&control->hit_count_per_device[device], 0U,
                         __ATOMIC_RELEASE);
        __atomic_store_n(&control->last_status_per_device[device], 0,
                         __ATOMIC_RELEASE);
    }
}

}  // namespace

int main(int argc, char **argv) {
    int fd = -1;
    Fault20Control *control = map_control(&fd);
    if (control == nullptr) {
        return 1;
    }
    int result = 0;
    if (argc == 2 && std::strcmp(argv[1], "status") == 0) {
        print_status(control);
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
        ++control->generation;
        print_status(control);
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
    } else if (argc == 8 && std::strcmp(argv[1], "arm-int4") == 0) {
        long long device = 0;
        long long m = 0;
        long long k = 0;
        long long call_index = 0;
        long long hits = 0;
        long long bytes = 0;
        if (!parse_int(argv[2], 0, 63, &device) ||
            !parse_int(argv[3], 1, 1048576, &m) ||
            !parse_int(argv[4], 1, 1048576, &k) ||
            !parse_int(argv[5], 1, 1048576, &call_index) ||
            !parse_int(argv[6], 1, 1024, &hits) ||
            !parse_int(argv[7], 1, 65536, &bytes)) {
            std::fprintf(stderr, "invalid arm-int4 arguments\n");
            result = 2;
        } else {
            __atomic_store_n(&control->enabled, 0U, __ATOMIC_RELEASE);
            control->mode = FAULT20_MODE_INT4_EQUIVALENT;
            control->target_device = static_cast<int32_t>(device);
            control->target_m = static_cast<int32_t>(m);
            control->target_k = static_cast<int32_t>(k);
            control->target_call_index = static_cast<uint32_t>(call_index);
            control->max_hits = static_cast<uint32_t>(hits);
            control->quant_step = 16;
            control->keep_mantissa_bits = 0;
            control->target_dtype = 2;
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
    } else if (argc == 8 && std::strcmp(argv[1], "arm-int2") == 0) {
        long long device = 0;
        long long m = 0;
        long long k = 0;
        long long call_index = 0;
        long long hits = 0;
        long long bytes = 0;
        if (!parse_int(argv[2], 0, 63, &device) ||
            !parse_int(argv[3], 1, 1048576, &m) ||
            !parse_int(argv[4], 1, 1048576, &k) ||
            !parse_int(argv[5], 1, 1048576, &call_index) ||
            !parse_int(argv[6], 1, 1024, &hits) ||
            !parse_int(argv[7], 1, 65536, &bytes)) {
            std::fprintf(stderr, "invalid arm-int2 arguments\n");
            result = 2;
        } else {
            __atomic_store_n(&control->enabled, 0U, __ATOMIC_RELEASE);
            control->mode = FAULT20_MODE_INT2_EQUIVALENT;
            control->target_device = static_cast<int32_t>(device);
            control->target_m = static_cast<int32_t>(m);
            control->target_k = static_cast<int32_t>(k);
            control->target_call_index = static_cast<uint32_t>(call_index);
            control->max_hits = static_cast<uint32_t>(hits);
            control->quant_step = 64;
            control->keep_mantissa_bits = 0;
            control->target_dtype = 2;
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
                     "  fault20_buffer_ctl arm-lossless DEVICE K MAX_HITS MAX_BYTES\n"
                     "  fault20_buffer_ctl arm-int4 DEVICE M K CALL_INDEX MAX_HITS MAX_BYTES\n"
                     "  fault20_buffer_ctl arm-int2 DEVICE M K CALL_INDEX MAX_HITS MAX_BYTES\n"
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
