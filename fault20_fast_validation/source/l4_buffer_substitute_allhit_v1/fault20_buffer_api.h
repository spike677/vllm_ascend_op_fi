#ifndef FAULT20_BUFFER_API_H
#define FAULT20_BUFFER_API_H

#include <cstddef>
#include <cstdint>

constexpr uint64_t FAULT20_CONTROL_MAGIC = 0x4632304255463133ULL;
constexpr uint32_t FAULT20_CONTROL_VERSION = 13;
constexpr const char *FAULT20_CONTROL_SHM = "/fault20_l4_buffer_substitute_ctl_v13";
constexpr uint32_t FAULT20_MODE_DISABLED = 0;
constexpr uint32_t FAULT20_MODE_LOSSLESS_D2D = 1;
constexpr uint32_t FAULT20_MODE_INT4_EQUIVALENT = 2;
constexpr uint32_t FAULT20_MODE_INT2_EQUIVALENT = 3;
constexpr uint32_t FAULT20_MODE_MANTISSA_TRUNCATE = 4;
constexpr uint32_t FAULT20_MODE_QMM_EXTREME_X1 = 5;
constexpr uint32_t FAULT20_MODE_QMM_OUTPUT_OBSERVE = 6;
constexpr uint32_t FAULT20_MODE_QMM_WEIGHT_OBSERVE = 7;
constexpr uint32_t FAULT20_MODE_QMM_EXTREME_X1_X2 = 8;
constexpr uint32_t FAULT20_MODE_QMM_LOSSLESS_X1_X2 = 9;
constexpr uint32_t FAULT20_MODE_QMM_X2_SCALE_OBSERVE = 10;
constexpr uint32_t FAULT20_MODE_QMM_X2_SCALE_LOSSLESS = 11;
constexpr uint32_t FAULT20_MODE_QMM_X2_SCALE_AMPLIFY = 12;
constexpr uint32_t FAULT20_MODE_FLOAT_MATMUL_LOSSLESS_SELF = 13;
constexpr uint32_t FAULT20_MODE_FLOAT_MATMUL_LOSSLESS_MAT2 = 14;
constexpr uint32_t FAULT20_MODE_FLOAT_MATMUL_LOSSLESS_DUAL = 15;
constexpr uint32_t FAULT20_MODE_FLOAT_MATMUL_OUTPUT_OBSERVE = 16;
constexpr uint32_t FAULT20_MODE_FLOAT_MATMUL_FINITE_DUAL = 17;
constexpr uint32_t FAULT20_MODE_FLOAT_MATMUL_CANCELLATION = 18;
constexpr uint32_t FAULT20_MODE_FLOAT_MATMUL_OUTPUT_PROXY = 19;
constexpr uint32_t FAULT20_CANCELLATION_HALF = 1;
constexpr uint32_t FAULT20_CANCELLATION_BLOCK = 2;
constexpr uint32_t FAULT20_CANCELLATION_ALTERNATING = 3;
constexpr uint32_t FAULT20_OUTPUT_PROXY_POS_INF = 1;
constexpr uint32_t FAULT20_OUTPUT_PROXY_NEG_INF = 2;
constexpr uint32_t FAULT20_OUTPUT_PROXY_QNAN = 3;
constexpr uint32_t FAULT20_EXTREME_POSITIVE_MAX = 1;
constexpr uint32_t FAULT20_EXTREME_NEGATIVE_MAX = 2;
constexpr uint32_t FAULT20_EXTREME_ALTERNATING = 3;
constexpr uint32_t FAULT20_SELECTION_BOUNDED = 0;
constexpr uint32_t FAULT20_SELECTION_ALL_HITS = 1;
constexpr uint32_t FAULT20_TARGET_API_NONE = 0;
constexpr uint32_t FAULT20_TARGET_API_QMM = 1;
constexpr uint32_t FAULT20_TARGET_API_MATMUL_WEIGHT_NZ = 2;
constexpr uint32_t FAULT20_TARGET_API_FLOAT_MATMUL = 3;
constexpr uint32_t FAULT20_ALL_HITS_LIMIT = 0xFFFFFFFFU;
constexpr size_t FAULT20_MAX_DIMS = 8;
constexpr size_t FAULT20_MAX_DEVICES = 64;

struct Fault20X1Metadata {
    int32_t dtype;
    int32_t format;
    uint32_t view_rank;
    uint32_t storage_rank;
    uint32_t stride_rank;
    uint32_t contiguous;
    int64_t view_shape[FAULT20_MAX_DIMS];
    int64_t storage_shape[FAULT20_MAX_DIMS];
    int64_t strides[FAULT20_MAX_DIMS];
    int64_t view_offset;
    int64_t numel;
    int64_t storage_elements;
    uint64_t element_size;
    uint64_t view_bytes;
    uint64_t storage_bytes;
    int64_t output_last_dim;
    int64_t group_size;
    uint32_t transpose_x1;
    uint32_t transpose_x2;
};

struct Fault20TensorMetadata {
    int32_t dtype;
    int32_t format;
    uint32_t present;
    uint32_t contiguous;
    uint32_t view_rank;
    uint32_t storage_rank;
    uint32_t stride_rank;
    uint32_t reserved32;
    int64_t view_shape[FAULT20_MAX_DIMS];
    int64_t storage_shape[FAULT20_MAX_DIMS];
    int64_t strides[FAULT20_MAX_DIMS];
    int64_t view_offset;
    int64_t numel;
    int64_t storage_elements;
    uint64_t element_size;
    uint64_t view_bytes;
    uint64_t storage_bytes;
    uint64_t tensor_ptr;
    uint64_t storage_ptr;
    uint64_t raw_addr;
    uint64_t storage_addr;
    uint64_t data_addr;
};

struct Fault20MatmulMetadata {
    Fault20TensorMetadata self;
    Fault20TensorMetadata mat2;
    Fault20TensorMetadata out;
    int32_t cube_math_type;
    uint32_t all_present;
    uint32_t any_empty_storage;
    uint32_t any_shared_storage;
};

struct Fault20QmmMetadata {
    Fault20X1Metadata x1_legacy;
    Fault20TensorMetadata x1;
    Fault20TensorMetadata x2;
    Fault20TensorMetadata x1_scale;
    Fault20TensorMetadata x2_scale;
    Fault20TensorMetadata y_scale;
    Fault20TensorMetadata x1_offset;
    Fault20TensorMetadata x2_offset;
    Fault20TensorMetadata y_offset;
    Fault20TensorMetadata bias;
    Fault20TensorMetadata out;
    int64_t group_size;
    uint32_t transpose_x1;
    uint32_t transpose_x2;
    uint32_t present_count;
    uint32_t any_empty_storage;
    uint32_t any_shared_storage;
};

struct Fault20Control {
    uint64_t magic;
    uint32_t version;
    uint32_t enabled;
    uint32_t mode;
    int32_t target_device;
    int32_t target_m;
    int32_t target_k;
    uint32_t target_call_index;
    uint32_t max_hits;
    uint32_t hit_count;
    uint32_t candidate_count;
    uint32_t quant_step;
    uint32_t keep_mantissa_bits;
    int32_t extreme_value;
    uint32_t extreme_ratio_bps;
    uint32_t extreme_pattern;
    int32_t target_dtype;
    int32_t last_status;
    uint32_t selection_mode;
    uint64_t max_bytes;
    uint64_t generation;
    uint64_t target_device_mask;
    uint32_t max_hits_per_device;
    uint32_t target_api;
    alignas(64) uint32_t candidate_count_per_device[FAULT20_MAX_DEVICES];
    alignas(64) uint32_t hit_count_per_device[FAULT20_MAX_DEVICES];
    alignas(64) int32_t last_status_per_device[FAULT20_MAX_DEVICES];
    uint64_t max_output_bytes;
    uint64_t max_weight_bytes;
    int32_t target_x2_n;
    uint32_t reserved32;
    uint64_t max_scale_bytes;
    int32_t target_scale_elements;
    uint32_t reserved_scale;
    double scale_factor;
};

extern "C" {
void fault20_qmm_gws_begin();
bool fault20_qmm_gws_commit(const Fault20QmmMetadata *metadata);
void fault20_qmm_gws_abort();
void fault20_matmul_gws_begin();
bool fault20_matmul_gws_commit(const Fault20MatmulMetadata *metadata);
void fault20_matmul_gws_abort();
void fault20_float_matmul_gws_begin();
bool fault20_float_matmul_gws_commit(const Fault20MatmulMetadata *metadata);
void fault20_float_matmul_gws_abort();
}

#endif
