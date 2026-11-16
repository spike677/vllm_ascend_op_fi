#pragma once
#include <stdint.h>
struct Fault39TensorMetadata {
    int32_t dtype;
    int32_t format;
    uint32_t present;
    uint32_t contiguous;
    uint32_t view_rank;
    uint32_t storage_rank;
    uint32_t stride_rank;
    uint32_t reserved32;
    int64_t view_shape[16];
    int64_t storage_shape[16];
    int64_t strides[16];
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

struct Fault39MatmulMetadata {
    Fault39TensorMetadata self;
    Fault39TensorMetadata mat2;
    Fault39TensorMetadata out;
    int32_t cube_math_type;
    uint32_t all_present;
    uint32_t any_empty_storage;
    uint32_t any_shared_storage;
};

struct Fault39MetadataEnvelopeV1 {
 uint64_t magic; uint32_t version, size, payload_size, reserved; const Fault39MatmulMetadata* payload;
};
#define FAULT39_METADATA_MAGIC UINT64_C(0x4633394d45544131)
extern "C" void fault39_gws_begin_v1();
extern "C" int fault39_gws_commit_v1(const Fault39MetadataEnvelopeV1*);
extern "C" void fault39_gws_abort_v1();