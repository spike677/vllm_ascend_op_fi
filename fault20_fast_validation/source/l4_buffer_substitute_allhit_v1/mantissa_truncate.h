#ifndef FAULT20_MANTISSA_TRUNCATE_H
#define FAULT20_MANTISSA_TRUNCATE_H

#include <cstdint>

inline uint16_t fault20_truncate_bf16_mantissa(uint16_t bits, uint32_t keep_bits) {
    constexpr uint16_t kExponentMask = 0x7F80U;
    if ((bits & kExponentMask) == kExponentMask || keep_bits >= 7U) {
        return bits;
    }
    const uint32_t clear_bits = 7U - keep_bits;
    const uint16_t keep_mask = static_cast<uint16_t>(
        ~static_cast<uint16_t>((1U << clear_bits) - 1U));
    return static_cast<uint16_t>(bits & keep_mask);
}

inline uint16_t fault20_truncate_fp16_mantissa(uint16_t bits, uint32_t keep_bits) {
    constexpr uint16_t kExponentMask = 0x7C00U;
    if ((bits & kExponentMask) == kExponentMask || keep_bits >= 10U) {
        return bits;
    }
    const uint32_t clear_bits = 10U - keep_bits;
    const uint16_t keep_mask = static_cast<uint16_t>(
        ~static_cast<uint16_t>((1U << clear_bits) - 1U));
    return static_cast<uint16_t>(bits & keep_mask);
}

#endif
