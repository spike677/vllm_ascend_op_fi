#ifndef FAULT20_EXTREME_TRANSFORM_H
#define FAULT20_EXTREME_TRANSFORM_H

#include "fault20_buffer_api.h"

#include <cstddef>
#include <cstdint>

struct Fault20ExtremeTransformStats {
    uint64_t selected_elements{0};
    uint64_t changed_elements{0};
};

inline bool fault20_valid_extreme_configuration(int32_t value, uint32_t ratio_bps,
                                                uint32_t pattern) {
    if (ratio_bps == 0 || ratio_bps > 10000) {
        return false;
    }
    if (pattern == FAULT20_EXTREME_POSITIVE_MAX) {
        return value == 127;
    }
    if (pattern == FAULT20_EXTREME_NEGATIVE_MAX) {
        return value == -128;
    }
    return pattern == FAULT20_EXTREME_ALTERNATING && (value == 127 || value == -128);
}

inline uint64_t fault20_extreme_selected_count(uint64_t elements, uint32_t ratio_bps) {
    return elements * static_cast<uint64_t>(ratio_bps) / 10000U;
}

inline bool fault20_extreme_select_index(uint64_t index, uint64_t elements,
                                         uint64_t selected_elements) {
    if (elements == 0 || selected_elements == 0 || index >= elements) {
        return false;
    }
    const uint64_t before = index * selected_elements / elements;
    const uint64_t after = (index + 1U) * selected_elements / elements;
    return after != before;
}

inline int8_t fault20_extreme_value(uint64_t selected_ordinal, int32_t first_value,
                                    uint32_t pattern) {
    if (pattern != FAULT20_EXTREME_ALTERNATING || (selected_ordinal % 2U) == 0) {
        return static_cast<int8_t>(first_value);
    }
    return static_cast<int8_t>(first_value == 127 ? -128 : 127);
}

inline Fault20ExtremeTransformStats fault20_apply_extreme_transform(
    int8_t *values, uint64_t elements, int32_t value, uint32_t ratio_bps,
    uint32_t pattern) {
    Fault20ExtremeTransformStats stats{};
    if (values == nullptr || !fault20_valid_extreme_configuration(value, ratio_bps, pattern)) {
        return stats;
    }
    const uint64_t selected = fault20_extreme_selected_count(elements, ratio_bps);
    for (uint64_t index = 0; index < elements; ++index) {
        if (!fault20_extreme_select_index(index, elements, selected)) {
            continue;
        }
        const int8_t after = fault20_extreme_value(stats.selected_elements, value, pattern);
        stats.changed_elements += values[index] != after ? 1U : 0U;
        values[index] = after;
        ++stats.selected_elements;
    }
    return stats;
}

#endif
