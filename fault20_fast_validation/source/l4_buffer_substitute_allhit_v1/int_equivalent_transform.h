#ifndef FAULT20_INT_EQUIVALENT_TRANSFORM_H
#define FAULT20_INT_EQUIVALENT_TRANSFORM_H

#include <cstdint>

inline int fault20_round_nearest_away_from_zero(int value, int step) {
    const int64_t wide = static_cast<int64_t>(value);
    const int64_t magnitude = wide >= 0 ? wide : -wide;
    const int64_t rounded = (magnitude + step / 2) / step;
    return static_cast<int>(wide >= 0 ? rounded : -rounded);
}

inline int8_t fault20_int4_equivalent_value(int8_t input) {
    int q4 = fault20_round_nearest_away_from_zero(static_cast<int>(input), 16);
    if (q4 < -8) {
        q4 = -8;
    } else if (q4 > 7) {
        q4 = 7;
    }
    return static_cast<int8_t>(q4 * 16);
}

inline int8_t fault20_int2_equivalent_value(int8_t input) {
    int q2 = fault20_round_nearest_away_from_zero(static_cast<int>(input), 64);
    if (q2 < -2) {
        q2 = -2;
    } else if (q2 > 1) {
        q2 = 1;
    }
    return static_cast<int8_t>(q2 * 64);
}

#endif
