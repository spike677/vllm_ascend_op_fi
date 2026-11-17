#ifndef DEMONCAT_FAULT28_CAUSAL_TILE_H
#define DEMONCAT_FAULT28_CAUSAL_TILE_H
#include <cstdint>

#if defined(__CCE_AICORE__)
#define F28_TILE_INLINE __aicore__ inline
#else
#define F28_TILE_INLINE inline
#endif

namespace fault28_causal_tile {
// BNSD's flattened M order is G,S1. Scope guarantees that an M tile never
// crosses a head boundary, S1 == S2 == 2048 and M is 128, 256 or 512.
// Only columns beyond the LAST query in this tile are removed. Therefore
// every unmasked key of every query in the tile remains exactly once.
F28_TILE_INLINE uint32_t InnerWidth(uint32_t gs1Start, uint32_t mRows,
                                 uint32_t s1, uint32_t s2Index,
                                 uint32_t s2Base, uint32_t nativeWidth)
{
    if (s1 == 0 || s2Base == 0 || mRows == 0) return 0;
    const uint64_t first = gs1Start % s1;
    const uint64_t end = first + mRows; // exclusive query bound
    const uint64_t beginK = uint64_t(s2Index) * s2Base;
    if (end > s1 || end <= beginK) return 0;
    const uint64_t valid = end - beginK;
    const uint64_t clipped = valid < nativeWidth ? valid : nativeWidth;
    const uint64_t aligned = ((clipped + 31U) / 32U) * 32U;
    return uint32_t(aligned < nativeWidth ? aligned : nativeWidth);
}
}
#undef F28_TILE_INLINE
#endif
