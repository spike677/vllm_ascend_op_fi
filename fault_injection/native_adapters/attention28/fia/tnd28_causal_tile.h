#ifndef DEMONCAT_TND28_CAUSAL_TILE_H
#define DEMONCAT_TND28_CAUSAL_TILE_H
#include <cstdint>
#if defined(__CCE_AICORE__)
#define DCAT_TND28_INLINE __aicore__ inline
#else
#define DCAT_TND28_INLINE inline
#endif
namespace tnd28_causal_tile {
// TND's flattened M is (S1,G), not (G,S1). Remove only columns after the
// last query in this M tile. The original byte mask still masks each row.
DCAT_TND28_INLINE uint32_t InnerWidth(uint32_t flat, uint32_t rows,
    uint32_t queries, uint32_t groups, uint32_t block, uint32_t base, uint32_t nativeWidth) {
    if (!queries || !groups || !rows || !base || !nativeWidth) return 0;
    const uint64_t endFlat=uint64_t(flat)+rows;
    if (endFlat>uint64_t(queries)*groups) return 0;
    const uint64_t endQuery=(endFlat-1)/groups+1;
    const uint64_t begin=uint64_t(block)*base;
    if (endQuery<=begin) return 0;
    const uint64_t valid=endQuery-begin;
    const uint64_t clipped=valid<nativeWidth?valid:nativeWidth;
    const uint64_t aligned=(clipped+31)/32*32;
    return static_cast<uint32_t>(aligned<nativeWidth?aligned:nativeWidth);
}
}
#undef DCAT_TND28_INLINE
#endif
