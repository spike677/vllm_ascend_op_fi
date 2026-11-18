#ifndef DEMONCAT_TND28_PREFILL_CONTRACT_H
#define DEMONCAT_TND28_PREFILL_CONTRACT_H
#include <cstdint>

// SDK-independent, normal-only host postcondition. No device calculation,
// tiling selection, replay, pointer inference or policy state lives here.
namespace tnd28_prefill_normal {
constexpr uint64_t kNormalKey = 103000000030000003ULL;
struct NormalTiling {
    uint64_t key = 0;
    uint32_t s1 = 0, s2 = 0, groups = 0, kvHeads = 0;
    uint32_t m = 0, s2Base = 0, loops = 0, tail = 0, aligned = 0;
    uint32_t aic = 0, aiv = 0, usedCores = 0, blockDim = 0, calculatedBlockDim = 0;
    int32_t rawBlockSize = -1, parsedBlockSize = -1;
    int64_t preToken = 0, nextToken = 0;
    bool cumulativeQ = false, cumulativeKv = false;
    bool flashDecode = true, brcb = true;
};
inline bool NormalPostcondition(const NormalTiling& n) {
    return n.key == kNormalKey && n.s1 == 2048 && n.s2 == 2048
        && n.groups == 2 && n.kvHeads == 8 && n.m == 512 && n.s2Base == 512
        && n.loops == 4 && n.tail == 512 && n.aligned == 512
        && n.aic > 0 && n.aic <= 26 && n.aiv == 2*n.aic
        && n.usedCores > 0 && n.usedCores <= n.aic
        && n.blockDim > 0 && n.blockDim == n.calculatedBlockDim
        && n.rawBlockSize == 128 && n.parsedBlockSize == 0
        && n.preToken == 2147483647 && n.nextToken == 0
        && n.cumulativeQ && n.cumulativeKv && !n.flashDecode && !n.brcb;
}
struct Coordinate { uint32_t query = 0, group = 0; };
inline bool TndCoordinate(uint32_t flat, uint32_t queryTokens, uint32_t groups, Coordinate& out) {
    if (queryTokens == 0 || queryTokens > 2048 || groups == 0 || groups > 2
        || flat >= queryTokens*groups) return false;
    out = {flat/groups, flat%groups};
    return true;
}
} // namespace tnd28_prefill_normal
#endif
