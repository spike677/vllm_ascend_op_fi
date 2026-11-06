#ifndef FAULT33_PA_SELECTOR_H
#define FAULT33_PA_SELECTOR_H
#include <cstdint>
#include <string>
namespace fault33pa {
// This selector is an experimental host policy, not an existing stock version.
// Selecting a smaller PA-aligned block leaves all dependent native tiling work
// downstream of CalcInnerSize (loop/tail, buffers, softmax, MM, workspace, key).
inline uint32_t RequestedInner(const std::string& mode) {
    if (mode == "inner512") return 512;
    if (mode == "inner1024") return 1024;
    return 0;
}
inline bool ValidMode(const std::string& mode) { return mode == "off" || RequestedInner(mode) != 0; }
inline bool LegalInner(uint32_t native, uint32_t seq, uint32_t block, uint32_t requested) {
    return block == 128 && (requested == 512 || requested == 1024)
        && requested <= native && requested < seq && requested % block == 0;
}
inline bool EqualLengths(const int64_t* lengths,int64_t count,uint32_t batch,uint32_t length) {
    if(!lengths||(count!=1&&count!=static_cast<int64_t>(batch)))return false;
    for(int64_t i=0;i<count;++i)if(lengths[i]!=static_cast<int64_t>(length))return false;
    return true;
}
} // namespace fault33pa
#endif
