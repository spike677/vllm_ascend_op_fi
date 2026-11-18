#ifndef DEMONCAT_TND28_PREFILL_CONTRACT_H
#define DEMONCAT_TND28_PREFILL_CONTRACT_H
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unistd.h>
#include <sys/syscall.h>

extern "C" int dcat_attention_observer_scope_matches_v1(const char*) __attribute__((weak));

// SDK-independent, normal-only host postcondition. No device calculation,
// tiling selection, replay or pointer inference lives here. Observation scope
// uses the current thread plus the existing observer callback; no owner is created.
namespace tnd28_prefill_normal {
inline std::string Env(const char* name) { const char* p=std::getenv(name); return p ? p : ""; }
inline bool HexId(const std::string& value, size_t size) {
    return value.size()==size && value.find_first_not_of("0123456789abcdef")==std::string::npos;
}
inline bool Decimal(const std::string& value, size_t maxSize, bool zero=false) {
    return !value.empty() && value.size()<=maxSize
        && value.find_first_not_of("0123456789")==std::string::npos
        && (value[0]!='0' || (zero && value=="0"));
}
inline std::string ProcessStart() {
    std::ifstream input("/proc/self/stat"); std::string line, field;
    if (!std::getline(input,line)) throw std::runtime_error("TND28_PROCESS_START_UNOBSERVED");
    const auto end=line.rfind(')');
    if (end==std::string::npos) throw std::runtime_error("TND28_PROCESS_STAT_INVALID");
    std::istringstream fields(line.substr(end+1));
    for (unsigned n=3;n<=22;++n) if (!(fields>>field)) throw std::runtime_error("TND28_PROCESS_STAT_SHORT");
    if (!Decimal(field,20)) throw std::runtime_error("TND28_PROCESS_START_INVALID");
    return field;
}
struct Scope {
    std::string mode,token,owner,generation,phase,target,targetSet,objectId,ordinal,path,start;
};
inline Scope ReadScope() {
    Scope s;
    s.mode=Env("DCAT28_TND_MODE"); s.token=Env("DCAT28_TND_TOKEN"); s.owner=Env("DCAT28_TND_OWNER");
    s.generation=Env("DCAT28_TND_GENERATION"); s.phase=Env("DCAT28_TND_PHASE");
    s.target=Env("DCAT28_TND_TARGET"); s.targetSet=Env("DCAT28_TND_TARGET_SET");
    s.objectId=Env("DCAT28_TND_OBJECT"); s.ordinal=Env("DCAT28_TND_ORDINAL");
    s.path=Env("DCAT28_TND_LOG"); s.start=ProcessStart();
    return s;
}
inline Scope CurrentScope() {
    const auto s=ReadScope();
    const bool phase=s.phase=="baseline" || s.phase=="baseline_repeat" || s.phase=="off" || s.phase=="recovery_off";
    if (s.mode!="prefill_normal" || !HexId(s.token,32) || !HexId(s.owner,32)
        || !Decimal(s.generation,18) || !phase || s.target.empty() || s.target.size()>511
        || s.target.find_first_not_of("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._")!=std::string::npos
        || !HexId(s.targetSet,64) || !Decimal(s.objectId,20) || !Decimal(s.ordinal,2,true)
        || std::stoul(s.ordinal)>=28 || s.path.rfind("/evidence/",0)!=0 || s.path.size()>4096
        || s.path.find("/../")!=std::string::npos || s.path.find("/./")!=std::string::npos
        || s.path.back()=='/' || s.path.find('\n')!=std::string::npos || s.path.find('\r')!=std::string::npos
        || Env("DCAT28_TND_PID")!=std::to_string(getpid())
        || Env("DCAT28_TND_TID")!=std::to_string(syscall(SYS_gettid))
        || Env("DCAT28_TND_PROCESS_START")!=s.start
        || !dcat_attention_observer_scope_matches_v1
        || dcat_attention_observer_scope_matches_v1(s.token.c_str())!=1)
        throw std::runtime_error("TND28_NORMAL_CURRENT_OWNER_SCOPE_REQUIRED");
    return s;
}
inline bool SameScope(const Scope& a, const Scope& b) {
    return a.mode==b.mode && a.token==b.token && a.owner==b.owner && a.generation==b.generation
        && a.phase==b.phase && a.target==b.target && a.targetSet==b.targetSet && a.objectId==b.objectId
        && a.ordinal==b.ordinal && a.path==b.path && a.start==b.start;
}
// Unrelated host calls may see process-wide environment values. They must not
// enter strict observation validation or disturb their original tiling result.
inline bool CurrentObservationCandidate(Scope& observed) noexcept {
    try {
        const auto token=Env("DCAT28_TND_TOKEN");
        if (Env("DCAT28_TND_MODE")!="prefill_normal" || !HexId(token,32)
            || Env("DCAT28_TND_PID")!=std::to_string(getpid())
            || Env("DCAT28_TND_TID")!=std::to_string(syscall(SYS_gettid))
            || !dcat_attention_observer_scope_matches_v1
            || dcat_attention_observer_scope_matches_v1(token.c_str())!=1) return false;
        observed=ReadScope();
        return observed.mode=="prefill_normal" && observed.token==token;
    } catch (...) { return false; }
}
inline void RequireCurrentScope(const Scope& scope) {
    if (!SameScope(scope,CurrentScope()))
        throw std::runtime_error("TND28_NORMAL_CAPTURE_OWNER_CHANGED");
}
template<class Geometry, class Body>
inline void CaptureCurrentScope(Geometry geometry, Body body) {
    Scope observed;
    if (!CurrentObservationCandidate(observed)) return;
    bool matches=false;
    try { matches=geometry(); } catch (...) { return; }
    if (!matches) return;
    // Once both current owner and actual geometry match, any further failure
    // is an observation failure, never a fallback to an unrelated-call result.
    const auto scope=CurrentScope();
    if (!SameScope(observed,scope)) throw std::runtime_error("TND28_NORMAL_MATCHED_OWNER_CHANGED");
    body(scope);
}
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
