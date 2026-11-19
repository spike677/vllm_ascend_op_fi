#ifndef DEMONCAT_TND28_DENOMINATOR_CONTRACT_H
#define DEMONCAT_TND28_DENOMINATOR_CONTRACT_H
#include "tnd28_prefill_contract.h"
#include <array>
#include <set>

// A separately reviewed live input adapter must implement this symbol. A
// journal window, saved report, bool from RPC or Python address/shape guess
// cannot implement it. No current deployment supplies it; F stays refused.
// The return must mean: THIS live original call's public operands, typed GWS
// inputs and prepared kernel inputs have been joined by actual observations.
extern "C" int dcat_tnd28_denominator_current_input_v1(
    const char*, const char*, uint64_t, uint64_t) __attribute__((weak));

namespace tnd28_denominator {
using tnd28_prefill_normal::Env;
using tnd28_prefill_normal::HexId;
using tnd28_prefill_normal::Decimal;
using tnd28_prefill_normal::ProcessStart;
constexpr uint64_t kNativeKey=103000000030000003ULL;
constexpr uint64_t kNativeBrcKey=103000000030400003ULL;
constexpr uint64_t kControlKey=109000000030400003ULL;
constexpr uint64_t kReplayKey=119000000030400003ULL;
struct Scope {
    std::string token,owner,generation,phase,target,request,inputHash,path,start,objectId;
    bool replay=false;
};
inline bool Candidate() {
    const auto mode=Env("DCAT28_TND_MODE"), token=Env("DCAT28_TND_TOKEN");
    return (mode=="denominator_control" || mode=="denominator_replay") && HexId(token,32)
        && Env("DCAT28_TND_PID")==std::to_string(getpid())
        && Env("DCAT28_TND_TID")==std::to_string(syscall(SYS_gettid))
        && dcat_attention_observer_scope_matches_v1
        && dcat_attention_observer_scope_matches_v1(token.c_str())==1;
}
inline Scope CurrentScope() {
    if (!Candidate()) throw std::runtime_error("TND28_REPLAY_CURRENT_SCOPE_REQUIRED");
    Scope s;
    s.token=Env("DCAT28_TND_TOKEN");s.owner=Env("DCAT28_TND_OWNER");
    s.generation=Env("DCAT28_TND_GENERATION");s.phase=Env("DCAT28_TND_PHASE");
    s.target=Env("DCAT28_TND_TARGET");s.objectId=Env("DCAT28_TND_OBJECT");
    s.request=Env("DCAT28_TND_REQUEST_SHA256");s.inputHash=Env("DCAT28_TND_INPUT_SHA256");
    s.path=Env("DCAT28_TND_LOG");s.start=ProcessStart();
    s.replay=Env("DCAT28_TND_MODE")=="denominator_replay";
    if (!HexId(s.owner,32) || !Decimal(s.generation,18) || !Decimal(s.objectId,20)
        || !HexId(s.request,64) || !HexId(s.inputHash,64) || s.target.empty() || s.target.size()>511
        || s.target.find_first_not_of("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._")!=std::string::npos
        || (s.replay?s.phase!="fault":s.phase!="control")
        || Env("DCAT28_TND_PROCESS_START")!=s.start || s.path.rfind("/evidence/",0)!=0
        || s.path.size()>4096 || s.path.find("/../")!=std::string::npos || s.path.find("/./")!=std::string::npos
        || s.path.back()=='/' || s.path.find_first_of("\r\n")!=std::string::npos)
        throw std::runtime_error("TND28_REPLAY_OWNER_REQUEST_SCOPE_DIFFERS");
    return s;
}
inline bool Same(const Scope& a,const Scope& b) {
    return a.token==b.token && a.owner==b.owner && a.generation==b.generation
        && a.phase==b.phase && a.target==b.target && a.objectId==b.objectId && a.request==b.request
        && a.inputHash==b.inputHash && a.path==b.path && a.start==b.start && a.replay==b.replay;
}
inline void RequireInput(const Scope& s) {
    if (!Same(s,CurrentScope())) throw std::runtime_error("TND28_REPLAY_SCOPE_CHANGED");
    if (s.replay && (!dcat_tnd28_denominator_current_input_v1
        || dcat_tnd28_denominator_current_input_v1(s.token.c_str(),s.inputHash.c_str(),
            std::stoull(s.generation),std::stoull(s.objectId))!=1))
        throw std::runtime_error("TND28_REPLAY_ACTUAL_PREPARED_INPUT_BINDING_MISSING");
}
// Whole causal S2 reductions only, interpreted from the original SplitCore
// endpoints. The matrices retain all original FD arrays; no FD flag is edited.
inline bool CompleteReduction(uint32_t cores,uint32_t used,uint32_t fd,
    const uint32_t* heads,const uint32_t* rows,const uint32_t* blocks) {
    if (!heads || !rows || !blocks || cores<16 || cores>26 || used!=8 || fd) return false;
    uint32_t previous=0;
    for (uint32_t i=0;i<used;++i) {
        if (heads[i]!=i+1 || rows[i]!=0 || blocks[i]!=0) return false;
        const uint32_t end=heads[i]*32+rows[i];
        if (end<=previous || end>256) return false;
        previous=end;
    }
    if (previous!=256) return false;
    for (uint32_t i=used;i<cores;++i) if (heads[i] || rows[i] || blocks[i]) return false;
    return true;
}
inline void Consume(const Scope& s) {
    // Each observer UUID belongs to one real invocation. A second matched
    // host invocation is rejected, including after an earlier failed tiling.
    static thread_local std::set<std::string> consumed;
    RequireInput(s);
    if (consumed.size()>=16 || !consumed.insert(s.token).second)
        throw std::runtime_error("TND28_REPLAY_DUPLICATE_OR_EXHAUSTED_CALL");
}
}
#endif
