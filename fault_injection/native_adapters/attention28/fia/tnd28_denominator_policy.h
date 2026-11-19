#ifndef DEMONCAT_TND28_DENOMINATOR_POLICY_H
#define DEMONCAT_TND28_DENOMINATOR_POLICY_H
#include "tnd28_prefill_normal.h"
#include "tnd28_denominator_contract.h"

namespace tnd28_denominator {
using tnd28_prefill_normal::Quote;
using tnd28_prefill_normal::Hex;
struct State {
    bool selected=false;
    uint32_t nativeM=0,nativeS2=0,aic=0,aiv=0;
    Scope scope;
};
inline bool UseRoute(gert::TilingContext* c) {
    tnd28_prefill_normal::Scope normal;
    return (Candidate() || tnd28_prefill_normal::CurrentObservationCandidate(normal))
        && tnd28_prefill_normal::Matches(c);
}
inline State Select(const optiling::FiaTilingInfo& i,gert::TilingContext* c,
    uint32_t m,uint32_t s2,uint32_t aic,uint32_t aiv) {
    State s;s.nativeM=m;s.nativeS2=s2;s.aic=aic;s.aiv=aiv;
    if (!Candidate()) return s;
    s.scope=CurrentScope();
    if (!tnd28_prefill_normal::ParsedGeometry(i,c) || m!=512 || s2!=512
        || aic<16 || aic>26 || aiv!=2*aic)
        throw std::runtime_error("TND28_REPLAY_CURRENT_NATIVE_GEOMETRY_DIFFERS");
    Consume(s.scope);s.selected=true;
    return s;
}
inline uint32_t SchedulingCores(const State& s,uint32_t physical) {
    if (!s.selected) return physical;
    RequireInput(s.scope);
    if (physical!=s.aic || physical<16 || physical>26)
        throw std::runtime_error("TND28_REPLAY_PHYSICAL_CORE_IDENTITY_CHANGED");
    // Eight equal KV-head workloads go through the ORIGINAL splitter. Its
    // result still has physical-sized storage. No endpoints or FD flags are
    // manufactured here; RequireSplit must verify one whole head per core.
    return 8;
}
inline void RequireSplit(const State& s,optiling::FusedInferAttentionScoreTilingData& d) {
    if (!s.selected) return;
    RequireInput(s.scope);
    if (!CompleteReduction(s.aic,d.baseParams.get_usedCoreNum(),d.fdParams.get_numOfFdHead(),
        d.outerSplitParams.get_bN2End(),d.outerSplitParams.get_gS1End(),d.outerSplitParams.get_s2End()))
        throw std::runtime_error("TND28_REPLAY_SPLITKV_OR_PARTIAL_REDUCTION_REFUSED");
    // Original SplitCore's FD ledger must really be empty, not just its count.
    for (uint32_t j=0;j<26;++j) {
        if (d.fdParams.get_bN2IdxOfFdHead()[j] || d.fdParams.get_gS1IdxOfFdHead()[j]
            || d.fdParams.get_s2SplitNumOfFdHead()[j] || d.fdParams.get_s2SplitStartIdxOfCore()[j]
            || d.fdParams.get_gS1SplitNumOfFdHead()[j] || d.fdParams.get_gS1LastPartSizeOfFdHead()[j])
            throw std::runtime_error("TND28_REPLAY_NONEMPTY_FD_LEDGER");
    }
    for (uint32_t j=0;j<52;++j)
        if (d.fdParams.get_gS1IdxEndOfFdHead()[j] || d.fdParams.get_gS1IdxEndOfFdHeadSplit()[j])
            throw std::runtime_error("TND28_REPLAY_NONEMPTY_FD_CONSUMER_LEDGER");
}
inline uint64_t ExtendKey(const State& s,uint64_t derived,
    optiling::FusedInferAttentionScoreTilingData& d) {
    if (!s.selected) return derived;
    RequireSplit(s,d);
    if (derived!=kNativeBrcKey || d.innerSplitParams.get_mBaseSize()!=128
        || d.innerSplitParams.get_s2BaseSize()!=128)
        throw std::runtime_error("TND28_REPLAY_HOST_DERIVED_TEMPLATE_DIFFERS");
    // Extend the actual host result, never substitute a key for wrong traits.
    return derived+(s.scope.replay?16000000000000000ULL:6000000000000000ULL);
}
inline void Capture(const State& s,const optiling::FiaTilingInfo& i,gert::TilingContext* c,
    optiling::FusedInferAttentionScoreTilingData& d,uint64_t workspace,
    uint32_t m,uint32_t s2,uint32_t loops,uint32_t tail,uint32_t aligned) {
    if (!s.selected) return;
    RequireSplit(s,d);
    if (!tnd28_prefill_normal::ParsedGeometry(i,c) || m!=128 || s2!=128 || loops!=16 || tail!=128 || aligned!=128
        || c->GetTilingKey()!=(s.scope.replay?kReplayKey:kControlKey)
        || d.innerSplitParams.get_mBaseSize()!=m || d.innerSplitParams.get_s2BaseSize()!=s2
        || d.maskParams.get_sparseMode()!=3 || d.maskParams.get_preToken()!=2147483647
        || d.maskParams.get_nextToken()!=0 || d.baseParams.get_accumQSeqFlag()!=1
        || d.baseParams.get_accumKVSeqFlag()!=1 || d.pageAttenParams.get_blockSize()!=0)
        throw std::runtime_error("TND28_REPLAY_FINAL_NATIVE_GEOMETRY_DIFFERS");
    auto platform=platform_ascendc::PlatformAscendC(c->GetPlatformInfo());
    const uint32_t used=d.baseParams.get_usedCoreNum();
    if (s.aic!=platform.GetCoreNumAic() || s.aiv!=platform.GetCoreNumAiv()
        || c->GetBlockDim()!=platform.CalcTschBlockDim(2*used,used,2*used))
        throw std::runtime_error("TND28_REPLAY_FINAL_CORE_BUDGET_DIFFERS");
    auto* raw=c->GetRawTilingData();
    if (!raw || !raw->GetData() || !raw->GetDataSize() || raw->GetDataSize()>65536 || raw->GetDataSize()!=d.GetDataSize())
        throw std::runtime_error("TND28_REPLAY_PUBLIC_SERIALIZER_SIZE");
    const auto bytes=Hex(raw->GetData(),raw->GetDataSize());
    std::vector<uint64_t> again((d.GetDataSize()+7)/8);d.SaveToBuffer(again.data(),again.size()*8);
    if (bytes!=Hex(again.data(),raw->GetDataSize())) throw std::runtime_error("TND28_REPLAY_PUBLIC_SERIALIZER_ROUNDTRIP");
    const auto& p=s.scope;
    std::ostringstream o;o<<std::boolalpha<<std::setprecision(17);
    o<<"{\"schema\":\"demoncat.attention28.tnd_denominator_host.v1\",\"pid\":"<<getpid()
        <<",\"tid\":"<<syscall(SYS_gettid)<<",\"process_start_ticks\":"<<Quote(p.start)
        <<",\"token\":"<<Quote(p.token)<<",\"owner\":"<<Quote(p.owner)<<",\"generation\":"<<p.generation
        <<",\"phase\":"<<Quote(p.phase)<<",\"target_path\":"<<Quote(p.target)<<",\"module_object_id\":"<<p.objectId
        <<",\"request_sha256\":"<<Quote(p.request)<<",\"input_sha256\":"<<Quote(p.inputHash)
        <<",\"native_m\":"<<s.nativeM<<",\"native_s2\":"<<s.nativeS2
        <<",\"selected_m\":"<<m<<",\"selected_s2\":"<<s2<<",\"s2_loops\":"<<loops
        <<",\"aic\":"<<s.aic<<",\"aiv\":"<<s.aiv<<",\"used_cores\":"<<used
        <<",\"scheduling_cores\":8,\"split_policy\":\"ORIGINAL_SPLIT_CORE_EIGHT_WHOLE_KV_HEADS\""
        <<",\"flash_decode\":false,\"complete_per_row_reduction\":true,\"causal_coordinate_order\":\"S1G\""
        <<",\"denominator_replay\":"<<p.replay<<",\"current_input_gate_required\":"<<p.replay
        <<",\"tiling_key\":"<<Quote(std::to_string(c->GetTilingKey()))<<",\"block_dim\":"<<c->GetBlockDim()
        <<",\"schedule_mode\":"<<c->GetScheduleMode()<<",\"workspace\":"<<workspace
        <<",\"raw_size\":"<<raw->GetDataSize()<<",\"raw_hex\":"<<Quote(bytes)<<",\"typed\":";
    tnd28_prefill_normal::AppendTyped(o,d);
    o<<",\"kernel_argument_raw_observed\":false,\"numerical_effect_observed\":false}\n";
    const auto text=o.str();RequireInput(p);
    if (text.size()>524288) throw std::runtime_error("TND28_REPLAY_CAPTURE_BOUND");
    const int fd=open(p.path.c_str(),O_WRONLY|O_CREAT|O_APPEND|O_NOFOLLOW|O_CLOEXEC,0600);
    if (fd<0) throw std::runtime_error("TND28_REPLAY_CAPTURE_OPEN");
    const auto n=write(fd,text.data(),text.size());const auto closed=close(fd);
    if (n!=static_cast<ssize_t>(text.size()) || closed) throw std::runtime_error("TND28_REPLAY_CAPTURE_WRITE");
}
}
#endif
