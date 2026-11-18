#include "tnd28_prefill_contract.h"
#include <array>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <thread>

using namespace tnd28_prefill_normal;
static void check(bool value) { if (!value) std::abort(); }
static thread_local bool observer_owns=true;
extern "C" int dcat_attention_observer_scope_matches_v1(const char* token) {
    return observer_owns && std::string(token)==std::string(32,'a') ? 1 : 0;
}
static void env(const char* name, const std::string& value) { check(setenv(name,value.c_str(),1)==0); }
static void scope_contract() {
    env("DCAT28_TND_MODE","prefill_normal"); env("DCAT28_TND_TOKEN",std::string(32,'a'));
    env("DCAT28_TND_OWNER",std::string(32,'b')); env("DCAT28_TND_GENERATION","1");
    env("DCAT28_TND_PHASE","baseline"); env("DCAT28_TND_TARGET","model.layers.0.attn");
    env("DCAT28_TND_TARGET_SET",std::string(64,'c')); env("DCAT28_TND_OBJECT","1234");
    env("DCAT28_TND_ORDINAL","0"); env("DCAT28_TND_LOG","/evidence/cpu-not-written.jsonl");
    env("DCAT28_TND_PID",std::to_string(getpid()));
    env("DCAT28_TND_TID",std::to_string(syscall(SYS_gettid)));
    env("DCAT28_TND_PROCESS_START",ProcessStart());
    unsigned geometry=0,captures=0;
    auto yes=[&] { ++geometry; return true; };
    auto body=[&](const Scope& scope) { ++captures; RequireCurrentScope(scope); };
    CaptureCurrentScope(yes,body); check(geometry==1 && captures==1);
    // An actual other thread sees the same environment but cannot consume it.
    std::thread foreign([&] { CaptureCurrentScope(yes,body); }); foreign.join();
    check(geometry==1 && captures==1);
    observer_owns=false; CaptureCurrentScope(yes,body); observer_owns=true;
    env("DCAT28_TND_TOKEN",std::string(32,'d')); CaptureCurrentScope(yes,body);
    env("DCAT28_TND_TOKEN",std::string(32,'a')); check(geometry==1 && captures==1);
    env("DCAT28_TND_PHASE","fault");
    CaptureCurrentScope([&] { ++geometry; return false; },body);
    CaptureCurrentScope([]()->bool { throw std::runtime_error("unmatched geometry"); },body);
    check(geometry==2 && captures==1);
    bool rejected=false;
    try { CaptureCurrentScope(yes,body); } catch (const std::runtime_error&) { rejected=true; }
    check(rejected && captures==1); env("DCAT28_TND_PHASE","baseline");
    rejected=false;
    try { CaptureCurrentScope([] { observer_owns=false; return true; },body); }
    catch (const std::runtime_error&) { rejected=true; }
    observer_owns=true; check(rejected && captures==1);
    rejected=false;
    try { CaptureCurrentScope([] { env("DCAT28_TND_OWNER",std::string(32,'c')); return true; },body); }
    catch (const std::runtime_error&) { rejected=true; }
    check(rejected && captures==1); env("DCAT28_TND_OWNER",std::string(32,'b'));
    rejected=false;
    try { CaptureCurrentScope(yes,[&](const Scope& scope) {
        ++captures; env("DCAT28_TND_GENERATION","2"); RequireCurrentScope(scope);
    }); } catch (const std::runtime_error&) { rejected=true; }
    check(rejected && captures==2); env("DCAT28_TND_GENERATION","1");
    CaptureCurrentScope(yes,body); check(captures==3);
}
static NormalTiling fixture() {
    NormalTiling n;
    n.key=kNormalKey; n.s1=n.s2=2048; n.groups=2; n.kvHeads=8;
    n.m=n.s2Base=n.tail=n.aligned=512; n.loops=4;
    // Explicit CPU platform fixture, not a claim about the real platform.
    n.aic=n.usedCores=20; n.aiv=40; n.blockDim=n.calculatedBlockDim=20;
    n.preToken=2147483647; n.nextToken=0;
    n.rawBlockSize=128; n.parsedBlockSize=0;
    n.cumulativeQ=n.cumulativeKv=true; n.flashDecode=n.brcb=false;
    return n;
}
int main() {
    scope_contract();
    check(NormalPostcondition(fixture()));
    check(!NormalPostcondition(NormalTiling{}));
    const std::array<uint32_t NormalTiling::*,14> fields = {
        &NormalTiling::s1,&NormalTiling::s2,&NormalTiling::groups,&NormalTiling::kvHeads,
        &NormalTiling::m,&NormalTiling::s2Base,&NormalTiling::loops,&NormalTiling::tail,
        &NormalTiling::aligned,&NormalTiling::aic,&NormalTiling::aiv,
        &NormalTiling::usedCores,&NormalTiling::blockDim,&NormalTiling::calculatedBlockDim};
    for (auto field:fields) {
        auto n=fixture(); n.*field=0; check(!NormalPostcondition(n));
        n=fixture(); n.*field=std::numeric_limits<uint32_t>::max(); check(!NormalPostcondition(n));
    }
    for (auto field:{&NormalTiling::cumulativeQ,&NormalTiling::cumulativeKv,
                    &NormalTiling::flashDecode,&NormalTiling::brcb}) {
        auto n=fixture(); n.*field=!(n.*field); check(!NormalPostcondition(n));
    }
    for (auto key:{103000000010300003ULL,109000000000000000ULL,119000000000400000ULL}) {
        auto n=fixture(); n.key=key; check(!NormalPostcondition(n));
    }
    auto n=fixture(); n.m=128; check(!NormalPostcondition(n));
    n=fixture(); n.s2Base=128; check(!NormalPostcondition(n));
    n=fixture(); n.preToken=0; check(!NormalPostcondition(n));
    n=fixture(); n.nextToken=2147483647; check(!NormalPostcondition(n));
    n=fixture(); n.rawBlockSize=16; check(!NormalPostcondition(n));
    n=fixture(); n.parsedBlockSize=128; check(!NormalPostcondition(n));
    for (uint32_t aic=1;aic<=26;++aic) {
        n=fixture(); n.aic=aic; n.aiv=2*aic; n.usedCores=1;
        n.blockDim=n.calculatedBlockDim=1; check(NormalPostcondition(n));
    }
    uint64_t causal=0;
    for (uint32_t q=0;q<2048;++q) for (uint32_t g=0;g<2;++g) {
        Coordinate c;
        check(TndCoordinate(q*2+g,2048,2,c) && c.query==q && c.group==g);
        causal+=q+1;
    }
    check(causal==4196352ULL);
    Coordinate c{99,99};
    check(!TndCoordinate(4096,2048,2,c) && c.query==99 && c.group==99);
    check(!TndCoordinate(0,0,2,c)); check(!TndCoordinate(0,2048,0,c));
    check(!TndCoordinate(0,2048,3,c));
    std::cout<<"PASS_CPU_TND28_NORMAL_GEOMETRY_COORDINATES_NOT_SDK_OR_NPU\n";
}
