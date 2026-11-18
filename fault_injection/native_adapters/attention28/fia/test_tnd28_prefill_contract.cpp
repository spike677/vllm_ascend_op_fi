#include "tnd28_prefill_contract.h"
#include <array>
#include <cstdlib>
#include <iostream>
#include <limits>

using namespace tnd28_prefill_normal;
static void check(bool value) { if (!value) std::abort(); }
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
