// CPU proof of logical causal coverage using the actual packaged width helper.
// This does not establish device-template legality or NPU numerical acceptance.
#include "fault28_causal_tile.h"
#include <cassert>
#include <cstdint>
#include <iostream>

int main() {
    uint64_t queries=0;
    for (uint32_t m : {32U,64U,128U,256U,512U}) {
        for (uint32_t n : {m,512U}) {
            assert(m*n<=512U*512U && m%32U==0 && n%32U==0);
            for (uint32_t group=0;group<2;++group) {
                for (uint32_t q=0;q<2048;++q) {
                    const uint32_t first=(q/m)*m;
                    const uint32_t end=first+m;
                    uint32_t next=0;
                    for(uint32_t begin=0;begin<end;begin+=n) {
                        const auto width=fault28_causal_tile::InnerWidth(
                            group*2048U+first,m,2048U,begin/n,n,n);
                        assert(width>0 && width<=n && width%32U==0 && begin+width<=2048U);
                        const uint32_t stop=begin+width<q+1U?begin+width:q+1U;
                        if(begin<stop) { assert(begin==next);next=stop; }
                    }
                    assert(next==q+1U);++queries;
                }
            }
        }
    }
    assert(fault28_causal_tile::InnerWidth(0,512,2048,0,512,512)==512);
    assert(fault28_causal_tile::InnerWidth(0,128,2048,0,512,512)==128);
    assert(fault28_causal_tile::InnerWidth(0,32,2048,0,512,512)==32);
    std::cout<<"CPU_CAUSAL_COVERAGE_PASS queries="<<queries<<" NPU_ACCEPTED=false\n";
}
