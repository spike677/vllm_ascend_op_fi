#include "fault28_causal_tile.h"
#include <cassert>
#include <cstdint>
#include <iostream>
#include <vector>

int main() {
    uint64_t queries=0, pruned=0;
    for (uint32_t m:{128U,256U,512U}) {
        for(uint32_t headGroup=0;headGroup<2;++headGroup) {
            for(uint32_t q=0;q<2048;++q) {
                const uint32_t first=(q/m)*m;
                const uint32_t last=first+m-1;
                std::vector<unsigned> valid(2048,0);
                for(uint32_t kBlock=0;kBlock<=last/512;++kBlock) {
                    const uint32_t n=fault28_causal_tile::InnerWidth(headGroup*2048+first,m,2048,kBlock,512,512);
                    assert(n>0&&n<=512&&n%32==0);
                    pruned+=n<512;
                    for(uint32_t j=0;j<n;++j) {
                        uint32_t k=kBlock*512+j;
                        assert(k<=last&&k<2048);
                        if(k<=q)++valid[k];
                    }
                    for(uint32_t j=n;j<512;++j)assert(kBlock*512+j>last);
                    if(m==512)assert(n==512);
                }
                for(uint32_t k=0;k<2048;++k)assert(valid[k]==(k<=q?1U:0U));
                ++queries;
            }
        }
    }
    assert(fault28_causal_tile::InnerWidth(0,0,2048,0,512,512)==0);
    assert(fault28_causal_tile::InnerWidth(2047,128,2048,0,512,512)==0);
    assert(fault28_causal_tile::InnerWidth(0,128,2048,1,512,512)==0);
    std::cout<<"{\"status\":\"PASS_CPU_CAUSAL_KEY_COVERAGE\",\"checked_queries\":"<<queries
             <<",\"query_partial_inner_groups\":"<<pruned<<",\"m_values\":[128,256,512],"
               "\"legal_keys_exactly_once\":true,\"removed_columns_all_future_masked\":true,"
               "\"m512_matches_native_inner_width\":true,\"npu_executed\":false}\n";
}
