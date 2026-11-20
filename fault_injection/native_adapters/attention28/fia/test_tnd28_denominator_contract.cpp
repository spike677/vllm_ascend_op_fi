#define DCAT_TND28_GATE_CPU_FIXTURE
#include "tnd28_denominator_contract.h"
#include "tnd28_causal_tile.h"
#include <cassert>
#include <iostream>
#include <thread>
bool observer=true,input=false;
extern "C" int dcat_attention_observer_scope_matches_v1(const char*) {return observer?1:0;}
#ifndef DCAT_TND28_NO_INPUT_GATE
extern "C" int dcat_tnd28_replay_prepare_gate_v2(const char*,const char*,uint64_t,uint64_t) {return input?1:0;}
#endif
void env(const char* n,const std::string& v){assert(!setenv(n,v.c_str(),1));}
template<class F> bool refused(F f){try{f();return false;}catch(const std::runtime_error&){return true;}}
int main(){
    using namespace tnd28_denominator;
    assert(!Candidate());
    env("DCAT28_TND_MODE","denominator_replay");env("DCAT28_TND_TOKEN",std::string(32,'a'));
    env("DCAT28_TND_OWNER",std::string(32,'b'));env("DCAT28_TND_GENERATION","1");
    env("DCAT28_TND_PHASE","fault");env("DCAT28_TND_TARGET","actual.discovered.object");
    env("DCAT28_TND_OBJECT","121");env("DCAT28_TND_REQUEST_SHA256",std::string(64,'c'));
    env("DCAT28_TND_INPUT_SHA256",std::string(64,'d'));env("DCAT28_TND_LOG","/evidence/cpu/host.jsonl");
    env("DCAT28_TND_PID",std::to_string(getpid()));env("DCAT28_TND_TID",std::to_string(syscall(SYS_gettid)));
    env("DCAT28_TND_PROCESS_START",ProcessStart());
    assert(Candidate());auto s=CurrentScope();
    assert(refused([&]{Consume(s);})); // journal=0/missing binding cannot select F
#ifdef DCAT_TND28_NO_INPUT_GATE
    std::cout<<"PASS_CPU_TND28_MISSING_INPUT_SYMBOL_REFUSED\n";return 0;
#endif
    input=true;Consume(s);assert(refused([&]{Consume(s);}));
    observer=false;assert(!Candidate());assert(refused([&]{RequireInput(s);}));observer=true;
    std::thread foreign([&]{assert(!Candidate());assert(refused([&]{RequireInput(s);}));});foreign.join();
    env("DCAT28_TND_REQUEST_SHA256",std::string(64,'e'));assert(refused([&]{RequireInput(s);}));
    env("DCAT28_TND_REQUEST_SHA256",s.request);
    env("DCAT28_TND_PHASE","control");assert(refused([&]{CurrentScope();}));env("DCAT28_TND_PHASE","fault");
    env("DCAT28_TND_OBJECT","122");assert(refused([&]{RequireInput(s);}));env("DCAT28_TND_OBJECT","121");
    uint32_t h[26]{},r[26]{},b[26]{};
    for(unsigned i=0;i<8;++i)h[i]=i+1;
    for(unsigned physical=16;physical<=26;++physical)assert(CompleteReduction(physical,8,0,h,r,b));
    assert(!CompleteReduction(16,16,0,h,r,b));
    assert(!CompleteReduction(8,8,0,h,r,b));
    b[2]=1;assert(!CompleteReduction(16,8,0,h,r,b));b[2]=0;
    assert(!CompleteReduction(16,8,1,h,r,b));
    h[8]=1;assert(!CompleteReduction(16,8,0,h,r,b));h[8]=0;
    h[0]=0;r[0]=16;assert(!CompleteReduction(16,8,0,h,r,b));h[0]=1;r[0]=0;
    r[4]=r[3];h[4]=h[3];assert(!CompleteReduction(16,8,0,h,r,b));
    for(uint32_t flat=0;flat<4096;flat+=128){
        const uint32_t last=(flat+127)/2;
        for(uint32_t k=0;k<16;++k){
            const auto width=tnd28_causal_tile::InnerWidth(flat,128,2048,2,k,128,128);
            assert(width<=128 && width%32==0);
            for(uint32_t row=flat;row<flat+128;++row)
                for(uint32_t key=k*128;key<(k+1)*128 && key<=row/2;++key)
                    assert(key<k*128+width);
            if(k*128>last) assert(width==0);
        }
    }
    assert(tnd28_causal_tile::InnerWidth(2048,128,2048,2,8,128,128)==64);
    std::cout<<"PASS_CPU_TND28_DENOMINATOR_CONTRACT_NOT_NPU\n";
}
