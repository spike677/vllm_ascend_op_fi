#include "fault33_pa_selector.h"
#include <iostream>
#include <stdexcept>
static void Check(bool p) { if(!p)throw std::runtime_error("selector CPU check failed"); }
int main() {
    try {
        using namespace fault33pa;
        Check(ValidMode("off")&&RequestedInner("off")==0);
        Check(!ValidMode("on")&&!ValidMode("inner0")&&!ValidMode("inner8192"));
        for(auto mode:{"inner512","inner1024"}) {
            const auto r=RequestedInner(mode);Check(ValidMode(mode)&&LegalInner(8192,16384,128,r));
            Check(!LegalInner(r-1,16384,128,r));Check(!LegalInner(8192,r,128,r));
            Check(!LegalInner(8192,16384,112,r));Check(!LegalInner(8192,16384,0,r));
        }
        Check(!LegalInner(8192,16384,128,0));Check(!LegalInner(8192,16384,128,528));
        // Native PA alignment is preserved before selection. A requested inner
        // block is never allowed to grow the native resource allocation.
        Check(LegalInner(2048,2048,128,1024));Check(!LegalInner(512,2048,128,1024));
        int64_t lengths[8]={16384,16384,16384,16384,16384,16384,16384,16384};
        Check(EqualLengths(lengths,1,8,16384));Check(EqualLengths(lengths,8,8,16384));
        Check(!EqualLengths(nullptr,8,8,16384));Check(!EqualLengths(lengths,0,8,16384));
        Check(!EqualLengths(lengths,2,8,16384));Check(!EqualLengths(lengths,9,8,16384));
        lengths[3]=8192;Check(!EqualLengths(lengths,8,8,16384));
        lengths[3]=16384;lengths[0]=16383;Check(!EqualLengths(lengths,1,8,16384));
        std::cout<<"{\"status\":\"PASS_SELECTOR_CPU_CHECK\",\"npu_executed\":false,\"abi_validated\":false}\n";
    } catch(const std::exception& e) { std::cerr<<e.what()<<'\n';return 1; }
}
