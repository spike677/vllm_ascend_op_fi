#include "fault28_fa_policy.h"
#include <iostream>
#include <stdexcept>
#include <vector>
int main() {
    // Scope-only opaque sentinel: Begin performs no context dereference.
    auto* sentinel = reinterpret_cast<gert::TilingContext*>(uintptr_t{0x1000});
    const std::vector<std::pair<std::string,std::string>> good = {
        {"FAULT28_FA_MODE","off"}, {"FAULT28_FA_SCOPE_PID",std::to_string(getpid())},
        {"FAULT28_FA_SCOPE_TID",std::to_string(syscall(SYS_gettid))},
        {"FAULT28_FA_SCOPE_LAYER","1"}, {"FAULT28_FA_SCOPE_PHASE","prefill"},
        {"FAULT28_FA_SCOPE_FORWARD","0"}, {"FAULT28_FA_SCOPE_TOKEN","f0_layer1_sdpa_ab12"},
        {"FAULT28_FA_LOG_PATH","/tmp/fault28-scope-only-unused.jsonl"}};
    auto reset=[&](){for(const auto& kv:good)setenv(kv.first.c_str(),kv.second.c_str(),1);};
    try {
        reset();fault28fa::Begin(sentinel);if(!fault28fa::state.active)throw std::runtime_error("valid off scope");
        setenv("FAULT28_FA_MODE","on",1);fault28fa::Begin(sentinel);if(!fault28fa::state.active)throw std::runtime_error("valid on scope");
        const std::vector<std::pair<std::string,std::string>> invalid = {
            {"FAULT28_FA_MODE","ON"},{"FAULT28_FA_SCOPE_PID","0"},{"FAULT28_FA_SCOPE_TID","0"},
            {"FAULT28_FA_SCOPE_LAYER","0"},{"FAULT28_FA_SCOPE_PHASE","decode"},
            {"FAULT28_FA_SCOPE_FORWARD","-1"},{"FAULT28_FA_SCOPE_FORWARD",""},
            {"FAULT28_FA_SCOPE_TOKEN","bad\"token"},{"FAULT28_FA_SCOPE_TOKEN",""},
            {"FAULT28_FA_SCOPE_TOKEN",std::string(129,'a')},{"FAULT28_FA_LOG_PATH","relative.jsonl"}};
        for(const auto& kv:invalid){reset();setenv(kv.first.c_str(),kv.second.c_str(),1);fault28fa::Begin(sentinel);if(fault28fa::state.active)throw std::runtime_error(kv.first);}
        reset();fault28fa::Begin(nullptr);if(fault28fa::state.active)throw std::runtime_error("null context");
        // Directly exercise the actual public StorageShape comparison helper.
        gert::StorageShape shape;
        shape.MutableStorageShape().SetDimNum(4);
        const int64_t dims[]={8,16,2048,128};for(size_t i=0;i<4;i++)shape.MutableStorageShape().SetDim(i,dims[i]);
        if(!fault28fa::Shape(&shape,{8,16,2048,128}) || fault28fa::Shape(&shape,{1,16,2048,128}) || fault28fa::Shape(nullptr,{8,16,2048,128}))throw std::runtime_error("shape guard");
        std::cout << "{\"status\":\"PASS_CPU_SCOPE_AND_SHAPE_ONLY\",\"valid_scopes\":2,\"invalid_scopes\":12,\"shape_checks\":3,\"npu_executed\":false,\"full_gert_eligibility_tested\":false}\n";
        return 0;
    }catch(const std::exception& e){std::cerr << e.what() << '\n';return 1;}
}
