#include "fault28_causal_policy.h"
#include <iostream>
#include <stdexcept>
int main() {
    auto* sentinel = reinterpret_cast<gert::TilingContext*>(uintptr_t{0x1000});
    const std::vector<std::pair<std::string,std::string>> good = {
      {"FAULT28_CAUSAL_PROFILE","base"}, {"FAULT28_CAUSAL_SCOPE_PID",std::to_string(getpid())},
      {"FAULT28_CAUSAL_SCOPE_TID",std::to_string(syscall(SYS_gettid))},
      {"FAULT28_CAUSAL_SCOPE_PHASE","operator_replay"}, {"FAULT28_CAUSAL_SCOPE_BATCH","8"},
      {"FAULT28_CAUSAL_SCOPE_TOKEN","replay_001"}, {"FAULT28_CAUSAL_LOG_PATH","/tmp/unused.jsonl"}};
    auto reset=[&](){for(const auto& kv:good)setenv(kv.first.c_str(),kv.second.c_str(),1);};
    try {
      size_t scopes=0,resolutions=0;
      for(const auto& p:{"base","m_only","r_only","joint","batch_m","batch_r","batch_joint"}) {
        for(const auto& b:{"1","8"}) {
          reset();setenv("FAULT28_CAUSAL_PROFILE",p,1);setenv("FAULT28_CAUSAL_SCOPE_BATCH",b,1);
          fault28causal::Begin(sentinel);if(!fault28causal::state.active)throw std::runtime_error("valid scope");++scopes;
          std::string expected=p;if(expected=="batch_m")expected=std::string(b)=="1"?"base":"m_only";
          if(expected=="batch_r")expected=std::string(b)=="1"?"base":"r_only";
          if(expected=="batch_joint")expected=std::string(b)=="1"?"base":"joint";
          if(fault28causal::ResolveProfile(p,std::stoll(b))!=expected)throw std::runtime_error("profile resolution");++resolutions;
        }
      }
      const std::vector<std::pair<std::string,std::string>> bad={
        {"FAULT28_CAUSAL_PROFILE","on"},{"FAULT28_CAUSAL_SCOPE_PID","0"},{"FAULT28_CAUSAL_SCOPE_TID","0"},
        {"FAULT28_CAUSAL_SCOPE_PHASE","prefill"},{"FAULT28_CAUSAL_SCOPE_PHASE","isolated_replay"},
        {"FAULT28_CAUSAL_SCOPE_BATCH","2"},{"FAULT28_CAUSAL_SCOPE_BATCH","08"},{"FAULT28_CAUSAL_SCOPE_BATCH",""},
        {"FAULT28_CAUSAL_SCOPE_TOKEN","bad\"token"},{"FAULT28_CAUSAL_SCOPE_TOKEN",""},
        {"FAULT28_CAUSAL_SCOPE_TOKEN",std::string(129,'a')},{"FAULT28_CAUSAL_LOG_PATH","relative"}};
      for(const auto& kv:bad){reset();setenv(kv.first.c_str(),kv.second.c_str(),1);fault28causal::Begin(sentinel);if(fault28causal::state.active)throw std::runtime_error(kv.first);}
      reset();fault28causal::Begin(nullptr);if(fault28causal::state.active)throw std::runtime_error("null context");
      if(!fault28causal::ResolveProfile("joint",2).empty()||!fault28causal::ResolveProfile("unknown",8).empty())throw std::runtime_error("invalid resolution");
      gert::StorageShape shape;shape.MutableStorageShape().SetDimNum(4);
      int64_t dims[]={8,16,2048,128};for(size_t i=0;i<4;++i)shape.MutableStorageShape().SetDim(i,dims[i]);
      if(!fault28causal::Shape(&shape,{8,16,2048,128})||fault28causal::Shape(&shape,{1,16,2048,128})||fault28causal::Shape(nullptr,{8,16,2048,128}))throw std::runtime_error("shape");
      gert::StorageShape rank0,empty,present;empty.MutableStorageShape().SetDimNum(1);empty.MutableStorageShape().SetDim(0,0);
      present.MutableStorageShape().SetDimNum(1);present.MutableStorageShape().SetDim(0,1);
      if(rank0.GetStorageShape().GetShapeSize()!=1||!fault28causal::NativeTensorOptionAbsent(&rank0)||!fault28causal::NativeTensorOptionAbsent(&empty)||fault28causal::NativeTensorOptionAbsent(&present)||!fault28causal::NativeTensorOptionAbsent(nullptr))throw std::runtime_error("optional");
      std::cout<<"{\"status\":\"PASS_CPU_SCOPE_PROFILE_AND_SHAPE\",\"valid_scopes\":"<<scopes<<",\"profile_resolutions\":"<<resolutions<<",\"invalid_scopes\":13,\"invalid_resolutions\":2,\"shape_checks\":3,\"optional_checks\":4,\"npu_executed\":false,\"full_gert_eligibility_tested\":false}\n";
    } catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
}
