#include "fa_bridge.h"
#include <fstream>
#include <iostream>
int main(int argc,char** argv) {
    try {
        nesfa::Require(argc==2,"Need synthetic plan fixture");std::ifstream f(argv[1]);nesfa::Json j;f>>j;
        j["scope"]["pid"]=getpid();j["scope"]["native_tid"]=syscall(SYS_gettid);
        const auto good=j;auto* sentinel=reinterpret_cast<gert::TilingContext*>(uintptr_t{0x1000});
        auto begin=[&](const nesfa::Json& plan){const auto value=plan.dump();setenv("NES_FA_PLAN_JSON",value.c_str(),1);nesfa::Begin(sentinel);};
        setenv("NES_FA_LOG_PATH","/tmp/unused_cpu_scope.jsonl",1);begin(good);nesfa::Require(nesfa::state.active,"Valid scope rejected");
        j=good;j["scope"]["pid"]=0;begin(j);nesfa::Require(!nesfa::state.active,"Wrong PID accepted");
        j=good;j["scope"]["native_tid"]=syscall(SYS_gettid)+1;begin(j);nesfa::Require(!nesfa::state.active,"Wrong TID accepted");
        setenv("NES_FA_LOG_PATH","relative.jsonl",1);begin(good);nesfa::Require(!nesfa::state.active,"Relative evidence path accepted");
        unsetenv("NES_FA_PLAN_JSON");nesfa::Begin(sentinel);nesfa::Require(!nesfa::state.active,"Missing plan accepted");
        setenv("NES_FA_LOG_PATH","/tmp/unused_cpu_scope.jsonl",1);begin(good);nesfa::Begin(nullptr);nesfa::Require(!nesfa::state.active,"Null context accepted");
        gert::StorageShape shape;shape.MutableStorageShape().SetDimNum(4);int64_t dims[]={1,16,2048,128};
        for(size_t i=0;i<4;++i)shape.MutableStorageShape().SetDim(i,dims[i]);
        nesfa::Require(nesfa::Shape(&shape,{1,16,2048,128})&&!nesfa::Shape(&shape,{8,16,2048,128}),"Actual shape helper");
        gert::StorageShape rank0,empty,present;empty.MutableStorageShape().SetDimNum(1);empty.MutableStorageShape().SetDim(0,0);
        present.MutableStorageShape().SetDimNum(1);present.MutableStorageShape().SetDim(0,1);
        nesfa::Require(nesfa::NativeTensorOptionAbsent(&rank0)&&nesfa::NativeTensorOptionAbsent(&empty)&&!nesfa::NativeTensorOptionAbsent(&present),"Pinned optional tensor semantics");
        std::cout<<"{\"status\":\"PASS_CPU_SCOPE_AND_PUBLIC_SHAPE\",\"valid_scopes\":1,\"invalid_scopes\":5,\"shape_checks\":2,\"optional_checks\":3,\"npu_executed\":false,\"full_gert_context_tested\":false}\n";
    }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
}
