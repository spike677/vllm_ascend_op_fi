// Explicit CPU fake SDK types. No CANN/torch runtime is loaded by this test.
#include <cstdint>
#include <vector>
#include <string>
#include <cassert>
#include <iostream>
struct Shape {std::vector<int64_t> values;size_t GetDimNum() const{return values.size();}
    int64_t GetDim(size_t i) const{return values.at(i);}};
struct aclTensor {
    Shape view{{2048,8,128}},storage{{2048,4096}};
    std::vector<int64_t> stride{4096,128,1};uint64_t address=0x100000,offset=3072;
    const Shape& GetViewShape() const{return view;}const Shape& GetStorageShape() const{return storage;}
    const std::vector<int64_t>& GetViewStrides() const{return stride;}
    void* GetStorageAddr() const{return reinterpret_cast<void*>(address);}
    void* GetData() const{return reinterpret_cast<void*>(address+2*offset);}
    uint64_t GetViewOffset() const{return offset;}uint64_t GetStorageOffset() const{return 0;}
    int GetDataType() const{return 1;}int GetStorageFormat() const{return 2;}
};
struct aclTensorList {std::vector<const aclTensor*> tensors;size_t Size() const{return tensors.size();}
    const aclTensor* operator[](size_t i) const{return tensors.at(i);}};
struct aclIntArray {std::vector<int64_t> values{2048};size_t Size() const{return values.size();}
    int64_t operator[](size_t i) const{return values.at(i);}};
struct aclOpExecutor {std::vector<uint64_t> offsets{4096};
    const std::vector<uint64_t>& GetWorkspaceOffsets() const{return offsets;}};
using aclnnStatus=int;using aclrtStream=void*;
static std::vector<std::string> tnd28_test_lines;
#define DCAT_TND28_INPUT_CPU_FIXTURE
#include "tnd28_input_observation.h"
static const std::string token(32,'a');
extern "C" int dcat_attention_observer_scope_matches_v1(const char* value){return token==value;}
static aclOpExecutor child,parent;static unsigned prepares=0,copies=0,workspaces=0,runs=0;static int copy_status=0;
static const aclTensorList* supplied=nullptr;static void* expected_workspace=reinterpret_cast<void*>(0x800000);
static aclnnStatus OriginalPrepare(const aclTensorList* list,uint64_t* bytes,aclOpExecutor** executor){
    assert(list==supplied);++prepares;*bytes=4194304+4096;*executor=&child;return 0;
}
static aclnnStatus OriginalWorkspace(void* executor,uint64_t* bytes){assert(executor==&parent);++workspaces;*bytes=8192;return 0;}
static aclnnStatus OriginalCopy(void* workspace,uint64_t bytes,aclOpExecutor* executor,aclrtStream stream){
    assert(workspace==expected_workspace&&bytes==4194304+4096&&executor==&child&&stream==reinterpret_cast<void*>(123));
    ++copies;return copy_status;
}
static aclnnStatus OriginalRun(void* executor,aclrtStream stream,void* workspace,uint64_t bytes){
    assert(executor==&parent&&stream==reinterpret_cast<void*>(123)&&tnd28_input::Ptr(workspace)+bytes==0x800000&&bytes==8192);
    ++runs;return 0;
}
static void Reset(){tnd28_test_lines.clear();tnd28_input::Live()={};tnd28_input::Seen().clear();child.offsets={4096};
    tnd28_input::Plan()={};assert(unsetenv("DCAT28_INPUT_EXECUTE_BORROW")==0);
    prepares=copies=workspaces=runs=0;copy_status=0;
    for(const auto& pair:std::vector<std::pair<std::string,std::string>>{
        {"MODE","observe"},{"TOKEN",token},{"OWNER",std::string(32,'b')},{"GENERATION","1"},{"MODULE","9"},
        {"REQUEST",std::string(64,'c')},{"HASH",std::string(64,'d')},{"PID",std::to_string(getpid())},
        {"TID",std::to_string(syscall(SYS_gettid))},{"START",tnd28_input::ProcessStart()},{"LOG","/evidence/cpu-only.jsonl"}})
        assert(setenv(("DCAT28_INPUT_"+pair.first).c_str(),pair.second.c_str(),1)==0);
}
static void Invoke(bool clone=false,bool changed_offset=false){
    aclTensor q,k,v,copy=v,mask,out,lse;aclTensorList keys{{&k}},values{{&v}},noncontiguous{{clone?&copy:&v}};aclIntArray lengths;
    supplied=&noncontiguous;uint64_t child_bytes=0,bytes=0;aclOpExecutor* actual_child=nullptr;aclOpExecutor* actual_parent=&parent;
    {
        tnd28_input::Prepare p(&OriginalPrepare,&q,&keys,&values,&mask,&lengths,&lengths,&out,&lse,
            {nullptr},{16,2147483647,2147483647,8,3,0,128},0.125,"TND",false);
        assert(p.Contiguous(&OriginalPrepare,&noncontiguous,&child_bytes,&actual_child)==0);
        assert(p.Workspace(&OriginalWorkspace,&parent,&bytes)==0);bytes+=child_bytes;
        assert(p.Return(0,&bytes,&actual_parent)==0);
    }
    if(changed_offset)child.offsets={8192};
    {
        tnd28_input::Execute e(&OriginalRun,&parent);
        auto status=e.Contiguous(&OriginalCopy,expected_workspace,child_bytes,actual_child,reinterpret_cast<void*>(123));
        if(status)return; // Same generated error-return behavior; no forced FIA.
        assert(e.Run(&OriginalRun,&parent,reinterpret_cast<void*>(123),reinterpret_cast<void*>(0x800000-8192),8192)==0);
        assert(e.Return(0)==0);
    }
}
static int callback_status=1;static unsigned callback_count=0;
static int PublicCallback(const char* actual,uint32_t stage,const char* bytes,size_t length){
    assert(token==actual&&(stage==1||stage==2)&&bytes&&length>0&&length<=32768);
    ++callback_count;return callback_status;
}
static bool GuardPrepare(){
    aclTensor q,k,v,mask,out,lse;aclTensorList keys{{&k}},values{{&v}};aclIntArray lengths;
    tnd28_input::Prepare p(&OriginalPrepare,&q,&keys,&values,&mask,&lengths,&lengths,&out,&lse,
        {nullptr},{16,2147483647,2147483647,8,3,0,128},0.125,"TND",false);
    return p.Admit(); // The source patch places this before original parent allocation.
}
int main(){
    Reset();Invoke();assert(prepares==1&&workspaces==1&&copies==1&&runs==1&&tnd28_test_lines.size()==14);
    for(const auto& line:tnd28_test_lines){assert(line.find("\"first_error\":\"\"")!=std::string::npos);
        assert(line.find("\"fault_authorized\":false")!=std::string::npos);}
    assert(!tnd28_input::Live().executor);
    Reset();Invoke(true);assert(prepares==1&&workspaces==1&&copies==1&&runs==1);
    bool seen=false;for(const auto& s:tnd28_test_lines)seen|=s.find("not original V")!=std::string::npos;assert(seen);
    Reset();Invoke(false,true);seen=false;for(const auto& s:tnd28_test_lines)seen|=s.find("offsets changed")!=std::string::npos;assert(seen);
    Reset();copy_status=37;Invoke();assert(copies==1&&runs==0&&tnd28_input::Live().executor);
    Reset();assert(setenv("DCAT28_INPUT_TID","999999999",1)==0);Invoke();assert(tnd28_test_lines.empty());
    assert(prepares==1&&workspaces==1&&copies==1&&runs==1);
    Reset();assert(setenv("DCAT28_INPUT_EXECUTE_BORROW","normal_guard",1)==0);
    assert(!GuardPrepare());assert(prepares==0&&workspaces==0&&copies==0&&runs==0); // no live callback
    Reset();assert(setenv("DCAT28_INPUT_EXECUTE_BORROW","normal_guard",1)==0);
    assert(tnd28_input::Current(tnd28_input::Plan().scope));tnd28_input::Plan().callback=&PublicCallback;
    callback_status=0;callback_count=0;assert(!GuardPrepare());
    assert(callback_count==1&&tnd28_input::Plan().safe_release&&!tnd28_input::Plan().prepare_admitted);
    assert(prepares==0&&workspaces==0&&copies==0&&runs==0);
    Reset();assert(setenv("DCAT28_INPUT_EXECUTE_BORROW","normal_guard",1)==0);
    assert(tnd28_input::Current(tnd28_input::Plan().scope));tnd28_input::Plan().callback=&PublicCallback;
    callback_status=1;callback_count=0;assert(GuardPrepare());assert(tnd28_input::Plan().prepare_admitted);
    assert(tnd28_input::CheckPublic(tnd28_input::Plan().scope,2,tnd28_input::Plan().typed));
    assert(callback_count==2);
    assert(!tnd28_input::CheckPublic(tnd28_input::Plan().scope,2,tnd28_input::Plan().typed));
    assert(callback_count==2); // A repeated stage is rejected before invoking the callback again.
    Reset();auto& read=tnd28_input::Plan();assert(tnd28_input::Current(read.scope));
    read.callback=+[](const char*,uint32_t,const char*,size_t){return 1;};
    read.checking=read.prepare_admitted=read.execute_entered=read.value_check_requested=true;
    read.value_addresses[0]=0x12340000;
    assert(!tnd28_input::ValueReadReady(read,token.c_str(),0,8388608));
    read.value_read_window=true;
    assert(tnd28_input::ValueReadReady(read,token.c_str(),0,8388608));
    assert(!tnd28_input::ValueReadReady(read,token.c_str(),0,8388607));
    assert(!tnd28_input::ValueReadReady(read,token.c_str(),5,8388608));
    assert(!tnd28_input::ValueReadReady(read,std::string(32,'d').c_str(),0,8388608));
    read.value_read_mask=1;assert(!tnd28_input::ValueReadReady(read,token.c_str(),0,8388608));
    read.value_read_mask=0;read.checking=false;
    assert(!tnd28_input::ValueReadReady(read,token.c_str(),0,8388608));
    std::cout<<"PASS_CPU_TND28_TYPED_INPUT_OBSERVATION_NO_DEVICE_NO_F\n";
}
