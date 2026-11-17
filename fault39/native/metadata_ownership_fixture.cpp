// CPU-only SDK ownership simulation. Production function bodies are inserted
// byte-for-byte by test_metadata_ownership.py; no CANN or NPU is loaded.
#include "metadata_array_owner.h"
#include "fault39_metadata.h"
#include "fault20_buffer_api.h"
#include <algorithm>
#include <cassert>
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <new>
#include <stdexcept>
#include <string>

static bool track_arrays=false;
static void* tracked_arrays[8192]{};
static uint64_t actual_allocations=0,actual_releases=0,actual_live=0;
void* operator new[](std::size_t size) {
    void* value=std::malloc(size?size:1);
    if(!value)throw std::bad_alloc();
    if(track_arrays) {
        bool found=false;
        for(auto& slot:tracked_arrays)if(!slot){slot=value;found=true;break;}
        if(!found)std::abort();
        ++actual_allocations;++actual_live;
    }
    return value;
}
void operator delete[](void* value) noexcept {
    if(!value)return;
    for(auto& slot:tracked_arrays)if(slot==value){slot=nullptr;++actual_releases;--actual_live;break;}
    std::free(value);
}
void operator delete[](void* value,std::size_t) noexcept {::operator delete[](value);}

using aclnnStatus=int;
using aclDataType=int;
using aclFormat=int;
constexpr int ACL_SUCCESS=0,ACL_DT_UNDEFINED=-1,ACL_FORMAT_UNDEFINED=-1;
struct aclStorage{};
struct aclOpExecutor{};
enum class Failure {None,BeforeReturn,AfterReturn,BeforeThrow,AfterThrow};
struct Control {
    int stage{},fail_at{};
    Failure failure{Failure::None};
    uint64_t rank{2};
    bool negative{},overflow{},null_array{},zero_element{},throw_accessor{},throw_log{};
} control;
struct aclTensor {
    const aclStorage* GetStorage()const {
        if(control.throw_accessor)throw std::runtime_error("storage accessor fixture");
        return reinterpret_cast<const aclStorage*>(0x1000);
    }
    void* GetStorageAddr()const{return reinterpret_cast<void*>(0x2000);}
    void* GetData()const{return reinterpret_cast<void*>(0x2000);}
    int64_t Numel()const{return 6;}
    int64_t Size()const{return 6;}
};
static int scalar_status() {
    const bool selected=++control.stage==control.fail_at;
    if(!selected)return 0;
    if(control.failure==Failure::BeforeThrow || control.failure==Failure::AfterThrow)
        throw std::runtime_error("scalar getter fixture");
    return 1;
}
int aclGetDataType(const aclTensor*,aclDataType* out){*out=1;return scalar_status();}
int aclGetFormat(const aclTensor*,aclFormat* out){*out=2;return scalar_status();}
int aclGetViewOffset(const aclTensor*,int64_t* out){*out=0;return scalar_status();}
int aclGetRawTensorAddr(const aclTensor*,void** out){*out=reinterpret_cast<void*>(0x2000);return scalar_status();}
size_t aclDataTypeSize(aclDataType){return control.zero_element?0:2;}
int array_getter(int64_t** output,uint64_t* count,bool strides) {
    const bool selected=++control.stage==control.fail_at;
    if(selected && control.failure==Failure::BeforeReturn)return 1;
    if(selected && control.failure==Failure::BeforeThrow)throw std::runtime_error("before array");
    *count=control.rank;
    if(!control.null_array) {
        *output=new int64_t[std::max<uint64_t>(1,*count)];
        for(uint64_t i=0;i<*count;++i)(*output)[i]=i?3:2;
        if(strides && *count==2){(*output)[0]=3;(*output)[1]=1;}
        if(control.negative && *count)(*output)[0]=-1;
        if(control.overflow && *count)(*output)[0]=std::numeric_limits<int64_t>::max();
    }
    if(selected && control.failure==Failure::AfterReturn)return 1;
    if(selected && control.failure==Failure::AfterThrow)throw std::runtime_error("after array");
    return 0;
}
int aclGetViewShape(const aclTensor*,int64_t** out,uint64_t* count){return array_getter(out,count,false);}
int aclGetStorageShape(const aclTensor*,int64_t** out,uint64_t* count){return array_getter(out,count,false);}
int aclGetViewStrides(const aclTensor*,int64_t** out,uint64_t* count){return array_getter(out,count,true);}
static uint64_t checked_bytes(int64_t n,size_t size) {
    return n<0 || !size || uint64_t(n)>UINT64_MAX/size?0:uint64_t(n)*size;
}
static std::atomic<uint64_t> g_sequence{0};
static uint64_t monotonic_ns(){return 1;}
static long getpid(){return 123;}
static long syscall(long){return 456;}
constexpr long SYS_gettid=1;
static void write_line(const char*,size_t){if(control.throw_log)throw std::runtime_error("log sink fixture");}
constexpr size_t kMaxDims=16;

namespace f20 {
// @F20_CAPTURE@
// @F20_APPEND_DIMS@
// @F20_LOG@
}
namespace f39 {
// @F39_CAPTURE@
}

static void clean() {
    DcatMetadataArrayStateV1 state;
    assert(dcat_metadata::snapshot(&state)==0);
    assert(state.active_captures==0 && state.active_getters==0 && state.live_arrays==0);
    assert(state.acquired_arrays==state.released_arrays && state.ownership_errors==0);
    assert(state.acquired_arrays==actual_allocations && state.released_arrays==actual_releases);
    assert(actual_live==0);
}
static bool capture(int route,const aclTensor* tensor,const char* api="api") {
    if(route==0){Fault20X1Metadata m{};return f20::capture_x1_metadata(tensor,tensor,false,false,0,&m);}
    if(route==1){Fault20TensorMetadata m{};return f20::capture_tensor_metadata(tensor,&m);}
    if(route==2){Fault39TensorMetadata m{};return f39::capture_tensor_metadata(tensor,&m);}
    f20::log_tensor(api,"input",1,nullptr,tensor);return true;
}
int main() {
    track_arrays=true;aclTensor tensor;uint64_t scenarios=0;
    // Full success, every getter failure before/after allocation, and unwind.
    for(int route=0;route<4;++route) {
        control={};assert(capture(route,&tensor));clean();++scenarios;
        for(int stage=1;stage<=7;++stage)for(auto mode:{Failure::BeforeReturn,Failure::AfterReturn,
                                                      Failure::BeforeThrow,Failure::AfterThrow}) {
            control={};control.fail_at=stage;control.failure=mode;
            const bool ok=capture(route,&tensor);if(route!=3)assert(!ok);
            clean();++scenarios;
        }
        control={};assert(route==3 || !capture(route,nullptr));clean();++scenarios;
    }
    // Validation and C++ accessor failures after all getter arrays exist.
    for(int route=0;route<3;++route)for(int defect=0;defect<7;++defect) {
        control={};
        if(defect==0)control.rank=0;
        if(defect==1)control.rank=17;
        if(defect==2)control.negative=true;
        if(defect==3)control.overflow=true;
        if(defect==4)control.null_array=true;
        if(defect==5)control.zero_element=true;
        if(defect==6)control.throw_accessor=true;
        const bool ok=capture(route,&tensor);
        if(!(defect==6 && route==0))assert(!ok);
        clean();++scenarios;
    }
    // log_tensor has formatting early returns and a throwing sink.
    control={};std::string long_api(5000,'a');capture(3,&tensor,long_api.c_str());clean();++scenarios;
    control={};control.throw_log=true;capture(3,&tensor);clean();++scenarios;
    control={};control.null_array=true;capture(3,&tensor);clean();++scenarios;
    // Additive query rejects ABI mistakes; no reset or pointer release export.
    DcatMetadataArrayStateV1 state;
    assert(dcat_metadata::snapshot(nullptr)==1);state.size=0;assert(dcat_metadata::snapshot(&state)==1);
    state={};state.version=2;assert(dcat_metadata::snapshot(&state)==1);
    state={};assert(dcat_metadata::snapshot(&state)==0 && state.instance);
    assert(state.getter_failures && state.getter_exceptions && state.capture_exceptions);
    std::cout<<"{\"scenarios\":"<<scenarios<<",\"acquired_arrays\":"<<state.acquired_arrays
             <<",\"released_arrays\":"<<state.released_arrays<<",\"live_arrays\":"<<state.live_arrays
             <<",\"actual_new_arrays\":"<<actual_allocations<<",\"actual_delete_arrays\":"<<actual_releases<<"}\n";
}
