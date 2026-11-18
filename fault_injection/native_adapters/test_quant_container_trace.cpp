// Compile against actual SDK declarations; all runtime functions below are
// explicit CPU doubles. This tests the real shim, not actual NPU resolution.
#include "quant_container_trace.cpp"
#include <cassert>
#include <iostream>
#include <stdexcept>
#include <thread>

namespace {
struct FakeTensor {int64_t shape[2]{8,2048};};
struct FakeObject {int kind;};
uint64_t calls[4]{};bool fail_destroy{},fail_create{},throw_shape{},partial_shape{};
FakeObject* repeated{};
aclTensorList* create_list(const aclTensor*const*,uint64_t){++calls[0];if(fail_create)return nullptr;
    return reinterpret_cast<aclTensorList*>(repeated?repeated:new FakeObject{1});}
aclnnStatus destroy_list(const aclTensorList* v){++calls[1];if(fail_destroy)return 17;
    if(v && reinterpret_cast<const FakeObject*>(v)!=repeated){delete reinterpret_cast<const FakeObject*>(v);}
    return 0;}
aclIntArray* create_ints(const int64_t*,uint64_t){++calls[2];if(fail_create)return nullptr;
    return reinterpret_cast<aclIntArray*>(new FakeObject{3});}
aclnnStatus destroy_ints(const aclIntArray* v){++calls[3];if(fail_destroy)return 19;
    delete reinterpret_cast<const FakeObject*>(v);return 0;}
aclnnStatus dtype(const aclTensor*,aclDataType* out){*out=ACL_BF16;return 0;}
aclnnStatus format(const aclTensor*,aclFormat* out){*out=ACL_FORMAT_ND;return 0;}
aclnnStatus offset(const aclTensor*,int64_t* out){*out=0;return 0;}
aclnnStatus shape(const aclTensor* p,int64_t** out,uint64_t* n){*n=2;*out=new int64_t[2];
    std::copy(reinterpret_cast<const FakeTensor*>(p)->shape,reinterpret_cast<const FakeTensor*>(p)->shape+2,*out);
    if(throw_shape)throw std::runtime_error("getter threw after assigning owned array");
    return partial_shape?7:0;}
aclnnStatus strides(const aclTensor*,int64_t** out,uint64_t* n){*n=2;*out=new int64_t[2]{2048,1};return 0;}
aclnnStatus storage(const aclTensor*,int64_t** out,uint64_t* n){*n=1;*out=new int64_t[1]{16384};return 0;}
aclnnStatus address(const aclTensor* p,void** out){*out=const_cast<aclTensor*>(p);return 0;}
Config begin(uint64_t max_calls=MaxCalls,uint64_t max_bytes=MaxBytes) {
    std::call_once(api_once,[]{});api.ready=true;
    api.create_list=create_list;api.destroy_list=destroy_list;api.create_ints=create_ints;api.destroy_ints=destroy_ints;
    api.dtype=dtype;api.format=format;api.offset=offset;api.shape=shape;api.strides=strides;api.storage=storage;api.address=address;
    Config c;c.owner_hi=11;c.owner_lo=22;c.generation=1;c.thread=tid();c.max_calls=max_calls;c.max_bytes=max_bytes;
    assert(dcat_qct_begin_v1(&c)==0);return c;
}
State query(){State s;assert(dcat_qct_state_v1(&s)==0);return s;}
void finish(const Config& c){assert(dcat_qct_stop_v1(&c)==0);assert(dcat_qct_release_v1(&c)==0);
    const auto s=query();assert(!s.installed && s.retired && !s.live_registrations && !s.active_calls
        && !s.record_capacity_bytes && !s.payload_capacity_bytes && !s.arrays.live_arrays);}
void test(const std::string& name) {
    auto c=begin(name=="call_budget"?1:MaxCalls,name=="byte_budget"?1:MaxBytes);
    FakeTensor t;const aclTensor* list[]={reinterpret_cast<const aclTensor*>(&t)};
    int64_t ints[]={0,64};
    if(name=="normal" || name=="payload" || name=="scope" || name=="foreign_thread") {
        Scope s;s.token=1;s.forward=7;s.operator_sequence=31;s.boundary_sequence=29;s.stage=3;s.thread=tid();
        if(name=="scope" || name=="foreign_thread")assert(dcat_qct_scope_begin_v1(&c,&s)==0);
        aclTensorList* out=nullptr;
        if(name=="foreign_thread"){std::thread worker([&]{out=aclCreateTensorList(list,1);aclDestroyTensorList(out);});worker.join();}
        else {out=aclCreateTensorList(list,1);assert(out);assert(aclDestroyTensorList(out)==0);}
        auto* integer=aclCreateIntArray(ints,2);assert(integer);assert(aclDestroyIntArray(integer)==0);
        const auto st=query();assert(st.total_calls==4 && st.total_events==8 && st.retained_events==8);
        assert(st.arrays.acquired_arrays==3 && st.arrays.released_arrays==3 && !st.arrays.live_arrays);
        Event e;assert(dcat_qct_event_v1(0,&e)==0);assert(e.original_calls==0 && e.operation==CreateTensors);
        if(name=="payload") {Tensor copy;assert(e.payload_bytes==sizeof(Tensor));
            assert(dcat_qct_payload_v1(e.payload_offset,sizeof(copy),&copy)==0);
            assert(copy.complete && copy.shape[0]==8 && copy.shape[1]==2048 && copy.storage[0]==16384);
            assert(copy.descriptor==reinterpret_cast<uint64_t>(&t));}
        if(name=="scope"){assert(e.scope_token==1 && e.forward==7 && e.stage==3);}
        if(name=="foreign_thread"){assert(!e.scope_token && st.foreign_scope_thread_calls==2 && st.first_error==WrongThread);}
        if(name=="scope" || name=="foreign_thread")assert(dcat_qct_scope_end_v1(&c,1)==0);
        for(unsigned i=0;i<4;++i)assert(calls[i]==1 && st.entries[i]==1 && st.returns[i]==1 && st.original_calls[i]==1);
    } else if(name=="getter_status" || name=="getter_throw") {
        partial_shape=name=="getter_status";throw_shape=name=="getter_throw";
        auto* out=aclCreateTensorList(list,1);assert(out && calls[0]==1);assert(aclDestroyTensorList(out)==0);
        const auto st=query();assert(st.first_error && !st.arrays.live_arrays && st.arrays.acquired_arrays==st.arrays.released_arrays);
        assert(!st.arrays.active_captures && !st.arrays.active_getters && !st.arrays.ownership_errors);
        if(throw_shape)assert(st.abi_exceptions && st.arrays.getter_exceptions==1 && st.arrays.capture_exceptions==1);
        else assert(st.getter_capture_failures==1 && st.arrays.getter_failures==1 && !st.arrays.capture_exceptions);
    } else if(name=="destroy_retry") {
        auto* out=aclCreateIntArray(ints,2);fail_destroy=true;assert(aclDestroyIntArray(out)==19);
        assert(dcat_qct_stop_v1(&c)==0);assert(dcat_qct_release_v1(&c)!=0 && query().live_registrations==1);
        fail_destroy=false;assert(aclDestroyIntArray(out)==0);assert(query().first_error==OriginalFailed && calls[3]==2);
    } else if(name=="unknown_destroy") {
        auto* out=reinterpret_cast<aclIntArray*>(new FakeObject{3});assert(aclDestroyIntArray(out)==0);
        assert(query().first_error==UnknownDestroy && calls[3]==1);
    } else if(name=="null_destroy") {
        assert(aclDestroyIntArray(nullptr)==0);assert(aclDestroyTensorList(nullptr)==0);
        assert(!query().first_error && !query().live_registrations);
    } else if(name=="failed_create") {
        fail_create=true;assert(!aclCreateIntArray(ints,2));assert(query().first_error==OriginalFailed && calls[2]==1);
    } else if(name=="reused_address" || name=="duplicate_live") {
        FakeObject same{1};repeated=&same;
        auto* first=aclCreateTensorList(list,1);Event e;assert(dcat_qct_event_v1(1,&e)==0);const auto registration=e.registration;
        if(name=="reused_address")assert(aclDestroyTensorList(first)==0);
        auto* second=aclCreateTensorList(list,1);assert(first==second);
        if(name=="duplicate_live")assert(query().first_error==DuplicateLive);
        else {assert(dcat_qct_event_v1(5,&e)==0);assert(e.registration!=registration && !query().first_error);}
        assert(aclDestroyTensorList(second)==0);repeated=nullptr;
    } else if(name=="call_budget" || name=="byte_budget" || name=="member_budget") {
        auto* out=aclCreateIntArray(ints,name=="member_budget"?MaxMembers+1:2);assert(out);
        assert(aclDestroyIntArray(out)==0);const auto st=query();assert(st.first_error && calls[2]==1 && calls[3]==1);
        if(name!="member_budget")assert(st.overflow);
    } else if(name=="owner" || name=="scope_retry") {
        Scope s;s.token=1;s.forward=1;s.operator_sequence=2;s.boundary_sequence=3;s.stage=3;s.thread=tid();
        assert(dcat_qct_scope_begin_v1(&c,&s)==0);auto wrong=c;++wrong.generation;
        assert(dcat_qct_scope_end_v1(&wrong,1)!=0 && query().scope_token==1);
        assert(dcat_qct_stop_v1(&c)!=0);assert(dcat_qct_release_v1(&c)!=0);
        assert(dcat_qct_scope_end_v1(&c,2)!=0);assert(dcat_qct_scope_end_v1(&c,1)==0);
        assert(dcat_qct_scope_begin_v1(&c,&s)!=0);++s.token;assert(dcat_qct_scope_begin_v1(&c,&s)==0);
        assert(dcat_qct_scope_end_v1(&c,2)==0);
    } else if(name=="read_bounds") {
        Event e;assert(dcat_qct_event_v1(0,&e)!=0);assert(dcat_qct_payload_v1(UINT64_MAX,1,&e)!=0);
        assert(dcat_qct_payload_v1(0,UINT64_MAX,&e)!=0);
    } else if(name!="empty")assert(false);
    finish(c);assert(dcat_qct_begin_v1(&c)!=0);
}
}
#ifndef DCAT_QCT_NO_TEST_MAIN
int main(int argc,char** argv){assert(argc==2);test(argv[1]);
    std::cout<<"PASS_CPU_SDK_TYPED_FUNCTION_DOUBLE_NOT_NPU "<<argv[1]<<"\n";}
#endif
