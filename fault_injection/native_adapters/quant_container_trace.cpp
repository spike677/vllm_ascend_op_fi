// Four public container APIs only. No GWS interception, private layout, device
// data copy, replacement, or fault operation is implemented by this library.
#include "quant_container_trace.h"
// This existing shared header is byte-pinned by the build profile. Its compact
// formatting triggers GCC's indentation warning; keep our own TU strict.
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wmisleading-indentation"
#include "../../fault39/native/runtime_identity.h"
#pragma GCC diagnostic pop
#include "../../fault39/native/metadata_array_owner.h"
#include "quant_container_profile.h"
#include <aclnn/acl_meta.h>
#include <atomic>
#include <chrono>
#include <map>
#include <mutex>
#include <sys/syscall.h>
#include <type_traits>
#include <unistd.h>
#include <vector>

namespace {
using namespace dcat_qct;
using CreateList=aclTensorList*(*)(const aclTensor*const*,uint64_t);
using DestroyList=aclnnStatus(*)(const aclTensorList*);
using CreateInts=aclIntArray*(*)(const int64_t*,uint64_t);
using DestroyInts=aclnnStatus(*)(const aclIntArray*);
static_assert(std::is_same<decltype(&aclCreateTensorList),CreateList>::value,"actual SDK list creator changed");
static_assert(std::is_same<decltype(&aclDestroyTensorList),DestroyList>::value,"actual SDK list destroy changed");
static_assert(std::is_same<decltype(&aclCreateIntArray),CreateInts>::value,"actual SDK integer creator changed");
static_assert(std::is_same<decltype(&aclDestroyIntArray),DestroyInts>::value,"actual SDK integer destroy changed");
struct Api {
    CreateList create_list{};DestroyList destroy_list{};CreateInts create_ints{};DestroyInts destroy_ints{};
    decltype(&aclGetDataType) dtype{};decltype(&aclGetFormat) format{};decltype(&aclGetViewOffset) offset{};
    decltype(&aclGetViewShape) shape{};decltype(&aclGetViewStrides) strides{};
    decltype(&aclGetStorageShape) storage{};decltype(&aclGetRawTensorAddr) address{};
    bool ready{};
} api;
std::once_flag api_once;
std::mutex mutex;
State state;
Config config;
Scope scope;
std::vector<Event> events;
std::vector<uint8_t> payload;
struct Registration {uint64_t id{},kind{};};
std::map<uint64_t,Registration> live;
std::atomic<uint64_t> exceptions{0};
uint64_t next_scope{},started{};
thread_local uint64_t call_depth{};
uint64_t tid(){return static_cast<uint64_t>(syscall(SYS_gettid));}
uint64_t now(){return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
    std::chrono::steady_clock::now().time_since_epoch()).count());}
void error(uint64_t code){if(!state.first_error)state.first_error=code;}
void poisoned() noexcept {++exceptions;try{std::lock_guard<std::mutex> lock(mutex);error(Exception);}catch(...) {}}
template<class T>T lookup(void* handle,const char* name){return reinterpret_cast<T>(dlsym(handle,name));}
void resolve() {
    // DT_NEEDED maps this SDK library. Never load a new CANN runtime here.
    void* nn=dlopen("libnnopbase.so",RTLD_NOW|RTLD_NOLOAD);
    if(!nn)return;
    api.create_list=lookup<CreateList>(nn,"aclCreateTensorList");
    api.destroy_list=lookup<DestroyList>(nn,"aclDestroyTensorList");
    api.create_ints=lookup<CreateInts>(nn,"aclCreateIntArray");
    api.destroy_ints=lookup<DestroyInts>(nn,"aclDestroyIntArray");
    api.dtype=lookup<decltype(api.dtype)>(nn,"aclGetDataType");
    api.format=lookup<decltype(api.format)>(nn,"aclGetFormat");
    api.offset=lookup<decltype(api.offset)>(nn,"aclGetViewOffset");
    api.shape=lookup<decltype(api.shape)>(nn,"aclGetViewShape");
    api.strides=lookup<decltype(api.strides)>(nn,"aclGetViewStrides");
    api.storage=lookup<decltype(api.storage)>(nn,"aclGetStorageShape");
    api.address=lookup<decltype(api.address)>(nn,"aclGetRawTensorAddr");
    api.ready=true;
    for(void* fn:{reinterpret_cast<void*>(api.create_list),reinterpret_cast<void*>(api.destroy_list),
        reinterpret_cast<void*>(api.create_ints),reinterpret_cast<void*>(api.destroy_ints),
        reinterpret_cast<void*>(api.dtype),reinterpret_cast<void*>(api.format),reinterpret_cast<void*>(api.offset),
        reinterpret_cast<void*>(api.shape),reinterpret_cast<void*>(api.strides),reinterpret_cast<void*>(api.storage),
        reinterpret_cast<void*>(api.address)}) api.ready=f39identity::matches(fn,qct_profile::nn) && api.ready;
    void* original=dlopen(qct_profile::custom_path,RTLD_NOW|RTLD_NOLOAD);
    api.ready=original && f39identity::matches(original?dlsym(original,"aclnnMoeInitRoutingCustom"):nullptr,
                                             qct_profile::custom_real) && api.ready;
    if(original)dlclose(original);
    dlclose(nn);
}
bool ready() noexcept {
    try{std::call_once(api_once,resolve);if(!api.ready){std::lock_guard<std::mutex> lock(mutex);error(Identity);}
        return api.ready;}catch(...){poisoned();return false;}
}
bool owner(const Config* c) {
    return c && c->version==1 && c->size==sizeof(*c) && state.installed
        && c->owner_hi==config.owner_hi && c->owner_lo==config.owner_lo && c->generation==config.generation
        && c->thread==config.thread && c->thread==tid() && c->max_calls==config.max_calls && c->max_bytes==config.max_bytes;
}
template<class T>bool valid(T* p){return p && p->version==1 && p->size==sizeof(T);}
bool append(Event value) {
    ++state.total_events;
    if(events.size()>=MaxEvents){++state.overflow;error(Budget);return false;}
    value.sequence=state.total_events;value.monotonic_ns=now();events.push_back(value);return true;
}
bool copy_payload(Event& event,const void* data,uint64_t bytes) {
    if(bytes>config.max_bytes-payload.size()){++state.overflow;error(Budget);event.error=Budget;return false;}
    event.payload_offset=payload.size();event.payload_bytes=bytes;
    const auto* begin=static_cast<const uint8_t*>(data);
    if(bytes)payload.insert(payload.end(),begin,begin+bytes);
    return true;
}
void capture(const aclTensor* value,Tensor& out) try {
    dcat_metadata::CaptureScope capture_scope;
    out.descriptor=reinterpret_cast<uint64_t>(value);if(!value)return;
    aclDataType dtype=ACL_DT_UNDEFINED;aclFormat format=ACL_FORMAT_UNDEFINED;void* address=nullptr;
    dcat_metadata::OwnedArray shape,strides,storage;
    out.called_mask|=1;out.statuses[0]=api.dtype(value,&dtype);out.dtype=static_cast<int32_t>(dtype);
    out.called_mask|=2;out.statuses[1]=api.format(value,&format);out.format=static_cast<int32_t>(format);
    out.called_mask|=4;out.statuses[2]=api.offset(value,&out.offset);
    out.called_mask|=8;out.statuses[3]=shape.read(api.shape,value,&out.counts[0]);
    out.called_mask|=16;out.statuses[4]=strides.read(api.strides,value,&out.counts[1]);
    out.called_mask|=32;out.statuses[5]=storage.read(api.storage,value,&out.counts[2]);
    out.called_mask|=64;out.statuses[6]=api.address(value,&address);out.address=reinterpret_cast<uint64_t>(address);
    const int64_t* inputs[]={shape.get(),strides.get(),storage.get()};
    int64_t* outputs[]={out.shape,out.strides,out.storage};
    for(unsigned k=0;k<3;++k) {
        if(!inputs[k] && out.counts[k])out.null_mask|=uint64_t(1)<<k;
        if(out.counts[k]>MaxRank)out.truncated_mask|=uint64_t(1)<<k;
        if(inputs[k])std::copy(inputs[k],inputs[k]+std::min(out.counts[k],MaxRank),outputs[k]);
    }
    out.complete=out.called_mask==127 && !out.null_mask && !out.truncated_mask
        && std::all_of(std::begin(out.statuses),std::end(out.statuses),[](int32_t x){return x==0;});
} catch(...) {
    // OwnedArray and CaptureScope have already unwound. Preserve the capture
    // failure in the shared counters before the outer ABI guard records it.
    dcat_metadata::capture_exception();
    throw;
}
struct Invocation {
    Event event{};bool observed{};
    Invocation(uint32_t operation,const void* caller,const void* function,uint64_t handle,uint64_t count):observed(false) {
        ++call_depth;
        try {
            std::lock_guard<std::mutex> lock(mutex);
            // After stop, only observe retirement of objects recorded earlier.
            if(!state.installed || (state.stopped && !live.count(handle)))return;
            observed=true;++state.active_calls;++state.total_calls;++state.entries[operation-1];
            event.operation=operation;event.phase=1;event.call=state.total_calls;
            event.pid=static_cast<uint64_t>(getpid());event.thread=tid();event.depth=call_depth;
            event.function=reinterpret_cast<uint64_t>(function);event.caller=reinterpret_cast<uint64_t>(caller);
            event.handle=handle;event.count=count;
            Dl_info info{};
            if(dladdr(function,&info)){event.function_base=reinterpret_cast<uint64_t>(info.dli_fbase);
                event.function_offset=event.function-event.function_base;}
            if(dladdr(caller,&info)){event.caller_base=reinterpret_cast<uint64_t>(info.dli_fbase);
                event.caller_offset=event.caller-event.caller_base;}
            if(f39identity::matches(const_cast<void*>(caller),qct_profile::torch_npu))event.caller_kind=1;
            else if(f39identity::matches(const_cast<void*>(caller),qct_profile::custom_extension))event.caller_kind=2;
            if(scope.token) {
                if(event.thread==scope.thread) {
                    event.scope_token=scope.token;event.forward=scope.forward;event.operator_sequence=scope.operator_sequence;
                    event.boundary_sequence=scope.boundary_sequence;event.stage=scope.stage;
                } else {++state.foreign_scope_thread_calls;event.error=WrongThread;error(WrongThread);}
            } else ++state.outside_scope_calls;
            if(operation==DestroyTensors || operation==DestroyIntegers) {
                auto found=live.find(handle);
                if(found!=live.end() && found->second.kind==uint64_t(operation-1))event.registration=found->second.id;
                else if(handle){event.error=UnknownDestroy;error(UnknownDestroy);}
            }
            if(state.total_calls>config.max_calls){++state.overflow;event.error=Budget;error(Budget);}
        } catch(...) {poisoned();}
    }
    void tensors(const aclTensor*const* values,uint64_t count) noexcept {
        if(!observed)return;
        try {
            std::lock_guard<std::mutex> lock(mutex);
            if(count>MaxMembers || (count && !values)){event.error=Metadata;error(Metadata);return;}
            if(state.total_calls>config.max_calls)return;
            std::vector<Tensor> snapshots(count);
            for(uint64_t i=0;i<count;++i) {
                capture(values[i],snapshots[i]);
                if(!snapshots[i].complete){++state.getter_capture_failures;event.error=Metadata;error(Metadata);}
            }
            copy_payload(event,snapshots.data(),count*sizeof(Tensor));
        } catch(...) {poisoned();event.error=Exception;}
    }
    void integers(const int64_t* values,uint64_t count) noexcept {
        if(!observed)return;
        try{std::lock_guard<std::mutex> lock(mutex);
            if(count>MaxMembers || (count && !values)){event.error=Metadata;error(Metadata);return;}
            if(state.total_calls<=config.max_calls)copy_payload(event,values,count*sizeof(int64_t));
        }catch(...){poisoned();event.error=Exception;}
    }
    void enter() noexcept {
        if(!observed)return;
        try{std::lock_guard<std::mutex> lock(mutex);append(event);}catch(...){poisoned();}
    }
    void returned(uint64_t handle,int32_t status) noexcept {
        if(!observed)return;
        try {
            std::lock_guard<std::mutex> lock(mutex);
            ++state.returns[event.operation-1];++state.original_calls[event.operation-1];
            event.phase=2;event.original_calls=1;event.status=status;
            event.payload_bytes=event.payload_offset=0;
            if(event.operation==CreateTensors || event.operation==CreateIntegers) {
                event.handle=handle;
                if(!handle || status){event.error=OriginalFailed;error(OriginalFailed);}
                else if(live.count(handle)){event.error=DuplicateLive;error(DuplicateLive);}
                else if(live.size()>=config.max_calls){event.error=Budget;++state.overflow;error(Budget);}
                else {event.registration=event.call;live.emplace(handle,Registration{event.registration,event.operation});}
            } else {
                if(status){event.error=OriginalFailed;error(OriginalFailed);}
                else if(event.registration)live.erase(event.handle);
            }
            append(event);
        } catch(...) {poisoned();}
    }
    ~Invocation() noexcept {
        if(observed)try{std::lock_guard<std::mutex> lock(mutex);--state.active_calls;}catch(...){poisoned();}
        --call_depth;
    }
};
}

extern "C" uint32_t dcat_qct_marker_v1(){return 1;}
extern "C" const char* dcat_qct_profile_v1(){return qct_profile::id;}
extern "C" int dcat_qct_sizes_v1(dcat_qct::Sizes* out){if(!valid(out))return 1;*out=Sizes{};return 0;}
extern "C" int dcat_qct_state_v1(dcat_qct::State* out) try {
    if(!valid(out))return 1;
    std::lock_guard<std::mutex> lock(mutex);
    *out=state;out->retained_events=events.size();out->metadata_bytes=payload.size();out->live_registrations=live.size();
    out->scope_token=scope.token;out->abi_exceptions=exceptions.load();out->record_capacity_bytes=events.capacity()*sizeof(Event);
    out->payload_capacity_bytes=payload.capacity();out->registration_count=live.size();out->elapsed_ns=started?now()-started:0;
    out->resolver_ready=api.ready;
    return dcat_metadata::snapshot(&out->arrays);
}catch(...){poisoned();return 1;}
extern "C" int dcat_qct_begin_v1(const dcat_qct::Config* c) try {
    if(!c || c->version!=1 || c->size!=sizeof(*c) || !(c->owner_hi|c->owner_lo) || !c->generation
        || c->thread!=tid() || !c->max_calls || c->max_calls>MaxCalls || !c->max_bytes || c->max_bytes>MaxBytes)return 1;
    if(!ready())return 2;
    std::lock_guard<std::mutex> lock(mutex);
    if(state.installed || state.retired || state.active_calls || !live.empty() || exceptions.load())return 3;
    std::vector<Event> new_events;std::vector<uint8_t> new_payload;
    new_events.reserve(MaxEvents);new_payload.reserve(c->max_bytes);
    events.swap(new_events);payload.swap(new_payload);config=*c;state=State{};scope=Scope{};
    state.installed=1;state.owner_hi=c->owner_hi;state.owner_lo=c->owner_lo;
    state.generation=c->generation;state.owner_thread=c->thread;started=now();return 0;
}catch(...){poisoned();return 4;}
extern "C" int dcat_qct_scope_begin_v1(const dcat_qct::Config* c,const dcat_qct::Scope* s) try {
    std::lock_guard<std::mutex> lock(mutex);
    if(!owner(c) || !s || s->version!=1 || s->size!=sizeof(*s) || state.stopped || scope.token
        || s->thread!=tid() || s->token<=next_scope || !s->forward || !s->operator_sequence
        || !s->boundary_sequence || s->stage<1 || s->stage>4)return 1;
    scope=*s;next_scope=s->token;return 0;
}catch(...){poisoned();return 2;}
extern "C" int dcat_qct_scope_end_v1(const dcat_qct::Config* c,uint64_t token) try {
    std::lock_guard<std::mutex> lock(mutex);
    if(!owner(c) || !token || scope.token!=token || state.active_calls)return 1;
    scope=Scope{};return 0;
}catch(...){poisoned();return 2;}
extern "C" int dcat_qct_stop_v1(const dcat_qct::Config* c) try {
    std::lock_guard<std::mutex> lock(mutex);if(!owner(c) || scope.token || state.active_calls)return 1;
    state.stopped=1;return 0;
}catch(...){poisoned();return 2;}
extern "C" int dcat_qct_event_v1(uint64_t index,dcat_qct::Event* out) try {
    if(!valid(out))return 1;
    std::lock_guard<std::mutex> lock(mutex);
    if(index>=events.size())return 2;
    *out=events[index];return 0;
}catch(...){poisoned();return 3;}
extern "C" int dcat_qct_payload_v1(uint64_t offset,uint64_t bytes,void* out) try {
    std::lock_guard<std::mutex> lock(mutex);
    if((bytes && !out) || offset>payload.size() || bytes>payload.size()-offset)return 1;
    if(bytes)std::memcpy(out,payload.data()+offset,bytes);
    return 0;
}catch(...){poisoned();return 2;}
extern "C" int dcat_qct_release_v1(const dcat_qct::Config* c) try {
    std::lock_guard<std::mutex> lock(mutex);
    if(!owner(c) || !state.stopped || scope.token || state.active_calls || !live.empty())return 1;
    std::vector<Event>().swap(events);std::vector<uint8_t>().swap(payload);
    state.installed=0;state.retired=1;return 0;
}catch(...){poisoned();return 2;}

extern "C" aclTensorList* aclCreateTensorList(const aclTensor*const* values,uint64_t size) {
    if(!ready())return nullptr;
    Invocation call(CreateTensors,__builtin_return_address(0),reinterpret_cast<void*>(api.create_list),0,size);
    call.tensors(values,size);call.enter();
    try{auto* result=api.create_list(values,size);call.returned(reinterpret_cast<uint64_t>(result),0);return result;}
    catch(...){poisoned();call.returned(0,161001);return nullptr;}
}
extern "C" aclnnStatus aclDestroyTensorList(const aclTensorList* value) {
    if(!ready())return 161001;
    Invocation call(DestroyTensors,__builtin_return_address(0),reinterpret_cast<void*>(api.destroy_list),reinterpret_cast<uint64_t>(value),0);
    call.enter();try{const auto status=api.destroy_list(value);call.returned(0,status);return status;}
    catch(...){poisoned();call.returned(0,161001);return 161001;}
}
extern "C" aclIntArray* aclCreateIntArray(const int64_t* values,uint64_t size) {
    if(!ready())return nullptr;
    Invocation call(CreateIntegers,__builtin_return_address(0),reinterpret_cast<void*>(api.create_ints),0,size);
    call.integers(values,size);call.enter();
    try{auto* result=api.create_ints(values,size);call.returned(reinterpret_cast<uint64_t>(result),0);return result;}
    catch(...){poisoned();call.returned(0,161001);return nullptr;}
}
extern "C" aclnnStatus aclDestroyIntArray(const aclIntArray* value) {
    if(!ready())return 161001;
    Invocation call(DestroyIntegers,__builtin_return_address(0),reinterpret_cast<void*>(api.destroy_ints),reinterpret_cast<uint64_t>(value),0);
    call.enter();try{const auto status=api.destroy_ints(value);call.returned(0,status);return status;}
    catch(...){poisoned();call.returned(0,161001);return 161001;}
}
