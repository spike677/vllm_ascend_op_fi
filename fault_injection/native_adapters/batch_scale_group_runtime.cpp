// Separate group DSO. This file never links or calls legacy dcat_dq_arm/finish.
#include "batch_scale_group_runtime.h"
#include <acl/acl.h>
#include <aclnnop/aclnn_dynamic_quant_v2.h>
#include <atomic>
#include <cstdlib>
#include <dlfcn.h>
#include <link.h>
#include <mutex>
#include <sys/syscall.h>
#include <unistd.h>

namespace {
using namespace dcat_dq_group;
constexpr aclnnStatus invalid=161001;
using GetDevice=aclError (*)(int32_t*);
using GetType=aclnnStatus (*)(const aclTensor*,aclDataType*);
using GetFormat=aclnnStatus (*)(const aclTensor*,aclFormat*);
using GetOffset=aclnnStatus (*)(const aclTensor*,int64_t*);
using GetShape=aclnnStatus (*)(const aclTensor*,int64_t**,uint64_t*);
using GetAddress=aclnnStatus (*)(const aclTensor*,void**);
using Create=aclTensor* (*)(const int64_t*,uint64_t,aclDataType,const int64_t*,int64_t,aclFormat,const int64_t*,uint64_t,void*);
using Destroy=aclnnStatus (*)(const aclTensor*);
using Allocate=aclError (*)(void**,size_t,aclrtMemMallocPolicy);
using Release=aclError (*)(void*);
using Copy=aclError (*)(void*,size_t,const void*,size_t,aclrtMemcpyKind);
using Sync=aclError (*)(aclrtStream);
struct Api {
    GetDevice device{};GetType type{};GetFormat format{};GetOffset offset{};
    GetShape shape{},strides{},storage{};GetAddress address{};Create create{};Destroy destroy{};
    Allocate allocate{};Release release{};Copy copy{};Sync sync{};
    decltype(&aclnnDynamicQuantV2GetWorkspaceSize) prepare{};
    decltype(&aclnnDynamicQuantV2) execute{};
    bool (*cache)(const char*){};
    void *nn{},*acl{},*op{};bool ready{};
} api;
template<class T>T symbol(void* handle,const char* name){return reinterpret_cast<T>(dlsym(handle,name));}
uint64_t tid(){return static_cast<uint64_t>(syscall(SYS_gettid));}
int device(){int32_t value=-1;return api.device && api.device(&value)==ACL_SUCCESS?value:-1;}
bool enabled(const char* key){const auto* v=std::getenv(key);return v && std::strcmp(v,"1")==0;}
bool resolve() {
    if(api.ready)return true;
    const auto* real=std::getenv("DCAT_REAL_OPAPI");if(!real || real[0]!='/')return false;
    // The NPU runtime must already be loaded by the real public caller. Merely
    // loading/querying this DSO never initializes a device or opens a runtime.
    if(!api.nn)api.nn=dlopen("libnnopbase.so",RTLD_NOW|RTLD_LOCAL|RTLD_NOLOAD);
    if(!api.acl)api.acl=dlopen("libascendcl.so",RTLD_NOW|RTLD_LOCAL|RTLD_NOLOAD);
    if(!api.nn || !api.acl)return false;
    if(!api.op)api.op=dlopen(real,RTLD_NOW|RTLD_LOCAL);
    if(!api.op)return false;
    api.device=symbol<GetDevice>(api.acl,"aclrtGetDevice");api.type=symbol<GetType>(api.nn,"aclGetDataType");
    api.format=symbol<GetFormat>(api.nn,"aclGetFormat");api.offset=symbol<GetOffset>(api.nn,"aclGetViewOffset");
    api.shape=symbol<GetShape>(api.nn,"aclGetViewShape");api.strides=symbol<GetShape>(api.nn,"aclGetViewStrides");
    api.storage=symbol<GetShape>(api.nn,"aclGetStorageShape");api.address=symbol<GetAddress>(api.nn,"aclGetRawTensorAddr");
    api.create=symbol<Create>(api.nn,"aclCreateTensor");api.destroy=symbol<Destroy>(api.nn,"aclDestroyTensor");
    api.allocate=symbol<Allocate>(api.acl,"aclrtMalloc");api.release=symbol<Release>(api.acl,"aclrtFree");
    api.copy=symbol<Copy>(api.acl,"aclrtMemcpy");api.sync=symbol<Sync>(api.acl,"aclrtSynchronizeStream");
    api.prepare=symbol<decltype(api.prepare)>(api.op,"aclnnDynamicQuantV2GetWorkspaceSize");
    api.execute=symbol<decltype(api.execute)>(api.op,"aclnnDynamicQuantV2");api.cache=symbol<decltype(api.cache)>(api.op,"CanUsePTACache");
    api.ready=api.device && api.type && api.format && api.offset && api.shape && api.strides && api.storage && api.address
        && api.create && api.destroy && api.allocate && api.release && api.copy && api.sync && api.prepare && api.execute && api.cache
        && api.prepare!=&aclnnDynamicQuantV2GetWorkspaceSize && api.execute!=&aclnnDynamicQuantV2;
    return api.ready;
}
bool legacy_present() {
    bool present=false;
    dl_iterate_phdr([](dl_phdr_info* row,size_t,void* out)->int {
        if(!row->dlpi_name || !*row->dlpi_name)return 0;
        void* handle=dlopen(row->dlpi_name,RTLD_NOW|RTLD_LOCAL|RTLD_NOLOAD);if(!handle)return 0;
        const bool old=dlsym(handle,"dcat_dq_arm") || dlsym(handle,"dcat_dq_finish");
        dlclose(handle);if(old){*static_cast<bool*>(out)=true;return 1;}return 0;
    },&present);
    return present;
}
Config owner{};
struct NativeOps final:Ops {
    bool on_device()const{return device()==owner.device;}
    bool allocate(void** p,size_t n)override{return on_device() && api.allocate(p,n,ACL_MEM_MALLOC_HUGE_FIRST)==ACL_SUCCESS;}
    bool release(void* p)override{return on_device() && api.release(p)==ACL_SUCCESS;}
    bool copy(void* out,size_t cap,const void* in,size_t n,CopyKind kind)override {
        const auto direction=kind==CopyKind::DeviceToHost?ACL_MEMCPY_DEVICE_TO_HOST:
            kind==CopyKind::HostToDevice?ACL_MEMCPY_HOST_TO_DEVICE:ACL_MEMCPY_DEVICE_TO_DEVICE;
        return on_device() && api.copy(out,cap,in,n,direction)==ACL_SUCCESS;
    }
    bool sync(uint64_t stream)override{return on_device() && stream && api.sync(reinterpret_cast<aclrtStream>(stream))==ACL_SUCCESS;}
    bool destroy_descriptor(uint64_t handle)override{return api.destroy(reinterpret_cast<const aclTensor*>(handle))==ACL_SUCCESS;}
} operations;
Journal journal;Engine engine(operations,journal);
std::recursive_mutex mutex;
std::atomic<uint64_t> entrants{0},overlaps{0};
struct Entry {
    bool overlap{};
    Entry(){overlap=entrants.fetch_add(1)!=0;if(overlap)++overlaps;}
    ~Entry(){--entrants;}
};
Admission admission{};Completed completed{};RuntimeState runtime{};
struct Envelope {
    bool active{},admitted{},returned{};uint64_t token{},gws_calls{};PublicInput input{};Completion completion{};
} envelope;
uint64_t next_lifetime{};Call callback_call{};bool decided{},accepted{};
struct PassRecord {Call call{};uint64_t prepare{},prepared{};};
std::map<uint64_t,PassRecord> passed;
std::set<uint64_t> orphan_descriptors;
int reject(int code){++runtime.rejected_calls;runtime.last_error=code;if(engine.state().pending_parent)engine.reject(code);return code;}
bool same_owner(const Config* c) {
    return c && c->version==1 && c->size==sizeof(Config) && !c->reserved && c->owner_hi==owner.owner_hi
        && c->owner_lo==owner.owner_lo && c->generation==owner.generation && c->thread==owner.thread
        && c->thread==tid() && c->device==owner.device && device()==owner.device;
}
View view(const aclTensor* tensor) {
    View v{};v.descriptor=reinterpret_cast<uint64_t>(tensor);if(!tensor || !api.ready)return v;
    aclDataType type{};aclFormat format{};int64_t *shape=nullptr,*strides=nullptr,*storage=nullptr;uint64_t count=0,scount=0,stored=0;
    void* address=nullptr;
    if(api.type(tensor,&type) || api.format(tensor,&format) || api.offset(tensor,&v.offset)
       || api.shape(tensor,&shape,&count) || api.strides(tensor,&strides,&scount)
       || api.storage(tensor,&storage,&stored) || api.address(tensor,&address))return v;
    v.address=reinterpret_cast<uint64_t>(address);v.dtype=type;v.format=format;v.dimensions=static_cast<uint32_t>(count);
    if(count<1 || count>2 || scount!=count || stored!=count || !shape || !strides || !storage)return v;
    v.rows=shape[0];v.columns=count==2?shape[1]:0;v.stride0=strides[0];v.stride1=count==2?strides[1]:0;
    if(!std::equal(shape,shape+count,storage)){v.reserved=1;return v;}
    const uint64_t element=type==ACL_INT8?1:(type==ACL_FLOAT16 || type==ACL_BF16)?2:type==ACL_FLOAT?4:0;
    if(v.rows<1 || v.rows>1048576 || (count==2 && (v.columns<1 || v.columns>1048576/v.rows)))return v;
    v.bytes=uint64_t(v.rows)*(count==2?uint64_t(v.columns):1)*element;return v;
}
LiveWitness live_witness() {
    LiveWitness w{};
    if(runtime.native_stack_active && envelope.active && envelope.admitted && callback_call.thread==tid()
       && callback_call.lifetime_token==envelope.token && device()==callback_call.device) {
        w.parent_token=callback_call.token;w.lifetime_token=envelope.token;
        w.input=callback_call.input;w.quant=callback_call.quant;w.scale=callback_call.scale;
    }
    return w;
}
int recover_inside_stack() {
    const auto keys=engine.target_keys();if(keys.empty())return Pending;
    for(const auto& key:keys)if(engine.stop(owner,key))return Pending;
    bool drained=true;for(const auto& key:keys)drained=engine.drain(owner,key) && drained;
    if(!drained){runtime.quarantined=1;return Pending;}
    const auto live=live_witness();const int code=engine.recover(owner,live.lifetime_token?&live:nullptr);
    runtime.quarantined=code!=Ok || !orphan_descriptors.empty();return code;
}
bool aliases(uint32_t role) {
    auto plan=engine.plan(role);
    for(unsigned formal=0;formal<3;++formal) {
        const auto& v=formal==0?plan.input:formal==1?plan.quant:plan.scale;if(v.descriptor)continue;
        const int64_t shape[2]{v.rows,v.columns},strides[2]{v.stride0,v.stride1};
        auto* tensor=api.create(shape,v.dimensions,static_cast<aclDataType>(v.dtype),strides,v.offset,
            static_cast<aclFormat>(v.format),shape,v.dimensions,reinterpret_cast<void*>(v.address));
        if(!tensor)return false;
        if(engine.descriptor(role,formal,reinterpret_cast<uint64_t>(tensor))) {
            orphan_descriptors.insert(reinterpret_cast<uint64_t>(tensor));runtime.quarantined=1;return false;
        }
    }
    return true;
}
aclnnStatus real_prepare(const aclTensor* x,const aclTensor* a,const aclTensor* b,int64_t dtype,
    const aclTensor* q,const aclTensor* s,const aclTensor* o,uint64_t* workspace,aclOpExecutor** executor,
    uint32_t role,uint64_t* enter_out,uint64_t* exit_out) {
    const auto enter=journal.prepare(view(x),view(q),view(s),tid(),device(),reinterpret_cast<uint64_t>(a),
        reinterpret_cast<uint64_t>(b),reinterpret_cast<uint64_t>(o),dtype);*enter_out=enter;
    if(!enter || (role && engine.bind_prepare(role,enter))) {
        *exit_out=journal.prepared(enter,0,invalid,0);return invalid;
    }
    *workspace=0;*executor=nullptr;
    const auto status=api.prepare(x,a,b,dtype,q,s,o,workspace,executor);
    *exit_out=journal.prepared(enter,reinterpret_cast<uint64_t>(*executor),status,1,*workspace);
    if(role && engine.prepared(role,*exit_out))return invalid;
    return status;
}
aclnnStatus prepare_child(uint32_t role) {
    if(!aliases(role)){reject(Allocation);return invalid;}
    const auto plan=engine.plan(role);uint64_t workspace=0,enter=0,exit=0;aclOpExecutor* executor=nullptr;
    return real_prepare(reinterpret_cast<const aclTensor*>(plan.input.descriptor),nullptr,nullptr,ACL_INT8,
        reinterpret_cast<const aclTensor*>(plan.quant.descriptor),reinterpret_cast<const aclTensor*>(plan.scale.descriptor),
        nullptr,&workspace,&executor,role,&enter,&exit);
}
aclnnStatus real_execute(void* workspace,uint64_t size,aclOpExecutor* executor,aclrtStream stream,
    uint32_t role,uint64_t* enter_out,uint64_t* exit_out) {
    const auto enter=journal.execute(reinterpret_cast<uint64_t>(executor),tid(),device(),reinterpret_cast<uint64_t>(stream),
        reinterpret_cast<uint64_t>(workspace),size);*enter_out=enter;
    if(!enter || (role && engine.bind_execute(role,enter))) {
        *exit_out=journal.executed(enter,invalid,0);return invalid;
    }
    const auto status=api.execute(workspace,size,executor,stream);*exit_out=journal.executed(enter,status,1);
    if(role && engine.executed(role,*exit_out))return invalid;
    return status;
}
aclnnStatus execute_child(uint32_t role) {
    const auto receipt=engine.receipt();const auto& child=role==CorrectReference?receipt.reference:receipt.effect;
    uint64_t enter=0,exit=0;
    return real_execute(reinterpret_cast<void*>(child.workspace),child.workspace_capacity,
        reinterpret_cast<aclOpExecutor*>(child.executor),reinterpret_cast<aclrtStream>(child.stream?child.stream:runtime.native_stack_active),role,&enter,&exit);
}
int notify(Completion& value) {
    if(!completed)return Ok;
    runtime.completion_active=1;const int code=completed(&value);runtime.completion_active=0;
    return code?reject(Observer):Ok;
}
}

extern "C" int dcat_dqg_abi_v1(AbiSizes* value) {
    if(!value || value->version!=1 || value->size!=sizeof(AbiSizes))return Identity;
    *value=AbiSizes{};value->view=sizeof(View);value->call=sizeof(Call);value->config=sizeof(Config);value->lease=sizeof(Lease);
    value->event=sizeof(NativeEvent);value->binding=sizeof(ChildBinding);value->journal=sizeof(JournalState);value->plan=sizeof(ChildPlan);
    value->child=sizeof(ChildReceipt);value->receipt=sizeof(Receipt);value->state=sizeof(State);value->public_input=sizeof(PublicInput);
    value->completion=sizeof(Completion);value->runtime=sizeof(RuntimeState);value->raw=sizeof(RawRecord);return Ok;
}
extern "C" int dcat_dqg_state_v1(RuntimeState* output) {
    std::lock_guard<std::recursive_mutex> lock(mutex);
    if(!output || output->version!=1 || output->size!=sizeof(RuntimeState))return Identity;
    runtime.core=engine.state();runtime.census=journal.state();runtime.last_sequence=journal.last_sequence();
    runtime.public_active=envelope.active;runtime.public_token=envelope.token;runtime.overlap_entries=overlaps.load();
    runtime.orphan_descriptors=orphan_descriptors.size();
    *output=runtime;return Ok;
}
extern "C" int dcat_dqg_raw_v1(uint64_t sequence,RawRecord* output) {
    std::lock_guard<std::recursive_mutex> lock(mutex);
    if(!output || output->version!=1 || output->size!=sizeof(RawRecord))return Identity;
    *output=RawRecord{};
    if(const auto* parent=journal.parent_record(sequence)){output->kind=1;output->parent=*parent;return Ok;}
    if(const auto* event=journal.event(sequence)) {
        output->kind=2;output->native=*event;if(const auto* binding=journal.binding(event->registration))output->binding=*binding;return Ok;
    }
    return Observer;
}
extern "C" int dcat_dqg_install_v1(const Config* c,Admission observed,Completed finished) {
    std::lock_guard<std::recursive_mutex> lock(mutex);
    if(!c || c->version!=1 || c->size!=sizeof(Config) || c->thread!=tid() || !observed || !finished || admission || completed
       || envelope.active || engine.state().pending_parent || runtime.quarantined || !passed.empty() || entrants.load()
       || !enabled("VLLM_SERVER_DEV_MODE") || !enabled("DCAT_DQ_GROUP_V1_ALLOWED") || enabled("DCAT_DQ_V2_ENABLE") || legacy_present()
       || !resolve() || device()!=c->device)return Identity;
    const auto census=journal.state();if(census.preparing || census.pending || census.executing || census.errors || census.overflow)return Observer;
    const int code=engine.open(*c);if(code)return code;owner=*c;admission=observed;completed=finished;return Ok;
}
extern "C" int dcat_dqg_public_begin_v1(const Config* c,const PublicInput* input,uint64_t* token) {
    std::lock_guard<std::recursive_mutex> lock(mutex);
    if(!same_owner(c) || !input || input->version!=1 || input->size!=sizeof(PublicInput) || input->reserved
       || input->thread!=tid() || input->device!=device() || !token || !admission || envelope.active || runtime.quarantined
       || engine.state().pending_parent || entrants.load() || legacy_present() || !span(input->input.address,input->input.bytes))return Identity;
    const auto census=journal.state();if(census.preparing || census.pending || census.executing || census.errors || census.overflow)return Observer;
    envelope=Envelope{};envelope.active=true;envelope.input=*input;envelope.token=++next_lifetime;*token=envelope.token;return Ok;
}
extern "C" int dcat_dqg_public_end_v1(const Config* c,uint64_t token,const View* q,const View* s,int failed,Completion* output) {
    std::lock_guard<std::recursive_mutex> lock(mutex);
    if(!same_owner(c) || !envelope.active || token!=envelope.token || runtime.native_stack_active || runtime.callback_active
       || !output || output->version!=1 || output->size!=sizeof(Completion))return Identity;
    const bool correct=!failed && envelope.returned && envelope.gws_calls==1 && !envelope.completion.status && q && s
        && same_view(*q,envelope.completion.call.quant,true,false) && same_view(*s,envelope.completion.call.scale,true,false)
        && (!envelope.admitted || (envelope.completion.core.publication_exact && envelope.completion.core.released));
    *output=envelope.completion;envelope.active=false;envelope.token=0;
    if(engine.state().pending_parent || journal.state().pending || journal.state().preparing || journal.state().executing) {
        runtime.quarantined=1;++runtime.unreturned_public_scope;
    }
    return correct?Ok:reject(Liveness);
}
extern "C" int dcat_dqg_pass_v1(const Config* c,const Call* call) {
    std::lock_guard<std::recursive_mutex> lock(mutex);
    if(!same_owner(c) || !call || !runtime.callback_active || decided || !same_call(*call,callback_call))return reject(Identity);
    decided=true;accepted=false;return Ok;
}
extern "C" int dcat_dqg_continue_v1(const Lease* lease) {
    std::lock_guard<std::recursive_mutex> lock(mutex);
    if(!lease || !runtime.callback_active || decided || !envelope.active || callback_call.lifetime_token!=envelope.token
       || !same_call(lease->expected,callback_call) || tid()!=owner.thread || device()!=owner.device || legacy_present())return reject(Identity);
    decided=true;const int code=engine.admit(callback_call.token,*lease);
    if(code)return reject(code);
    accepted=true;envelope.admitted=true;++runtime.admitted_calls;return Ok;
}
extern "C" int dcat_dqg_seal_v1(const Config* c,const Key* keys,uint64_t count,uint64_t unresolved,const Key* scope) {
    std::lock_guard<std::recursive_mutex> lock(mutex);
    if(!same_owner(c) || !keys || !count || count>512 || !scope || envelope.active || entrants.load())return Identity;
    return engine.seal(std::vector<Key>(keys,keys+count),unresolved,*scope);
}
extern "C" int dcat_dqg_stop_v1(const Config* c,const Key* target) {
    std::lock_guard<std::recursive_mutex> lock(mutex);
    if(!same_owner(c) || !target || runtime.callback_active)return Identity;
    return engine.stop(*c,*target);
}
extern "C" int dcat_dqg_drain_v1(const Config* c,const Key* target) {
    std::lock_guard<std::recursive_mutex> lock(mutex);
    if(!same_owner(c) || !target || runtime.callback_active)return Identity;
    return engine.drain(*c,*target)?Ok:Pending;
}
extern "C" int dcat_dqg_recover_v1(const Config* c) {
    std::lock_guard<std::recursive_mutex> lock(mutex);
    if(!same_owner(c) || runtime.callback_active)return Identity;
    // No exported caller-supplied lifetime witness exists. Outside the actual
    // execute stack, public storage cannot be read or restored, even when its
    // previous address/shape still happen to match.
    if(engine.state().pending_parent && (!runtime.native_stack_active || !envelope.active))return Liveness;
    const auto live=live_witness();const int code=engine.recover(*c,live.lifetime_token?&live:nullptr);
    if(!code && orphan_descriptors.empty())runtime.quarantined=0;
    return code;
}
extern "C" int dcat_dqg_remove_v1(const Config* c,Admission observed,Completed finished) {
    std::lock_guard<std::recursive_mutex> lock(mutex);
    const auto state=engine.state();const auto census=journal.state();
    if(!same_owner(c) || observed!=admission || finished!=completed || envelope.active || entrants.load() || state.installed
       || state.pending_parent || state.device_bytes || state.host_bytes || state.owned_descriptors || state.borrowed_workspace_bytes
       || state.quarantined_bytes || runtime.quarantined || !orphan_descriptors.empty() || !passed.empty()
       || census.preparing || census.pending || census.executing)return Pending;
    admission=nullptr;completed=nullptr;return Ok;
}

extern "C" bool CanUsePTACache(const char* name) {
    std::lock_guard<std::recursive_mutex> lock(mutex);
    if(name && std::strcmp(name,"aclnnDynamicQuantV2")==0)return false;
    return resolve() && api.cache(name);
}
extern "C" aclnnStatus aclnnDynamicQuantV2GetWorkspaceSize(const aclTensor* x,const aclTensor* a,const aclTensor* b,int64_t dtype,
    const aclTensor* q,const aclTensor* s,const aclTensor* o,uint64_t* workspace,aclOpExecutor** executor) {
    Entry entry;std::lock_guard<std::recursive_mutex> lock(mutex);++runtime.public_gws_entries;
    const bool ready=resolve();
    Call call{};call.thread=tid();call.device=device();call.input=view(x);call.quant=view(q);call.scale=view(s);
    if(!ready || a || b || o || dtype!=ACL_INT8)call.reserved=1;
    if(envelope.active) {
        ++envelope.gws_calls;
        if(envelope.gws_calls==1 && same_view(call.input,envelope.input.input,true,false))call.lifetime_token=envelope.token;
    }
    call=journal.parent(call); // Independent actual public entry, before admission.
    if(!ready || !workspace || !executor || !call.token || entry.overlap || runtime.quarantined || legacy_present()
       || (admission && (tid()!=owner.thread || device()!=owner.device)) || engine.state().pending_parent) {
        reject(Identity);runtime.quarantined=runtime.quarantined || engine.state().pending_parent!=0;return invalid;
    }
    *workspace=0;*executor=nullptr;callback_call=call;decided=accepted=false;
    if(admission) {
        runtime.callback_active=1;const int code=admission(&callback_call);runtime.callback_active=0;
        if(code || !decided){reject(Observer);if(engine.state().pending_parent)recover_inside_stack();return invalid;}
    }
    if(accepted) {
        auto status=engine.receipt().mode==Fault?prepare_child(CorrectReference):aclnnStatus(0);
        if(!status)status=prepare_child(ModelEffect);
        const auto receipt=engine.receipt();*workspace=receipt.effect.workspace_required;
        *executor=reinterpret_cast<aclOpExecutor*>(receipt.effect.executor);
        if(status){reject(Native);recover_inside_stack();return status;}return 0;
    }
    uint64_t enter=0,exit=0;
    const auto status=real_prepare(x,a,b,dtype,q,s,o,workspace,executor,0,&enter,&exit);
    if(*executor && !passed.emplace(reinterpret_cast<uint64_t>(*executor),PassRecord{call,enter,exit}).second) {
        reject(Observer);runtime.quarantined=1;return invalid;
    }
    if(status && *executor)runtime.quarantined=1;
    return status;
}
extern "C" aclnnStatus aclnnDynamicQuantV2(void* workspace,uint64_t size,aclOpExecutor* executor,aclrtStream stream) {
    Entry entry;std::lock_guard<std::recursive_mutex> lock(mutex);++runtime.public_execute_entries;
    if(!resolve())return invalid;
    const auto before=engine.receipt();const bool selected=engine.state().pending_parent && before.effect.executor==reinterpret_cast<uint64_t>(executor);
    if(entry.overlap || runtime.quarantined || legacy_present() || (admission && (tid()!=owner.thread || device()!=owner.device))
       || (selected && (!envelope.active || before.lifetime_token!=envelope.token || !envelope.admitted))) {
        const auto raw=journal.execute(reinterpret_cast<uint64_t>(executor),tid(),device(),reinterpret_cast<uint64_t>(stream),reinterpret_cast<uint64_t>(workspace),size);
        journal.executed(raw,invalid,0);reject(Liveness);runtime.quarantined=1;return invalid;
    }
    if(!selected) {
        const auto found=passed.find(reinterpret_cast<uint64_t>(executor));
        if(found==passed.end()) {
            const auto raw=journal.execute(reinterpret_cast<uint64_t>(executor),tid(),device(),reinterpret_cast<uint64_t>(stream),reinterpret_cast<uint64_t>(workspace),size);
            journal.executed(raw,invalid,0);reject(Observer);runtime.quarantined=1;return invalid;
        }
        Completion result{};result.call=found->second.call;result.prepare_enter=found->second.prepare;result.prepare_exit=found->second.prepared;
        result.status=real_execute(workspace,size,executor,stream,0,&result.execute_enter,&result.execute_exit);
        passed.erase(found);++runtime.passthrough_calls;++runtime.completed_calls;
        if(notify(result))result.status=invalid;
        if(envelope.active && result.call.lifetime_token==envelope.token){envelope.completion=result;envelope.returned=true;}
        return result.status;
    }
    runtime.native_stack_active=reinterpret_cast<uint64_t>(stream);
    int code=engine.start(reinterpret_cast<uint64_t>(stream),reinterpret_cast<uint64_t>(workspace),size);
    if(!code && before.mode==Fault) {
        code=execute_child(CorrectReference);if(!code)code=engine.readback(CorrectReference);
    }
    if(!code){code=execute_child(ModelEffect);if(!code)code=engine.readback(ModelEffect);}
    if(!code)code=engine.publish();
    Completion result{};result.call=callback_call;result.admitted=1;result.core=engine.receipt();result.status=code;
    if(!code)code=notify(result);
    if(!code)code=engine.acknowledge();
    if(code){reject(code);recover_inside_stack();}
    result.core=engine.receipt();result.status=code?invalid:0;
    envelope.completion=result;envelope.returned=true;++runtime.completed_calls;
    // This is the last point at which the public caller's output allocations
    // remain borrowed under the actual Native execute stack contract.
    runtime.native_stack_active=0;
    return result.status;
}
