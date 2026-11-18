// Independent v2 DSO. Never link/preload either legacy v1 writer beside this file.
// All writer/consumer identity comes from the independent trace, before admission.
#include "writer26_group_core.h"
#include "../../fault39/native/metadata_array_owner.h"
#include "../../fault20_fast_validation/source/fault26_readonly_trace_v1/fault26_qmm_underflow_api.h"
#include "../../fault20_fast_validation/source/fault26_readonly_trace_v1/fault26_inventory.h"
#include <cstdlib>
#include <dlfcn.h>
#include <mutex>
#include <sys/syscall.h>
#include <unistd.h>

namespace {
using namespace dcat26_group;
using Event=dcat26_inventory::Event;
using TraceState=dcat26_inventory::State;
using Admit=int (*)(const Call*);
using Complete=int (*)(const Receipt*);
using GetDevice=aclError (*)(int32_t*);
using GetType=aclnnStatus (*)(const aclTensor*,aclDataType*);
using GetFormat=aclnnStatus (*)(const aclTensor*,aclFormat*);
using GetOffset=aclnnStatus (*)(const aclTensor*,int64_t*);
using GetShape=aclnnStatus (*)(const aclTensor*,int64_t**,uint64_t*);
using GetAddress=aclnnStatus (*)(const aclTensor*,void**);
using CreateTensor=aclTensor* (*)(const int64_t*,uint64_t,aclDataType,const int64_t*,int64_t,
    aclFormat,const int64_t*,uint64_t,void*);
using DestroyTensor=aclnnStatus (*)(const aclTensor*);
using Malloc=aclError (*)(void**,size_t,aclrtMemMallocPolicy);
using FreeFn=aclError (*)(void*);
using Memcpy=aclError (*)(void*,size_t,const void*,size_t,aclrtMemcpyKind);
using SyncStream=aclError (*)(aclrtStream);
struct Api {
    GetDevice device{};GetType type{};GetFormat format{};GetOffset offset{};
    GetShape shape{},storage{},strides{};GetAddress address{};CreateTensor create{};DestroyTensor destroy{};
    Malloc allocate{};FreeFn release{};Memcpy copy{};SyncStream sync{};
    int (*trace_state)(TraceState*){};
    int (*trace_read)(uint64_t,Event*){};
    bool ready{};
} api;
struct NativeOps final:Ops {
    bool sync(uint64_t stream) override {return api.sync && api.sync(reinterpret_cast<aclrtStream>(stream))==ACL_SUCCESS;}
    bool allocate(void** out,size_t bytes) override {return api.allocate && api.allocate(out,bytes,ACL_MEM_MALLOC_HUGE_FIRST)==ACL_SUCCESS;}
    bool release(void* pointer) override {return api.release && api.release(pointer)==ACL_SUCCESS;}
    bool copy(void* out,size_t capacity,const void* in,size_t bytes,CopyKind kind) override {
        const aclrtMemcpyKind direction=kind==CopyKind::DeviceToHost?ACL_MEMCPY_DEVICE_TO_HOST:
            kind==CopyKind::HostToDevice?ACL_MEMCPY_HOST_TO_DEVICE:ACL_MEMCPY_DEVICE_TO_DEVICE;
        return api.copy && api.copy(out,capacity,in,bytes,direction)==ACL_SUCCESS;
    }
} operations;
Engine engine(operations);
std::recursive_mutex mutex;
Config owner{};
Admit admit{};Complete complete{};
uint64_t next_token{};
// Visible control counters are independent of the authoritative trace counters.
struct RuntimeState {
    uint32_t version{2},size{sizeof(RuntimeState)};
    State core{};
    uint64_t callback_active{},completion_active{},prepared_fia{},fia_executor{},descriptor{};
    uint64_t callback_calls{},passthrough_calls{},admitted_calls{},completion_calls{},rejected_calls{},foreign_calls{};
    int32_t last_rejection{},reserved{};
} runtime;
Call callback_call{};
Event callback_event{};
bool continued{},accepted{};
uint64_t qmm_stream{};
struct FiaRecord {
    bool active{},gws_done{},execute_allowed{};
    Call call{};
    const aclTensor* original{};
    const aclTensor* effective{};
    aclTensor* owned{};
    aclOpExecutor* executor{};
} fia;

uint64_t tid() {return static_cast<uint64_t>(syscall(SYS_gettid));}
bool one(const char* name) {const char* value=std::getenv(name);return value && std::strcmp(value,"1")==0;}
bool same_owner(const Config* value) {
    return value && value->version==2 && value->size==sizeof(Config) && value->thread==tid()
        && engine.matches_owner(value->owner_hi,value->owner_lo,value->generation,value->thread);
}
int reject(int code) {
    ++runtime.rejected_calls;runtime.last_rejection=code;
    if(engine.state().installed)engine.reject(code);
    return code;
}
bool owned_thread() {
    if(owner.thread==tid())return true;
    ++runtime.foreign_calls;reject(Identity);return false;
}
template<class T> T symbol(void* handle,const char* name) {return reinterpret_cast<T>(dlsym(handle,name));}
bool resolve() {
    // RTLD_NOLOAD forbids this shim from creating a new CANN/device context.
    void* nn=dlopen("libnnopbase.so",RTLD_NOW|RTLD_LOCAL|RTLD_NOLOAD);
    void* acl=dlopen("libascendcl.so",RTLD_NOW|RTLD_LOCAL|RTLD_NOLOAD);
    if(!nn || !acl){if(nn)dlclose(nn);if(acl)dlclose(acl);return false;}
    api.device=symbol<GetDevice>(acl,"aclrtGetDevice");api.type=symbol<GetType>(nn,"aclGetDataType");
    api.format=symbol<GetFormat>(nn,"aclGetFormat");api.offset=symbol<GetOffset>(nn,"aclGetViewOffset");
    api.shape=symbol<GetShape>(nn,"aclGetViewShape");api.storage=symbol<GetShape>(nn,"aclGetStorageShape");
    api.strides=symbol<GetShape>(nn,"aclGetViewStrides");api.address=symbol<GetAddress>(nn,"aclGetRawTensorAddr");
    api.create=symbol<CreateTensor>(nn,"aclCreateTensor");api.destroy=symbol<DestroyTensor>(nn,"aclDestroyTensor");
    api.allocate=symbol<Malloc>(acl,"aclrtMalloc");api.release=symbol<FreeFn>(acl,"aclrtFree");
    api.copy=symbol<Memcpy>(acl,"aclrtMemcpy");api.sync=symbol<SyncStream>(acl,"aclrtSynchronizeStream");
    api.trace_state=symbol<decltype(api.trace_state)>(RTLD_DEFAULT,"dcat26_inventory_state_v1");
    api.trace_read=symbol<decltype(api.trace_read)>(RTLD_DEFAULT,"dcat26_inventory_read_v1");
    api.ready=api.device && api.type && api.format && api.offset && api.shape && api.storage && api.strides
        && api.address && api.create && api.destroy && api.allocate && api.release && api.copy && api.sync
        && api.trace_state && api.trace_read;
    dlclose(nn);dlclose(acl);return api.ready;
}
bool trace_state(TraceState& value) {
    return api.trace_state && api.trace_state(&value)==0 && value.version==1 && value.size==sizeof(TraceState)
        && value.flags==7 && value.capacity==32768 && !value.overflow_events && !value.errors
        && value.stored_events==value.total_events && value.all_runs==value.qmm_runs+value.non_target_runs;
}
bool tail(uint32_t kind,Event& value) {
    TraceState state{};
    return trace_state(state) && state.stored_events && api.trace_read(state.stored_events-1,&value)==0
        && value.version==1 && value.size==sizeof(Event) && value.sequence==state.total_events
        && value.kind==kind && value.thread==tid();
}
bool contiguous(uint32_t dimensions,const int64_t* shape,const int64_t* strides,uint64_t bytes) {
    if(!dimensions || dimensions>4)return false;
    uint64_t elements=1;
    for(uint32_t i=dimensions;i>0;--i) {
        const auto n=shape[i-1];
        if(n<=0 || (n>1 && strides[i-1]!=static_cast<int64_t>(elements))
            || static_cast<uint64_t>(n)>4*1024*1024/elements)return false;
        elements*=static_cast<uint64_t>(n);
    }
    return bytes==elements*2;
}
bool complete_event(const Event& event,uint32_t mechanism) {
    return event.flags==(mechanism==Qmm?7U:3U) && event.device>=0 && !event.status && event.call
        && event.producer && (event.dtype==ACL_FLOAT16 || event.dtype==ACL_BF16) && event.format==ACL_FORMAT_ND
        && contiguous(event.dimensions,event.shape,event.strides,event.output_bytes)
        && contiguous(event.residual_dimensions,event.residual_shape,event.residual_strides,event.residual_bytes)
        && span(event.output,event.output_bytes) && span(event.residual,event.residual_bytes);
}
Call make_call(const Event& event,uint32_t mechanism) {
    Call call{};call.mechanism=mechanism;call.token=++next_token;call.thread=event.thread;call.device=event.device;
    call.dtype=event.dtype;call.format=event.format;call.dimensions=event.dimensions;
    call.independent_call=event.call;call.registration=event.registration;call.cache=event.cache;
    call.producer_sequence=event.producer;call.output=event.output;call.output_bytes=event.output_bytes;
    call.residual=event.residual;call.residual_bytes=event.residual_bytes;
    std::copy(event.shape,event.shape+4,call.shape);return call;
}
bool same_device(int device) {int32_t actual=-1;return api.device && api.device(&actual)==ACL_SUCCESS && actual==device;}
bool descriptor_matches(const aclTensor* tensor,const Call& call) try {
    dcat_metadata::CaptureScope capture_scope;
    if(!tensor || call.dimensions>4 || !same_device(call.device))return false;
    aclDataType type{};aclFormat format{};int64_t offset=-1;void* address=nullptr;
    dcat_metadata::OwnedArray shape,storage,strides;
    uint64_t rank=0,storage_rank=0,stride_rank=0;
    if(api.type(tensor,&type) || api.format(tensor,&format) || api.offset(tensor,&offset)
        || api.address(tensor,&address) || shape.read(api.shape,tensor,&rank)
        || storage.read(api.storage,tensor,&storage_rank) || strides.read(api.strides,tensor,&stride_rank))return false;
    return rank==call.dimensions && storage_rank==rank && stride_rank==rank && shape && storage && strides
        && !offset && type==call.dtype && format==call.format && reinterpret_cast<uint64_t>(address)==call.output
        && std::equal(shape.get(),shape.get()+rank,call.shape) && std::equal(storage.get(),storage.get()+rank,call.shape)
        && contiguous(static_cast<uint32_t>(rank),shape,strides,call.output_bytes);
} catch (...) {
    dcat_metadata::capture_exception();
    return false;
}
bool destroy_descriptor() {
    if(!fia.owned)return true;
    if(!api.destroy || api.destroy(fia.owned)!=ACL_SUCCESS){reject(Free);return false;}
    fia.owned=nullptr;engine.descriptor(false);return true;
}
bool dispatch(const Event& event,uint32_t mechanism) {
    if(!admit || !owned_thread())return false;
    const auto state=engine.state();
    const auto pending=engine.receipt(engine.token());
    const bool awaiting_consumer=state.pending && !state.error && !state.executing
        && pending.writer_complete && !pending.consumer_complete;
    // A completed writer can precede ordinary, unselected model calls before
    // its actual consumer. Observe those calls before deciding to pass them.
    // Engine::prepare still rejects a second selected lease while one is held.
    if(runtime.callback_active || runtime.completion_active || (state.pending && !awaiting_consumer)) {
        reject(Busy);return false;
    }
    callback_call=make_call(event,mechanism);callback_event=event;continued=accepted=false;
    runtime.callback_active=1;++runtime.callback_calls;
    const int status=admit(&callback_call);
    runtime.callback_active=0;
    if(status || !continued){reject(Identity);return false;}
    return accepted;
}
void notify_completion() {
    if(!complete || !engine.token())return;
    auto receipt=engine.receipt(engine.token());
    runtime.completion_active=1;++runtime.completion_calls;
    const int status=complete(&receipt);
    runtime.completion_active=0;
    if(status)reject(Consumer);
}
}

extern "C" int dcat26_group_metadata_arrays_v1(DcatMetadataArrayStateV1* output) {
    return dcat_metadata::snapshot(output);
}
extern "C" int dcat26_group_state_v2(RuntimeState* output) {
    std::lock_guard<std::recursive_mutex> lock(mutex);
    if(!output || output->version!=2 || output->size!=sizeof(RuntimeState))return Identity;
    runtime.core=engine.state();runtime.prepared_fia=fia.active;runtime.fia_executor=reinterpret_cast<uint64_t>(fia.executor);
    runtime.descriptor=reinterpret_cast<uint64_t>(fia.owned);*output=runtime;return Ok;
}
extern "C" int dcat26_group_install_v2(const Config* config,Admit callback,Complete finished) {
    std::lock_guard<std::recursive_mutex> lock(mutex);
    if(!config || config->version!=2 || config->size!=sizeof(Config) || config->thread!=tid() || !callback || !finished
        || admit || complete || fia.active || runtime.callback_active || runtime.completion_active
        || !one("VLLM_SERVER_DEV_MODE") || !one("DCAT26_GROUP_V2_ALLOWED")
        || one("FAULT26_QMM_UNDERFLOW_ENABLED") || one("FAULT26_FIA_WRITER_PROXY_ENABLED")
        || dlsym(RTLD_DEFAULT,"dcat26q_install_v1") || dlsym(RTLD_DEFAULT,"dcat26f_install_v1"))return Identity;
    TraceState trace{};
    if(!resolve() || !trace_state(trace))return Identity;
    const int status=engine.open(*config);if(status)return status;
    owner=*config;admit=callback;complete=finished;runtime=RuntimeState{};return Ok;
}
extern "C" int dcat26_group_pass_v2(const Config* config,const Call* call) {
    std::lock_guard<std::recursive_mutex> lock(mutex);
    if(!same_owner(config) || !call || !runtime.callback_active || continued || !same_call(*call,callback_call))return reject(Identity);
    continued=true;accepted=false;++runtime.passthrough_calls;return Ok;
}
extern "C" int dcat26_group_continue_v2(const Lease* lease) {
    std::lock_guard<std::recursive_mutex> lock(mutex);
    if(!lease || !runtime.callback_active || continued || tid()!=owner.thread
        || !same_call(lease->expected,callback_call))return reject(Identity);
    continued=true;
    if(!complete_event(callback_event,callback_call.mechanism) || !same_device(callback_call.device))return reject(Identity);
    const int status=engine.prepare(callback_call,*lease);
    if(status)return reject(status);
    accepted=true;++runtime.admitted_calls;return Ok;
}
extern "C" int dcat26_group_receipt_v2(const Config* config,uint64_t token,Receipt* output) {
    std::lock_guard<std::recursive_mutex> lock(mutex);
    if(!same_owner(config) || !output || output->version!=2 || output->size!=sizeof(Receipt))return Identity;
    *output=engine.receipt(token);return output->token==token?Ok:Identity;
}
extern "C" int dcat26_group_acknowledge_v2(const Config* config,uint64_t token) {
    std::lock_guard<std::recursive_mutex> lock(mutex);
    if(!same_owner(config) || runtime.callback_active || fia.owned)return Identity;
    const int status=engine.acknowledge(*config,token);
    if(status==Ok && fia.active && fia.call.token==token)fia=FiaRecord{};
    return status;
}
extern "C" int dcat26_group_seal_v2(const Config* config,const Key* keys,uint64_t count) {
    std::lock_guard<std::recursive_mutex> lock(mutex);
    if(!same_owner(config) || !keys || !count || count>512 || runtime.callback_active || runtime.completion_active)return Identity;
    return engine.seal(*config,std::vector<Key>(keys,keys+count));
}
extern "C" int dcat26_group_stop_v2(const Config* config) {
    std::lock_guard<std::recursive_mutex> lock(mutex);
    if(!same_owner(config) || runtime.callback_active || runtime.completion_active)return Identity;
    return engine.stop(*config);
}
extern "C" int dcat26_group_drain_v2(const Config* config) {
    std::lock_guard<std::recursive_mutex> lock(mutex);
    if(!same_owner(config) || runtime.callback_active || runtime.completion_active)return Identity;
    return engine.drain(*config);
}
extern "C" int dcat26_group_restore_v2(const Config* config,uint64_t token) {
    std::lock_guard<std::recursive_mutex> lock(mutex);
    if(!same_owner(config) || runtime.callback_active || runtime.completion_active)return Identity;
    const int status=engine.restore(*config,token);
    if(status)return status;
    return destroy_descriptor()?Ok:Free;
}
extern "C" int dcat26_group_remove_v2(const Config* config,Admit callback,Complete finished) {
    std::lock_guard<std::recursive_mutex> lock(mutex);
    if(!same_owner(config) || callback!=admit || finished!=complete || runtime.callback_active || runtime.completion_active
        || fia.active || fia.owned)return Identity;
    const int status=engine.close(*config);
    if(status==Ok){admit=nullptr;complete=nullptr;callback_call=Call{};callback_event=Event{};}
    return status;
}

extern "C" bool fault26_qmm_underflow_prepare(const Fault26QmmUnderflowContext* context,aclrtStream stream,
    Fault26QmmUnderflowDecision* decision) {
    std::lock_guard<std::recursive_mutex> lock(mutex);
    if(decision)*decision=Fault26QmmUnderflowDecision{};
    if(!admit || engine.mechanism()!=Qmm)return false;
    Event event{};
    if(!context || !decision || context->abi_version!=FAULT26_QMM_UNDERFLOW_ABI_VERSION || !tail(dcat26_inventory::QmmEnter,event)
        || event.device!=context->device || event.cache!=reinterpret_cast<uint64_t>(context->cache)
        || event.producer!=context->producer_sequence || context->consumer_sequence
        || event.output!=reinterpret_cast<uint64_t>(context->original_output) || event.output_bytes!=context->output_bytes
        || event.residual!=reinterpret_cast<uint64_t>(context->residual) || event.residual_bytes!=context->residual_bytes
        || event.dimensions!=2 || event.shape[0]!=context->output_m || event.shape[1]!=context->output_n
        || event.residual_dimensions!=2 || event.residual_shape[0]!=context->residual_m || event.residual_shape[1]!=context->residual_k) {
        reject(Identity);return false;
    }
    if(!dispatch(event,Qmm))return false;
    qmm_stream=reinterpret_cast<uint64_t>(stream);
    if(engine.start(callback_call.token,qmm_stream)!=Ok){reject(StateError);return false;}
    decision->abi_version=FAULT26_QMM_UNDERFLOW_ABI_VERSION;decision->claimed=1;
    decision->token=callback_call.token;decision->replacement_output=reinterpret_cast<void*>(engine.replacement());return true;
}
extern "C" void fault26_qmm_underflow_complete(const Fault26QmmUnderflowContext* context,
    const Fault26QmmUnderflowDecision* decision,aclrtStream stream,aclnnStatus status) {
    std::lock_guard<std::recursive_mutex> lock(mutex);
    if(!admit || engine.mechanism()!=Qmm || !owned_thread())return;
    Event event{};const auto receipt=engine.receipt(engine.token());
    if(!context || !decision || decision->abi_version!=FAULT26_QMM_UNDERFLOW_ABI_VERSION || decision->claimed!=1
        || decision->token!=receipt.token || reinterpret_cast<uint64_t>(decision->replacement_output)!=receipt.replacement
        || reinterpret_cast<uint64_t>(stream)!=qmm_stream) {reject(Identity);return;}
    if(!tail(dcat26_inventory::QmmExit,event)
        || event.call!=receipt.independent_call || event.registration!=receipt.registration || event.cache!=receipt.cache
        || event.output!=receipt.output || event.residual!=receipt.residual || event.status!=status) {
        engine.failed_writer_completion(decision->token,qmm_stream,status,1,Identity);reject(Identity);return;
    }
    engine.writer_done(decision->token,reinterpret_cast<uint64_t>(stream),status,1);
}
extern "C" const aclTensor* fault26_fia_writer_prepare_output(const aclTensor* output) {
    std::lock_guard<std::recursive_mutex> lock(mutex);
    if(!admit || engine.mechanism()!=Fia)return output;
    Event event{};
    if(!tail(dcat26_inventory::FiaBefore,event)){reject(Identity);return output;}
    if(!dispatch(event,Fia))return output;
    fia.active=true;fia.call=callback_call;fia.original=fia.effective=output;
    if(!descriptor_matches(output,callback_call)){reject(Identity);return output;}
    if(engine.replacement()!=callback_call.output) {
        int64_t strides[4]{};int64_t step=1;
        for(uint32_t i=callback_call.dimensions;i>0;--i){strides[i-1]=step;step*=callback_call.shape[i-1];}
        fia.owned=api.create(callback_call.shape,callback_call.dimensions,static_cast<aclDataType>(callback_call.dtype),
            strides,0,ACL_FORMAT_ND,callback_call.shape,callback_call.dimensions,reinterpret_cast<void*>(engine.replacement()));
        if(!fia.owned){reject(Allocation);return output;}
        engine.descriptor(true);fia.effective=fia.owned;
    }
    return fia.effective;
}
extern "C" void fault26_fia_writer_gws_complete(const aclTensor* original,const aclTensor* effective,
    aclOpExecutor* executor,aclnnStatus status) {
    std::lock_guard<std::recursive_mutex> lock(mutex);
    if(!admit || engine.mechanism()!=Fia || !fia.active)return;
    if(!owned_thread() || fia.gws_done || original!=fia.original || effective!=fia.effective || !executor || status) {
        reject(Identity);return;
    }
    fia.executor=executor;fia.gws_done=true;
}
extern "C" bool fault26_fia_writer_before_execute(aclOpExecutor* executor,aclrtStream stream) {
    std::lock_guard<std::recursive_mutex> lock(mutex);
    if(!admit || engine.mechanism()!=Fia || !fia.active)return true;
    // A new overlapping/nonowned executor still executes its original once.
    // It remains in the independent denominator and invalidates this scope.
    if(executor!=fia.executor){reject(Busy);return true;}
    Event event{};
    if(!owned_thread() || !fia.gws_done || executor!=fia.executor || !tail(dcat26_inventory::FiaEnter,event)
        || event.call!=fia.call.independent_call || !event.registration || event.executor!=reinterpret_cast<uint64_t>(executor)
        || event.output!=engine.replacement() || event.output_bytes!=fia.call.output_bytes || event.device!=fia.call.device
        || event.residual!=fia.call.residual || event.residual_bytes!=fia.call.residual_bytes
        || event.producer!=fia.call.producer_sequence || !complete_event(event,Fia) || !same_device(event.device)) {
        reject(Identity);return false;
    }
    fia.execute_allowed=engine.start(fia.call.token,reinterpret_cast<uint64_t>(stream),reinterpret_cast<uint64_t>(executor),event.registration)==Ok;
    return fia.execute_allowed;
}
extern "C" void fault26_fia_writer_after_execute(aclOpExecutor* executor,aclrtStream stream,aclnnStatus status) {
    std::lock_guard<std::recursive_mutex> lock(mutex);
    if(!admit || engine.mechanism()!=Fia || !fia.active)return;
    if(!owned_thread() || executor!=fia.executor){reject(Identity);return;}
    if(fia.execute_allowed)engine.writer_done(fia.call.token,reinterpret_cast<uint64_t>(stream),status,1);
    else reject(Writer); // Original was skipped by the reviewed proxy; never claim an execution or hit.
    // The descriptor is retained on any failure until stop/drain/restore retries.
    if(!engine.state().error)destroy_descriptor();
}
extern "C" void fault26_qmm_underflow_consumer_complete(int32_t device,uint64_t sequence,void* residual,
    uint64_t residual_bytes,void* output,uint64_t output_bytes,aclrtStream stream,aclnnStatus status) {
    std::lock_guard<std::recursive_mutex> lock(mutex);
    // This existing trace callback observes actual AddRms consumers for BOTH writers.
    if(!admit || !engine.state().pending || !owned_thread())return;
    const auto receipt=engine.receipt(engine.token());
    if(!receipt.writer_complete)return;
    Event event{};
    if(!tail(dcat26_inventory::AddRmsExit,event) || event.flags!=3 || event.device!=device || event.status!=status
        || event.call!=sequence || event.producer!=sequence || !event.executor
        || event.residual!=reinterpret_cast<uint64_t>(residual) || event.residual_bytes!=residual_bytes
        || event.output!=reinterpret_cast<uint64_t>(output) || event.output_bytes!=output_bytes) {reject(Consumer);return;}
    engine.consumer_done(sequence,reinterpret_cast<uint64_t>(residual),residual_bytes,reinterpret_cast<uint64_t>(output),
        output_bytes,device,reinterpret_cast<uint64_t>(stream),status,1);
    notify_completion();
}
