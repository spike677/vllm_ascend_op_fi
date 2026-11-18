// Included only by the separately fingerprinted DCAT_QCT_GMM2_EXTENSION build.
// The four-container journal, scope, mutex and copied payload remain its owner.
#include "quant_gmm2_trace.h"
#include <aclnnop/aclnn_grouped_matmul_weight_nz.h>
#include <acl/acl_rt.h>
#include <array>

namespace {
using GPrepare=decltype(&aclnnGroupedMatmulWeightNzGetWorkspaceSize);
using GExecute=decltype(&aclnnGroupedMatmulWeightNz);
bool reviewed_gmm_owner(const void* p){return f39identity::matches(const_cast<void*>(p),qct_profile::opapi);}
bool reviewed_gmm_caller(const void* p,uint32_t operation) {
    Dl_info info{};
    return f39identity::matches(const_cast<void*>(p),qct_profile::torch_npu) && dladdr(p,&info)
        && reinterpret_cast<uint64_t>(p)-reinterpret_cast<uint64_t>(info.dli_fbase)
            == (operation==dcat_qg2::Prepare?0x175bc10ULL:0x1747cb4ULL);
}
struct GApi {
    GPrepare prepare{};GExecute execute{};
    decltype(&aclGetTensorListSize) list_size{};decltype(&aclGetIntArraySize) int_size{};
    decltype(&aclrtGetDevice) device{};
    bool(*owner_matches)(const void*){reviewed_gmm_owner};
    bool(*caller_matches)(const void*,uint32_t){reviewed_gmm_caller};
    bool ready{};
} gapi;
std::once_flag gapi_once;
std::atomic<bool> gresolved{false};
dcat_qg2::State gstate;
dcat_qg2::Limits glimits;
std::vector<dcat_qg2::Event> gevents;
struct GPending {
    uint64_t executor{},registration{},scope_token{},thread{},workspace_bytes{};
    int32_t device{-1};bool failed_prepare{},executing{};
};
std::array<GPending,dcat_qg2::MaxPairs> gpending{};
std::atomic<uint64_t> gexceptions{0};
uint64_t gstarted{};
bool gmm2_active(){return gstate.active_calls!=0;}
bool gmm2_owns_journal(){return gstate.installed || gstate.pending_executors || gstate.untracked_executors;}
void gerror(uint64_t e){if(!gstate.first_error)gstate.first_error=e;}
void gpoisoned() noexcept {
    ++gexceptions;
    try{std::lock_guard<std::mutex> lock(mutex);gerror(dcat_qg2::Exception);}catch(...){}
}
void gresolve() {
    // These libraries must be startup dependencies; NOLOAD cannot repair a
    // missing preload. The canonical files and executable segments are pinned.
    void* op=dlopen(qct_profile::opapi_path,RTLD_NOW|RTLD_NOLOAD);
    void* nn=dlopen(qct_profile::nn_path,RTLD_NOW|RTLD_NOLOAD);
    void* device=dlopen(qct_profile::device_path,RTLD_NOW|RTLD_NOLOAD);
    if(op){gapi.prepare=lookup<GPrepare>(op,"aclnnGroupedMatmulWeightNzGetWorkspaceSize");
        gapi.execute=lookup<GExecute>(op,"aclnnGroupedMatmulWeightNz");}
    if(nn){gapi.list_size=lookup<decltype(gapi.list_size)>(nn,"aclGetTensorListSize");
        gapi.int_size=lookup<decltype(gapi.int_size)>(nn,"aclGetIntArraySize");}
    if(device)gapi.device=lookup<decltype(gapi.device)>(device,"aclrtGetDevice");
    gapi.ready=ready() && op && nn && device
        && gapi.owner_matches(reinterpret_cast<void*>(gapi.prepare))
        && gapi.owner_matches(reinterpret_cast<void*>(gapi.execute))
        && f39identity::matches(reinterpret_cast<void*>(gapi.list_size),qct_profile::nn)
        && f39identity::matches(reinterpret_cast<void*>(gapi.int_size),qct_profile::nn)
        && f39identity::matches(reinterpret_cast<void*>(gapi.device),qct_profile::device_runtime);
    if(op)dlclose(op);
    if(nn)dlclose(nn);
    if(device)dlclose(device);
}
bool gready() noexcept {
    try{std::call_once(gapi_once,gresolve);gresolved.store(true,std::memory_order_release);
        if(!gapi.ready){std::lock_guard<std::mutex> lock(mutex);gerror(dcat_qg2::Owner);}
        return gapi.ready;
    }catch(...){gpoisoned();return false;}
}
bool gowner(const Config* c){return owner(c) && gstate.installed && c->owner_hi==gstate.owner_hi
    && c->owner_lo==gstate.owner_lo && c->generation==gstate.generation && c->thread==gstate.thread;}
void gappend(dcat_qg2::Event& e) {
    e.sequence=++gstate.total_events;e.monotonic_ns=now();e.container_events=state.total_events;
    if(gevents.size()>=glimits.max_calls*2){++gstate.overflow;gerror(dcat_qg2::Budget);return;}
    gevents.push_back(e);
}
GPending* pending(uint64_t executor) {
    if(!executor)return nullptr;
    for(auto& p:gpending)if(p.executor==executor)return &p;
    return nullptr;
}
struct GInvocation {
    dcat_qg2::Event e{};bool observed{},function_owned{},joined{};
    GInvocation(uint32_t operation,const void* caller,const void* function,uint64_t executor=0) noexcept {
        try {
            function_owned=gapi.owner_matches(function);
            std::lock_guard<std::mutex> lock(mutex);
            if(!gstate.installed || (gstate.stopped && !pending(executor)))return;
            observed=true;e.operation=operation;e.phase=1;e.call=++gstate.total_calls;
            ++gstate.active_calls;
            if(operation==dcat_qg2::Prepare)++gstate.prepares;else ++gstate.executes;
            e.owner_hi=gstate.owner_hi;e.owner_lo=gstate.owner_lo;e.generation=gstate.generation;
            e.pid=getpid();e.thread=tid();e.function=reinterpret_cast<uint64_t>(function);
            e.caller=reinterpret_cast<uint64_t>(caller);
            Dl_info info{};
            if(dladdr(function,&info)){e.function_base=reinterpret_cast<uint64_t>(info.dli_fbase);
                e.function_offset=e.function-e.function_base;}
            if(dladdr(caller,&info)){e.caller_base=reinterpret_cast<uint64_t>(info.dli_fbase);
                e.caller_offset=e.caller-e.caller_base;}
            if(!function_owned || !gapi.caller_matches(caller,operation))fail(dcat_qg2::Owner);
            if(scope.token){e.scope_token=scope.token;e.forward=scope.forward;
                e.operator_sequence=scope.operator_sequence;e.boundary_sequence=scope.boundary_sequence;e.stage=scope.stage;}
            if(!scope.token || scope.stage!=3){++gstate.outside_scope_calls;fail(dcat_qg2::Scope);}
            if(e.thread!=gstate.thread || (scope.token && e.thread!=scope.thread)){
                ++gstate.foreign_thread_calls;fail(dcat_qg2::Thread);}
            e.device_status=gapi.device(&e.device);
            if(e.device_status || e.device<0)fail(dcat_qg2::Device);
            if(gstate.active_calls!=1)fail(dcat_qg2::Overlap);
            if(gstate.total_calls>glimits.max_calls){++gstate.overflow;fail(dcat_qg2::Budget);}
        }catch(...){gpoisoned();e.error=dcat_qg2::Exception;}
    }
    void fail(uint64_t code){if(!e.error)e.error=code;gerror(code);}
    void slots() {
        static constexpr uint32_t indices[]={0,1,2,3,4,5,6,7,9,10,11,16,18,19,20};
        for(size_t n=0;n<15;++n) {
            auto& slot=e.containers[n];slot.formal_index=indices[n];slot.handle=e.formals[indices[n]];
            slot.kind=indices[n]==16?CreateIntegers:CreateTensors;
            if(!slot.handle){slot.joined=1;slot.live_at_return=1;continue;}
            auto it=live.find(slot.handle);
            if(it==live.end() || it->second.kind!=slot.kind){fail(dcat_qg2::Container);continue;}
            slot.registration=it->second.id;
            const Event* entry=nullptr;const Event* returned=nullptr;
            for(const auto& raw:events)if(raw.call==slot.registration) {
                if(raw.phase==1)entry=&raw;else if(raw.phase==2)returned=&raw;
            }
            if(!entry || !returned || returned->handle!=slot.handle || returned->status || returned->error
                || entry->error || entry->operation!=slot.kind || entry->scope_token!=e.scope_token
                || entry->forward!=e.forward || entry->thread!=e.thread || entry->stage!=3
                || entry->operator_sequence!=e.operator_sequence || entry->boundary_sequence!=e.boundary_sequence
                || returned->original_calls!=1 || returned->sequence<=entry->sequence
                || returned->sequence>state.total_events){fail(dcat_qg2::Container);continue;}
            slot.creator_entry=entry->sequence;slot.creator_return=returned->sequence;
            slot.count=entry->count;slot.payload_offset=entry->payload_offset;slot.payload_bytes=entry->payload_bytes;
            const uint64_t stride=slot.kind==CreateTensors?sizeof(Tensor):sizeof(int64_t);
            if(slot.count>MaxMembers || slot.payload_bytes!=slot.count*stride
                || slot.payload_offset>payload.size() || slot.payload_bytes>payload.size()-slot.payload_offset){
                fail(dcat_qg2::Container);continue;}
            slot.size_status=slot.kind==CreateTensors
                ?gapi.list_size(reinterpret_cast<const aclTensorList*>(slot.handle),&slot.actual_count)
                :gapi.int_size(reinterpret_cast<const aclIntArray*>(slot.handle),&slot.actual_count);
            if(slot.size_status || slot.actual_count!=slot.count){++gstate.getter_failures;fail(dcat_qg2::Getter);continue;}
            slot.joined=1;
        }
    }
    void prepare(const uint64_t* formals,const int64_t* scalars,const aclTensor* group) noexcept {
        if(!observed)return;
        try{std::lock_guard<std::mutex> lock(mutex);
            std::copy(formals,formals+23,e.formals);std::copy(scalars,scalars+5,e.scalars);
            e.prepare_registration=e.call;
            slots();
            if(group){capture(group,e.group_list);if(!e.group_list.complete){++gstate.getter_failures;fail(dcat_qg2::Getter);}}
            if(!formals[21] || !formals[22])fail(dcat_qg2::Workspace);
            gappend(e);
        }catch(...){gpoisoned();e.error=dcat_qg2::Exception;
            try{std::lock_guard<std::mutex> lock(mutex);gappend(e);}catch(...){gpoisoned();}}
    }
    void execute(void* workspace,uint64_t bytes,aclOpExecutor* executor,aclrtStream stream) noexcept {
        if(!observed)return;
        try{std::lock_guard<std::mutex> lock(mutex);
            e.workspace=reinterpret_cast<uint64_t>(workspace);e.workspace_bytes=bytes;
            e.executor=reinterpret_cast<uint64_t>(executor);e.stream=reinterpret_cast<uint64_t>(stream);
            auto* p=pending(e.executor);
            if(!p)fail(dcat_qg2::Executor);
            else {
                e.prepare_registration=p->registration;
                if(e.error || p->failed_prepare || p->executing || p->scope_token!=e.scope_token || p->thread!=e.thread
                    || p->device!=e.device || e.device_status)fail(dcat_qg2::Executor);
                else if(bytes<p->workspace_bytes || (p->workspace_bytes && !workspace))fail(dcat_qg2::Workspace);
                else {joined=true;p->executing=true;}
            }
            gappend(e);
        }catch(...){gpoisoned();e.error=dcat_qg2::Exception;}
    }
    void prepared(int status,uint64_t* bytes,aclOpExecutor** executor,bool original_called) noexcept {
        if(!observed)return;
        try{std::lock_guard<std::mutex> lock(mutex);
            ++gstate.prepare_returns;gstate.original_prepares+=original_called;
            e.phase=2;e.status=status;e.original_calls=original_called;
            // Output pointers are read only within this actual GWS stack.
            e.workspace_bytes=original_called && bytes?*bytes:0;
            e.executor=original_called && executor?reinterpret_cast<uint64_t>(*executor):0;
            if(status || !original_called)fail(dcat_qg2::Original);
            if(e.executor) {
                if(pending(e.executor)){++gstate.untracked_executors;fail(dcat_qg2::Executor);}
                else {
                    GPending* free=nullptr;
                    for(uint64_t i=0;i<glimits.max_pairs;++i)if(!gpending[i].executor){free=&gpending[i];break;}
                    if(!free){++gstate.untracked_executors;++gstate.overflow;fail(dcat_qg2::Budget);}
                    else {*free=GPending{e.executor,e.call,e.scope_token,e.thread,e.workspace_bytes,e.device,
                                        status!=0 || !original_called,false};++gstate.pending_executors;}
                }
                if(status)++gstate.failed_with_executor;
            }else if(!status)fail(dcat_qg2::Executor);
            for(auto& slot:e.containers)if(slot.handle && slot.joined) {
                const auto it=live.find(slot.handle);
                slot.live_at_return=it!=live.end() && it->second.id==slot.registration && it->second.kind==slot.kind;
                if(!slot.live_at_return)fail(dcat_qg2::Container);
            }
            gappend(e);
        }catch(...){gpoisoned();}
    }
    void executed(int status,bool original_called) noexcept {
        if(!observed)return;
        try{std::lock_guard<std::mutex> lock(mutex);
            ++gstate.execute_returns;gstate.original_executes+=original_called;
            e.phase=2;e.status=status;e.original_calls=original_called;
            auto* p=pending(e.executor);
            if(p && joined && p->registration==e.prepare_registration) {
                p->executing=false;
                if(!status && original_called){*p=GPending{};--gstate.pending_executors;++gstate.paired_execute_returns;}
            }
            if(status || !original_called)fail(dcat_qg2::Original);
            gappend(e);
        }catch(...){gpoisoned();}
    }
    ~GInvocation() noexcept {
        if(observed)try{std::lock_guard<std::mutex> lock(mutex);--gstate.active_calls;}catch(...){gpoisoned();}
    }
};
}

extern "C" uint32_t dcat_qct_gmm2_marker_v1(){return 1;}
extern "C" int dcat_qct_gmm2_resolve_v1(){return gready()?0:1;}
extern "C" int dcat_qct_gmm2_sizes_v1(dcat_qg2::Sizes* out){if(!valid(out))return 1;*out=dcat_qg2::Sizes{};return 0;}
extern "C" int dcat_qct_gmm2_begin_v1(const Config* c,const dcat_qg2::Limits* limits) try {
    if(!limits || limits->version!=1 || limits->size!=sizeof(*limits) || !limits->max_calls
        || limits->max_calls>dcat_qg2::MaxCalls || !limits->max_pairs || limits->max_pairs>dcat_qg2::MaxPairs
        || limits->max_bytes>dcat_qg2::MaxBytes || limits->max_calls*2*sizeof(dcat_qg2::Event)>limits->max_bytes)return 1;
    if(!gready())return 2;
    std::lock_guard<std::mutex> lock(mutex);
    if(!owner(c) || scope.token || state.stopped || state.active_calls || gstate.installed || gstate.retired
        || gstate.active_calls || gstate.pending_executors || gstate.untracked_executors || gstate.first_error || gexceptions.load())return 3;
    std::vector<dcat_qg2::Event> allocated;allocated.reserve(limits->max_calls*2);gevents.swap(allocated);
    glimits=*limits;gstate=dcat_qg2::State{};gstate.installed=1;gstate.owner_hi=c->owner_hi;
    gstate.owner_lo=c->owner_lo;gstate.generation=c->generation;gstate.thread=c->thread;gstarted=now();return 0;
}catch(...){gpoisoned();return 4;}
extern "C" int dcat_qct_gmm2_state_v1(dcat_qg2::State* out) try {
    if(!valid(out))return 1;
    std::lock_guard<std::mutex> lock(mutex);*out=gstate;out->retained_events=gevents.size();
    out->record_capacity_bytes=gevents.capacity()*sizeof(dcat_qg2::Event);
    out->elapsed_ns=gstarted?now()-gstarted:0;out->abi_exceptions=gexceptions.load();
    if(gresolved.load(std::memory_order_acquire)) {
        out->resolver_ready=gapi.ready;
        out->prepare_function=reinterpret_cast<uint64_t>(gapi.prepare);out->execute_function=reinterpret_cast<uint64_t>(gapi.execute);
        out->device_function=reinterpret_cast<uint64_t>(gapi.device);out->list_size_function=reinterpret_cast<uint64_t>(gapi.list_size);
        out->int_size_function=reinterpret_cast<uint64_t>(gapi.int_size);
    }
    return 0;
}catch(...){gpoisoned();return 2;}
extern "C" int dcat_qct_gmm2_event_v1(uint64_t index,dcat_qg2::Event* out) try {
    if(!valid(out))return 1;
    std::lock_guard<std::mutex> lock(mutex);if(index>=gevents.size())return 2;*out=gevents[index];return 0;
}catch(...){gpoisoned();return 3;}
extern "C" int dcat_qct_gmm2_stop_v1(const Config* c) try {
    std::lock_guard<std::mutex> lock(mutex);if(!gowner(c) || scope.token || gstate.active_calls)return 1;
    gstate.stopped=1;return 0;
}catch(...){gpoisoned();return 2;}
extern "C" int dcat_qct_gmm2_release_v1(const Config* c) try {
    std::lock_guard<std::mutex> lock(mutex);
    if(!gowner(c) || !gstate.stopped || scope.token || gstate.active_calls || gstate.pending_executors
        || gstate.untracked_executors || state.active_calls || !live.empty())return 1;
    std::vector<dcat_qg2::Event>().swap(gevents);gstate.installed=0;gstate.retired=1;return 0;
}catch(...){gpoisoned();return 2;}

extern "C" aclnnStatus aclnnGroupedMatmulWeightNzGetWorkspaceSize(
    const aclTensorList* x,const aclTensorList* weight,const aclTensorList* bias,
    const aclTensorList* scale,const aclTensorList* offset,const aclTensorList* anti_scale,
    const aclTensorList* anti_offset,const aclTensorList* token_scale,const aclTensor* groups,
    const aclTensorList* activation,const aclTensorList* activation_scale,const aclTensorList* activation_offset,
    int64_t split,int64_t group_type,int64_t group_list_type,int64_t act_type,aclIntArray* tuning,int64_t quant_group,
    aclTensorList* out,aclTensorList* feature_out,aclTensorList* scale_out,uint64_t* bytes,aclOpExecutor** executor) {
    if(!gready())return 161001;
    GInvocation call(dcat_qg2::Prepare,__builtin_return_address(0),reinterpret_cast<void*>(gapi.prepare));
    const uint64_t formals[]={reinterpret_cast<uint64_t>(x),reinterpret_cast<uint64_t>(weight),
        reinterpret_cast<uint64_t>(bias),reinterpret_cast<uint64_t>(scale),reinterpret_cast<uint64_t>(offset),
        reinterpret_cast<uint64_t>(anti_scale),reinterpret_cast<uint64_t>(anti_offset),reinterpret_cast<uint64_t>(token_scale),
        reinterpret_cast<uint64_t>(groups),reinterpret_cast<uint64_t>(activation),reinterpret_cast<uint64_t>(activation_scale),
        reinterpret_cast<uint64_t>(activation_offset),static_cast<uint64_t>(split),static_cast<uint64_t>(group_type),
        static_cast<uint64_t>(group_list_type),static_cast<uint64_t>(act_type),reinterpret_cast<uint64_t>(tuning),
        static_cast<uint64_t>(quant_group),reinterpret_cast<uint64_t>(out),reinterpret_cast<uint64_t>(feature_out),
        reinterpret_cast<uint64_t>(scale_out),reinterpret_cast<uint64_t>(bytes),reinterpret_cast<uint64_t>(executor)};
    const int64_t scalars[]={split,group_type,group_list_type,act_type,quant_group};
    call.prepare(formals,scalars,groups);
    try {const auto status=gapi.prepare(x,weight,bias,scale,offset,anti_scale,anti_offset,token_scale,groups,
        activation,activation_scale,activation_offset,split,group_type,group_list_type,act_type,tuning,quant_group,
        out,feature_out,scale_out,bytes,executor);call.prepared(status,bytes,executor,true);return status;}
    catch(...){gpoisoned();call.prepared(161001,bytes,executor,true);return 161001;}
}
extern "C" aclnnStatus aclnnGroupedMatmulWeightNz(void* workspace,uint64_t bytes,aclOpExecutor* executor,aclrtStream stream) {
    if(!gready())return 161001;
    GInvocation call(dcat_qg2::Execute,__builtin_return_address(0),reinterpret_cast<void*>(gapi.execute),
                     reinterpret_cast<uint64_t>(executor));
    call.execute(workspace,bytes,executor,stream);
    try {const auto status=gapi.execute(workspace,bytes,executor,stream);call.executed(status,true);return status;}
    catch(...){gpoisoned();call.executed(161001,true);return 161001;}
}
