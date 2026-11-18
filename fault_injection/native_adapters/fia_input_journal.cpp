// Separate opt-in build. Reuse the byte-pinned four-container journal unchanged.
// No writer26 callback and no GMM2 extension is linked into this translation unit.
#ifdef DCAT_QCT_GMM2_EXTENSION
#error FIA_INPUT_JOURNAL_MUST_NOT_ENABLE_GMM2
#endif
#include "quant_container_trace.cpp"
#include "fia_input_journal.h"
#include "fia_input_journal_profile.h"
#include <aclnnop/aclnn_fused_infer_attention_score_v3.h>
#include <acl/acl_rt.h>
#include <array>
#include <set>

namespace {
using FPrepare=decltype(&aclnnFusedInferAttentionScoreV3GetWorkspaceSize);
using FExecute=decltype(&aclnnFusedInferAttentionScoreV3);
bool fowner_matches(const void* p){return f39identity::matches(const_cast<void*>(p),fij_profile::opapi);}
bool fcaller_matches(const void* p){return f39identity::matches(const_cast<void*>(p),qct_profile::torch_npu);}
struct FApi {
    FPrepare prepare{};FExecute execute{};
    decltype(&aclGetTensorListSize) list_size{};decltype(&aclGetIntArraySize) int_size{};
    decltype(&aclrtGetDevice) device{};int(*scope_matches)(const char*){};
    bool(*owner_matches)(const void*){fowner_matches};bool(*caller_matches)(const void*){fcaller_matches};
    bool ready{};
} fapi;
std::once_flag fonce;
dcat_fij::State fs;
dcat_fij::Limits flimits;
dcat_fij::ScopeIdentity fscope;
std::vector<dcat_fij::Event> fevents;
struct FPending {uint64_t executor{},registration{},token{},thread{},workspace_bytes{};int32_t device{-1};bool failed{},executing{};};
std::array<FPending,dcat_fij::MaxPairs> fpending{};
std::set<std::string> ftokens;
std::atomic<uint64_t> fexceptions{0};
constexpr uint32_t tensor_indices[]={0,3,4,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,24,25,26,40,41};
constexpr uint32_t container_indices[]={1,2,5,6,23};
void ferr(uint64_t e){if(!fs.first_error)fs.first_error=e;}
void fpoison() noexcept {++fexceptions;try{std::lock_guard<std::mutex> lock(mutex);ferr(dcat_fij::Exception);}catch(...) {}}
void fresolve() {
    void* op=dlopen(fij_profile::opapi_path,RTLD_NOW|RTLD_NOLOAD);
    void* nn=dlopen(qct_profile::nn_path,RTLD_NOW|RTLD_NOLOAD);
    void* device=dlopen(fij_profile::device_path,RTLD_NOW|RTLD_NOLOAD);
    void* observer=dlopen(fij_profile::observer_path,RTLD_NOW|RTLD_NOLOAD);
    if(op){fapi.prepare=lookup<FPrepare>(op,"aclnnFusedInferAttentionScoreV3GetWorkspaceSize");
        fapi.execute=lookup<FExecute>(op,"aclnnFusedInferAttentionScoreV3");}
    if(nn){fapi.list_size=lookup<decltype(fapi.list_size)>(nn,"aclGetTensorListSize");
        fapi.int_size=lookup<decltype(fapi.int_size)>(nn,"aclGetIntArraySize");}
    if(device)fapi.device=lookup<decltype(fapi.device)>(device,"aclrtGetDevice");
    if(observer)fapi.scope_matches=lookup<decltype(fapi.scope_matches)>(observer,"dcat_attention_observer_scope_matches_v1");
    fapi.ready=ready() && op && nn && device && observer
        && fapi.owner_matches(reinterpret_cast<void*>(fapi.prepare))
        && fapi.owner_matches(reinterpret_cast<void*>(fapi.execute))
        && f39identity::matches(reinterpret_cast<void*>(fapi.list_size),qct_profile::nn)
        && f39identity::matches(reinterpret_cast<void*>(fapi.int_size),qct_profile::nn)
        && f39identity::matches(reinterpret_cast<void*>(fapi.device),fij_profile::device_runtime)
        && f39identity::matches(reinterpret_cast<void*>(fapi.scope_matches),fij_profile::observer);
    if(op)dlclose(op);if(nn)dlclose(nn);if(device)dlclose(device);if(observer)dlclose(observer);
}
bool fready() noexcept {
    try{std::call_once(fonce,fresolve);return fapi.ready;}catch(...){fpoison();return false;}
}
bool fowner(const Config* c){return owner(c) && fs.installed && fs.pid==uint64_t(getpid())
    && fs.thread==tid() && fs.owner_hi==c->owner_hi && fs.owner_lo==c->owner_lo && fs.generation==c->generation;}
FPending* find_pending(uint64_t executor){for(auto& p:fpending)if(executor && p.executor==executor)return &p;return nullptr;}
void fappend(dcat_fij::Event& e) {
    e.sequence=++fs.total_events;e.monotonic_ns=now();
    if(fevents.size()>=flimits.max_calls*2){++fs.overflow;ferr(dcat_fij::Budget);return;}
    fevents.push_back(e);
}
uint64_t word(double x){uint64_t out;static_assert(sizeof(x)==sizeof(out),"double ABI");std::memcpy(&out,&x,sizeof(out));return out;}
struct FInvocation {
    dcat_fij::Event e{};bool observed{},joined{};
    FInvocation(uint32_t operation,const void* caller,const void* fn) noexcept {
        try{std::lock_guard<std::mutex> lock(mutex);
            if(!fs.installed || fs.stopped)return;
            if(!scope.token){++fs.outside_scope_calls;return;}
            observed=true;++fs.active_calls;e.call=++fs.total_calls;e.operation=operation;e.phase=1;
            e.owner_hi=config.owner_hi;e.owner_lo=config.owner_lo;e.generation=config.generation;
            e.scope_token=scope.token;e.forward=scope.forward;e.operator_sequence=scope.operator_sequence;
            e.boundary_sequence=scope.boundary_sequence;e.pid=getpid();e.thread=tid();
            e.worker=fscope.worker;e.model=fscope.model;e.module=fscope.module;
            std::copy(std::begin(fscope.call_id),std::end(fscope.call_id),e.call_id);
            e.function=reinterpret_cast<uint64_t>(fn);e.caller=reinterpret_cast<uint64_t>(caller);
            Dl_info info{};
            if(dladdr(fn,&info)){e.function_base=reinterpret_cast<uint64_t>(info.dli_fbase);e.function_offset=e.function-e.function_base;}
            if(dladdr(caller,&info)){e.caller_base=reinterpret_cast<uint64_t>(info.dli_fbase);e.caller_offset=e.caller-e.caller_base;}
            if(e.thread!=config.thread){++fs.foreign_thread_calls;fail(dcat_fij::Thread);}
            if(!fapi.owner_matches(fn) || !fapi.caller_matches(caller))fail(dcat_fij::Owner);
            if(fapi.scope_matches(e.call_id)!=1)fail(dcat_fij::Scope);
            e.device_status=fapi.device(&e.device);if(e.device_status || e.device<0)fail(dcat_fij::Device);
            if(fs.total_calls>flimits.max_calls){++fs.overflow;fail(dcat_fij::Budget);}
        }catch(...){fpoison();e.error=dcat_fij::Exception;}
    }
    void fail(uint64_t code){if(!e.error)e.error=code;ferr(code);}
    void tensors() {
        for(size_t n=0;n<dcat_fij::TensorCount;++n){const auto p=e.formals[tensor_indices[n]];
            e.tensors[n]=Tensor{};
            if(p){capture(reinterpret_cast<const aclTensor*>(p),e.tensors[n]);
                if(!e.tensors[n].complete){++fs.getter_failures;fail(dcat_fij::Metadata);}}}
    }
    void containers() {
        for(size_t n=0;n<dcat_fij::ContainerCount;++n){auto& s=e.containers[n];s.formal=container_indices[n];
            s.kind=n<2?CreateTensors:CreateIntegers;s.handle=e.formals[s.formal];
            if(!s.handle){s.joined=s.live_at_return=1;continue;}
            auto it=live.find(s.handle);
            if(it==live.end() || it->second.kind!=s.kind){fail(dcat_fij::Container);continue;}
            s.registration=it->second.id;
            const Event* before=nullptr;const Event* after=nullptr;
            for(const auto& row:events)if(row.call==s.registration){if(row.phase==1)before=&row;else if(row.phase==2)after=&row;}
            if(!before || !after || before->operation!=s.kind || before->error || after->error || after->status
                || after->original_calls!=1 || after->handle!=s.handle || after->sequence<=before->sequence
                || before->scope_token!=e.scope_token || before->forward!=e.forward || before->stage!=3
                || before->operator_sequence!=e.operator_sequence || before->boundary_sequence!=e.boundary_sequence
                || before->pid!=e.pid || before->thread!=e.thread){fail(dcat_fij::Container);continue;}
            s.creator_entry=before->sequence;s.creator_return=after->sequence;s.count=before->count;
            s.payload_offset=before->payload_offset;s.payload_bytes=before->payload_bytes;
            const uint64_t stride=s.kind==CreateTensors?sizeof(Tensor):sizeof(int64_t);
            if(s.count>MaxMembers || s.payload_bytes!=s.count*stride || s.payload_offset>payload.size()
                || s.payload_bytes>payload.size()-s.payload_offset){fail(dcat_fij::Container);continue;}
            s.size_status=n<2?fapi.list_size(reinterpret_cast<const aclTensorList*>(s.handle),&s.actual_count)
                :fapi.int_size(reinterpret_cast<const aclIntArray*>(s.handle),&s.actual_count);
            if(s.size_status || s.actual_count!=s.count){++fs.getter_failures;fail(dcat_fij::Metadata);continue;}
            s.joined=1;
        }
    }
    void prepare(const uint64_t* args,const int64_t* scalars,double scale,bool lse,const char* layout) noexcept {
        if(!observed)return;
        try{std::lock_guard<std::mutex> lock(mutex);std::copy(args,args+dcat_fij::FormalCount,e.formals);
            std::copy(scalars,scalars+10,e.signed_scalars);e.scale=scale;e.softmax_lse=lse;
            if(layout){e.layout_length=strnlen(layout,sizeof(e.layout));
                std::copy(layout,layout+e.layout_length,e.layout);}
            if(!e.layout_length || e.layout_length>=sizeof(e.layout))fail(dcat_fij::Metadata);
            e.prepare_registration=e.call;tensors();containers();
            if(!args[42] || !args[43])fail(dcat_fij::Workspace);
            fappend(e);
        }catch(...){fpoison();e.error=dcat_fij::Exception;try{std::lock_guard<std::mutex> lock(mutex);fappend(e);}catch(...) {}}
    }
    void prepared(uint64_t* bytes,aclOpExecutor** executor,int status,bool exception=false) noexcept {
        if(!observed)return;
        try{std::lock_guard<std::mutex> lock(mutex);++fs.prepare_returns;++fs.original_prepares;
            e.phase=2;e.original_calls=1;e.original_status_valid=!exception;e.original_exception=exception;e.status=status;
            // The SDK does not promise initialized output parameters on failure.
            // Do not read indeterminate outputs or infer that no executor exists.
            // Retain this session until process exit instead of inventing cleanup.
            if(status || exception){fail(dcat_fij::Original);++fs.failed_prepare_outputs_unknown;
                ++fs.untracked_executors;fappend(e);return;}
            e.workspace_bytes=bytes?*bytes:0;e.executor=executor?reinterpret_cast<uint64_t>(*executor):0;
            if(e.executor){if(find_pending(e.executor)){++fs.untracked_executors;fail(dcat_fij::Executor);}
                else {FPending* free=nullptr;for(uint64_t i=0;i<flimits.max_pairs;++i)if(!fpending[i].executor){free=&fpending[i];break;}
                    if(!free){++fs.untracked_executors;++fs.overflow;fail(dcat_fij::Budget);}
                    else {*free=FPending{e.executor,e.call,e.scope_token,e.thread,e.workspace_bytes,e.device,status!=0||exception,false};++fs.pending_executors;}}
            }else if(!status && !exception)fail(dcat_fij::Executor);
            // These are still real arguments of this GWS call, not borrowed
            // descriptors re-read later at execute time.
            tensors();
            for(auto& s:e.containers)if(s.handle && s.joined){auto it=live.find(s.handle);
                s.live_at_return=it!=live.end() && it->second.id==s.registration && it->second.kind==s.kind;
                if(!s.live_at_return)fail(dcat_fij::Container);}
            fappend(e);
        }catch(...){fpoison();}
    }
    void execute(void* workspace,uint64_t bytes,aclOpExecutor* executor,aclrtStream stream) noexcept {
        if(!observed)return;
        try{std::lock_guard<std::mutex> lock(mutex);e.workspace=reinterpret_cast<uint64_t>(workspace);
            e.workspace_bytes=bytes;e.executor=reinterpret_cast<uint64_t>(executor);
            e.formals[0]=e.workspace;e.formals[1]=bytes;e.formals[2]=e.executor;e.formals[3]=reinterpret_cast<uint64_t>(stream);
            auto* p=find_pending(e.executor);
            if(!p)fail(dcat_fij::Executor);
            else {e.prepare_registration=p->registration;
                if(p->failed || p->executing || p->token!=e.scope_token || p->thread!=e.thread || p->device!=e.device || e.device_status)
                    fail(dcat_fij::Executor);
                else if(bytes<p->workspace_bytes || (p->workspace_bytes && !workspace))fail(dcat_fij::Workspace);
                else {joined=true;p->executing=true;}}
            fappend(e);
        }catch(...){fpoison();}
    }
    void executed(int status,bool exception=false) noexcept {
        if(!observed)return;
        try{std::lock_guard<std::mutex> lock(mutex);++fs.execute_returns;++fs.original_executes;
            e.phase=2;e.status=status;e.original_calls=1;e.original_status_valid=!exception;e.original_exception=exception;
            auto* p=find_pending(e.executor);
            if(p && joined && p->registration==e.prepare_registration){p->executing=false;
                if(!status && !exception){*p=FPending{};--fs.pending_executors;++fs.paired_execute_returns;}}
            if(status || exception)fail(dcat_fij::Original);fappend(e);
        }catch(...){fpoison();}
    }
    ~FInvocation() noexcept {if(observed)try{std::lock_guard<std::mutex> lock(mutex);--fs.active_calls;}catch(...) {fpoison();}}
};
}

extern "C" uint32_t dcat_fia_input_journal_marker_v1(){return 1;}
extern "C" int dcat_fia_input_journal_resolve_v1(){return fready()?0:1;}
extern "C" int dcat_fia_input_journal_sizes_v1(dcat_fij::Sizes* p){if(!valid(p))return 1;*p=dcat_fij::Sizes{};
    p->float16_tag=ACL_FLOAT16;p->int8_tag=ACL_INT8;p->nd_tag=ACL_FORMAT_ND;return 0;}
extern "C" int dcat_fia_input_journal_begin_v1(const Config* c,const dcat_fij::Limits* limits) try {
    if(!fready() || !valid(limits) || !limits->max_calls || limits->max_calls>dcat_fij::MaxCalls
        || !limits->max_pairs || limits->max_pairs>dcat_fij::MaxPairs || limits->max_bytes>dcat_fij::MaxBytes
        || !limits->max_windows || limits->max_windows>dcat_fij::MaxCalls
        || limits->max_calls*2*sizeof(dcat_fij::Event)>limits->max_bytes)return 1;
    std::lock_guard<std::mutex> lock(mutex);
    if(!owner(c) || fs.installed || fs.retired || scope.token)return 2;
    fevents.reserve(limits->max_calls*2);flimits=*limits;fs.installed=1;
    fs.owner_hi=c->owner_hi;fs.owner_lo=c->owner_lo;fs.generation=c->generation;fs.pid=getpid();fs.thread=tid();return 0;
}catch(...){fpoison();return 3;}
extern "C" int dcat_fia_input_journal_scope_begin_v1(const Config* c,const dcat_fij::ScopeIdentity* s) try {
    {std::lock_guard<std::mutex> lock(mutex);
        if(!fowner(c) || fs.stopped || !valid(s) || scope.token || fs.active_calls || fs.pending_executors
            || fs.untracked_executors || s->pid!=fs.pid || !s->worker || !s->model || !s->module
            || s->scope.thread!=tid() || s->scope.stage!=3 || s->call_id[32]!='\0'
            || std::any_of(s->call_id,s->call_id+32,[](char x){return !((x>='0'&&x<='9')||(x>='a'&&x<='f'));})
            || ftokens.count(s->call_id))return 1;
        if(ftokens.size()>=flimits.max_windows){ferr(dcat_fij::Budget);return 1;}}
    if(dcat_qct_scope_begin_v1(c,&s->scope))return 2;
    std::lock_guard<std::mutex> lock(mutex);fscope=*s;ftokens.insert(s->call_id);++fs.accepted_windows;return 0;
}catch(...){fpoison();return 3;}
extern "C" int dcat_fia_input_journal_scope_end_v1(const Config* c,uint64_t token) try {
    {std::lock_guard<std::mutex> lock(mutex);if(!fowner(c) || fs.active_calls || fs.pending_executors || fs.untracked_executors)return 1;}
    if(dcat_qct_scope_end_v1(c,token))return 2;
    std::lock_guard<std::mutex> lock(mutex);fscope=dcat_fij::ScopeIdentity{};return 0;
}catch(...){fpoison();return 3;}
extern "C" int dcat_fia_input_journal_state_v1(dcat_fij::State* p) try {
    if(!valid(p))return 1;std::lock_guard<std::mutex> lock(mutex);*p=fs;
    p->exceptions=fexceptions.load();p->retained_events=fevents.size();p->record_capacity_bytes=fevents.capacity()*sizeof(dcat_fij::Event);
    p->resolver_ready=fapi.ready;p->prepare_function=reinterpret_cast<uint64_t>(fapi.prepare);p->execute_function=reinterpret_cast<uint64_t>(fapi.execute);
    p->device_function=reinterpret_cast<uint64_t>(fapi.device);p->list_size_function=reinterpret_cast<uint64_t>(fapi.list_size);
    p->int_size_function=reinterpret_cast<uint64_t>(fapi.int_size);p->observer_scope_function=reinterpret_cast<uint64_t>(fapi.scope_matches);
    p->window_limit=flimits.max_windows;p->retained_scope_tokens=ftokens.size();
    p->retained_scope_token_characters=0;for(const auto& token:ftokens)p->retained_scope_token_characters+=token.size();return 0;
}catch(...){fpoison();return 3;}
extern "C" int dcat_fia_input_journal_event_v1(uint64_t i,dcat_fij::Event* p) try {
    if(!valid(p))return 1;std::lock_guard<std::mutex> lock(mutex);if(i>=fevents.size())return 2;*p=fevents[i];return 0;
}catch(...){fpoison();return 3;}
extern "C" int dcat_fia_input_journal_stop_v1(const Config* c) try {
    std::lock_guard<std::mutex> lock(mutex);if(!fowner(c) || scope.token || fs.active_calls || fs.pending_executors || fs.untracked_executors)return 1;
    fs.stopped=1;return 0;
}catch(...){fpoison();return 3;}
extern "C" int dcat_fia_input_journal_release_v1(const Config* c) try {
    std::lock_guard<std::mutex> lock(mutex);if(!fowner(c) || !fs.stopped || scope.token || fs.active_calls || fs.pending_executors
        || fs.untracked_executors || !live.empty())return 1;
    std::vector<dcat_fij::Event>().swap(fevents);std::set<std::string>().swap(ftokens);fs.installed=0;fs.retired=1;return 0;
}catch(...){fpoison();return 3;}

// Signature is declared by the exact CANN 8.5.1 SDK header, SHA
// bf73aea4146a9b12f91eda466927c23c5d43e56f78aa5039bd9644732f521550.
extern "C" aclnnStatus aclnnFusedInferAttentionScoreV3GetWorkspaceSize(
    const aclTensor* q,const aclTensorList* k,const aclTensorList* v,const aclTensor* pse,const aclTensor* mask,
    const aclIntArray* seq,const aclIntArray* seqkv,const aclTensor* dq1,const aclTensor* qs1,const aclTensor* dq2,
    const aclTensor* qs2,const aclTensor* qo2,const aclTensor* as,const aclTensor* ao,const aclTensor* blocks,
    const aclTensor* qpad,const aclTensor* kvpad,const aclTensor* kas,const aclTensor* kao,const aclTensor* vas,
    const aclTensor* vao,const aclTensor* kp,const aclTensor* vp,const aclIntArray* plen,const aclTensor* qr,
    const aclTensor* kr,const aclTensor* kras,int64_t nh,double scale,int64_t pre,int64_t next,char* layout,
    int64_t nkh,int64_t sparse,int64_t precise,int64_t block,int64_t am,bool lse,int64_t kam,int64_t vam,
    const aclTensor* out,const aclTensor* sout,uint64_t* bytes,aclOpExecutor** executor) {
    if(!fready())return 161001; // pre-LLM resolver gate must reject this package.
    FInvocation call(dcat_fij::Prepare,__builtin_return_address(0),reinterpret_cast<void*>(fapi.prepare));
    const uint64_t a[]={reinterpret_cast<uint64_t>(q),reinterpret_cast<uint64_t>(k),reinterpret_cast<uint64_t>(v),
        reinterpret_cast<uint64_t>(pse),reinterpret_cast<uint64_t>(mask),reinterpret_cast<uint64_t>(seq),reinterpret_cast<uint64_t>(seqkv),
        reinterpret_cast<uint64_t>(dq1),reinterpret_cast<uint64_t>(qs1),reinterpret_cast<uint64_t>(dq2),reinterpret_cast<uint64_t>(qs2),
        reinterpret_cast<uint64_t>(qo2),reinterpret_cast<uint64_t>(as),reinterpret_cast<uint64_t>(ao),reinterpret_cast<uint64_t>(blocks),
        reinterpret_cast<uint64_t>(qpad),reinterpret_cast<uint64_t>(kvpad),reinterpret_cast<uint64_t>(kas),reinterpret_cast<uint64_t>(kao),
        reinterpret_cast<uint64_t>(vas),reinterpret_cast<uint64_t>(vao),reinterpret_cast<uint64_t>(kp),reinterpret_cast<uint64_t>(vp),
        reinterpret_cast<uint64_t>(plen),reinterpret_cast<uint64_t>(qr),reinterpret_cast<uint64_t>(kr),reinterpret_cast<uint64_t>(kras),
        uint64_t(nh),word(scale),uint64_t(pre),uint64_t(next),reinterpret_cast<uint64_t>(layout),uint64_t(nkh),uint64_t(sparse),
        uint64_t(precise),uint64_t(block),uint64_t(am),uint64_t(lse),uint64_t(kam),uint64_t(vam),
        reinterpret_cast<uint64_t>(out),reinterpret_cast<uint64_t>(sout),reinterpret_cast<uint64_t>(bytes),reinterpret_cast<uint64_t>(executor)};
    static_assert(sizeof(a)/sizeof(a[0])==dcat_fij::FormalCount,"SDK formal count");
    const int64_t scalars[]={nh,pre,next,nkh,sparse,precise,block,am,kam,vam};call.prepare(a,scalars,scale,lse,layout);
    try{auto rc=fapi.prepare(q,k,v,pse,mask,seq,seqkv,dq1,qs1,dq2,qs2,qo2,as,ao,blocks,qpad,kvpad,kas,kao,vas,vao,kp,vp,
        plen,qr,kr,kras,nh,scale,pre,next,layout,nkh,sparse,precise,block,am,lse,kam,vam,out,sout,bytes,executor);
        call.prepared(bytes,executor,rc);return rc;
    }catch(...){call.prepared(bytes,executor,0,true);throw;}
}
extern "C" aclnnStatus aclnnFusedInferAttentionScoreV3(void* workspace,uint64_t bytes,aclOpExecutor* executor,aclrtStream stream) {
    if(!fready())return 161001;
    FInvocation call(dcat_fij::Execute,__builtin_return_address(0),reinterpret_cast<void*>(fapi.execute));
    call.execute(workspace,bytes,executor,stream);
    try{auto rc=fapi.execute(workspace,bytes,executor,stream);call.executed(rc);return rc;}
    catch(...){call.executed(0,true);throw;}
}
