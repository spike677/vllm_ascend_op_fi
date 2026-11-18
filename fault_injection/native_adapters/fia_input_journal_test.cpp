// Real production TU and real SDK types; numerical/original functions are CPU doubles.
#include "fia_input_journal.cpp"
#include <cassert>
#include <iostream>
#include <stdexcept>
#include <thread>

namespace {
struct FakeTensor {int64_t dims[3]{2048,8,128},stride[3]{4096,128,1},offset{3072},storage{8388608};};
struct FakeObject {uint64_t count;};
uint64_t originals[4]{},prepares{},executes{},devices{};
bool getter_throw{},getter_status{},size_bad{},owner_good{true},caller_good{true},scope_good{true},
    prepare_throw{},execute_throw{},null_executor{},reuse_executor{},active_guard{},destroy_fail{},untouched_outputs{};
int prepare_status{},execute_status{};
FakeObject reuse{};FakeObject* repeated_list{};
const Config* held{};
aclTensorList* clist(const aclTensor*const*,uint64_t n){++originals[0];auto* p=repeated_list?repeated_list:new FakeObject{};p->count=n;return reinterpret_cast<aclTensorList*>(p);}
aclnnStatus dlist(const aclTensorList* p){++originals[1];if(destroy_fail)return 13;
    if(p && reinterpret_cast<const FakeObject*>(p)!=repeated_list)delete reinterpret_cast<const FakeObject*>(p);return 0;}
aclIntArray* cints(const int64_t*,uint64_t n){++originals[2];return reinterpret_cast<aclIntArray*>(new FakeObject{n});}
aclnnStatus dints(const aclIntArray* p){++originals[3];delete reinterpret_cast<const FakeObject*>(p);return 0;}
aclnnStatus dt(const aclTensor*,aclDataType* p){*p=ACL_FLOAT16;return 0;}
aclnnStatus fmt(const aclTensor*,aclFormat* p){*p=ACL_FORMAT_ND;return 0;}
aclnnStatus off(const aclTensor* t,int64_t* p){*p=reinterpret_cast<const FakeTensor*>(t)->offset;return 0;}
aclnnStatus sh(const aclTensor* t,int64_t** p,uint64_t* n){*n=3;*p=new int64_t[3];auto* f=reinterpret_cast<const FakeTensor*>(t);
    std::copy(f->dims,f->dims+3,*p);if(getter_throw)throw std::runtime_error("owned getter array assigned before throw");return getter_status?5:0;}
aclnnStatus st(const aclTensor* t,int64_t** p,uint64_t* n){*n=3;*p=new int64_t[3];auto* f=reinterpret_cast<const FakeTensor*>(t);std::copy(f->stride,f->stride+3,*p);return 0;}
aclnnStatus ss(const aclTensor* t,int64_t** p,uint64_t* n){*n=1;*p=new int64_t[1]{reinterpret_cast<const FakeTensor*>(t)->storage};return 0;}
aclnnStatus addr(const aclTensor* t,void** p){*p=const_cast<aclTensor*>(t);return 0;}
aclnnStatus ls(const aclTensorList* t,uint64_t* n){*n=reinterpret_cast<const FakeObject*>(t)->count+size_bad;return 0;}
aclnnStatus is(const aclIntArray* t,uint64_t* n){*n=reinterpret_cast<const FakeObject*>(t)->count+size_bad;return 0;}
aclError dev(int32_t* p){++devices;*p=0;return 0;}
int active_scope(const char*){return scope_good?1:0;}
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-parameter"
aclnnStatus prep(const aclTensor* q,const aclTensorList* k,const aclTensorList* v,const aclTensor* pse,const aclTensor* mask,
    const aclIntArray* seq,const aclIntArray* seqkv,const aclTensor* dq1,const aclTensor* qs1,const aclTensor* dq2,
    const aclTensor* qs2,const aclTensor* qo2,const aclTensor* as,const aclTensor* ao,const aclTensor* blocks,
    const aclTensor* qpad,const aclTensor* kvpad,const aclTensor* kas,const aclTensor* kao,const aclTensor* vas,
    const aclTensor* vao,const aclTensor* kp,const aclTensor* vp,const aclIntArray* plen,const aclTensor* qr,
    const aclTensor* kr,const aclTensor* kras,int64_t nh,double scale,int64_t pre,int64_t next,char* layout,
    int64_t nkh,int64_t sparse,int64_t precise,int64_t block,int64_t am,bool lse,int64_t kam,int64_t vam,
    const aclTensor* out,const aclTensor* sout,uint64_t* bytes,aclOpExecutor** ex) {
    ++prepares;assert(q && k && v && mask && seq && seqkv && out);
    assert(nh==16 && scale==0.125 && pre==-7 && next==11 && nkh==8 && sparse==3 && precise==0
        && block==128 && am==0 && !lse && kam==0 && vam==0 && std::string(layout)=="TND");
    if(active_guard){assert(dcat_fia_input_journal_scope_end_v1(held,1));assert(dcat_fia_input_journal_stop_v1(held));}
    if(untouched_outputs)return prepare_status;
    *bytes=64;*ex=null_executor?nullptr:reinterpret_cast<aclOpExecutor*>(reuse_executor?&reuse:new FakeObject{prepares});
    if(prepare_throw)throw std::runtime_error("original prepare exception");return prepare_status;
}
#pragma GCC diagnostic pop
aclnnStatus execute(void*,uint64_t,aclOpExecutor* ex,aclrtStream){++executes;
    if(execute_throw)throw std::runtime_error("original execute exception");
    if(!execute_status && reinterpret_cast<FakeObject*>(ex)!=&reuse)delete reinterpret_cast<FakeObject*>(ex);return execute_status;}
dcat_fij::State fq(){dcat_fij::State s;assert(!dcat_fia_input_journal_state_v1(&s));return s;}
State cq(){State s;assert(!dcat_qct_state_v1(&s));return s;}
struct Case {
    Config c;dcat_fij::ScopeIdentity s;dcat_fij::Limits limits;FakeTensor tensor;
    aclTensorList* key{};aclTensorList* value{};aclIntArray* lengths{};aclIntArray* lengths_kv{};
    uint64_t bytes{};aclOpExecutor* ex{};char workspace[64]{};
    Case(){std::call_once(api_once,[]{});api.ready=true;api.create_list=clist;api.destroy_list=dlist;api.create_ints=cints;api.destroy_ints=dints;
        api.dtype=dt;api.format=fmt;api.offset=off;api.shape=sh;api.strides=st;api.storage=ss;api.address=addr;
        std::call_once(fonce,[]{});fapi.ready=true;fapi.prepare=prep;fapi.execute=execute;fapi.list_size=ls;fapi.int_size=is;fapi.device=dev;
        fapi.scope_matches=active_scope;fapi.owner_matches=[](const void*){return owner_good;};fapi.caller_matches=[](const void*){return caller_good;};
        assert(!dcat_fia_input_journal_resolve_v1() && !devices && !prepares && !executes);
        c.owner_hi=11;c.owner_lo=22;c.generation=1;c.thread=tid();held=&c;
        s.pid=getpid();s.worker=101;s.model=102;s.module=103;
        s.scope.token=1;s.scope.forward=1;s.scope.operator_sequence=1;s.scope.boundary_sequence=1;s.scope.stage=3;s.scope.thread=tid();
        std::memset(s.call_id,'a',32);s.call_id[32]=0;}
    void start(){assert(!dcat_qct_begin_v1(&c));assert(!dcat_fia_input_journal_begin_v1(&c,&limits));assert(!dcat_fia_input_journal_scope_begin_v1(&c,&s));}
    void members(){const aclTensor* t[]={reinterpret_cast<const aclTensor*>(&tensor)};const int64_t n[]={2048};
        key=aclCreateTensorList(t,1);value=aclCreateTensorList(t,1);lengths=aclCreateIntArray(n,1);lengths_kv=aclCreateIntArray(n,1);}
    int prepare(){bytes=0;ex=nullptr;const auto* t=reinterpret_cast<const aclTensor*>(&tensor);char layout[]="TND";
        return aclnnFusedInferAttentionScoreV3GetWorkspaceSize(t,key,value,nullptr,t,lengths,lengths_kv,
            nullptr,nullptr,nullptr,nullptr,nullptr,nullptr,nullptr,nullptr,nullptr,nullptr,nullptr,nullptr,nullptr,nullptr,
            nullptr,nullptr,nullptr,nullptr,nullptr,nullptr,16,.125,-7,11,layout,8,3,0,128,0,false,0,0,t,nullptr,&bytes,&ex);}
    int run(uint64_t n=64){return aclnnFusedInferAttentionScoreV3(workspace,n,ex,reinterpret_cast<aclrtStream>(0x1234));}
    void destroy(){if(key){assert(!aclDestroyTensorList(key));key=nullptr;}if(value){assert(!aclDestroyTensorList(value));value=nullptr;}
        if(lengths){assert(!aclDestroyIntArray(lengths));lengths=nullptr;}if(lengths_kv){assert(!aclDestroyIntArray(lengths_kv));lengths_kv=nullptr;}}
    void arrays(){auto a=cq().arrays;assert(a.acquired_arrays==a.released_arrays && !a.live_arrays
        && !a.active_captures && !a.active_getters && !a.ownership_errors);}
    void finish(bool close=true){if(close)assert(!dcat_fia_input_journal_scope_end_v1(&c,1));
        assert(fq().accepted_windows==1 && fq().retained_scope_tokens==1 && fq().retained_scope_token_characters==32);
        assert(!dcat_fia_input_journal_stop_v1(&c));assert(!dcat_qct_stop_v1(&c));
        assert(!dcat_fia_input_journal_release_v1(&c));assert(!dcat_qct_release_v1(&c));
        auto f=fq();auto q=cq();assert(f.retired && !f.installed && !f.record_capacity_bytes && !f.pending_executors
            && !f.retained_scope_tokens && !f.retained_scope_token_characters && f.accepted_windows==1
            && !q.installed && !q.live_registrations && !q.record_capacity_bytes && !q.payload_capacity_bytes);arrays();}
    void retained(){assert(dcat_fia_input_journal_scope_end_v1(&c,1));assert(dcat_fia_input_journal_stop_v1(&c));
        assert(dcat_fia_input_journal_release_v1(&c));assert(fq().installed && fq().record_capacity_bytes);arrays();}
};
void test(const std::string& name){
    Case c;
    if(name=="inactive"){c.members();assert(!c.prepare() && !c.run());c.destroy();assert(!fq().total_calls && !devices);return;}
    if(name=="window_limit_invalid"){
        c.limits.max_windows=0;assert(!dcat_qct_begin_v1(&c.c));
        assert(dcat_fia_input_journal_begin_v1(&c.c,&c.limits));assert(!fq().installed && !fq().record_capacity_bytes);
        assert(!dcat_qct_stop_v1(&c.c));assert(!dcat_qct_release_v1(&c.c));c.arrays();return;}
    if(name=="empty_window_budget"){
        c.limits.max_windows=1;c.start();assert(!dcat_fia_input_journal_scope_end_v1(&c.c,1));
        c.s.scope.token=2;std::memset(c.s.call_id,'b',32);
        assert(dcat_fia_input_journal_scope_begin_v1(&c.c,&c.s));
        assert(!fq().total_calls && fq().first_error==dcat_fij::Budget && !cq().scope_token);
        assert(!devices && !prepares && !executes);c.finish(false);return;}
    if(name=="call_budget")c.limits.max_calls=1;
    if(name=="pair_budget")c.limits.max_pairs=1;
    c.start();c.members();
    if(name=="getter_throw")getter_throw=true;
    if(name=="getter_status")getter_status=true;
    if(name=="size_mismatch")size_bad=true;
    if(name=="wrong_owner")owner_good=false;
    if(name=="wrong_caller")caller_good=false;
    if(name=="wrong_scope")scope_good=false;
    if(name=="unknown_member"){assert(!aclDestroyTensorList(c.key));c.key=reinterpret_cast<aclTensorList*>(new FakeObject{1});}
    if(name=="reused_container"){assert(!aclDestroyTensorList(c.key));FakeObject* saved=repeated_list;repeated_list=&reuse;
        const aclTensor* p[]={reinterpret_cast<const aclTensor*>(&c.tensor)};c.key=aclCreateTensorList(p,1);
        auto old=live.at(reinterpret_cast<uint64_t>(c.key)).id;assert(!aclDestroyTensorList(c.key));c.key=aclCreateTensorList(p,1);
        assert(live.at(reinterpret_cast<uint64_t>(c.key)).id!=old);(void)saved;}
    if(name=="failed_prepare" || name=="failed_prepare_null" || name=="failed_prepare_untouched")prepare_status=37;
    if(name=="failed_prepare_untouched")untouched_outputs=true;
    if(name=="failed_prepare_null")null_executor=true;
    if(name=="prepare_throw")prepare_throw=true;
    if(name=="executor_reused")reuse_executor=true;
    if(name=="active_guard")active_guard=true;
    if(name=="foreign_thread"){std::thread t([&]{assert(!c.prepare());});t.join();}
    else if(prepare_throw){try{c.prepare();assert(false);}catch(const std::runtime_error& e){assert(std::string(e.what())=="original prepare exception");}}
    else assert(c.prepare()==prepare_status);
    assert(prepares==1);c.arrays();
    if(name=="normal" || name=="reused_container" || name=="active_guard"){
        dcat_fij::Event e;assert(!dcat_fia_input_journal_event_v1(0,&e));assert(!e.error && e.operation==1 && e.phase==1);
        assert(e.tensors[0].strides[0]==4096 && e.tensors[0].offset==3072 && e.tensors[0].storage[0]==8388608);
        assert(e.formals[27]==16 && e.signed_scalars[1]==-7 && e.scale==.125 && std::string(e.layout)=="TND");
        for(const auto& slot:e.containers)assert(slot.joined && slot.count==slot.actual_count);
    }
    bool retain=prepare_throw || prepare_status || name=="no_execute" || name=="foreign_thread";
    if(prepare_throw || prepare_status){assert(fq().failed_prepare_outputs_unknown==1 && fq().untracked_executors==1);
        dcat_fij::Event failed;assert(!dcat_fia_input_journal_event_v1(1,&failed));assert(!failed.executor && !failed.workspace_bytes);}
    if(name=="pair_budget" || name=="executor_reused"){
        auto* first=c.ex;assert(!c.prepare());auto* second=c.ex;c.ex=first;assert(!c.run());
        if(second!=first){c.ex=second;assert(!c.run());}retain=true;assert(fq().untracked_executors==1);
    }else if(!retain && name!="failed_prepare_null"){
        if(name=="execute_error" || name=="execute_retry")execute_status=41;
        if(name=="execute_throw")execute_throw=true;
        if(execute_throw){try{c.run();assert(false);}catch(const std::runtime_error& e){assert(std::string(e.what())=="original execute exception");}retain=true;}
        else {assert(c.run(name=="workspace_short"?1:64)==execute_status);retain=execute_status || name=="workspace_short";}
        if(name=="execute_retry"){assert(fq().pending_executors==1);execute_status=0;assert(!c.run());retain=false;}
    }
    if(name=="normal" || name=="reused_container" || name=="active_guard")assert(!fq().first_error && fq().paired_execute_returns==1);
    else if(name!="no_execute")assert(fq().first_error || name=="call_budget");
    getter_throw=false;getter_status=false;c.destroy();
    if(retain)c.retained();else c.finish();
}
}
int main(int argc,char** argv){assert(argc==2);test(argv[1]);std::cout<<"PASS_CPU_FIA_INPUT_JOURNAL_NOT_NPU "<<argv[1]<<"\n";}
