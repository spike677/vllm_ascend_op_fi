// Actual typed shim and SDK signatures; numerical functions are CPU doubles.
#ifndef DCAT_QCT_GMM2_EXTENSION
#define DCAT_QCT_GMM2_EXTENSION
#endif
#define DCAT_QCT_NO_TEST_MAIN
#include "test_quant_container_trace.cpp"

namespace {
std::map<uint64_t,uint64_t> sizes;
uint64_t gprepare_calls{},gexecute_calls{},device_calls{},last_formals[23]{};
int prepare_status{},execute_status{},device_status{},logical_device{};
bool prepare_throw{},execute_throw{},prepare_no_executor{},owner_good{true},caller_good{true},size_bad{},size_error{};
struct FakeExecutor {uint64_t serial;};
FakeExecutor repeat_executor{};bool executor_reuse{},active_scope_guard{};
const Config* held_config{};
aclTensorList* counted_list(const aclTensor*const* v,uint64_t n){auto* p=create_list(v,n);sizes[reinterpret_cast<uint64_t>(p)]=n;return p;}
aclIntArray* counted_ints(const int64_t* v,uint64_t n){auto* p=create_ints(v,n);sizes[reinterpret_cast<uint64_t>(p)]=n;return p;}
aclnnStatus list_count(const aclTensorList* p,uint64_t* out){*out=sizes.at(reinterpret_cast<uint64_t>(p))+size_bad;return size_error?13:0;}
aclnnStatus int_count(const aclIntArray* p,uint64_t* out){*out=sizes.at(reinterpret_cast<uint64_t>(p))+size_bad;return size_error?13:0;}
aclError current_device(int32_t* out){++device_calls;*out=logical_device;return device_status;}
aclnnStatus prepare_double(const aclTensorList* x,const aclTensorList* weight,const aclTensorList* bias,
    const aclTensorList* scale,const aclTensorList* off,const aclTensorList* as,const aclTensorList* ao,
    const aclTensorList* token,const aclTensor* group,const aclTensorList* act,const aclTensorList* acts,
    const aclTensorList* acto,int64_t split,int64_t gt,int64_t glt,int64_t at,aclIntArray* tune,int64_t qg,
    aclTensorList* output,aclTensorList* feature,aclTensorList* dyn,uint64_t* ws,aclOpExecutor** ex) {
    ++gprepare_calls;
    const uint64_t actual[]={reinterpret_cast<uint64_t>(x),reinterpret_cast<uint64_t>(weight),reinterpret_cast<uint64_t>(bias),
        reinterpret_cast<uint64_t>(scale),reinterpret_cast<uint64_t>(off),reinterpret_cast<uint64_t>(as),
        reinterpret_cast<uint64_t>(ao),reinterpret_cast<uint64_t>(token),reinterpret_cast<uint64_t>(group),
        reinterpret_cast<uint64_t>(act),reinterpret_cast<uint64_t>(acts),reinterpret_cast<uint64_t>(acto),
        static_cast<uint64_t>(split),static_cast<uint64_t>(gt),static_cast<uint64_t>(glt),static_cast<uint64_t>(at),
        reinterpret_cast<uint64_t>(tune),static_cast<uint64_t>(qg),reinterpret_cast<uint64_t>(output),
        reinterpret_cast<uint64_t>(feature),reinterpret_cast<uint64_t>(dyn),reinterpret_cast<uint64_t>(ws),reinterpret_cast<uint64_t>(ex)};
    std::copy(actual,actual+23,last_formals);
    if(active_scope_guard){assert(gstate.active_calls==1 && dcat_qct_scope_end_v1(held_config,scope.token));
        assert(dcat_qct_stop_v1(held_config) && dcat_qct_release_v1(held_config));}
    *ws=64;*ex=prepare_no_executor?nullptr:reinterpret_cast<aclOpExecutor*>(
        executor_reuse?&repeat_executor:new FakeExecutor{gprepare_calls});
    if(prepare_throw)throw std::runtime_error("real CPU double produced executor then threw");
    return prepare_status;
}
aclnnStatus execute_double(void*,uint64_t,aclOpExecutor* ex,aclrtStream) {
    ++gexecute_calls;
    if(execute_throw)throw std::runtime_error("real CPU execute double threw");
    if(!execute_status && reinterpret_cast<FakeExecutor*>(ex)!=&repeat_executor)delete reinterpret_cast<FakeExecutor*>(ex);
    return execute_status;
}
dcat_qg2::State gquery(){dcat_qg2::State s;assert(!dcat_qct_gmm2_state_v1(&s));return s;}
struct Case {
    Config c;
    Scope s;
    dcat_qg2::Limits limits;
    FakeTensor tensor;
    aclTensorList* lists[14]{};
    aclIntArray* integers{};
    uint64_t workspace_bytes{};
    aclOpExecutor* executor{};
    char workspace[64]{};
    Case(uint64_t max_calls=dcat_qg2::MaxCalls,uint64_t max_pairs=dcat_qg2::MaxPairs):c(begin()) {
        api.create_list=counted_list;api.create_ints=counted_ints;
        std::call_once(gapi_once,[]{});gapi.ready=true;gapi.prepare=prepare_double;gapi.execute=execute_double;
        gapi.list_size=list_count;gapi.int_size=int_count;gapi.device=current_device;
        gapi.owner_matches=[](const void*){return owner_good;};
        gapi.caller_matches=[](const void*,uint32_t){return caller_good;};
        assert(!dcat_qct_gmm2_resolve_v1() && !device_calls && !gprepare_calls && !gexecute_calls);
        limits.max_calls=max_calls;limits.max_pairs=max_pairs;
        held_config=&c;
    }
    void start() {assert(!dcat_qct_gmm2_begin_v1(&c,&limits));
        s.token=1;s.forward=5;s.operator_sequence=761;s.boundary_sequence=8;s.stage=3;s.thread=tid();
        assert(!dcat_qct_scope_begin_v1(&c,&s));}
    void containers() {
        const aclTensor* values[]={reinterpret_cast<const aclTensor*>(&tensor)};
        // Reverse construction order deliberately differs from the named ABI.
        for(int i=13;i>=0;--i)lists[i]=aclCreateTensorList(values,(i==0||i==1||i==3||i==7||i==11)?1:0);
        const int64_t values_i[]={3,9};integers=aclCreateIntArray(values_i,2);
    }
    int prepare() {workspace_bytes=0;executor=nullptr;
        return aclnnGroupedMatmulWeightNzGetWorkspaceSize(lists[0],lists[1],lists[2],lists[3],lists[4],lists[5],lists[6],lists[7],
            reinterpret_cast<const aclTensor*>(&tensor),lists[8],lists[9],lists[10],2,0,1,0,integers,-11,
            lists[11],lists[12],lists[13],&workspace_bytes,&executor);}
    int execute(uint64_t bytes=64){return aclnnGroupedMatmulWeightNz(workspace,bytes,executor,reinterpret_cast<aclrtStream>(0x1234));}
    void destroy() {for(auto*& p:lists){if(p){assert(!aclDestroyTensorList(p));p=nullptr;}}
        if(integers){assert(!aclDestroyIntArray(integers));integers=nullptr;}}
    void stop() {assert(!dcat_qct_scope_end_v1(&c,s.token));assert(!dcat_qct_gmm2_stop_v1(&c));assert(!dcat_qct_stop_v1(&c));}
    void release() {assert(!dcat_qct_gmm2_release_v1(&c));assert(!dcat_qct_release_v1(&c));
        const auto g=gquery();assert(!g.installed && g.retired && !g.record_capacity_bytes && !g.pending_executors);
        const auto four=query();assert(!four.installed && !four.record_capacity_bytes && !four.payload_capacity_bytes
            && !four.live_registrations && !four.arrays.live_arrays);}
    void retained() {assert(dcat_qct_gmm2_release_v1(&c));assert(dcat_qct_release_v1(&c));
        assert(gquery().installed && gquery().record_capacity_bytes && query().installed);}
};
void run_gmm(const std::string& name) {
    if(name=="original_four"){test("normal");return;}
    Case c(name=="call_budget"?1:dcat_qg2::MaxCalls,name=="pair_budget"?1:dcat_qg2::MaxPairs);
    if(name=="byte_budget") {c.limits.max_bytes=1;assert(dcat_qct_gmm2_begin_v1(&c.c,&c.limits));
        assert(!gquery().installed && !gquery().record_capacity_bytes);finish(c.c);return;}
    c.start();c.containers();
    if(name=="size_mismatch")size_bad=true;
    if(name=="size_status")size_error=true;
    if(name=="getter_throw")throw_shape=true;
    if(name=="wrong_caller")caller_good=false;
    if(name=="changed_owner")owner_good=false;
    if(name=="device_error")device_status=23;
    if(name=="failed_prepare" || name=="failed_prepare_null")prepare_status=37;
    if(name=="failed_prepare_null")prepare_no_executor=true;
    if(name=="prepare_throw")prepare_throw=true;
    if(name=="executor_reused")executor_reuse=true;
    if(name=="active_scope_guard")active_scope_guard=true;
    if(name=="null_optional"){assert(!aclDestroyTensorList(c.lists[2]));c.lists[2]=nullptr;}
    if(name=="unknown_container") {
        assert(!aclDestroyTensorList(c.lists[0]));c.lists[0]=reinterpret_cast<aclTensorList*>(new FakeObject{1});
        // It has no observed current registration; no size getter may dereference it.
    }
    if(name=="foreign_thread") {std::thread other([&]{assert(!c.prepare());});other.join();}
    else {const int result=c.prepare();assert(result==(prepare_throw?161001:prepare_status));}
    assert(gprepare_calls==1);
    dcat_qg2::Event first;assert(!dcat_qct_gmm2_event_v1(0,&first));
    assert(std::equal(std::begin(first.formals),std::end(first.formals),std::begin(last_formals)));
    assert(first.scalars[4]==-11 && first.formals[17]==static_cast<uint64_t>(-11));
    if(name=="normal") {
        assert(!first.error);
        for(const auto& slot:first.containers)assert(slot.joined && slot.actual_count==slot.count);
        assert(first.containers[0].registration>first.containers[14].registration);
    }
    bool retained=false;
    if(name=="pair_budget") {
        auto* first_executor=c.executor;assert(!c.prepare());auto* second=c.executor;
        c.executor=first_executor;assert(!c.execute());c.executor=second;assert(!c.execute());retained=true;
        assert(gquery().untracked_executors==1);
    } else if(name=="no_execute" || name=="failed_prepare" || name=="prepare_throw")retained=true;
    else if(name=="failed_prepare_null")assert(!c.executor);
    else {
        if(name=="executor_swapped")c.executor=reinterpret_cast<aclOpExecutor*>(new FakeExecutor{999});
        if(name=="scope_changed"){assert(!dcat_qct_scope_end_v1(&c.c,c.s.token));++c.s.token;++c.s.forward;
            assert(!dcat_qct_scope_begin_v1(&c.c,&c.s));}
        if(name=="wrong_device")logical_device=1;
        if(name=="execute_error" || name=="execute_retry")execute_status=41;
        if(name=="execute_throw")execute_throw=true;
        const auto result=c.execute(name=="workspace_short"?63:64);
        assert(result==(execute_throw?161001:execute_status));assert(gexecute_calls==1);
        retained=name=="execute_error" || name=="execute_throw" || name=="workspace_short" || name=="wrong_device"
            || name=="device_error" || name=="foreign_thread" || name=="wrong_caller" || name=="changed_owner" || name=="call_budget"
            || name=="executor_swapped" || name=="scope_changed";
        if(name=="execute_retry") {assert(gquery().pending_executors==1);execute_status=0;assert(!c.execute());
            assert(gexecute_calls==2 && !gquery().pending_executors && gquery().first_error==dcat_qg2::Original);}
        if(name=="executor_reused") {auto* old=c.executor;assert(!c.prepare() && c.executor==old);assert(!c.execute());
            dcat_qg2::Event next;assert(!dcat_qct_gmm2_event_v1(4,&next));
            assert(next.prepare_registration!=first.prepare_registration && !gquery().pending_executors);}
    }
    if(name=="normal") {
        auto s=gquery();assert(s.total_calls==2 && s.total_events==4 && s.retained_events==4
            && s.paired_execute_returns==1 && !s.first_error && !s.pending_executors);
        dcat_qg2::Event last;assert(!dcat_qct_gmm2_event_v1(3,&last));
        assert(last.prepare_registration==first.call && last.original_calls==1 && last.stream==0x1234
            && last.workspace_bytes==64 && last.workspace==reinterpret_cast<uint64_t>(c.workspace));
        assert(query().total_calls==15); // GWS/execute never enter the four-API denominator.
    }else if(name!="executor_reused" && name!="null_optional" && name!="release_order" && name!="active_scope_guard")
        assert(gquery().first_error || name=="no_execute");
    throw_shape=false;c.destroy();c.stop();
    if(name=="release_order")assert(dcat_qct_release_v1(&c.c) && query().installed && gquery().installed);
    if(retained)c.retained();else c.release();
    assert(!query().arrays.live_arrays && !query().arrays.active_captures && !gquery().active_calls);
}
}
int main(int argc,char** argv){assert(argc==2);run_gmm(argv[1]);
    std::cout<<"PASS_CPU_SDK_GMM2_TYPED_FUNCTION_DOUBLE_NOT_NPU "<<argv[1]<<"\n";return 0;}
