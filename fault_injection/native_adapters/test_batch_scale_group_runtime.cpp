// The production runtime is compiled into this CPU-only test translation unit
// so its private Api can point to explicit memory mocks. No injection API is
// exported by the actual DSO. SDK builds use real headers; host-only checks use
// tests/native_quant_group_mock_sdk and cannot claim CANN ABI validation.
#include "batch_scale_group_runtime.cpp"
#include <cassert>
#include <iostream>
#include <sys/wait.h>
namespace {
struct Tensor {
    void* data{};aclDataType type{};uint64_t rank{};
    int64_t shape[2]{},strides[2]{};
};
struct Executor {Tensor x{},q{},s{};uint64_t workspace{16};};
Tensor* tensor(const aclTensor* t){return reinterpret_cast<Tensor*>(const_cast<aclTensor*>(t));}
std::map<void*,size_t> allocations;
std::set<const aclTensor*> descriptors;
uint64_t allocations_made{},releases{},real_prepares{},real_executes{},copies{},late_copies{};
bool public_storage_live=true,uniform_outputs=false,fail_completion=false;
int fail_prepare_at=0,fail_execute_at=0;uint64_t fail_sync_stream=0;
int selected_mode=Reference;
Config configured{};Key selected_target{{1,2,3,4}},other_target{{5,6,7,8}},rows_proof{{11,12,13,14}};
aclError fake_device(int32_t* d){*d=0;return 0;}
aclnnStatus fake_type(const aclTensor* t,aclDataType* out){*out=tensor(t)->type;return 0;}
aclnnStatus fake_format(const aclTensor*,aclFormat* out){*out=ACL_FORMAT_ND;return 0;}
aclnnStatus fake_offset(const aclTensor*,int64_t* out){*out=0;return 0;}
aclnnStatus fake_shape(const aclTensor* t,int64_t** out,uint64_t* n){*out=tensor(t)->shape;*n=tensor(t)->rank;return 0;}
aclnnStatus fake_strides(const aclTensor* t,int64_t** out,uint64_t* n){*out=tensor(t)->strides;*n=tensor(t)->rank;return 0;}
aclnnStatus fake_address(const aclTensor* t,void** out){*out=tensor(t)->data;return 0;}
aclTensor* fake_create(const int64_t* shape,uint64_t rank,aclDataType type,const int64_t* strides,int64_t offset,
    aclFormat format,const int64_t* storage,uint64_t stored,void* address) {
    assert(rank>=1 && rank<=2 && !offset && format==ACL_FORMAT_ND && stored==rank && std::equal(shape,shape+rank,storage));
    auto* value=new Tensor{};value->data=address;value->type=type;value->rank=rank;
    std::copy(shape,shape+rank,value->shape);std::copy(strides,strides+rank,value->strides);
    auto* result=reinterpret_cast<aclTensor*>(value);descriptors.insert(result);return result;
}
aclnnStatus fake_destroy(const aclTensor* t){assert(descriptors.erase(t)==1);delete tensor(t);return 0;}
aclError fake_allocate(void** p,size_t bytes,aclrtMemMallocPolicy){*p=std::malloc(bytes);assert(*p);allocations[*p]=bytes;++allocations_made;return 0;}
aclError fake_release(void* p){assert(allocations.erase(p)==1);std::free(p);++releases;return 0;}
aclError fake_copy(void* out,size_t cap,const void* in,size_t bytes,aclrtMemcpyKind) {
    ++copies;if(!public_storage_live){++late_copies;return 1;}
    assert(cap>=bytes);std::memmove(out,in,bytes);return 0;
}
aclError fake_sync(aclrtStream stream){return reinterpret_cast<uint64_t>(stream)==fail_sync_stream?1:0;}
bool fake_cache(const char*){return true;}
aclnnStatus fake_prepare(const aclTensor* x,const aclTensor* a,const aclTensor* b,int64_t dtype,const aclTensor* q,
    const aclTensor* s,const aclTensor* o,uint64_t* workspace,aclOpExecutor** executor) {
    assert(!a && !b && !o && dtype==ACL_INT8);++real_prepares;
    auto* record=new Executor{*tensor(x),*tensor(q),*tensor(s),16};*workspace=16;*executor=reinterpret_cast<aclOpExecutor*>(record);
    return int(real_prepares)==fail_prepare_at?37:0;
}
aclnnStatus fake_execute(void* workspace,uint64_t size,aclOpExecutor* handle,aclrtStream) {
    ++real_executes;auto* record=reinterpret_cast<Executor*>(handle);assert(workspace && size>=record->workspace);
    const int8_t correct[8]{1,2,3,4,5,6,7,8},fault[8]{1,1,2,2,3,3,4,4};
    const bool flattened=record->x.shape[0]==1;
    assert(record->x.data && record->q.data && record->s.data);
    std::memcpy(record->q.data,(flattened || uniform_outputs)?fault:correct,8);
    const float scale[2]{uniform_outputs?.5f:.25f,.5f};
    if(flattened){const float scalar=.5f;std::memcpy(record->s.data,&scalar,4);}
    else std::memcpy(record->s.data,scale,8);
    delete record;return int(real_executes)==fail_execute_at?19:0;
}
int observe(const Call* call) {
    if(!selected_mode)return dcat_dqg_pass_v1(&configured,call);
    Lease lease{};lease.mode=selected_mode;lease.owner_hi=configured.owner_hi;lease.owner_lo=configured.owner_lo;
    lease.generation=configured.generation;lease.thread=configured.thread;lease.target=selected_target;
    lease.actual_row_proof=rows_proof;lease.expected=*call;return dcat_dqg_continue_v1(&lease);
}
int finish(const Completion* value){assert(value->call.token);return fail_completion?1:0;}
void install() {
    api.device=fake_device;api.type=fake_type;api.format=fake_format;api.offset=fake_offset;
    api.shape=fake_shape;api.storage=fake_shape;api.strides=fake_strides;api.address=fake_address;
    api.create=fake_create;api.destroy=fake_destroy;api.allocate=fake_allocate;api.release=fake_release;
    api.copy=fake_copy;api.sync=fake_sync;api.prepare=fake_prepare;api.execute=fake_execute;api.cache=fake_cache;api.ready=true;
    setenv("VLLM_SERVER_DEV_MODE","1",1);setenv("DCAT_DQ_GROUP_V1_ALLOWED","1",1);unsetenv("DCAT_DQ_V2_ENABLE");
    configured.max_targets=8;configured.max_calls=64;configured.owner_hi=51;configured.owner_lo=52;configured.generation=1;
    configured.thread=tid();configured.device=0;configured.max_device_bytes=1<<20;configured.max_host_bytes=1<<20;
    assert(dcat_dqg_install_v1(&configured,observe,finish)==Ok);
}
struct Invocation {
    std::array<uint16_t,8> x{{1,2,3,4,5,6,7,8}};
    std::array<int8_t,8> q{{-1,-1,-1,-1,-1,-1,-1,-1}};
    std::array<float,2> s{{-1,-1}};
    Tensor xx{x.data(),ACL_BF16,2,{2,4},{4,1}},qq{q.data(),ACL_INT8,2,{2,4},{4,1}},ss{s.data(),ACL_FLOAT,1,{2,0},{1,0}};
    uint64_t token{},workspace_bytes{};aclOpExecutor* executor{};Completion result{};
    std::vector<unsigned char> workspace;
    void begin() {
        PublicInput input{};input.thread=tid();input.device=0;input.input=view(reinterpret_cast<const aclTensor*>(&xx));
        input.input.descriptor=0;assert(dcat_dqg_public_begin_v1(&configured,&input,&token)==Ok);
    }
    int prepare() {
        return aclnnDynamicQuantV2GetWorkspaceSize(reinterpret_cast<const aclTensor*>(&xx),nullptr,nullptr,ACL_INT8,
            reinterpret_cast<const aclTensor*>(&qq),reinterpret_cast<const aclTensor*>(&ss),nullptr,&workspace_bytes,&executor);
    }
    int execute(uint64_t stream=71) {
        workspace.resize(workspace_bytes);return aclnnDynamicQuantV2(workspace.data(),workspace.size(),executor,reinterpret_cast<aclrtStream>(stream));
    }
    int end(int failed=0) {
        auto quant=view(reinterpret_cast<const aclTensor*>(&qq)),scale=view(reinterpret_cast<const aclTensor*>(&ss));
        quant.descriptor=scale.descriptor=0;
        return dcat_dqg_public_end_v1(&configured,token,&quant,&scale,failed,&result);
    }
    void success(uint64_t stream=71) {begin();assert(!prepare());assert(!execute(stream));assert(end()==Ok);}
};
void qualify(uint64_t stream=71) {for(int mode:{Reference,Lossless,Off}){selected_mode=mode;Invocation call;call.success(stream);}}
void seal() {assert(dcat_dqg_seal_v1(&configured,&selected_target,1,0,&rows_proof)==Ok);}
void stop_drain_recover() {
    assert(dcat_dqg_stop_v1(&configured,&selected_target)==Ok);
    assert(dcat_dqg_drain_v1(&configured,&selected_target)==Ok);
    assert(dcat_dqg_recover_v1(&configured)==Ok);
}
void happy_actual_typed_children_and_off() {
    install();qualify();seal();selected_mode=Fault;const auto census=journal.state();const auto prepares=real_prepares,executes=real_executes;
    Invocation call;call.x[7]=999;call.success();
    assert(real_prepares-prepares==2 && real_executes-executes==2 && journal.state().total-census.total==9);
    assert(call.result.core.reference_effect_compared && call.result.core.changed_quant_bytes && call.result.core.changed_scale_bytes
        && call.result.core.input_unchanged && call.result.core.released && call.x[7]==999);
    const auto& r=call.result.core;
    RawRecord raw{};assert(dcat_dqg_raw_v1(r.reference.prepare_enter,&raw)==Ok);
    assert(raw.kind==2 && raw.binding.role==CorrectReference && raw.native.input.address==r.effect.plan.input.address
        && raw.native.quant.address!=r.effect.plan.quant.address && raw.native.input.rows==2 && r.effect.plan.input.rows==1);
    assert(allocations.empty() && descriptors.empty());
    selected_mode=Off;Invocation off;off.success();stop_drain_recover();
    assert(dcat_dqg_remove_v1(&configured,observe,finish)==Ok);
}
void pass_has_one_real_execution_and_no_group_allocation() {
    install();selected_mode=0;Invocation call;call.success();
    assert(real_prepares==1 && real_executes==1 && allocations_made==0 && !call.result.admitted);
    assert(journal.state().helper_prepares==0 && journal.state().unclassified_prepares==1);
}
void prepared_without_execute_blocks_and_expires_lifetime() {
    install();qualify();seal();selected_mode=Fault;Invocation call;call.begin();assert(!call.prepare());
    assert(journal.state().pending==2 && !runtime.native_stack_active);
    assert(call.end(1)==Liveness);public_storage_live=false;const auto before=copies;
    assert(dcat_dqg_stop_v1(&configured,&selected_target)==Ok && dcat_dqg_drain_v1(&configured,&selected_target)==Pending);
    assert(dcat_dqg_recover_v1(&configured)==Liveness && copies==before && !late_copies && !allocations.empty());
    assert(dcat_dqg_remove_v1(&configured,observe,finish)==Pending);
}
void gws_error_with_nonnull_executor_is_never_ignored() {
    install();qualify();seal();selected_mode=Fault;fail_prepare_at=int(real_prepares)+1;
    Invocation call;call.begin();assert(call.prepare());assert(journal.state().pending==1);
    assert(call.end(1)==Liveness && runtime.quarantined && !allocations.empty());
    public_storage_live=false;assert(dcat_dqg_recover_v1(&configured)==Liveness && !late_copies);
}
void stack_failure_can_restore_before_return_only() {
    install();qualify();seal();selected_mode=Fault;fail_completion=true;
    Invocation call;const auto q=call.q;const auto s=call.s;call.begin();assert(!call.prepare());assert(call.execute());
    assert(call.q==q && call.s==s && allocations.empty() && descriptors.empty());
    assert(call.end(1)==Liveness && call.result.core.public_bytes_restored && call.result.core.released);
    assert(!engine.state().installed && !engine.state().pending_parent && !runtime.native_stack_active);
}
void failed_stream_keeps_ownership_after_return() {
    install();qualify();seal();selected_mode=Fault;Invocation call;call.begin();assert(!call.prepare());
    fail_sync_stream=71;assert(call.execute());assert(call.end(1)==Liveness);
    public_storage_live=false;fail_sync_stream=0;const auto before=copies;
    assert(dcat_dqg_recover_v1(&configured)==Liveness && copies==before && !late_copies && !allocations.empty());
}
void genuine_no_effect_does_not_publish() {
    install();qualify();seal();selected_mode=Fault;uniform_outputs=true;
    Invocation call;const auto q=call.q;const auto s=call.s;call.begin();assert(!call.prepare());assert(call.execute());
    assert(call.end(1)==Liveness && call.result.core.error==NoEffect && !call.result.core.published);
    assert(call.q==q && call.s==s && call.result.core.released && allocations.empty());
}
void inactive_stack_rejects_forged_lifetime_and_unknown_row() {
    install();qualify();seal();selected_mode=Fault;Invocation call;call.begin();
    const auto saved=rows_proof;rows_proof=Key{};assert(call.prepare());rows_proof=saved;
    assert(call.end(1)==Liveness && real_executes==3 && !engine.state().pending_parent);
    assert(dcat_dqg_recover_v1(&configured)==Pending);
}
void exported_recovery_requires_all_target_drains() {
    install();const Key first=selected_target;qualify(71);selected_target=other_target;qualify(72);
    const Key keys[2]{first,other_target};assert(dcat_dqg_seal_v1(&configured,keys,2,0,&rows_proof)==Ok);
    const auto host_bytes=engine.state().host_bytes;
    assert(dcat_dqg_stop_v1(&configured,&first)==Ok && dcat_dqg_drain_v1(&configured,&first)==Pending);
    assert(dcat_dqg_stop_v1(&configured,&other_target)==Ok && dcat_dqg_drain_v1(&configured,&first)==Ok);
    fail_sync_stream=72;assert(dcat_dqg_drain_v1(&configured,&other_target)==Pending);
    assert(dcat_dqg_recover_v1(&configured)==Pending && engine.state().host_bytes==host_bytes);
    fail_sync_stream=0;assert(dcat_dqg_drain_v1(&configured,&other_target)==Ok);
    assert(dcat_dqg_recover_v1(&configured)==Ok && !engine.state().host_bytes);
    assert(dcat_dqg_remove_v1(&configured,observe,finish)==Ok);
}
void run(const char* name,void(*test)()) {
    const pid_t pid=fork();assert(pid>=0);
    if(!pid){test();std::cout<<"PASS "<<name<<"\n"<<std::flush;_exit(0);}
    int status=0;assert(waitpid(pid,&status,0)==pid);assert(WIFEXITED(status) && WEXITSTATUS(status)==0);
}
}
int main() {
    AbiSizes sizes{};assert(dcat_dqg_abi_v1(&sizes)==Ok);assert(sizes.runtime==sizeof(RuntimeState) && sizes.event==sizeof(NativeEvent));
    assert(dcat_dqg_install_v1(nullptr,nullptr,nullptr)==Identity);RuntimeState state{};assert(dcat_dqg_state_v1(&state)==Ok);
    assert(!state.core.installed && !state.census.native_prepares && !state.public_active);
    run("typed children / OFF",happy_actual_typed_children_and_off);
    run("passthrough denominator",pass_has_one_real_execution_and_no_group_allocation);
    run("prepared executor / expired lifetime",prepared_without_execute_blocks_and_expires_lifetime);
    run("GWS nonnull executor failure",gws_error_with_nonnull_executor_is_never_ignored);
    run("in-stack exact rollback",stack_failure_can_restore_before_return_only);
    run("stream failure quarantine",failed_stream_keeps_ownership_after_return);
    run("actual fixture no effect",genuine_no_effect_does_not_publish);
    run("unknown rows refused",inactive_stack_rejects_forged_lifetime_and_unknown_row);
    run("exported whole-group recovery barrier",exported_recovery_requires_all_target_drains);
    std::cout<<"PASS 9 typed-shim CPU memory contracts; actual NPU calls=0; mocked Native functions; SDK status depends on build headers\n";
}
