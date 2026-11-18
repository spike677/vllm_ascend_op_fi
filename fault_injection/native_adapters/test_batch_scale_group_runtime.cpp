// The production runtime is compiled into this CPU-only test translation unit
// so its private Api can point to explicit memory mocks. No injection API is
// exported by the actual DSO. SDK builds use real headers; host-only checks use
// tests/native_quant_group_mock_sdk and cannot claim CANN ABI validation.
#include "batch_scale_group_runtime.cpp"
#include <cassert>
#include <iostream>
#include <new>
#include <sys/wait.h>
namespace array_accounting {
void* live[256]{};uint64_t allocated{},released{};
uint64_t count(){return std::count_if(std::begin(live),std::end(live),[](void* p){return p!=nullptr;});}
}
void* operator new[](size_t bytes) {
    void* p=std::malloc(bytes?bytes:1);if(!p)throw std::bad_alloc();
    auto slot=std::find(std::begin(array_accounting::live),std::end(array_accounting::live),nullptr);
    assert(slot!=std::end(array_accounting::live));*slot=p;++array_accounting::allocated;return p;
}
void operator delete[](void* p) noexcept {
    if(!p)return;
    auto slot=std::find(std::begin(array_accounting::live),std::end(array_accounting::live),p);
    assert(slot!=std::end(array_accounting::live));*slot=nullptr;++array_accounting::released;std::free(p);
}
void operator delete[](void* p,size_t) noexcept {::operator delete[](p);}
namespace {
struct Tensor {
    void* data{};aclDataType type{};uint64_t rank{};
    int64_t shape[12]{},strides[12]{};
    uint64_t storage_rank{UINT64_MAX},stride_rank{UINT64_MAX};int64_t storage[12]{};
};
struct Executor {Tensor x{},q{},s{};uint64_t workspace{16};};
Tensor* tensor(const aclTensor* t){return reinterpret_cast<Tensor*>(const_cast<aclTensor*>(t));}
std::map<void*,size_t> allocations;
std::set<const aclTensor*> descriptors;
uint64_t allocations_made{},releases{},real_prepares{},real_executes{},copies{},late_copies{},executor_handles{},getter_calls{};
bool public_storage_live=true,uniform_outputs=false,fail_completion=false;
bool throw_completion=false,throw_device=false;
uint64_t throw_copy_at{},throw_getter_at{},fail_getter_at{},null_getter_at{};
int fail_prepare_at=0,fail_execute_at=0;uint64_t fail_sync_stream=0;
int selected_mode=Reference;
Config configured{};Key selected_target{{1,2,3,4}},other_target{{5,6,7,8}},rows_proof{{11,12,13,14}};
aclError fake_device(int32_t* d){if(throw_device)throw std::bad_alloc();*d=0;return 0;}
aclnnStatus fake_type(const aclTensor* t,aclDataType* out){*out=tensor(t)->type;return 0;}
aclnnStatus fake_format(const aclTensor*,aclFormat* out){*out=ACL_FORMAT_ND;return 0;}
aclnnStatus fake_offset(const aclTensor*,int64_t* out){*out=0;return 0;}
aclnnStatus fake_dimensions(const aclTensor* t,int64_t** out,uint64_t* n,unsigned kind) {
    ++getter_calls;const auto* value=tensor(t);
    *n=kind==1 && value->stride_rank!=UINT64_MAX?value->stride_rank:
        kind==2 && value->storage_rank!=UINT64_MAX?value->storage_rank:value->rank;
    if(getter_calls==null_getter_at){*out=nullptr;return 0;}
    *out=new int64_t[*n];
    const auto* source=kind==1?value->strides:kind==2 && value->storage_rank!=UINT64_MAX?value->storage:value->shape;
    std::copy(source,source+*n,*out);
    if(getter_calls==throw_getter_at)throw std::bad_alloc();
    return getter_calls==fail_getter_at?37:0;
}
aclnnStatus fake_shape(const aclTensor* t,int64_t** out,uint64_t* n){return fake_dimensions(t,out,n,0);}
aclnnStatus fake_strides(const aclTensor* t,int64_t** out,uint64_t* n){return fake_dimensions(t,out,n,1);}
aclnnStatus fake_storage(const aclTensor* t,int64_t** out,uint64_t* n){return fake_dimensions(t,out,n,2);}
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
    ++copies;if(copies==throw_copy_at)throw std::bad_alloc();
    if(!public_storage_live){++late_copies;return 1;}
    assert(cap>=bytes);std::memmove(out,in,bytes);return 0;
}
aclError fake_sync(aclrtStream stream){return reinterpret_cast<uint64_t>(stream)==fail_sync_stream?1:0;}
bool fake_cache(const char*){return true;}
aclnnStatus fake_prepare(const aclTensor* x,const aclTensor* a,const aclTensor* b,int64_t dtype,const aclTensor* q,
    const aclTensor* s,const aclTensor* o,uint64_t* workspace,aclOpExecutor** executor) {
    assert(!a && !b && !o && dtype==ACL_INT8);++real_prepares;
    auto* record=new Executor{*tensor(x),*tensor(q),*tensor(s),16};++executor_handles;
    *workspace=16;*executor=reinterpret_cast<aclOpExecutor*>(record);
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
    if(int(real_executes)==fail_execute_at)return 19; // Consumption on error is deliberately NOT assumed.
    delete record;--executor_handles;return 0;
}
int observe(const Call* call) {
    if(!selected_mode)return dcat_dqg_pass_v1(&configured,call);
    Lease lease{};lease.mode=selected_mode;lease.owner_hi=configured.owner_hi;lease.owner_lo=configured.owner_lo;
    lease.generation=configured.generation;lease.thread=configured.thread;lease.target=selected_target;
    lease.actual_row_proof=rows_proof;lease.expected=*call;return dcat_dqg_continue_v1(&lease);
}
int finish(const Completion* value){assert(value->call.token);if(throw_completion)throw std::bad_alloc();return fail_completion?1:0;}
void install() {
    api.device=fake_device;api.type=fake_type;api.format=fake_format;api.offset=fake_offset;
    api.shape=fake_shape;api.storage=fake_storage;api.strides=fake_strides;api.address=fake_address;
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
void execute_error_cannot_infer_executor_consumption_from_sync() {
    install();qualify();seal();selected_mode=Fault;fail_execute_at=int(real_executes)+2;
    Invocation call;call.begin();assert(!call.prepare());const auto freed=releases;
    assert(call.execute() && executor_handles==1 && journal.state().pending==1);
    assert(releases==freed && !allocations.empty() && !runtime.native_stack_active);
    assert(dcat_dqg_drain_v1(&configured,&selected_target)==Pending);
    assert(call.end(1)==Liveness);public_storage_live=false;
    assert(dcat_dqg_recover_v1(&configured)==Liveness && !late_copies && releases==freed);
}
void throwing_execute_region_cannot_leave_a_live_witness(bool during_completion) {
    install();qualify();seal();selected_mode=Fault;Invocation call;call.begin();assert(!call.prepare());
    const auto freed=releases;
    if(during_completion)throw_completion=true;else throw_copy_at=copies+1;
    assert(call.execute());
    RuntimeState state{};assert(dcat_dqg_state_v1(&state)==Ok);
    assert(state.abi_exceptions==1 && state.quarantined && !state.native_stack_active
        && !state.callback_active && !state.completion_active && state.public_active);
    // The public envelope has deliberately not ended yet. A stale numerical
    // stack marker would therefore be enough to forge the old live witness.
    public_storage_live=false;const auto copy_count=copies;
    assert(dcat_dqg_recover_v1(&configured)==Liveness && copies==copy_count && !late_copies && releases==freed);
    assert(call.end(1)==Liveness);
}
void throw_during_start(){throwing_execute_region_cannot_leave_a_live_witness(false);}
void throw_during_completion(){throwing_execute_region_cannot_leave_a_live_witness(true);}
void control_abi_catches_exceptions() {
    install();PublicInput input{};input.thread=tid();input.device=0;uint64_t token=0;
    throw_device=true;assert(dcat_dqg_public_begin_v1(&configured,&input,&token)==Observer);throw_device=false;
    RuntimeState state{};assert(dcat_dqg_state_v1(&state)==Ok);
    assert(state.abi_exceptions==1 && state.quarantined && !state.public_active && !state.native_stack_active);
}
void sdk_metadata_arrays_release_on_all_view_exits() {
    install();Invocation call;const auto* input=reinterpret_cast<const aclTensor*>(&call.xx);
    const auto alive=array_accounting::count();
    for(uint64_t failure:{0u,1u,2u,3u}) {
        getter_calls=0;fail_getter_at=failure;
        const auto allocated=array_accounting::allocated,released_arrays=array_accounting::released;
        const auto observed=view(input);
        assert((failure && !observed.bytes) || (!failure && observed.bytes==sizeof(call.x)));
        const auto expected=failure?failure:3;
        assert(array_accounting::allocated-allocated==expected && array_accounting::released-released_arrays==expected
            && array_accounting::count()==alive);
    }
    fail_getter_at=0;call.xx.rank=3;assert(!view(input).bytes && array_accounting::count()==alive);call.xx.rank=2;
    getter_calls=0;throw_getter_at=2;bool caught=false;
    try{(void)view(input);}catch(const std::bad_alloc&){caught=true;}
    assert(caught && array_accounting::count()==alive);throw_getter_at=0;
}
void raw_metadata_keeps_rank_refusal_without_public_fill() {
    install();selected_mode=0;Invocation call;
    call.xx.storage_rank=1;call.xx.storage[0]=8;
    call.qq.storage_rank=1;call.qq.storage[0]=8;
    const auto arrays=array_accounting::count();
    assert(!call.prepare() && !call.execute());
    MetadataState state{};assert(dcat_dqg_metadata_state_v1(&state)==Ok);
    assert(state.total==1 && state.retained==1 && !state.overflow && state.snapshot_size==sizeof(MetadataView)
        && state.record_size==sizeof(MetadataRecord) && state.dimensions==8);
    MetadataRecord row{},by_parent{};assert(dcat_dqg_metadata_v1(1,&row)==Ok);
    assert(row.sequence==1 && row.public_entry==1 && row.parent_token==1 && row.capture_complete
        && row.thread==tid() && row.device==0 && row.ready);
    assert(dcat_dqg_metadata_parent_v1(row.parent_token,&by_parent)==Ok && by_parent.sequence==row.sequence);
    const auto& x=row.input;
    assert(x.stage==MetadataRank && x.called_mask==127 && !x.null_arrays && !x.truncated_arrays
        && x.counts[0]==2 && x.counts[1]==2 && x.counts[2]==1 && x.retained[2]==1
        && x.shape[0]==2 && x.shape[1]==4 && x.strides[0]==4 && x.strides[1]==1 && x.storage[0]==8
        && x.dtype==ACL_BF16 && x.format==ACL_FORMAT_ND && x.address==reinterpret_cast<uint64_t>(call.x.data())
        && !x.result.rows && !x.result.columns && !x.result.bytes && x.result.dimensions==2);
    assert(row.quant.stage==MetadataRank && row.scale.stage==MetadataComplete && row.scale.result.bytes==8);
    RawRecord raw{};assert(dcat_dqg_raw_v1(row.parent_token,&raw)==Ok && same_view(raw.parent.input,x.result,true,true));
    assert(real_prepares==1 && real_executes==1 && !engine.state().admitted_model_calls && !allocations_made
        && array_accounting::count()==arrays);
    const auto getters=getter_calls;
    assert(dcat_dqg_metadata_v1(1,&row)==Ok && getter_calls==getters); // No late descriptor reads.
}
void raw_metadata_rank_arrays_partial_failures_and_truncation() {
    install();Invocation call;MetadataView captured{};const auto alive=array_accounting::count();
    const auto* x=reinterpret_cast<const aclTensor*>(&call.xx);
    assert(view(x,&captured).bytes==16 && captured.stage==MetadataComplete && captured.counts[0]==2);
    assert(view(reinterpret_cast<const aclTensor*>(&call.ss),&captured).bytes==8 && captured.counts[0]==1);
    call.xx.rank=3;assert(!view(x,&captured).bytes && captured.stage==MetadataRank && captured.retained[0]==3);
    call.xx.rank=9;assert(!view(x,&captured).bytes && captured.stage==MetadataRank && captured.counts[0]==9
        && captured.retained[0]==8 && captured.truncated_arrays==7);
    call.xx.rank=2;call.xx.stride_rank=1;
    assert(!view(x,&captured).bytes && captured.stage==MetadataRank && captured.counts[1]==1);call.xx.stride_rank=UINT64_MAX;
    for(uint64_t failure:{1u,2u,3u}) {
        getter_calls=0;fail_getter_at=failure;
        assert(!view(x,&captured).bytes && captured.stage==MetadataOffset+failure
            && captured.getter_status[2+failure]==37 && captured.counts[failure-1]==2
            && array_accounting::count()==alive);
    }
    fail_getter_at=0;getter_calls=0;null_getter_at=2;
    assert(!view(x,&captured).bytes && captured.stage==MetadataRank && captured.null_arrays==2
        && captured.counts[1]==2 && captured.retained[1]==0 && array_accounting::count()==alive);null_getter_at=0;
    getter_calls=0;throw_getter_at=3;bool caught=false;
    try{(void)view(x,&captured);}catch(const std::bad_alloc&){caught=true;}
    assert(caught && captured.stage==MetadataException && captured.last_getter==MetadataStorage
        && captured.counts[2]==2 && array_accounting::count()==alive);throw_getter_at=0;
}
void raw_metadata_budget_retains_counts_without_native_mutation() {
    MetadataState state{};
    for(uint64_t index=0;index<MetadataCapacity+2;++index)(void)begin_metadata();
    assert(dcat_dqg_metadata_state_v1(&state)==Ok && state.total==MetadataCapacity+2
        && state.retained==MetadataCapacity && state.overflow==2 && journal.state().total==0);
    MetadataRecord row{};assert(dcat_dqg_metadata_v1(MetadataCapacity,&row)==Ok);
    assert(dcat_dqg_metadata_v1(MetadataCapacity+1,&row)==Observer && dcat_dqg_metadata_parent_v1(0,&row)==Observer);
}
void empty_owner_cancel_drains_and_removes_exact_callbacks() {
    install();selected_mode=0;Invocation call;call.success(71);
    assert(!engine.state().targets && !journal.state().pending && admission==observe);
    fail_sync_stream=71;
    assert(dcat_dqg_cancel_empty_v1(&configured)==Pending && engine.state().installed && engine.state().stopped);
    assert(dcat_dqg_remove_v1(&configured,observe,finish)==Pending && admission==observe);
    PublicInput input{};input.thread=tid();input.device=0;input.input=view(reinterpret_cast<const aclTensor*>(&call.xx));uint64_t token=0;
    assert(dcat_dqg_public_begin_v1(&configured,&input,&token)==Identity);
    fail_sync_stream=0;
    assert(dcat_dqg_cancel_empty_v1(&configured)==Ok && !engine.state().installed);
    assert(dcat_dqg_remove_v1(&configured,nullptr,finish)==Pending && admission==observe);
    assert(dcat_dqg_remove_v1(&configured,observe,finish)==Ok && !admission && !completed);
    assert(real_prepares==1 && real_executes==1 && allocations_made==0);
}
void empty_cancel_refuses_live_scope_pending_executor_and_nonempty_group() {
    install();selected_mode=0;Invocation call;call.begin();
    assert(dcat_dqg_cancel_empty_v1(&configured)==Identity && engine.state().installed);
    assert(!call.prepare() && journal.state().pending==1);
    assert(call.end(1)==Liveness);public_storage_live=false;
    assert(dcat_dqg_cancel_empty_v1(&configured)==Pending && engine.state().installed && admission==observe);
    assert(dcat_dqg_remove_v1(&configured,observe,finish)==Pending && !late_copies);
}
void empty_cancel_does_not_release_an_admitted_target() {
    install();selected_mode=Reference;Invocation call;call.success();
    const auto host=engine.state().host_bytes;
    assert(host && dcat_dqg_cancel_empty_v1(&configured)==Identity);
    assert(engine.state().installed && engine.state().host_bytes==host && admission==observe);
    stop_drain_recover();assert(dcat_dqg_remove_v1(&configured,observe,finish)==Ok);
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
    run("failed execute retains executor registration",execute_error_cannot_infer_executor_consumption_from_sync);
    run("exception at native start revokes stack",throw_during_start);
    run("exception at completion revokes stack",throw_during_completion);
    run("C control ABI catches exceptions",control_abi_catches_exceptions);
    run("SDK metadata ownership on all exits",sdk_metadata_arrays_release_on_all_view_exits);
    run("actual GWS raw storage rank without geometry fill",raw_metadata_keeps_rank_refusal_without_public_fill);
    run("raw metadata ranks errors nulls and bounded arrays",raw_metadata_rank_arrays_partial_failures_and_truncation);
    run("raw metadata bounded retention",raw_metadata_budget_retains_counts_without_native_mutation);
    run("empty owner actual raw stream drain and exact callback removal",empty_owner_cancel_drains_and_removes_exact_callbacks);
    run("empty cancel retains live scope and pending executor",empty_cancel_refuses_live_scope_pending_executor_and_nonempty_group);
    run("empty cancel cannot release admitted target",empty_cancel_does_not_release_an_admitted_target);
    std::cout<<"PASS 20 typed-shim CPU memory contracts; actual NPU calls=0; mocked Native functions; SDK status depends on build headers\n";
}
