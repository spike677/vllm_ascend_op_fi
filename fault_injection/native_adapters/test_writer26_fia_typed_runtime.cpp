// Actual group shim and owned-array paths; host-memory SDK doubles only.
#include "writer26_group_runtime.cpp"
#include <cassert>
#include <iostream>
#include <set>
#include <stdexcept>
#include <sys/wait.h>
#include <vector>
namespace typed_fixture {
struct Tensor {
    void* address{};int32_t format{47};int64_t offset{};
    std::array<int64_t,3> shape{{2,2,4}},storage{{2,2,4}},stride{{8,4,1}};
};
std::vector<Event> events;
std::vector<dcat26_fia_typed::Record> typed;
std::set<Tensor*> owned;
uint64_t getter_calls{},fail_getter_at{},throw_getter_at{},created{},destroyed{},writer_calls{},consumer_calls{};
bool throw_admission{},bad_before_call{},prepared_registration{};
FiaSnapshotsV3 completed_snapshot{};
Tensor* tensor(const aclTensor* value){return reinterpret_cast<Tensor*>(const_cast<aclTensor*>(value));}
aclError device(int32_t* value){*value=0;return 0;}
aclError sync(aclrtStream){return 0;}
aclError copy(void* out,size_t capacity,const void* in,size_t bytes,aclrtMemcpyKind) {
    assert(out && in && capacity>=bytes);std::memmove(out,in,bytes);return 0;
}
aclnnStatus type(const aclTensor*,aclDataType* value){*value=ACL_BF16;return 0;}
aclnnStatus format(const aclTensor* value,aclFormat* out){*out=static_cast<aclFormat>(tensor(value)->format);return 0;}
aclnnStatus offset(const aclTensor* value,int64_t* out){*out=tensor(value)->offset;return 0;}
aclnnStatus address(const aclTensor* value,void** out){*out=tensor(value)->address;return 0;}
aclnnStatus array(const std::array<int64_t,3>& value,int64_t** out,uint64_t* rank) {
    ++getter_calls;*rank=3;*out=new int64_t[3];std::copy(value.begin(),value.end(),*out);
    if(getter_calls==throw_getter_at)throw std::runtime_error("GETTER_AFTER_ALLOCATION");
    return getter_calls==fail_getter_at?71:0;
}
aclnnStatus shape(const aclTensor* value,int64_t** out,uint64_t* rank){return array(tensor(value)->shape,out,rank);}
aclnnStatus storage(const aclTensor* value,int64_t** out,uint64_t* rank){return array(tensor(value)->storage,out,rank);}
aclnnStatus stride(const aclTensor* value,int64_t** out,uint64_t* rank){return array(tensor(value)->stride,out,rank);}
aclTensor* create(const int64_t* dims,uint64_t rank,aclDataType dtype,const int64_t* strides,int64_t off,
    aclFormat fmt,const int64_t* physical,uint64_t storage_rank,void* raw) {
    assert(rank==3 && storage_rank==3 && dtype==ACL_BF16 && static_cast<int>(fmt)==47 && !off);
    auto* out=new Tensor{};out->address=raw;out->format=fmt;
    std::copy(dims,dims+3,out->shape.begin());std::copy(physical,physical+3,out->storage.begin());std::copy(strides,strides+3,out->stride.begin());
    owned.insert(out);++created;return reinterpret_cast<aclTensor*>(out);
}
aclnnStatus destroy(const aclTensor* value){assert(owned.erase(tensor(value))==1);delete tensor(value);++destroyed;return 0;}
int state(TraceState* value){*value=TraceState{};value->flags=7;value->capacity=32768;value->total_events=value->stored_events=events.size();return 0;}
int read(uint64_t index,Event* out){if(index>=events.size())return 1;*out=events[index];return 0;}
int typed_state(dcat26_fia_typed::State* out) {
    *out=dcat26_fia_typed::State{};out->capacity=32768;out->total=out->stored=typed.size();
    for(const auto& row:typed) {
        if(row.phase==dcat26_fia_typed::OriginalBeforeGws)++out->before_gws;
        if(row.phase==dcat26_fia_typed::OriginalAndEffectiveAfterGws)++out->after_gws;
        if(row.phase==dcat26_fia_typed::OwnedEffectiveBeforeExecute)++out->before_execute;
    }
    if(!typed.empty()) {
        const auto phase=typed.back().phase;
        out->preparing=phase==dcat26_fia_typed::OriginalBeforeGws;
        out->prepared=prepared_registration;
        out->executing=prepared_registration && phase==dcat26_fia_typed::OwnedEffectiveBeforeExecute;
    }
    return 0;
}
int typed_read(uint64_t index,dcat26_fia_typed::Record* out){if(index>=typed.size())return 1;*out=typed[index];return 0;}
void append(Event value,uint32_t kind){value.kind=kind;value.sequence=events.size()+1;events.push_back(value);}
void append(dcat26_fia_typed::Record value){value.sequence=typed.size()+1;typed.push_back(value);}
int before_execute(const aclTensor* value,aclOpExecutor* executor) {
    assert(owned.count(tensor(value))==1);auto row=typed.back();
    row.phase=dcat26_fia_typed::OwnedEffectiveBeforeExecute;row.event_sequence=events.back().sequence;
    row.registration=events.back().registration;row.executor=reinterpret_cast<uint64_t>(executor);
    row.effective=capture_fia_descriptor(value);append(row);return 0;
}
int observer(const Call* call) {
    if(throw_admission)throw std::runtime_error("ADMISSION_FAILURE");
    FiaSnapshotsV3 snapshot{};assert(dcat26_group_fia_snapshots_v3(&owner,call->token,&snapshot)==Ok);
    FiaLeaseV3 proposal{};proposal.lease.mode=Reference;proposal.lease.owner_hi=owner.owner_hi;
    proposal.lease.owner_lo=owner.owner_lo;proposal.lease.generation=owner.generation;proposal.lease.thread=owner.thread;
    proposal.lease.target={{1,2,3,4}};proposal.lease.expected=*call;proposal.original_before=snapshot.before;
    if(bad_before_call)++proposal.original_before.call;
    return dcat26_group_continue_fia_v3(&proposal);
}
int completion(const Receipt* value) {
    ++consumer_calls;assert(!value->error && value->writer_complete && value->consumer_complete);
    assert(dcat26_group_fia_snapshots_v3(&owner,value->token,&completed_snapshot)==Ok);
    return dcat26_group_acknowledge_v2(&owner,value->token);
}
struct Case {
    std::array<unsigned char,32> output{},residual{},consumed{};
    Tensor original{};Event event{};const aclTensor* effective{};
    aclOpExecutor* executor=reinterpret_cast<aclOpExecutor*>(456);
    aclrtStream stream=reinterpret_cast<aclrtStream>(123);
    Case() {
        output.fill(1);residual.fill(3);original.address=output.data();
        owner=Config{};owner.mechanism=Fia;owner.owner_hi=101;owner.owner_lo=102;owner.generation=1;owner.thread=tid();
        owner.max_targets=4;owner.max_calls=16;owner.max_host_bytes=65536;owner.max_device_bytes=8192;
        assert(engine.open(owner)==Ok);admit=observer;complete=completion;
        api.device=device;api.sync=sync;api.copy=copy;api.type=type;api.format=format;api.offset=offset;api.address=address;
        api.shape=shape;api.storage=storage;api.strides=stride;api.create=create;api.destroy=destroy;
        api.trace_state=state;api.trace_read=read;api.fia_typed_state=typed_state;api.fia_typed_read=typed_read;api.fia_typed_execute=before_execute;
        event.flags=3;event.thread=tid();event.device=0;event.call=11;event.producer=9;event.dtype=ACL_BF16;event.format=47;
        event.output=reinterpret_cast<uint64_t>(output.data());event.output_bytes=32;
        event.residual=reinterpret_cast<uint64_t>(residual.data());event.residual_bytes=32;
        event.dimensions=3;event.residual_dimensions=2;
        std::copy(original.shape.begin(),original.shape.end(),event.shape);std::copy(original.stride.begin(),original.stride.end(),event.strides);
        event.residual_shape[0]=2;event.residual_shape[1]=8;event.residual_strides[0]=8;event.residual_strides[1]=1;
        append(event,dcat26_inventory::FiaBefore);
        dcat26_fia_typed::Record before{};before.phase=dcat26_fia_typed::OriginalBeforeGws;
        before.event_sequence=events.back().sequence;before.call=event.call;before.thread=tid();before.device=0;
        std::memcpy(before.layout,"TND",4);before.layout_length=3;
        before.original=capture_fia_descriptor(reinterpret_cast<const aclTensor*>(&original));before.effective=before.original;append(before);
    }
    void prepare() {effective=fault26_fia_writer_prepare_output(reinterpret_cast<const aclTensor*>(&original));}
    void gws() {
        auto next=event;next.executor=reinterpret_cast<uint64_t>(executor);next.output=engine.replacement();append(next,dcat26_inventory::FiaGws);
        auto row=typed.front();row.phase=dcat26_fia_typed::OriginalAndEffectiveAfterGws;row.event_sequence=events.back().sequence;
        row.executor=reinterpret_cast<uint64_t>(executor);row.original=capture_fia_descriptor(reinterpret_cast<const aclTensor*>(&original));
        row.effective=capture_fia_descriptor(effective);append(row);
        prepared_registration=true;
    }
    void complete_gws(){fault26_fia_writer_gws_complete(reinterpret_cast<const aclTensor*>(&original),effective,executor,0);}
    bool start() {
        auto next=event;next.executor=reinterpret_cast<uint64_t>(executor);next.registration=7;next.output=engine.replacement();
        append(next,dcat26_inventory::FiaEnter);return fault26_fia_writer_before_execute(executor,stream);
    }
    void execute() {
        assert(start());++writer_calls;std::memset(tensor(effective)->address,17,32);
        fault26_fia_writer_after_execute(executor,stream,0);
        prepared_registration=false; // Successful proxy end consumes its independent registration.
    }
    void consumer() {
        auto next=event;next.call=next.producer=event.producer+1;next.executor=789;next.output=reinterpret_cast<uint64_t>(consumed.data());
        append(next,dcat26_inventory::AddRmsExit);
        fault26_qmm_underflow_consumer_complete(0,next.call,residual.data(),32,consumed.data(),32,stream,0);
    }
    void retain() {
        assert(engine.state().installed && engine.state().pending && engine.state().host_bytes && !writer_calls && !consumer_calls);
        assert(dcat26_group_stop_v2(&owner)==Ok);
        assert(dcat26_group_drain_v2(&owner)!=Ok && dcat26_group_restore_v2(&owner,fia.call.token)!=Ok);
        assert(dcat26_group_remove_v2(&owner,observer,completion)!=Ok);
        assert(created==1 && destroyed==0 && owned.size()==1);
    }
};
void arrays_zero() {
    DcatMetadataArrayStateV1 arrays{};assert(dcat26_group_metadata_arrays_v1(&arrays)==0);
    assert(!arrays.active_captures && !arrays.active_getters && !arrays.live_arrays && !arrays.ownership_errors);
    assert(arrays.acquired_arrays==arrays.released_arrays);
}
void success() {
    Case test;test.prepare();assert(fia.typed && created==1 && test.effective!=nullptr);
    assert(test.effective!=reinterpret_cast<const aclTensor*>(&test.original));
    assert(tensor(test.effective)->address==test.output.data() && tensor(test.effective)->format==47);
    test.gws();test.complete_gws();test.execute();test.consumer();
    assert(writer_calls==1 && consumer_calls==1 && created==destroyed && owned.empty());
    assert(completed_snapshot.created && completed_snapshot.released && completed_snapshot.native_return_success);
    assert(completed_snapshot.before.original.format==47 && completed_snapshot.after.original.format==47
        && completed_snapshot.before_execute.effective.format==47);
    assert(dcat26_group_stop_v2(&owner)==Ok && dcat26_group_drain_v2(&owner)==Ok);
    assert(dcat26_group_remove_v2(&owner,observer,completion)==Ok);
    assert(!engine.state().pending && !engine.state().host_bytes && !engine.state().device_bytes);arrays_zero();
}
void changed_after(unsigned field) {
    Case test;test.prepare();test.gws();auto& value=typed.back().original;
    switch(field) {
        case 0:++value.storage[0];break;case 1:--value.storage_rank;break;case 2:++value.offset;break;
        case 3:value.format=2;break;case 4:++value.strides[0];break;case 5:++value.descriptor;break;
        case 6:++value.raw;break;default:++typed.back().call;break;
    }
    test.complete_gws();assert(!test.start());test.retain();arrays_zero();
}
void changed_before_execute() {
    Case test;test.prepare();test.gws();test.complete_gws();tensor(test.effective)->offset=1;
    assert(!test.start());test.retain();arrays_zero();
}
void gws_binding_failure_keeps_unclassified_executor() {
    Case test;test.prepare();test.gws();
    fault26_fia_writer_gws_complete(nullptr,test.effective,test.executor,0);
    assert(!fia.executor && prepared_registration && !test.start());
    test.retain();arrays_zero();
}
void getter_fail(bool exception) {
    Case test;test.prepare();test.gws();test.complete_gws();
    if(exception)throw_getter_at=getter_calls+2;else fail_getter_at=getter_calls+2;
    assert(!test.start());test.retain();arrays_zero();
    if(exception)assert(fia_abi_exceptions.load());
}
void callback_throw() {
    Case test;throw_admission=true;test.prepare();assert(!runtime.callback_active && fia_abi_exceptions.load());
    assert(engine.state().installed && !engine.state().pending && !created && !writer_calls);arrays_zero();
}
void wrong_call() {
    Case test;bad_before_call=true;test.prepare();assert(!engine.state().pending && !created && !writer_calls);arrays_zero();
}
void nd_and_qmm_do_not_gain_ncl() {
    Case test;const auto call=make_call(test.event,Fia);Lease lease{};lease.owner_hi=owner.owner_hi;lease.owner_lo=owner.owner_lo;
    lease.generation=1;lease.thread=owner.thread;lease.target={{1,2,3,4}};lease.mode=Reference;lease.expected=call;
    assert(engine.prepare(call,lease)==Geometry);auto qmm=call;qmm.mechanism=Qmm;lease.expected=qmm;
    assert(engine.prepare_fia_ncl_v3(qmm,lease,typed.front().original)==Geometry);
    auto changed=typed.front().original;changed.format=2;lease.expected=call;
    assert(engine.prepare_fia_ncl_v3(call,lease,changed)==Geometry);assert(!engine.state().pending);arrays_zero();
}
void run(void (*test)()) {
    const auto child=fork();assert(child>=0);if(child==0){test();_exit(0);}
    int result=0;assert(waitpid(child,&result,0)==child && WIFEXITED(result) && WEXITSTATUS(result)==0);
}
}
int main() {
    using namespace typed_fixture;
    run(success);run(changed_before_execute);run(callback_throw);run(wrong_call);run(nd_and_qmm_do_not_gain_ncl);
    run(gws_binding_failure_keeps_unclassified_executor);
    run([]{getter_fail(false);});run([]{getter_fail(true);});
    run([]{changed_after(0);});run([]{changed_after(1);});run([]{changed_after(2);});run([]{changed_after(3);});
    run([]{changed_after(4);});run([]{changed_after(5);});run([]{changed_after(6);});run([]{changed_after(7);});
    std::cout<<"{\"status\":\"PASS_CPU_TYPED_FIA_RUNTIME\",\"actual_npu_calls\":0,\"cases\":16}\n";
}
