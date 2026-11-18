// Actual v2 shim callbacks with explicit host-memory SDK doubles, no NPU calls.
#include "writer26_group_runtime.cpp"
#include <cassert>
#include <iostream>
#include <vector>

namespace fixture {
std::vector<Event> events;
std::vector<uint64_t> observed;
std::vector<Receipt> completed;
bool select_second{};
aclError device(int32_t* value) {*value=0;return ACL_SUCCESS;}
aclError sync(aclrtStream) {return ACL_SUCCESS;}
aclError copy(void* out,size_t capacity,const void* in,size_t bytes,aclrtMemcpyKind) {
    assert(out && in && capacity>=bytes);std::memcpy(out,in,bytes);return ACL_SUCCESS;
}
int state(TraceState* value) {
    *value=TraceState{};value->flags=7;value->capacity=32768;
    value->total_events=value->stored_events=events.size();return 0;
}
int read(uint64_t index,Event* value) {
    if(index>=events.size())return -1;
    *value=events[index];return 0;
}
void append(Event value,uint32_t kind) {
    value.kind=kind;value.sequence=events.size()+1;events.push_back(value);
}
int observer(const Call* call) {
    observed.push_back(call->independent_call);
    if(call->independent_call!=1821 && !select_second)return dcat26_group_pass_v2(&owner,call);
    Lease lease{};lease.mode=Reference;lease.owner_hi=owner.owner_hi;lease.owner_lo=owner.owner_lo;
    lease.generation=owner.generation;lease.thread=owner.thread;lease.target={call->independent_call,2,3,4};
    lease.expected=*call;lease.output_allocation_begin=call->output;
    lease.residual_allocation_end=call->output;lease.shift_bytes=48;
    return dcat26_group_continue_v2(&lease);
}
int completion(const Receipt* receipt) {
    completed.push_back(*receipt);
    if(receipt->error || !receipt->writer_complete || !receipt->consumer_complete
        || !receipt->residual_restored_exact)return Consumer;
    return dcat26_group_acknowledge_v2(&owner,receipt->token);
}
struct Case {
    std::array<unsigned char,208> arena{},other{};
    std::array<unsigned char,128> consumed{};
    aclrtStream stream=reinterpret_cast<aclrtStream>(123);
    Event event{};
    Fault26QmmUnderflowContext context{};
    Fault26QmmUnderflowDecision selected{};
    explicit Case(uint64_t identity) {
        events.clear();observed.clear();completed.clear();select_second=false;
        owner=Config{};owner.mechanism=Qmm;owner.owner_hi=identity;owner.owner_lo=5;
        owner.generation=1;owner.thread=tid();owner.max_targets=4;owner.max_calls=16;
        owner.max_host_bytes=65536;owner.max_device_bytes=8192;
        assert(engine.open(owner)==Ok);runtime=RuntimeState{};admit=observer;complete=completion;
        api.device=device;api.sync=sync;api.copy=copy;api.trace_state=state;api.trace_read=read;
        arena.fill(3);other.fill(4);
        event.flags=7;event.device=0;event.thread=tid();event.call=1821;
        event.registration=1;event.cache=41;event.producer=249;
        event.output=reinterpret_cast<uint64_t>(arena.data()+80);event.output_bytes=128;
        event.residual=reinterpret_cast<uint64_t>(arena.data());event.residual_bytes=64;
        event.dtype=ACL_BF16;event.format=ACL_FORMAT_ND;event.dimensions=event.residual_dimensions=2;
        event.shape[0]=4;event.shape[1]=16;event.strides[0]=16;event.strides[1]=1;
        event.residual_shape[0]=4;event.residual_shape[1]=8;
        event.residual_strides[0]=8;event.residual_strides[1]=1;
        context.abi_version=FAULT26_QMM_UNDERFLOW_ABI_VERSION;context.device=0;
        context.producer_sequence=249;context.cache=reinterpret_cast<void*>(41);
        context.original_output=arena.data()+80;context.output_bytes=128;context.output_m=4;context.output_n=16;
        context.residual=arena.data();context.residual_bytes=64;context.residual_m=4;context.residual_k=8;
    }
    void start_writer() {
        append(event,dcat26_inventory::QmmEnter);
        assert(fault26_qmm_underflow_prepare(&context,stream,&selected));
        assert(selected.claimed && selected.replacement_output==context.original_output);
        std::memset(selected.replacement_output,17,context.output_bytes); // One original reference writer.
    }
    void finish_writer() {
        append(event,dcat26_inventory::QmmExit);
        fault26_qmm_underflow_complete(&context,&selected,stream,0);
        assert(engine.receipt(selected.token).writer_complete && !engine.state().executing);
    }
    void other_writer() {
        auto next=event;next.call=1823;next.cache=42;
        next.output=reinterpret_cast<uint64_t>(other.data()+80);
        auto next_context=context;next_context.cache=reinterpret_cast<void*>(42);
        next_context.original_output=other.data()+80;
        append(next,dcat26_inventory::QmmEnter);
        Fault26QmmUnderflowDecision decision{};
        assert(!fault26_qmm_underflow_prepare(&next_context,stream,&decision));
        assert(!decision.claimed && !decision.replacement_output);
        std::memset(next_context.original_output,29,next_context.output_bytes); // Original non-target exactly once.
        append(next,dcat26_inventory::QmmExit);
    }
    void consumer() {
        auto row=event;row.flags=3;row.call=row.producer=250;row.executor=77;
        row.output=reinterpret_cast<uint64_t>(consumed.data());row.output_bytes=consumed.size();
        append(row,dcat26_inventory::AddRmsExit);
        fault26_qmm_underflow_consumer_complete(0,250,arena.data(),64,consumed.data(),consumed.size(),stream,0);
    }
    void close() {
        assert(dcat26_group_stop_v2(&owner)==Ok);assert(dcat26_group_drain_v2(&owner)==Ok);
        if(engine.token()) {
            const auto token=engine.token();
            assert(dcat26_group_restore_v2(&owner,token)==Ok);
            assert(dcat26_group_acknowledge_v2(&owner,token)==Ok);
        }
        assert(dcat26_group_remove_v2(&owner,observer,completion)==Ok);
        assert(!engine.state().pending && !engine.state().host_bytes && !engine.state().device_bytes);
    }
};
}

int main() {
    using namespace fixture;
    {Case test(101);test.start_writer();test.finish_writer();test.other_writer();
     assert((observed==std::vector<uint64_t>{1821,1823}));
     assert(runtime.callback_calls==2 && runtime.passthrough_calls==1 && runtime.admitted_calls==1);
     assert(!runtime.rejected_calls && engine.state().pending && !engine.state().error);
     assert(!engine.receipt(test.selected.token).error);
     test.consumer();assert(completed.size()==1 && !completed[0].error && completed[0].consumer_complete);
     assert(!engine.state().pending && runtime.completion_calls==1);
     assert(std::all_of(test.arena.begin(),test.arena.begin()+80,[](unsigned char n){return n==3;}));test.close();}
    {Case test(102);test.start_writer();test.finish_writer();select_second=true;test.other_writer();
     assert(observed.size()==2 && runtime.admitted_calls==1 && engine.state().pending);
     assert(engine.state().error && runtime.rejected_calls);test.close();} // A second selected lease still fails.
    {Case test(103);test.start_writer();test.other_writer();
     assert(observed.size()==1 && engine.state().executing && engine.state().error==Busy);
     engine.failed_writer_completion(test.selected.token,123,0,1,Busy);test.close();} // True overlap remains refused.
    std::cout<<"{\"status\":\"PASS_CPU_PENDING_CONSUMER_RUNTIME\",\"actual_npu_calls\":0,\"cases\":3}\n";
}
