// Explicit host memory simulation of failure/recovery, never native qualification.
#include "writer26_group_core.h"
#include <cassert>
#include <cstdlib>
#include <iostream>
#include <set>
using namespace dcat26_group;
struct Memory final:Ops {
    std::set<void*> allocations;
    int sync_fail{},copy_fail{},free_fail{};
    bool sync(uint64_t) override {if(sync_fail){--sync_fail;return false;}return true;}
    bool allocate(void** output,size_t bytes) override {*output=std::malloc(bytes);if(*output)allocations.insert(*output);return *output;}
    bool release(void* pointer) override {
        if(free_fail){--free_fail;return false;}
        assert(allocations.erase(pointer)==1);std::free(pointer);return true;
    }
    bool copy(void* output,size_t capacity,const void* input,size_t bytes,CopyKind kind) override {
        if(copy_fail){--copy_fail;return false;}
        assert(capacity>=bytes && output && input);
        if(kind==CopyKind::DeviceToDevice)assert(!overlap(reinterpret_cast<uint64_t>(output),bytes,reinterpret_cast<uint64_t>(input),bytes));
        std::memcpy(output,input,bytes);return true;
    }
    ~Memory(){assert(allocations.empty());}
};
struct Fixture {
    Memory memory;Engine engine{memory};Config owner{};
    std::array<unsigned char,208> arena{};std::array<unsigned char,128> consumer_output{};
    uint64_t next{};
    explicit Fixture(uint32_t mechanism) {
        owner.mechanism=mechanism;owner.owner_hi=12;owner.owner_lo=34;owner.generation=1;owner.thread=55;
        owner.max_targets=4;owner.max_calls=64;owner.max_device_bytes=8192;owner.max_host_bytes=65536;
        assert(engine.open(owner)==Ok);arena.fill(3);
    }
    Call call() {
        Call x{};x.mechanism=owner.mechanism;x.device=0;x.dtype=27;x.format=2;x.thread=owner.thread;
        x.token=++next;x.independent_call=next*4;x.registration=1;x.cache=2;x.producer_sequence=next*2;
        x.residual=reinterpret_cast<uint64_t>(arena.data());x.residual_bytes=64;
        x.output=reinterpret_cast<uint64_t>(arena.data()+80);x.output_bytes=owner.mechanism==Qmm?128:64;
        x.dimensions=2;x.shape[0]=4;x.shape[1]=owner.mechanism==Qmm?16:8;return x;
    }
    Lease lease(const Call& call,uint32_t mode,Key key={1,2,3,4}) {
        Lease value{};value.mode=mode;value.owner_hi=owner.owner_hi;value.owner_lo=owner.owner_lo;
        value.generation=owner.generation;value.thread=owner.thread;value.target=key;value.expected=call;
        if(call.mechanism==Qmm){value.residual_allocation_end=call.output;value.output_allocation_begin=call.output;value.shift_bytes=48;}
        return value;
    }
    void writer(const Call& call) {
        assert(engine.start(call.token,123,456,789)==Ok);
        std::memset(reinterpret_cast<void*>(engine.replacement()),17,call.output_bytes); // One simulated original writer.
        assert(engine.writer_done(call.token,123,0,1)==Ok);
    }
    int consumer(const Call& call,uint64_t sequence=0,uint64_t stream=123) {
        return engine.consumer_done(sequence?sequence:call.producer_sequence+1,call.residual,call.residual_bytes,
            reinterpret_cast<uint64_t>(consumer_output.data()),consumer_output.size(),0,stream,0,1);
    }
    void phase(uint32_t mode,Key key={1,2,3,4}) {
        auto x=call();assert(engine.prepare(x,lease(x,mode,key))==Ok);writer(x);
        assert(engine.state().pending==1 && !engine.receipt(x.token).consumer_complete);
        assert(consumer(x)==Ok);
        auto receipt=engine.receipt(x.token);
        assert(receipt.writer_calls==1 && receipt.consumer_calls==1 && receipt.output_readback_exact
            && receipt.residual_restored_exact && receipt.padding_readback_exact);
        if(mode==Fault)assert(receipt.changed_residual_bytes==(owner.mechanism==Qmm?32:64));
        else assert(receipt.residual_unchanged);
        assert(engine.acknowledge(owner,x.token)==Ok);
        assert(!engine.state().pending && !engine.state().device_bytes);
        assert(std::all_of(arena.begin(),arena.begin()+80,[](unsigned char n){return n==3;}));
    }
    void qualify(Key key={1,2,3,4}) {phase(Reference,key);phase(Lossless,key);phase(Off,key);}
    void recover() {
        const auto token=engine.token();assert(engine.stop(owner)==Ok);assert(engine.drain(owner)==Ok);
        if(token){assert(engine.restore(owner,token)==Ok);assert(engine.acknowledge(owner,token)==Ok);}
        assert(engine.close(owner)==Ok);assert(!engine.state().host_bytes && !engine.state().device_bytes);
    }
};
int main() {
    for(uint32_t mechanism:{uint32_t(Qmm),uint32_t(Fia)}) {
        {Fixture f(mechanism);f.qualify();f.qualify({9,8,7,6});
         assert(f.engine.seal(f.owner,{{1,2,3,4}})!=Ok);
         assert(f.engine.seal(f.owner,{{1,2,3,4},{9,8,7,6}})==Ok);
         f.phase(Fault);f.phase(Fault,{9,8,7,6});f.phase(Off);f.phase(Off,{9,8,7,6});f.recover();}
        {Fixture f(mechanism);auto x=f.call(),other=f.call();
         auto stale=f.lease(x,Reference);stale.expected.residual+=2;
         assert(f.engine.prepare(x,stale)==Identity && !f.engine.state().pending);
         assert(f.engine.prepare(x,f.lease(x,Reference))==Ok);
         assert(f.engine.prepare(other,f.lease(other,Reference))==Busy); // Overlap refused, not a smaller denominator.
         f.writer(x);assert(f.consumer(x)==Ok);assert(f.engine.acknowledge(f.owner,x.token)==Ok);
         assert(f.engine.prepare(x,f.lease(x,Reference))==Identity);f.recover();}
        {Fixture f(mechanism);f.qualify();assert(f.engine.seal(f.owner,{{1,2,3,4}})==Ok);
         auto x=f.call();assert(f.engine.prepare(x,f.lease(x,Fault))==Ok);f.writer(x);
         assert(f.consumer(x,x.producer_sequence+2)==Consumer);
         assert(f.engine.state().pending && f.engine.state().host_bytes>0);
         assert(f.engine.acknowledge(f.owner,x.token)!=Ok);f.recover();
         assert(std::all_of(f.arena.begin(),f.arena.begin()+80,[](unsigned char n){return n==3;}));}
        {Fixture f(mechanism);f.qualify();assert(f.engine.seal(f.owner,{{1,2,3,4}})==Ok);
         auto x=f.call();assert(f.engine.prepare(x,f.lease(x,Fault))==Ok);f.writer(x);
         f.memory.copy_fail=1;assert(f.consumer(x)==Restore);
         assert(f.engine.state().pending && !f.engine.receipt(x.token).released);
         assert(f.engine.stop(f.owner)==Ok);assert(f.engine.restore(f.owner,x.token)==StateError);
         assert(f.engine.drain(f.owner)==Ok);f.memory.copy_fail=1;assert(f.engine.restore(f.owner,x.token)==Restore);
         assert(f.engine.state().pending);f.recover();}
        {Fixture f(mechanism);f.phase(Reference);auto x=f.call();
         assert(f.engine.prepare(x,f.lease(x,Lossless))==Ok);f.writer(x);assert(f.consumer(x)==Ok);
         f.memory.free_fail=1;assert(f.engine.acknowledge(f.owner,x.token)==Free);
         assert(f.engine.state().device_bytes==x.output_bytes && f.engine.state().pending);f.recover();}
        {Fixture f(mechanism);auto x=f.call();assert(f.engine.prepare(x,f.lease(x,Reference))==Ok);
         f.memory.sync_fail=1;assert(f.engine.start(x.token,123,456,789)==Sync);f.recover();}
        {Fixture f(mechanism);auto x=f.call();assert(f.engine.prepare(x,f.lease(x,Reference))==Ok);
         f.writer(x);assert(f.consumer(x,0,124)==Consumer);f.recover();}
        {Fixture f(mechanism);auto x=f.call();assert(f.engine.prepare(x,f.lease(x,Reference))==Ok);
         assert(f.engine.start(x.token,123,456,789)==Ok);std::memset(reinterpret_cast<void*>(f.engine.replacement()),88,x.output_bytes);
         assert(f.engine.writer_done(x.token,123,4,1)==Writer);f.recover();
         assert(std::all_of(f.arena.begin(),f.arena.end(),[](unsigned char n){return n==3;}));}
        {Fixture f(mechanism);auto x=f.call();assert(f.engine.prepare(x,f.lease(x,Reference))==Ok);
         assert(f.engine.start(x.token,123,456,789)==Ok);std::memset(reinterpret_cast<void*>(f.engine.replacement()),88,x.output_bytes);
         assert(f.engine.failed_writer_completion(x.token,123,0,1,Identity)==Identity);
         assert(!f.engine.state().executing && f.engine.state().pending);f.recover();
         assert(std::all_of(f.arena.begin(),f.arena.end(),[](unsigned char n){return n==3;}));}
        {Fixture f(mechanism);f.qualify();auto x=f.call();
         assert(f.engine.prepare(x,f.lease(x,Reference))==Qualification);f.recover();}
    }
    std::cout<<"{\"status\":\"PASS_CPU_MEMORY_SIMULATION\",\"actual_npu_calls\":0,\"mechanisms\":2,\"config_bytes\":"
        <<sizeof(Config)<<",\"call_bytes\":"<<sizeof(Call)<<",\"lease_bytes\":"<<sizeof(Lease)
        <<",\"receipt_bytes\":"<<sizeof(Receipt)<<",\"state_bytes\":"<<sizeof(State)<<"}\n";
}
