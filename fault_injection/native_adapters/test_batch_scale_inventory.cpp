// Host-only contract, no CANN dependency or device calls.
#include "batch_scale_inventory.h"
#include <cassert>
#include <iostream>
using namespace dcat_dq_inventory;
Inventory<32768> production_capacity;
int main() {
    Inventory<8> inventory;
    Event event{};event.thread=10;event.input=123;event.quant=456;event.scale=789;
    event.rows=8;event.columns=2048;event.flags=31;
    auto first=inventory.begin(event);
    inventory.prepared(first,900,0,1);
    assert(inventory.state().pending==1);
    auto running=inventory.enter(900,10);
    assert(running.call==first.call && inventory.state().executing==1);
    inventory.leave(running,0,1);
    auto second=inventory.begin(event); // Executor address reused, different registration lifetime.
    inventory.prepared(second,900,0,1);
    running=inventory.enter(900,10);inventory.leave(running,0,1);
    assert(first.call!=second.call);
    Event output{};assert(inventory.read(7,&output) && output.call==second.call && output.original_calls==1);
    auto orphan=inventory.enter(900,10);inventory.leave(orphan,7,1);
    auto state=inventory.state();
    assert(state.capacity==8 && state.total_events==10 && state.stored_events==8 && state.overflow_events==2);
    assert(state.prepare_enter==2 && state.prepare_exit==2 && state.execute_enter==3 && state.execute_exit==3);
    assert(state.metadata_complete==2 && state.registrations==2 && state.orphan_executions==1);
    assert(!state.errors && !state.pending && !state.executing && !inventory.read(8,&output));
    Inventory<20> collision;
    auto a=collision.begin(event),b=collision.begin(event);
    collision.prepared(a,10,0,1);collision.prepared(b,10,0,1);
    assert(collision.state().errors==1 && collision.state().pending==1);
    auto foreign=collision.enter(10,11);assert(!foreign.call);
    collision.leave(foreign,7,0);
    assert(collision.state().pending==1 && collision.state().orphan_executions==1);
    running=collision.enter(10,10);collision.leave(running,0,1);
    assert(!collision.state().pending && !collision.state().executing);
    auto failed=collision.begin(event);collision.prepared(failed,0,1,1);
    assert(!collision.state().pending);
    production_capacity.begin(event);
    assert(production_capacity.state().capacity==32768 && production_capacity.read(0,&output));
    assert(output.version==1 && output.size==sizeof(Event) && output.device==-1 && output.input_dtype==-1);
    std::cout << "{\"status\":\"PASS_CPU_ONLY\",\"actual_npu_calls\":0,\"event_bytes\":"
              << sizeof(Event) << ",\"state_bytes\":" << sizeof(State) << "}\n";
}
