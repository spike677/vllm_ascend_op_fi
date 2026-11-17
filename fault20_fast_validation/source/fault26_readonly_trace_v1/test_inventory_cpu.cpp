// Host-only counter/order/overflow checks. No CANN headers, device or callback.
#include "fault26_inventory.h"
#include <cassert>
#include <iostream>
#include <thread>
// Match the production global capacity; compilation must not unroll thousands
// of nonzero Event default constructors into the load-time initializer.
dcat26_inventory::Inventory<32768> production_capacity;
int main() {
    using namespace dcat26_inventory;
    Inventory<5> inventory;
    inventory.run(false);inventory.run(true);
    Event writer{};writer.kind=QmmEnter;writer.cache=123;writer.registration=1;
    const auto call=inventory.append(writer);
    writer.kind=QmmExit;writer.call=call;inventory.append(writer);
    Event reused{};reused.kind=QmmRegistration;reused.cache=123;reused.registration=2;
    inventory.append(reused);
    Event consumer{};consumer.kind=AddRmsExit;consumer.call=8;consumer.producer=8;
    inventory.append(consumer);
    Event record{};
    assert(!production_capacity.read(0,&record));
    production_capacity.append(writer);
    assert(production_capacity.state().capacity==32768);
    assert(production_capacity.read(0,&record));
    assert(record.version==1 && record.size==sizeof(Event) && record.device==-1
        && record.dtype==-1 && record.format==-1 && record.shape[7]==0 && record.residual_strides[7]==0);
    assert(inventory.read(0,&record) && record.call==call && record.registration==1);
    assert(inventory.read(2,&record) && record.cache==123 && record.registration==2);
    std::thread other([&] {for(int i=0;i<3;++i){inventory.run(true);inventory.append(writer);}});
    other.join();inventory.error();
    const auto state=inventory.state();
    assert(state.version==1 && state.size==sizeof(State));
    assert(state.total_events==7 && state.stored_events==5 && state.overflow_events==2);
    assert(state.all_runs==5 && state.qmm_runs==4 && state.non_target_runs==1);
    assert(state.qmm_enter==1 && state.qmm_exit==4 && state.add_rms_exit==1);
    assert(state.qmm_registration==1 && state.errors==1);
    assert(!inventory.read(5,&record) && !inventory.read(0,nullptr));
    std::cout << "{\"status\":\"PASS_CPU_ONLY\",\"actual_npu_calls\":0,\"event_bytes\":"
              << sizeof(Event) << ",\"state_bytes\":" << sizeof(State) << "}\n";
}
