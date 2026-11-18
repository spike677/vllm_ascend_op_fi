// Device-free control/ABI test against actual installed SDK declarations.
#include "writer26_group_runtime.cpp"
#include <cassert>
#include <iostream>
int main() {
    uint64_t address=999;
    assert(dcat26_group_cached_address_v1(0,nullptr)==1);
    assert(dcat26_group_cached_address_v1(0,&address)==3 && address==0);
    // Explicit CPU function holder; the getter must never call it or resolve.
    auto cpu_device=[](int32_t*)->aclError {assert(false);return ACL_SUCCESS;};
    Config cpu_owner{};cpu_owner.mechanism=Qmm;cpu_owner.max_targets=1;
    cpu_owner.owner_hi=1;cpu_owner.generation=1;cpu_owner.thread=tid();
    cpu_owner.max_calls=1;cpu_owner.max_host_bytes=1024;
    assert(engine.open(cpu_owner)==Ok);
    api.device=cpu_device;api.ready=true;owner=cpu_owner;
    assert(dcat26_group_cached_address_v1(0,&address)==0
        && address==reinterpret_cast<uintptr_t>(api.device));
    assert(dcat26_group_cached_address_v1(1,&address)==3 && address==0);
    assert(dcat26_group_cached_address_v1(64,&address)==2 && address==0);
    runtime.callback_active=1;
    assert(dcat26_group_cached_address_v1(0,&address)==3 && address==0);
    runtime.callback_active=0;owner.thread=0;
    assert(dcat26_group_cached_address_v1(0,&address)==3 && address==0);
    assert(engine.stop(cpu_owner)==Ok && engine.close(cpu_owner)==Ok);
    owner=cpu_owner;
    assert(dcat26_group_cached_address_v1(0,&address)==3 && address==0);
    api={};owner={};
    RuntimeState state{};
    assert(dcat26_group_state_v2(&state)==Ok);
    assert(state.version==2 && state.size==sizeof(RuntimeState));
    assert(!state.core.installed && !state.core.pending && !state.core.device_bytes && !state.core.host_bytes);
    assert(!state.callback_active && !state.completion_active && !state.descriptor && !state.prepared_fia);
    RuntimeState wrong{};wrong.size=0;
    assert(dcat26_group_state_v2(&wrong)==Identity);
    assert(dcat26_group_install_v2(nullptr,nullptr,nullptr)==Identity);
    assert(dcat26_group_continue_v2(nullptr)==Identity);
    assert(dcat26_group_pass_v2(nullptr,nullptr)==Identity);
    assert(dcat26_group_stop_v2(nullptr)==Identity);
    assert(dcat26_group_drain_v2(nullptr)==Identity);
    assert(dcat26_group_restore_v2(nullptr,1)==Identity);
    assert(dcat26_group_acknowledge_v2(nullptr,1)==Identity);
    assert(dcat26_group_remove_v2(nullptr,nullptr,nullptr)==Identity);
    static_assert(ACL_FLOAT16==1 && ACL_BF16==27 && ACL_FORMAT_ND==2);
    static_assert(sizeof(Config)==72 && sizeof(Call)==144 && sizeof(Lease)==248);
    static_assert(sizeof(Receipt)==296 && sizeof(State)==136 && sizeof(RuntimeState)==240);
    FiaAbiV3 typed{};assert(dcat26_group_fia_abi_v3(&typed)==Ok);
    static_assert(sizeof(dcat26_fia_typed::Descriptor)==280 && sizeof(dcat26_fia_typed::Record)==656);
    static_assert(sizeof(dcat26_fia_typed::State)==104 && sizeof(FiaLeaseV3)==912 && sizeof(FiaSnapshotsV3)==2048);
    assert(dcat26_group_continue_fia_v3(nullptr)==Identity);
    assert(dcat26_group_fia_snapshots_v3(nullptr,1,nullptr)==Identity);
    std::cout<<"{\"status\":\"PASS_SDK_ABI_CONTROL_ONLY\",\"actual_npu_calls\":0,\"runtime_state_bytes\":"
        <<sizeof(RuntimeState)<<",\"fia_typed_abi\":{\"version\":3,\"inventory_version\":2,\"descriptor\":"<<typed.descriptor
        <<",\"record\":"<<typed.record<<",\"state\":"<<typed.state<<",\"lease\":"<<typed.lease
        <<",\"snapshots\":"<<typed.snapshots<<"}}\n";
}
