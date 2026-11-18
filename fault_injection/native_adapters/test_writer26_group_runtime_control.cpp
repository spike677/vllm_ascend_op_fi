// Device-free control/ABI test against actual installed SDK declarations.
#include "writer26_group_runtime.cpp"
#include <cassert>
#include <iostream>
int main() {
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
