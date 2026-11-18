#ifndef DCAT26_FIA_TYPED_ADMISSION_V3_H
#define DCAT26_FIA_TYPED_ADMISSION_V3_H
#include "writer26_group_core.h"
namespace dcat26_group {
struct FiaLeaseV3 {
    uint32_t version{3},size{sizeof(FiaLeaseV3)};
    Lease lease{};
    dcat26_fia_typed::Record original_before{};
};
struct FiaSnapshotsV3 {
    uint32_t version{3},size{sizeof(FiaSnapshotsV3)};
    uint64_t token{},owner_hi{},owner_lo{},generation{},descriptor{},created{},released{},native_return_success{},exceptions{};
    dcat26_fia_typed::Record before{},after{},before_execute{};
};
struct FiaAbiV3 {
    uint32_t version{3},size{sizeof(FiaAbiV3)};
    uint64_t descriptor{},record{},state{},lease{},snapshots{};
};
}
#endif
