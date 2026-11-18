#ifndef DCAT_DQ_GROUP_RUNTIME_V1_H
#define DCAT_DQ_GROUP_RUNTIME_V1_H
#include "batch_scale_group_core.h"
namespace dcat_dq_group {
struct PublicInput {
    uint32_t version{1},size{sizeof(PublicInput)};
    uint64_t thread{};
    int32_t device{-1},reserved{};
    View input{};
};
struct Completion {
    uint32_t version{1},size{sizeof(Completion)};
    Call call{};Receipt core{};
    uint64_t admitted{},prepare_enter{},prepare_exit{},execute_enter{},execute_exit{};
    int32_t status{},reserved{};
};
struct RuntimeState {
    uint32_t version{1},size{sizeof(RuntimeState)};
    State core{};JournalState census{};
    uint64_t last_sequence{},public_active{},public_token{},native_stack_active{},callback_active{},completion_active{},
        quarantined{},overlap_entries{},public_gws_entries{},public_execute_entries{},passthrough_calls{},
        admitted_calls{},completed_calls{},rejected_calls{},unreturned_public_scope{},orphan_descriptors{},abi_exceptions{};
    int32_t last_error{},reserved{};
};
struct RawRecord {
    uint32_t version{1},size{sizeof(RawRecord)},kind{},reserved{}; // 1 model parent, 2 actual Native event
    Call parent{};NativeEvent native{};ChildBinding binding{};
};
// Diagnostic snapshots only: captured once from the actual public GWS
// descriptors. They never supply geometry or authority to View/Engine.
constexpr uint64_t MetadataDimensions=8,MetadataCapacity=8192;
enum MetadataStage:uint32_t {
    MetadataNotAttempted=0,MetadataUnavailable=1,MetadataType=2,MetadataFormat=3,MetadataOffset=4,
    MetadataShape=5,MetadataStrides=6,MetadataStorage=7,MetadataAddress=8,MetadataRank=9,
    MetadataStorageMismatch=10,MetadataExtent=11,MetadataComplete=12,MetadataException=13
};
struct MetadataView {
    uint64_t descriptor{},address{};
    int64_t offset{};
    int32_t dtype{-1},format{-1};
    uint32_t stage{},last_getter{};
    uint64_t called_mask{},null_arrays{},truncated_arrays{};
    int32_t getter_status[7]{};uint32_t reserved{};
    uint64_t counts[3]{},retained[3]{};
    int64_t shape[MetadataDimensions]{},strides[MetadataDimensions]{},storage[MetadataDimensions]{};
    View result{};
};
struct MetadataRecord {
    uint32_t version{1},size{sizeof(MetadataRecord)};
    uint64_t sequence{},public_entry{},parent_token{},thread{};
    int32_t device{-1},ready{};
    uint64_t capture_complete{};
    MetadataView input{},quant{},scale{};
};
struct MetadataState {
    uint32_t version{1},size{sizeof(MetadataState)};
    uint64_t snapshot_size{sizeof(MetadataView)},record_size{sizeof(MetadataRecord)},dimensions{MetadataDimensions},
        capacity{MetadataCapacity},total{},retained{},overflow{},storage_bytes{MetadataCapacity*sizeof(MetadataRecord)};
};
struct AbiSizes {
    uint32_t version{1},size{sizeof(AbiSizes)};
    uint64_t view{},call{},config{},lease{},event{},binding{},journal{},plan{},child{},receipt{},state{},
        public_input{},completion{},runtime{},raw{};
};
using Admission=int (*)(const Call*);
using Completed=int (*)(const Completion*);
}
extern "C" {
int dcat_dqg_abi_v1(dcat_dq_group::AbiSizes*);
int dcat_dqg_state_v1(dcat_dq_group::RuntimeState*);
int dcat_dqg_raw_v1(uint64_t,dcat_dq_group::RawRecord*);
int dcat_dqg_metadata_state_v1(dcat_dq_group::MetadataState*);
int dcat_dqg_metadata_v1(uint64_t,dcat_dq_group::MetadataRecord*);
int dcat_dqg_metadata_parent_v1(uint64_t,dcat_dq_group::MetadataRecord*);
int dcat_dqg_install_v1(const dcat_dq_group::Config*,dcat_dq_group::Admission,dcat_dq_group::Completed);
int dcat_dqg_public_begin_v1(const dcat_dq_group::Config*,const dcat_dq_group::PublicInput*,uint64_t*);
int dcat_dqg_public_end_v1(const dcat_dq_group::Config*,uint64_t,const dcat_dq_group::View*,const dcat_dq_group::View*,int,dcat_dq_group::Completion*);
int dcat_dqg_pass_v1(const dcat_dq_group::Config*,const dcat_dq_group::Call*);
int dcat_dqg_continue_v1(const dcat_dq_group::Lease*);
int dcat_dqg_seal_v1(const dcat_dq_group::Config*,const dcat_dq_group::Key*,uint64_t,uint64_t,const dcat_dq_group::Key*);
int dcat_dqg_stop_v1(const dcat_dq_group::Config*,const dcat_dq_group::Key*);
int dcat_dqg_drain_v1(const dcat_dq_group::Config*,const dcat_dq_group::Key*);
int dcat_dqg_cancel_empty_v1(const dcat_dq_group::Config*);
int dcat_dqg_recover_v1(const dcat_dq_group::Config*);
int dcat_dqg_remove_v1(const dcat_dq_group::Config*,dcat_dq_group::Admission,dcat_dq_group::Completed);
}
#endif
