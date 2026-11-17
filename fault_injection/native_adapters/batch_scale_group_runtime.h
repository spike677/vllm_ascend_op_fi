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
int dcat_dqg_install_v1(const dcat_dq_group::Config*,dcat_dq_group::Admission,dcat_dq_group::Completed);
int dcat_dqg_public_begin_v1(const dcat_dq_group::Config*,const dcat_dq_group::PublicInput*,uint64_t*);
int dcat_dqg_public_end_v1(const dcat_dq_group::Config*,uint64_t,const dcat_dq_group::View*,const dcat_dq_group::View*,int,dcat_dq_group::Completion*);
int dcat_dqg_pass_v1(const dcat_dq_group::Config*,const dcat_dq_group::Call*);
int dcat_dqg_continue_v1(const dcat_dq_group::Lease*);
int dcat_dqg_seal_v1(const dcat_dq_group::Config*,const dcat_dq_group::Key*,uint64_t,uint64_t,const dcat_dq_group::Key*);
int dcat_dqg_stop_v1(const dcat_dq_group::Config*,const dcat_dq_group::Key*);
int dcat_dqg_drain_v1(const dcat_dq_group::Config*,const dcat_dq_group::Key*);
int dcat_dqg_recover_v1(const dcat_dq_group::Config*);
int dcat_dqg_remove_v1(const dcat_dq_group::Config*,dcat_dq_group::Admission,dcat_dq_group::Completed);
}
#endif
