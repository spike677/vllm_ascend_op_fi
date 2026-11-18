#ifndef DCAT_QUANT_GMM2_TRACE_V1_H
#define DCAT_QUANT_GMM2_TRACE_V1_H
#include "quant_container_trace.h"

namespace dcat_qg2 {
constexpr uint64_t MaxCalls=128, MaxPairs=32, MaxBytes=1024*1024;
enum Operation:uint32_t {Prepare=1, Execute=2};
enum Error:uint64_t {None=0, Owner=1, Scope=2, Thread=3, Device=4, Budget=5,
    Container=6, Getter=7, Executor=8, Workspace=9, Original=10, Exception=11, Overlap=12};
struct Limits {
    uint32_t version{1},size{sizeof(Limits)};
    uint64_t max_calls{MaxCalls},max_pairs{MaxPairs},max_bytes{MaxBytes};
};
struct Slot {
    uint32_t formal_index{},kind{};
    uint64_t handle{},registration{},creator_entry{},creator_return{},count{},payload_offset{},payload_bytes{},
        actual_count{},joined{},live_at_return{};
    int32_t size_status{};uint32_t reserved{};
};
struct Event {
    uint32_t version{1},size{sizeof(Event)},operation{},phase{};
    uint64_t sequence{},call{},prepare_registration{},owner_hi{},owner_lo{},generation{},
        scope_token{},forward{},operator_sequence{},boundary_sequence{},stage{},pid{},thread{},
        function{},function_base{},function_offset{},caller{},caller_base{},caller_offset{},
        original_calls{},container_events{},monotonic_ns{},error{};
    int32_t device{-1},device_status{},status{};uint32_t reserved{};
    uint64_t formals[23]{};
    int64_t scalars[5]{}; // splitItem, groupType, groupListType, actType, quantGroupSize
    Slot containers[15]{}; // named GWS indices, never inferred from creator order
    dcat_qct::Tensor group_list{};
    uint64_t workspace{},workspace_bytes{},executor{},stream{};
};
struct State {
    uint32_t version{1},size{sizeof(State)};
    uint64_t installed{},stopped{},retired{},owner_hi{},owner_lo{},generation{},thread{},
        total_calls{},total_events{},retained_events{},active_calls{},pending_executors{},
        untracked_executors{},overflow{},first_error{},abi_exceptions{},outside_scope_calls{},
        foreign_thread_calls{},getter_failures{},record_capacity_bytes{},elapsed_ns{},resolver_ready{},
        prepares{},prepare_returns{},executes{},execute_returns{},original_prepares{},original_executes{},
        paired_execute_returns{},failed_with_executor{},prepare_function{},execute_function{},
        device_function{},list_size_function{},int_size_function{};
};
struct Sizes {
    uint32_t version{1},size{sizeof(Sizes)};
    uint64_t limits{sizeof(Limits)},slot{sizeof(Slot)},event{sizeof(Event)},state{sizeof(State)};
};
}
extern "C" {
uint32_t dcat_qct_gmm2_marker_v1();
int dcat_qct_gmm2_resolve_v1(); // startup only: lookup/identity, no device or operator call
int dcat_qct_gmm2_sizes_v1(dcat_qg2::Sizes*);
int dcat_qct_gmm2_begin_v1(const dcat_qct::Config*,const dcat_qg2::Limits*);
int dcat_qct_gmm2_state_v1(dcat_qg2::State*);
int dcat_qct_gmm2_event_v1(uint64_t,dcat_qg2::Event*);
int dcat_qct_gmm2_stop_v1(const dcat_qct::Config*);
int dcat_qct_gmm2_release_v1(const dcat_qct::Config*);
}
#endif
