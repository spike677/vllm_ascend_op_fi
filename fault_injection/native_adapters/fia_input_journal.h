#ifndef DCAT_FIA_INPUT_JOURNAL_V1_H
#define DCAT_FIA_INPUT_JOURNAL_V1_H
#include "quant_container_trace.h"

// Observation only: no argument replacement, device read, descriptor creation,
// Slice decoder, F admission, or claim about an executor's internal tensors.
namespace dcat_fij {
constexpr uint64_t MaxCalls=256, MaxPairs=128, MaxBytes=16*1024*1024;
constexpr uint32_t TensorCount=24, ContainerCount=5, FormalCount=44;
enum Operation:uint32_t {Prepare=1, Execute=2};
enum Error:uint64_t {None=0, Owner=1, Scope=2, Thread=3, Device=4, Budget=5,
    Metadata=6, Container=7, Executor=8, Original=9, Exception=10, Workspace=11};
struct Limits {uint32_t version{1},size{sizeof(Limits)};uint64_t max_calls{MaxCalls},max_pairs{MaxPairs},max_bytes{MaxBytes},max_windows{MaxCalls};};
struct ScopeIdentity {
    uint32_t version{1},size{sizeof(ScopeIdentity)};
    dcat_qct::Scope scope;
    uint64_t pid{},worker{},model{},module{};
    char call_id[33]{}; // The actual launch observer's UUID, not a new token.
};
struct Slot {
    uint32_t formal{},kind{};
    uint64_t handle{},registration{},creator_entry{},creator_return{},count{},actual_count{},
        payload_offset{},payload_bytes{},joined{},live_at_return{};
    int32_t size_status{};uint32_t reserved{};
};
struct Event {
    uint32_t version{1},size{sizeof(Event)},operation{},phase{};
    uint64_t sequence{},call{},prepare_registration{},owner_hi{},owner_lo{},generation{},
        scope_token{},forward{},operator_sequence{},boundary_sequence{},pid{},thread{},
        worker{},model{},module{},function{},function_base{},function_offset{},caller{},caller_base{},caller_offset{},
        monotonic_ns{},original_calls{},original_status_valid{},original_exception{},error{},
        executor{},workspace{},workspace_bytes{},formals[FormalCount]{};
    int64_t signed_scalars[10]{};
    double scale{};
    uint32_t softmax_lse{},layout_length{};
    char layout[16]{},call_id[33]{};
    int32_t device{-1},device_status{},status{};uint32_t reserved{};
    dcat_qct::Tensor tensors[TensorCount]{};
    Slot containers[ContainerCount]{};
};
struct State {
    uint32_t version{1},size{sizeof(State)};
    uint64_t installed{},stopped{},retired{},owner_hi{},owner_lo{},generation{},pid{},thread{},
        total_calls{},total_events{},retained_events{},active_calls{},pending_executors{},untracked_executors{},
        overflow{},first_error{},exceptions{},foreign_thread_calls{},outside_scope_calls{},getter_failures{},
        prepare_returns{},execute_returns{},original_prepares{},original_executes{},paired_execute_returns{},
        failed_prepare_outputs_unknown{},record_capacity_bytes{},resolver_ready{},prepare_function{},execute_function{},
        device_function{},list_size_function{},int_size_function{},observer_scope_function{},window_limit{},accepted_windows{},
        retained_scope_tokens{},retained_scope_token_characters{};
};
struct Sizes {uint32_t version{1},size{sizeof(Sizes)};uint64_t limits{sizeof(Limits)},scope{sizeof(ScopeIdentity)},
    slot{sizeof(Slot)},event{sizeof(Event)},state{sizeof(State)};int32_t float16_tag{},int8_tag{},nd_tag{},reserved{};};
}
extern "C" {
uint32_t dcat_fia_input_journal_marker_v1();
int dcat_fia_input_journal_resolve_v1(); // No device or operator call.
int dcat_fia_input_journal_sizes_v1(dcat_fij::Sizes*);
int dcat_fia_input_journal_begin_v1(const dcat_qct::Config*,const dcat_fij::Limits*);
int dcat_fia_input_journal_scope_begin_v1(const dcat_qct::Config*,const dcat_fij::ScopeIdentity*);
int dcat_fia_input_journal_scope_end_v1(const dcat_qct::Config*,uint64_t);
int dcat_fia_input_journal_state_v1(dcat_fij::State*);
int dcat_fia_input_journal_event_v1(uint64_t,dcat_fij::Event*);
int dcat_fia_input_journal_stop_v1(const dcat_qct::Config*);
int dcat_fia_input_journal_release_v1(const dcat_qct::Config*);
}
#endif
