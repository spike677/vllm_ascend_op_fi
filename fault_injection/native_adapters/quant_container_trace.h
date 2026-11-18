#ifndef DCAT_QUANT_CONTAINER_TRACE_V1_H
#define DCAT_QUANT_CONTAINER_TRACE_V1_H
#include <cstdint>
#include "../../fault39/native/metadata_array_state.h"

namespace dcat_qct {
constexpr uint64_t MaxCalls=4096, MaxEvents=8192, MaxMembers=128, MaxBytes=8*1024*1024, MaxRank=8;
enum Operation:uint32_t {CreateTensors=1, DestroyTensors=2, CreateIntegers=3, DestroyIntegers=4};
enum Error:uint64_t {NoError=0, Identity=1, Budget=2, Metadata=3, DuplicateLive=4,
    UnknownDestroy=5, OriginalFailed=6, BadScope=7, Exception=8, WrongThread=9};
struct Config {
    uint32_t version{1},size{sizeof(Config)};
    uint64_t owner_hi{},owner_lo{},generation{},thread{},max_calls{MaxCalls},max_bytes{MaxBytes};
};
struct Scope {
    uint32_t version{1},size{sizeof(Scope)};
    uint64_t token{},forward{},operator_sequence{},boundary_sequence{},stage{},thread{};
};
// Host descriptor metadata only. No Tensor private layout or device data read.
struct Tensor {
    uint64_t descriptor{},address{},counts[3]{},called_mask{},null_mask{},truncated_mask{};
    int64_t offset{},shape[MaxRank]{},strides[MaxRank]{},storage[MaxRank]{};
    int32_t dtype{-1},format{-1},statuses[7]{};
    uint32_t complete{};
};
struct Event {
    uint32_t version{1},size{sizeof(Event)},operation{},phase{}; // phase 1 entry, 2 return
    uint64_t sequence{},call{},registration{},scope_token{},forward{},operator_sequence{},boundary_sequence{},stage{};
    uint64_t pid{},thread{},depth{},function{},function_base{},function_offset{},caller{},caller_base{},caller_offset{};
    uint64_t caller_kind{},handle{},count{},payload_offset{},payload_bytes{},original_calls{},monotonic_ns{},error{};
    int32_t status{},reserved{};
};
struct State {
    uint32_t version{1},size{sizeof(State)};
    uint64_t installed{},stopped{},retired{},owner_hi{},owner_lo{},generation{},owner_thread{},scope_token{};
    uint64_t total_calls{},total_events{},retained_events{},metadata_bytes{},live_registrations{},active_calls{},overflow{},first_error{},
        abi_exceptions{},getter_capture_failures{},outside_scope_calls{},foreign_scope_thread_calls{},
        record_capacity_bytes{},payload_capacity_bytes{},registration_count{},elapsed_ns{},resolver_ready{};
    uint64_t entries[4]{},returns[4]{},original_calls[4]{};
    DcatMetadataArrayStateV1 arrays{};
};
struct Sizes {
    uint32_t version{1},size{sizeof(Sizes)};
    uint64_t config{sizeof(Config)},scope{sizeof(Scope)},tensor{sizeof(Tensor)},event{sizeof(Event)},state{sizeof(State)};
};
}
extern "C" {
uint32_t dcat_qct_marker_v1();
const char* dcat_qct_profile_v1();
int dcat_qct_sizes_v1(dcat_qct::Sizes*);
int dcat_qct_state_v1(dcat_qct::State*);
int dcat_qct_begin_v1(const dcat_qct::Config*);
int dcat_qct_scope_begin_v1(const dcat_qct::Config*,const dcat_qct::Scope*);
int dcat_qct_scope_end_v1(const dcat_qct::Config*,uint64_t);
int dcat_qct_stop_v1(const dcat_qct::Config*);
int dcat_qct_event_v1(uint64_t,dcat_qct::Event*);
int dcat_qct_payload_v1(uint64_t,uint64_t,void*);
int dcat_qct_release_v1(const dcat_qct::Config*);
}
#endif
