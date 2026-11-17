// Current typed GWS -> executor -> Execute binding. No cached address/layout
// is used to manufacture a current tensor. This header needs no CANN runtime.
#pragma once
#include <array>
#include <cstdint>
#include <limits>
#include <sstream>
#include <thread>
#include <vector>
#include "../../fault20_fast_validation/source/l4_buffer_substitute_allhit_v1/fault20_buffer_api.h"

namespace dcat_current {
enum Kind : uint32_t { qmm = 1, weight_nz = 2 };

struct Call {
    uint64_t token = 0;
    uint32_t kind = 0;
    const void* executor = nullptr;
    const void* cache_created = nullptr;
    uint32_t add_count = 0, parameter_count = 0;
    std::array<uintptr_t, 16> parameters{};
    int32_t device = -1;
    Fault20QmmMetadata quant{};
    Fault20MatmulMetadata floating{};
};

// Private runtime observation, not a substitute for live_storage_slot or typed
// scope admission. Addresses belong to this invocation only, never stable keys.
inline std::string tensor_observation(const char* formal, const Fault20TensorMetadata& tensor,
                                      int32_t slot, uint64_t run_address) {
    if (!formal || tensor.view_rank > FAULT20_MAX_DIMS || tensor.storage_rank > FAULT20_MAX_DIMS
        || tensor.stride_rank > FAULT20_MAX_DIMS) return "{}";
    std::ostringstream out;
    out << "{\"formal\":\"" << formal << "\",\"present\":" << (tensor.present ? "true" : "false")
        << ",\"dtype\":" << tensor.dtype << ",\"format\":" << tensor.format
        << ",\"contiguous\":" << (tensor.contiguous ? "true" : "false")
        << ",\"view_offset\":" << tensor.view_offset << ",\"numel\":" << tensor.numel
        << ",\"storage_elements\":" << tensor.storage_elements << ",\"element_size\":" << tensor.element_size
        << ",\"view_bytes\":" << tensor.view_bytes << ",\"storage_bytes\":" << tensor.storage_bytes
        << ",\"raw_addr\":" << tensor.raw_addr << ",\"storage_addr\":" << tensor.storage_addr
        << ",\"data_addr\":" << tensor.data_addr << ",\"run_slot\":" << slot
        << ",\"run_address\":" << run_address;
    const auto dimensions = [&](const char* key, const int64_t* values, uint32_t count) {
        out << ",\"" << key << "\":[";
        for (uint32_t index = 0; index < count; ++index) {
            if (index) out << ',';
            out << values[index];
        }
        out << ']';
    };
    dimensions("view_shape", tensor.view_shape, tensor.view_rank);
    dimensions("storage_shape", tensor.storage_shape, tensor.storage_rank);
    dimensions("strides", tensor.strides, tensor.stride_rank);
    out << '}';
    return out.str();
}

inline bool live_storage_slot(const Fault20TensorMetadata& tensor, int32_t slot,
                              const std::vector<void*>& addresses) {
    if (!tensor.present || !tensor.storage_ptr || !tensor.storage_bytes || tensor.view_offset
        || slot < 0 || static_cast<size_t>(slot) >= addresses.size() || !tensor.storage_addr
        || reinterpret_cast<uintptr_t>(addresses[slot]) != tensor.storage_addr) return false;
    int32_t matched = -1;
    for (size_t i = 0; i < addresses.size(); ++i) {
        const auto actual = reinterpret_cast<uintptr_t>(addresses[i]);
        if ((tensor.raw_addr && tensor.raw_addr == actual)
            || (tensor.storage_addr && tensor.storage_addr == actual)
            || (tensor.data_addr && tensor.data_addr == actual)) {
            if (matched >= 0) return false;
            matched = static_cast<int32_t>(i);
        }
    }
    return matched == slot;
}

inline bool formals_match(const Call& call) {
    if (call.device < 0 || call.add_count > 1
        || (call.add_count == 1) != (call.cache_created != nullptr)) return false;
    if (call.kind == qmm) {
        const auto& m = call.quant;
        const Fault20TensorMetadata* formals[] = {&m.x1, &m.x2, &m.x1_scale, &m.x2_scale,
            &m.y_scale, &m.x1_offset, &m.x2_offset, &m.y_offset, &m.bias, &m.out};
        if (call.parameter_count != 10 || !m.x1.present || !m.x2.present || !m.out.present)
            return false;
        for (size_t i = 0; i < 10; ++i) {
            if (call.parameters[i] != formals[i]->tensor_ptr
                || bool(formals[i]->present) != bool(formals[i]->tensor_ptr)) return false;
        }
        return true;
    }
    if (call.kind == weight_nz) {
        const auto& m = call.floating;
        return call.parameter_count == 3 && m.all_present && !m.any_empty_storage
            && !m.any_shared_storage && m.self.present && m.mat2.present && m.out.present
            && m.self.tensor_ptr && m.mat2.tensor_ptr && m.out.tensor_ptr
            && call.parameters[0] == m.self.tensor_ptr
            && call.parameters[1] == m.mat2.tensor_ptr
            && call.parameters[2] == m.out.tensor_ptr;
    }
    return false;
}

// One synchronous typed call is supported per thread. Queued/reordered GWS,
// duplicate executors and nested Execute are rejected, never guessed. Caller
// keeps this object thread_local; the explicit check is also tested on CPU.
class Binding {
    const std::thread::id owner_ = std::this_thread::get_id();
    uint64_t serial_ = 0, building_ = 0;
    uint32_t building_kind_ = 0;
    bool invalid_build_ = false;
    bool invalid_active_ = false;
    Call pending_{}, active_{};
    const char* reason_ = "INITIAL";

    bool on_owner() const { return owner_ == std::this_thread::get_id(); }
public:
    const char* reason() const { return on_owner() ? reason_ : "FOREIGN_THREAD"; }

    uint64_t begin(uint32_t kind) {
        if (!on_owner()) return 0;
        if (kind != qmm && kind != weight_nz) { reason_ = "WRONG_OPERATOR"; return 0; }
        if (active_.token) {
            invalid_active_ = true;
            reason_ = "NESTED_GWS_DURING_EXECUTE";
            return 0;
        }
        if (building_) {
            invalid_build_ = true;
            reason_ = "NESTED_GWS";
            return 0;
        }
        if (serial_ == std::numeric_limits<uint64_t>::max()) {
            reason_ = "SEQUENCE_EXHAUSTED";
            return 0;
        }
        // If a prior typed call was never executed, the same executor address
        // can be ambiguous. Refuse both rather than replacing its evidence.
        invalid_build_ = pending_.token != 0;
        pending_ = {};
        building_ = ++serial_;
        building_kind_ = kind;
        reason_ = invalid_build_ ? "UNCONSUMED_GWS" : "GWS_STARTED";
        return building_;
    }

    bool publish(uint64_t token, Call call) {
        if (!on_owner()) return false;
        if (!token || token != building_ || call.kind != building_kind_) {
            reason_ = "GWS_TOKEN_MISMATCH";
            return false;
        }
        building_ = 0;
        building_kind_ = 0;
        if (invalid_build_) { reason_ = "GWS_SCOPE_AMBIGUOUS"; return false; }
        if (!call.executor || !formals_match(call)) {
            reason_ = "CURRENT_TYPED_FORMALS_UNPROVEN";
            return false;
        }
        call.token = token;
        pending_ = call;
        reason_ = "CURRENT_TYPED_GWS_BOUND";
        return true;
    }

    void abort(uint64_t token) {
        if (!on_owner()) return;
        if (token && token == building_) {
            building_ = 0;
            building_kind_ = 0;
            invalid_build_ = false;
            pending_ = {};
            reason_ = "GWS_ABORTED";
        }
    }

    uint64_t enter(uint32_t kind, const void* executor) {
        if (!on_owner()) return 0;
        if (active_.token) {
            invalid_active_ = true;
            reason_ = "NESTED_EXECUTE";
            return 0;
        }
        if (building_ || !pending_.token) { reason_ = "MISSING_CURRENT_TYPED_GWS"; return 0; }
        // Consume before checking the executor. An uncertain/reordered caller
        // cannot retry this typed-call capability with another executor.
        const auto next = pending_;
        pending_ = {};
        if (next.kind != kind || next.executor != executor) {
            reason_ = "EXECUTOR_OR_OPERATOR_MISMATCH";
            return 0;
        }
        active_ = next;
        invalid_active_ = false;
        reason_ = "CURRENT_EXECUTE_BOUND";
        return active_.token;
    }

    const Call* current(uint32_t kind, const void* cache) const {
        if (!on_owner() || invalid_active_ || active_.kind != kind || !active_.token) return nullptr;
        if (active_.cache_created && active_.cache_created != cache) return nullptr;
        return &active_;
    }

    bool leave(uint64_t token, uint32_t kind, const void* executor) {
        if (!on_owner()) return false;
        // A refused nested entry has token 0 and cannot clear the outer scope.
        if (!token) return false;
        const bool match = active_.token == token && active_.kind == kind && active_.executor == executor;
        active_ = {};
        invalid_active_ = false;
        reason_ = match ? "EXECUTE_RELEASED" : "EXECUTE_RELEASE_MISMATCH";
        return match;
    }
};
} // namespace dcat_current
