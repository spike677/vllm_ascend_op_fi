// Host-only parts of the synchronous CANN Run bridge. Mechanisms provide the
// proven argument vector, CANN memory operations and original transformation.
#pragma once
#include "../../fault39/native/runtime_identity.h"
#include <functional>
#include <mutex>
#include <sstream>
#include <thread>
#include <vector>

namespace dcat_matmul {
enum Arm : uint32_t { original = 0, reference = 1, lossless = 2, fault = 3 };
enum StateFlag : uint32_t {
    callback_installed = 1, call_inflight = 2, recovery_required = 4,
    callback_owned = 8, runtime_matched = 16,
};
constexpr uint64_t byte_limit = 512ULL * 1024 * 1024;
using Callback = uint32_t (*)(uint64_t, const char*);
using Run = std::function<uint32_t(const std::vector<void*>&)>;
using Transform = std::function<bool(size_t, void*, size_t)>;

struct Slot {
    const char* formal;
    size_t index;
    uint64_t bytes;
    bool output;
    bool replace;
};

inline bool layout_safe(const std::vector<Slot>& slots, const std::vector<void*>& addresses) {
    uint64_t total = 0;
    size_t outputs = 0, replacements = 0;
    for (size_t i = 0; i < slots.size(); ++i) {
        const auto& s = slots[i];
        if (s.index >= addresses.size() || !addresses[s.index] || !s.bytes || s.bytes > byte_limit - total
            || (s.output && s.replace)) return false;
        const auto start = reinterpret_cast<uintptr_t>(addresses[s.index]);
        if (s.bytes > UINTPTR_MAX - start) return false;
        for (size_t j = 0; j < i; ++j) {
            const auto& t = slots[j];
            const auto prior = reinterpret_cast<uintptr_t>(addresses[t.index]);
            if (start < prior + t.bytes && prior < start + s.bytes) return false;
        }
        total += s.bytes;
        outputs += s.output;
        replacements += s.replace;
    }
    return outputs == 1 && replacements == 1;
}

struct Memory {
    std::function<int(void**, size_t)> alloc_host, alloc_device;
    std::function<int(void*)> free_host, free_device;
    std::function<int(void*, const void*, size_t)> read, write;
    std::function<int()> synchronize;
    std::function<int(void*, const void*, size_t)> copy_device;
};

struct Audit {
    uint32_t arm = original;
    uint32_t original_calls = 0, native_status = 1;
    uint32_t replacements = 0, changed_slots = 0;
    bool resources_released = true, temporary_readback_verified = false;
    bool device_copy = false;
    std::vector<std::string> inputs_before, inputs_after, replaced_formals;
    std::string output_hash;

    std::string json() const {
        std::ostringstream out;
        out << "{\"mode\":" << arm << ",\"original_calls\":" << original_calls
            << ",\"native_status\":" << native_status << ",\"replacements\":" << replacements
            << ",\"changed_slots\":" << changed_slots << ",\"resources_released\":" << std::boolalpha
            << resources_released << ",\"temporary_readback_verified\":" << temporary_readback_verified
            << ",\"device_copy\":" << device_copy;
        auto hashes = [&](const char* key, const std::vector<std::string>& values) {
            out << ",\"" << key << "\":[";
            bool first = true;
            for (const auto& value : values) {
                if (!first) out << ',';
                first = false;
                out << '"' << value << '"';
            }
            out << ']';
        };
        hashes("input_sha256_before", inputs_before);
        hashes("input_sha256_after", inputs_after);
        hashes("replaced_formals", replaced_formals);
        out << ",\"output_sha256\":\"" << output_hash << "\"}";
        return out.str();
    }
};

inline uint32_t audited_run(uint32_t arm, const std::vector<Slot>& slots,
                            const std::vector<void*>& addresses, const Memory& memory,
                            const Transform& transform, const Run& run, Audit& audit,
                            const std::vector<std::string>* expected_inputs = nullptr) {
    audit.arm = arm;
    if (arm == original) return run(addresses);
    const bool copying = arm == lossless;
    if (arm > fault || (copying && !memory.copy_device)
        || (expected_inputs && expected_inputs->size() + 1 != slots.size())
        || !layout_safe(slots, addresses) || memory.synchronize()) return 1;
    const auto largest = std::max_element(slots.begin(), slots.end(),
        [](const Slot& a, const Slot& b) { return a.bytes < b.bytes; })->bytes;
    void* host = nullptr;
    auto replacement = addresses;
    std::vector<void*> allocated(slots.size(), nullptr);
    bool uncertain_stream = false;
    bool completed = false;
    auto transfer_and_run = [&]() -> bool {
        if (memory.alloc_host(&host, largest) || !host) return false;
        for (size_t i = 0; i < slots.size(); ++i) {
            const auto& slot = slots[i];
            if (slot.output) continue;
            const auto address = addresses[slot.index];
            if (memory.read(host, address, slot.bytes)) return false;
            const auto initial = f39identity::digest(host, slot.bytes);
            // The selected Run must consume the same qualified inputs. Check
            // the backing bytes before making a replacement for this slot.
            if (expected_inputs && (*expected_inputs)[audit.inputs_before.size()] != initial) return false;
            audit.inputs_before.push_back(initial);
            if (arm == reference || !slot.replace) continue;
            if (arm == fault && !transform(slot.index, host, slot.bytes)) return false;
            const auto temporary = f39identity::digest(host, slot.bytes);
            if (temporary != initial) ++audit.changed_slots;
            if (memory.alloc_device(&allocated[i], slot.bytes) || !allocated[i]) return false;
            replacement[slot.index] = allocated[i];
            // Include other temporary allocations in the overlap check.
            const auto start = reinterpret_cast<uintptr_t>(allocated[i]);
            if (slot.bytes > UINTPTR_MAX - start) {
                uncertain_stream = true;
                return false;
            }
            for (size_t j = 0; j < slots.size(); ++j) {
                const auto& other = slots[j];
                for (const void* existing : {addresses[other.index], j < i ? allocated[j] : nullptr}) {
                    if (!existing) continue;
                    const auto prior = reinterpret_cast<uintptr_t>(existing);
                    if (start < prior + other.bytes && prior < start + slot.bytes) {
                        uncertain_stream = true;
                        return false;
                    }
                }
            }
            const auto status = copying ? memory.copy_device(allocated[i], address, slot.bytes)
                                        : memory.write(allocated[i], host, slot.bytes);
            if (status || memory.read(host, allocated[i], slot.bytes)
                || f39identity::digest(host, slot.bytes) != temporary) return false;
            ++audit.replacements;
            audit.replaced_formals.emplace_back(slot.formal);
        }
        audit.temporary_readback_verified = audit.replacements > 0;
        audit.device_copy = copying && audit.temporary_readback_verified;
        if (copying && !audit.device_copy) return false;
        if (arm == fault && !audit.changed_slots) return false;
        ++audit.original_calls;
        uncertain_stream = true;
        audit.native_status = run(replacement);
        if (memory.synchronize()) return false;
        uncertain_stream = false;
        if (audit.native_status) return false;
        for (const auto& slot : slots) {
            if (memory.read(host, addresses[slot.index], slot.bytes)) return false;
            auto hash = f39identity::digest(host, slot.bytes);
            if (slot.output) audit.output_hash = std::move(hash);
            else audit.inputs_after.push_back(std::move(hash));
        }
        return audit.inputs_before == audit.inputs_after;
    };
    try {
        completed = transfer_and_run();
    } catch (...) {
        completed = false;
    }
    // Unconfirmed execution may still read the replacements. Keep those
    // allocations alive and require process recovery rather than free early.
    audit.resources_released = !uncertain_stream;
    if (!uncertain_stream) {
        for (auto* buffer : allocated)
            if (buffer && memory.free_device(buffer)) audit.resources_released = false;
        if (host && memory.free_host(host)) audit.resources_released = false;
    }
    return completed && audit.resources_released ? audit.native_status : 1;
}

class Bridge {
    using Execute = std::function<uint32_t(uint32_t, Audit&, const std::vector<std::string>*)>;
    enum class Gate { empty, observed, copied, ready };
    struct Flight {
        uint64_t token;
        const Execute& execute;
        const std::string& descriptor;
        bool entered = false, invalid = false;
        uint32_t status = 1;
        std::string receipt;
    };
    std::mutex control_;
    Callback callback_ = nullptr;
    std::thread::id owner_;
    uint64_t serial_ = 0;
    Flight* flight_ = nullptr;
    bool quarantined_ = false;
    Gate gate_ = Gate::empty;
    bool fault_attempted_ = false;
    std::string target_;
    std::vector<std::string> reference_inputs_;
    std::string reference_output_;
    static inline thread_local Bridge* active_ = nullptr;

    bool finish_gate(uint32_t arm, const std::string& details, const Audit& audit) {
        if (arm == original || arm == fault) return true;
        if (audit.original_calls != 1 || audit.native_status || !audit.resources_released
            || audit.inputs_before.empty() || audit.inputs_before != audit.inputs_after
            || audit.output_hash.empty()) return false;
        if (arm == reference && gate_ == Gate::empty) {
            if (audit.replacements) return false;
            target_ = details;
            reference_inputs_ = audit.inputs_before;
            reference_output_ = audit.output_hash;
            gate_ = Gate::observed;
            return true;
        }
        const bool equal = target_ == details && reference_inputs_ == audit.inputs_before
            && reference_output_ == audit.output_hash;
        if (!equal) {
            gate_ = Gate::empty;
            return false;
        }
        if (arm == reference && gate_ == Gate::copied && !audit.replacements) {
            gate_ = Gate::ready;
            return true;
        }
        if (arm != lossless || audit.replacements != 1 || !audit.temporary_readback_verified
            || !audit.device_copy || audit.changed_slots)
            return false;
        gate_ = Gate::copied;
        return true;
    }

public:
    int install(Callback callback) {
        std::lock_guard<std::mutex> guard(control_);
        if (!callback || callback_ || quarantined_) return 1;
        callback_ = callback;
        owner_ = std::this_thread::get_id();
        gate_ = Gate::empty;
        fault_attempted_ = false;
        return 0;
    }

    int remove(Callback callback) {
        std::lock_guard<std::mutex> guard(control_);
        if (callback != callback_ || owner_ != std::this_thread::get_id()
            || flight_ || quarantined_) return 1;
        callback_ = nullptr;
        owner_ = {};
        target_.clear();
        reference_inputs_.clear();
        reference_output_.clear();
        return 0;
    }

    bool owned() {
        std::lock_guard<std::mutex> guard(control_);
        return callback_ != nullptr;
    }

    uint32_t state(Callback requester) {
        std::lock_guard<std::mutex> guard(control_);
        uint32_t flags = 0;
        if (callback_) flags |= callback_installed;
        if (flight_) flags |= call_inflight;
        if (quarantined_) flags |= recovery_required;
        if (requester && requester == callback_) flags |= callback_owned;
        return flags;
    }

    bool on_owner() {
        std::lock_guard<std::mutex> guard(control_);
        return callback_ && owner_ == std::this_thread::get_id() && !flight_;
    }

    uint32_t dispatch(const std::string& details, const Execute& execute) {
        Callback callback = nullptr;
        Flight flight{0, execute, details, false, false, 1, {}};
        {
            std::lock_guard<std::mutex> guard(control_);
            if (quarantined_ || !callback_ || owner_ != std::this_thread::get_id() || flight_ || active_) return 1;
            callback = callback_;
            flight.token = ++serial_;
            flight_ = &flight;
        }
        active_ = this;
        uint32_t rc = 1;
        try { rc = callback(flight.token, details.c_str()); } catch (...) {}
        active_ = nullptr;
        {
            std::lock_guard<std::mutex> guard(control_);
            flight_ = nullptr;
        }
        if (rc || !flight.entered || flight.invalid || flight.status) {
            gate_ = Gate::empty;
            return flight.status ? flight.status : 1;
        }
        return 0;
    }

    uint32_t invoke(uint64_t token, uint32_t arm) {
        if (active_ != this || !flight_) return 1;
        auto& flight = *flight_;
        if (flight.token != token || flight.entered || arm > fault) {
            flight.invalid = true;
            return 1;
        }
        flight.entered = true;
        if (arm == lossless && (gate_ != Gate::observed || target_ != flight.descriptor)) return 1;
        if (arm == fault) {
            if (fault_attempted_ || gate_ != Gate::ready || target_ != flight.descriptor) return 1;
            fault_attempted_ = true;
            gate_ = Gate::empty;
        }
        Audit audit;
        try {
            flight.status = flight.execute(arm, audit, arm == fault ? &reference_inputs_ : nullptr);
            if (arm != original) flight.receipt = audit.json();
            if (flight.status || !finish_gate(arm, flight.descriptor, audit)) {
                if (arm != original) gate_ = Gate::empty;
                if (!flight.status) flight.status = 1;
            }
        } catch (...) {
            if (arm != original) gate_ = Gate::empty;
            flight.status = 1;
        }
        if (!audit.resources_released) {
            std::lock_guard<std::mutex> guard(control_);
            quarantined_ = true;
        }
        return flight.status;
    }

    const char* receipt(uint64_t token) const {
        if (active_ != this || !flight_ || flight_->token != token || flight_->receipt.empty()) return nullptr;
        return flight_->receipt.c_str();
    }
};

inline std::string descriptor(const char* operator_name, const std::string& signature,
                              const void* cache, int device, const char* build_profile,
                              const char* dtype, const int64_t* shape, const int64_t* stride,
                              int64_t offset, const std::vector<Slot>& slots, uint64_t cache_lifecycle = 0) {
    // All strings here are bounded internal constants or numerically generated.
    std::ostringstream out;
    out << "{\"version\":1,\"boundary\":\"op_exec_cache_run\",\"operator\":\"" << operator_name
        << "\",\"native_signature\":\"" << signature << "\",\"cache_identity\":\""
        << f39identity::digest(&cache, sizeof(cache)) << "\",\"candidate_count\":1,\"device\":" << device
        << ",\"build_profile_sha256\":\"" << build_profile << "\",\"runtime_identity\":\"MATCHED_BUILD\","
        << "\"layout\":{\"dtype\":\"" << dtype << "\",\"format\":\"ND\",\"shape\":["
        << shape[0] << ',' << shape[1] << "],\"stride\":[" << stride[0] << ',' << stride[1]
        << "],\"offset\":" << offset << "},\"slots\":[";
    for (size_t i = 0; i < slots.size(); ++i) {
        if (i) out << ',';
        out << "{\"formal\":\"" << slots[i].formal << "\",\"slot\":" << slots[i].index
            << ",\"storage_bytes\":" << slots[i].bytes << '}';
    }
    out << ']';
    if (cache_lifecycle) out << ",\"cache_lifecycle\":\"" << cache_lifecycle << '"';
    out << '}';
    return out.str();
}
}  // namespace dcat_matmul
