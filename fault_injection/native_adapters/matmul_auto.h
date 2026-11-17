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
    callback_paused = 32, request_controlled = 64,
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
    std::string request_owner_;
    uint64_t request_generation_ = 0;
    bool paused_ = false;
    uint64_t serial_ = 0;
    Flight* flight_ = nullptr;
    bool quarantined_ = false;
    Gate gate_ = Gate::empty;
    bool fault_attempted_ = false;
    std::string target_;
    std::vector<std::string> reference_inputs_;
    std::string reference_output_;
    // v3 is a distinct, bounded multi-target contract. The original single
    // target state and all v1/v2 qualification rules remain unchanged.
    struct GroupTarget {
        Gate gate = Gate::empty;
        std::string descriptor;
        std::vector<std::string> inputs;
        std::string output;
        uint32_t faults = 0;
    };
    std::map<std::string, GroupTarget> group_targets_;
    bool group_mode_ = false, group_sealed_ = false, group_failed_ = false;
    uint32_t group_target_limit_ = 0, group_fault_limit_ = 0;
    uint64_t group_call_limit_ = 0, group_calls_ = 0;
    std::string group_identity_;
    static inline thread_local Bridge* active_ = nullptr;

    static bool sha256_key(const char* key) {
        if (!key || std::strlen(key) != 64) return false;
        for (const char* p = key; *p; ++p)
            if (!((*p >= '0' && *p <= '9') || (*p >= 'a' && *p <= 'f'))) return false;
        return true;
    }

    bool request_owner_matches(Callback callback, const char* owner, uint64_t generation) const {
        return callback && callback == callback_ && owner && !request_owner_.empty()
            && request_owner_ == owner && generation && generation == request_generation_
            && owner_ == std::this_thread::get_id();
    }

    void clear_owner() {
        callback_ = nullptr;
        owner_ = {};
        request_owner_.clear();
        request_generation_ = 0;
        paused_ = false;
        gate_ = Gate::empty;
        fault_attempted_ = false;
        target_.clear();
        reference_inputs_.clear();
        reference_output_.clear();
        group_targets_.clear();
        group_mode_ = group_sealed_ = group_failed_ = false;
        group_target_limit_ = group_fault_limit_ = 0;
        group_call_limit_ = group_calls_ = 0;
        group_identity_.clear();
    }

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
        // A matched-phase baseline may make more read-only observations after
        // qualification. They must preserve the exact target/input/output and
        // cannot reset the spent fault budget or introduce replacements.
        if (arm == reference && (gate_ == Gate::copied || gate_ == Gate::ready) && !audit.replacements) {
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

    // Request control retains the callback owner and qualification while idle.
    // It is a distinct ABI; legacy remove cannot revoke this reservation.
    int install_request(Callback callback, const char* owner, uint64_t generation) {
        std::lock_guard<std::mutex> guard(control_);
        if (!callback || !owner || !*owner || std::strlen(owner) > 128 || !generation
            || callback_ || quarantined_ || flight_ || active_) return 1;
        callback_ = callback;
        owner_ = std::this_thread::get_id();
        request_owner_ = owner;
        request_generation_ = generation;
        paused_ = true;
        gate_ = Gate::empty;
        fault_attempted_ = false;
        return 0;
    }

    int pause(Callback callback, const char* owner, uint64_t generation) {
        std::lock_guard<std::mutex> guard(control_);
        if (!request_owner_matches(callback, owner, generation) || flight_ || active_ || quarantined_)
            return 1;
        paused_ = true; // idempotent cleanup; does not reset qualification/spent
        return 0;
    }

    int install_group_request(Callback callback, const char* owner, uint64_t generation,
                              uint32_t max_targets, uint32_t max_faults, uint64_t max_calls) {
        if (!max_targets || max_targets > 4096 || !max_faults || max_faults > 8
            || max_calls < 3ULL * max_targets || max_calls > 65536) return 1;
        // Only the installing thread may resume a request owner. No callback
        // can enter between the paused installation and this configuration.
        if (install_request(callback, owner, generation)) return 1;
        std::lock_guard<std::mutex> guard(control_);
        group_mode_ = true;
        group_target_limit_ = max_targets;
        group_fault_limit_ = max_faults;
        group_call_limit_ = max_calls;
        return 0;
    }

    int seal_group(Callback callback, const char* owner, uint64_t generation,
                   uint32_t expected_targets, const char* identity) {
        std::lock_guard<std::mutex> guard(control_);
        if (!request_owner_matches(callback, owner, generation) || !paused_ || flight_ || active_
            || quarantined_ || !group_mode_ || group_sealed_ || group_failed_
            || !expected_targets || expected_targets != group_targets_.size() || !sha256_key(identity)) return 1;
        std::string keys;
        for (const auto& row : group_targets_) {
            if (row.second.gate != Gate::ready || row.second.faults) return 1;
            keys += row.first + '\n';
        }
        if (f39identity::digest(keys.data(), keys.size()) != identity) return 1;
        group_identity_ = identity;
        group_sealed_ = true;
        return 0;
    }

    std::string group_state(Callback callback, const char* owner, uint64_t generation) {
        std::lock_guard<std::mutex> guard(control_);
        const bool owned = request_owner_matches(callback, owner, generation);
        if (flight_) {
            // Target audits are mutated by the synchronous owner while Run is
            // active. A foreign query must not race that state or imply an
            // idle recovery point.
            return std::string("{\"version\":3,\"inflight\":true,\"owned\":") + (owned ? "true}" : "false}");
        }
        std::ostringstream out;
        out << "{\"version\":3,\"owned\":" << std::boolalpha << owned
            << ",\"group_mode\":" << group_mode_ << ",\"sealed\":" << group_sealed_
            << ",\"failed\":" << group_failed_ << ",\"targets\":" << group_targets_.size()
            << ",\"calls\":" << group_calls_ << ",\"max_calls\":" << group_call_limit_
            << ",\"max_targets\":" << group_target_limit_ << ",\"max_faults\":" << group_fault_limit_
            << ",\"group_sha256\":\"" << group_identity_ << "\",\"entries\":[";
        bool first = true;
        if (owned) for (const auto& row : group_targets_) {
            if (!first) out << ',';
            first = false;
            out << "{\"key\":\"" << row.first << "\",\"ready\":" << (row.second.gate == Gate::ready)
                << ",\"fault_attempts\":" << row.second.faults << '}';
        }
        out << "]}";
        return out.str();
    }

    int resume(Callback callback, const char* owner, uint64_t generation) {
        std::lock_guard<std::mutex> guard(control_);
        if (!request_owner_matches(callback, owner, generation) || !paused_
            || flight_ || active_ || quarantined_) return 1;
        paused_ = false;
        return 0;
    }

    int remove_request(Callback callback, const char* owner, uint64_t generation) {
        std::lock_guard<std::mutex> guard(control_);
        if (!request_owner_matches(callback, owner, generation) || !paused_
            || flight_ || active_ || quarantined_) return 1;
        clear_owner();
        return 0;
    }

    int remove(Callback callback) {
        std::lock_guard<std::mutex> guard(control_);
        if (callback != callback_ || owner_ != std::this_thread::get_id() || !request_owner_.empty()
            || flight_ || quarantined_) return 1;
        clear_owner();
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

    uint32_t request_state(Callback requester, const char* owner, uint64_t generation) {
        std::lock_guard<std::mutex> guard(control_);
        uint32_t flags = 0;
        if (callback_) flags |= callback_installed;
        if (flight_) flags |= call_inflight;
        if (quarantined_) flags |= recovery_required;
        if (paused_) flags |= callback_paused;
        if (!request_owner_.empty()) flags |= request_controlled;
        const bool matches = request_owner_.empty() ? requester && requester == callback_
            : request_owner_matches(requester, owner, generation);
        if (matches) flags |= callback_owned;
        return flags;
    }

    bool on_owner() {
        std::lock_guard<std::mutex> guard(control_);
        // A quarantined owner still routes to dispatch's hard refusal. Do not
        // turn quarantine into the caller's unowned/original passthrough path.
        return callback_ && !paused_ && owner_ == std::this_thread::get_id() && !flight_;
    }

    uint32_t dispatch(const std::string& details, const Execute& execute, const std::string& stable_target = {}) {
        Callback callback = nullptr;
        // The packet keeps every current typed-call fact. Qualification compares
        // a separate stable native identity, never a reused GWS token/address.
        Flight flight{0, execute, stable_target.empty() ? details : stable_target, false, false, 1, {}};
        {
            std::lock_guard<std::mutex> guard(control_);
            if (quarantined_ || !callback_ || paused_ || owner_ != std::this_thread::get_id() || flight_ || active_) return 1;
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
            if (group_mode_ && (rc || !flight.entered || flight.invalid || flight.status)) group_failed_ = true;
        }
        if (rc || !flight.entered || flight.invalid || flight.status) {
            gate_ = Gate::empty;
            return flight.status ? flight.status : 1;
        }
        return 0;
    }

    uint32_t invoke(uint64_t token, uint32_t arm) {
        if (active_ != this || !flight_ || paused_ || group_mode_) return 1;
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

    uint32_t invoke_group(uint64_t token, uint32_t arm, const char* target_key) {
        if (active_ != this || !flight_ || paused_ || !group_mode_) return 1;
        auto& flight = *flight_;
        if (flight.token != token || flight.entered || arm > fault) {
            flight.invalid = true;
            group_failed_ = true;
            return 1;
        }
        flight.entered = true;
        const auto refuse = [&]() -> uint32_t { group_failed_ = true; return flight.status = 1; };
        if (++group_calls_ > group_call_limit_) return refuse();
        GroupTarget* target = nullptr;
        if (arm != original) {
            if (!sha256_key(target_key)) return refuse();
            auto found = group_targets_.find(target_key);
            if (found == group_targets_.end()) {
                if (arm != reference || group_sealed_ || group_failed_
                    || group_targets_.size() >= group_target_limit_) return refuse();
                GroupTarget initial;
                initial.descriptor = flight.descriptor;
                found = group_targets_.emplace(target_key, std::move(initial)).first;
            }
            target = &found->second;
            if (target->descriptor != flight.descriptor) return refuse();
            if (arm == lossless && (group_sealed_ || target->gate != Gate::observed)) return refuse();
            if (arm == fault) {
                if (!group_sealed_ || group_failed_ || target->gate != Gate::ready
                    || target->faults >= group_fault_limit_) return refuse();
                ++target->faults; // A failed attempt cannot replenish its lease.
            }
        }
        Audit audit;
        try {
            // Fresh propagated values are permitted only by this v3 group
            // contract. Typed current-call storage and exact transformation
            // are still audited. Healthy phases and OFF retain byte equality.
            const auto* expected = target && arm != fault && target->gate != Gate::empty ? &target->inputs : nullptr;
            flight.status = flight.execute(arm, audit, expected);
            if (arm != original) {
                const bool invariant = audit.original_calls == 1 && !audit.native_status && audit.resources_released
                    && !audit.inputs_before.empty() && audit.inputs_before == audit.inputs_after && !audit.output_hash.empty();
                bool accepted = invariant;
                if (arm == reference && target->gate == Gate::empty) {
                    accepted = accepted && !audit.replacements;
                    if (accepted) {
                        target->inputs = audit.inputs_before;
                        target->output = audit.output_hash;
                        target->gate = Gate::observed;
                    }
                } else if (arm == reference) {
                    accepted = accepted && !audit.replacements && audit.inputs_before == target->inputs
                        && audit.output_hash == target->output
                        && (target->gate == Gate::copied || target->gate == Gate::ready);
                    if (accepted) target->gate = Gate::ready;
                } else if (arm == lossless) {
                    accepted = accepted && audit.replacements == 1 && audit.temporary_readback_verified
                        && audit.device_copy && !audit.changed_slots && audit.inputs_before == target->inputs
                        && audit.output_hash == target->output;
                    if (accepted) target->gate = Gate::copied;
                } else if (arm == fault) {
                    accepted = accepted && audit.replacements == 1 && audit.changed_slots == 1
                        && audit.temporary_readback_verified && !audit.device_copy;
                }
                const std::string raw = audit.json();
                std::ostringstream receipt;
                receipt << raw.substr(0, raw.size() - 1) << ",\"group_v3\":{\"target_key\":\"" << target_key
                    << "\",\"group_sha256\":\"" << group_identity_ << "\",\"call_index\":" << group_calls_
                    << ",\"fault_attempts\":" << target->faults << ",\"input_equal_healthy_reference\":"
                    << std::boolalpha << (audit.inputs_before == target->inputs) << "}}";
                flight.receipt = receipt.str();
                if (!accepted && !flight.status) flight.status = 1;
            }
        } catch (...) { flight.status = 1; }
        if (flight.status) group_failed_ = true;
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

// This denominator has no callback, owner, arm, gate, or hit-state inputs.
// Capacity loss is explicit and totals keep counting. Registry lifetime and
// actual current typed Run identity are encoded in each native descriptor.
class RunInventory {
    struct Row { std::string descriptor; uint64_t count = 0, last_gws_token = 0; };
    std::mutex mutex_;
    size_t capacity_;
    uint64_t total_ = 0, eligible_ = 0, rejected_ = 0, other_ = 0, overflow_ = 0;
    std::map<std::string, Row> rows_;
    std::map<std::string, uint64_t> reasons_;
public:
    explicit RunInventory(size_t capacity = 4096) : capacity_(capacity) {}
    void begin_run() { std::lock_guard<std::mutex> lock(mutex_); ++total_; }
    void non_target() { std::lock_guard<std::mutex> lock(mutex_); ++other_; }
    void reject(uint32_t kind, const char* reason) {
        std::lock_guard<std::mutex> lock(mutex_);
        ++rejected_;
        const auto key = std::to_string(kind) + ':' + reason;
        auto found = reasons_.find(key);
        if (found != reasons_.end()) ++found->second;
        else if (reasons_.size() < 128) reasons_.emplace(key, 1);
        else ++overflow_;
    }
    void eligible(const std::string& descriptor, uint64_t current_token) {
        std::lock_guard<std::mutex> lock(mutex_);
        ++eligible_;
        const auto key = f39identity::digest(descriptor.data(), descriptor.size());
        auto found = rows_.find(key);
        if (found == rows_.end()) {
            if (rows_.size() >= capacity_ || descriptor.empty() || descriptor.size() > 32768) { ++overflow_; return; }
            found = rows_.emplace(key, Row{descriptor, 0, 0}).first;
        }
        if (found->second.descriptor != descriptor) { ++overflow_; return; }
        ++found->second.count;
        found->second.last_gws_token = current_token;
    }
    std::string snapshot() {
        std::lock_guard<std::mutex> lock(mutex_);
        std::ostringstream out;
        out << "{\"schema\":\"dcat20.independent_run_inventory.v1\",\"total_run_calls\":" << total_
            << ",\"eligible_run_calls\":" << eligible_ << ",\"rejected_candidate_calls\":" << rejected_
            << ",\"non_target_run_calls\":" << other_ << ",\"overflow_calls\":" << overflow_
            << ",\"capacity\":" << capacity_ << ",\"complete\":" << std::boolalpha
            << (!overflow_ && total_ == eligible_ + rejected_ + other_) << ",\"targets\":[";
        bool first = true;
        for (const auto& row : rows_) {
            if (!first) out << ',';
            first = false;
            out << "{\"identity_sha256\":\"" << row.first << "\",\"observations\":" << row.second.count
                << ",\"last_gws_token\":" << row.second.last_gws_token
                << ",\"native\":" << row.second.descriptor << '}';
        }
        out << "],\"rejections\":[";
        first = true;
        for (const auto& reason : reasons_) {
            if (!first) out << ',';
            first = false;
            out << "{\"reason\":\"" << reason.first << "\",\"calls\":" << reason.second << '}';
        }
        out << "]}";
        return out.str();
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
