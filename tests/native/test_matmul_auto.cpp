// Host memory and a CPU function stand in for ACL in this ABI test.
#include "../../fault_injection/native_adapters/matmul_auto.h"
#include "../../fault20_fast_validation/source/l4_buffer_substitute_allhit_v1/int_equivalent_transform.h"
#include "../../fault20_fast_validation/source/l4_buffer_substitute_allhit_v1/mantissa_truncate.h"
#include <cassert>
#include <cstdlib>
#include <iostream>

using namespace dcat_matmul;

Memory host_memory(unsigned* releases = nullptr, bool* fail_sync = nullptr) {
    auto allocate = [](void** p, size_t n) { *p = std::malloc(n); return *p ? 0 : 1; };
    auto release = [releases](void* p) {
        if (releases) ++*releases;
        std::free(p);
        return 0;
    };
    auto copy = [](void* to, const void* from, size_t n) { std::memcpy(to, from, n); return 0; };
    return {allocate, allocate, release, release, copy, copy,
            [fail_sync]() { return fail_sync && *fail_sync ? 1 : 0; }, copy};
}

const char* profile = "1234567812345678123456781234567812345678123456781234567812345678";
Bridge bridge;
Bridge weight_nz_bridge;

extern "C" const char* dcat20_qmm_profile_v1() { return profile; }
extern "C" int dcat20_qmm_install_v1(Callback callback, int32_t) { return bridge.install(callback); }
extern "C" int dcat20_qmm_remove_v1(Callback callback) { return bridge.remove(callback); }
extern "C" uint32_t dcat20_qmm_invoke_v1(uint64_t token, uint32_t arm) { return bridge.invoke(token, arm); }
extern "C" const char* dcat20_qmm_receipt_v1(uint64_t token) { return bridge.receipt(token); }
extern "C" uint32_t dcat20_qmm_state_v1(Callback requester) {
    return bridge.state(requester) | runtime_matched;
}
extern "C" int dcat20_weight_nz_install_v1(Callback callback, int32_t) { return weight_nz_bridge.install(callback); }
extern "C" int dcat20_weight_nz_remove_v1(Callback callback) { return weight_nz_bridge.remove(callback); }
extern "C" uint32_t dcat20_weight_nz_invoke_v1(uint64_t token, uint32_t arm) {
    return weight_nz_bridge.invoke(token, arm);
}
extern "C" const char* dcat20_weight_nz_receipt_v1(uint64_t token) { return weight_nz_bridge.receipt(token); }
extern "C" uint32_t dcat20_weight_nz_state_v1(Callback requester) {
    return weight_nz_bridge.state(requester) | runtime_matched;
}

#define FIXTURE_REQUEST_ABI(prefix, instance) \
extern "C" int prefix##install_v2(Callback cb, int32_t, const char* owner, uint64_t generation) { \
    return instance.install_request(cb, owner, generation); \
} \
extern "C" int prefix##pause_v2(Callback cb, const char* owner, uint64_t generation) { \
    return instance.pause(cb, owner, generation); \
} \
extern "C" int prefix##resume_v2(Callback cb, const char* owner, uint64_t generation) { \
    return instance.resume(cb, owner, generation); \
} \
extern "C" int prefix##remove_v2(Callback cb, const char* owner, uint64_t generation) { \
    return instance.remove_request(cb, owner, generation); \
} \
extern "C" uint32_t prefix##state_v2(Callback cb, const char* owner, uint64_t generation) { \
    return instance.request_state(cb, owner, generation) | runtime_matched; \
}
FIXTURE_REQUEST_ABI(dcat20_qmm_, bridge)
FIXTURE_REQUEST_ABI(dcat20_weight_nz_, weight_nz_bridge)
#undef FIXTURE_REQUEST_ABI

int fixture_qmm_run_for_generation(uint64_t generation, int8_t first = 37, int* run_calls = nullptr,
                                    uint64_t current_token = 0) {
    int8_t x[]{first, -39};
    int8_t weight[]{1, 3};
    int32_t out = 0;
    std::vector<void*> addresses{x, weight, &out};
    std::vector<Slot> slots{{"x1", 0, sizeof(x), false, true},
                            {"x2", 1, sizeof(weight), false, false},
                            {"out", 2, sizeof(out), true, false}};
    int64_t shape[]{1, 2}, stride[]{2, 1};
    const auto details = descriptor("aclnnQuantMatmulWeightNz", "fixture:qmm", &bridge, 0, profile,
                                    "INT8", shape, stride, 0, slots, generation);
    const auto packet = current_token ? details.substr(0, details.size() - 1)
        + ",\"current_typed_call\":{\"gws_token\":" + std::to_string(current_token) + "}}" : details;
    auto run = [&](const auto& set) {
        if (run_calls) ++*run_calls;
        const auto* input = static_cast<const int8_t*>(set[0]);
        *static_cast<int32_t*>(set[2]) = input[0] * weight[0] + input[1] * weight[1];
        return 0;
    };
    if (!bridge.on_owner()) { run(addresses); return out; }
    uint32_t status = bridge.dispatch(packet, [&](uint32_t arm, Audit& audit, const auto* expected) {
        const auto transform = [](size_t index, void* host, size_t n) {
            assert(index == 0);
            auto* values = static_cast<int8_t*>(host);
            for (size_t i = 0; i < n; ++i) values[i] = fault20_int4_equivalent_value(values[i]);
            return true;
        };
        return audited_run(arm, slots, addresses, host_memory(), transform, run, audit, expected);
    }, details);
    assert(x[0] == first && x[1] == -39 && weight[0] == 1 && weight[1] == 3);
    return status ? 9999 : out;
}

extern "C" int fixture_qmm_run() { return fixture_qmm_run_for_generation(1); }
extern "C" int fixture_qmm_run_current(uint64_t token) { return fixture_qmm_run_for_generation(1, 37, nullptr, token); }

extern "C" int fixture_weight_nz_run() {
    uint16_t input[]{0x3f89, 0x400f}, weight[]{0x3f80, 0x4000}, out = 0;
    const std::vector<void*> addresses{input, weight, &out};
    const std::vector<Slot> slots{{"self", 0, sizeof(input), false, true},
                                 {"mat2", 1, sizeof(weight), false, false},
                                 {"out", 2, sizeof(out), true, false}};
    const int64_t shape[]{1, 2}, stride[]{2, 1};
    const auto details = descriptor("aclnnMatmulWeightNz", "fixture:weight_nz", &weight_nz_bridge, 0,
                                    profile, "BF16", shape, stride, 0, slots, 1);
    const auto run = [&](const auto& set) {
        assert(set[1] == weight && set[2] == &out);
        const auto* x = static_cast<const uint16_t*>(set[0]);
        *static_cast<uint16_t*>(set[2]) = x[0] ^ x[1];
        return 0;
    };
    if (!weight_nz_bridge.on_owner()) { run(addresses); return out; }
    const auto transform = [](size_t index, void* host, size_t bytes) {
        assert(index == 0 && bytes == 4);
        auto* values = static_cast<uint16_t*>(host);
        for (size_t i = 0; i < bytes / sizeof(*values); ++i)
            values[i] = fault20_truncate_bf16_mantissa(values[i], 3);
        return true;
    };
    auto status = weight_nz_bridge.dispatch(details, [&](uint32_t arm, Audit& audit, const auto* expected) {
        return audited_run(arm, slots, addresses, host_memory(), transform, run, audit, expected);
    });
    assert(input[0] == 0x3f89 && input[1] == 0x400f && weight[0] == 0x3f80 && weight[1] == 0x4000);
    return status ? 9999 : out;
}

#ifndef DCAT_TEST_SHARED
uint32_t expected_arm = original;
bool retry = false;
bool refused = false;
bool other_thread = false;

uint32_t callback(uint64_t token, const char* packet) {
    assert(std::strstr(packet, "\"operator\":\"aclnnQuantMatmulWeightNz\""));
    assert(bridge.state(callback) == (callback_installed | call_inflight | callback_owned));
    if (other_thread) {
        std::thread foreign([&] {
            assert(bridge.state(nullptr) == (callback_installed | call_inflight));
            assert(bridge.invoke(token, fault) != 0);
            assert(fixture_qmm_run() == -80);
        });
        foreign.join();
    }
    const auto status = bridge.invoke(token, expected_arm);
    if (refused) {
        assert(status != 0);
        return 1;
    }
    assert(status == 0);
    if (expected_arm != original)
        assert(std::strstr(bridge.receipt(token), "\"resources_released\":true"));
    if (retry) assert(bridge.invoke(token, expected_arm) != 0);
    return 0;
}

int main() {
    const int baseline = fixture_qmm_run();
    assert(bridge.state(callback) == 0);
    assert(bridge.install(callback) == 0);
    assert(bridge.state(callback) == (callback_installed | callback_owned));
    assert(weight_nz_bridge.state(callback) == 0);
    assert(bridge.install(callback) != 0);
    refused = true;
    expected_arm = fault;
    assert(fixture_qmm_run() == 9999);
    refused = false;
    for (auto arm : {original, reference, lossless, reference}) {
        expected_arm = arm;
        assert(fixture_qmm_run() == baseline);
    }
    expected_arm = fault;
    other_thread = true;
    assert(fixture_qmm_run() != baseline);
    other_thread = false;
    refused = true;
    assert(fixture_qmm_run() == 9999);
    refused = false;
    expected_arm = original;
    retry = true;
    assert(fixture_qmm_run() == 9999);
    retry = false;
    assert(bridge.remove(callback) == 0);
    assert(bridge.state(callback) == 0);
    assert(fixture_qmm_run() == baseline);
    assert(bridge.install(callback) == 0);
    for (auto arm : {reference, lossless, reference}) {
        expected_arm = arm;
        assert(fixture_qmm_run() == baseline);
    }
    expected_arm = fault;
    refused = true;
    // Same executor address and tensor geometry. Registration lifecycle differs.
    assert(fixture_qmm_run_for_generation(2) == 9999);
    refused = false;
    assert(bridge.remove(callback) == 0);

    assert(bridge.install(callback) == 0);
    for (auto arm : {reference, lossless, reference}) {
        expected_arm = arm;
        assert(fixture_qmm_run() == baseline);
    }
    expected_arm = fault;
    refused = true;
    int mismatched_calls = 0;
    // The cache, owner, layout and generation match, but the input has
    // changed since lossless. The original Run must not execute.
    assert(fixture_qmm_run_for_generation(1, 49, &mismatched_calls) == 9999);
    assert(mismatched_calls == 0);
    refused = false;
    assert(bridge.remove(callback) == 0);

    uint16_t left[]{0x3f80, 0x4000}, right[]{0x3f80, 0x4000}, out = 0;
    const std::vector<void*> addresses{left, right, &out};
    const std::vector<Slot> slots{{"self", 0, sizeof(left), false, true},
                                 {"mat2", 1, sizeof(right), false, false},
                                 {"out", 2, sizeof(out), true, false}};
    unsigned releases = 0;
    int calls = 0;
    const auto run = [&](const auto& set) {
        assert(set[0] != left && set[1] == right && set[2] == &out);
        ++calls;
        *static_cast<uint16_t*>(set[2]) = 0x40a0;
        return 0;
    };
    const auto transform = [](size_t index, void* host, size_t) {
        assert(index == 0);
        *static_cast<uint16_t*>(host) = 0x3f00;
        return true;
    };
    Audit lossless_receipt, fault_receipt;
    assert(audited_run(lossless, slots, addresses, host_memory(&releases), transform, run, lossless_receipt) == 0);
    assert(lossless_receipt.replacements == 1 && lossless_receipt.changed_slots == 0);
    assert(lossless_receipt.device_copy && lossless_receipt.temporary_readback_verified
           && lossless_receipt.resources_released);
    assert(audited_run(fault, slots, addresses, host_memory(&releases), transform, run, fault_receipt) == 0);
    assert(calls == 2 && releases == 4 && fault_receipt.replacements == 1 && fault_receipt.changed_slots == 1);
    assert(fault_receipt.inputs_before == fault_receipt.inputs_after);
    assert(fault_receipt.inputs_before == lossless_receipt.inputs_before);
    assert(left[0] == 0x3f80 && right[0] == 0x3f80);

    Audit wrong_slot;
    assert(audited_run(lossless, slots, {left, left, &out}, host_memory(), transform, run, wrong_slot) != 0);
    assert(calls == 2 && wrong_slot.original_calls == 0);

    // A successful Run followed by a failed stream sync must retain both
    // buffers because the native stream may still be reading them.
    uint16_t pending_host[2]{}, pending_input[2]{};
    auto pending = host_memory(&releases);
    pending.alloc_host = [&](void** value, size_t) { *value = pending_host; return 0; };
    pending.alloc_device = [&](void** value, size_t) { *value = pending_input; return 0; };
    pending.free_host = pending.free_device = [](void*) { assert(false); return 1; };
    int sync_checks = 0;
    pending.synchronize = [&]() { return sync_checks++; };
    Audit incomplete;
    assert(audited_run(fault, slots, addresses, pending, transform, run, incomplete) != 0);
    assert(incomplete.original_calls == 1 && !incomplete.resources_released);

    Bridge quarantined;
    static Bridge* active = &quarantined;
    const auto callback_with_failure = +[](uint64_t token, const char*) {
        return active->invoke(token, reference);
    };
    assert(quarantined.install(callback_with_failure) == 0);
    assert(quarantined.dispatch("fixture:pending", [](uint32_t, Audit& audit, const auto*) {
        audit.resources_released = false;
        return 1;
    }) != 0);
    assert(quarantined.state(nullptr) == (callback_installed | recovery_required));
    assert(quarantined.remove(callback_with_failure) != 0);
    std::cout << "PASS local C++ callback, lossless, substitutions, OFF and slot rejection\n";
}
#endif
