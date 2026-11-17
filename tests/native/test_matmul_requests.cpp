// Owned host arrays/CPU Run only. No vendor library, model, or NPU acceptance.
#define DCAT_TEST_SHARED
#include "test_matmul_auto.cpp"

namespace {
constexpr const char* lease = "real-lease-fixture";
uint64_t generation = 17;
uint32_t arm = original;
bool refusal = false;
unsigned callbacks = 0;
std::vector<std::string> packets;

uint32_t callback(uint64_t token, const char* raw) {
    ++callbacks;
    packets.emplace_back(raw);
    const auto flags = bridge.request_state(callback, lease, generation);
    assert(flags == (callback_installed | callback_owned | call_inflight | request_controlled));
    // Lifecycle changes cannot occur inside the active native continuation.
    assert(bridge.pause(callback, lease, generation) != 0);
    assert(bridge.resume(callback, lease, generation) != 0);
    assert(bridge.remove_request(callback, lease, generation) != 0);
    std::thread foreign([&] {
        assert(bridge.pause(callback, lease, generation) != 0);
        assert(bridge.resume(callback, lease, generation) != 0);
        assert(bridge.remove_request(callback, lease, generation) != 0);
        assert(bridge.invoke(token, arm) != 0);
    });
    foreign.join();
    auto status = bridge.invoke(token, arm);
    if (refusal) { assert(status); return 1; }
    assert(status == 0);
    return 0;
}

void idle() {
    assert(bridge.pause(callback, lease, generation) == 0);
    assert(bridge.pause(callback, lease, generation) == 0);
    assert(bridge.request_state(callback, lease, generation) ==
           (callback_installed | callback_owned | callback_paused | request_controlled));
    assert(bridge.owned() && !bridge.on_owner());
    const auto before = callbacks;
    assert(fixture_qmm_run_current(999) == -80);
    assert(callbacks == before);
    assert(bridge.invoke(1, fault) != 0);
    assert(bridge.dispatch("forbidden", [](uint32_t, Audit&, const auto*) {
        assert(false); return 1;
    }) != 0);
}

int stage(uint32_t next, uint64_t token, uint64_t incarnation = 1,
          int8_t input = 37, int* calls = nullptr) {
    arm = next;
    assert(bridge.resume(callback, lease, generation) == 0);
    assert(bridge.resume(callback, lease, generation) != 0);
    assert(bridge.remove_request(callback, lease, generation) != 0);
    const int output = fixture_qmm_run_for_generation(incarnation, input, calls, token);
    idle();
    return output;
}
}

int main() {
    assert(bridge.install_request(callback, "", 1) != 0);
    assert(bridge.install_request(callback, lease, 0) != 0);
    assert(bridge.install_request(callback, lease, generation) == 0);
    idle();
    assert(bridge.install_request(callback, lease, generation) != 0);
    assert(bridge.install_request(callback, "foreign", generation + 1) != 0);
    assert(bridge.remove(callback) != 0);
    for (auto wrong : {generation - 1, generation + 1}) {
        assert(bridge.pause(callback, lease, wrong) != 0);
        assert(bridge.resume(callback, lease, wrong) != 0);
        assert(bridge.remove_request(callback, lease, wrong) != 0);
        assert(!(bridge.request_state(callback, lease, wrong) & callback_owned));
    }
    assert(bridge.resume(callback, "foreign", generation) != 0);
    std::thread foreign([] { assert(bridge.resume(callback, lease, generation) != 0); });
    foreign.join();

    assert(stage(reference, 101) == -80);
    assert(stage(lossless, 102) == -80);
    assert(stage(reference, 103) == -80);
    assert(packets.size() == 3 && packets[0] != packets[1] && packets[1] != packets[2]);
    assert(packets[0].find("\"gws_token\":101") != std::string::npos);
    // A baseline comparison phase is another exact, read-only observation.
    assert(stage(reference, 1031) == -80);
    assert(stage(reference, 1032) == -80);
    assert(stage(fault, 104) != -80);
    refusal = true;
    assert(stage(fault, 105) == 9999);
    refusal = false;
    // Fresh reference/lossless cannot replenish an already-spent fault lease.
    assert(stage(reference, 106) == -80);
    assert(stage(lossless, 107) == -80);
    assert(stage(reference, 108) == -80);
    assert(stage(reference, 1081) == -80);
    refusal = true;
    assert(stage(fault, 109) == 9999);
    refusal = false;
    assert(bridge.remove_request(callback, lease, generation) == 0);
    assert(bridge.request_state(callback, lease, generation) == 0);

    ++generation;
    assert(bridge.install_request(callback, lease, generation) == 0);
    assert(stage(reference, 201) == -80);
    assert(stage(lossless, 202) == -80);
    assert(stage(reference, 203) == -80);
    // Cache incarnation/geometry identity remains part of the stable target.
    refusal = true;
    int runs = 0;
    assert(stage(fault, 204, 2, 37, &runs) == 9999 && runs == 0);
    refusal = false;
    assert(bridge.remove_request(callback, lease, generation) == 0);

    ++generation;
    assert(bridge.install_request(callback, lease, generation) == 0);
    assert(stage(reference, 301) == -80);
    assert(stage(lossless, 302) == -80);
    assert(stage(reference, 303) == -80);
    refusal = true;
    assert(stage(fault, 304, 1, 49, &runs) == 9999 && runs == 0);
    refusal = false;
    assert(bridge.remove_request(callback, lease, generation) == 0);
    assert(fixture_qmm_run() == -80);
    ++generation;
    assert(bridge.install_request(callback, lease, generation) == 0);
    assert(stage(reference, 401) == -80);
    assert(stage(lossless, 402) == -80);
    assert(stage(reference, 403) == -80);
    refusal = true;
    runs = 0;
    assert(stage(reference, 404, 1, 49, &runs) == 9999 && runs == 1);
    // Changed read-only input invalidates qualification; it never admits fault.
    assert(stage(fault, 405, 1, 37, &runs) == 9999 && runs == 1);
    refusal = false;
    assert(bridge.remove_request(callback, lease, generation) == 0);
    Bridge pending;
    static Bridge* unsafe = &pending;
    const auto fail = +[](uint64_t token, const char*) { return unsafe->invoke(token, reference); };
    assert(pending.install_request(fail, "pending", 1) == 0);
    assert(pending.resume(fail, "pending", 1) == 0);
    assert(pending.dispatch("pending", [](uint32_t, Audit& audit, const auto*) {
        audit.resources_released = false;
        return 1;
    }) != 0);
    assert(pending.request_state(fail, "pending", 1) & recovery_required);
    assert(pending.pause(fail, "pending", 1) != 0);
    assert(pending.resume(fail, "pending", 1) != 0);
    assert(pending.remove_request(fail, "pending", 1) != 0);
    assert(pending.on_owner()); // must reach hard refusal, never unowned passthrough
    assert(pending.dispatch("quarantined", [](uint32_t, Audit&, const auto*) {
        assert(false); return 1;
    }) != 0);
    std::cout << "PASS CPU request owner/generation, paused non-execution, qualification retention, stable/current separation, one-fault, identity drift and OFF\n";
}
