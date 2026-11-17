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

namespace group_contract {
const char* owner = "cpu-group-owner";
uint64_t epoch = 100;
uint32_t selected_arm = original;
std::string key(64, 'a'), receipt;
const std::string other_key(64, 'b');
uint32_t callback(uint64_t token, const char*) {
    // Neither a wrong thread nor the legacy unkeyed ABI can use a v3 lease.
    assert(bridge.invoke(token, selected_arm) != 0);
    std::thread foreign([&] {
        assert(bridge.invoke_group(token, selected_arm, key.c_str()) != 0);
        const auto state = bridge.group_state(callback, owner, epoch);
        assert(state.find("\"inflight\":true") != std::string::npos);
        assert(state.find("\"owned\":false") != std::string::npos);
        assert(state.find("\"entries\"") == std::string::npos);
    });
    foreign.join();
    const auto status = bridge.invoke_group(token, selected_arm, key.c_str());
    const char* raw = bridge.receipt(token);
    receipt = raw ? raw : "";
    return status;
}
int stage(uint32_t arm, const std::string& target, int8_t input = 37, uint64_t lifetime = 1, int* calls = nullptr) {
    selected_arm = arm;
    key = target;
    assert(bridge.resume(callback, owner, epoch) == 0);
    const auto output = fixture_qmm_run_for_generation(lifetime, input, calls, 10000 + epoch);
    assert(bridge.pause(callback, owner, epoch) == 0);
    return output;
}
std::string group_hash(const std::vector<std::string>& targets) {
    std::string keys;
    for (const auto& target : targets) keys += target + '\n';
    return f39identity::digest(keys.data(), keys.size());
}
void prepare(uint32_t fault_limit = 1) {
    ++epoch;
    assert(bridge.install_group_request(callback, owner, epoch, 2, fault_limit, 100) == 0);
    const std::string first_key(64, 'a');
    for (auto arm : {reference, lossless, reference}) {
        assert(stage(arm, first_key) == -80);
        assert(stage(arm, other_key) == -80);
    }
    const auto hash = group_hash({first_key, other_key});
    assert(bridge.seal_group(callback, owner, epoch + 1, 2, hash.c_str()) != 0);
    assert(bridge.seal_group(callback, owner, epoch, 1, hash.c_str()) != 0);
    assert(bridge.seal_group(callback, owner, epoch, 2, hash.c_str()) == 0);
    assert(bridge.seal_group(callback, owner, epoch, 2, hash.c_str()) != 0);
}
void run() {
    const std::string first_key(64, 'a'), added_key(64, 'c');
    assert(bridge.install_group_request(callback, owner, epoch, 0, 1, 100) != 0);
    assert(bridge.install_group_request(callback, owner, epoch, 2, 0, 100) != 0);
    assert(bridge.install_group_request(callback, owner, epoch, 2, 1, 1) != 0);
    prepare();
    int calls = 0;
    assert(stage(fault, first_key, 37, 1, &calls) != -80 && calls == 1);
    // A downstream target's actual propagated input is independently audited;
    // this does not change the original C1 input-equality contract above.
    const auto propagated = stage(fault, other_key, 49, 1, &calls);
    assert(propagated != 9999 && propagated != -68 && calls == 2);
    assert(receipt.find("\"input_equal_healthy_reference\":false") != std::string::npos);
    assert(stage(reference, first_key) == -80);
    assert(stage(reference, other_key) == -80);
    assert(stage(fault, first_key, 37, 1, &calls) == 9999 && calls == 2);
    assert(bridge.group_state(callback, owner, epoch).find("\"failed\":true") != std::string::npos);
    assert(bridge.remove_request(callback, owner, epoch) == 0);
    assert(bridge.group_state(callback, owner, epoch).find("\"targets\":0") != std::string::npos);

    prepare(2);
    for (int request = 0; request < 2; ++request) {
        assert(stage(fault, first_key) != 9999);
        assert(stage(fault, other_key) != 9999);
    }
    assert(stage(reference, first_key) == -80 && stage(reference, other_key) == -80);
    assert(stage(fault, first_key) == 9999); // immutable bounded every-match budget
    assert(bridge.remove_request(callback, owner, epoch) == 0);

    prepare();
    calls = 0;
    assert(stage(fault, first_key, 37, 2, &calls) == 9999 && calls == 0); // registration reuse
    assert(stage(fault, other_key, 37, 1, &calls) == 9999 && calls == 0); // group refuses further faults
    assert(stage(reference, first_key) == -80 && stage(reference, other_key) == -80); // OFF still available
    assert(bridge.remove_request(callback, owner, epoch) == 0);

    prepare();
    assert(stage(reference, added_key) == 9999); // no expansion of sealed targets
    assert(bridge.remove_request(callback, owner, epoch) == 0);

    RunInventory observer(1);
    observer.begin_run(); observer.non_target();
    observer.begin_run(); observer.reject(1, "CURRENT_SLOT_UNPROVEN");
    observer.begin_run(); observer.eligible("{\"cache_lifecycle\":1}", 10);
    observer.begin_run(); observer.eligible("{\"cache_lifecycle\":1}", 11);
    auto observed = observer.snapshot();
    assert(observed.find("\"complete\":true") != std::string::npos);
    assert(observed.find("\"observations\":2") != std::string::npos);
    observer.begin_run(); observer.eligible("{\"cache_lifecycle\":2}", 12);
    observed = observer.snapshot();
    assert(observed.find("\"total_run_calls\":5") != std::string::npos);
    assert(observed.find("\"eligible_run_calls\":3") != std::string::npos);
    assert(observed.find("\"overflow_calls\":1") != std::string::npos);
    assert(observed.find("\"complete\":false") != std::string::npos);
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
    group_contract::run();
    std::cout << "PASS CPU request owner/generation, C1 unchanged, bounded v3 target groups, independent inventory overflow, propagated inputs, sealed targets and OFF\n";
}
