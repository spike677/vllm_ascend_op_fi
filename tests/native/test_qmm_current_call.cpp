// CPU contract regression. Typed metadata and byte arrays are host fixtures;
// this test does not claim that CANN calls GWS on its cache-hit path.
#include "../../fault_injection/native_adapters/qmm_current_call.h"
#include <cassert>
#include <iostream>

using namespace dcat_current;

Fault20TensorMetadata tensor(uintptr_t identity, void* address, uint64_t bytes = 8) {
    Fault20TensorMetadata value{};
    value.present = 1;
    value.tensor_ptr = identity;
    value.storage_ptr = identity + 100;
    value.raw_addr = value.storage_addr = value.data_addr = reinterpret_cast<uintptr_t>(address);
    value.storage_bytes = bytes;
    return value;
}

Call qmm_call(void* executor, void* input, void* weight, void* output) {
    Call call;
    call.kind = qmm;
    call.executor = executor;
    call.device = 0;
    call.parameter_count = 10;
    call.quant.x1 = tensor(11, input);
    call.quant.x2 = tensor(12, weight);
    call.quant.out = tensor(20, output);
    call.parameters[0] = 11;
    call.parameters[1] = 12;
    call.parameters[9] = 20;
    return call;
}

Call nz_call(void* executor, void* input, void* weight, void* output) {
    Call call;
    call.kind = weight_nz;
    call.executor = executor;
    call.device = 0;
    call.parameter_count = 3;
    call.floating.self = tensor(31, input);
    call.floating.mat2 = tensor(32, weight);
    call.floating.out = tensor(33, output);
    call.floating.all_present = 1;
    call.parameters[0] = 31;
    call.parameters[1] = 32;
    call.parameters[2] = 33;
    return call;
}

int main() {
    int cache = 0, other_cache = 0, executor = 0, other_executor = 0;
    uint64_t input_a = 1, input_b = 2, weight = 3, output_a = 0, output_b = 0;
    Binding binding;
    const auto first = qmm_call(&executor, &input_a, &weight, &output_a);
    auto miss = first;
    miss.add_count = 1;
    miss.cache_created = &cache;
    auto token = binding.begin(qmm);
    assert(binding.publish(token, miss));
    assert(binding.enter(qmm, &executor) == token);
    assert(binding.current(qmm, &cache));
    assert(!binding.current(qmm, &other_cache));
    assert(binding.leave(token, qmm, &executor));

    // The same cached executor is now given entirely current input/output
    // storage. No AddOpExecCache is required on this invocation.
    const auto hit = qmm_call(&executor, &input_b, &weight, &output_b);
    token = binding.begin(qmm);
    assert(binding.publish(token, hit));
    assert(binding.enter(qmm, &executor) == token);
    const auto* current = binding.current(qmm, &cache);
    assert(current && current->add_count == 0);
    const std::vector<void*> addresses{&input_b, &weight, &output_b};
    assert(live_storage_slot(current->quant.x1, 0, addresses));
    assert(live_storage_slot(current->quant.x2, 1, addresses));
    assert(live_storage_slot(current->quant.out, 2, addresses));
    assert(!live_storage_slot(first.quant.x1, 0, addresses));
    assert(!live_storage_slot(first.quant.out, 2, addresses));
    assert(binding.leave(token, qmm, &executor));
    assert(!binding.enter(qmm, &executor)); // never reuse a consumed GWS

    // Wrong executor/operator consume the pending capability instead of
    // allowing a second attempt to find an address that happens to match.
    token = binding.begin(qmm);
    assert(binding.publish(token, hit));
    assert(!binding.enter(qmm, &other_executor));
    assert(!binding.enter(qmm, &executor));
    token = binding.begin(qmm);
    assert(binding.publish(token, hit));
    assert(!binding.enter(weight_nz, &executor));
    assert(!binding.current(qmm, &cache));

    // The typed AddParam order remains mandatory, even if addresses match.
    auto wrong = hit;
    std::swap(wrong.parameters[0], wrong.parameters[1]);
    token = binding.begin(qmm);
    assert(!binding.publish(token, wrong));
    wrong = hit;
    wrong.parameter_count = 0;
    token = binding.begin(qmm);
    assert(!binding.publish(token, wrong));
    wrong = hit;
    wrong.parameters[8] = 99; // optional bias absent, but observation says present
    token = binding.begin(qmm);
    assert(!binding.publish(token, wrong));
    wrong = hit;
    wrong.add_count = 1; // no observed cache object
    token = binding.begin(qmm);
    assert(!binding.publish(token, wrong));

    // Two outstanding GWS calls are ambiguous, even with the same executor.
    token = binding.begin(qmm);
    assert(binding.publish(token, hit));
    auto newer = binding.begin(qmm);
    assert(!binding.publish(newer, hit));
    assert(!binding.enter(qmm, &executor));
    token = binding.begin(qmm);
    assert(binding.publish(token, hit));
    assert(binding.enter(qmm, &executor) == token);
    assert(!binding.enter(qmm, &executor)); // nested Execute invalidates outer admission
    assert(!binding.current(qmm, &cache));
    assert(!binding.leave(0, qmm, &executor));
    assert(binding.leave(token, qmm, &executor));

    token = binding.begin(qmm);
    assert(!binding.begin(qmm)); // nested GWS cannot replace the outer call
    assert(!binding.publish(token, hit));
    token = binding.begin(qmm);
    binding.abort(token);
    assert(!binding.enter(qmm, &executor));

    token = binding.begin(qmm);
    assert(binding.publish(token, hit));
    bool foreign_refused = false;
    std::thread foreign([&] {
        foreign_refused = !binding.enter(qmm, &executor) && !binding.current(qmm, &cache);
    });
    foreign.join();
    assert(foreign_refused);
    assert(binding.enter(qmm, &executor) == token);
    assert(!binding.begin(qmm)); // nested GWS during Execute poisons admission
    assert(!binding.current(qmm, &cache));
    assert(binding.leave(token, qmm, &executor));

    // Exact live pointer equality and uniqueness have not been weakened.
    assert(!live_storage_slot(hit.quant.x1, 1, addresses));
    auto aliased = addresses;
    aliased.push_back(&input_b);
    assert(!live_storage_slot(hit.quant.x1, 0, aliased));
    auto offset = hit.quant.x1;
    offset.view_offset = 1;
    assert(!live_storage_slot(offset, 0, addresses));
    auto absent = hit.quant.x1;
    absent.storage_ptr = 0;
    assert(!live_storage_slot(absent, 0, addresses));

    // WeightNz uses its own exact three-formal contract, not the QMM layout.
    auto nz = nz_call(&executor, &input_b, &weight, &output_b);
    token = binding.begin(weight_nz);
    assert(binding.publish(token, nz));
    assert(binding.enter(weight_nz, &executor) == token);
    current = binding.current(weight_nz, &cache);
    assert(current && live_storage_slot(current->floating.out, 2, addresses));
    assert(binding.leave(token, weight_nz, &executor));
    nz.floating.any_shared_storage = 1;
    token = binding.begin(weight_nz);
    assert(!binding.publish(token, nz));

    // The private observer reports actual current shape/storage/address facts
    // verbatim without changing eligibility or conflating them with a hash.
    auto observed = tensor(31, &input_b, 8);
    observed.dtype = 1; observed.format = 2; observed.contiguous = 1;
    observed.view_rank = observed.stride_rank = 2; observed.storage_rank = 1;
    observed.view_shape[0] = 1; observed.view_shape[1] = 4;
    observed.strides[0] = 4; observed.strides[1] = 1; observed.storage_shape[0] = 4;
    observed.element_size = 2; observed.view_bytes = 8;
    const auto raw = tensor_observation("self", observed, 0, reinterpret_cast<uintptr_t>(&input_b));
    assert(raw.find("\"present\":true") != std::string::npos);
    assert(raw.find("\"view_shape\":[1,4]") != std::string::npos);
    assert(raw.find("\"strides\":[4,1]") != std::string::npos);
    assert(raw.find("\"storage_shape\":[4]") != std::string::npos);
    assert(raw.find("\"run_address\":" + std::to_string(reinterpret_cast<uintptr_t>(&input_b))) != std::string::npos);
    Fault20TensorMetadata absent_formal{};
    assert(tensor_observation("bias", absent_formal, -1, 0).find("\"present\":false") != std::string::npos);
    observed.view_rank = FAULT20_MAX_DIMS + 1;
    assert(tensor_observation("self", observed, 0, 1) == "{}");

    std::cout << "PASS CPU current typed GWS/executor scope, cache address refresh, stale/nested/foreign rejection and live slot equality\n";
}
