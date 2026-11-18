// CPU host memory behind explicit ACL stubs. This is NOT an NPU test.
// g++ -std=c++17 -pthread -I "$SDK/include" test_original_snapshot.cpp -o test-original
#include "postwrite.cpp"
#include <cassert>
#include <iostream>

namespace { unsigned acl_calls = 0; }
extern "C" aclError aclrtGetDevice(int32_t *device) {
    ++acl_calls; *device = 0; return ACL_SUCCESS;
}
extern "C" aclError aclrtSynchronizeDevice() { ++acl_calls; return ACL_SUCCESS; }
extern "C" aclError aclrtMemcpy(void *dst, size_t capacity, const void *src,
                                 size_t bytes, aclrtMemcpyKind) {
    ++acl_calls; assert(bytes <= capacity); std::memcpy(dst, src, bytes); return ACL_SUCCESS;
}

int main() {
    constexpr uint64_t owner = 5, generation = 7;
    uint16_t tensor[] = {0x3f80, 0x8000, 0x7f7f, 0x0001};
    uint64_t handle = 0, changed = 99, index = 0;
    assert(dcat_pw_open(tensor, sizeof(tensor), 0, owner, generation, &handle) == 0);
    const auto original = leases.at(handle).original;
    uint8_t host[sizeof(tensor)] = {};
    dcat_pw_original_info_v1 info{};
    auto before = acl_calls;
    assert(dcat_pw_copy_original_v1(handle, owner, generation, host, sizeof(host), &info) == 0);
    assert(acl_calls == before && leases.size() == 1 && !leases.at(handle).applied);
    assert(info.version == 1 && info.little_endian_bf16 == 1 && info.applied == 0);
    assert(info.handle == handle && info.owner == owner && info.generation == generation);
    assert(info.tensor_address == reinterpret_cast<uintptr_t>(tensor));
    assert(info.elements == 4 && info.span_bytes == sizeof(tensor) && info.device == 0);
    for (size_t i = 0; i < original.size(); ++i)
        assert((host[2*i] | uint16_t(host[2*i+1]) << 8) == original[i]);
    host[0] = 0;  // A caller-owned copy cannot change lease.original.
    assert(leases.at(handle).original == original);

    // Missing/foreign/other-thread handles, wrong sizes and all alias paths refuse.
    const auto saved_info = info;
    const auto host_before = std::vector<uint8_t>(host, host+sizeof(host));
    assert(dcat_pw_copy_original_v1(handle+1, owner, generation, host, sizeof(host), &info) == 2);
    assert(dcat_pw_copy_original_v1(handle, owner+1, generation, host, sizeof(host), &info) == 2);
    assert(dcat_pw_copy_original_v1(handle, owner, generation+1, host, sizeof(host), &info) == 2);
    assert(dcat_pw_copy_original_v1(handle, owner, generation, host, sizeof(host)-1, &info) == 1);
    assert(dcat_pw_copy_original_v1(handle, owner, generation, nullptr, sizeof(host), &info) == 1);
    assert(dcat_pw_copy_original_v1(handle, owner, generation, host, sizeof(host), nullptr) == 1);
    assert(dcat_pw_copy_original_v1(handle, owner, generation, tensor, sizeof(tensor), &info) == 14);
    assert(dcat_pw_copy_original_v1(handle, owner, generation, leases.at(handle).original.data(),
                                  sizeof(tensor), &info) == 14);
    assert(dcat_pw_copy_original_v1(handle, owner, generation, leases.at(handle).expected.data(),
                                  sizeof(tensor), &info) == 14);
    alignas(dcat_pw_original_info_v1) uint8_t both[sizeof(dcat_pw_original_info_v1)] = {};
    assert(dcat_pw_copy_original_v1(handle, owner, generation, both, sizeof(host),
        reinterpret_cast<dcat_pw_original_info_v1 *>(both)) == 14);
    std::thread foreign([&] {
        assert(dcat_pw_copy_original_v1(handle, owner, generation, host, sizeof(host), &info) == 2);
    });
    foreign.join();
    assert(std::memcmp(&info, &saved_info, sizeof(info)) == 0);
    assert(host_before == std::vector<uint8_t>(host, host+sizeof(host)));
    assert(acl_calls == before);

    // Successful apply writes Inf into live tensor; snapshot remains finite original.
    assert(dcat_pw_apply(handle, owner, generation, 1, &index, 1, &changed) == 0);
    assert(changed == 1 && tensor[0] == 0x7f80);
    before = acl_calls;
    assert(dcat_pw_copy_original_v1(handle, owner, generation, host, sizeof(host), &info) == 0);
    assert(acl_calls == before && info.applied == 1 && host[1] == 0x3f);
    assert(leases.at(handle).original == original && leases.at(handle).expected[0] == 0x7f80);
    assert(dcat_pw_restore(handle, owner, generation) == 0);

    // External change is still return5, not return4 based on the fresh tensor.
    tensor[2] = 0x7fc0;
    before = acl_calls;
    assert(dcat_pw_copy_original_v1(handle, owner, generation, host, sizeof(host), &info) == 0);
    assert(acl_calls == before && host[4] == 0x7f && host[5] == 0x7f);
    assert(dcat_pw_apply(handle, owner, generation, 1, &index, 1, &changed) == 5);
    tensor[2] = original[2];
    assert(dcat_pw_release(handle, owner, generation) == 0);
    assert(dcat_pw_copy_original_v1(handle, owner, generation, host, sizeof(host), &info) == 2);

    // Same original nonfinite bits cause unchanged return4 BEFORE any ACL fresh read.
    uint16_t nonfinite[] = {0x3f80, 0x7f80, 0xff80, 0x7fc1, 0xff81};
    uint8_t nonfinite_host[sizeof(nonfinite)] = {};
    assert(dcat_pw_open(nonfinite, sizeof(nonfinite), 0, owner, generation, &handle) == 0);
    before = acl_calls; changed = 99;
    assert(dcat_pw_apply(handle, owner, generation, 1, &index, 1, &changed) == 4);
    assert(changed == 99 && acl_calls == before); // No changed-elements comparison ran.
    assert(dcat_pw_copy_original_v1(handle, owner, generation, nonfinite_host,
                                  sizeof(nonfinite_host), &info) == 0);
    assert(acl_calls == before && info.applied == 0);
    for (size_t i = 0; i < 5; ++i)
        assert((nonfinite_host[2*i] | uint16_t(nonfinite_host[2*i+1]) << 8) == nonfinite[i]);
    assert(dcat_pw_restore(handle, owner, generation) == 0);
    assert(dcat_pw_release(handle, owner, generation) == 0);
    assert(leases.empty() && live_span_bytes == 0);
    std::cout << "{\"status\":\"PASS_CPU_LEASE_ORIGINAL_DIAGNOSTIC\",\"npu_compute\":false}\n";
}
