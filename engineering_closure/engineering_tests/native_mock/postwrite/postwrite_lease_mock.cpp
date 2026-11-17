// Link the production lease registry against CPU-only fake ACL transfers.
#include "postwrite.h"
#include <acl/acl.h>
#include <array>
#include <cassert>
#include <cstring>
#include <iostream>
#include <thread>
#include <vector>

namespace {
thread_local int32_t current_device = 0;
bool fail_next_sync = false;
bool fail_partial_write = false;
constexpr uint64_t owner = 17;
constexpr uint64_t generation = 3;
constexpr uint64_t zero = 0;
}

int aclrtGetDevice(int32_t *device) {
    *device = current_device;
    return ACL_SUCCESS;
}

int aclrtSynchronizeDevice() {
    if (!fail_next_sync) return ACL_SUCCESS;
    fail_next_sync = false;
    return 1;
}

int aclrtMemcpy(void *destination, size_t capacity, const void *source, size_t bytes, int direction) {
    assert(direction == ACL_MEMCPY_HOST_TO_DEVICE || direction == ACL_MEMCPY_DEVICE_TO_HOST);
    assert(capacity >= bytes);
    if (direction == ACL_MEMCPY_HOST_TO_DEVICE && fail_partial_write) {
        fail_partial_write = false;
        std::memcpy(destination, source, sizeof(uint16_t));
        return 1;
    }
    std::memcpy(destination, source, bytes);
    return ACL_SUCCESS;
}

int main() {
    std::array<uint16_t, 6> value{0x3f80, 0x4000, 0x4040, 0x4080, 0x40a0, 0x40c0};
    const auto baseline = value;
    uint64_t first = 0, second = 0, extra = 0, changed = 0;
    assert(dcat_pw_abi() == 1);
    assert(dcat_pw_open(value.data(), 4, 0, owner, generation, &first) == 0);
    assert(dcat_pw_open(value.data() + 1, 4, 0, owner, generation, &extra) == 14);
    assert(dcat_pw_open(value.data() + 2, 4, 0, owner, generation + 1, &extra) == 2);
    assert(dcat_pw_open(value.data() + 2, 4, 0, owner + 1, generation, &extra) == 2);
    assert(dcat_pw_open(value.data() + 2, 4, 1, owner, generation, &extra) == 2);
    assert(dcat_pw_open(value.data() + 2, 4, 0, owner, generation, &second) == 0);
    assert(first != second);
    std::thread foreign([&] {
        assert(dcat_pw_live(first, owner, generation) == -1);
        assert(dcat_pw_apply(first, owner, generation, 1, &zero, 1, &changed) == 2);
        assert(dcat_pw_restore(first, owner, generation) == 2);
        assert(dcat_pw_release(first, owner, generation) == 2);
    });
    foreign.join();
    assert(value == baseline);
    assert(dcat_pw_apply(first, owner, generation, 1, &zero, 1, &changed) == 0 && changed == 1);
    assert(dcat_pw_apply(second, owner, generation, 4, &zero, 1, &changed) == 0 && changed == 1);
    assert(value[0] == 0x7f80 && value[2] == (baseline[2] ^ 0x8000));
    assert(value[1] == baseline[1] && value[3] == baseline[3] && value[4] == baseline[4]);
    assert(dcat_pw_release(first, owner, generation) == 2);

    // A failed drain retains the exact lease. A separate disjoint lease can
    // still be restored, and an external writer is never silently overwritten.
    fail_next_sync = true;
    assert(dcat_pw_restore(first, owner, generation) == 11);
    assert(dcat_pw_live(first, owner, generation) == 1);
    assert(dcat_pw_restore(second, owner, generation) == 0);
    assert(dcat_pw_release(second, owner, generation) == 0);
    assert(dcat_pw_live(second, owner, generation) == 0);
    value[1] = 0x7fc0;
    assert(dcat_pw_restore(first, owner, generation) == 5);
    assert(value[1] == 0x7fc0);
    value[1] = baseline[1];
    assert(dcat_pw_restore(first, owner, generation) == 0);
    assert(dcat_pw_release(first, owner, generation) == 0);
    assert(dcat_pw_live(first, owner, generation) == 0);
    assert(value == baseline);
    assert(dcat_pw_open(value.data(), 4, 0, owner, generation + 1, &extra) == 0);
    assert(first != extra && dcat_pw_live(first, owner, generation) == 0);
    assert(dcat_pw_restore(extra, owner, generation) == 2);
    assert(dcat_pw_release(extra, owner, generation + 1) == 0);

    assert(dcat_pw_open(value.data(), 4, 0, owner, generation + 1, &extra) == 0);
    std::array<uint64_t, 2> both{0, 1};
    fail_partial_write = true;
    assert(dcat_pw_apply(extra, owner, generation + 1, 1, both.data(), both.size(), &changed) == 13);
    assert(value[0] == 0x7f80 && value[1] == baseline[1]);
    assert(dcat_pw_restore(extra, owner, generation + 1) == 5);
    assert(dcat_pw_release(extra, owner, generation + 1) == 2);
    value[0] = baseline[0];
    assert(dcat_pw_restore(extra, owner, generation + 1) == 0);
    assert(dcat_pw_release(extra, owner, generation + 1) == 0);

    const auto capacity = dcat_pw_lease_capacity();
    assert(capacity == 256);
    assert(dcat_pw_lease_byte_capacity() == 160 * 1024 * 1024);
    std::vector<uint16_t> slots(capacity + 1, 0x3f80);
    std::vector<uint64_t> handles(capacity);
    for (uint32_t i = 0; i < capacity; ++i) {
        assert(dcat_pw_open(&slots[i], 2, 0, owner, generation + 1, &handles[i]) == 0);
    }
    assert(dcat_pw_open(&slots.back(), 2, 0, owner, generation + 1, &extra) == 15);
    for (uint64_t handle : handles) {
        assert(dcat_pw_release(handle, owner, generation + 1) == 0);
        assert(dcat_pw_live(handle, owner, generation + 1) == 0);
    }
    std::cout << "postwrite native-mock lease checks passed\n";
}
