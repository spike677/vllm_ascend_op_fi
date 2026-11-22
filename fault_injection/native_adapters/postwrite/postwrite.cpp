#include "postwrite.h"
#include <acl/acl.h>
#include <algorithm>
#include <cstring>
#include <limits>
#include <map>
#include <mutex>
#include <thread>
#include <vector>

namespace {
struct Lease {
    void *tensor;
    uint64_t owner, generation;
    int32_t device;
    std::thread::id thread;
    std::vector<uint16_t> original, expected;
    bool applied = false;
    aclrtStream stream = nullptr;
};
std::mutex guard;
std::map<uint64_t, Lease> leases;
uint64_t serial = 0;
uint64_t live_span_bytes = 0;
constexpr uint64_t max_bytes = 1024 * 1024;
constexpr uint32_t max_leases = 256;
// Original and expected BF16 snapshots consume twice this payload budget.
constexpr uint64_t max_span_bytes = 160 * 1024 * 1024;
bool owns(const Lease &lease, uint64_t owner, uint64_t generation) {
    return lease.owner == owner && lease.generation == generation &&
        lease.thread == std::this_thread::get_id();
}
Lease *owned(uint64_t handle, uint64_t owner, uint64_t generation) {
    const auto found = leases.find(handle);
    return found != leases.end() && owns(found->second, owner, generation) ? &found->second : nullptr;
}
int sync(int32_t expected_device, aclrtStream stream = nullptr) {
    int32_t device = -1;
    if (aclrtGetDevice(&device) != ACL_SUCCESS || device != expected_device) return 10;
    return (stream ? aclrtSynchronizeStream(stream) : aclrtSynchronizeDevice()) == ACL_SUCCESS ? 0 : 11;
}
int read(const Lease &l, std::vector<uint16_t> &out) {
    const int result = sync(l.device, l.stream);
    if (result) return result;
    out.resize(l.original.size());
    const auto bytes = out.size() * sizeof(uint16_t);
    return aclrtMemcpy(out.data(), bytes, l.tensor, bytes, ACL_MEMCPY_DEVICE_TO_HOST)
        == ACL_SUCCESS ? 0 : 12;
}
int write(const Lease &l, const std::vector<uint16_t> &values) {
    const int result = sync(l.device, l.stream);
    if (result) return result;
    const auto bytes = values.size() * sizeof(uint16_t);
    if (aclrtMemcpy(l.tensor, bytes, values.data(), bytes, ACL_MEMCPY_HOST_TO_DEVICE)
            != ACL_SUCCESS) return 13;
    return sync(l.device, l.stream);
}
bool finite(uint16_t bits) { return (bits & 0x7f80U) != 0x7f80U; }
}
extern "C" uint32_t dcat_pw_abi(void) { return 1; }
extern "C" int dcat_pw_linked_address_v1(uint32_t slot,uint64_t* output) {
    if(!output)return 1;
    *output=0;
    // These are the direct linked relocations used above, not a new dlsym.
    const uintptr_t values[]={reinterpret_cast<uintptr_t>(&aclrtGetDevice),
        reinterpret_cast<uintptr_t>(&aclrtSynchronizeDevice),reinterpret_cast<uintptr_t>(&aclrtMemcpy)};
    if(slot>=sizeof(values)/sizeof(values[0]))return 2;
    if(!values[slot])return 3;
    *output=values[slot];return 0;
}
extern "C" uint32_t dcat_pw_lease_capacity(void) { return max_leases; }
extern "C" uint64_t dcat_pw_lease_byte_capacity(void) { return max_span_bytes; }
extern "C" int dcat_pw_read_span(const void *tensor, uint64_t bytes, int32_t device, void *host) {
    std::lock_guard<std::mutex> lock(guard);
    if (!tensor || !host || device < 0 || bytes == 0 || bytes > max_bytes) return 1;
    const int result = sync(device);
    if (result) return result;
    return aclrtMemcpy(host, bytes, tensor, bytes, ACL_MEMCPY_DEVICE_TO_HOST) == ACL_SUCCESS ? 0 : 12;
}
static int open_span(void *tensor, uint64_t bytes, int32_t device,
                     uint64_t owner, uint64_t generation, uint64_t *handle, aclrtStream stream) {
    std::lock_guard<std::mutex> lock(guard);
    if (!tensor || !handle || !owner || !generation || device < 0 ||
        bytes == 0 || bytes > max_bytes || bytes % 2) return 1;
    const auto begin = reinterpret_cast<uintptr_t>(tensor);
    if (bytes > std::numeric_limits<uintptr_t>::max() - begin) return 1;
    if (leases.size() == max_leases || bytes > max_span_bytes - live_span_bytes ||
        serial == std::numeric_limits<uint64_t>::max()) return 15;
    for (const auto &entry : leases) {
        const Lease &lease = entry.second;
        if (!owns(lease, owner, generation) || lease.device != device) return 2;
        const auto start = reinterpret_cast<uintptr_t>(lease.tensor);
        const auto end = start + lease.original.size() * sizeof(uint16_t);
        if (begin < end && start < begin + bytes) return 14;
    }
    try {
        Lease next{tensor, owner, generation, device, std::this_thread::get_id(), {}, {}, false, stream};
        next.original.resize(bytes / 2);
        std::vector<uint16_t> observed;
        const int result = read(next, observed);
        if (result) return result;
        next.original = observed; next.expected = observed;
        leases.emplace(++serial, std::move(next));
        live_span_bytes += bytes;
        *handle = serial;
        return 0;
    } catch (...) { return 90; }
}
extern "C" int dcat_pw_open(void *tensor, uint64_t bytes, int32_t device,
                              uint64_t owner, uint64_t generation, uint64_t *handle) {
    return open_span(tensor, bytes, device, owner, generation, handle, nullptr);
}
extern "C" int dcat_pw_open_stream_v2(void *tensor, uint64_t bytes, int32_t device,
                              uint64_t owner, uint64_t generation, uint64_t stream, uint64_t *handle) {
    if (!stream) return 1;
    return open_span(tensor, bytes, device, owner, generation, handle,
                     reinterpret_cast<aclrtStream>(static_cast<uintptr_t>(stream)));
}
/* Copy only between already-open owned live objects. Never a new raw-address input. */
extern "C" int dcat_pw_copy_region_v2(uint64_t target, uint64_t source,
        uint64_t owner, uint64_t generation, uint64_t *changed) {
    std::lock_guard<std::mutex> lock(guard);
    Lease *dst = owned(target, owner, generation);
    Lease *src = owned(source, owner, generation);
    if (!dst || !src || dst == src || !changed || dst->applied || src->applied ||
        !dst->stream || dst->stream != src->stream || dst->device != src->device ||
        dst->original.size() != src->original.size()) return 2;
    try {
        std::vector<uint16_t> source_now, target_now;
        int result = read(*src, source_now); if (result) return result;
        result = read(*dst, target_now); if (result) return result;
        if (source_now != src->original || target_now != dst->original) return 5;
        *changed = 0;
        for (size_t i=0; i<source_now.size(); ++i) *changed += source_now[i] != target_now[i];
        dst->applied = true; dst->expected = source_now;
        result = write(*dst, source_now); if (result) return result;
        result = read(*dst, target_now); if (result) return result;
        return target_now == source_now ? 0 : 6;
    } catch (...) { return 90; }
}
extern "C" int dcat_pw_apply(uint64_t handle, uint64_t owner, uint64_t generation,
                               uint32_t mode, const uint64_t *indices, uint64_t count,
                               uint64_t *changed) {
    std::lock_guard<std::mutex> lock(guard);
    Lease *live = owned(handle, owner, generation);
    if (!live || !changed || live->applied || mode > 5 ||
        !indices || count == 0 || count > live->original.size()) return 2;
    try {
        auto selected = std::vector<uint64_t>(indices, indices + count);
        std::sort(selected.begin(), selected.end());
        if (selected.back() >= live->original.size() ||
            std::adjacent_find(selected.begin(), selected.end()) != selected.end()) return 3;
        if (!std::all_of(live->original.begin(), live->original.end(), finite)) return 4;
        std::vector<uint16_t> observed;
        int result = read(*live, observed);
        if (result) return result;
        if (observed != live->original) return 5; // Another writer owns this state now.
        auto replacement = live->original;
        *changed = 0;
        for (uint64_t i : selected) {
            uint16_t bits = replacement[i];
            if (mode == 1) bits = 0x7f80U;
            else if (mode == 2) bits = 0xff80U;
            else if (mode == 3) bits = 0x7fc0U;
            else if (mode == 4) bits ^= 0x8000U;
            else if (mode == 5) bits = 0;
            if (bits != replacement[i]) ++*changed;
            replacement[i] = bits;
        }
        /* Mark before writing: a failed transfer may have written some bytes.
           Retain the lease rather than reporting release or blind recovery. */
        live->applied = true; live->expected = replacement;
        result = write(*live, replacement);
        if (result) return result;
        result = read(*live, observed);
        if (result) return result;
        return observed == replacement ? 0 : 6;
    } catch (...) { return 90; }
}
extern "C" int dcat_pw_restore(uint64_t handle, uint64_t owner, uint64_t generation) {
    std::lock_guard<std::mutex> lock(guard);
    Lease *live = owned(handle, owner, generation);
    if (!live) return 2;
    try {
        std::vector<uint16_t> observed;
        int result = read(*live, observed);
        if (result) return result;
        if (observed != live->expected && observed != live->original) return 5;
        result = write(*live, live->original);
        if (result) return result;
        result = read(*live, observed);
        if (result) return result;
        if (observed != live->original) return 6;
        live->expected = live->original; live->applied = false;
        return 0;
    } catch (...) { return 90; }
}
extern "C" int dcat_pw_release(uint64_t handle, uint64_t owner, uint64_t generation) {
    std::lock_guard<std::mutex> lock(guard);
    Lease *live = owned(handle, owner, generation);
    if (!live || live->applied) return 2;
    try {
        std::vector<uint16_t> observed;
        const int result = read(*live, observed);
        if (result) return result;
        if (observed != live->original) return 5;
        live_span_bytes -= live->original.size() * sizeof(uint16_t);
        leases.erase(handle); return 0;
    } catch (...) { return 90; }
}
extern "C" int dcat_pw_live(uint64_t handle, uint64_t owner, uint64_t generation) {
    std::lock_guard<std::mutex> lock(guard);
    const auto found = leases.find(handle);
    if (found == leases.end()) return 0;
    return owns(found->second, owner, generation) ? 1 : -1;
}
extern "C" int dcat_pw_copy_original_v1(uint64_t handle, uint64_t owner, uint64_t generation,
                                         void *host, uint64_t capacity,
                                         dcat_pw_original_info_v1 *info) {
    std::lock_guard<std::mutex> lock(guard);
    const Lease *live = owned(handle, owner, generation);
    if (!live) return 2;
    const uint64_t bytes = live->original.size() * sizeof(uint16_t);
    if (!host || !info || capacity != bytes) return 1;
    const auto payload = reinterpret_cast<uintptr_t>(host);
    const auto metadata = reinterpret_cast<uintptr_t>(info);
    const auto limit = std::numeric_limits<uintptr_t>::max();
    if (bytes > limit - payload || sizeof(*info) > limit - metadata) return 1;
    const auto overlaps = [](uintptr_t a, uint64_t na, uintptr_t b, uint64_t nb) {
        return a < b + nb && b < a + na;
    };
    if (overlaps(payload, bytes, metadata, sizeof(*info))) return 14;
    // Neither output may alias any owned tensor or either retained snapshot.
    // This diagnostic must not become an alternative writer into a lease.
    for (const auto &entry : leases) {
        const Lease &other = entry.second;
        const uint64_t size = other.original.size() * sizeof(uint16_t);
        const uintptr_t protected_spans[] = {reinterpret_cast<uintptr_t>(other.tensor),
            reinterpret_cast<uintptr_t>(other.original.data()),
            reinterpret_cast<uintptr_t>(other.expected.data())};
        for (uintptr_t start : protected_spans) {
            if (overlaps(payload, bytes, start, size) ||
                overlaps(metadata, sizeof(*info), start, size)) return 14;
        }
    }
    auto *output = static_cast<uint8_t *>(host);
    for (uint64_t i = 0; i < live->original.size(); ++i) {
        const uint16_t bits = live->original[i];
        output[2 * i] = static_cast<uint8_t>(bits & 0xffU);
        output[2 * i + 1] = static_cast<uint8_t>(bits >> 8);
    }
    *info = {1, 1, handle, owner, generation,
        static_cast<uint64_t>(reinterpret_cast<uintptr_t>(live->tensor)),
        static_cast<uint64_t>(live->original.size()), bytes, live->device,
        static_cast<uint32_t>(live->applied)};
    return 0;
}
