#ifndef DCAT_METADATA_ARRAY_OWNER_V1_H
#define DCAT_METADATA_ARRAY_OWNER_V1_H

#include <atomic>
#include <cstdint>
#include "metadata_array_state.h"

// Additive, read-only ABI. Each metadata proxy owns an independent instance.
// Counts concern SDK getter arrays only, not all host or device allocations.

// Internal linkage is essential: opapi and opapi_nn must not share counters
// through ELF interposition or GNU-unique inline variables.
namespace { namespace dcat_metadata {
struct Counters {
    std::atomic<uint64_t> capture_calls{0}, active_captures{0}, getter_calls{0}, active_getters{0};
    std::atomic<uint64_t> getter_failures{0}, getter_exceptions{0}, capture_exceptions{0};
    std::atomic<uint64_t> acquired_arrays{0}, released_arrays{0}, live_arrays{0}, peak_live_arrays{0};
    std::atomic<uint64_t> ownership_errors{0}, revision{0};
};
static Counters counters;

static void changed() noexcept { counters.revision.fetch_add(1); }

struct CaptureScope {
    CaptureScope() noexcept {
        counters.active_captures.fetch_add(1);
        counters.capture_calls.fetch_add(1);
        changed();
    }
    ~CaptureScope() noexcept { counters.active_captures.fetch_sub(1); changed(); }
    CaptureScope(const CaptureScope&) = delete;
    CaptureScope& operator=(const CaptureScope&) = delete;
};

static void capture_exception() noexcept {
    counters.capture_exceptions.fetch_add(1);
    changed();
}

class OwnedArray {
    int64_t* value_{nullptr};
    bool counted_{false};

    void returned_pointer() noexcept {
        if (!value_) return;
        if (counted_) { counters.ownership_errors.fetch_add(1); changed(); return; }
        counted_ = true;
        counters.acquired_arrays.fetch_add(1);
        const auto live = counters.live_arrays.fetch_add(1) + 1;
        auto peak = counters.peak_live_arrays.load();
        while (peak < live && !counters.peak_live_arrays.compare_exchange_weak(peak, live)) {}
        changed();
    }
public:
    OwnedArray() = default;
    OwnedArray(const OwnedArray&) = delete;
    OwnedArray& operator=(const OwnedArray&) = delete;
    ~OwnedArray() noexcept {
        if (!value_) return;
        // The SDK allocates these three getter results with new[]. Even a
        // failing/throwing getter may have assigned a pointer before failing.
        if (!counted_) counters.ownership_errors.fetch_add(1);
        delete[] value_;
        counters.released_arrays.fetch_add(1);
        if (counted_) counters.live_arrays.fetch_sub(1);
        changed();
    }

    template<class Getter, class Tensor>
    auto read(Getter getter, const Tensor* tensor, uint64_t* count)
        -> decltype(getter(tensor, static_cast<int64_t**>(nullptr), count)) {
        using Status = decltype(getter(tensor, static_cast<int64_t**>(nullptr), count));
        if (value_ || counted_) {
            counters.ownership_errors.fetch_add(1); changed();
            return static_cast<Status>(161001);
        }
        counters.active_getters.fetch_add(1);
        counters.getter_calls.fetch_add(1);
        changed();
        try {
            const auto status = getter(tensor, &value_, count);
            returned_pointer();
            if (status != 0) counters.getter_failures.fetch_add(1);
            counters.active_getters.fetch_sub(1); changed();
            return status;
        } catch (...) {
            returned_pointer();
            counters.getter_exceptions.fetch_add(1);
            counters.active_getters.fetch_sub(1); changed();
            throw;
        }
    }

    operator const int64_t*() const noexcept { return value_; }
    const int64_t* get() const noexcept { return value_; }
    const int64_t& operator[](uint64_t index) const noexcept { return value_[index]; }
};

static int snapshot(DcatMetadataArrayStateV1* out) noexcept {
    if (!out || out->version != 1 || out->size != sizeof(*out)) return 1;
    for (unsigned attempt = 0; attempt != 4; ++attempt) {
        const auto before = counters.revision.load();
        DcatMetadataArrayStateV1 value;
        value.instance = reinterpret_cast<uint64_t>(&counters);
#define DCAT_METADATA_READ(name) value.name = counters.name.load()
        DCAT_METADATA_READ(capture_calls); DCAT_METADATA_READ(active_captures);
        DCAT_METADATA_READ(getter_calls); DCAT_METADATA_READ(active_getters);
        DCAT_METADATA_READ(getter_failures); DCAT_METADATA_READ(getter_exceptions);
        DCAT_METADATA_READ(capture_exceptions); DCAT_METADATA_READ(acquired_arrays);
        DCAT_METADATA_READ(released_arrays); DCAT_METADATA_READ(live_arrays);
        DCAT_METADATA_READ(peak_live_arrays); DCAT_METADATA_READ(ownership_errors);
#undef DCAT_METADATA_READ
        value.revision = counters.revision.load();
        if (before == value.revision) { *out = value; return 0; }
    }
    return 2; // Concurrent changes: no quiescent resource proof.
}
}} // unnamed namespace / dcat_metadata
#endif
