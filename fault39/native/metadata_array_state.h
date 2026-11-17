#ifndef DCAT_METADATA_ARRAY_STATE_V1_H
#define DCAT_METADATA_ARRAY_STATE_V1_H
#include <cstdint>
// Read-only state layout shared by the owner and cached-registration witness.
struct DcatMetadataArrayStateV1 {
    uint32_t version{1}, size{sizeof(DcatMetadataArrayStateV1)};
    uint64_t instance{}, capture_calls{}, active_captures{};
    uint64_t getter_calls{}, active_getters{}, getter_failures{}, getter_exceptions{};
    uint64_t capture_exceptions{}, acquired_arrays{}, released_arrays{}, live_arrays{};
    uint64_t peak_live_arrays{}, ownership_errors{}, revision{};
};
#endif
