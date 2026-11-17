#ifndef DCAT_METADATA_ARRAY_WITNESS_V1_H
#define DCAT_METADATA_ARRAY_WITNESS_V1_H
#include "metadata_array_state.h"
#include <sstream>
#include <string>

namespace dcat_metadata_witness {
struct MatmulCapture {
    DcatMetadataArrayStateV1 before{},after{};
};
inline bool quiescent(const DcatMetadataArrayStateV1& value) {
    return value.version==1 && value.size==sizeof(value) && value.instance
        && !value.active_captures && !value.active_getters && !value.getter_failures
        && !value.getter_exceptions && !value.capture_exceptions && !value.live_arrays
        && !value.ownership_errors && value.acquired_arrays==value.released_arrays;
}
inline bool valid(const MatmulCapture& value) {
    const auto& b=value.before;const auto& a=value.after;
    // Exactly self/mat2/out, each with three owned getter arrays. This is an
    // actual capture interval, never an inferred all-zero process origin.
    return quiescent(b) && quiescent(a) && b.instance==a.instance
        && a.capture_calls>=b.capture_calls && a.capture_calls-b.capture_calls==3
        && a.getter_calls>=b.getter_calls && a.getter_calls-b.getter_calls==9
        && a.acquired_arrays>=b.acquired_arrays && a.acquired_arrays-b.acquired_arrays==9
        && a.released_arrays>=b.released_arrays && a.released_arrays-b.released_arrays==9
        && a.peak_live_arrays>=b.peak_live_arrays && a.revision>b.revision;
}
inline std::string state_json(const DcatMetadataArrayStateV1& v) {
    std::ostringstream out;
    out<<"{\"version\":"<<v.version<<",\"size\":"<<v.size;
#define DCAT_WITNESS_JSON(name) out<<",\"" #name "\":"<<v.name
    DCAT_WITNESS_JSON(instance);DCAT_WITNESS_JSON(capture_calls);DCAT_WITNESS_JSON(active_captures);
    DCAT_WITNESS_JSON(getter_calls);DCAT_WITNESS_JSON(active_getters);DCAT_WITNESS_JSON(getter_failures);
    DCAT_WITNESS_JSON(getter_exceptions);DCAT_WITNESS_JSON(capture_exceptions);
    DCAT_WITNESS_JSON(acquired_arrays);DCAT_WITNESS_JSON(released_arrays);DCAT_WITNESS_JSON(live_arrays);
    DCAT_WITNESS_JSON(peak_live_arrays);DCAT_WITNESS_JSON(ownership_errors);DCAT_WITNESS_JSON(revision);
#undef DCAT_WITNESS_JSON
    out<<'}';return out.str();
}
inline std::string json(const MatmulCapture& value) {
    if(!valid(value))return "null";
    return "{\"source\":\"actual_typed_gws_capture\",\"before\":"+state_json(value.before)
        +",\"after\":"+state_json(value.after)+"}";
}
}
#endif
