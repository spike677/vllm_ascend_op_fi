// Pure routing contract for the explicit Fault20 compatibility mode.
// This does not disable the CANN OpExecCache used by the injection boundary.
#pragma once
#include <cstring>

namespace dcat_pta {
enum class Mode { disabled, require_typed_gws, invalid };
enum class Route { original, require_gws, refused };

inline Mode parse_mode(const char* value) {
    if (!value || !*value || std::strcmp(value, "0") == 0) return Mode::disabled;
    return std::strcmp(value, "1") == 0 ? Mode::require_typed_gws : Mode::invalid;
}

inline unsigned target_kind(const char* api) {
    if (!api) return 0;
    if (std::strcmp(api, "aclnnQuantMatmulWeightNz") == 0) return 1;
    if (std::strcmp(api, "aclnnMatmulWeightNz") == 0) return 2;
    return 0;
}

inline Route route(Mode mode, bool identity, bool profile, bool config_unchanged,
                   bool already_refused, const char* api) {
    if (!config_unchanged || already_refused || mode == Mode::invalid) return Route::refused;
    if (mode == Mode::disabled) return Route::original;
    if (!identity || !profile) return Route::refused;
    return target_kind(api) ? Route::require_gws : Route::original;
}
} // namespace dcat_pta
