#include "../../fault_injection/native_adapters/qmm_pta_gate.h"
#include <cassert>
#include <iostream>
int main() {
    using namespace dcat_pta;
    assert(parse_mode(nullptr) == Mode::disabled);
    assert(parse_mode("") == Mode::disabled);
    assert(parse_mode("0") == Mode::disabled);
    assert(parse_mode("1") == Mode::require_typed_gws);
    for (const char* value : {"true", "01", "2", " 1", "1 ", "false"}) assert(parse_mode(value) == Mode::invalid);
    for (const char* api : {"aclnnMatmul", "aclnnQuantMatmul", "aclnnMatmulWeightNzGetWorkspaceSize",
            "aclnnQuantMatmulWeightNzGetWorkspaceSize", "aclnnMatmulWeightNzV2", "", "AC LNN"}) {
        assert(!target_kind(api));
        assert(route(Mode::require_typed_gws, true, true, true, false, api) == Route::original);
    }
    assert(!target_kind(nullptr));
    for (const char* api : {"aclnnMatmulWeightNz", "aclnnQuantMatmulWeightNz"}) {
        assert(route(Mode::require_typed_gws, true, true, true, false, api) == Route::require_gws);
        assert(route(Mode::disabled, false, false, true, false, api) == Route::original);
        assert(route(Mode::require_typed_gws, false, true, true, false, api) == Route::refused);
        assert(route(Mode::require_typed_gws, true, false, true, false, api) == Route::refused);
        assert(route(Mode::require_typed_gws, true, true, false, false, api) == Route::refused);
        assert(route(Mode::require_typed_gws, true, true, true, true, api) == Route::refused);
        assert(route(Mode::invalid, true, true, true, false, api) == Route::refused);
    }
    std::cout << "PASS_CPU_ONLY exact targets, explicit opt-in, identity/profile/config refusal, passthrough\n";
}
