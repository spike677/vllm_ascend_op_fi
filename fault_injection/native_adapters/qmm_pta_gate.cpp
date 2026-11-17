// Linked into the isolated opapi proxies, not merely LD_PRELOAD: torch_npu
// resolves CanUsePTACache by dlsym on its actual opapi library handle.
// A deployment must consume status before admission and after collection.
#include "qmm_pta_gate.h"
#include "fault39_build_profile.h"
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <dlfcn.h>
#include <mutex>
#include <sstream>
#include <string>

namespace {
using CanUse = bool (*)(const char*);
using Profile = const char* (*)();
struct State {
    std::string configured;
    dcat_pta::Mode mode = dcat_pta::Mode::disabled;
    CanUse original = nullptr;
    bool identity = false, profile = false, math_identity = false;
};
#if defined(DEMONCAT_QMM_MATH_GATE)
constexpr const char* entry = "libopapi_math.so";
#elif defined(DEMONCAT_QMM_SPLIT)
constexpr const char* entry = "libopapi_nn.so";
#else
constexpr const char* entry = "libopapi.so";
#endif
#ifdef DCAT20_HAS_PTA_RESOLVER_WHEEL_PROFILE
constexpr const char* resolver_wheel = f39profile::pta_resolver_wheel_sha256;
#else
constexpr const char* resolver_wheel = ""; // CPU fixtures; production admission refuses this
#endif
State state;
std::once_flag once;
std::atomic<bool> refused{false};
std::atomic<const char*> reason{"NOT_INITIALIZED"};
std::atomic<uint64_t> original_calls{0}, qmm_bypass{0}, nz_bypass{0}, refused_calls{0};

std::string configuration() {
    const char* value = std::getenv("DEMONCAT_F20_REQUIRE_TYPED_GWS");
    return value ? value : "";
}

void refuse(const char* why) {
    refused.store(true);
    reason.store(why);
}

void initialize() {
    state.configured = configuration();
    state.mode = dcat_pta::parse_mode(state.configured.c_str());
    // Only inspect an existing library. No RTLD_DEFAULT guess or new CANN load.
    void* handle = dlopen("libnnopbase.so", RTLD_LAZY | RTLD_NOLOAD);
    void* original = handle ? dlsym(handle, "CanUsePTACache") : nullptr;
    state.original = reinterpret_cast<CanUse>(original);
    state.identity = f39identity::matches(original, f39profile::nn);
#if defined(DEMONCAT_QMM_MATH_GATE)
    // The math proxy must retain the original math implementation as a real
    // dependency. Check its own anchor, never an address resolved globally.
    void* math_handle = dlopen("libopapi_math_real.so", RTLD_LAZY | RTLD_NOLOAD);
    void* math_anchor = math_handle ? dlsym(math_handle, "aclnnAdd") : nullptr;
    state.math_identity = f39identity::matches(math_anchor, f39profile::math);
#endif
    const auto profile = reinterpret_cast<Profile>(dlsym(RTLD_DEFAULT, "dcat20_qmm_profile_v1"));
    const char* actual_profile = profile ? profile() : nullptr;
    state.profile = actual_profile && std::strcmp(actual_profile, f39profile::id) == 0;
    // The explicit handle pins the original for the life of this proxy.
    if (!state.original) refuse("ORIGINAL_CAN_USE_PTA_MISSING");
    else if (state.mode == dcat_pta::Mode::invalid) refuse("INVALID_EXPLICIT_CONTROL_VALUE");
    else if (state.mode == dcat_pta::Mode::require_typed_gws && !state.identity)
        refuse("LOADED_NNOPBASE_IDENTITY_MISMATCH");
    else if (state.mode == dcat_pta::Mode::require_typed_gws && !state.profile)
        refuse("PRELOAD_PROFILE_MISMATCH");
#if defined(DEMONCAT_QMM_MATH_GATE)
    else if (state.mode == dcat_pta::Mode::require_typed_gws && !state.math_identity)
        refuse("LOADED_ORIGINAL_MATH_IDENTITY_MISMATCH");
#endif
    else reason.store(state.mode == dcat_pta::Mode::require_typed_gws
        ? "CONTROL_READY_REQUIRES_REAL_ROUTING" : "DISABLED_ORIGINAL_ROUTING");
}

bool config_unchanged() {
    if (configuration() == state.configured) return true;
    refuse("PROCESS_CONTROL_CONFIGURATION_CHANGED");
    return false;
}
} // namespace

extern "C" bool CanUsePTACache(const char* api) {
    std::call_once(once, initialize);
    const auto selected = dcat_pta::route(state.mode, state.identity, state.profile,
        config_unchanged(), refused.load(), api);
    if (selected == dcat_pta::Route::require_gws) {
        if (dcat_pta::target_kind(api) == 1) ++qmm_bypass; else ++nz_bypass;
        return false;
    }
    if (selected == dcat_pta::Route::refused) ++refused_calls;
    // Refusal does not pretend that the target reached typed GWS. Preserve the
    // original decision when it exists; the provider must reject status.refused.
    if (!state.original) {
        std::fprintf(stderr, "Fault20 PTA control refused: ORIGINAL_CAN_USE_PTA_MISSING\n");
        std::abort(); // no original result exists to forward; never invent false
    }
    ++original_calls;
    return state.original(api);
}

extern "C" const char* dcat20_pta_gate_status_v1() {
    std::call_once(once, initialize);
    (void)config_unchanged();
    thread_local std::string result;
    const bool requested = state.mode == dcat_pta::Mode::require_typed_gws;
    std::ostringstream out;
    out << "{\"schema\":\"dcat20.pta_gate.v1\",\"profile_sha256\":\"" << f39profile::id
        << "\",\"entry\":\"" << entry
        << "\",\"resolver_wheel_sha256\":\"" << resolver_wheel
        << "\",\"nnopbase_sha256\":\"" << f39profile::nn.file_sha256 << "\",\"requested\":"
        << (requested ? "true" : "false") << ",\"config_valid\":"
        << (state.mode != dcat_pta::Mode::invalid ? "true" : "false")
        << ",\"identity_valid\":" << (state.identity ? "true" : "false")
        << ",\"profile_match\":" << (state.profile ? "true" : "false")
        << ",\"original_available\":" << (state.original ? "true" : "false")
        << ",\"original_math_identity_valid\":" << (state.math_identity ? "true" : "false")
        << ",\"refused\":" << (refused.load() ? "true" : "false")
        << ",\"ready\":" << (requested && !refused.load() && state.identity && state.profile ? "true" : "false")
        << ",\"reason\":\"" << reason.load() << "\",\"original_calls\":" << original_calls.load()
        << ",\"qmm_bypass\":" << qmm_bypass.load() << ",\"weight_nz_bypass\":" << nz_bypass.load()
        << ",\"refused_calls\":" << refused_calls.load() << "}";
    result = out.str();
    return result.c_str();
}
