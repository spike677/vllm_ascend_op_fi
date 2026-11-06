# Offline dependency target, matching cann-cmake master-026 json.cmake's
# header-only include and namespace contract. This uses the frozen single
# header and avoids that module's undefined third_party_json target when its
# local-header find_path succeeds. No operator/tiling source is modified.
if(NOT TARGET json)
    if(NOT EXISTS "/deps/json/single_include/nlohmann/json.hpp")
        message(FATAL_ERROR "Frozen offline JSON header is missing")
    endif()
    add_library(json INTERFACE)
    target_include_directories(json INTERFACE /deps/json/single_include)
    target_compile_definitions(json INTERFACE nlohmann=ascend_nlohmann)
    set(JSON_INCLUDE_DIR /deps/json/single_include)
    set(JSON_SOURCE_PATH /deps/json)
endif()
