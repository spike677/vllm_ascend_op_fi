// Only compiled with DEMONCAT_QMM_AUTO. Legacy shared-control paths are not
// consumers of this current-call binding or its diagnostics.
namespace {
thread_local dcat_current::Binding auto_current_binding;
std::atomic<uint64_t> auto_diagnostic_serial{0};
std::atomic<uint64_t> auto_gws_bound{0}, auto_gws_refused{0};
std::atomic<uint64_t> auto_execute_bound{0}, auto_execute_refused{0};
std::atomic<uint64_t> auto_run_bound{0}, auto_run_refused{0};

void auto_call_diagnostic(const char* event, uint32_t kind, uint64_t token,
                          const char* reason, const void* cache = nullptr) {
    const char* path = std::getenv("DEMONCAT_F20_DIAGNOSTICS_PATH");
    if (!path || !*path) return;
    const uint64_t sequence = auto_diagnostic_serial.fetch_add(1) + 1;
    if (sequence > 20000) return;
    char line[768];
    const int length = std::snprintf(line, sizeof(line),
        "{\"event\":\"%s\",\"kind\":%u,\"pid\":%ld,\"tid\":%ld,"
        "\"sequence\":%llu,\"gws_token\":%llu,\"reason\":\"%s\",\"cache\":\"%p\"}\n",
        event, kind, static_cast<long>(getpid()), static_cast<long>(syscall(SYS_gettid)),
        static_cast<unsigned long long>(sequence), static_cast<unsigned long long>(token), reason, cache);
    if (length <= 0 || static_cast<size_t>(length) >= sizeof(line)) return;
    const int fd = open(path, O_WRONLY | O_APPEND | O_CREAT | O_CLOEXEC, 0600);
    if (fd >= 0) {
        // Diagnostic I/O cannot change the serving calculation or admission.
        (void)write(fd, line, static_cast<size_t>(length));
        (void)close(fd);
    }
}

std::string with_current_call(const std::string& descriptor, const dcat_current::Call& call,
                              const std::vector<dcat_matmul::Slot>& slots,
                              const std::vector<void*>& addresses) {
    std::ostringstream out;
    out << descriptor.substr(0, descriptor.size() - 1)
        << ",\"current_typed_call\":{\"source\":\"typed_gws_execute_scope\","
        "\"executor_bound\":true,\"formal_params_current\":true,\"cached_layout_used\":false,"
        "\"gws_token\":" << call.token << ",\"gws_add_count\":" << call.add_count
        << ",\"formals\":[";
    bool first = true;
    const auto emit = [&](const char* name, const Fault20TensorMetadata& tensor) {
        int32_t index = -1;
        for (const auto& slot : slots)
            if (std::strcmp(name, slot.formal) == 0) index = static_cast<int32_t>(slot.index);
        const uint64_t address = index >= 0 && static_cast<size_t>(index) < addresses.size()
            ? reinterpret_cast<uintptr_t>(addresses[index]) : 0;
        if (!first) out << ',';
        first = false;
        out << dcat_current::tensor_observation(name, tensor, index, address);
    };
    if (call.kind == dcat_current::weight_nz) {
        emit("self", call.floating.self);
        emit("mat2", call.floating.mat2);
        emit("out", call.floating.out);
    } else if (call.kind == dcat_current::qmm) {
        emit("x1", call.quant.x1); emit("x2", call.quant.x2);
        emit("x1_scale", call.quant.x1_scale); emit("x2_scale", call.quant.x2_scale);
        emit("y_scale", call.quant.y_scale); emit("x1_offset", call.quant.x1_offset);
        emit("x2_offset", call.quant.x2_offset); emit("y_offset", call.quant.y_offset);
        emit("bias", call.quant.bias); emit("out", call.quant.out);
    }
    out << "]}}";
    return out.str();
}

const dcat_current::Call* current_call(uint32_t kind, const void* cache) {
    const auto* current = auto_current_binding.current(kind, cache);
    if (!current) {
        ++auto_run_refused;
        auto_call_diagnostic("run_refused", kind, 0, "NO_CURRENT_TYPED_EXECUTE_FOR_CACHE", cache);
    }
    return current;
}

uint32_t auto_passthrough(uint32_t kind, const dcat_current::Call& call,
                         const void* cache, const char* reason,
                         const std::vector<void*>& addresses, const dcat_matmul::Run& original) {
    ++auto_run_refused;
    auto_call_diagnostic("run_refused", kind, call.token, reason, cache);
    return original(addresses);
}
} // namespace

extern "C" uint64_t dcat20_auto_gws_begin_v1(uint32_t kind) {
    const auto token = auto_current_binding.begin(kind);
    auto_call_diagnostic("gws_begin", kind, token, auto_current_binding.reason());
    return token;
}

extern "C" void dcat20_auto_gws_abort_v1(uint64_t token, uint32_t kind) {
    auto_current_binding.abort(token);
    ++auto_gws_refused;
    auto_call_diagnostic("gws_refused", kind, token, "GWS_FAILED_OR_METADATA_UNAVAILABLE");
}

extern "C" bool dcat20_auto_gws_commit_v1(uint64_t token, uint32_t kind,
    const Fault20MetadataAbiDescriptor* descriptor, const void* executor) {
    // Copy the current AddParamToBuf observations before the legacy cache-miss
    // registration consumes its capture. Cache hits have zero AddOpExecCache.
    const auto capture = g_capture;
    dcat_current::Call call;
    call.kind = kind;
    call.executor = executor;
    call.add_count = capture.add_count;
    call.cache_created = capture.cache;
    call.parameter_count = capture.tensor_param_count;
    std::copy(std::begin(capture.tensor_params), std::end(capture.tensor_params), call.parameters.begin());
    bool metadata_ok = capture.active;
    if (kind == dcat_current::qmm) {
        const auto* typed = validated_metadata_payload<Fault20QmmMetadata>(descriptor, FAULT20_METADATA_PAYLOAD_QMM);
        metadata_ok = metadata_ok && typed && capture.kind == CaptureKind::kQmm;
        if (typed) call.quant = *typed;
    } else if (kind == dcat_current::weight_nz) {
        const auto* typed = validated_metadata_payload<Fault20MatmulMetadata>(descriptor, FAULT20_METADATA_PAYLOAD_MATMUL_WEIGHT_NZ);
        metadata_ok = metadata_ok && typed && capture.kind == CaptureKind::kMatmulWeightNz;
        if (typed) call.floating = *typed;
    } else {
        metadata_ok = false;
    }
    resolve_runtime_api();
    metadata_ok = metadata_ok && g_get_device && g_get_device(&call.device) == ACL_SUCCESS;
    bool registered = true;
    if (metadata_ok && capture.add_count == 1) {
        registered = kind == dcat_current::qmm ? fault20_qmm_gws_commit_v17(descriptor)
                                              : fault20_matmul_gws_commit_v17(descriptor);
    } else {
        g_capture = CaptureState{};
    }
    const bool accepted = metadata_ok && registered && auto_current_binding.publish(token, call);
    if (accepted) {
        ++auto_gws_bound;
        auto_call_diagnostic("gws_bound", kind, token, "CURRENT_TYPED_GWS_BOUND", capture.cache);
    } else {
        ++auto_gws_refused;
        const char* reason = !metadata_ok ? "CURRENT_METADATA_UNAVAILABLE"
            : (!registered ? "CACHE_CREATION_UNPROVEN" : auto_current_binding.reason());
        auto_call_diagnostic("gws_refused", kind, token, reason, capture.cache);
        auto_current_binding.abort(token);
    }
    return accepted;
}

extern "C" uint64_t dcat20_auto_execute_begin_v1(uint32_t kind, const void* executor) {
    const auto token = auto_current_binding.enter(kind, executor);
    if (token) ++auto_execute_bound; else ++auto_execute_refused;
    auto_call_diagnostic(token ? "execute_bound" : "execute_refused", kind, token, auto_current_binding.reason());
    return token;
}

extern "C" void dcat20_auto_execute_end_v1(uint64_t token, uint32_t kind, const void* executor) {
    if (token) {
        const bool closed = auto_current_binding.leave(token, kind, executor);
        auto_call_diagnostic("execute_end", kind, token, closed ? "EXECUTE_RELEASED" : "EXECUTE_RELEASE_MISMATCH");
    }
}

extern "C" const char* dcat20_auto_diagnostics_v1() {
    thread_local std::string result;
    std::ostringstream out;
    out << "{\"gws_bound\":" << auto_gws_bound.load() << ",\"gws_refused\":" << auto_gws_refused.load()
        << ",\"execute_bound\":" << auto_execute_bound.load() << ",\"execute_refused\":" << auto_execute_refused.load()
        << ",\"run_bound\":" << auto_run_bound.load() << ",\"run_refused\":" << auto_run_refused.load() << '}';
    result = out.str();
    return result.c_str();
}
