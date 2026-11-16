// Included by the opt-in Fault20 build. QMM and WeightNz have distinct formal
// contracts, and neither call uses the process-shared v17 arm.
namespace {
dcat_matmul::Bridge qmm_auto;
dcat_matmul::Bridge weight_nz_auto;
std::mutex auto_install_mutex;

struct AutoAcl {
    GetDeviceFn device = nullptr;
    MallocFn allocate = nullptr;
    FreeFn release = nullptr;
    MallocHostFn host_alloc = nullptr;
    FreeHostFn host_release = nullptr;
    MemcpyFn copy = nullptr;
    SynchronizeStreamFn sync = nullptr;
};

AutoAcl auto_acl;

bool resolve_auto_acl(bool install = true) {
    pthread_once(&g_resolve_once, resolve_real);
    if (!g_abi_ok || !g_developer_mode_ok) return false;
    for (void* symbol : {reinterpret_cast<void*>(g_real_run), reinterpret_cast<void*>(g_real_add),
            reinterpret_cast<void*>(g_real_add_param_const_tensor), reinterpret_cast<void*>(g_real_add_param_tensor),
            reinterpret_cast<void*>(g_get_log_api_info)}) {
        if (!f39identity::matches(symbol, f39profile::nn)) return false;
    }
    void* handle = dlopen("libascendcl.so", RTLD_LAZY | RTLD_NOLOAD);
    if (!handle) return false;
    auto bind = [handle](auto& function, const char* symbol) {
        auto address = dlsym(handle, symbol);
        if (!f39identity::matches(address, f39profile::acl)) return false;
        function = reinterpret_cast<std::remove_reference_t<decltype(function)>>(address);
        return true;
    };
    AutoAcl verified;
    const bool ok = bind(verified.device, "aclrtGetDevice") && bind(verified.allocate, "aclrtMalloc")
        && bind(verified.release, "aclrtFree") && bind(verified.host_alloc, "aclrtMallocHost")
        && bind(verified.host_release, "aclrtFreeHost") && bind(verified.copy, "aclrtMemcpy")
        && bind(verified.sync, "aclrtSynchronizeStream");
    dlclose(handle);
    if (ok && install) auto_acl = verified;
    return ok;
}

dcat_matmul::Memory auto_memory(aclrtStream stream) {
    return {
        auto_acl.host_alloc,
        [](void** p, size_t n) { return auto_acl.allocate(p, n, ACL_MEM_MALLOC_HUGE_FIRST); },
        auto_acl.host_release, auto_acl.release,
        [](void* to, const void* from, size_t n) { return auto_acl.copy(to, n, from, n, ACL_MEMCPY_DEVICE_TO_HOST); },
        [](void* to, const void* from, size_t n) { return auto_acl.copy(to, n, from, n, ACL_MEMCPY_HOST_TO_DEVICE); },
        [stream]() { return auto_acl.sync(stream); },
        [](void* to, const void* from, size_t n) { return auto_acl.copy(to, n, from, n, ACL_MEMCPY_DEVICE_TO_DEVICE); },
    };
}

bool append_signature(std::ostringstream& signature, const Fault20TensorMetadata& tensor,
                      const char* name, int32_t slot) {
    if (slot < 0 || tensor.view_rank > FAULT20_MAX_DIMS || tensor.stride_rank > FAULT20_MAX_DIMS
        || tensor.storage_rank > FAULT20_MAX_DIMS) return false;
    signature << ':' << name << ',' << slot << ',' << tensor.dtype << ','
              << tensor.format << ',' << tensor.storage_bytes << ',' << tensor.view_offset;
    signature << ",v";
    for (size_t j = 0; j < tensor.view_rank; ++j) signature << ',' << tensor.view_shape[j];
    signature << ",s";
    for (size_t j = 0; j < tensor.stride_rank; ++j) signature << ',' << tensor.strides[j];
    signature << ",p";
    for (size_t j = 0; j < tensor.storage_rank; ++j) signature << ',' << tensor.storage_shape[j];
    return true;
}

bool live_storage_slot(const Fault20TensorMetadata& tensor, int32_t slot,
                       const std::vector<void*>& addresses) {
    return tensor.present && tensor.storage_ptr && tensor.storage_bytes && !tensor.view_offset
        && slot >= 0 && static_cast<size_t>(slot) < addresses.size() && tensor.storage_addr
        && reinterpret_cast<uintptr_t>(addresses[slot]) == tensor.storage_addr
        && find_unique_address_slot(addresses, tensor) == slot;
}

aclnnStatus run_auto_qmm(op::internal::OpExecCache* cache, aclrtStream stream,
                        const std::vector<void*>& addresses, const dcat_matmul::Run& original) {
    RegisteredMetadata registered{};
    uint64_t run_index = 0;
    int32_t device = -1;
    if (!lookup_metadata(cache, &registered, &run_index) || auto_acl.device(&device) != ACL_SUCCESS
        || registered.device != device) return original(addresses);
    const auto& meta = registered.tensors;
    const auto order = prove_qmm_storage_order(registered, addresses);
    const auto& x1 = meta.x1;
    if (x1.view_shape[0] <= 0 || x1.view_shape[0] > INT32_MAX ||
        x1.view_shape[1] <= 0 || x1.view_shape[1] > INT32_MAX) return original(addresses);
    const SemanticTarget selector{
        FAULT20_TARGET_API_QMM, TensorRole::kQmmX1,
        static_cast<int32_t>(x1.view_shape[0]), static_cast<int32_t>(x1.view_shape[1]), 65536};
    const auto resolved = ResolveSemanticTarget(selector, registered, addresses, order,
        FAULT20_PROOF_MODE_SEMANTIC_SLOT, false, false);
    // A cached GWS output address is not a current output slot. Refuse that
    // layout instead of reading old or recycled output storage.
    if (!resolved.eligible || !order.weight_slot_proven || !order.x2_scale_slot_proven || !order.output_slot_proven ||
        !qmm_weight_metadata_valid(meta.x2, selector.k, dcat_matmul::byte_limit) ||
        !qmm_output_metadata_valid(meta.out, dcat_matmul::byte_limit)) return original(addresses);
    const Fault20TensorMetadata* formal[] = {&meta.x1, &meta.x2, &meta.x1_scale, &meta.x2_scale,
        &meta.y_scale, &meta.x1_offset, &meta.x2_offset, &meta.y_offset, &meta.bias, &meta.out};
    static const char* names[] = {"x1", "x2", "x1_scale", "x2_scale",
        "y_scale", "x1_offset", "x2_offset", "y_offset", "bias", "out"};
    std::vector<dcat_matmul::Slot> slots;
    std::ostringstream signature;
    signature << "qmm:" << meta.transpose_x1 << ':' << meta.transpose_x2 << ':' << meta.group_size;
    for (size_t i = 0; i < 10; ++i) {
        const auto& tensor = *formal[i];
        if (!tensor.present || !tensor.storage_bytes) continue;
        if (!live_storage_slot(tensor, order.slots[i], addresses)
            || !append_signature(signature, tensor, names[i], order.slots[i])) return original(addresses);
        slots.push_back({names[i], size_t(order.slots[i]), tensor.storage_bytes, i == 9, i == 0});
    }
    if (!dcat_matmul::layout_safe(slots, addresses)) return original(addresses);

    const auto details = dcat_matmul::descriptor("aclnnQuantMatmulWeightNz", signature.str(), cache,
        device, f39profile::id, "INT8", x1.view_shape, x1.strides, x1.view_offset, slots,
        registered.cache_lifecycle);
    const auto transform = [](size_t i, void* host, size_t bytes) {
        if (i != 0) return false;
        auto* values = static_cast<int8_t*>(host);
        for (size_t j = 0; j < bytes; ++j) values[j] = fault20_int4_equivalent_value(values[j]);
        return true;
    };
    return qmm_auto.dispatch(details, [&](uint32_t arm, dcat_matmul::Audit& audit, const auto* expected) {
        return dcat_matmul::audited_run(arm, slots, addresses, auto_memory(stream), transform, original, audit, expected);
    });
}

aclnnStatus run_auto_weight_nz(op::internal::OpExecCache* cache, aclrtStream stream,
                              const std::vector<void*>& addresses, const dcat_matmul::Run& original) {
    RegisteredMatmulMetadata registered{};
    uint64_t run_index = 0;
    int32_t device = -1;
    if (!lookup_matmul_metadata(cache, MatmulApiKind::kWeightNz, &registered, &run_index)
        || auto_acl.device(&device) != ACL_SUCCESS || device != registered.device) return original(addresses);
    const auto& meta = registered.tensors;
    const auto order = prove_matmul_storage_order(registered, addresses);
    const auto& self = meta.self;
    const bool bf16 = self.dtype == ACL_BF16;
    constexpr int32_t fractal_nz = 29;
    constexpr int64_t max_elements = 65536 / sizeof(uint16_t);
    if (!order.self_slot_proven || !order.mat2_slot_proven || !order.output_slot_proven
        || (!bf16 && self.dtype != ACL_FLOAT16)
        || self.format != ACL_FORMAT_ND || self.view_rank != 2 || self.storage_rank != 1
        || self.stride_rank != 2 || !self.contiguous || self.view_offset != 0
        || self.view_shape[1] <= 0 || self.view_shape[0] <= 0
        || self.view_shape[0] > max_elements / self.view_shape[1]
        || self.strides[0] != self.view_shape[1] || self.strides[1] != 1
        || self.element_size != sizeof(uint16_t) || meta.mat2.format != fractal_nz
        || meta.mat2.dtype != self.dtype || meta.out.dtype != self.dtype
        || meta.mat2.view_rank != 2 || meta.mat2.storage_rank != 4
        || meta.mat2.element_size != sizeof(uint16_t) || !qmm_output_metadata_valid(meta.out, dcat_matmul::byte_limit))
        return original(addresses);
    const auto elements = self.view_shape[0] * self.view_shape[1];
    if (self.storage_shape[0] != elements || self.numel != elements || self.storage_elements != elements
        || self.view_bytes != self.storage_bytes || self.storage_bytes != elements * sizeof(uint16_t))
        return original(addresses);
    const std::vector<dcat_matmul::Slot> slots{
        {"self", 0, self.storage_bytes, false, true},
        {"mat2", 1, meta.mat2.storage_bytes, false, false},
        {"out", 2, meta.out.storage_bytes, true, false},
    };
    std::ostringstream signature;
    constexpr uint32_t keep_mantissa_bits = 3;
    signature << "weight_nz:" << meta.cube_math_type << ':' << keep_mantissa_bits;
    if (!live_storage_slot(self, 0, addresses) || !live_storage_slot(meta.mat2, 1, addresses)
        || !live_storage_slot(meta.out, 2, addresses) || !dcat_matmul::layout_safe(slots, addresses)
        || !append_signature(signature, self, "self", 0)
        || !append_signature(signature, meta.mat2, "mat2", 1)
        || !append_signature(signature, meta.out, "out", 2)) return original(addresses);
    const auto details = dcat_matmul::descriptor("aclnnMatmulWeightNz", signature.str(), cache, device,
        f39profile::id, bf16 ? "BF16" : "FP16", self.view_shape, self.strides, self.view_offset, slots,
        registered.cache_lifecycle);
    const auto transform = [bf16](size_t slot, void* host, size_t bytes) {
        if (slot != 0) return false;
        auto* values = static_cast<uint16_t*>(host);
        for (size_t index = 0; index < bytes / sizeof(*values); ++index) {
            auto& value = values[index];
            value = bf16 ? fault20_truncate_bf16_mantissa(value, keep_mantissa_bits)
                         : fault20_truncate_fp16_mantissa(value, keep_mantissa_bits);
        }
        return true;
    };
    return weight_nz_auto.dispatch(details, [&](uint32_t arm, dcat_matmul::Audit& audit, const auto* expected) {
        return dcat_matmul::audited_run(arm, slots, addresses, auto_memory(stream), transform, original, audit, expected);
    });
}

aclnnStatus run_auto_fault20(op::internal::OpExecCache* cache, void* workspace, aclrtStream stream,
                            const std::vector<void*>& addresses) {
    pthread_once(&g_resolve_once, resolve_real);
    if (!g_real_run) return 1;
    const auto original = [&](const auto& selected) { return g_real_run(cache, workspace, stream, selected); };
    if (qmm_auto.on_owner() && starts_with(g_get_log_api_info(), "aclnnQuantMatmulWeightNz_"))
        return run_auto_qmm(cache, stream, addresses, original);
    if (weight_nz_auto.on_owner() && starts_with(g_get_log_api_info(), "aclnnMatmulWeightNz_"))
        return run_auto_weight_nz(cache, stream, addresses, original);
    return original(addresses);
}

int install_auto_bridge(dcat_matmul::Bridge& bridge, dcat_matmul::Callback callback, int32_t rank) {
    std::lock_guard<std::mutex> guard(auto_install_mutex);
    const char* task_queue = std::getenv("TASK_QUEUE_ENABLE");
    if (rank < 0 || !task_queue || std::strcmp(task_queue, "0")
        || qmm_auto.owned() || weight_nz_auto.owned() || !resolve_auto_acl()) return 1;
    return bridge.install(callback);
}

uint32_t auto_bridge_state(dcat_matmul::Bridge& bridge, dcat_matmul::Callback requester) {
    const char* task_queue = std::getenv("TASK_QUEUE_ENABLE");
    const bool matched = task_queue && !std::strcmp(task_queue, "0") && resolve_auto_acl(false);
    return bridge.state(requester) | (matched ? dcat_matmul::runtime_matched : 0);
}
}

extern "C" int dcat20_qmm_install_v1(dcat_matmul::Callback callback, int32_t rank) {
    return install_auto_bridge(qmm_auto, callback, rank);
}
extern "C" int dcat20_qmm_remove_v1(dcat_matmul::Callback callback) { return qmm_auto.remove(callback); }
extern "C" uint32_t dcat20_qmm_invoke_v1(uint64_t token, uint32_t arm) { return qmm_auto.invoke(token, arm); }
extern "C" const char* dcat20_qmm_receipt_v1(uint64_t token) { return qmm_auto.receipt(token); }
extern "C" const char* dcat20_qmm_profile_v1() { return f39profile::id; }
extern "C" uint32_t dcat20_qmm_state_v1(dcat_matmul::Callback requester) {
    return auto_bridge_state(qmm_auto, requester);
}

extern "C" int dcat20_weight_nz_install_v1(dcat_matmul::Callback callback, int32_t rank) {
    return install_auto_bridge(weight_nz_auto, callback, rank);
}
extern "C" int dcat20_weight_nz_remove_v1(dcat_matmul::Callback callback) { return weight_nz_auto.remove(callback); }
extern "C" uint32_t dcat20_weight_nz_invoke_v1(uint64_t token, uint32_t arm) {
    return weight_nz_auto.invoke(token, arm);
}
extern "C" const char* dcat20_weight_nz_receipt_v1(uint64_t token) { return weight_nz_auto.receipt(token); }
extern "C" uint32_t dcat20_weight_nz_state_v1(dcat_matmul::Callback requester) {
    return auto_bridge_state(weight_nz_auto, requester);
}
