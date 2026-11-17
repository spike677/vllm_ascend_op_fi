// Extracted from historical CANN 8.5.1 metadata lifecycle proxy; strict required tensors.
bool checked_product(const int64_t *dims, uint64_t count, int64_t *product) {
    if (dims == nullptr || count == 0 || count > 16) {
        return false;
    }
    uint64_t value = 1;
    for (uint64_t i = 0; i < count; ++i) {
        if (dims[i] < 0) {
            return false;
        }
        const uint64_t dim = static_cast<uint64_t>(dims[i]);
        if (dim != 0 && value > static_cast<uint64_t>(std::numeric_limits<int64_t>::max()) / dim) {
            return false;
        }
        value *= dim;
    }
    *product = static_cast<int64_t>(value);
    return true;
}

bool is_contiguous(const int64_t *shape, uint64_t shape_count, const int64_t *strides,
                   uint64_t stride_count) {
    if (shape == nullptr || strides == nullptr || shape_count == 0 ||
        shape_count != stride_count || shape_count > 16) {
        return false;
    }
    int64_t expected = 1;
    for (uint64_t reverse = 0; reverse < shape_count; ++reverse) {
        const uint64_t i = shape_count - reverse - 1;
        if (strides[i] != expected || shape[i] < 0) {
            return false;
        }
        if (shape[i] != 0 &&
            expected > std::numeric_limits<int64_t>::max() / shape[i]) {
            return false;
        }
        expected *= shape[i];
    }
    return true;
}

bool capture_tensor_metadata(const aclTensor *tensor, Fault39TensorMetadata *metadata) try {
    dcat_metadata::CaptureScope capture_scope;
    if (tensor == nullptr || metadata == nullptr) {
        return false;
    }
    aclDataType dtype = ACL_DT_UNDEFINED;
    aclFormat format = ACL_FORMAT_UNDEFINED;
    int64_t view_offset = 0;
    dcat_metadata::OwnedArray view_dims;
    dcat_metadata::OwnedArray storage_dims;
    dcat_metadata::OwnedArray strides;
    uint64_t view_count = 0;
    uint64_t storage_count = 0;
    uint64_t stride_count = 0;
    void *raw_addr = nullptr;
    if (aclGetDataType(tensor, &dtype) != ACL_SUCCESS ||
        aclGetFormat(tensor, &format) != ACL_SUCCESS ||
        aclGetViewOffset(tensor, &view_offset) != ACL_SUCCESS ||
        view_dims.read(aclGetViewShape, tensor, &view_count) != ACL_SUCCESS ||
        storage_dims.read(aclGetStorageShape, tensor, &storage_count) != ACL_SUCCESS ||
        strides.read(aclGetViewStrides, tensor, &stride_count) != ACL_SUCCESS ||
        aclGetRawTensorAddr(tensor, &raw_addr) != ACL_SUCCESS || view_count == 0 ||
        storage_count == 0 || stride_count == 0 || view_count > 16 ||
        storage_count > 16 || stride_count > 16) {
        return false;
    }
    int64_t numel = 0;
    int64_t storage_elements = 0;
    if (!checked_product(view_dims, view_count, &numel) ||
        !checked_product(storage_dims, storage_count, &storage_elements)) {
        return false;
    }
    const size_t element_size = aclDataTypeSize(dtype);
    const uint64_t view_bytes = checked_bytes(numel, element_size);
    const uint64_t storage_bytes = checked_bytes(storage_elements, element_size);
    if (element_size == 0) {
        return false;
    }
    Fault39TensorMetadata value{};
    value.dtype = static_cast<int32_t>(dtype);
    value.format = static_cast<int32_t>(format);
    value.present = 1;
    value.contiguous = is_contiguous(view_dims, view_count, strides, stride_count) ? 1U : 0U;
    value.view_rank = static_cast<uint32_t>(view_count);
    value.storage_rank = static_cast<uint32_t>(storage_count);
    value.stride_rank = static_cast<uint32_t>(stride_count);
    for (uint64_t i = 0; i < view_count; ++i) {
        value.view_shape[i] = view_dims[i];
    }
    for (uint64_t i = 0; i < storage_count; ++i) {
        value.storage_shape[i] = storage_dims[i];
    }
    for (uint64_t i = 0; i < stride_count; ++i) {
        value.strides[i] = strides[i];
    }
    value.view_offset = view_offset;
    value.numel = numel;
    value.storage_elements = storage_elements;
    value.element_size = element_size;
    value.view_bytes = view_bytes;
    value.storage_bytes = storage_bytes;
    value.tensor_ptr = reinterpret_cast<uintptr_t>(tensor);
    value.storage_ptr = reinterpret_cast<uintptr_t>(tensor->GetStorage());
    value.raw_addr = reinterpret_cast<uintptr_t>(raw_addr);
    value.storage_addr = reinterpret_cast<uintptr_t>(tensor->GetStorageAddr());
    value.data_addr = reinterpret_cast<uintptr_t>(tensor->GetData());
    *metadata = value;
    return true;
} catch (...) {
    dcat_metadata::capture_exception();
    return false;
}
