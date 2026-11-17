"""Version 1 native ABI client; never dlopen an interposer after worker start."""
import ctypes
import json
import threading
from contextlib import contextmanager
from typing import Protocol
from .schema import Fault39Config

ABI_VERSION = 1
PATTERN_IDS = {"half": 0, "alternating": 1, "block256": 2, "block512": 3, "block1024": 4}
MODE_IDS = {"discover": 1, "inject": 2, "lossless": 3, "observer": 4}
SERVING_INPUT_FIELDS = ("self_address", "self_storage_bytes", "mat2_address", "mat2_storage_bytes",
                        "M", "K", "N", "self_offset", "mat2_offset",
                        "self_stride0", "self_stride1", "mat2_stride0", "mat2_stride1")
ERRORS = {-1: "INVALID_CONFIG", -2: "ABI_MISMATCH", -3: "BUSY",
          -4: "OWNER_MISMATCH", -5: "GENERATION_MISMATCH", -6: "AMBIGUOUS_TARGET",
          -7: "TARGET_NOT_FOUND", -8: "UNSUPPORTED_RUNTIME", -9: "RECOVERY_FAILED",
          -10: "DRAIN_TIMEOUT", -11: "NO_OVERFLOW_SOLUTION"}


class NativeError(RuntimeError):
    def __init__(self, code, operation):
        self.code, self.operation = code, operation
        self.status = ERRORS.get(code, "UNKNOWN_NATIVE_ERROR")
        super().__init__(f"{operation}: {self.status} ({code})")


class NativeConfigV1(ctypes.Structure):
    _fields_ = [("abi_version", ctypes.c_uint32), ("struct_size", ctypes.c_uint32),
                ("mode", ctypes.c_uint32), ("pattern", ctypes.c_uint32),
                ("exponent", ctypes.c_int32), ("rank", ctypes.c_int32),
                ("device", ctypes.c_int32), ("max_hits", ctypes.c_uint32),
                ("max_bytes", ctypes.c_uint64), ("owner", ctypes.c_char * 64),
                ("target_signature", ctypes.c_char * 256),
                ("input_mask", ctypes.c_uint32), ("reserved", ctypes.c_uint32)]


class NativeProvider(Protocol):
    """Explicit standalone protocol, not an invented internal DemonCAT API."""
    def prepare(self) -> dict: ...
    def configure(self, owner: str, config: Fault39Config, rank: int, device: int,
                  mode: str = "inject", target_signature: str = "", input_mask: int = 3) -> int: ...
    def snapshot(self) -> dict: ...
    def disable(self, owner: str, generation: int, timeout_ms: int) -> dict: ...
    def clean(self, owner: str, generation: int, timeout_ms: int) -> dict: ...


def _bounded_text(value, capacity, name):
    if not isinstance(value, str) or "\0" in value:
        raise ValueError(f"Invalid {name}")
    encoded = value.encode("utf-8")
    if len(encoded) >= capacity:
        raise ValueError(f"{name} exceeds ABI capacity")
    return encoded


class CtypesProvider:
    """Attach to symbols already in this worker's global process namespace.

    This API has no DSO pathname argument. Presence of symbols proves attachment
    only, never interception. Native run/metadata evidence is checked separately.
    A test may construct this with a fake symbol table, explicitly marked mock.
    """
    def __init__(self, symbols, *, evidence_kind="NATIVE_ATTACHMENT_UNVERIFIED"):
        self._symbols = symbols
        self.evidence_kind = evidence_kind
        self._context_local = threading.local()
        signatures = {
            "fault39_abi_version_v1": ([], ctypes.c_uint32),
            "fault39_configure_v1": ([ctypes.POINTER(NativeConfigV1), ctypes.POINTER(ctypes.c_uint64)], ctypes.c_int),
            "fault39_disable_v1": ([ctypes.c_char_p, ctypes.c_uint64, ctypes.c_uint32], ctypes.c_int),
            "fault39_clean_v1": ([ctypes.c_char_p, ctypes.c_uint64, ctypes.c_uint32], ctypes.c_int),
            "fault39_snapshot_v1": ([ctypes.c_void_p, ctypes.c_size_t], ctypes.c_int),
            "fault39_set_context_v1": ([ctypes.c_char_p, ctypes.c_char_p], ctypes.c_int),
            "fault39_clear_context_v1": ([], None),
            "fault39_bind_worker_rank_v1": ([ctypes.c_int32], ctypes.c_int),
        }
        for name, (args, result) in signatures.items():
            try:
                fn = getattr(symbols, name)
            except AttributeError as exc:
                raise RuntimeError("NATIVE_PRELOAD_NOT_ATTACHED: " + name) from exc
            fn.argtypes, fn.restype = args, result
        if symbols.fault39_abi_version_v1() != ABI_VERSION:
            raise NativeError(-2, "attach")

    @classmethod
    def attach_preloaded(cls):
        # Never replace None with a library path. Late dlopen does not establish
        # that CANN's already-resolved calls will enter an interposer.
        return cls(ctypes.CDLL(None))

    def prepare(self):
        result = self.snapshot()
        result["provider_evidence_kind"] = self.evidence_kind
        result["attachment_verified"] = True
        result["interception_verified"] = False
        return result

    def bind_worker_rank(self, worker_rank):
        """Called from the actual worker rank API, not configuration targets."""
        if type(worker_rank) is not int or not 0 <= worker_rank < 2**31:
            raise ValueError("Actual nonnegative worker rank required")
        code = self._symbols.fault39_bind_worker_rank_v1(worker_rank)
        if code:
            raise NativeError(code, "bind_worker_rank")

    def configure(self, owner, config, rank, device, mode="inject", target_signature="", input_mask=3):
        return self._configure(owner, config, rank, device, mode, target_signature, input_mask)

    def configure_cache(self, owner, config, rank, device, *, cache_address, cache_epoch,
                        mode="inject", target_signature="", input_mask=3):
        """One exact live cache epoch, never a replacement for v1 unique selection."""
        if (any(type(value) is not int or not 0 < value < 2**64 for value in (cache_address, cache_epoch))
                or config.max_hits != 1 or mode == 'discover' or not target_signature
                or not getattr(self._context_local, 'active', False)
                or getattr(self._context_local, 'owner', None) != owner):
            raise ValueError('EXACT_CACHE_SINGLE_CALL_AND_OWNED_CONTEXT_REQUIRED')
        return self._configure(owner, config, rank, device, mode, target_signature, input_mask,
                               cache_identity=(cache_address, cache_epoch))

    def _configure(self, owner, config, rank, device, mode, target_signature, input_mask,
                   cache_identity=None):
        if mode not in MODE_IDS:
            raise ValueError("Unknown native mode")
        if not owner or type(rank) is not int or type(device) is not int or not 0 <= rank < 2**31 or not 0 <= device < 2**31:
            raise ValueError("Explicit owner, rank, and device are required")
        if type(input_mask) is not int or input_mask not in (1, 2, 3) or (mode != "lossless" and input_mask != 3):
            raise ValueError("Only lossless mode accepts self-only/mat2-only masks")
        raw = NativeConfigV1(ABI_VERSION, ctypes.sizeof(NativeConfigV1), MODE_IDS[mode],
                             PATTERN_IDS[config.pattern], config.exponent, rank, device,
                             config.max_hits, config.max_bytes,
                             _bounded_text(owner, 64, "owner"),
                             _bounded_text(target_signature, 256, "target_signature"), input_mask, 0)
        generation = ctypes.c_uint64()
        if cache_identity is None:
            code = self._symbols.fault39_configure_v1(ctypes.byref(raw), ctypes.byref(generation))
        else:
            fn = getattr(self._symbols, 'fault39_configure_cache_v2', None)
            if fn is None:
                raise RuntimeError('EXACT_CACHE_CONFIGURATION_UNAVAILABLE')
            fn.argtypes = [ctypes.POINTER(NativeConfigV1), ctypes.c_uint64, ctypes.c_uint64,
                           ctypes.POINTER(ctypes.c_uint64)]
            fn.restype = ctypes.c_int
            code = fn(ctypes.byref(raw), *cache_identity, ctypes.byref(generation))
        if code:
            raise NativeError(code, "configure")
        if not generation.value:
            raise RuntimeError("INVALID_NATIVE_GENERATION")
        return generation.value

    def snapshot(self):
        size = self._symbols.fault39_snapshot_v1(None, 0)
        for _ in range(4):
            if size <= 0:
                raise NativeError(size, "snapshot")
            if size > 16 * 1024 * 1024:
                raise RuntimeError("NATIVE_SNAPSHOT_SIZE_LIMIT")
            buffer = ctypes.create_string_buffer(size)
            needed = self._symbols.fault39_snapshot_v1(buffer, size)
            if needed <= 0:
                raise NativeError(needed, "snapshot")
            if needed > size:
                size = needed
                continue
            result = json.loads(buffer.value.decode("utf-8"))
            if not isinstance(result, dict):
                raise RuntimeError("INVALID_NATIVE_SNAPSHOT")
            return result
        raise RuntimeError("NATIVE_SNAPSHOT_UNSTABLE_SIZE")

    def metadata_arrays(self):
        """Read actual loaded-proxy counters; no symbol-presence qualification."""
        from fault_injection.metadata_arrays import LoadedMetadataArrays
        if not hasattr(self, '_metadata_arrays'):
            self._metadata_arrays = LoadedMetadataArrays(kind='39',
                build_profile=self.snapshot().get('build_profile_sha256'), anchor=self._symbols.fault39_snapshot_v1)
        return self._metadata_arrays.read()

    def _stop(self, operation, owner, generation, timeout_ms):
        if type(generation) is not int or not 1 <= generation < 2**64:
            raise ValueError("Invalid generation")
        if type(timeout_ms) is not int or not 0 <= timeout_ms < 2**32:
            raise ValueError("Invalid drain timeout")
        name = "fault39_" + operation + "_v1"
        code = getattr(self._symbols, name)(_bounded_text(owner, 64, "owner"), generation, timeout_ms)
        if code not in (0, -9, -10):
            raise NativeError(code, operation)
        state = self.snapshot()
        status = "RECOVERY_FAILED" if code == -9 else "DISABLE_PENDING" if code == -10 else "DISABLED_CONFIRMED"
        # A successful return still requires an actual zero-lease snapshot.
        if code == 0 and (state.get("enabled") or state.get("inflight") != 0 or state.get("leases") != 0):
            status = "DISABLE_PENDING"
        if code == 0 and operation == "disable" and (state.get("owner") != owner or state.get("generation") != generation):
            status = "DISABLE_PENDING"
        return {"status": status, "native_code": code, "generation": generation,
                "owner": owner, "snapshot": state}

    def disable(self, owner, generation, timeout_ms=10000):
        return self._stop("disable", owner, generation, timeout_ms)

    def clean(self, owner, generation, timeout_ms=10000):
        return self._stop("clean", owner, generation, timeout_ms)

    def resume_discovery(self, owner, generation):
        """Optional build-pinned continuation; no fallback to reconfigure."""
        fn = getattr(self._symbols, 'fault39_resume_discovery_v1', None)
        if fn is None:
            raise RuntimeError('SERVING_DISCOVERY_CONTINUATION_UNAVAILABLE')
        if type(generation) is not int or not 0 < generation < 2**64:
            raise ValueError('Invalid generation')
        fn.argtypes = [ctypes.c_char_p, ctypes.c_uint64]
        fn.restype = ctypes.c_int
        code = fn(_bounded_text(owner, 64, 'owner'), generation)
        if code:
            raise NativeError(code, 'resume_discovery')

    def bind_serving_inputs(self, owner, values):
        """Bind current model storage/layout on the actual owner model thread.

        This additive symbol is optional for legacy operator use, mandatory for
        the guarded serving path. No fallback may report live-input verification.
        Values follow SERVING_INPUT_FIELDS; strides/offsets count elements.
        """
        fn = getattr(self._symbols, 'fault39_set_serving_inputs_v1', None)
        if fn is None:
            raise RuntimeError('SERVING_LIVE_INPUT_GUARD_UNAVAILABLE')
        if (not getattr(self._context_local, 'active', False)
                or getattr(self._context_local, 'owner', None) != owner):
            raise RuntimeError('OWNED_SAME_THREAD_REQUEST_CONTEXT_REQUIRED')
        encoded_owner = _bounded_text(owner, 64, 'owner')
        fn.argtypes = [ctypes.c_char_p, ctypes.POINTER(ctypes.c_uint64), ctypes.c_size_t]
        fn.restype = ctypes.c_int
        if (not isinstance(values, (list, tuple)) or len(values) != len(SERVING_INPUT_FIELDS)
                or any(type(value) is not int or not 0 <= value < 2**64 for value in values)):
            # Fail closed even if a caller accidentally catches this exception
            # and continues executing inside the same request context.
            fn(encoded_owner, None, 0)
            raise ValueError('EXACTLY_13_UINT64_SERVING_INPUT_VALUES_REQUIRED')
        raw = (ctypes.c_uint64 * len(SERVING_INPUT_FIELDS))(*values)
        code = fn(encoded_owner, raw, len(raw))
        if code:
            raise NativeError(code, 'set_serving_inputs')

    @contextmanager
    def context(self, owner, request_ids):
        """Bind actual scheduler IDs on the calling model thread only.

        A set of multiple IDs preserves ambiguity; it does not prove which
        request owns any individual matmul row. Nested scopes are rejected.
        """
        if getattr(self._context_local, "active", False):
            raise RuntimeError("NESTED_REQUEST_CONTEXT_UNSUPPORTED")
        if not isinstance(request_ids, (list, tuple)) or not request_ids or any(not isinstance(v, str) or not v for v in request_ids):
            raise ValueError("Actual nonempty scheduler request IDs required")
        encoded = json.dumps(list(request_ids), ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        if len(encoded) > 4096:
            raise ValueError("Request context exceeds ABI capacity")
        code = self._symbols.fault39_set_context_v1(_bounded_text(owner, 64, "owner"), encoded)
        if code:
            raise NativeError(code, "set_context")
        self._context_local.active = True
        self._context_local.owner = owner
        try:
            yield
        finally:
            try:
                self._symbols.fault39_clear_context_v1()
            finally:
                self._context_local.active = False
                self._context_local.owner = None
