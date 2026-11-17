"""CPU-only ABI/guard tests against the production DSO. No fake vendor backend.

These tests do not claim a successful configure, Run, drain, or NPU lifetime test.
"""
import ctypes as C
import json
import sys
import threading


class Config(C.Structure):
    _fields_ = [(k, C.c_uint32) for k in ('abi_version', 'struct_size', 'mode', 'pattern')]
    _fields_ += [(k, C.c_int32) for k in ('exponent', 'rank', 'device')]
    _fields_ += [('max_hits', C.c_uint32), ('max_bytes', C.c_uint64), ('owner', C.c_char * 64),
                ('target_signature', C.c_char * 256), ('input_mask', C.c_uint32), ('reserved', C.c_uint32)]


lib = C.CDLL(sys.argv[1])
assert C.sizeof(Config) == 368
lib.fault39_configure_v1.argtypes = [C.POINTER(Config), C.POINTER(C.c_uint64)]
lib.fault39_configure_cache_v2.argtypes = [C.POINTER(Config), C.c_uint64, C.c_uint64, C.POINTER(C.c_uint64)]
lib.fault39_snapshot_v1.argtypes = [C.c_void_p, C.c_size_t]
lib.fault39_disable_v1.argtypes = [C.c_char_p, C.c_uint64, C.c_uint32]
lib.fault39_clean_v1.argtypes = [C.c_char_p, C.c_uint64, C.c_uint32]
lib.fault39_set_context_v1.argtypes = [C.c_char_p, C.c_char_p]
lib.fault39_set_serving_inputs_v1.argtypes = [C.c_char_p, C.POINTER(C.c_uint64), C.c_size_t]
lib.fault39_set_serving_inputs_v1.restype = C.c_int
lib.fault39_clear_context_v1.argtypes = []
lib.fault39_clear_context_v1.restype = None
lib.fault39_bind_worker_rank_v1.argtypes = [C.c_int32]
lib.fault39_build_profile_v1.restype = C.c_char_p


def snapshot():
    need = lib.fault39_snapshot_v1(None, 0)
    assert need > 0
    small = C.create_string_buffer(b'x' * 8, 9)
    assert lib.fault39_snapshot_v1(small, 9) == need
    assert small.value == b'x' * 8  # Never publish truncated JSON.
    buf = C.create_string_buffer(need)
    assert lib.fault39_snapshot_v1(buf, need) == need
    return json.loads(buf.value)


assert lib.fault39_abi_version_v1() == 1
initial = snapshot()
assert initial['enabled'] is False and initial['owner'] == ''
assert initial['generation'] == 0 and initial['inflight'] == 0 and initial['leases'] == 0
assert initial['runtime_ready'] is False
assert initial['runtime_identity'] == 'MISSING_OR_MISMATCHED_LIBRARIES'
assert initial['build_profile_sha256'] == lib.fault39_build_profile_v1().decode()
assert len(initial['build_profile_sha256']) == 64 and initial['cann_build_version']
cfg = Config(1, C.sizeof(Config), 1, 0, -1, 0, 0, 1, 1 << 20, b'cpu-test', b'', 3, 0)
generation = C.c_uint64(999)
cfg.abi_version = 0
assert lib.fault39_configure_v1(C.byref(cfg), C.byref(generation)) == -2
cfg.abi_version = 1
cfg.max_hits = 0
assert lib.fault39_configure_v1(C.byref(cfg), C.byref(generation)) == -1
cfg.max_hits = 1
assert lib.fault39_configure_v1(C.byref(cfg), C.byref(generation)) == -8
cfg.mode, cfg.input_mask = 2, 1
assert lib.fault39_configure_v1(C.byref(cfg), C.byref(generation)) == -1
assert generation.value == 999
assert initial['exact_cache_api_version'] == 2
assert initial['selected_cache_address'] == initial['selected_cache_epoch'] == 0
assert lib.fault39_configure_cache_v2(C.byref(cfg), 0, 1, C.byref(generation)) == -1
assert lib.fault39_configure_cache_v2(C.byref(cfg), 1, 0, C.byref(generation)) == -1
assert lib.fault39_configure_cache_v2(C.byref(cfg), 1, 1, C.byref(generation)) == -8
assert generation.value == 999
assert lib.fault39_bind_worker_rank_v1(-1) == -1
assert lib.fault39_bind_worker_rank_v1(3) == 0
assert lib.fault39_bind_worker_rank_v1(3) == 0
assert lib.fault39_bind_worker_rank_v1(4) == -4
assert snapshot()['worker_rank'] == 3
for invalid in (b'[]', b'[ ]', b'[null]', b'[""]', b'["a",]', b'["a" "b"]', b'["\\uQQQQ"]'):
    assert lib.fault39_set_context_v1(b'cpu-test', invalid) == -1, invalid
assert lib.fault39_set_context_v1(b'cpu-test', b'["actual-request-1","request\\u0032"]') == 0
lib.fault39_clear_context_v1()
# Synthetic addresses are never dereferenced: this setter is a CPU/TLS guard.
values = [0x1000, 32, 0x2000, 48, 2, 8, 3, 0, 0, 8, 1, 1, 8]
raw = (C.c_uint64 * 13)(*values)
assert lib.fault39_set_serving_inputs_v1(b'cpu-test', raw, 13) == -1
assert lib.fault39_set_context_v1(b'cpu-test', b'["guard-request"]') == 0
assert lib.fault39_set_serving_inputs_v1(b'foreign', raw, 13) == -4
thread_results = []
thread = threading.Thread(target=lambda: thread_results.append(
    lib.fault39_set_serving_inputs_v1(b'cpu-test', raw, 13)))
thread.start()
thread.join()
assert thread_results == [-1]
assert lib.fault39_set_serving_inputs_v1(b'cpu-test', raw, 13) == 0
assert lib.fault39_set_serving_inputs_v1(b'cpu-test', raw, 13) == -3
lib.fault39_clear_context_v1()
for index, invalid in ((0, 0), (1, 31), (1, 30), (2, 0x1000), (4, 0),
                       (5, 2**63), (7, 2**64 - 1), (9, 0), (12, 2**63)):
    assert lib.fault39_set_context_v1(b'cpu-test', b'["guard-invalid"]') == 0
    changed = list(values)
    changed[index] = invalid
    assert lib.fault39_set_serving_inputs_v1(b'cpu-test', (C.c_uint64 * 13)(*changed), 13) == -1
    assert lib.fault39_set_serving_inputs_v1(b'cpu-test', raw, 13) == -3
    lib.fault39_clear_context_v1()
assert lib.fault39_set_context_v1(b'cpu-test', b'["guard-short"]') == 0
assert lib.fault39_set_serving_inputs_v1(b'cpu-test', raw, 12) == -1
lib.fault39_clear_context_v1()
assert lib.fault39_set_context_v1(b'cpu-test', b'["guard-new-context"]') == 0
assert lib.fault39_set_serving_inputs_v1(b'cpu-test', raw, 13) == 0
lib.fault39_clear_context_v1()
assert snapshot()['last_observation']['serving_live_inputs_verified'] is False
before = snapshot()
assert lib.fault39_disable_v1(b'foreign-owner', 0, 0) == -4
assert lib.fault39_clean_v1(b'foreign-owner', 0, 0) == -4
after = snapshot()
assert after == before  # Rejected foreign control cannot alter status/evidence.
assert not after['enabled'] and after['generation'] == 0
print('fault39 production ABI/guard tests passed (no vendor execution)')
