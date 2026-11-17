"""Explicit CPU ABI reports. These never stand for loaded-image/NPU evidence."""
import ctypes
import os

from fault_injection.metadata_arrays import COUNTERS, MetadataArrayStateV1, SCHEMA


def metadata_report(count, build_profile):
    rows = []
    for index, name in enumerate(('libopapi.so', 'libopapi_nn.so')):
        n = count if index == 0 else 0
        state = dict.fromkeys(COUNTERS, 0)
        state.update(version=1, size=ctypes.sizeof(MetadataArrayStateV1), instance=index + 1,
                     capture_calls=n, getter_calls=n * 3, acquired_arrays=n * 3,
                     released_arrays=n * 3, peak_live_arrays=3 if n else 0, revision=n * 20)
        rows.append(dict(name=name, path='/CPU-fixture/' + name, sha256='a' * 64,
                         symbol='EXPLICIT_CPU_MOCK', state=state))
    return dict(schema=SCHEMA, source='EXPLICIT_CPU_NATIVE_MOCK_NOT_NPU', pid=os.getpid(),
                build_profile_sha256=build_profile, build_manifest_sha256='f' * 64,
                images=rows, scope='SDK_GETTER_ARRAYS_ONLY', reset_supported=False)
