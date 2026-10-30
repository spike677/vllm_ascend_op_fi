"""Public torch_npu allocator metering outside statistics capture.

Call-window wall time is a conservative device-acquisition duration, including
Python and collection overhead through synchronized completion. It is not the
sum of kernel durations. Allocator peaks are process/device tensor allocations,
including already live inputs; they are not whole-card HBM or container RAM.
"""
import copy
import hashlib
import math
from pathlib import Path
import time

MEMORY_MODULE_SHA256 = 'b489a7dc22b77d5f50b3329801fe64cb4c7f908cf2abaea75c0efbd7f4c436c1'
ALLOCATOR_PEAK_KEY = 'allocated_bytes.all.peak'
PUBLIC_APIS = ('reset_peak_memory_stats', 'memory_stats', 'max_memory_allocated')


def begin_window(torch_npu, device=0):
    """Run after pre-call synchronize and before debugger.start()."""
    namespace = torch_npu.npu
    row = {'schema': 'nes.resource_measurement.v1', 'logical_device': device,
           'public_api_available': {name: callable(getattr(namespace, name, None)) for name in PUBLIC_APIS},
           'reset_outside_msprobe_capture': True, 'peak_reset_completed': False,
           'audited_memory_module_sha256': MEMORY_MODULE_SHA256,
           'allocator_metric': ALLOCATOR_PEAK_KEY,
           'allocator_scope': 'Current process, selected logical NPU, allocated tensor bytes; includes already-live inputs.',
           'whole_card_hbm_measured': False, 'container_host_memory_measured': False,
           'device_seconds_semantics': 'Conservative device-acquisition wall seconds: original API entry through synchronized completion; includes Python/collection overhead, not kernel execution time.',
           'errors': []}
    module_path = getattr(getattr(namespace, 'memory', None), '__file__', None)
    if module_path:
        try:
            path = Path(module_path)
            actual = hashlib.sha256(path.read_bytes()).hexdigest()
            row['actual_memory_module'] = {'path': str(path), 'sha256': actual, 'matches_audited_source': actual == MEMORY_MODULE_SHA256}
        except (OSError, TypeError) as error:
            row['errors'].append({'stage': 'source_metadata', 'error_type': type(error).__name__, 'message': str(error)})
    if not all(row['public_api_available'].values()):
        row['errors'].append({'stage': 'availability', 'reason': 'Required public allocator API unavailable; peak remains unknown.'})
        return row
    row['reset_started_monotonic_ns'] = time.monotonic_ns()
    try:
        namespace.reset_peak_memory_stats(device)
        row['peak_reset_completed'] = True
        row['reset_completed_monotonic_ns'] = time.monotonic_ns()
    except Exception as error:
        row['errors'].append({'stage': 'reset', 'error_type': type(error).__name__, 'message': str(error)})
    return row


def finish_window(torch_npu, started, call):
    """Sample after debugger.stop(); do not perform any extra NPU computation."""
    row = copy.deepcopy(started)
    row['sample_outside_msprobe_capture'] = True
    row['resources'] = {'device_seconds': None, 'peak_memory_bytes': None}
    enter, exit = call.get('enter_monotonic_ns'), call.get('exit_monotonic_ns')
    if call.get('status') == 'returned' and type(enter) is int and type(exit) is int and exit >= enter:
        elapsed = (exit - enter) / 1_000_000_000
        if math.isfinite(elapsed):
            row['resources']['device_seconds'] = elapsed
            row['call_window_monotonic_ns'] = {'enter': enter, 'exit_after_synchronize': exit, 'elapsed': exit - enter}
    if row['resources']['device_seconds'] is None:
        row['errors'].append({'stage': 'elapsed', 'reason': 'No complete synchronized monotonic call window; duration remains unknown.'})
    row['sample_started_monotonic_ns'] = time.monotonic_ns()
    if row['peak_reset_completed']:
        try:
            stats = torch_npu.npu.memory_stats(row['logical_device'])
            # The installed getter defaults to 0 when this key is absent. Such a
            # fallback is not a measured zero and must remain unobservable.
            peak = stats.get(ALLOCATOR_PEAK_KEY) if isinstance(stats, dict) else None
            if type(peak) is not int or peak < 0:
                row['errors'].append({'stage': 'allocator_stats', 'reason': 'Allocator peak key missing or invalid; peak remains unknown.'})
            else:
                public_peak = torch_npu.npu.max_memory_allocated(row['logical_device'])
                if type(public_peak) is not int or public_peak < 0:
                    row['errors'].append({'stage': 'allocator_getter', 'reason': 'Public max_memory_allocated returned an invalid value.'})
                else:
                    row['allocator_observations'] = {'memory_stats_peak_bytes': peak, 'max_memory_allocated_bytes': public_peak}
                    row['resources']['peak_memory_bytes'] = max(peak, public_peak)
        except Exception as error:
            row['errors'].append({'stage': 'allocator_sample', 'error_type': type(error).__name__, 'message': str(error)})
    row['sample_completed_monotonic_ns'] = time.monotonic_ns()
    row['allocator_window_semantics'] = 'Reset before capture through post-stop sampling; conservative peak including any collection allocations and already-live input tensors.'
    row['accounting_complete'] = all(value is not None for value in row['resources'].values())
    row['accounting_status'] = 'MEASURED_WINDOW_AND_ALLOCATOR_PEAK' if row['accounting_complete'] else 'RESOURCE_ACCOUNTING_UNKNOWN'
    row['hard_limits_verified_by_this_meter'] = False
    return row
