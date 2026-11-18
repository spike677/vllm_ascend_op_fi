"""Public cache control for an isolated test Worker, never proof of execution."""
import hashlib
import inspect
from pathlib import Path


IDLE_PHASES = frozenset((
    'healthy_0', 'healthy_1', 'native_baseline', 'baseline', 'baseline_repeat',
    'lossless', 'c2_fault', 'c2_off', 'c3_fault_0', 'c3_fault_1', 'c3_off',
    'cleanup_healthy', 'recovery_off', 'recovery_health',
))


def reset_idle_mm_cache(llm, phase):
    if phase not in IDLE_PHASES or llm.llm_engine.has_unfinished_requests():
        raise RuntimeError('MM_CACHE_RESET_REQUIRES_IDLE_DECLARED_PHASE')
    method = getattr(llm, 'reset_mm_cache', None)
    function = getattr(method, '__func__', None)
    if not callable(method) or function is None or getattr(method, '__self__', None) is not llm:
        raise RuntimeError('PUBLIC_BOUND_MM_CACHE_RESET_REQUIRED')
    path = inspect.getsourcefile(function)
    if path is None or not Path(path).is_file():
        raise RuntimeError('MM_CACHE_RESET_SOURCE_UNAVAILABLE')
    before = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    result = method()
    if result is not None or llm.llm_engine.has_unfinished_requests():
        raise RuntimeError('MM_CACHE_RESET_RETURN_OR_IDLE_CHANGED')
    if hashlib.sha256(Path(path).read_bytes()).hexdigest() != before:
        raise RuntimeError('MM_CACHE_RESET_SOURCE_CHANGED')
    return dict(phase=phase, method=function.__qualname__, module=function.__module__,
                source_sha256=before, returned_none=True, idle_before=True, idle_after=True,
                encoder_execution_proven=False, qualification_granted=False)
