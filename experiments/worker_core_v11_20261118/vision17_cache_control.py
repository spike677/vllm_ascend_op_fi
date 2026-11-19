"""Public cache control for an isolated test Worker, never proof of execution."""
import hashlib
import inspect
from pathlib import Path


IDLE_PHASES = frozenset((
    'normal', 'fault', 'off', 'health',
    'healthy_0', 'healthy_1', 'native_baseline', 'baseline', 'baseline_repeat',
    'lossless', 'c2_fault', 'c2_off', 'c3_fault_0', 'c3_fault_1', 'c3_off',
    'cleanup_healthy', 'recovery_off', 'recovery_health',
))


def _public_reset(owner, name):
    method = getattr(owner, name, None)
    function = getattr(method, '__func__', None)
    if not callable(method) or function is None or getattr(method, '__self__', None) is not owner:
        raise RuntimeError('PUBLIC_BOUND_MM_CACHE_RESET_REQUIRED:' + name)
    path = inspect.getsourcefile(function)
    if path is None or not Path(path).is_file():
        raise RuntimeError('MM_CACHE_RESET_SOURCE_UNAVAILABLE:' + name)
    return method, function, Path(path), hashlib.sha256(Path(path).read_bytes()).hexdigest()


def reset_idle_mm_cache(llm, phase, *, reset_encoder_cache=False):
    if phase not in IDLE_PHASES or llm.llm_engine.has_unfinished_requests():
        raise RuntimeError('MM_CACHE_RESET_REQUIRES_IDLE_DECLARED_PHASE')
    if type(reset_encoder_cache) is not bool:
        raise RuntimeError('EXPLICIT_ENCODER_CACHE_RESET_FLAG_REQUIRED')
    planned = [_public_reset(llm, 'reset_mm_cache')]
    if reset_encoder_cache:
        planned.append(_public_reset(llm.llm_engine, 'reset_encoder_cache'))
    records = []
    for method, function, path, before in planned:
        if llm.llm_engine.has_unfinished_requests():
            raise RuntimeError('MM_CACHE_RESET_REQUIRES_IDLE_DECLARED_PHASE')
        result = method()
        if result is not None or llm.llm_engine.has_unfinished_requests():
            raise RuntimeError('MM_CACHE_RESET_RETURN_OR_IDLE_CHANGED')
        if hashlib.sha256(path.read_bytes()).hexdigest() != before:
            raise RuntimeError('MM_CACHE_RESET_SOURCE_CHANGED')
        records.append(dict(method=function.__qualname__, module=function.__module__, source_sha256=before))
    return dict(phase=phase, **records[0], public_resets=records,
                encoder_cache_reset_requested=reset_encoder_cache,
                returned_none=True, idle_before=True, idle_after=True,
                encoder_execution_proven=False, qualification_granted=False)
