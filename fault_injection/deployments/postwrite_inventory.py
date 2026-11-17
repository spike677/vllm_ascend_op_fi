"""Admission-independent inventory of every original projection API and module.

The dispatcher calls this before it asks a postwrite injector to admit anything.
This module has no Native handles, hit counters, mutation modes or skip policy.
"""
from dataclasses import dataclass, field

from .postwrite_boundaries import PROJECTION_OPERATORS, tensor_geometry


@dataclass
class ProjectionCall:
    key: tuple
    module_path: str
    module_object_identity: int
    operator: str
    schema: str
    output_geometry: dict | None
    original_calls: int = 1
    candidate: object = None
    rejection: str | None = None

    def report(self):
        candidate = self.candidate
        return dict(key=self.key, module_path=self.module_path, module_object_identity=self.module_object_identity,
                    operator=self.operator, schema=self.schema, output_geometry=self.output_geometry,
                    original_calls=self.original_calls, rejection=self.rejection,
                    eligible=bool(candidate and candidate.consumers),
                    selected_indices=list(candidate.indices) if candidate else [],
                    consumers=list(candidate.consumers) if candidate else [])


class ProjectionWorkloadInventory:
    def __init__(self, *, max_calls=8192):
        self.max_calls = max_calls
        self.calls, self.module_calls, self.operator_counts = [], {}, {}
        self._ordinals = {}

    def module_entered(self, path, module):
        key = (path, id(module))
        self.module_calls[key] = self.module_calls.get(key, 0) + 1

    def original_returned(self, func, module_path, module, result):
        import torch
        operation = str(func)
        self.operator_counts[operation] = self.operator_counts.get(operation, 0) + 1
        if operation not in PROJECTION_OPERATORS:return None
        if len(self.calls) >= self.max_calls:raise RuntimeError('POSTWRITE_OBSERVER_CALL_BOUND_EXCEEDED')
        signature = (id(module), operation)
        ordinal = self._ordinals.get(signature, 0)
        self._ordinals[signature] = ordinal + 1
        record = ProjectionCall(signature + (ordinal,), module_path, id(module), operation,
                                str(func._schema), tensor_geometry(result) if isinstance(result, torch.Tensor) else None)
        self.calls.append(record)
        return record

    def report(self):
        return dict(calls=[call.report() for call in self.calls], operator_counts=dict(sorted(self.operator_counts.items())),
                    modules=[dict(path=path, object_identity=identity, calls=count)
                             for (path, identity), count in self.module_calls.items()],
                    admission_independent=True, dropped_calls=0)
