"""Read-only current source/operator windows for the four-container diagnostic."""
import sys
import threading
import time

import torch

from .quant_row_provenance import metadata, tensor_values, snapshot
from .quant_token_routes import SCHEMAS, OPERATORS, _formals, _describe
from .serving_context import require

STAGES = dict(routing=(1, 'route_dispatch'), gmm1=(2, 'rchain_gmm1'),
              gmm2=(3, 'rchain_gmm2'), unpermute=(4, 'route_combine'))


class ContainerScopes:
    def __init__(self, boundary, native):
        self.boundary, self.native = boundary, native
        self.active = None; self.errors = []; self.calls = 0; self.elapsed_ns = 0

    def before_operator(self, state):
        started = time.monotonic_ns()
        name = self.boundary.operator_aliases.get(state['func'], str(state['func']))
        stage = OPERATORS.get(name)
        if stage not in STAGES:
            return
        try:
            b, g = self.boundary, self.boundary.graph
            require(self.active is None and b.active and not b.ending and g.active and not g.ending
                and not b.errors and not g.errors and not g.overflow and sys.getprofile() is b.callback
                and threading.get_ident() == b.thread and torch.is_inference_mode_enabled()
                and not torch.is_grad_enabled() and not b.model.training,
                'CONTAINER_CURRENT_SOURCE_OBSERVER_REQUIRED')
            b._verify_bindings()
            require(str(state['func']._schema) == SCHEMAS[stage], 'CONTAINER_ACTUAL_OPERATOR_SCHEMA_CHANGED')
            number, role = STAGES[stage]
            frames = [row for row in b.stack if row['role'] == role]
            require(len(frames) == 1 and self.calls < 128, 'CONTAINER_CURRENT_SOURCE_FRAME_OR_BUDGET_REQUIRED')
            frame = frames[0]; bound, origins = _formals(state['func'], state['args'], state['kwargs'])
            evidence = dict(forward_id=g.forward_id, request_ids=list(b.scope['request_ids']),
                stage=stage, operator=name, schema=SCHEMAS[stage], operator_sequence=state['event']['sequence'],
                boundary_sequence=frame['sequence'], function=frame['function'], source_sha256=frame['source_sha256'],
                formals={key: _describe(value) for key, value in bound.items()}, argument_origins=origins,
                input_objects=[dict(tensor=metadata(t), write_epoch=g.epochs.get(g.storage_key(metadata(t)), 0))
                               for t in tensor_values((state['args'], state['kwargs']))])
            self.active = state; state['container_trace_scope'] = True
            self.native.enter(forward=1, sequence=state['event']['sequence'], boundary=frame['sequence'],
                              stage=number, evidence=evidence)
            self.calls += 1
        except BaseException as error:
            if self.active is state and self.native.active_scope is None:
                self.active = None; state.pop('container_trace_scope', None)
            self.errors.append(type(error).__name__ + ':' + str(error))
        finally:
            self.elapsed_ns += time.monotonic_ns() - started

    def after_operator(self, state, error=None):
        if state is None or not state.get('container_trace_scope'):
            return
        started = time.monotonic_ns()
        try:
            require(state is self.active, 'CONTAINER_PUBLIC_SCOPE_RETURN_ORDER_CHANGED')
            row = self.native.active_scope
            g = self.boundary.graph
            row['after_input_metadata'] = [dict(tensor=metadata(t), write_epoch=g.epochs.get(g.storage_key(metadata(t)), 0))
                for t in tensor_values((state['args'], state['kwargs']))]
            self.native.leave(original_calls=state['event']['original_calls'],
                              error=None if error is None else type(error).__name__)
            self.active = None
        except BaseException as failure:
            self.errors.append(type(failure).__name__ + ':' + str(failure))
        finally:
            self.elapsed_ns += time.monotonic_ns() - started

    def report(self):
        return snapshot(dict(scope='FOUR_CONTAINER_SOURCE_WINDOWS_NOT_ROW_PROOF',
            calls=self.calls, active=int(self.active is not None), errors=self.errors, elapsed_ns=self.elapsed_ns,
            native_qualified=False, request_rows_proven=False))

    def close(self):
        if self.active is not None:
            self.native.leave(original_calls=self.active['event']['original_calls'],
                              error='CLEANUP_RETRY_AFTER_SCOPE_FAILURE')
            self.active = None
