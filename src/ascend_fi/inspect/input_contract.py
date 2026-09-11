"""Opt-in read-only input guards before the UNCHANGED reference replacement hook."""
from __future__ import annotations

import os
from .surface import stored, is_tensor
from .model_descriptor import describe_model
from .plan_builder import verify_plan


def check_input(binding, args, kwargs, predicate=is_tensor):
    path = binding['path']
    if len(path) != 2:
        raise ValueError('NESTED_BINDING_UNSUPPORTED')
    try:
        value = args[path[1]] if path[0] == 'args' else kwargs[path[1]]
    except (IndexError, KeyError):
        raise ValueError('BOUND_INPUT_PATH_MISSING') from None
    floats = [v for v in (*args, *kwargs.values()) if predicate(v) and v.is_floating_point()]
    if len(floats) != 1 or floats[0] is not value:
        raise ValueError('CALL_SEMANTIC_OR_ALIAS_CONTRACT_CHANGED')
    # Reject nested auxiliaries; the V1 resolver only certifies simple linear inputs.
    if any(type(v) in (list, tuple, dict) for v in (*args, *kwargs.values())):
        raise ValueError('CALL_STRUCTURE_CHANGED')
    if (str(value.dtype) != binding['dtype'] or str(value.layout) != binding['layout']
            or value.is_meta or len(value.shape) != binding['ndim']
            or value.shape[-1] != binding['feature_size'] or value.shape[0] <= 0
            or not value.is_contiguous()):
        raise ValueError('CALL_INPUT_METADATA_CHANGED')
    return value


class PlanInputContractMixin:
    """No numerical transforms, no alternate fault state. Explicit owner opt-in."""
    def fake_requant_bind_target_plan(self, plan=None, clear=False):
        if os.environ.get('VLLM_SERVER_DEV_MODE') != '1':
            return {'ok': False, 'error': 'DEVELOPER_MODE_REQUIRED'}
        try:
            if clear:
                if self.fake_requant_query().get('enabled') is not False:
                    raise ValueError('DISARM_BEFORE_CLEARING_INPUT_GUARDS')
                pending = []
                for handle in stored(self, '_target_plan_guards', []):
                    try:
                        handle.remove()
                    except Exception:
                        pending.append(handle)
                self._target_plan_guards = pending
                if pending:
                    raise ValueError('INPUT_GUARD_CLEANUP_FAILED')
                self._target_plan_guard_status = {'installed': False, 'handles': 0}
                return {'ok': True, **self._target_plan_guard_status}
            verify_plan(plan)
            if stored(self, '_target_plan_guards'):
                raise ValueError('INPUT_GUARDS_ALREADY_INSTALLED')
            state = self.fake_requant_query()
            if state.get('enabled') is not False or state.get('installed') is not False:
                raise ValueError('CLEAN_REFERENCE_STATE_REQUIRED')
            session = stored(self, '_demoncat_inspect_session_v1')
            if session is None or session.active or session.handles:
                raise ValueError('STOP_LIVE_INSPECT_BEFORE_BINDING')
            d, _ = describe_model(session, self)
            target = next((w for w in plan['workers'] if w['service_epoch'] == session.epoch), None)
            if target is None or target['model_fingerprint'] != d['model_fingerprint']:
                raise ValueError('STALE_MODEL_OR_PLAN_WORKER')
            self._target_plan_guards = []
            self._target_plan_guard_status = {'installed': True, 'plan_sha256': plan['plan_sha256'],
                                             'calls_checked': 0, 'violations': 0, 'last_error': None}
            try:
                for t in target['targets']:
                    module = session.surface.modules[t['module_name']]
                    binding = t['binding']
                    def guard(_module, args, kwargs, binding=binding):
                        try:
                            check_input(binding, args, kwargs)
                            self._target_plan_guard_status['calls_checked'] += 1
                        except Exception as e:
                            self._target_plan_guard_status['violations'] += 1
                            self._target_plan_guard_status['last_error'] = str(e)[:200]
                            raise RuntimeError('TARGET_PLAN_INPUT_CONTRACT_FAILED:' + str(e)) from e
                        return None
                    self._target_plan_guards.append(module.register_forward_pre_hook(guard, with_kwargs=True, prepend=True))
            except Exception:
                self.fake_requant_bind_target_plan(clear=True)
                raise
            self._target_plan_guard_status['handles'] = len(self._target_plan_guards)
            return {'ok': True, **self._target_plan_guard_status}
        except Exception as e:
            return {'ok': False, 'error': type(e).__name__ + ':' + str(e)[:300]}

    def fake_requant_query(self):
        value = super().fake_requant_query()
        value['input_contract'] = dict(stored(self, '_target_plan_guard_status',
                                              {'installed': False, 'handles': 0}))
        return value
