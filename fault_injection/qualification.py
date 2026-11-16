"""Lossless admission for profiled providers, using the existing NativeSession."""
import copy
import hashlib
import json
import re

from .core import UnsupportedCall
from .environment import environment_fingerprint
from .native import NativeSession, _read_proofs
from .runtime_discovery import native_facts, plan_adaptation


_SHA = re.compile(r'[0-9a-f]{64}\Z')
_MODES = {'reference': 1, 'lossless': 2, 'fault': 3}
_REPLACE_FORMALS = {
    'aclnnMatmulWeightNz': ('self',),
    'aclnnQuantMatmulWeightNz': ('x1',),
}


def _digest_output(output):
    import torch

    # Deployment replay chooses the complete deterministic output under test,
    # typically logits. Reject sampling text or a summary as evidence.
    if not isinstance(output, torch.Tensor) or not output.numel() or output.layout != torch.strided:
        raise UnsupportedCall('Worker replay must return a complete nonempty output tensor')
    if output.numel() * output.element_size() > 512 * 1024 * 1024:
        raise UnsupportedCall('Worker replay output exceeds the byte verification budget')
    flat = output.detach().contiguous().reshape(-1).view(torch.uint8).cpu()
    digest = hashlib.sha256(json.dumps([str(output.dtype), list(output.shape)]).encode())
    digest.update(memoryview(flat.numpy()))
    return digest.hexdigest()


def _validate_receipt(receipt, stage, operator, input_count):
    if not isinstance(receipt, dict):
        raise UnsupportedCall('Native call returned no scoped receipt')
    formals = () if stage == 'reference' else _REPLACE_FORMALS[operator]
    expected = len(formals)
    fields = {'mode': _MODES[stage], 'original_calls': 1, 'native_status': 0, 'replacements': expected}
    if any(type(receipt.get(key)) is not int or receipt[key] != value for key, value in fields.items()):
        raise UnsupportedCall('Incorrect native execution count, mode or substitution count')
    if (receipt.get('resources_released') is not True
            or receipt.get('temporary_readback_verified') is not bool(expected)
            or receipt.get('replaced_formals') != list(formals)
            or receipt.get('device_copy') is not stage.startswith('lossless')):
        raise UnsupportedCall('Native replacement readback or cleanup was not confirmed')
    changed = receipt.get('changed_slots')
    expected_change = (1 <= changed <= expected if stage == 'fault' else changed == 0) if type(changed) is int else False
    if not expected_change:
        raise UnsupportedCall('Native replacement contents do not match the requested mode')
    before, after = receipt.get('input_sha256_before'), receipt.get('input_sha256_after')
    valid_hash = lambda value: isinstance(value, str) and _SHA.fullmatch(value)
    if (not isinstance(before, list) or len(before) != input_count
            or before != after or not all(valid_hash(value) for value in before)
            or not valid_hash(receipt.get('output_sha256'))):
        raise UnsupportedCall('Native input protection or complete output readback is unproven')
    return receipt


class QualifiedNativeSession(NativeSession):
    """Discover, prove one target lossless and admit at most one fault call."""

    def __init__(self, model, fault, binding, arm='fault', unsupported='error', evidence_dir=None):
        runtime = binding.factory(model)
        if not callable(getattr(runtime, 'execute_scoped', None)):
            raise UnsupportedCall('Provider does not implement scoped native continuation')
        super().__init__(model, fault, runtime=runtime, arm='observe', unsupported=unsupported,
                         evidence_dir=evidence_dir, profiles=binding.profiles)
        self.requested_arm = arm
        self.replay = binding.replay
        self.first_live_validation = binding.first_live_validation
        self.stage = 'discovery'
        self.qualification = {'status': 'PENDING', 'selected': None, 'replays': [],
                              'first_live_validation': self.first_live_validation}
        self._selection = self._selected_facts = None
        self._attempted = False
        self._input_reference = ()

    def _replay(self, stage, expected=None):
        import torch

        self.stage = stage
        forwards, events = len(self._forwards), len(self._events)
        with torch.inference_mode():
            result = self.replay(self.model)
        if len(self._forwards) != forwards + 1 or self._forwards[-1]['status'] != 'returned':
            raise UnsupportedCall('Worker replay must execute exactly one complete loaded model forward')
        current = _digest_output(result)
        row = {'stage': stage, 'model_output_sha256': current}
        self.qualification['replays'].append(row)
        if expected is not None and current != expected:
            raise UnsupportedCall('Replay diverged from the healthy complete model output: ' + stage)
        if stage == 'discovery':
            return current, None
        observations = [event for event in self._events[events:] if event.get('executed_stage') == stage]
        if len(observations) != 1:
            raise UnsupportedCall('Expected one verified selected native call during ' + stage)
        row['native'] = observations[0]['provider_observation']
        return current, row['native']

    def _matching_environment(self, observed):
        levels = {'NATIVE_VERIFIED', 'MODEL_VERIFIED', 'GENERALIZED'}
        if self.first_live_validation:
            levels.add('CPU_VERIFIED')
        return any(
            profile.environment_fingerprint == environment_fingerprint(observed, profile.required_libraries)
            and profile.validation_level in levels
            and (self._selection is None or
                 profile.environment_fingerprint == self._selection['environment_fingerprint'])
            for profile in self._profiles
        )

    def __enter__(self):
        if getattr(self.model, 'training', None) is not False:
            raise UnsupportedCall('Native inject requires an already loaded evaluation model')
        try:
            super().__enter__()
            if not self._matching_environment(self._environment_before):
                raise UnsupportedCall('Loaded worker has no validated environment for this mode')
            healthy, _ = self._replay('discovery')
            self._replay('discovery', healthy)
            self._environment_after = self._observe_environment()
            decision = plan_adaptation(super().report(), self._model_structure, self._profiles,
                                       self._environment_before, self._environment_after, discovery_finished=True,
                                       first_live_validation=self.first_live_validation)
            self.qualification['discovery'] = decision
            if decision['status'] != 'LOSSLESS_GATE_REQUIRED':
                raise UnsupportedCall('Automatic native target refused: ' + decision['status'])
            self._selection = decision['target']
            selected_call = self._selection['call_ids'][0]
            self._selected_facts = next(event['native_facts'] for event in self._events
                                        if event['call_id'] == selected_call)
            worker = self._selected_facts['worker']
            if (not self._selected_facts.get('cache_lifecycle')
                    or not worker.get('worker_id') or not worker.get('model_instance_id')):
                raise UnsupportedCall('Qualified injection requires live worker, model and cache lifecycle proofs')
            slots = self._selected_facts['slots']
            required = {'self', 'mat2', 'out'} if self._selection['operator'] == 'aclnnMatmulWeightNz' else {
                'x1', 'x2', 'out'
            }
            if not required <= {slot['formal'] for slot in slots} or len({slot['slot'] for slot in slots}) != len(slots):
                raise UnsupportedCall('Qualified injection requires distinct proven input and output slots')
            self.qualification['selected'] = self._selection
            _, reference = self._replay('reference', healthy)
            self._input_reference = tuple(reference['input_sha256_before'])
            for stage in ('lossless', 'off'):
                _, replay = self._replay(stage, healthy)
                for metric in ('input_sha256_before', 'input_sha256_after', 'output_sha256'):
                    if reference[metric] != replay[metric]:
                        raise UnsupportedCall('Native bytes differ from reference: ' + stage + '/' + metric)
            if not self._matching_environment(self._observe_environment()):
                raise UnsupportedCall('Loaded environment changed during lossless qualification')
            self.qualification['status'] = 'LOSSLESS_VERIFIED'
            self.stage = 'fault' if self.requested_arm == 'fault' else 'baseline'
            return self
        except BaseException as exc:
            self.stage = 'refused'
            self.qualification.update(status='REFUSED', reason=str(exc))
            self.close()
            raise

    def _observe_invocation(self, call, event, original):
        if self.stage == 'discovery':
            return super()._observe_invocation(call, event, original)
        selected = self._selection
        if (selected is None or id(call.owner) != selected['object_identity'] or call.operator != selected['operator']
                or self.stage in ('baseline', 'spent', 'refused')):
            return self._pass_through(event, original)
        try:
            if call.call_id in self._seen:
                raise UnsupportedCall('Reused native invocation token')
            self._seen.add(call.call_id)
            proofs, details = _read_proofs(call, self.fault)
            facts = native_facts(details.get('runtime_call'), details.keys(), self.fault)
            if facts['native_signature'] != selected['native_signature']:
                return self._pass_through(event, original)
            if facts != self._selected_facts:
                raise UnsupportedCall('Live worker, cache or native layout changed after discovery')
            if event['scope_ordinal'] != selected['scope_ordinal']:
                raise UnsupportedCall('Native call position changed within model forward')
            phase = self.stage
            if phase == 'fault':
                import torch

                if (self._attempted or torch.is_grad_enabled() or self.model.training
                        or not self._matching_environment(self._observe_environment())):
                    raise UnsupportedCall('Qualified fault requires the matched worker and an inference forward')
                # Consume admission before entering native code. Never retry
                # after an uncertain result or silently arm another call.
                self._attempted = True
                self.stage = 'spent'
            elif any(row.get('executed_stage') == phase and row['forward'] == event['forward']
                     for row in self._events):
                raise UnsupportedCall('Repeated selected call during ' + phase)
            stage = 'reference' if phase == 'off' else phase

            def once():
                if event['actual_original_calls']:
                    raise UnsupportedCall('Native continuation invoked more than once')
                event['actual_original_calls'] = 1
                return original()

            result, receipt = self.runtime.execute_scoped(call, once, _MODES[stage])
            if event['actual_original_calls'] != 1:
                raise UnsupportedCall('Native continuation was not invoked exactly once')
            input_count = sum(slot['formal'] != 'out' for slot in facts['slots'])
            _validate_receipt(receipt, stage, call.operator, input_count)
            if phase == 'fault' and tuple(receipt['input_sha256_before']) != self._input_reference:
                raise UnsupportedCall('Fault receipt did not match qualified input bytes')
            event.update(status='routed', executed_stage=phase, proofs=proofs, native_facts=facts,
                         provider_observation=receipt)
            if phase == 'fault':
                self.qualification['fault_execution'] = 'NATIVE_COMPLETION_REPORTED'
            return result
        except BaseException as exc:
            event.update(status='failed', reason=str(exc))
            if self._attempted or self.stage == 'fault':
                self.stage = 'refused'
                self.qualification.update(status='FAULT_REFUSED_OR_FAILED', reason=str(exc))
            raise

    @staticmethod
    def _pass_through(event, original):
        event.update(status='passthrough', actual_original_calls=1)
        try:
            return original()
        except BaseException as exc:
            event.update(status='failed', reason=type(exc).__name__)
            raise

    def report(self):
        report = super().report()
        report['observation_only'] = self.stage == 'discovery'
        report['qualification'] = copy.deepcopy(self.qualification)
        report['qualification']['phase'] = self.stage
        report['qualification']['fault_admitted'] = self._active and self.stage == 'fault'
        return report

    def assert_coverage(self):
        # This path qualifies one target, not every structural candidate.
        faults = [event for event in self._events if event.get('executed_stage') == 'fault']
        if len(faults) != 1:
            raise UnsupportedCall('No completed selected fault call')
        forward = self._forwards[faults[0]['forward']]
        if forward['status'] != 'returned' or forward['errors']:
            raise UnsupportedCall('Selected fault did not complete a verified model forward')
        return self.report()
