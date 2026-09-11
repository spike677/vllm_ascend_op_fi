"""Opt-in bounded observer over the PINNED, unmodified reference transform.

Only use in a new, isolated experimental worker epoch. The original combined
extension and original runtime remain byte-for-byte unchanged.
"""
import time
from .combined import FakeRequantReferenceExtension
from .extension import InspectWorkerExtension
from .effect_stats import EffectStats
from .input_contract import PlanInputContractMixin


class ExplorationWorkerExtension(PlanInputContractMixin, InspectWorkerExtension, FakeRequantReferenceExtension):
    @property
    def _state(self):
        state = FakeRequantReferenceExtension._state.fget(self)
        if not isinstance(state.tensor_effect_history, EffectStats):
            if state.tensor_effect_history:
                raise RuntimeError('REQUIRES_FRESH_REFERENCE_STATE')
            state.tensor_effect_history = EffectStats()
        return state

    def fake_requant_query(self):
        value = super().fake_requant_query()
        with self._state.lock:
            value['effect_summary'] = self._state.tensor_effect_history.snapshot()
        return value

    def _requant_tensor(self, module, tensor, module_name, layer, rank, rank_source, hit_index, config):
        # The numeric output is returned by the ORIGINAL function, unaltered.
        faulty, effect = FakeRequantReferenceExtension._requant_tensor(
            self, module, tensor, module_name, layer, rank, rank_source, hit_index, config)
        effect['wall_time_ns'] = time.time_ns()
        effect['request_generation'] = config.request_generation
        effect['candidate_index'] = self._state.candidate_seen
        prior = self._state.per_module[module_name]['hit_count']
        if prior % 64 == 0:
            import torch
            with torch.no_grad():
                working, result = tensor.detach().float(), faulty.detach().float()
                # FP32 difference/sums, not BF16 subtraction; diagnostics only.
                diff = result - working
                packed = torch.stack(((working != 0).sum().float(),
                                      ((working != 0) & (result == 0)).sum().float(),
                                      working.square().sum(), diff.square().sum())).cpu().tolist()
            effect['dose_sample'] = {'elements': tensor.numel(), 'nonzero': int(packed[0]),
                'new_zero': int(packed[1]), 'input_sq': packed[2], 'error_sq': packed[3],
                'relative_l2': (packed[3] / packed[2]) ** .5 if packed[2] > 0 else None,
                'precision': 'FP32', 'module_call_ordinal': prior + 1}
        return faulty, effect

    def _identity_effect(self, *args, **kwargs):
        effect = FakeRequantReferenceExtension._identity_effect(self, *args, **kwargs)
        effect['wall_time_ns'] = time.time_ns()
        return effect
