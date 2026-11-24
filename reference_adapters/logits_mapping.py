"""Library12 column mapping proxy at the true sampler softmax reader.

Wait until legitimate temperature/mask writes finish, then lease the live
compute_logits storage for the softmax only. This does not restore over those
legitimate writes. Entropy is invariant under a column permutation; distribution
L1 measures the changed token mapping. No lm_head weight is modified.
"""
import torch
from torch.utils._python_dispatch import TorchDispatchMode, _disable_current_modes
from experiments.worker_core_v11_20261118.external_input_examples import BaseSite, changed
from fault_injection.sites.mla28 import require, fingerprint


class LogitsMode(TorchDispatchMode):
    def __init__(self, site):
        super().__init__(); self.site = site

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}; site = self.site
        if func._schema.name != 'aten::_softmax' or not args or not isinstance(args[0], torch.Tensor):
            return func(*args, **kwargs)
        tensor = args[0]
        if tensor.untyped_storage()._cdata != site.logits.untyped_storage()._cdata:
            return func(*args, **kwargs)
        require(tensor.shape == site.logits.shape and tensor.storage_offset() == site.logits.storage_offset()
            and tensor.is_contiguous() and tensor.ndim == 2 and tensor.shape[0] == 1,
            'REFERENCE_LOGITS_STORAGE_MAPPING')
        require(not site.forward_calls[-1], 'REFERENCE_LOGITS_MULTIPLE_SAMPLER_READERS')
        with _disable_current_modes():
            event = site.event('compute_logits.storage->sampler.softmax', tensor)
            event.update(target='all_vocab_columns_rotate_one', named_variant='row12_logits_columns_rotate',
                effect='PHENOMENON_PROXY_NOT_ORIGINAL_MECHANISM', producer_sha256=site.producer_sha,
                restore_boundary='AFTER_REAL_SOFTMAX_BEFORE_RETURN;AFTER_LEGITIMATE_TEMPERATURE_WRITES')
            before = tensor.detach().float().cpu().clone()
            value = torch.roll(tensor, 1, -1)
            event['before_sha256'] = fingerprint(tensor)
            event['changed_elements'] = changed(tensor, value) if site.fault else 0
            token = site.region_leases.acquire(tensor)
            if site.fault:
                site.region_leases.write(token, value)
                event.update(applied=True, changed=int(event['changed_elements'] > 0))
            event['after_sha256'] = fingerprint(tensor)
            result = func(*args, **kwargs)
            site.region_leases.consumed(token)
            event.update(consumed=True, consumer_operator=func._schema.name,
                consumer_sha256=fingerprint(tensor), probabilities_sha256=fingerprint(result))
            after = result.detach().float().cpu()
            normal_probs = before.softmax(-1)
            event['distribution_l1'] = float((after-normal_probs).abs().sum())
            if site.fault: site.region_leases.restore(token)
            else: site.region_leases.release_unmodified(token)
            event.update(restored=True, restored_sha256=fingerprint(tensor))
            require(event['before_sha256'] == event['restored_sha256'], 'REFERENCE_LOGITS_RESTORE')
            return result


class LogitsColumnsRotate(BaseSite):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mode = LogitsMode(self); self.logits = None; self.wrapper = None

    def install(self):
        require('compute_logits' not in vars(self.model), 'REFERENCE_LOGITS_ALREADY_PATCHED')
        original = self.model.compute_logits
        def compute(*args, **kwargs):
            value = original(*args, **kwargs)
            require(not self.active and self.logits is None and isinstance(value, torch.Tensor),
                'REFERENCE_LOGITS_PRODUCER_ORDER')
            self.logits = value; self.producer_sha = fingerprint(value)
            return value
        self.wrapper = compute; self.model.compute_logits = compute
        self.sites = ['compute_logits.storage->sampler.softmax']
        return self.sites

    def begin_sample(self):
        require(self.logits is not None, 'REFERENCE_LOGITS_NOT_FROM_REAL_MODEL')
        self.enter()

    def finish_sample(self):
        require(self.forward_calls[-1], 'REFERENCE_LOGITS_SOFTMAX_NOT_OBSERVED')
        super().finish_forward(); self.logits = None

    def finish_forward(self):
        pass  # Wait for the actual later sampler.

    def close(self):
        require(not self.active and self.logits is None and not self.pending, 'REFERENCE_LOGITS_PENDING')
        require(vars(self.model).get('compute_logits') is self.wrapper, 'REFERENCE_LOGITS_OWNER_CHANGED')
        del self.model.compute_logits
        super().close()


FACTORIES = {'row12_logits_columns_rotate': LogitsColumnsRotate}
