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
        # vLLM Sampler.forward converts BF16 compute_logits to FP32 before
        # temperature and softmax. Track this actual conversion, not an
        # unrelated equal-shaped tensor. The lease is on the live FP32 reader.
        if (func._schema.name == 'aten::_to_copy' and args
                and same_live_logits_row(args[0], site.logits)
                and kwargs.get('dtype') == torch.float32):
            result = func(*args, **kwargs)
            require(result.shape == args[0].shape and result.device == site.logits.device,
                'REFERENCE_LOGITS_CAST_MAPPING')
            with _disable_current_modes():
                require(torch.equal(result, args[0].float()), 'REFERENCE_LOGITS_CAST_VALUES')
                site.cast_chain.append(dict(operator=func._schema.name,
                    input_sha256=fingerprint(args[0]), output_sha256=fingerprint(result),
                    producer_shape=list(site.logits.shape), sampler_shape=list(args[0].shape),
                    first_live_request_row=True))
            site.logits = result
            return result
        if func._schema.name != 'aten::_softmax' or not args or not isinstance(args[0], torch.Tensor):
            return func(*args, **kwargs)
        tensor = args[0]
        if not same_live_logits_row(tensor, site.logits):
            return func(*args, **kwargs)
        require(not site.forward_calls[-1], 'REFERENCE_LOGITS_MULTIPLE_SAMPLER_READERS')
        with _disable_current_modes():
            event = site.event('compute_logits.storage->sampler.softmax', tensor)
            event.update(target='all_vocab_columns_rotate_one', named_variant='row12_logits_columns_rotate',
                effect='PHENOMENON_PROXY_NOT_ORIGINAL_MECHANISM', producer_sha256=site.producer_sha,
                producer_to_sampler_casts=list(site.cast_chain),
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


def same_live_logits_row(tensor, producer):
    # NPUModelRunner._sample slices logits[:input_batch.num_reqs] when
    # lmhead_tp_enable(). With this site's single real request, that is row0.
    # Object identity changes even when the entire row's storage is identical.
    return (isinstance(tensor,torch.Tensor) and isinstance(producer,torch.Tensor)
        and tensor.ndim==producer.ndim==2 and tensor.shape[0]==1 and producer.shape[0]>=1
        and tensor.shape[1]==producer.shape[1] and tensor.dtype==producer.dtype
        and tensor.device==producer.device and tensor.is_contiguous() and producer.is_contiguous()
        and tensor.untyped_storage()._cdata==producer.untyped_storage()._cdata
        and tensor.storage_offset()==producer.storage_offset())


class LogitsColumnsRotate(BaseSite):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mode = LogitsMode(self); self.logits = None; self.wrapper = None
        self.cast_chain = []

    def install(self):
        require('compute_logits' not in vars(self.model), 'REFERENCE_LOGITS_ALREADY_PATCHED')
        original = self.model.compute_logits
        def compute(*args, **kwargs):
            value = original(*args, **kwargs)
            require(not self.active and self.logits is None and isinstance(value, torch.Tensor),
                'REFERENCE_LOGITS_PRODUCER_ORDER')
            self.logits = value; self.producer_sha = fingerprint(value); self.cast_chain = []
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
