"""Trusted A2 example, not a public built-in or a yellow capability grant."""
import torch
from torch.utils._python_dispatch import _disable_current_modes
from fault_injection.sites.norm18 import Norm18Site
from fault_injection.sites.mla28 import require


class ExternalGainSite(Norm18Site):
    """Reuse source-checked discovery and true dispatch consumption, gain 3/4."""
    def output(self, name, tensor):
        require(self.active and self.forward_calls, 'EXTERNAL_GAIN_OUTSIDE_MODEL')
        require(tensor.dtype == torch.bfloat16 and tensor.numel() > 0 and tensor.ndim >= 1,
                'EXTERNAL_GAIN_BF16_REQUIRED')
        calls = self.forward_calls[-1]
        calls[name] = calls.get(name, 0) + 1
        with _disable_current_modes():
            region = torch.as_strided(tensor, (1,), (1,), tensor.storage_offset()
                                      + (tensor.shape[0] - 1) * tensor.stride(0))
            event = dict(site=name, request_ids=list(self.runner.input_batch.req_ids),
                         eligible=True, applied=False, changed=0, consumed=False,
                         restored=False, failed=False, effect='external_gain_three_quarters')
            self.events.append(event)
            self.mode.watch(region, event, inject=self.fault, replacement=lambda x: x * 0.75)

    def finish_forward(self):
        require(self.active and not self.pending and not self.parents,
                'EXTERNAL_GAIN_LOGITS_CONSUMER_PENDING')
        self.mode.__exit__(None, None, None)
        self.active = False
