"""Library 7/8 fixed zero-row proxies; no MoE capacity or CP mechanism claim."""
import torch
from torch.utils._python_dispatch import _disable_current_modes
from experiments.worker_core_v11_20261118.external_input_examples import BaseSite, InputReadMode, changed
from fault_injection.sites.mla28 import require
from fault_injection.sites.projection39 import contains_element


class HiddenReadMode(InputReadMode):
    metadata_name='vllm::maybe_chunk_residual'
    def __torch_dispatch__(self,func,types,args=(),kwargs=None):
        if func._schema.name==self.metadata_name:
            # Actual 0.26 _maybe_chunk_residual_impl uses x.size(0) only and
            # returns/chunks residual. Its custom schema does not mark aliases.
            # Never count x as consumed or restore before npu_add_rms_norm.
            require(len(args)>=2,'REFERENCE_RESIDUAL_METADATA_SCHEMA')
            require(not any(contains_element(args[1],p['region']) for p in self.pending),
                'REFERENCE_RESIDUAL_ALIAS_PATH_UNSUPPORTED')
            return func(*args,**(kwargs or {}))
        return super().__torch_dispatch__(func,types,args,kwargs)


class HiddenRowSite(BaseSite):
    target_kind = None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mode = HiddenReadMode(self)

    @property
    def pending(self):
        return len(self.mode.pending)

    def install(self):
        self.hooks.append(self.model.register_forward_pre_hook(self.enter))
        visual = getattr(self.model, 'visual', None)
        excluded = {id(m) for m in visual.modules()} if visual is not None else set()
        for name, module in self.model.named_modules():
            if id(module) in excluded:
                continue
            mlp = all(hasattr(module, k) for k in ('gate_up_proj', 'down_proj', 'act_fn'))
            decoder = all(hasattr(module, k) for k in ('mlp', 'input_layernorm', 'post_attention_layernorm'))
            if not (mlp if self.target_kind == 'mlp' else decoder):
                continue
            self.sites.append(name)
            def output(mod, args, result, _name=name):
                require(self.active, 'REFERENCE_HIDDEN_OUTSIDE_FORWARD')
                t = result[0] if isinstance(result, tuple) else result
                require(isinstance(t, torch.Tensor) and t.dtype == torch.bfloat16
                    and t.ndim == 2 and t.is_contiguous() and t.shape[0] > 0,
                    'REFERENCE_HIDDEN_LAYOUT')
                with _disable_current_modes():
                    event = self.event(_name, t)
                    event.update(named_variant=type(self).__name__,
                        effect='PHENOMENON_PROXY_NOT_ORIGINAL_MECHANISM', target='last_actual_token_row')
                    def zero(region):
                        value = torch.zeros_like(region)
                        event['changed_elements'] = changed(region, value)
                        return value
                    self.mode.watch(t[-1], event, inject=self.fault, replacement=zero)
            self.hooks.append(module.register_forward_hook(output))
        require(self.sites, 'REFERENCE_NO_NATURAL_HIDDEN_BOUNDARY')
        return self.sites


class MlpContributionLost(HiddenRowSite):
    target_kind = 'mlp'


class DecoderPaddingRow(HiddenRowSite):
    target_kind = 'decoder'


FACTORIES = {'row7_mlp_last_row_zero': MlpContributionLost,
             'row8_decoder_last_row_zero': DecoderPaddingRow}
