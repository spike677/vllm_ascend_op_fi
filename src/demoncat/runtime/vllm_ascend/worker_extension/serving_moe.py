"""Actual DeepSeek MoE combine deployment; no router or model reconstruction."""
from .serving_tensor import ServingTensorGroup
from .serving_context import require
from .providers.tensor_natural import _sha


class ServingMoEGroup(ServingTensorGroup):
    def __init__(self, worker, *, _trace_factory=None, **kwargs):
        from fault_injection.moe_census import NaturalMoETrace
        require(_trace_factory is None or kwargs.get('_simulation') is True,
                'MOE_TRACE_OVERRIDE_IS_CPU_CONTRACT_ONLY')
        self.trace_factory = _trace_factory or NaturalMoETrace.from_runtime
        model = worker.model_runner.model
        template = self.trace_factory(model)
        self.trace_source_identity = _sha(template.sources)
        self.structural_paths = tuple(path or '<root>' for path, module in model.named_modules()
            if hasattr(module, 'top_k') and callable(getattr(module, 'forward_impl', None)))
        require(self.structural_paths, 'NATURAL_STRUCTURAL_MOE_MODULES_REQUIRED')
        super().__init__(worker, _mechanism='8.restore_offset', **kwargs)
        self.provenance['moe_sources'] = template.sources
        self.provenance['structural_moe_paths'] = list(self.structural_paths)

    def _make_adapter(self, arm):
        from fault_injection.route8 import RestoreOffsetSession
        return RestoreOffsetSession(self.model, arm=arm, admission=self.provider._admission,
            strict_group_oracle=True, simulation=self.provider.simulation)

    def _make_observer(self):
        from fault_injection.moe_restore import MoEPhaseObserver
        template = self.trace_factory(self.model)
        require(_sha(template.sources) == self.trace_source_identity, 'NATURAL_MOE_SOURCE_OR_CALLABLE_CHANGED')
        return MoEPhaseObserver(self.model, template, self.provider, self.adapter, self.structural_paths)
