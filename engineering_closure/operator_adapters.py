"""E3 operator entry points over the existing dispatch and provider lifecycles."""
from dataclasses import asdict, dataclass
from types import MappingProxyType

from fault_injection.core import UnsupportedCall
from fault_injection.coverage import BoundaryTargetIdentity, CandidateSet, TargetIdentity
from fault_injection.runtime_discovery import RuntimeCandidateInventory


@dataclass(frozen=True)
class OperatorPath:
    family: str
    controller: str
    lifecycle: str


_MODEL = 'fault_injection.lifecycle.ModelFaultBackend'
_TENSOR = 'demoncat.runtime.vllm_ascend.worker_extension.providers.tensor_natural.NaturalTensorGroupProvider'
_POSTWRITE = 'fault_injection.deployments.postwrite'
_KV = 'demoncat.runtime.vllm_ascend.worker_extension.postwrite41.Postwrite41WorkerProvider'

# The first path is the existing default. Alt routes are explicit because the
# difference can include the actual writer, scope, device ABI and OFF contract.
OPERATOR_PATHS = MappingProxyType({
    '8.restore_offset': (OperatorPath('MoE Unpermute', 'fault_injection.route8.RestoreOffsetSession', 'dispatch'),),
    '15.alias_detach': (OperatorPath('AsStrided', _MODEL, 'model'),),
    '17.spatial_padding': (OperatorPath('Conv3d', _MODEL, 'model'),),
    '18.api_half': (OperatorPath('RMSNorm', _MODEL, 'model'), OperatorPath('RMSNorm', _TENSOR, 'group')),
    '20.buffer_substitute': (OperatorPath('MatMul / QMM', _MODEL, 'model'),),
    '22.wrong_row': (OperatorPath('RoPE / IndexSelect', 'fault_injection.index22.WrongRowSession', 'dispatch'),
                     OperatorPath('RoPE / IndexSelect', _TENSOR, 'group')),
    '25.reduce_scatter': (OperatorPath('ReduceScatter', 'fault_injection.collective25.ReduceScatterSession', 'collective'),),
    '26.residual_stomp': (OperatorPath('MatMul / QMM', _MODEL, 'model'),),
    '26.fia_destination': (OperatorPath('FIA destination', _MODEL, 'model'),),
    '28.fa.legal_tiling': (OperatorPath('FA', 'fault_injection.strategy.NativeStrategyBridge', 'strategy'),),
    '28.fia.legal_tiling': (OperatorPath('FIA tiling', _MODEL, 'model'),),
    '28.fia.denominator_replay': (OperatorPath('FIA kernel', _MODEL, 'model'),),
    '33.paged_decode': (OperatorPath('Paged IFA', _MODEL, 'model'),),
    '39.matmul_range': (OperatorPath('MatMul / QMM', 'demoncat_fault39.backend.NativeBackend', 'native39'),),
    '39.output_proxy': (OperatorPath('MatMul / QMM', _POSTWRITE + '.Fault39OutputProxyProvider', 'postwrite'),),
    '41.tail_column': (OperatorPath('KV Cache', _KV, 'kv'),
                       OperatorPath('KV Cache', _POSTWRITE + '.Fault41TailColumnProvider', 'postwrite')),
    'quant.batch_shared_scale': (OperatorPath('DynamicQuant', _MODEL, 'model'),),
})


class OperatorAdapter:
    """Select observed identities and attach an already configured provider.

    Providers still hold every live gate. Candidate selection here supplies no
    raw address, no native permission, and does not bypass provider discovery.
    """

    def __init__(self, mechanism):
        if mechanism not in OPERATOR_PATHS:
            raise UnsupportedCall('OPERATOR_ADAPTER_NOT_IMPLEMENTED:' + str(mechanism))
        self.mechanism = mechanism
        self.paths = OPERATOR_PATHS[mechanism]

    def discover(self, inventory, observer, *, artifact_sha256):
        if not isinstance(inventory, RuntimeCandidateInventory):
            raise TypeError('Use a RuntimeCandidateInventory populated by the actual observer')
        return inventory.seal(observer, artifact_sha256=artifact_sha256)

    def identify(self, candidates, policy='unique', *, target=None, priority=None):
        if not isinstance(candidates, CandidateSet):
            raise TypeError('Expected the provider candidate set')
        return candidates.select(policy, target=target, priority=priority)

    def inspect_metadata(self, target):
        if not isinstance(target, (TargetIdentity, BoundaryTargetIdentity)):
            raise TypeError('Expected an actual typed observation identity')
        return dict(mechanism=self.mechanism, target=asdict(target),
                    execution_boundary=target.execution_boundary, execution_authorized=False)

    def bind_target(self, controller):
        classes = {t.__module__ + '.' + t.__qualname__ for t in type(controller).__mro__}
        for path in self.paths:
            if path.controller in classes:
                break
        else:
            raise UnsupportedCall('OPERATOR_CONTROLLER_DOES_NOT_MATCH_ROUTE:' + self.mechanism)
        if path.lifecycle == 'model':
            registered = getattr(controller.config, 'mechanism', None)
            if registered != self.mechanism:
                raise UnsupportedCall('OPERATOR_MODEL_CONFIG_DOES_NOT_MATCH_MECHANISM')
        elif path.lifecycle in ('group', 'dispatch', 'collective', 'postwrite'):
            if controller.mechanism != self.mechanism:
                raise UnsupportedCall('OPERATOR_CONTROLLER_DOES_NOT_MATCH_MECHANISM')
        elif path.lifecycle == 'strategy':
            from native_execution_strategy.adapters.flash_attention import FlashAttentionAdapter

            if not isinstance(controller.session.adapter, FlashAttentionAdapter):
                raise UnsupportedCall('FA_STRATEGY_CONTROLLER_REQUIRED')
        return BoundOperator(controller, path.lifecycle, self.mechanism)


class BoundOperator:
    """Thin lifecycle bridge. Its methods return the concrete provider receipts."""

    def __init__(self, controller, lifecycle, mechanism):
        self.controller = controller
        self.lifecycle = lifecycle
        self.mechanism = mechanism

    def _active_controller(self):
        if self.lifecycle == 'model' and getattr(self.controller.config, 'mechanism', None) != self.mechanism:
            raise UnsupportedCall('OPERATOR_MODEL_CONFIG_CHANGED')
        return self.controller

    def discover(self, *args, **kwargs):
        controller = self._active_controller()
        if self.lifecycle in ('postwrite', 'group'):
            return controller.qualify(*args, **kwargs)
        if self.lifecycle == 'native39':
            return controller.discover(*args, **kwargs)
        raise UnsupportedCall('DISCOVERY_REQUIRES_ACTUAL_OBSERVER_OR_PROVIDER_REPLAY')

    def observe(self, *args, phase=None, **kwargs):
        controller = self._active_controller()
        if phase is not None and self.lifecycle != 'kv':
            raise ValueError('Only the KV request route accepts a phase override')
        if self.lifecycle == 'strategy':
            return controller.execute_attention(*args, **kwargs, mode='native')
        if self.lifecycle == 'native39':
            return controller.observe_inventory(*args, **kwargs)
        if self.lifecycle == 'postwrite':
            return controller.run('baseline', *args, **kwargs)
        if self.lifecycle == 'kv':
            phase = 'observe' if phase is None else phase
            if phase not in ('observe', 'lossless', 'baseline', 'off'):
                raise UnsupportedCall('KV_OBSERVATION_CANNOT_ARM_FAULT')
            return controller.expect(phase, *args, **kwargs)
        if self.lifecycle == 'collective' and controller.mode_name in ('observe', 'off'):
            return controller.__enter__(*args, **kwargs)
        if self.lifecycle == 'model' and controller.config.arm != 'fault':
            return controller.enable(*args, **kwargs)
        if self.lifecycle == 'dispatch' and controller.arm == 'baseline':
            return controller.__enter__(*args, **kwargs)
        raise UnsupportedCall('READ_ONLY_OBSERVATION_REQUIRES_AN_OBSERVER_OR_BASELINE_PATH')

    def inject(self, *args, **kwargs):
        controller = self._active_controller()
        if self.lifecycle == 'native39' or self.lifecycle == 'model' and controller.config.arm == 'fault':
            return controller.enable(*args, **kwargs)
        if self.lifecycle == 'postwrite':
            return controller.run('fault', *args, **kwargs)
        if self.lifecycle == 'group':
            return controller.run(*args, **kwargs)
        if self.lifecycle == 'strategy':
            return controller.execute_attention(*args, **kwargs, mode='controlled')
        if self.lifecycle == 'kv':
            return controller.expect('fault', *args, **kwargs)
        if (self.lifecycle == 'dispatch' and controller.arm == 'fault'
                or self.lifecycle == 'collective' and controller.mode_name == 'replay_delta'):
            return controller.__enter__(*args, **kwargs)
        raise UnsupportedCall('INJECTION_REQUIRES_FAULT_CONFIGURATION')

    def verify_effect(self):
        controller = self._active_controller()
        if self.lifecycle in ('model', 'native39', 'postwrite'):
            return controller.query()
        if self.lifecycle == 'kv':
            return controller.finish_request()
        if self.lifecycle == 'group':
            if controller.group is None:
                raise UnsupportedCall('NATURAL_GROUP_NOT_OPEN')
            return controller.group.report()
        return controller.report()

    def restore(self, *args, **kwargs):
        controller = self._active_controller()
        if self.lifecycle in ('model', 'native39'):
            return controller.disable(*args, **kwargs)
        if self.lifecycle == 'postwrite':
            controller.recover()
            return controller.run('off', *args, **kwargs)
        if self.lifecycle == 'kv':
            return controller.recover(*args, **kwargs)
        if self.lifecycle == 'strategy':
            return self.observe(*args, **kwargs)
        return self.cleanup(*args, **kwargs)

    def cleanup(self, *args, **kwargs):
        controller = self._active_controller()
        if self.lifecycle in ('model', 'native39'):
            return controller.clean(*args, **kwargs)
        if self.lifecycle == 'postwrite':
            return controller.recover(*args, **kwargs)
        if self.lifecycle == 'strategy':
            # The bridge owns a per-invocation context that expires on return.
            return controller.report()
        if self.lifecycle == 'group':
            return controller.close_group(*args, **kwargs)
        return controller.close(*args, **kwargs)
