"""FA semantics and typed selector plans. Imports no torch, driver or NPU library."""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
import json
import math
import os
from pathlib import Path
import re
import struct
import threading

from ..certificates import load_certificate
from ..models import (BuildFingerprints, EvidenceRef, InvocationContext, NativeStrategy,
                      ResourceUsage, StrategyObservation, canonical_sha256, require)
from ..policies import PolicyDecision
from .fa_schema import (POD_SCHEMA_SHA256, decode, execution_plan, input_contract,
                        validate_execution_plan)

SOURCE_COMMIT = 'c77c0829f1f0f0b8f882db1d9b051a758b605472'
CANN_BUILD = 'V100R001C11SPC001B243'
BRIDGE_ABI = 'nes.fa.cann9.1.s1s2.v1'
TILING_KEY = '22619644976'
EFFECTIVE_SCALE = struct.unpack('<f', struct.pack('<f', 1 / math.sqrt(128)))[0]
ENV_KEYS = ('NES_FA_PLAN_JSON', 'NES_FA_LOG_PATH')
TARGETS = ('query_tile', 'key_group', 'joint')
_ENV_LEASE = threading.Lock()


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def strategy_from_pod(pod: dict, *, raw: bytes) -> NativeStrategy:
    m, ratio, tile = (pod[key] for key in ('coreParams.s1BaseSize', 'coreParams.nRatio', 'coreParams.s2BaseSize'))
    return NativeStrategy(f'fa.s1s2.m{m}.r{ratio}', {'query_tile': m, 'key_group_ratio': ratio,
                          'key_tile': tile, 'execution_plan': execution_plan(pod, raw=raw)})


def selector_projection(strategy: NativeStrategy) -> dict:
    return {key: strategy.parameters[key] for key in ('query_tile', 'key_group_ratio', 'key_tile')}


@contextmanager
def _environment(plan: str, log_path: str):
    require(_ENV_LEASE.acquire(blocking=False), 'Concurrent FA environment scopes require separate processes')
    try:
        require(all(os.environ.get(key) is None for key in ENV_KEYS), 'Inherited or nested FA scope is forbidden')
        os.environ['NES_FA_PLAN_JSON'], os.environ['NES_FA_LOG_PATH'] = plan, log_path
        try:
            yield
        finally:
            for key in ENV_KEYS:
                os.environ.pop(key, None)
    finally:
        _ENV_LEASE.release()


class FlashAttentionAdapter:
    name = 'FlashAttentionScore'
    route = 'S1S2'

    def __init__(self, certificate_path: str | Path, *, evidence_root: str | Path,
                 selection_target: str = 'query_tile', allow_experimental: bool = False,
                 on_unsupported: str = 'reject'):
        require(selection_target in TARGETS, 'Unsupported FA selection target')
        require(on_unsupported in ('reject', 'passthrough'), 'Invalid on_unsupported')
        self.loaded = load_certificate(certificate_path, evidence_root=evidence_root,
                                       allow_experimental=allow_experimental)
        self.certificate = self.loaded.data
        self.version = self.certificate['adapter']['version']
        self.fingerprints = self.loaded.fingerprints
        self.selection_target = selection_target
        self.on_unsupported = on_unsupported
        self._native: dict[str, StrategyObservation] = {}
        self._observation_dirs: set[Path] = set()
        runtime = self.certificate['runtime']
        require(runtime['source_commit'] == SOURCE_COMMIT and runtime['cann_build'] == CANN_BUILD
                and runtime['bridge_abi'] == BRIDGE_ABI, 'FA runtime/version is not supported by this adapter')
        require(self.certificate['projection_schema_sha256'] == POD_SCHEMA_SHA256, 'Wrong public POD projection')
        self._contexts = {row['request_count']: row for row in self.certificate['contexts']}

    def bind_evidence_directory(self, directory: str | Path) -> None:
        """Explicit readonly import boundary for previously captured observations."""
        path = Path(directory).resolve(strict=True)
        require(path.is_dir(), 'Observation root must be a directory')
        self._observation_dirs.add(path)

    def inspect(self, actual_call: dict) -> InvocationContext:
        """Interpret named BNSD axes and explicit workload metadata; never generic shape[0]."""
        require(actual_call.get('operator') == self.name, 'Wrong actual operator')
        request_count = actual_call.get('request_count')
        require(request_count is None or type(request_count) is int and request_count > 0, 'Invalid explicit request_count')
        attrs = dict(actual_call.get('attributes', {}))
        require(type(attrs.get('scale')) in (int, float) and math.isfinite(attrs['scale']), 'Missing finite actual scale')
        attrs['scale'] = struct.unpack('<f', struct.pack('<f', attrs['scale']))[0]
        attrs['device'] = actual_call.get('device')
        inputs = actual_call.get('inputs', {})
        for name in ('query', 'key', 'value', 'atten_mask'):
            require(type(inputs.get(name)) is dict, 'Missing actual tensor metadata: ' + name)
        attrs['tensor_shapes'] = {name: inputs[name].get('shape') for name in inputs}
        attrs['tensor_dtypes'] = {name: inputs[name].get('dtype') for name in inputs}
        dims = {}
        if actual_call.get('layout') == 'BNSD' and all(type(inputs[key].get('shape')) is list and len(inputs[key]['shape']) == 4 for key in ('query', 'key', 'value')):
            q, k, v = (inputs[key]['shape'] for key in ('query', 'key', 'value'))
            dims = {'query_heads': q[1], 'kv_heads': k[1], 'query_length': q[2],
                    'key_length': k[2], 'head_dim': q[3], 'value_dim': v[3]}
        context = InvocationContext(self.name, request_count=request_count,
            token_count=actual_call.get('token_count'), seq_lens=actual_call.get('seq_lens'),
            matrix_dims=dims,
            dtype=inputs['query'].get('dtype'), layout=actual_call.get('layout'), attributes=attrs)
        reason = self.unsupported_reason(context, controlled=False)
        if reason and self.on_unsupported == 'reject':
            raise ValueError(reason)
        return context

    def unsupported_reason(self, context: InvocationContext, *, controlled: bool) -> str | None:
        if context.operator != self.name or context.layout != 'BNSD':
            return 'unsupported_operator_or_layout: TND requires its own sequence-metadata adapter'
        b, attrs = context.request_count, context.attributes
        if type(b) is not int or b <= 0:
            return 'request_count_unknown'
        expected_shapes = {'query': [b, 16, 2048, 128], 'key': [b, 8, 2048, 128],
                           'value': [b, 8, 2048, 128], 'atten_mask': [2048, 2048]}
        if attrs.get('tensor_shapes') != expected_shapes:
            return 'unsupported_shape'
        if attrs.get('tensor_dtypes') != {'query': 'float16', 'key': 'float16', 'value': 'float16', 'atten_mask': 'bool'}:
            return 'unsupported_dtype'
        if context.token_count != b * 2048 or context.seq_lens != (2048,) * b:
            return 'inconsistent_explicit_sequence_metadata'
        expected = {'scale': EFFECTIVE_SCALE, 'keep_prob': 1, 'pre_tokens': 2147483647,
                    'next_tokens': 0, 'heads': 16, 'inner_precise': 0, 'sparse_mode': 2,
                    'optional_inputs_absent': True, 'device': self.certificate['runtime']['device']}
        if any(type(attrs.get(key)) is bool and type(value) is not bool or attrs.get(key) != value
               for key, value in expected.items()):
            return 'unsupported_native_attributes_or_device'
        if controlled and b not in self._contexts:
            return 'context_not_controlled_validated: discovery does not promote support'
        return None

    def observe_current(self, context: InvocationContext) -> StrategyObservation:
        key = canonical_sha256(context.as_dict())
        require(key in self._native, 'No same-context native observation; execute an original native scope first')
        observation = self._native[key]
        require(observation.fingerprints == self.fingerprints, 'Stock/other-build discovery cannot substitute for this bridge native observation')
        return observation

    def legal_alternatives(self, context: InvocationContext, current: NativeStrategy):
        if self.unsupported_reason(context, controlled=True):
            if self.on_unsupported == 'reject':
                raise ValueError(self.unsupported_reason(context, controlled=True))
            return ()
        result = []
        before = selector_projection(current)
        for item in self._contexts[context.request_count]['strategies']:
            candidate = NativeStrategy(item['strategy_id'], item['parameters'])
            after = selector_projection(candidate)
            m, r = before['query_tile'] != after['query_tile'], before['key_group_ratio'] != after['key_group_ratio']
            if ((self.selection_target == 'query_tile' and m and not r)
                    or (self.selection_target == 'key_group' and r and not m)
                    or (self.selection_target == 'joint' and m and r)) and self.validate_candidate(context, candidate):
                result.append(candidate)
        return tuple(result)

    def validate_candidate(self, context: InvocationContext, candidate: NativeStrategy) -> bool:
        if self.unsupported_reason(context, controlled=True):
            return False
        row = self._contexts[context.request_count]
        return any(candidate.as_dict() == item for item in row['strategies'])

    @contextmanager
    def native_scope(self, decision: PolicyDecision | None, invocation_id: str, evidence_dir: Path,
                     *, context: InvocationContext, current: NativeStrategy | None = None):
        """One genuine call is owned by the caller, never executed or replaced here."""
        require(type(invocation_id) is str and re.fullmatch(r'[A-Za-z0-9_.-]{1,128}', invocation_id), 'Invalid native invocation token')
        evidence_dir = Path(evidence_dir).resolve()
        evidence_dir.mkdir(parents=True, exist_ok=True)
        self._observation_dirs.add(evidence_dir)
        requested = decision is not None and decision.requires_execution_change
        reason = self.unsupported_reason(context, controlled=requested)
        if reason:
            if self.on_unsupported == 'reject':
                raise ValueError(reason)
            require(_ENV_LEASE.acquire(blocking=False), 'Concurrent FA environment scopes require separate processes')
            try:
                require(all(os.environ.get(key) is None for key in ENV_KEYS), 'Cannot pass through inside an inherited FA scope')
                yield {'mode': 'uninstrumented_passthrough', 'reason': reason, 'strategy_switch_verified': None}
            finally:
                _ENV_LEASE.release()
            return
        selected = None
        if requested:
            require(current is not None and decision.current_semantic_sha256 == current.semantic_sha256, 'Decision does not bind its real native observation')
            require(self.validate_candidate(context, decision.requested_strategy), 'Uncertified FA strategy')
            selected = {'m': decision.requested_strategy.parameters['query_tile'],
                        'n_ratio': decision.requested_strategy.parameters['key_group_ratio']}
        attrs = context.attributes
        native_pair = None if not requested else {'m': current.parameters['query_tile'], 'n_ratio': current.parameters['key_group_ratio']}
        evidence = self.certificate['validation_evidence']
        require(evidence, 'Certificate must bind validation or explicitly experimental build evidence')
        plan = {'schema': 'nes.fa.typed_plan.v1', 'operator': self.name,
                'mode': 'controlled' if requested else 'native',
                'scope': {'pid': os.getpid(), 'native_tid': threading.get_native_id(), 'token': invocation_id,
                          'phase': 'operator_execution', 'request_count': context.request_count},
                'context': {**{name + '_shape': attrs['tensor_shapes'][name] for name in ('query', 'key', 'value')},
                            'mask_shape': attrs['tensor_shapes']['atten_mask'], 'dtype': 'float16', 'mask_dtype': 'bool',
                            'layout': context.layout, **{key: attrs[key] for key in ('scale', 'keep_prob', 'pre_tokens', 'next_tokens', 'heads', 'inner_precise', 'sparse_mode', 'optional_inputs_absent')}},
                'version': {key: self.certificate['runtime'][key] for key in ('source_commit', 'cann_build', 'bridge_abi')},
                'requested': {'native': native_pair, 'selected': selected},
                'certificate': {'id': self.certificate['id'], 'source_manifest_sha256': self.fingerprints.source_sha256,
                                'build_manifest_sha256': self.fingerprints.build_sha256,
                                'validation_sha256': evidence[0]['sha256'], 'validation_state': self.certificate['validation_state']}}
        text = json.dumps(plan, separators=(',', ':'), allow_nan=False)
        require(len(text.encode()) <= 16384, 'Typed FA plan exceeds native parser limit')
        with _environment(text, str(evidence_dir / 'host_events.jsonl')):
            yield plan

    def parse_native_evidence(self, context: InvocationContext, record: dict) -> StrategyObservation:
        """Consume separately parsed actual launch bytes, not selected/desired labels."""
        require(not self.unsupported_reason(context, controlled=False), 'Cannot interpret an unsupported FA route')
        native = record['native_launch']
        raw = bytes.fromhex(native['raw_hex'])
        pod = decode(raw)
        require(record.get('pod') == pod, 'Observer public POD differs from actual launch bytes')
        require(str(native.get('key', native.get('tiling_key'))) == TILING_KEY, 'Unsupported actual FA kernel route/key')
        require(type(native.get('block_dim')) is int and native['block_dim'] > 0, 'Missing actual launch geometry')
        expected = {'inputParams.bSize': context.request_count, 'inputParams.n2Size': 8, 'inputParams.gSize': 2,
                    'inputParams.s1Size': 2048, 'inputParams.s2Size': 2048, 'inputParams.dSize': 128,
                    'inputParams.scaleValue': EFFECTIVE_SCALE, 'inputParams.keepProb': 1,
                    'inputParams.preTokens': 2048, 'inputParams.nextTokens': 0,
                    'inputParams.layoutType': 3, 'inputParams.implMode': 0, 'inputParams.sparseType': 3,
                    'inputParams.needDropMaskOp': 0}
        require(all(pod.get(key) == value for key, value in expected.items()), 'Actual native workload/precision/mask semantics differ')
        host = record.get('host_event')
        fingerprints = self.fingerprints
        if host is not None:
            require(host['schema'] == 'nes.fa.host_observation.v1' and host['callback_status'] == 0
                    and host['eligible'] is True and host['raw_hex'] == native['raw_hex'], 'Host/native raw binding or guards failed')
            require(host['plan']['certificate']['source_manifest_sha256'] == fingerprints.source_sha256
                    and host['plan']['certificate']['build_manifest_sha256'] == fingerprints.build_sha256,
                    'Native host source/build differs from certified deployment')
            require(host['plan']['scope']['request_count'] == context.request_count, 'Host request interpretation differs')
            require(host['pid'] == host['plan']['scope']['pid'] and host['tid'] == host['plan']['scope']['native_tid']
                    and host['token'] == host['plan']['scope']['token'], 'Host scope identity differs')
            require(host['plan']['certificate']['id'] == self.certificate['id']
                    and host['plan']['certificate']['validation_state'] == self.certificate['validation_state']
                    and host['plan']['certificate']['validation_sha256'] == self.certificate['validation_evidence'][0]['sha256'], 'Host certificate binding differs')
            actual = strategy_from_pod(pod, raw=raw)
            require(host['selected_factors'] == {'m': actual.parameters['query_tile'], 'n_ratio': actual.parameters['key_group_ratio']},
                    'Actual selector result differs from final native launch')
            require(host['mode'] == host['plan']['mode'], 'Host mode differs from scoped plan')
            if host['mode'] == 'controlled':
                require(host['selection_applied'] is True
                        and host['plan']['requested']['native'] == host['native_factors']
                        and host['plan']['requested']['selected'] == host['selected_factors'],
                        'Requested controlled strategy was not actually applied')
            else:
                require(host['mode'] == 'native' and host['selection_applied'] is False
                        and host['native_factors'] == host['selected_factors']
                        and host['plan']['requested']['selected'] is None, 'Native observation changed the strategy')
        else:
            require(record.get('origin') == 'stock', 'Private bridge observation cannot omit host callback evidence')
            declared = record.get('fingerprints')
            require(type(declared) is dict, 'Original stock discovery must bind its actual source/build independently')
            fingerprints = BuildFingerprints(**declared)
        evidence = tuple(EvidenceRef(**row) for row in record['evidence'])
        require(bool(evidence), 'Native launch observation needs original evidence')
        for ref in evidence:
            path = Path(ref.path).resolve(strict=True)
            require(path.is_file() and any(path.is_relative_to(directory) for directory in self._observation_dirs), 'Observation evidence outside this invocation directories')
            require(_sha(path) == ref.sha256, 'Original native observation file changed')
        actual = strategy_from_pod(pod, raw=raw)
        contract = self._contexts.get(context.request_count, {}).get('input_contract')
        if contract is not None:
            require(input_contract(pod) == contract, 'Native input contract differs from certified context')
        if self.certificate['validation_state'] == 'validated' and context.request_count in self._contexts:
            require(self.validate_candidate(context, actual), 'Actual execution plan is outside the certified context recipes')
        usage = ResourceUsage(**record.get('resources', {}))
        observation = StrategyObservation(context, actual, fingerprints, evidence, hashlib.sha256(raw).hexdigest(), usage)
        # A controlled result is never mistaken for the next call's normal strategy.
        if host is not None and host['mode'] == 'native':
            self._native[canonical_sha256(context.as_dict())] = observation
        return observation
