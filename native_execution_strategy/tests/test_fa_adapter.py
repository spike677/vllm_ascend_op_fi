"""CPU contract fixtures. No new native execution or new-bridge validation."""
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
import unittest

from native_execution_strategy.adapters.fa_schema import FIELDS, PADDING_FIELDS, OPAQUE_SECTIONS, decode, execution_plan, input_contract, validate_execution_plan
from native_execution_strategy.adapters.flash_attention import FlashAttentionAdapter
from native_execution_strategy.config import PolicyConfig
from native_execution_strategy.policies import Policy

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = json.loads((ROOT / 'tests/fixtures/legacy_raw_pod.json').read_bytes())['records']


def actual(b=1):
    return {'operator': 'FlashAttentionScore', 'request_count': b, 'token_count': b * 2048,
            'seq_lens': [2048] * b, 'layout': 'BNSD', 'device': 'Ascend910B4',
            'inputs': {'query': {'shape': [b, 16, 2048, 128], 'dtype': 'float16'},
                       'key': {'shape': [b, 8, 2048, 128], 'dtype': 'float16'},
                       'value': {'shape': [b, 8, 2048, 128], 'dtype': 'float16'},
                       'atten_mask': {'shape': [2048, 2048], 'dtype': 'bool'}},
            'attributes': {'scale': 1 / math.sqrt(128), 'keep_prob': 1, 'pre_tokens': 2147483647,
                           'next_tokens': 0, 'heads': 16, 'inner_precise': 0, 'sparse_mode': 2,
                           'optional_inputs_absent': True}}


class AdapterTests(unittest.TestCase):
    def setUp(self):
        self.temp_root = Path.cwd() / '.codex-tmp'
        self.temp_root.mkdir(exist_ok=True)
        self.tmp = tempfile.TemporaryDirectory(dir=self.temp_root)
        self.directory = Path(self.tmp.name)
        self.adapter = FlashAttentionAdapter(ROOT / 'data/fa_s1s2_experimental.json',
            evidence_root=ROOT, allow_experimental=True)

    def tearDown(self):
        self.assertNotIn('NES_FA_PLAN_JSON', os.environ)
        self.assertNotIn('NES_FA_LOG_PATH', os.environ)
        self.assertTrue(self.directory.resolve().is_relative_to(self.temp_root.resolve()))
        self.tmp.cleanup()

    def record(self, context, decision=None, profile='base', token='fixture_native'):
        folder = self.directory / token
        current = None if decision is None else self.adapter.observe_current(context).strategy
        with self.adapter.native_scope(decision, token, folder, context=context, current=current) as plan:
            captured = copy.deepcopy(plan)
            self.assertIn('NES_FA_PLAN_JSON', os.environ)
        raw = FIXTURE[f'{profile}_b{context.request_count}']['raw_hex']
        pod = decode(bytes.fromhex(raw))
        factors = {'m': pod['coreParams.s1BaseSize'], 'n_ratio': pod['coreParams.nRatio']}
        native = {'m': 128, 'n_ratio': 8}
        evidence = folder / 'cpu_fixture.txt'
        evidence.write_text('Synthetic CPU host envelope around legacy public POD; not a new NPU run.', encoding='utf-8')
        return {'host_event': {'schema': 'nes.fa.host_observation.v1', 'callback_status': 0,
                    'eligible': True, 'raw_hex': raw, 'plan': captured, 'mode': captured['mode'],
                    'pid': captured['scope']['pid'], 'tid': captured['scope']['native_tid'],
                    'token': captured['scope']['token'], 'selected_factors': factors, 'native_factors': native,
                    'selection_applied': captured['mode'] == 'controlled'},
                'native_launch': {'raw_hex': raw, 'tiling_key': '22619644976', 'block_dim': 20},
                'pod': pod, 'evidence': [{'path': str(evidence), 'sha256': hashlib.sha256(evidence.read_bytes()).hexdigest()}],
                'resources': {}}

    def test_target_selection_is_not_B8_condition(self):
        for b in (1, 8):
            for target, expected in [('query_tile', (64, 8)), ('key_group', (128, 4)), ('joint', (64, 4))]:
                self.adapter.selection_target = target
                context = self.adapter.inspect(actual(b))
                baseline = self.adapter.parse_native_evidence(context, self.record(context, token=f'b{b}_{target}'))
                policy = Policy(PolicyConfig(target=target))
                for _ in range(2):
                    decision = policy.evaluate(context, baseline.strategy, self.adapter.legal_alternatives(context, baseline.strategy), self.adapter.validate_candidate)
                    self.assertTrue(decision.requires_execution_change)
                    p = decision.requested_strategy.parameters
                    self.assertEqual((p['query_tile'], p['key_group_ratio']), expected)

    def test_controlled_observation_does_not_toggle_native_baseline(self):
        context = self.adapter.inspect(actual())
        baseline = self.adapter.parse_native_evidence(context, self.record(context))
        decision = Policy(PolicyConfig()).evaluate(context, baseline.strategy,
            self.adapter.legal_alternatives(context, baseline.strategy), self.adapter.validate_candidate)
        observed = self.adapter.parse_native_evidence(context, self.record(context, decision, 'm_only', 'fixture_control'))
        self.assertNotEqual(observed.strategy.semantic_sha256, baseline.strategy.semantic_sha256)
        self.assertEqual(self.adapter.observe_current(context), baseline)

    def test_new_B2_B4_discovery_does_not_authorize_control(self):
        for b in (2, 4):
            context = self.adapter.inspect(actual(b))
            self.assertIsNone(self.adapter.unsupported_reason(context, controlled=False))
            with self.assertRaises(ValueError):
                self.adapter.legal_alternatives(context, None)

    def test_unknown_TND_is_not_shape0_request_inference(self):
        call = actual()
        call.update(layout='TND', request_count=None, token_count=None, seq_lens=None)
        with self.assertRaises(ValueError):
            self.adapter.inspect(call)
        self.adapter.on_unsupported = 'passthrough'
        context = self.adapter.inspect(call)
        self.assertIsNone(context.request_count)
        self.assertEqual(context.matrix_dims, {})
        with self.adapter.native_scope(None, 'unsupported', self.directory / 'unsupported', context=context) as plan:
            self.assertEqual(plan['mode'], 'uninstrumented_passthrough')
            self.assertNotIn('NES_FA_PLAN_JSON', os.environ)

    def test_nesting_rejected_without_destroying_owner(self):
        context = self.adapter.inspect(actual())
        with self.adapter.native_scope(None, 'outer', self.directory / 'outer', context=context):
            original = os.environ['NES_FA_PLAN_JSON']
            with self.assertRaises(ValueError):
                with self.adapter.native_scope(None, 'inner', self.directory / 'inner', context=context):
                    pass
            self.assertEqual(os.environ['NES_FA_PLAN_JSON'], original)

    def test_host_intent_cannot_replace_actual_launch(self):
        context = self.adapter.inspect(actual())
        record = self.record(context)
        record['native_launch']['raw_hex'] = FIXTURE['m_only_b1']['raw_hex']
        with self.assertRaises(ValueError):
            self.adapter.parse_native_evidence(context, record)

    def test_private_host_callback_cannot_be_omitted(self):
        context = self.adapter.inspect(actual())
        record = self.record(context)
        record['host_event'] = None
        with self.assertRaises(ValueError):
            self.adapter.parse_native_evidence(context, record)

    def test_changed_evidence_bytes_rejected(self):
        context = self.adapter.inspect(actual())
        record = self.record(context)
        Path(record['evidence'][0]['path']).write_text('changed', encoding='utf-8')
        with self.assertRaises(ValueError):
            self.adapter.parse_native_evidence(context, record)

    def test_source_audited_padding_is_retained_raw_but_not_a_strategy_change(self):
        raw = bytes.fromhex(FIXTURE['base_b1']['raw_hex'])
        changed = bytearray(raw)
        for name, (offset, count) in PADDING_FIELDS.items():
            for index in range(offset, offset + count):
                changed[index] ^= 0xA5
        first, second = decode(raw), decode(bytes(changed))
        self.assertNotEqual(raw, bytes(changed))
        self.assertEqual(len(first), 205)
        self.assertTrue(all(first[name] != second[name] for name in PADDING_FIELDS))
        self.assertEqual(input_contract(first), input_contract(second))
        self.assertEqual(execution_plan(first, raw=raw), execution_plan(second, raw=bytes(changed)))
        self.assertIn('inputParams.remain', input_contract(first))
        self.assertIn('coreParams.s1SparseValidSize', execution_plan(first, raw=raw))

    def test_all_opaque_section_bytes_participate_in_strategy_identity(self):
        raw = bytes.fromhex(FIXTURE['base_b1']['raw_hex'])
        original = execution_plan(decode(raw), raw=raw)
        self.assertEqual(len(original), 169)
        for name, (offset, size) in OPAQUE_SECTIONS.items():
            for index in range(offset, offset + size):
                changed = bytearray(raw)
                changed[index] ^= 1
                self.assertEqual(decode(raw), decode(bytes(changed)))
                observed = execution_plan(decode(bytes(changed)), raw=bytes(changed))
                self.assertNotEqual(observed[name], original[name])
                validate_execution_plan(observed)

    def test_execution_plan_rejects_missing_or_malformed_opaque_sections(self):
        raw = bytes.fromhex(FIXTURE['base_b1']['raw_hex'])
        for name in OPAQUE_SECTIONS:
            plan = execution_plan(decode(raw), raw=raw)
            del plan[name]
            with self.assertRaises(ValueError):
                validate_execution_plan(plan)
            plan[name] = 'g' * 128
            with self.assertRaises(ValueError):
                validate_execution_plan(plan)


if __name__ == '__main__':
    unittest.main()
