"""CPU input/ownership refusals for the preparation-only natural FIA boundary."""
import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace as NS
import unittest

import torch

from demoncat.runtime.vllm_ascend.worker_extension import attention28_prefill as boundary


def fixture():
    mask = torch.empty(2048, 2048, dtype=torch.bool)
    actual = dict(query=torch.empty(1, 16, 2048, 128, dtype=torch.float16),
        key=torch.empty(1, 8, 2048, 128, dtype=torch.float16),
        value=torch.empty(1, 8, 2048, 128, dtype=torch.float16), atten_mask=mask,
        input_layout='BNSD', actual_seq_lengths=[2048], actual_seq_lengths_kv=[2048],
        num_heads=16, num_key_value_heads=8, scale=128**-.5, pre_tokens=2147483647,
        next_tokens=0, sparse_mode=2, inner_precise=0)
    metadata = NS(attn_mask=mask, num_prefills=1, num_decodes=0, num_actual_tokens=2048,
                  causal=True, seq_lens_list=[2048], actual_seq_lengths_q=[2048])
    scheduler = dict(status='READ_ONLY_SCHEDULER_SNAPSHOT_CORROBORATED_NOT_NPU_ACCEPTANCE',
        num_requests=1, num_computed_tokens=0, num_prompt_tokens=2048, num_scheduled_tokens=2048)
    return actual, metadata, scheduler


class PrefillBoundaryTests(unittest.TestCase):
    def inspect(self, actual, metadata, scheduler, **kwargs):
        return boundary.inspect_prefill_input('npu_fused_infer_attention_score', actual,
            metadata=metadata, scheduler=scheduler, allow_cpu_test=True, **kwargs)

    def test_candidate_keeps_exact_objects_and_cannot_claim_mask_or_native_proof(self):
        actual, metadata, scheduler = fixture()
        identities = {name: id(actual[name]) for name in ('query', 'key', 'value', 'atten_mask')}
        row = self.inspect(actual, metadata, scheduler)
        self.assertTrue(row['input_contract_matches'])
        self.assertEqual(row['qualification_scope'], 'CPU_CONTRACT_ONLY')
        for field in ('api_arguments_changed', 'tensor_values_read', 'mask_values_verified',
                      'native_route_verified', 'public_worker_qualified', 'native_launch_verified'):
            self.assertFalse(row[field])
        self.assertEqual(identities, {name: id(actual[name]) for name in identities})
        with self.assertRaisesRegex(RuntimeError, 'NPU_REQUIRED'):
            boundary.inspect_prefill_input('npu_fused_infer_attention_score', actual,
                metadata=metadata, scheduler=scheduler)

    def test_tnd_page16_no_longer_falls_through_to_bnsd(self):
        actual, metadata, scheduler = fixture()
        actual.update(input_layout='TND', block_size=16, block_table=torch.empty(1, 128, dtype=torch.int32))
        with self.assertRaisesRegex(RuntimeError, 'TND_PAGE16_IS_SEPARATE'):self.inspect(actual, metadata, scheduler)

    def test_prefill_and_no_page_requirements_cannot_be_claimed_by_shape(self):
        actual, metadata, scheduler = fixture()
        for field, value, reason in [('num_computed_tokens', 7, 'FULL_PREFILL'),
                                     ('num_requests', 8, 'FULL_PREFILL'),
                                     ('num_scheduled_tokens', 1024, 'FULL_PREFILL')]:
            changed = dict(scheduler);changed[field] = value
            with self.assertRaisesRegex(RuntimeError, reason):self.inspect(actual, metadata, changed)
        for changed, reason in [(dict(actual, block_table=torch.empty(1, 128)), 'OPTIONAL_FEATURE'),
                                (dict(actual, query_rope=torch.empty(1)), 'OPTIONAL_FEATURE'),
                                (dict(actual, unseen_mode=1), 'UNKNOWN_OPTIONAL'),
                                (dict(actual, actual_seq_lengths=[1024]), 'NONACCUMULATED_LENGTHS')]:
            with self.assertRaisesRegex(RuntimeError, reason):self.inspect(changed, metadata, scheduler)

    def test_same_shape_new_mask_and_noncontiguous_q_are_refused(self):
        actual, metadata, scheduler = fixture()
        other = dict(actual, atten_mask=torch.empty_like(actual['atten_mask']))
        with self.assertRaisesRegex(RuntimeError, 'MASK_NOT_ACTUAL_METADATA'):self.inspect(other, metadata, scheduler)
        other = dict(actual, query=actual['query'].transpose(-1, -2).contiguous().transpose(-1, -2))
        with self.assertRaisesRegex(RuntimeError, 'QKV_GEOMETRY'):self.inspect(other, metadata, scheduler)

    def test_bind_preserves_original_objects_and_refuses_duplicate_or_extra_positionals(self):
        q, k, v = object(), object(), object();kwargs = dict(input_layout='TND')
        row = boundary.bind_actual_api((q, k, v), kwargs)
        self.assertIs(row['query'], q);self.assertIs(row['key'], k);self.assertIs(row['value'], v)
        self.assertEqual(kwargs, dict(input_layout='TND'))
        with self.assertRaisesRegex(RuntimeError, 'DUPLICATE_ACTUAL'):boundary.bind_actual_api((q,), dict(query=q))
        with self.assertRaisesRegex(RuntimeError, 'POSITIONAL_OPTIONALS'):boundary.bind_actual_api((q, k, v, 16), {})

    def test_two_mechanisms_have_distinct_requests_and_no_implicit_execution(self):
        legal_b = boundary.strategy_request(boundary.MECHANISMS[0], 'baseline')
        legal_f = boundary.strategy_request(boundary.MECHANISMS[0], 'fault')
        replay_b = boundary.strategy_request(boundary.MECHANISMS[1], 'baseline')
        replay_f = boundary.strategy_request(boundary.MECHANISMS[1], 'fault')
        self.assertEqual(legal_b['requested_m'], 512)
        self.assertEqual(legal_f['requested_key'], replay_b['requested_key'])
        self.assertNotEqual(replay_b['requested_key'], replay_f['requested_key'])
        for row in (legal_b, legal_f, replay_b, replay_f):
            self.assertFalse(row['native_scope_entered']);self.assertFalse(row['worker_qualified'])
        with self.assertRaises(RuntimeError):boundary.strategy_request('33.paged_decode', 'fault')

    def test_actual_sdk_files_required_and_tnd33_completion_refused(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder);sdk = root / 'sdk';sdk.mkdir()
            header = sdk / 'actual.h';header.write_bytes(b'exact 9.1 SDK interface')
            digest = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
            contract = root / 'contract.json'
            pinned = [dict(kind='sdk', path='actual.h', bytes=header.stat().st_size, sha256=digest(header))]
            contract.write_text(json.dumps(dict(files=pinned)))
            data = dict(schema='demoncat.fia.fresh_build.v1', status='CPU_BUILD_ONLY_REQUIRES_NATIVE_QUALIFICATION',
                variant='legal_tiling', source_commit='c77c0829f1f0f0b8f882db1d9b051a758b605472',
                loading_allowed=False, npu_executed=False, previous_experiment_objects_used=False,
                contract_sha256=digest(contract), sdk_interface_files=pinned, sdk_interface_manifest_sha256=None)
            completion = root / 'completion.json';completion.write_text(json.dumps(data))
            report = boundary.verify_sdk_binding(completion, digest(completion), contract, sdk)
            self.assertFalse(report['loading_allowed']);self.assertFalse(report['library_coexistence_verified'])
            header.write_bytes(b'actual 8.5.1 SDK interface')
            with self.assertRaisesRegex(RuntimeError, 'RUNTIME_SDK_ABI_DIFFERS'):
                boundary.verify_sdk_binding(completion, digest(completion), contract, sdk)
            del data['sdk_interface_files'];completion.write_text(json.dumps(data))
            with self.assertRaisesRegex(RuntimeError, 'EXACT_ORIGINAL_CONTRACT'):
                boundary.verify_sdk_binding(completion, digest(completion), contract, sdk)
            original = Path(boundary.__file__).resolve().parents[5] / 'fault_injection/native_adapters/attention28/fia/build_contract.json'
            data['contract_sha256'] = digest(original);completion.write_text(json.dumps(data))
            # The known original receipt still requires actual SDK bytes.
            with self.assertRaisesRegex(RuntimeError, 'RUNTIME_SDK_ABI_DIFFERS'):
                boundary.verify_sdk_binding(completion, digest(completion), original, sdk)
            data['variant'] = 'tnd_page16_fai_s2_256';completion.write_text(json.dumps(data))
            with self.assertRaisesRegex(RuntimeError, 'INDEPENDENT_SOURCE_BUILD_REQUIRED'):
                boundary.verify_sdk_binding(completion, digest(completion), contract, sdk)

    def audit_fixture(self):
        # No vLLM source qualification is mocked here: only wrapper ownership
        # and pass-through behavior are tested, without entering a request.
        import threading
        sentinel, calls = object(), []
        def original(*args, **kwargs):calls.append((args, kwargs));return sentinel
        module = NS(npu_fused_infer_attention_score=original)
        audit = boundary.FiaPrefillAudit.__new__(boundary.FiaPrefillAudit)
        audit.model = torch.nn.Identity();audit.api_module = module
        audit.candidates = [];audit.owner = threading.get_ident()
        audit.handles = [];audit.patches = [];audit.stack = [];audit.rows = [];audit.errors = []
        audit.expected = audit.current = None;audit.installed = False
        audit.max_calls = 1;audit.overflow_calls = 0
        audit.install()
        return audit, module, original, sentinel, calls

    def test_pass_through_calls_original_once_and_returns_exact_object(self):
        audit, api, original, sentinel, calls = self.audit_fixture()
        q, k, v, mask = object(), object(), object(), object()
        try:
            result = api.npu_fused_infer_attention_score(q, k, v, atten_mask=mask, input_layout='TND')
            self.assertIs(result, sentinel);self.assertEqual(len(calls), 1)
            self.assertEqual(calls[0][0], (q, k, v));self.assertIs(calls[0][1]['atten_mask'], mask)
        finally:audit.close()
        self.assertIs(api.npu_fused_infer_attention_score, original)

    def test_unproven_boundary_and_overflow_preserve_each_real_call(self):
        audit, api, original, sentinel, calls = self.audit_fixture()
        audit.current = dict(scheduler=None)
        try:
            self.assertIs(api.npu_fused_infer_attention_score(input_layout='TND'), sentinel)
            self.assertIs(api.npu_fused_infer_attention_score(input_layout='TND'), sentinel)
            self.assertEqual(len(calls), 2)
            self.assertEqual(len(audit.rows), 1);self.assertEqual(audit.overflow_calls, 1)
            self.assertIn('SCHEDULER_OR_MODULE_UNPROVEN', audit.rows[0]['refusal'])
            self.assertTrue(audit.rows[0]['consumer_returned'])
            self.assertFalse(audit.rows[0]['input_contract_matches'])
        finally:
            audit.current = None;audit.close()

    def test_cleanup_never_overwrites_a_later_api_owner(self):
        audit, api, original, sentinel, calls = self.audit_fixture()
        installed = api.npu_fused_infer_attention_score
        foreign = lambda *args, **kwargs: None
        api.npu_fused_infer_attention_score = foreign
        with self.assertRaisesRegex(RuntimeError, 'API_WRAPPER_OWNERSHIP_CHANGED'):audit.close()
        self.assertIs(api.npu_fused_infer_attention_score, foreign)
        api.npu_fused_infer_attention_score = installed;audit.close()
        self.assertIs(api.npu_fused_infer_attention_score, original)


if __name__ == '__main__':unittest.main()
