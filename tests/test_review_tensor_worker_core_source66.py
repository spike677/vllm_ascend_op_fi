"""Independent reader negatives from actual public CPU Host/Runtime requests."""
import copy
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import test_core_tensor_serving as core
import test_tensor_worker_core_probe as driver_fixture
from test_rmsnorm_natural import FusedNorm, CustomFusedNorm
from experiments.c2c3_closure_20261117 import natural_tensor_worker_core_probe as probe


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / 'experiments/c2c3_closure_20261117/review_tensor_worker_core_source66.py'
spec = importlib.util.spec_from_file_location('tensor_original_reader', SCRIPT)
review = importlib.util.module_from_spec(spec); spec.loader.exec_module(review)
AUDITOR = review.load_module(ROOT/'fault_injection/coverage_audit.py', 'independent_coverage_reader', review.AUDIT_SHA)


class NativeCPUModel(core.RMSModel):
    """Real CPU dispatcher calls for both observed native RMS schema families."""
    def __init__(self):
        super().__init__(); self.first = FusedNorm(); self.last = CustomFusedNorm()
    def forward(self, input_ids, positions):
        self.forward_calls += 1
        hidden = self.embedding[input_ids]
        value, residual = self.first(hidden, core.torch.ones_like(hidden))
        value, residual = self.last(value + 1, residual)
        return value + 0


def make_fixture(kind, *, fail=False, native=False):
    case = (core.CoreMoEServingTests if kind == '8.restore_offset' else core.CoreTensorServingTests)()
    if native: case.model_type = NativeCPUModel
    case.setUp()
    try:
        if kind == '8.restore_offset':
            # Actual CPU calls use the real reviewed six-way routing shape.
            case.model.first.top_k = case.model.last.top_k = 6
        driver = driver_fixture.Driver(case)
        if fail: driver.fail_index = 4
        healthy = case.actual('unmanaged-warmup')
        result = dict(status='STARTED', mechanism=kind, standard_vllm_worker=False,
            public_qualification_granted=False, historical_scope_inherited=False, healthy_before=[int(healthy.argmax())])
        with tempfile.TemporaryDirectory() as folder:
            probe.drive(driver, case.client, prompt={'prompt_token_ids':[1, 2, 3]}, params=object(), out=Path(folder), report=result)
            host = case.client.snapshot()
            result['binding_removal'] = core.binding.remove_tensor_serving_binding(case.worker, kind)
            # Keep detached JSON; no live provider object is accepted by reader.
            return json.loads(json.dumps(result)), json.loads(json.dumps(host))
    finally:
        case.tearDown()


class PublicTensorOriginalReviewTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rms, cls.rms_host = make_fixture('18.api_half')
        cls.moe, cls.moe_host = make_fixture('8.restore_offset')
        cls.native, cls.native_host = make_fixture('18.api_half', native=True)
        cls.failed, cls.failed_host = make_fixture('18.api_half', fail=True)

    def audit(self, d=None, h=None):
        return review.review(d or self.rms, h or self.rms_host, AUDITOR, natural=False)

    def test_both_actual_ten_request_protocols_and_live_proof_reconcile(self):
        for d, h in ((self.rms, self.rms_host), (self.moe, self.moe_host), (self.native, self.native_host)):
            with self.subTest(mechanism=d['mechanism']):
                result = self.audit(d, h)
                self.assertTrue(result['consistent'], result['issues'])
                self.assertEqual(result['groups']['c2']['recomputed_hits'], 2)
                self.assertEqual(result['groups']['c3']['recomputed_hits'], 4)
                self.assertEqual(result['actual_requests'], 10)

    def test_probe_pass_neither_needed_nor_sufficient(self):
        d = copy.deepcopy(self.rms); d['status'] = 'UNTRUSTED_LABEL'
        self.assertTrue(self.audit(d)['consistent'])
        d['c3']['calls'].pop(); d['status'] = probe.SUCCESS
        self.assertFalse(self.audit(d)['consistent'])

    def test_c3_missing_second_target_occurrence_is_not_hidden_by_total(self):
        d = copy.deepcopy(self.rms); d['c3']['calls'][2] = copy.deepcopy(d['c3']['calls'][0])
        self.assertFalse(self.audit(d)['consistent'])

    def test_c2_original_proof_cannot_be_reconstructed_or_used_elsewhere(self):
        for change in ('object', 'scope', 'owner', 'generation', 'report', 'issued'):
            d = copy.deepcopy(self.rms); proof = d['c3']['live_c2_proof']['consumed']
            if change == 'object': proof['object_binding']['python_object_id'] += 1
            elif change == 'scope': proof['consumer']['scope']['workload_sha256'] = '0'*64
            elif change == 'owner': proof['consumer']['owner'] = 'foreign'
            elif change == 'generation': proof['consumer']['generation'] -= 1
            elif change == 'report': proof['proof']['report_sha256'] = '0'*64
            else: d['c2']['live_c2_proof']['issued'].clear()
            with self.subTest(change=change): self.assertFalse(self.audit(d)['consistent'])

    def test_wrong_rpc_ticket_uncertain_reply_or_query_advancing_refuse(self):
        for change in ('ticket', 'uncertain', 'query', 'reserved', 'owner'):
            h = copy.deepcopy(self.rms_host); reply = h['history'][18]['response']
            if change == 'ticket': reply['ticket']['generation'] += 1
            elif change == 'uncertain': reply['status'] = 'RPC_UNCERTAIN'
            elif change == 'query': reply['provider_state']['serving']['next_phase'] = 'c2_off'
            elif change == 'reserved': reply['provider_state']['binding_reserved'] = False
            else: reply['provider_state']['owner'] = 'foreign'
            with self.subTest(change=change): self.assertFalse(self.audit(h=h)['consistent'])

    def test_rms_consumer_readback_and_half_semantics_required(self):
        for change in ('consumer', 'readback', 'gain', 'auxiliary', 'call_input'):
            d = copy.deepcopy(self.rms); row = d['requests'][4]['worker']; event = row['mechanism']['events'][0]
            if change == 'consumer': event['consumers'][0]['actual_return_lineage'] = False
            elif change == 'readback': event['return_readback_exact'] = False
            elif change == 'gain': event['gain'] = 0.25
            elif change == 'auxiliary': event['auxiliary_return_storage_unchanged'] = False
            else: row['calls'][0]['metadata'][0]['object_id'] += 1
            with self.subTest(change=change): self.assertFalse(self.audit(d)['consistent'])

    def test_moe_native_producer_consumer_and_complete_oracle_required(self):
        for change in ('producer', 'double_call', 'oracle', 'groups', 'hidden'):
            d = copy.deepcopy(self.moe); row = d['requests'][4]['worker']; trace = row['independent']['moe_trace']
            native = next(e for e in trace['native_operators'] if e['operator'] == 'npu.npu_moe_token_unpermute.default')
            if change == 'producer': native['arguments']['probs']['object_id'] += 1
            elif change == 'double_call': native['original_calls'] = 2
            elif change == 'oracle': row['mechanism']['events'][0]['complete_native_output_oracle_exact'] = False
            elif change == 'groups': row['mechanism']['events'][0]['source_groups'][0] = 0
            else: native['consumed_arguments']['permuted_tokens']['object_id'] += 1
            with self.subTest(change=change): self.assertFalse(self.audit(d, self.moe_host)['consistent'])

    def test_full_off_and_cleanup_health_not_replaced_by_unchanged_tokens(self):
        for change in ('off', 'health', 'lease', 'hook', 'host'):
            d = copy.deepcopy(self.rms); h = copy.deepcopy(self.rms_host)
            if change == 'off': d['requests'][8]['worker']['output']['logits']['sha256'] = '0'*64
            elif change == 'health': d['health_request']['worker']['output']['hidden']['sha256'] = '0'*64
            elif change == 'lease': d['c3']['recovery']['held_leases'].append('owned')
            elif change == 'hook': d['worker_core']['final_cleanup']['owned_serving_hooks'] = 1
            else: h['ownership_retained'] = True
            with self.subTest(change=change): self.assertFalse(self.audit(d, h)['consistent'])

    def test_failed_fault_records_safe_cleanup_without_manufacturing_c3(self):
        result = self.audit(self.failed, self.failed_host)
        self.assertFalse(result['source_bound_public_c2_c3_receipts_reconciled'])
        self.assertFalse(result['ownership_retained'])
        self.assertTrue(result['cleanup']['resource_release_confirmed'])
        self.assertIn('actual logits failed', result['original_error'])

    def test_partial_failure_cannot_claim_safe_release_without_original_zero_reply(self):
        h = copy.deepcopy(self.failed_host)
        row = next(r for r in h['history'] if r['action'] == 'unregister')
        row['response']['provider_state']['hook_owned'] = True
        result = self.audit(self.failed, h)
        self.assertFalse(result['cleanup']['resource_release_confirmed'])
        self.assertIn('ZERO_OWNED_RESOURCES_BEFORE_ORIGINAL_OWNER_RELEASE', {r['code'] for r in result['issues']})

    def test_source66_moe_error_cannot_use_zero_counters_as_stack_cleanup_proof(self):
        d = copy.deepcopy(self.failed); d['mechanism'] = '8.restore_offset'
        h = copy.deepcopy(self.failed_host)
        row = next(r for r in h['history'] if r['action'] == 'unregister')
        row['response']['provider_state']['serving']['failures'].append(dict(failed_phase=dict(
            independent=dict(moe_trace=dict(errors=['CPU nested mode exit failed'])))))
        result = self.audit(d, h)
        self.assertFalse(result['cleanup']['resource_release_confirmed'])
        self.assertFalse(result['ownership_retained'])  # Recorded release is distinct from proof of safety.
        self.assertIn('SOURCE66_MOE_ERROR_PATH_MODE_CLEANUP_NOT_PROVEN', {r['code'] for r in result['issues']})

    def test_native_rms_weight_object_and_all_independent_calls_are_required(self):
        for change in ('owner', 'omitted_native', 'output_storage', 'native_count'):
            d = copy.deepcopy(self.native); row = d['requests'][1]['worker']
            native = next(e for e in row['independent']['events'] if e.get('operator') in review.NATIVE_RMS)
            if change == 'owner': native['weight_owner_sources'][0]['module_object_id'] += 1
            elif change == 'output_storage': row['mechanism']['events'][0]['native_output_storage'] += 1
            elif change == 'native_count': row['independent']['operator_counts'][native['operator']] += 1
            else:
                extra = copy.deepcopy(native); extra['weight_owner_paths'] = ['omitted.actual.native']
                row['independent']['events'].append(extra)
                row['independent']['operator_counts'][native['operator']] += 1
            with self.subTest(change=change): self.assertFalse(self.audit(d, self.native_host)['consistent'])

    def test_real_captured_input_digest_binds_dtype_shape_and_original_objects(self):
        from demoncat.runtime.vllm_ascend.worker_extension.providers.tensor_modules import _identity
        from demoncat.runtime.vllm_ascend.worker_extension.serving_context import tensor_metadata
        from fault_injection.tensor_observer import metadata
        ids = core.torch.tensor([1, 2, 3], dtype=core.torch.int32); positions = core.torch.arange(3)
        item = copy.deepcopy(self.rms['requests'][0]); row = item['worker']; rid = row['request_id']
        row['input'] = _identity(((), dict(input_ids=ids, positions=positions, inputs_embeds=None, intermediate_tensors=None)))
        meta = dict(input_ids=tensor_metadata(ids), positions=tensor_metadata(positions))
        observed = [metadata(ids), metadata(positions)]
        for value in [*meta.values(), *observed]: value['device'] = 'npu:0'  # Schema fixture, never an NPU execution.
        row['independent']['forwards'][0]['input'] = observed
        row['scheduler'] = dict(internal_request_id=rid, request_object_id=1, num_requests=1,
            num_computed_tokens=0, num_prompt_tokens=3, num_scheduled_tokens=3, token_ids=[1, 2, 3],
            positions=[0, 1, 2], token_ids_sha256=review.sha([1, 2, 3]), block_ids=[1], block_size=16,
            slot_mapping=[16, 17, 18], input_tensor_metadata=meta)
        a = review.Review(); review.scheduler(a, item, row, True); self.assertFalse(a.errors)
        for change in ('digest', 'dtype', 'shape', 'same_value_other_object', 'slots'):
            altered = copy.deepcopy(item); r = altered['worker']
            if change == 'digest': r['input'][1]['input_ids']['sha256'] = '0'*64
            elif change == 'dtype': r['scheduler']['input_tensor_metadata']['input_ids']['dtype'] = 'torch.int64'
            elif change == 'shape': r['scheduler']['input_tensor_metadata']['input_ids']['shape'] = [1, 3]
            elif change == 'same_value_other_object': r['independent']['forwards'][0]['input'][0]['object_id'] += 1
            else: r['scheduler']['slot_mapping'][0] += 1
            a = review.Review(); review.scheduler(a, altered, row, True)
            with self.subTest(change=change): self.assertTrue(a.errors)

    def test_raw_rpc_and_host_artifact_hashes_are_checked(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); d = copy.deepcopy(self.rms); h = self.rms_host
            host = root/'host_receipt.json'; host.write_text(json.dumps(h))
            d['worker_core']['final_host_receipt'] = dict(path=host.name, sha256=review.file_sha(host))
            refs = []
            for i, row in enumerate(h['history']):
                relative = f"worker_core_rpc/{i:03d}-{row['action']}.json"; p = root/relative
                p.parent.mkdir(exist_ok=True); p.write_text(json.dumps(row))
                refs.append(dict(action=row['action'], path=relative, sha256=review.file_sha(p)))
            d['worker_core']['receipts'] = refs; path = root/'result.json'; path.write_text(json.dumps(d))
            a = review.Review(); review.load_artifacts(a, path, review.file_sha(path)); self.assertFalse(a.errors)
            (root/refs[6]['path']).write_text('{}')
            a = review.Review(); review.load_artifacts(a, path, review.file_sha(path))
            self.assertIn('ORIGINAL_RPC_BYTES_AND_HOST_HISTORY', a.errors)

    def test_frozen_source_scope_is_recomputed_and_source_edit_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); files = []
            for name in review.SOURCES:
                target = root/name; target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes((ROOT/name).read_bytes())
                files.append(dict(path=name, sha256=review.file_sha(target)))
            manifest = root/'source_manifest.json'
            manifest.write_text(json.dumps(dict(base_commit=review.SOURCE_COMMIT, files=files)))
            # Fixture pin only. The production CLI pin remains source66's exact manifest.
            with patch.object(review, 'SOURCE_MANIFEST', review.file_sha(manifest)):
                a = review.Review(); source, _ = review.source_review(a, root)
                self.assertFalse(a.errors)
                self.assertEqual(source['implementation_sha256'], self.rms['c2']['scope']['implementation_sha256'])
                (root/review.BINDINGS[0]).write_text('changed provider')
                a = review.Review(); review.source_review(a, root)
                self.assertIn('FROZEN_SOURCE_REHASH', a.errors)

    def test_import_and_moe_helpers_use_only_standard_library(self):
        program = """import importlib.util,sys
s=importlib.util.spec_from_file_location('reader',sys.argv[1]);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);m.moe_checks()
assert not any(n=='torch' or n.startswith(('torch.', 'torch_npu', 'demoncat.', 'fault_injection.')) for n in sys.modules)
print('STDLIB_ONLY')"""
        r = subprocess.run([sys.executable, '-I', '-B', '-c', program, str(SCRIPT)], capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stderr); self.assertEqual(r.stdout.strip(), 'STDLIB_ONLY')


class PublicTensorIndependentCleanupReviewTests(unittest.TestCase):
    def test_actual_late_logits_change_is_captured_and_off_refused(self):
        case = core.CoreTensorServingTests(); case.setUp()
        handle = None
        try:
            case.first_four(); case.request('c2-fault')
            def late(module, args, kwargs, output):
                if case.provider.pending and case.provider.pending[1] == 'c2_off': return output + 1
            handle = case.model.processor.register_forward_hook(late, with_kwargs=True)
            with self.assertRaisesRegex(RuntimeError, 'FULL_HIDDEN_LOGITS_OFF_NOT_EXACT'):
                case.request('c2-off')
            self.assertTrue(case.client.snapshot()['ownership_retained'])
            self.assertNotIn('C2', case.provider.reports)
            handle.remove(); handle = None
            case.recovery(); case.client.detach(); case.assert_clean()
        finally:
            if handle: handle.remove()
            case.tearDown()

    def test_actual_after_removal_exception_preserves_retryable_owner(self):
        from unittest.mock import patch
        case = core.CoreTensorServingTests(); case.setUp()
        try:
            case.start(); handle = case.provider.session.handles[0]; original = handle.remove
            def removed_then_fail():
                original(); raise RuntimeError('CPU after actual hook removal')
            with patch.object(handle, 'remove', side_effect=removed_then_fail):
                result = case.client.unregister()
            self.assertEqual(result['status'], 'PROVIDER_ACTION_FAILED')
            self.assertTrue(result['ownership_retained'])
            self.assertNotIn(handle, case.provider.session.handles)
            case.ok(case.client.unregister()); case.client.detach(); case.assert_clean()
        finally:
            case.tearDown()


if __name__ == '__main__': unittest.main()
