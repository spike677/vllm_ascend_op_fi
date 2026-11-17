"""Actual CPU calls exercise group contracts; never natural Worker evidence."""
from types import SimpleNamespace as NS
import sys
import threading
import unittest
import uuid
from unittest.mock import patch

import torch

from demoncat.runtime.vllm_ascend.worker_extension import postwrite41 as adapter
from demoncat.runtime.vllm_ascend.worker_extension.postwrite41_coverage import KVGroupCoverage
from fault_injection.coverage_session import CoverageScope
from fault_injection.deployments.postwrite_evidence import process_start
from fault_injection.deployments.postwrite_kv import live_value_tail_plan
from fault_injection.deployments.postwrite_kv_inventory import KVWorkloadInventory
from tests.test_postwrite_natural_kv import CpuSpanTransport, fixture


class CpuAttention(torch.nn.Module):
    def __init__(self, path):
        super().__init__()
        self.fixture = fixture()
        self.impl, self.layer_name = self.fixture['impl'], path
        self.kv_cache = [(self.impl.key_cache, self.impl.value_cache)]

    def forward(self, query, harness):
        p = harness.p
        p.stack.append((self.layer_name, self))
        p.workload.module_entered(self)
        # CPU fixture's actual writer runs before its actual value consumer.
        self.impl.value_cache.copy_(harness.initial_values[id(self)])
        p.writers[id(self)] = self.fixture['writer']
        p.workload.writer_returned(self, self.fixture['writer'])
        actual = dict(self.fixture['actual'], query=query)
        try:
            return p._consume('npu_fused_infer_attention_score',
                lambda **kw:kw['query'] + self.impl.value_cache[1, 1].sum(), (), actual)
        finally:p.stack.pop()


class CpuModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.a, self.b = CpuAttention('a'), CpuAttention('b')
    def forward(self, query, harness):return self.b(self.a(query, harness), harness)


class Harness:
    def __init__(self):
        p = adapter.Postwrite41WorkerProvider.__new__(adapter.Postwrite41WorkerProvider)
        self.p = p
        p.model, p.worker, p.runner = CpuModel(), NS(), NS(with_prefill=False)
        p.candidates = [('a',p.model.a),('b',p.model.b)]
        p.structural_candidates = list(p.candidates)
        p.transport, p.source = CpuSpanTransport(), {'cpu_contract':True}
        p.owner, p.generation, p.thread = uuid.uuid4().int % (2**63), 0, threading.get_ident()
        p.handles, p.patches, p.stack, p.collectors, p.cleanup_receipts = [], [], [], [], []
        p.current = p.expected = p.lease = p.collector = p.pending_run = None
        p.phase = p.target = p.coverage = None
        p.qualified = p.lossless = p.fault_attempted = False
        p.installed, p.tail_elements = True, 2
        p.rows, p.errors, p.history, p.outputs, p.independent_calls = [], [], [], [], []
        p.writers, p.forward_serial = {}, 0
        p.workload, p.process_start = KVWorkloadInventory(), process_start()
        p.model_identity = p._identity()
        p.install = lambda:setattr(p,'installed',True)
        self.context = NS(virtual_engine=0,
            attn_metadata={name:module.fixture['metadata'] for name,module in p.candidates},
            no_compile_layers=dict(p.candidates))
        self.initial_values = {id(module):module.impl.value_cache.clone() for _,module in p.candidates}
        self.input = torch.ones(1,4,4,dtype=torch.bfloat16)
        for phase in ('observe','observe','lossless','fault','off'):self.request(phase)
        scope = CoverageScope('41.tail_column', 'a'*64, 'b'*64, 'c'*64, 'cpu_contract')
        self.group = KVGroupCoverage(p,scope)
        for _ in range(2):self.request('group_observe')

    def request(self, phase):
        p = self.p
        p.expect(phase,'cpu-request-'+str(p.forward_serial+1),[1,2,3,4,5])
        p.forward_serial += 1
        p.current = dict(p.model.a.fixture['scheduler'], request_ids=list(p.expected),
                         prefixes=[dict(prompt=[1,2,3,4,5],output=[7])])
        p.pending_run = dict(phase=phase,forward_id=str(p.forward_serial),new_fault_hits=0,
            actual_scheduler=p.current, rows=p.rows)
        p.workload.begin(p.forward_serial,p.current)
        cpu_plan = lambda **kw:live_value_tail_plan(**kw,allow_cpu_test=True)
        try:
            with patch.dict(sys.modules,{'vllm.forward_context':NS(get_forward_context=lambda:self.context)}), \
                    patch.object(adapter,'live_value_tail_plan',side_effect=cpu_plan):
                result = p.model(self.input,self)
            p.outputs.append(result.detach().clone())
            p.pending_run['model_output_sha256'] = adapter._output_hash(p.outputs[-1])
            p.pending_run['independent_inventory'] = p.workload.finish(p.pending_run['model_output_sha256'])
        finally:
            p.workload.abort('CPU_MODEL_FORWARD_INCOMPLETE')
            p.current=None
        return p.finish_request()

    def recover(self):
        self.request('group_off')
        pending = self.group.begin_recovery()
        assert not pending['complete']
        assert not self.p.installed
        # An actual separate CPU model call after provider-owned hooks are gone.
        query = self.input
        for _,module in self.p.candidates:
            query = query + module.impl.value_cache[1,1].sum()
        self.group.health_artifact = dict(model_output_sha256=adapter._output_hash(query),cpu_contract=True)
        return self.group.finish_health()

    def run(self, trigger='once', baseline_control=False):
        self.group.prepare(trigger,baseline_control=baseline_control)
        self.request('group_lossless')
        for _ in range(2):self.request('group_fault')
        return self.recover()


class KVGroupCoverageTests(unittest.TestCase):
    def test_group_uninstall_failure_retains_ownership_and_retries_without_resealing(self):
        h=Harness();h.group.prepare('once');h.request('group_lossless')
        for _ in range(2):h.request('group_fault')
        h.request('group_off')
        original=h.p.close
        with patch.object(h.p,'close',side_effect=RuntimeError('external API owner')):
            with self.assertRaisesRegex(RuntimeError,'UNINSTALL_NOT_CONFIRMED.*external API owner'):
                h.group.begin_recovery()
        self.assertEqual(h.group.stage,'recovery_pending')
        self.assertTrue(h.p.installed)
        self.assertEqual(len(h.group.session.recovery_report()['held_leases']),2)
        before=h.group.observer.evidence
        report=h.group.retry_recovery()
        self.assertFalse(report['complete']) # Independent normal request still required.
        self.assertFalse(h.p.installed)
        self.assertEqual(h.group.observer.evidence,before)
        self.assertEqual(len(h.group.recovery_attempts),2)
        self.assertIn('external API owner',str(h.group.recovery_attempts[0]['errors']))
        h.group.health_artifact=dict(model_output_sha256=adapter._output_hash(h.p.outputs[0]))
        self.assertTrue(h.group.finish_health()['recovery']['complete'])

    def test_all_layers_once_then_every_match_have_independent_denominator_and_health(self):
        h = Harness()
        report = h.run()
        self.assertTrue(report['contract_satisfied'],report['rejection_reasons'])
        self.assertEqual((report['planned_targets'],report['hits'],report['observed_eligible_calls']),(2,2,4))
        self.assertTrue(report['recovery']['complete'])
        self.assertEqual((report['live_native_leases'],report['owned_hooks'],report['owned_api_patches']),(0,0,0))
        self.assertFalse(report['real_worker_evidence'])
        self.assertFalse(report['public_qualification_granted'])
        report = h.run('every_match')
        self.assertTrue(report['contract_satisfied'],report['rejection_reasons'])
        self.assertEqual((report['hits'],report['observed_eligible_calls']),(4,4))
        self.assertTrue(report['recovery']['complete'])

    def test_baseline_control_executes_each_target_without_c2_proof(self):
        h=Harness(); report=h.run(baseline_control=True)
        self.assertEqual((report['hits'],report['observed_eligible_calls']),(0,4))
        self.assertTrue(report['all_targets_controlled_once'])
        self.assertFalse(report['contract_satisfied'])
        self.assertIsNone(h.group.c2_live_proof)
        self.assertTrue(all(row['native']['changed_elements']==0 for row in report['executions']))

    def test_c3_requires_live_c2_and_c1_requires_real_effect(self):
        h=Harness()
        with self.assertRaisesRegex(RuntimeError,'LIVE_C2'):
            h.group.prepare('every_match')
        h.p.history=[row for row in h.p.history if row['phase']!='fault']
        with self.assertRaisesRegex(RuntimeError,'SAME_PROCESS_C1_EFFECT'):
            KVGroupCoverage(h.p,h.group.scope)

    def test_incomplete_module_universe_refuses_before_admission(self):
        h=Harness(); h.p.structural_candidates.append(('unsupported',NS()))
        h.p.history=[row for row in h.p.history if not row['phase'].startswith('group_')]
        with self.assertRaisesRegex(RuntimeError,'UNREVIEWED_ATTENTION_IMPL'):
            KVGroupCoverage(h.p,h.group.scope)

    def test_changed_healthy_query_refuses_target_lossless_before_session(self):
        h=Harness(); h.group.prepare('once'); h.input += 1
        with self.assertRaisesRegex(RuntimeError,'LOSSLESS_BYTES_OR_CONSUMER_MISMATCH'):
            h.request('group_lossless')
        self.assertIsNone(h.group.session)
        self.assertIsNone(h.p.lease)

    def test_recovery_retains_group_until_normal_request_exact_health(self):
        h=Harness(); h.group.prepare('once'); h.request('group_lossless')
        for _ in range(2):h.request('group_fault')
        h.request('group_off')
        report=h.group.begin_recovery()
        self.assertFalse(report['complete'])
        self.assertEqual(len(report['held_leases']),2)
        self.assertFalse(h.p.installed)
        self.assertIsNone(h.p.lease)
        h.group.health_artifact=dict(model_output_sha256='f'*64)
        with self.assertRaisesRegex(RuntimeError,'HEALTH_OUTPUT_MISMATCH'):h.group.finish_health()
        self.assertFalse(h.group.session.report()['contract_satisfied'])
        h.group.health_artifact=dict(model_output_sha256=adapter._output_hash(h.p.outputs[0]))
        self.assertTrue(h.group.finish_health()['recovery']['complete'])

    def test_bounded_stop_preserves_original_call_and_recovers_with_failed_contract(self):
        h=Harness(); h.group.prepare('once'); h.request('group_lossless')
        with patch('fault_injection.coverage_session.time.monotonic',return_value=1e30):
            with self.assertRaisesRegex(RuntimeError,'BOUNDED_STOP'):
                h.request('group_fault')
        h.group.abort_request('actual session deadline')
        report=h.recover()
        self.assertFalse(report['contract_satisfied'])
        self.assertTrue(report['recovery']['complete'])
        self.assertEqual(report['hits'],0)
        self.assertEqual(report['calls'][0]['receipt']['original_calls'],1)

    def test_c3_nonfinite_is_retained_as_failed_eligible_call_and_exact_off_recovers(self):
        h=Harness();self.assertTrue(h.run()['contract_satisfied'])
        h.group.prepare('every_match');h.request('group_lossless')
        h.input.fill_(float('nan'))
        with self.assertRaisesRegex(RuntimeError,'NONFINITE_INPUT_REFUSED'):h.request('group_fault')
        self.assertIsNone(h.p.lease)
        h.group.abort_request('actual CPU nonfinite query')
        h.input.fill_(1)
        report=h.recover()
        self.assertFalse(report['contract_satisfied'])
        self.assertEqual(report['planned_targets'],2)
        self.assertEqual(report['observed_eligible_calls'],1)
        self.assertEqual(report['calls'][0]['receipt']['original_calls'],1)
        self.assertEqual(report['hits'],0)
        self.assertTrue(report['recovery']['complete'])
        self.assertEqual(report['failures'][0]['input_before']['query_nonfinite'],16)


if __name__=='__main__':unittest.main()
