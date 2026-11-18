"""Actual public CPU driver budget stop, retaining incomplete C2 and recovery."""
from copy import deepcopy
import json
import os
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import test_output39_refusal_probe as fixture_module
import fault_injection.coverage_session as session_module
from experiments.worker_core_v11_20261118 import review39_refusal_prefix as P


class RefusalPrefixTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fixture=fixture_module.DiagnosticProbeTests();fixture.setUp();cls.addClassCleanup(fixture.doCleanups)
        fixture.report.update(schema='demoncat.output39.fixed_c3_refusal_probe.v1',mechanism='39.output_proxy',
            c2=False,c3=False,e5=False,qualification_granted=False,sources=fixture.pins,plan=fixture.plan.__dict__)
        count=[0]
        def monotonic():
            count[0]+=1
            # Session start, first and second calls are at zero. The third,
            # non-target call of the real first F observes the expired budget.
            return 0.0 if count[0]<4 else 1201.0
        with patch.object(session_module,'time',SimpleNamespace(monotonic=monotonic)):
            result=fixture.run_driver()
        cls.document=json.loads(json.dumps(result));cls.fixture=fixture
        cls.host=P.C.read(fixture.out/result['host_receipt']['path'])
        cls.records=[dict(P.C.read(fixture.out/'rpc'/r['path']),_reference=r) for r in result['rpc_receipts']]
        cls.diag=P.C.read(result['diagnostic_original']['path'])
        cls.root=os.path.commonpath([str(fixture.out),result['requests'][0]['original_observations']['path']])
        cls.final=P.C.read(result['recovery_requests'][-1]['original_observations']['path'])

    def review(self,**overrides):
        params=dict(d=deepcopy(self.document),host=deepcopy(self.host),records=deepcopy(self.records),
            originals=P.C.Originals(self.root,natural=False),diag=deepcopy(self.diag),natural=False)
        params.update(overrides);return P.review(**params)

    def test_actual_budget_stop_is_consistent_partial_not_apply4_or_c3(self):
        result=self.review()
        self.assertEqual(result['issues'],[],result)
        self.assertTrue(result['consistent'])
        self.assertEqual(result['details']['prefix']['completed_independent_fault_targets'],1)
        self.assertEqual(result['details']['prefix']['remaining_targets'],1)
        self.assertEqual(result['details']['prefix']['first_stop']['sequence'],2)
        self.assertEqual(result['details']['prefix']['first_stop']['cause_interpretation'],
                         'ELAPSED_LIMIT_BY_PINNED_SOURCE_BRANCH_ELIMINATION')
        self.assertEqual(result['details']['native_universe']['unique_leases'],5)
        self.assertFalse(result['c2_completed']);self.assertFalse(result['c3_started'])
        self.assertFalse(result['native_apply4_reproduced']);self.assertFalse(result['complete_counterfactual_chain'])

    def test_toplevel_error_string_cannot_substitute_first_budget_decision(self):
        group=deepcopy(self.final['finished_groups']['c2']);group['coverage']['calls'][2]['decision']['reason']='NON_TARGET'
        a=P.C.Audit();P.ledger(a,group,self.final['actual_requests'],natural=False)
        self.assertTrue(a.issues)
        self.assertIn('ACTUAL_FIRST_BUDGET_STOP_AND_ELIGIBLE_ABORT',[x['code'] for x in a.issues])

    def test_omitted_non_target_before_stop_is_rejected(self):
        group=deepcopy(self.final['finished_groups']['c2']);group['coverage']['calls'].pop(1)
        a=P.C.Audit();P.ledger(a,group,self.final['actual_requests'],natural=False)
        self.assertIn('FULL_LEDGER_INCLUDES_FAILED_PREFIX_AND_NONTARGETS',[x['code'] for x in a.issues])

    def test_c3_or_new_live_proof_cannot_be_added_to_partial_group(self):
        group=deepcopy(self.final['finished_groups']['c2']);group['coverage']['live_c2_proof']['issued']=[{'fabricated':True}]
        a=P.C.Audit();P.ledger(a,group,self.final['actual_requests'],natural=False)
        self.assertIn('NO_LIVE_C2_PROOF_ISSUED_OR_CONSUMED',[x['code'] for x in a.issues])

    def test_a_forged_completed_target_is_not_accepted_by_hit_count(self):
        group=deepcopy(self.final['finished_groups']['c2']);group['coverage']['hits']+=1
        a=P.C.Audit();P.ledger(a,group,self.final['actual_requests'],natural=False)
        self.assertIn('INCOMPLETE_C2_COUNTS_REBUILT_WITHOUT_PROMOTION',[x['code'] for x in a.issues])

    def test_partial_group_cannot_borrow_a_different_worker_lifetime(self):
        group=deepcopy(self.final['finished_groups']['c2']);worker=deepcopy(group['worker']);worker['pid']+=1
        a=P.C.Audit();P.ledger(a,group,self.final['actual_requests'],natural=False,worker=worker)
        self.assertIn('EXACT_PARTIAL_GROUP_WORKER_LIFETIME_AND_SCOPE',[x['code'] for x in a.issues])

    def test_script_main_helper_namespace_still_requires_same_pinned_source(self):
        records=deepcopy(self.records)
        for r in records:
            if r['method'] in P.R.INSTALL_METHODS+P.R.FINISH_METHODS:
                r['method']='__main__.'+r['method'].rsplit('.',1)[1]
        self.assertTrue(self.review(records=records)['consistent'])
        diag=deepcopy(self.diag);diag['helper']['source_sha256']='0'*64
        self.assertFalse(self.review(records=records,diag=diag)['consistent'])


if __name__=='__main__':unittest.main()
