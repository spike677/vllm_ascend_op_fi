"""CPU protocol and synthetic Native ABI tests; no NPU qualification."""
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest

from experiments.worker_core_v11_20261118 import review26_group_originals as r


class Core26OriginalReviewTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from test_writer26_core_coverage_probe import Core26CoverageProbeTests
        cls.case=Core26CoverageProbeTests();cls.case.setUp();cls.addClassCleanup(cls.case.doCleanups)
        cls.document=cls.case.drive()
        cls.host,cls.rpc=r.C.load_rpc(r.C.Originals(cls.case.out,natural=False),cls.document,prefix='')
        cls.originals=r.C.Originals(cls.case.case.evidence_root,natural=False)
        cls.ref=cls.document['c2_attempt']['original_observations']['originals'][-1]
        cls.final=cls.originals.load(cls.ref,mounted=True);cls.b=r.OLD.helper('review_native26_single_source54.py')
        cls.context=r.C.protocol(r.C.Audit(),cls.document,cls.host,cls.rpc,'26.residual_stomp',natural=False)

    def run_review(self,d=None,natural=False,count=2):
        return r.review(self.document if d is None else d,self.host,self.rpc,self.originals,
            natural=natural,mechanism='26.residual_stomp',expected_count=count)

    def native(self,inv):
        a=r.C.Audit();artifacts=r.GroupArtifacts(self.originals,self.ref)
        value=a.section('native',r.native_group,self.b,a,inv,self.host['worker'],self.context,artifacts,natural=False,expected_count=2)
        return a,value,artifacts

    def test_real_public_cpu_topology_originals(self):
        result=self.run_review();self.assertTrue(result['consistent'],result['issues'])
        self.assertEqual(result['details']['coverage']['eligible_targets'],2)
        self.assertFalse(result['c2'] or result['c3'] or result['e5'])

    def test_cpu_native_double_is_rejected_in_natural_mode(self):
        self.assertFalse(self.run_review(natural=True)['consistent'])

    def test_wrong_expected_denominator_rejected(self):
        self.assertFalse(self.run_review(count=1)['consistent'])

    def test_probe_status_c2_success_ignored(self):
        d=deepcopy(self.document);d.update(status='UNRELATED_FLAG',c2=True,c3=True)
        result=self.run_review(d);self.assertTrue(result['consistent'],result['issues']);self.assertFalse(result['c2'])

    def test_failed_attempt_remains_failed_even_with_success_flags(self):
        d=deepcopy(self.document);d.update(error='Exact workload scope required',status='SUCCESS',c2=True)
        result=self.run_review(d)
        self.assertEqual(result['classification'],'FAILED_OR_PARTIAL_NOT_QUALIFIED')
        self.assertFalse(result['group_completion_reconciled'] or result['c2'])
        self.assertEqual(result['original_error'],d['error'])

    def test_missing_writer_callback_does_not_shrink_denominator(self):
        inv=deepcopy(self.final['inventory']);inv['rows'][5]['candidates'].pop()
        a,_,_=self.native(inv);self.assertFalse(a.report()['consistent'])

    def test_missing_target_lossless_outcome_rejected(self):
        inv=deepcopy(self.final['inventory']);inv['outcomes'].pop(2)
        a,_,_=self.native(inv);self.assertFalse(a.report()['consistent'])

    def test_foreign_writer_consumer_target_rejected(self):
        inv=deepcopy(self.final['inventory']);inv['outcomes'][6]['request_id']='foreign'
        a,_,_=self.native(inv);self.assertFalse(a.report()['consistent'])

    def test_reference_off_residual_hash_mismatch_rejected(self):
        inv=deepcopy(self.final['inventory']);inv['outcomes'][-1]['residual_after_sha256']='0'*64
        a,_,_=self.native(inv);self.assertFalse(a.report()['consistent'])

    def test_recovery_resource_leak_rejected(self):
        inv=deepcopy(self.final['inventory']);inv['coverage']['recovery']['workers'][0]['host_bytes']=8
        a,values,artifacts=self.native(inv)
        a.section('ledger',r.ledger,a,self.b,inv,*values,artifacts,self.context,natural=False)
        self.assertFalse(a.report()['consistent'])

    def test_original_rpc_hash_required(self):
        d=deepcopy(self.document);d['rpc_receipts'][1]['sha256']='0'*64
        with self.assertRaises(ValueError):r.C.load_rpc(r.C.Originals(self.case.out,natural=False),d,prefix='')


class Native26GroupABIReviewTests(unittest.TestCase):
    def setUp(self):
        from test_native26_single_source54_review import Fixture
        self.tmp=self.enterContext(tempfile.TemporaryDirectory());self.fixture=Fixture(self.tmp)
        self.b=r.OLD.helper('review_native26_single_source54.py')

    def inspect(self,change=None):
        f=self.fixture;effect=deepcopy(f.outcomes[3]);row=deepcopy(f.rows[5])
        candidate=next(x for x in row['candidates'] if x['callback']['target_key']==effect['target'])
        # Explicit synthetic group test envelope, not an original device receipt.
        candidate['callback']['complete_group_current_binding_verified']=True
        if change:change(effect,candidate)
        a=r.C.Audit();r.N.group_outcome(self.b,a,effect,candidate,row,
            f.inventory['single_site_c1']['reference_lossless_off'],'synthetic-group')
        return a.report()

    def test_original_abi_writer_consumer_restore_contract(self):
        result=self.inspect();self.assertTrue(result['consistent'],result['issues'])

    def test_zero_fault_effect_rejected(self):
        result=self.inspect(lambda e,c:e['consumer_receipt'].update(changed_residual_bytes=0))
        self.assertFalse(result['consistent'])

    def test_wrong_consumer_or_owner_rejected(self):
        self.assertFalse(self.inspect(lambda e,c:e['consumer_receipt'].update(consumer_calls=0))['consistent'])
        self.assertFalse(self.inspect(lambda e,c:e['lease'].update(owner_hi=0))['consistent'])

    def test_only_prior_single_site_flag_cannot_prove_complete_group_binding(self):
        self.assertFalse(self.inspect(lambda e,c:c['callback'].update(complete_group_current_binding_verified=False))['consistent'])


class CurrentForwardAbsenceReviewTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from test_core_writer26_coverage import CoreWriter26CoverageTests
        cls.case=CoreWriter26CoverageTests();cls.case.setUp();cls.addClassCleanup(cls.case.doCleanups)
        backend=cls.case.stale_producer();cls.row=deepcopy(backend.rows[1]);cls.lifetime=deepcopy(backend.report()['lifetime'])
        cls.b=r.OLD.helper('review_native26_single_source54.py');cls.case.ok(cls.case.client.unregister())

    def inspect(self,change=None):
        row=deepcopy(self.row)
        if change:change(row)
        a=r.C.Audit();value=a.section('absence',r.current_absence,self.b,a,row['candidates'][0],row,self.lifetime)
        return a.report(),value

    def test_stale_tracker_is_excluded_only_with_complete_current_trace(self):
        report,value=self.inspect();self.assertTrue(report['consistent'],report['issues'])
        self.assertEqual(value,('UNSUPPORTED',['CURRENT_FRESH_FORWARD_HAS_NO_RESIDUAL_PRODUCER']))

    def test_trace_gap_or_changed_thread_cannot_exclude_current_writer(self):
        self.assertFalse(self.inspect(lambda row:row['native_events'].pop(0))[0]['consistent'])
        self.assertFalse(self.inspect(lambda row:row['native_events'][0].update(thread=0))[0]['consistent'])

    def test_false_absence_witness_or_wrong_next_producer_rejected(self):
        self.assertFalse(self.inspect(lambda row:row['candidates'][0]['current_forward_residual_absence'].update(
            complete_native_events_sha256='0'*64))[0]['consistent'])
        self.assertFalse(self.inspect(lambda row:next(x for x in row['native_events'] if x['kind_name']=='add_rms_exit').update(
            producer=999))[0]['consistent'])


class Fia26TypedGroupReviewTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from test_core_writer26_coverage import CoreWriter26CoverageTests
        from fault_injection.deployments.writer26_group import packet
        cls.case=CoreWriter26CoverageTests();cls.case.setUp();cls.addClassCleanup(cls.case.doCleanups)
        cls.case.make(fia=True,count=3);native=cls.case.native
        cls.b=r.OLD.helper('review_native26_single_source54.py');cls.t=r.OLD.helper('review_native26_fia_single_source62.py')
        execute=native.execute
        def execute_typed(call,mode,target,allocation,*,typed_before):
            execute(call,mode,target,allocation,typed_before=typed_before)
            return dict(version=3,size=912,original_before=deepcopy(typed_before),lease=dict(version=2,size=248,
                mode=cls.b.MODE_IDS[mode],reserved=0,owner_hi=1,owner_lo=2,generation=1,thread=call.thread,
                target=cls.b.words(target),expected=packet(call),residual_allocation_end=0,output_allocation_begin=0,shift_bytes=0))
        start=cls.case.start
        def start_with_typed_receipt():
            start();complete=native.finished
            def complete_typed(pointer):
                token=pointer.contents.token;destination=native.fia_typed.snapshots[token]['before_execute']['effective']['raw']
                pointer.contents.replacement=destination;native.receipts[token]['replacement']=destination
                return complete(pointer)
            native.finished=complete_typed
        cls.case.start=start_with_typed_receipt;native.execute=execute_typed
        cls.inventory=cls.case.complete(3,2)

    def inspect(self,change=None):
        inv=deepcopy(self.inventory)
        if change:change(inv)
        return r.N.typed_group_review(self.b,self.t,inv,self.t.FiaArtifacts(self.b,self.case.evidence_root,
            recorded_root=self.case.evidence_root),expected_count=3)

    def test_every_selected_fia_has_typed_calls_and_stock_bytes(self):
        result=self.inspect();self.assertTrue(result['consistent'],result['issues'])
        self.assertEqual(result['stock_targets'],3);self.assertEqual(result['full_output_bins_rehashed'],16)
        self.assertEqual(result['typed_records'],58)

    def test_missing_second_target_stock_bridge_rejected(self):
        self.assertFalse(self.inspect(lambda v:v['fia_typed_stock_bridge']['reference_bridge_targets'].pop())['consistent'])

    def test_typed_descriptor_or_native_executor_drift_rejected(self):
        self.assertFalse(self.inspect(lambda v:v['outcomes'][0]['fia_typed_consumer_snapshot'].update(released=0))['consistent'])
        self.assertFalse(self.inspect(lambda v:v['outcomes'][3]['consumer_receipt'].update(executor=999))['consistent'])


if __name__=='__main__':unittest.main()
