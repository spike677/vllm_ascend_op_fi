"""Current public CPU producer originals; no constructed receipt qualifies NPU."""
from copy import deepcopy
from pathlib import Path
import unittest
from unittest.mock import patch

from experiments.worker_core_v11_20261118 import review26_group_originals as r
from experiments.worker_core_v11_20261118 import review26_isolated_originals as isolated


class Isolated26OriginalReviewTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from test_writer26_core_isolated_probe import Core26IsolatedProbeTests
        cls.case=Core26IsolatedProbeTests();cls.case.setUp();cls.addClassCleanup(cls.case.doCleanups)
        cls.document=cls.case.drive()
        cls.host,cls.rpc=r.C.load_rpc(r.C.Originals(cls.case.out,natural=False),cls.document,prefix='')
        cls.originals=r.C.Originals(cls.case.case.evidence_root,natural=False)
        cls.ref=cls.document['c2_attempt']['original_observations']['originals'][-1]
        cls.final=cls.originals.load(cls.ref,mounted=True);cls.b=r.OLD.helper('review_native26_single_source54.py')
        cls.context=r.C.protocol(r.C.Audit(),cls.document,cls.host,cls.rpc,'26.residual_stomp',natural=False)

    def inspect(self,d=None,*,natural=False,count=3):
        return r.review(self.document if d is None else d,self.host,self.rpc,self.originals,
            mechanism='26.residual_stomp',expected_count=count,natural=natural)

    def native(self,inv):
        a=r.C.Audit();artifacts=r.GroupArtifacts(self.originals,self.ref)
        result=a.section('native',isolated.native,r,a,self.b,inv,self.context,artifacts,natural=False,expected_count=3)
        return a,result,artifacts

    def test_current_public_driver_twelve_group_requests_and_three_isolated_pairs(self):
        value=self.inspect();self.assertTrue(value['consistent'],value['issues'])
        self.assertTrue(value['group_completion_reconciled'])
        self.assertEqual(value['completed_original_requests'],12)
        summary=value['details']['coverage']
        self.assertEqual((summary['full_calls_per_request'],summary['eligible_targets'],summary['ledger_calls'],summary['owned_native_executes']),
            (3,3,18,15))
        self.assertFalse(value['c2'] or value['c3'] or value['e5'] or value['public_qualification_granted'])

    def test_saved_status_or_pass_booleans_are_not_authority(self):
        d=deepcopy(self.document);d.update(status='SUCCESS',c2=True,c3=True,qualification_granted=True)
        value=self.inspect(d);self.assertTrue(value['consistent'],value['issues']);self.assertFalse(value['c2'] or value['c3'])

    def test_current_cpu_fixture_cannot_pass_natural_mode_or_wrong_denominator(self):
        self.assertFalse(self.inspect(natural=True)['consistent'])
        self.assertFalse(self.inspect(count=2)['consistent'])

    def test_dispatch_is_explicit_and_old_cumulative_shape_cannot_accept_isolated(self):
        d=deepcopy(self.document);d['stage']='c2_once'
        self.assertFalse(self.inspect(d)['consistent'])

    def test_missing_original_delta_or_changed_rpc_mapping_refused(self):
        for change in (lambda d:d['requests'][5].pop('original_observations'),
                lambda d:d['requests'][6]['mapping'].update(internal_request_id='foreign'),
                lambda d:d['requests'].pop(8)):
            with self.subTest(change=change):
                d=deepcopy(self.document);change(d);self.assertFalse(self.inspect(d)['consistent'])

    def test_original_byte_change_is_rejected_without_rewriting_reference(self):
        ref=self.document['requests'][0]['original_observations'];path=Path(ref['path']);data=path.read_bytes()
        try:
            path.write_bytes(data+b' ');self.assertFalse(self.inspect()['consistent'])
        finally:path.write_bytes(data)

    def test_delta_chain_offset_null_and_final_merged_collection_changes_refused(self):
        load=self.originals.load;first=self.document['requests'][0]['original_observations']
        mutations=(
            (first,lambda d:d.update(previous_original_sha256='a'*64)),
            (first,lambda d:d['inventory']['complete_request_delta'].update(start=1)),
            (first,lambda d:d['inventory'].update(rows=None)),
            (self.ref,lambda d:d['inventory']['outcomes'].pop()),
            (self.ref,lambda d:d['inventory']['coverage']['calls'].pop()),
            (self.ref,lambda d:d['serving_requests'].pop()))
        for target,change in mutations:
            def wrong(ref,*args,**kwargs):
                value=load(ref,*args,**kwargs)
                if ref==target:value=deepcopy(value);change(value)
                return value
            with self.subTest(change=change),patch.object(self.originals,'load',side_effect=wrong):
                self.assertFalse(self.inspect()['consistent'])

    def test_missing_target_outcome_or_second_fault_in_same_request_refused(self):
        for change in (lambda v:v['rows'][5]['candidates'].pop(),lambda v:v['outcomes'].pop(10),
                lambda v:v['rows'][5]['callbacks'][1].update(claimed=1),
                lambda v:v['isolated_plan']['target_order'].pop()):
            with self.subTest(change=change):
                inv=deepcopy(self.final['inventory']);change(inv);a,_,_=self.native(inv)
                self.assertFalse(a.report()['consistent'])

    def test_prior_fault_contamination_and_different_off_are_not_independent_f_off(self):
        for change in (lambda v:v['outcomes'][9].update(residual_before_sha256='a'*64,residual_after_sha256='a'*64),
                lambda v:v['outcomes'][10].update(writer_output_sha256='a'*64),
                lambda v:v['rows'][5]['callbacks'][0]['isolated_pre_fault_baseline'].update(earlier_faults_this_request=1)):
            with self.subTest(change=change):
                inv=deepcopy(self.final['inventory']);change(inv);a,_,_=self.native(inv)
                self.assertFalse(a.report()['consistent'])

    def test_fake_isolated_pairs_do_not_supply_missing_native_execution(self):
        inv=deepcopy(self.final['inventory']);inv['isolated_pairs'][0]['off_request_id']='invented'
        a,_,_=self.native(inv);self.assertFalse(a.report()['consistent'])

    def test_ledger_false_admission_or_wrong_passthrough_is_rejected(self):
        for change in (lambda v:v['coverage']['calls'][1]['decision'].update(admitted=True,reason='ADMITTED'),
                lambda v:v['coverage']['calls'][1]['receipt'].update(native_applied=True),
                lambda v:v['coverage']['recovery']['targets'].pop()):
            with self.subTest(change=change):
                inv=deepcopy(self.final['inventory']);change(inv);a,values,artifacts=self.native(inv)
                a.section('ledger',isolated.ledger,r,a,self.b,inv,*values[:3],artifacts,self.context,natural=False,native_build_binding=None)
                self.assertFalse(a.report()['consistent'])

    def test_boolean_counts_and_target_pair_index_are_not_integer_observations(self):
        for change in (lambda v:v['rows'][5].update(admissions=True),
                lambda v:v['rows'][5]['callbacks'][0].update(claimed=True),
                lambda v:v['rows'][5]['plan'].update(isolated_pair=False),
                lambda v:v['rows'][5]['callbacks'][0]['isolated_pre_fault_baseline'].update(earlier_faults_this_request=False)):
            with self.subTest(change=change):
                inv=deepcopy(self.final['inventory']);change(inv);a,_,_=self.native(inv)
                self.assertFalse(a.report()['consistent'])

    def test_isolated_scope_adds_actual_source_identity_without_relabeling_cumulative(self):
        from types import SimpleNamespace
        root=Path(r.__file__).resolve().parents[2];prefix='fault_injection/deployments/'
        names=[prefix+name+'.py' for name in ('writer26_coverage','writer26_experiment','writer26_public','writer26_fia_typed')]
        names+=['fault_injection/environment.py','fault_injection/metadata_arrays.py']
        names+=[r.OLD.PREFIX+name+'.py' for name in ('serving26_core','serving26_isolated','providers/fault26_core')]
        document=dict(stage='c2_isolated_once',sources={name:r.C.file_sha(root/name) for name in names})
        args=SimpleNamespace(source_root=root,native_root=root,source_commit='a'*40,source_manifest_sha256='b'*64,
            native_build_sha256='c'*64,native_source_root=None,native_source_commit=None,native_source_manifest_sha256=None)
        with patch.object(r.OLD,'provenance',return_value=dict(consistent=True)):
            value=r.provenance(document,self.final,args)
        self.assertTrue(value['group_scope_source']['consistent'],value)
        document['stage']='c2_once'
        with patch.object(r.OLD,'provenance',return_value=dict(consistent=True)):
            value=r.provenance(document,self.final,args)
        self.assertFalse(value['group_scope_source']['consistent'])


class Isolated26PartialOriginalReviewTests(unittest.TestCase):
    def test_actual_pre_f_budget_refusal_preserves_five_completed_controls_no_f(self):
        from test_writer26_core_isolated_probe import Core26IsolatedProbeTests,probe
        case=Core26IsolatedProbeTests();case.setUp();self.addCleanup(case.doCleanups)
        original=probe.isolated_budget_preflight
        def too_slow(requests,plan,**kwargs):
            return original([dict(v,complete_cycle_seconds=200) for v in requests],plan,**kwargs)
        with patch.object(probe,'isolated_budget_preflight',too_slow):d=case.drive()
        host,records=r.C.load_rpc(r.C.Originals(case.out,natural=False),d,prefix='')
        result=r.review(d,host,records,r.C.Originals(case.case.evidence_root,natural=False),mechanism='26.residual_stomp',expected_count=3,natural=False)
        self.assertEqual(result['classification'],'FAILED_OR_PARTIAL_NOT_QUALIFIED')
        self.assertEqual(result['completed_original_requests'],5)
        self.assertFalse(result['group_completion_reconciled'] or result['c2'])
        self.assertFalse(any(row['phase']=='group_fault' for row in d['requests']))
        self.assertEqual(result['details']['partial']['complete_request_native_counts'],[3]*5)
        self.assertEqual(result['details']['partial']['recorded_fault_outcomes'],0)


class Isolated26FiaOriginalReviewTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import test_writer26_core_isolated_probe as fixture
        from fault_injection.deployments.writer26_group import packet
        make=fixture.fixture.CoreWriter26IsolatedTests.make
        def fia(self,**kwargs):return make(self,fia=True,count=4)
        cls.case=fixture.Core26IsolatedProbeTests()
        with patch.object(fixture.fixture.CoreWriter26IsolatedTests,'make',fia):cls.case.setUp()
        cls.addClassCleanup(cls.case.doCleanups)
        # Only the CPU double's ABI wire envelope is completed. The current
        # public driver still emits every original from real CPU execution.
        native=cls.case.case.native;execute=native.execute;install=native.install
        cls.b=r.OLD.helper('review_native26_single_source54.py');cls.t=r.OLD.helper('review_native26_fia_single_source62.py')
        def execute_typed(call,mode,target,allocation,*,typed_before):
            execute(call,mode,target,allocation,typed_before=typed_before)
            return dict(version=3,size=912,original_before=deepcopy(typed_before),lease=dict(version=2,size=248,
                mode=cls.b.MODE_IDS[mode],reserved=0,owner_hi=native.config.owner_hi,owner_lo=native.config.owner_lo,
                generation=native.config.generation,thread=call.thread,target=cls.b.words(target),expected=packet(call),
                residual_allocation_end=0,output_allocation_begin=0,shift_bytes=0))
        def install_typed(*args):
            result=install(*args);complete=native.finished
            def complete_typed(pointer):
                token=pointer.contents.token;destination=native.fia_typed.snapshots[token]['before_execute']['effective']['raw']
                pointer.contents.replacement=destination;native.receipts[token]['replacement']=destination
                return complete(pointer)
            native.finished=complete_typed;return result
        native.execute=execute_typed;native.install=install_typed
        cls.document=cls.case.drive()
        cls.host,cls.rpc=r.C.load_rpc(r.C.Originals(cls.case.out,natural=False),cls.document,prefix='')
        cls.originals=r.C.Originals(cls.case.case.evidence_root,natural=False)
        cls.ref=cls.document['c2_attempt']['original_observations']['originals'][-1]
        cls.final=cls.originals.load(cls.ref,mounted=True)

    def typed(self,change=None):
        inv=deepcopy(self.final['inventory'])
        if change:change(inv)
        artifacts=self.t.FiaArtifacts(self.b,self.case.case.evidence_root,recorded_root=self.case.case.evidence_root)
        a=r.C.Audit();value=a.section('typed',r.N.typed_group_review,self.b,self.t,inv,artifacts,expected_count=4,isolated=True)
        return value if value is not None else a.report()

    def test_actual_fia_cpu_driver_full_census_includes_unsupported_call(self):
        value=r.review(self.document,self.host,self.rpc,self.originals,mechanism='26.fia_destination',expected_count=4,natural=False)
        self.assertTrue(value['consistent'],value['issues'])
        summary=value['details']['coverage']
        self.assertEqual((summary['full_calls_per_request'],summary['eligible_targets'],summary['ledger_calls'],summary['owned_native_executes']),
            (4,3,24,15))
        self.assertFalse(value['c2'] or value['c3'] or value['e5'])

    def test_typed_every_isolated_execute_and_complete_stock_bytes(self):
        value=self.typed();self.assertTrue(value['consistent'],value['issues'])
        self.assertEqual(value['typed_records'],12*4*2+15)
        self.assertEqual(value['full_output_bins_rehashed'],2*4+15)
        self.assertFalse(value['c2'] or value['c3'] or value['e5'])

    def test_typed_wrong_selected_target_missing_record_or_owned_descriptor_leak_refused(self):
        for change in (lambda v:v['rows'][5]['fia_typed_records'].pop(),
                lambda v:v['rows'][5]['callbacks'][2].update(claimed=1),
                lambda v:v['outcomes'][9]['fia_typed_consumer_snapshot'].update(released=0)):
            with self.subTest(change=change):self.assertFalse(self.typed(change)['consistent'])


if __name__=='__main__':unittest.main()
