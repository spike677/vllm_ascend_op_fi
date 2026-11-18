"""Current combined producer CPU originals, never fabricated NPU acceptance."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from experiments.worker_core_v11_20261118 import review26_group_originals as r
from experiments.worker_core_v11_20261118 import review26_isolated_originals as i
from experiments.worker_core_v11_20261118 import review26_isolated_c3_originals as t


class IsolatedC326OriginalReviewTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from test_writer26_core_isolated_transition_probe import Core26IsolatedTransitionProbeTests
        cls.case=Core26IsolatedTransitionProbeTests();cls.case.setUp();cls.addClassCleanup(cls.case.doCleanups)
        cls.document=cls.case.drive()
        cls.host,cls.rpc=r.C.load_rpc(r.C.Originals(cls.case.out,natural=False),cls.document,prefix='')
        cls.originals=r.C.Originals(cls.case.case.evidence_root,natural=False)
        cls.context=r.C.protocol(r.C.Audit(),cls.document,cls.host,cls.rpc,'26.fia_destination',natural=False)
        cls.refs={run:cls.document[run+'_attempt']['original_observations']['originals'][-1] for run in ('c2','c3')}
        cls.finals={run:cls.originals.load(ref,mounted=True) for run,ref in cls.refs.items()}
        cls.b=r.OLD.helper('review_native26_single_source54.py')

    def inspect(self,document=None,*,natural=False,count=3):
        return r.review(self.document if document is None else document,self.host,self.rpc,self.originals,
            mechanism='26.fia_destination',expected_count=count,natural=natural)

    def mutate_original(self,reference,change):
        load=self.originals.load
        def mutated(ref,*args,**kwargs):
            value=load(ref,*args,**kwargs)
            if ref==reference:
                value=deepcopy(value);change(value)
            return value
        return patch.object(self.originals,'load',mutated)

    def test_actual_current_driver_both_delta_chains_and_distinct_plans(self):
        value=self.inspect();self.assertTrue(value['consistent'],value['issues'])
        self.assertTrue(value['group_completion_reconciled'])
        self.assertEqual(value['completed_original_requests'],19)
        c2,c3=value['details']['c2']['coverage'],value['details']['c3']['coverage']
        self.assertEqual((c2['eligible_targets'],c2['group_requests'],c2['owned_native_executes'],c2['ledger_calls'],c2['recomputed_hits']),
            (2,10,10,12,2))
        self.assertEqual((c3['eligible_targets'],c3['recomputed_hits']),(2,4))
        self.assertTrue(value['details']['transition']['same_saved_object_binding'])
        self.assertFalse(any(value[k] for k in ('c2','c3','e5','public_qualification_granted')))

    def test_probe_success_flags_never_grant_live_authority(self):
        d=deepcopy(self.document);d.update(status='SUCCESS',c2=True,c3=True,qualification_granted=True)
        value=self.inspect(d);self.assertTrue(value['consistent'],value['issues'])
        self.assertFalse(value['c2'] or value['c3'] or value['e5'])

    def test_cpu_originals_wrong_native_denominator_or_old_stage_refused(self):
        self.assertFalse(self.inspect(natural=True)['consistent'])
        self.assertFalse(self.inspect(count=2)['consistent'])
        for stage in ('c2_isolated_once','c2_c3','c2_once'):
            d=deepcopy(self.document);d['stage']=stage
            with self.subTest(stage=stage):self.assertFalse(self.inspect(d)['consistent'])

    def test_original_bytes_cannot_change_without_pinned_reference(self):
        ref=self.document['requests'][10]['original_observations'];path=Path(ref['path']);data=path.read_bytes()
        try:
            path.write_bytes(data+b' ');self.assertFalse(self.inspect()['consistent'])
        finally:path.write_bytes(data)

    def test_missing_delta_request_or_wrong_generation_cannot_inherit_other_run(self):
        changes=(lambda d:d['requests'][10].pop('original_observations'),lambda d:d['requests'].pop(12),
            lambda d:d['requests'][10].update(run='c2'),lambda d:d['requests'][10]['mapping'].update(internal_request_id='foreign'))
        for change in changes:
            d=deepcopy(self.document);change(d)
            with self.subTest(change=change):self.assertFalse(self.inspect(d)['consistent'])

    def test_c3_delta_chain_must_restart_and_full_merge_cannot_drop_native_calls(self):
        first=self.document['requests'][10]['original_observations']
        changes=((first,lambda d:d.update(previous_original_sha256=self.refs['c2']['sha256'])),
            (first,lambda d:d['inventory']['complete_request_delta'].update(start=10)),
            (first,lambda d:d.update(native_generation=d['rpc_generation'])),
            (first,lambda d:d['inventory'].update(rows=None)),
            (self.refs['c3'],lambda d:d['inventory']['rows'].pop()),
            (self.refs['c3'],lambda d:d['inventory']['coverage']['calls'].pop()),
            (self.refs['c3'],lambda d:d['inventory']['outcomes'].pop()))
        for ref,change in changes:
            with self.subTest(change=change),self.mutate_original(ref,change):self.assertFalse(self.inspect()['consistent'])

    def test_bad_later_c3_original_preserves_c2_without_marking_bad_c3_complete(self):
        first=self.document['requests'][10]['original_observations']
        with self.mutate_original(first,lambda d:d['inventory'].update(rows=None)):
            value=self.inspect()
        self.assertFalse(value['consistent'] or value['group_completion_reconciled'])
        self.assertTrue(value['details']['c2']['complete_originals_reconciled'])
        self.assertFalse(value['details']['c3']['complete_originals_reconciled'])
        with self.mutate_original(self.refs['c3'],lambda d:d['inventory']['coverage']['calls'].pop()):
            value=self.inspect()
        self.assertFalse(value['details']['c3']['complete_originals_reconciled'])

    def test_transition_driver_reference_must_be_c2_cleanup_not_last_collect(self):
        d=deepcopy(self.document);d['requests'][9]['original_observations']=d['c2_attempt']['original_observations']['originals'][-2]
        self.assertFalse(self.inspect(d)['consistent'])

    def test_isolated_pre_fault_contamination_and_different_off_are_rejected(self):
        changes=(lambda v:v['outcomes'][6].update(residual_before_sha256='a'*64),
            lambda v:v['outcomes'][7].update(writer_output_sha256='b'*64),
            lambda v:v['rows'][5]['callbacks'][1]['isolated_pre_fault_baseline'].update(earlier_faults_this_request=1))
        for change in changes:
            inv=deepcopy(self.finals['c2']['inventory']);change(inv)
            a=r.C.Audit();artifacts=r.GroupArtifacts(self.originals,self.refs['c2'])
            a.section('native',i.native,r,a,self.b,inv,self.context,artifacts,natural=False,expected_count=3,transition=True)
            with self.subTest(change=change):self.assertFalse(a.report()['consistent'])

    def test_c3_cumulative_cannot_relabel_as_isolated_or_drop_target(self):
        changes=(lambda v:v.update(independent_per_target_replay=True),lambda v:v['isolated_plan']['target_order'].pop(),
            lambda v:v['isolated_plan'].update(current_phase_scope=i.SCOPE),lambda v:v['isolated_pairs'].append({'index':0}))
        for change in changes:
            inv=deepcopy(self.finals['c3']['inventory']);change(inv);a=r.C.Audit()
            a.section('scope',t.cumulative_scope,r,a,inv,3)
            with self.subTest(change=change):self.assertFalse(a.report()['consistent'])

    def test_old_scope_proof_new_object_owner_or_generation_is_not_transferable(self):
        changes=(lambda v:v['coverage']['live_c2_proof']['consumed']['object_binding'].update(python_object_id=1),
            lambda v:v['coverage'].update(generation=v['coverage']['generation']+1),
            lambda v:v['coverage'].update(owner='another-owner'),
            lambda v:v['coverage']['scope'].update(implementation_sha256='a'*64),
            lambda v:v['candidates'].pop())
        for change in changes:
            final=deepcopy(self.finals['c3']);change(final['inventory']);a=r.C.Audit()
            _,old=t.helpers(r)
            a.section('transition',old.transition,r,a,self.finals['c2'],final,
                dict(projection_calls_per_request=3,eligible_targets=2,recomputed_hits=2),
                dict(projection_calls_per_request=3,eligible_targets=2,recomputed_hits=4))
            with self.subTest(change=change):self.assertFalse(a.report()['consistent'])

    def test_final_native_resource_leak_or_replaced_prior_c2_is_rejected(self):
        with self.mutate_original(self.refs['c3'],lambda d:d['inventory'].update(owned_hooks=1)):
            self.assertFalse(self.inspect()['consistent'])
        d=deepcopy(self.document);d['c2_attempt']['coverage']['live_c2_proof']['issued'][0]['object_binding']['python_object_id']=1
        self.assertFalse(self.inspect(d)['consistent'])

    def test_both_source_scopes_are_new_transition_hash_not_source82_scope(self):
        root=Path(r.__file__).resolve().parents[2];prefix='fault_injection/deployments/'
        names=[prefix+v+'.py' for v in ('writer26_coverage','writer26_experiment','writer26_public','writer26_fia_typed')]
        names+=['fault_injection/environment.py','fault_injection/metadata_arrays.py']
        names+=[r.OLD.PREFIX+v+'.py' for v in ('serving26_core','serving26_isolated','providers/fault26_core')]
        d=dict(stage=t.STAGE,sources={name:r.C.file_sha(root/name) for name in names})
        args=SimpleNamespace(source_root=root,native_root=root,source_commit='a'*40,source_manifest_sha256='b'*64,
            native_build_sha256='c'*64,native_source_root=None,native_source_commit=None,native_source_manifest_sha256=None)
        for run,final in self.finals.items():
            with self.subTest(run=run),patch.object(r.OLD,'provenance',return_value=dict(consistent=True)):
                self.assertTrue(r.provenance(d,final,args)['group_scope_source']['consistent'])
                old=dict(d,stage='c2_isolated_once')
                self.assertFalse(r.provenance(old,final,args)['group_scope_source']['consistent'])


class IsolatedC326PartialOriginalReviewTests(unittest.TestCase):
    def run_refusal(self,run):
        from test_writer26_core_isolated_transition_probe import Core26IsolatedTransitionProbeTests,probe
        case=Core26IsolatedTransitionProbeTests();case.setUp();self.addCleanup(case.doCleanups)
        actual=probe.transition_budget
        def late(requests,plan,reference,**kwargs):
            if kwargs['run']==run:kwargs['elapsed_seconds']=1800
            return actual(requests,plan,reference,**kwargs)
        with patch.object(probe,'transition_budget',late):d=case.drive()
        host,records=r.C.load_rpc(r.C.Originals(case.out,natural=False),d,prefix='')
        value=r.review(d,host,records,r.C.Originals(case.case.evidence_root,natural=False),mechanism='26.fia_destination',expected_count=3,natural=False)
        return d,value

    def test_actual_c2_budget_refusal_retains_controls_and_no_f(self):
        d,value=self.run_refusal('c2');self.assertTrue(value['consistent'],value['issues'])
        self.assertEqual(value['classification'],'FAILED_OR_PARTIAL_NOT_QUALIFIED')
        self.assertEqual(value['completed_original_requests'],5)
        self.assertEqual(value['details']['c2']['recorded_fault_outcomes'],0)
        self.assertFalse(value['group_completion_reconciled'] or value['c2'] or value['c3'])

    def test_actual_model_unwind_preserves_completed_c2_and_reads_failed_c3_cleanup(self):
        from test_writer26_core_isolated_transition_probe import Core26IsolatedTransitionProbeTests
        case=Core26IsolatedTransitionProbeTests();case.setUp();self.addCleanup(case.doCleanups)
        case.driver.fail_index=12;d=case.drive()
        host,records=r.C.load_rpc(r.C.Originals(case.out,natural=False),d,prefix='')
        value=r.review(d,host,records,r.C.Originals(case.case.evidence_root,natural=False),mechanism='26.fia_destination',expected_count=3,natural=False)
        self.assertTrue(value['consistent'],value['issues'])
        self.assertEqual(value['completed_original_requests'],12)
        self.assertTrue(value['details']['c2']['complete_originals_reconciled'])
        self.assertEqual(value['details']['partial_cleanup']['c3']['collected_before_failure'],2)
        self.assertFalse(value['group_completion_reconciled'] or value['c3'])

    def test_actual_c3_budget_refusal_keeps_completed_isolated_c2_and_no_c3_f(self):
        d,value=self.run_refusal('c3');self.assertTrue(value['consistent'],value['issues'])
        self.assertEqual(value['classification'],'FAILED_OR_PARTIAL_NOT_QUALIFIED')
        self.assertEqual(value['completed_original_requests'],15)
        self.assertTrue(value['details']['c2']['complete_originals_reconciled'])
        self.assertEqual(value['details']['c3']['recorded_fault_outcomes'],0)
        self.assertFalse(value['details']['c3']['complete_originals_reconciled'])
        self.assertFalse(value['group_completion_reconciled'] or value['c2'] or value['c3'])


class IsolatedC326TypedOriginalReviewTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from test_writer26_core_isolated_transition_probe import Core26IsolatedTransitionProbeTests
        from fault_injection.deployments.writer26_group import packet
        cls.case=Core26IsolatedTransitionProbeTests();cls.case.setUp();cls.addClassCleanup(cls.case.doCleanups)
        # Complete the explicit CPU double's ABI wire envelope. All files still
        # come from the actual public producer, with no Native/NPU qualification.
        native=cls.case.case.native;execute=native.execute;install=native.install
        cls.b=r.OLD.helper('review_native26_single_source54.py');cls.typed=r.OLD.helper('review_native26_fia_single_source62.py')
        def execute_typed(call,mode,target,allocation,*,typed_before):
            execute(call,mode,target,allocation,typed_before=typed_before)
            return dict(version=3,size=912,original_before=deepcopy(typed_before),lease=dict(version=2,size=248,
                mode=cls.b.MODE_IDS[mode],reserved=0,owner_hi=native.config.owner_hi,owner_lo=native.config.owner_lo,
                generation=native.config.generation,thread=call.thread,target=cls.b.words(target),expected=packet(call),
                residual_allocation_end=0,output_allocation_begin=0,shift_bytes=0))
        def install_typed(*args):
            value=install(*args);complete=native.finished
            def complete_typed(pointer):
                token=pointer.contents.token;destination=native.fia_typed.snapshots[token]['before_execute']['effective']['raw']
                pointer.contents.replacement=destination;native.receipts[token]['replacement']=destination
                return complete(pointer)
            native.finished=complete_typed;return value
        native.execute=execute_typed;native.install=install_typed
        cls.document=cls.case.drive();cls.originals=r.C.Originals(cls.case.case.evidence_root,natural=False)
        cls.inv={run:cls.originals.load(cls.document[run+'_attempt']['original_observations']['originals'][-1],mounted=True)['inventory']
            for run in ('c2','c3')}

    def inspect(self,run,change=None):
        inv=deepcopy(self.inv[run])
        if change:change(inv)
        artifacts=self.typed.FiaArtifacts(self.b,self.case.case.evidence_root,recorded_root=self.case.case.evidence_root)
        a=r.C.Audit();value=a.section('typed',r.N.typed_group_review,self.b,self.typed,inv,artifacts,expected_count=3,
            isolated=run=='c2',isolated_scope=t.SCOPE)
        return value if value is not None else a.report()

    def test_every_generation_typed_record_and_stock_owned_output_bytes(self):
        for run,requests,owned in (('c2',10,10),('c3',9,12)):
            with self.subTest(run=run):
                value=self.inspect(run);self.assertTrue(value['consistent'],value['issues'])
                self.assertEqual(value['typed_records'],requests*3*2+owned)
                self.assertEqual(value['full_output_bins_rehashed'],6+owned)
                self.assertFalse(value['c2'] or value['c3'] or value['e5'])

    def test_c3_missing_typed_call_or_wrong_current_destination_rejected(self):
        for change in (lambda v:v['rows'][6]['fia_typed_records'].pop(),
                lambda v:v['outcomes'][8]['fia_typed_consumer_snapshot'].update(released=0),
                lambda v:v['outcomes'][8]['consumer_receipt'].update(replacement=1)):
            with self.subTest(change=change):self.assertFalse(self.inspect('c3',change)['consistent'])

    def test_owned_full_output_byte_change_is_not_hidden_by_summary(self):
        artifact=self.inv['c3']['fia_typed_stock_bridge']['artifacts'][-1]
        # Artifact names are read from the real emitted record, not guessed.
        paths=[v for k,v in artifact.items() if k in ('path','file','binary_path','data_path')]
        self.assertEqual(len(paths),1,artifact)
        path=Path(paths[0]);data=path.read_bytes()
        try:
            path.write_bytes(data[:-1]+bytes([data[-1]^1]));self.assertFalse(self.inspect('c3')['consistent'])
        finally:path.write_bytes(data)


if __name__=='__main__':unittest.main()
