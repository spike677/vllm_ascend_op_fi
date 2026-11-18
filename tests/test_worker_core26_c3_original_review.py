"""Current public CPU producer/driver originals, not edited acceptance receipts."""
from copy import deepcopy
from pathlib import Path
import unittest

from experiments.worker_core_v11_20261118 import review26_group_originals as r
from experiments.worker_core_v11_20261118 import review26_c3_originals as c3


class Core26C3OriginalReviewTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from test_writer26_core_every_match_probe import Core26EveryMatchProbeTests
        cls.case=Core26EveryMatchProbeTests();cls.case.setUp();cls.addClassCleanup(cls.case.doCleanups)
        cls.document=cls.case.drive()
        cls.host,cls.rpc=r.C.load_rpc(r.C.Originals(cls.case.out,natural=False),cls.document,prefix='')
        cls.originals=r.C.Originals(cls.case.case.evidence_root,natural=False)
        cls.context=r.C.protocol(r.C.Audit(),cls.document,cls.host,cls.rpc,'26.residual_stomp',natural=False)
        cls.finals={key:cls.originals.load(cls.document[key+'_attempt']['original_observations']['originals'][-1],mounted=True)
            for key in ('c2','c3')}
        cls.b=r.OLD.helper('review_native26_single_source54.py')

    def inspect(self,d=None,count=2,natural=False):
        return r.review(d or self.document,self.host,self.rpc,self.originals,
                        mechanism='26.residual_stomp',expected_count=count,natural=natural)

    def native(self,inv):
        a=r.C.Audit();context=dict(self.context,run='c3',native_generation=self.context['ticket']['generation']+1)
        ref=self.document['c3_attempt']['original_observations']['originals'][-1]
        artifacts=r.GroupArtifacts(self.originals,ref,run='c3')
        value=a.section('native',r.native_group,self.b,a,inv,self.host['worker'],context,artifacts,natural=False,expected_count=2)
        return a,value,artifacts,context

    def test_current_public_driver_two_generations_all_originals(self):
        result=self.inspect();self.assertTrue(result['consistent'],result['issues'])
        self.assertTrue(result['group_completion_reconciled'])
        self.assertEqual(result['completed_original_requests'],17)
        self.assertEqual((result['details']['c2']['coverage']['recomputed_hits'],result['details']['c3']['coverage']['recomputed_hits']),(2,4))
        self.assertEqual(result['details']['c3']['coverage']['per_target_native_outcomes'],6)
        self.assertFalse(result['c2'] or result['c3'] or result['e5'] or result['public_qualification_granted'])

    def test_saved_status_and_qualification_flags_are_not_authority(self):
        d=deepcopy(self.document);d.update(status='SAVED_SUCCESS',c2=True,c3=True,qualification_granted=True)
        result=self.inspect(d);self.assertTrue(result['consistent'],result['issues'])
        self.assertFalse(result['c2'] or result['c3'] or result['e5'])

    def test_current_native_double_cannot_pass_natural_or_wrong_denominator(self):
        self.assertFalse(self.inspect(natural=True)['consistent'])
        self.assertFalse(self.inspect(count=1)['consistent'])

    def test_each_run_and_final_summary_must_join_raw_rpc(self):
        for change in (lambda d:d['requests'][8].update(run='c2'),
                       lambda d:d['c3_attempt']['public_core'].update(native_generation=999),
                       lambda d:d['c2_attempt']['coverage']['live_c2_proof']['issued'][0]['object_binding'].update(python_object_id=1)):
            with self.subTest(change=change):
                d=deepcopy(self.document);change(d);self.assertFalse(self.inspect(d)['consistent'])

    def test_native_second_fault_missing_target_or_outcome_rejected(self):
        for change in (lambda v:v['rows'][6]['candidates'].pop(),
                       lambda v:v['outcomes'].pop(8),
                       lambda v:v['outcomes'][8].update(request_id='foreign')):
            with self.subTest(change=change):
                inv=deepcopy(self.finals['c3']['inventory']);change(inv)
                a,_,_,_=self.native(inv);self.assertFalse(a.report()['consistent'])

    def test_c3_recovery_missing_target_or_original_sha_rejected(self):
        for change in (lambda v:v['coverage']['recovery']['targets'].pop(),
                       lambda v:v['coverage']['recovery']['targets'][0].update(proof_sha256='0'*64),
                       lambda v:v['coverage']['recovery']['workers'][0].update(host_bytes=1)):
            with self.subTest(change=change):
                inv=deepcopy(self.finals['c3']['inventory']);change(inv)
                a,values,artifacts,context=self.native(inv)
                a.section('ledger',r.ledger,a,self.b,inv,*values,artifacts,context,natural=False)
                self.assertFalse(a.report()['consistent'])

    def test_saved_consumed_proof_scope_object_or_generation_changed_rejected(self):
        complete=self.inspect();one=complete['details']['c2']['coverage'];two=complete['details']['c3']['coverage']
        for change in (lambda v:v['inventory']['coverage']['live_c2_proof']['consumed']['object_binding'].update(python_object_id=1),
                       lambda v:v['inventory']['coverage']['scope'].update(workload_sha256='0'*64),
                       lambda v:v.update(native_generation=1),
                       lambda v:v['inventory']['candidates'].pop()):
            with self.subTest(change=change):
                second=deepcopy(self.finals['c3']);change(second);a=r.C.Audit()
                a.section('transition',c3.transition,r,a,self.finals['c2'],second,one,two)
                self.assertFalse(a.report()['consistent'])

    def test_original_generation_fields_are_strict(self):
        from unittest.mock import patch
        ref=self.document['c3_attempt']['original_observations']['originals'][-1]
        state=next(v['state'] for v in self.context['states'] if v['action']=='finish')
        for generation in (999,True):
            final=deepcopy(self.finals['c3']);final['rpc_generation']=generation;a=r.C.Audit()
            with patch.object(self.originals,'load',return_value=final):
                r.load_original(a,self.originals,ref,state,self.context,natural=False,run='c3')
            self.assertFalse(a.report()['consistent'])


class PartialC3OriginalReviewTests(unittest.TestCase):
    def drive_and_review(self,*,blocked=False,failure=False):
        from test_writer26_core_every_match_probe import Core26EveryMatchProbeTests
        case=Core26EveryMatchProbeTests('test_blocked_c2_keeps_full_set_and_never_starts_c3_or_faults' if blocked else 'runTest')
        case.setUp();self.addCleanup(case.doCleanups)
        if failure:case.driver.fail_index=10
        doc=case.drive();originals=r.C.Originals(case.case.evidence_root,natural=False)
        host,records=r.C.load_rpc(r.C.Originals(case.out,natural=False),doc,prefix='')
        return r.review(doc,host,records,originals,mechanism='26.residual_stomp',expected_count=2,natural=False)

    def test_actual_c3_model_failure_preserves_completed_c2_and_no_grant(self):
        result=self.drive_and_review(failure=True)
        self.assertTrue(result['consistent'],result['issues'])
        self.assertEqual(result['classification'],'FAILED_OR_PARTIAL_NOT_QUALIFIED')
        self.assertTrue(result['details']['c2']['complete_originals_reconciled'])
        self.assertEqual(result['details']['c3']['collected_requests'],2)
        self.assertEqual(result['details']['c3']['observed_complete_request_native_counts'],[2,2])
        self.assertFalse(result['group_completion_reconciled'] or result['c2'] or result['c3'])

    def test_actual_blocked_c2_preserves_whole_inventory_no_fault_or_c3(self):
        result=self.drive_and_review(blocked=True)
        self.assertTrue(result['consistent'],result['issues'])
        self.assertEqual(result['blocked_run'],'c2')
        self.assertEqual(result['details']['c2']['observed_complete_request_native_counts'],[2,2,2])
        self.assertEqual(result['details']['c3']['collected_requests'],0)
        self.assertFalse(result['group_completion_reconciled'] or result['c3'])


class Fia26C3PublicOriginalReviewTests(unittest.TestCase):
    def test_actual_current_fia_public_driver_keeps_unsupported_writer_in_both_generations(self):
        from unittest.mock import patch
        from test_writer26_core_every_match_probe import Core26EveryMatchProbeTests
        from test_core_writer26_every_match import CoreWriter26EveryMatchTests
        original=CoreWriter26EveryMatchTests.make
        def fia(case,**kwargs):return original(case,fia=True,count=3,max_targets=kwargs.get('max_targets',128))
        case=Core26EveryMatchProbeTests()
        with patch.object(CoreWriter26EveryMatchTests,'make',fia):case.setUp()
        self.addCleanup(case.doCleanups);doc=case.drive()
        originals=r.C.Originals(case.case.evidence_root,natural=False)
        host,records=r.C.load_rpc(r.C.Originals(case.out,natural=False),doc,prefix='')
        result=r.review(doc,host,records,originals,mechanism='26.fia_destination',expected_count=3,natural=False)
        self.assertTrue(result['consistent'],result['issues'])
        self.assertTrue(result['group_completion_reconciled'])
        for name in ('c2','c3'):
            self.assertEqual(result['details'][name]['coverage']['projection_calls_per_request'],3)
            self.assertEqual(result['details'][name]['coverage']['eligible_targets'],2)
        self.assertEqual(result['details']['c3']['coverage']['recomputed_hits'],4)
        self.assertFalse(result['c2'] or result['c3'] or result['e5'])


class Fia26C3TypedOriginalReviewTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from test_core_writer26_every_match import CoreWriter26EveryMatchTests
        from fault_injection.deployments.writer26_group import packet
        cls.case=CoreWriter26EveryMatchTests();cls.case.setUp();cls.addClassCleanup(cls.case.doCleanups)
        cls.case.make(fia=True,count=3);native=cls.case.native
        cls.b=r.OLD.helper('review_native26_single_source54.py');cls.t=r.OLD.helper('review_native26_fia_single_source62.py')
        execute=native.execute;install=native.install
        # Extend this explicit CPU double's wire envelope, not a saved original
        # or a device producer. Both real public generations still execute.
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
        cls.case.complete(3,2)
        cls.inventory=cls.case.provider.backend.report()

    def inspect(self,change=None):
        inv=deepcopy(self.inventory)
        if change:change(inv)
        return r.N.typed_group_review(self.b,self.t,inv,self.t.FiaArtifacts(self.b,self.case.evidence_root,
            recorded_root=self.case.evidence_root),expected_count=3)

    def test_both_fia_faults_full_typed_denominator_and_complete_output_bytes(self):
        result=self.inspect();self.assertTrue(result['consistent'],result['issues'])
        self.assertEqual(result['stock_targets'],3)
        self.assertEqual(result['full_output_bins_rehashed'],18)
        self.assertEqual(result['typed_records'],66)
        self.assertFalse(result['c2'] or result['c3'] or result['e5'])

    def test_second_fault_missing_typed_record_or_descriptor_release_rejected(self):
        self.assertFalse(self.inspect(lambda v:v['rows'][6]['fia_typed_records'].pop())['consistent'])
        self.assertFalse(self.inspect(lambda v:v['outcomes'][8]['fia_typed_consumer_snapshot'].update(released=0))['consistent'])


class BoundEnvironmentOriginalReviewTests(unittest.TestCase):
    """Synthetic saved-address records only; no mapping or Native qualification."""
    def setUp(self):
        from experiments.worker_core_v11_20261118 import review26_environment_originals as env
        self.env=env
        names=['libdcat26_group_v2.so','libcust_opapi.so','libfault26_readonly_trace_abi0.so','libnnopbase.so','libascendcl.so']
        rows=[]
        for i,name in enumerate(names):
            rows.append(dict(path='/native/'+name,name=name,sha256=str(i+1)*64,elf_build_id=None,status='MAPPED_FILE_VERIFIED',
                mapping=dict(deleted=False,executable_segments=[dict(address=f'{(i+1)*4096:x}-{(i+2)*4096:x}',file_offset='0')]),
                path_identity_current=dict(inode=i+1)))
        bindings=[]
        def origin(row,symbol):
            base=int(row['mapping']['executable_segments'][0]['address'].split('-')[0],16)
            return dict(path=row['path'],mapped_path=row['path'],symbol=symbol,address=base+1,symbol_address=base+1,base=base)
        for getter,image,symbols in (('dcat26_group_cached_address_v1',rows[0],env.GROUP_SYMBOLS),
                ('dcat26_custom_cached_address_v1',rows[1],('aclnnAddRmsNormBias','aclnnAddRmsNormBiasGetWorkspaceSize'))):
            for i,symbol in enumerate(symbols):
                target=rows[3]
                bindings.append(dict(kind='CACHED_FUNCTION_POINTER',slot=i,symbol=symbol,getter=origin(image,getter),
                    getter_sha256=image['sha256'],target=origin(target,symbol),target_sha256=target['sha256'],target_elf_build_id=None))
        common=dict(machine='CPU_FIXTURE',python='fixture',packages={},torch_cxx11_abi=False,
                    driver=dict(running_module_versions={},installation_sha256=None))
        material=dict(common,profile='demoncat.bound_environment.v1',
            images=[{k:row[k] for k in ('path','name','sha256','elf_build_id')} for row in sorted(rows,key=lambda row:row['path'])],
            bindings=[dict(kind=v['kind'],getter_path=v['getter']['mapped_path'],getter_symbol=v['getter']['symbol'],
                getter_sha256=v['getter_sha256'],slot=v['slot'],symbol=v['symbol'],target_path=v['target']['mapped_path'],
                target_sha256=v['target_sha256']) for v in bindings])
        self.fingerprint=r.C.digest(material)
        self.value=dict(common,pid=123,maps_status='OBSERVED',loaded_libraries=rows,
            driver=dict(running_module_versions={},installation=None),bound_symbol_evidence=dict(
                schema='demoncat.bound_environment.v1',status='EXACT_MAPPED_FUNCTION_BINDINGS_OBSERVED',
                qualification_granted=False,serializable_authority=False,bindings=bindings,collision_groups=[],
                scope_material=material,fingerprint=self.fingerprint))
        self.images={row['name']:row['sha256'] for row in rows}

    def inspect(self,change=None):
        value=deepcopy(self.value)
        if change:change(value)
        a=r.C.Audit();a.section('bound_environment',self.env.review,r.C,a,value,self.fingerprint,
            dict(pid=123,native_writer_sha256=self.images['libdcat26_group_v2.so']),native_images=self.images)
        return a.report()

    def test_all_held_edges_and_scope_material_recomputed_without_symbol_resolution(self):
        result=self.inspect();self.assertTrue(result['consistent'],result['issues'])
        self.assertFalse(result['c2'] or result['c3'] or result['e5'])

    def test_missing_binding_foreign_image_address_or_changed_collateral_image_refused(self):
        for change in (lambda v:v['bound_symbol_evidence']['bindings'].pop(),
                       lambda v:v['bound_symbol_evidence']['bindings'][0]['target'].update(address=1),
                       lambda v:v['loaded_libraries'][2].update(sha256='f'*64),
                       lambda v:v['loaded_libraries'][1].update(status='MAPPED_FILE_IDENTITY_CHANGED'),
                       lambda v:v['bound_symbol_evidence'].update(fingerprint='0'*64)):
            with self.subTest(change=change):self.assertFalse(self.inspect(change)['consistent'])


if __name__=='__main__':unittest.main()
