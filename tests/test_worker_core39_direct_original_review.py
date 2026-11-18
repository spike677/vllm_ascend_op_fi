"""Actual CPU Host/Runtime originals, with an explicit test rotary profile.

Mutations affect audit copies only. The reader never imports the provider and
cannot turn the CPU profile or a saved success boolean into NPU qualification.
"""
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from experiments.worker_core_v11_20261118 import review39_group_originals as r


class DirectOriginalReviewTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import test_core_output39_direct_discovery as direct
        import test_core_output39_group as group
        from test_output39_group_worker_core_probe import Output39GroupProbeTests
        from demoncat.runtime.vllm_ascend.worker_extension import serving39_output_discovery as discovery
        from experiments.c2c3_postwrite_20261117.model_identity import model_source_identity
        direct.CoreOutput39DirectDiscoveryTests.setUpClass()
        cls.addClassCleanup(direct.CoreOutput39DirectDiscoveryTests.tearDownClass)
        sha = hashlib.sha256(Path(direct.__file__).read_bytes()).hexdigest()
        cls.cpu_profile = dict(module=direct.__name__, source_sha256=sha,
            operator_schema='vllm::npu_rotary_embedding(Tensor query, Tensor key) -> (Tensor, Tensor)')
        for modification in (patch.object(group, 'ParallelProjections', direct.DirectModel),
            patch.object(discovery, '_ROTARY_MODULE', direct.__name__),
            patch.object(discovery, '_ROTARY_SOURCE_SHA256', sha)):
            modification.start(); cls.addClassCleanup(modification.stop)
        cls.cases = {}
        shared = None
        for label, fail_index in [('complete', None), ('partial', 12)]:
            case = Output39GroupProbeTests()
            if shared is None:
                case.setUp(); cls.addClassCleanup(case.doCleanups); shared = case.case
            else:
                # A fresh public client/recorder on the same detached CPU
                # Worker avoids nesting the fixture's global device doubles.
                from test_output39_worker_core_probe import Driver
                from demoncat.runtime.vllm_ascend.worker_extension.host import HostAdapter, WorkerFaultClient
                from experiments.c2c3_postwrite_20261117 import natural39_worker_core_probe as probe
                from demoncat.runtime.vllm_ascend.worker_extension import serving39_output
                serving39_output.remove_fault39_output_serving_binding(shared.worker)
                folder = tempfile.TemporaryDirectory(); cls.addClassCleanup(folder.cleanup)
                case.case, case.out = shared, Path(folder.name)
                case.driver = Driver(shared)
                case.recorder = probe.RecordingExecutor(shared.executor, case.out/'worker_core_rpc')
                shared.client = WorkerFaultClient(HostAdapter(case.recorder,
                    prepare_binding=shared.client.host._binding), shared.client.config)
                case.report = dict(status='STARTED', stage='c2_c3', healthy_before=dict(tokens=[0], text='CPU fixture'))
            model_source = model_source_identity(case.case.model)
            case.driver.fail_index = fail_index
            document = case.drive()
            originals = r.C.Originals(case.out, natural=False)
            host, rpc = r.C.load_rpc(originals, document, prefix='')
            refs = [x['original_observations'] for x in document.get('requests', [])+document.get('recovery_requests', [])
                    if 'original_observations' in x]
            if not refs:
                raise AssertionError((label, document.get('error'), document.get('traceback')))
            root = os.path.commonpath([str(case.out), refs[0]['path']])
            final = r.C.read(refs[-1]['path'])
            cls.cases[label] = dict(case=case, document=document, host=host, rpc=rpc,
                root=root, final=final, model_source=model_source)

    def run_review(self, label='complete', document=None):
        data = self.cases[label]
        return r.review(data['document'] if document is None else document,
            data['host'], data['rpc'], r.C.Originals(data['root'], natural=False), natural=False,
            direct_cpu_profile=self.cpu_profile, frame_model_source=data['model_source'])

    def frames(self, mutate=None, *, natural=False):
        data = self.cases['complete']; tree = deepcopy(dict(group=data['final']['finished_groups']['c2'],
            rows=data['final']['actual_requests'], aliases=data['final']['module_aliases']))
        if mutate:
            mutate(tree)
        audit = r.C.Audit()
        audit.section('frames', r.FORWARD.check, audit, r.C, tree['group'], tree['rows'], data['host']['worker'],
            tree['aliases'], natural=natural, model_source=data['model_source'], cpu_profile=self.cpu_profile)
        return audit.report()

    @staticmethod
    def inventories(tree):
        values = tree['group']['independent_discovery']+[x['inventory'] for x in tree['group']['forward_artifacts']]
        values += [x['complete_inventory'] for x in tree['rows'] if x.get('complete_inventory') is not None]
        seen = set()
        for value in values:
            if id(value) not in seen:
                seen.add(id(value)); yield value

    def rejects(self, mutation, code):
        result = self.frames(mutation)
        self.assertFalse(result['consistent'])
        self.assertIn(code, [x['code'] for x in result['issues']])

    def test_current_real_cpu_direct_protocol_and_frames(self):
        result = self.run_review()
        self.assertTrue(result['consistent'], result['issues'])
        self.assertEqual((result['details']['c2']['recomputed_hits'], result['details']['c3']['recomputed_hits']), (2,4))
        self.assertTrue(result['details']['live_c2_history']['consumption_recorded'])
        self.assertFalse(result['c2'] or result['c3'] or result['e5'] or result['public_qualification_granted'])
        self.assertTrue(self.frames()['consistent'])

    def test_actual_partial_c3_preserves_completed_c2_and_recovery(self):
        original = self.cases['partial']['document']
        self.assertEqual(original['status'], 'REFUSED_OR_FAILED')
        self.assertIn('CPU_ROOT_FORWARD_FAILED', original['error'])
        result = self.run_review('partial')
        self.assertTrue(result['consistent'], result['issues'])
        self.assertEqual(result['classification'], 'FAILED_OR_PARTIAL_NOT_QUALIFIED')
        self.assertEqual(result['completed_stage_originals_reconciled'], {'c2': True})
        self.assertEqual(result['details']['c2']['recomputed_hits'], 2)
        self.assertTrue(result['details']['live_c2_history']['consumption_recorded'])
        self.assertFalse(result['group_completion_reconciled'] or result['c2'] or result['c3'] or result['e5'])
        self.assertIn('c3', result['failed_stage_reports'])

    def test_saved_success_cannot_erase_actual_partial(self):
        document = deepcopy(self.cases['partial']['document']); document.update(status='SUCCESS', c2=True, c3=True)
        result = self.run_review('partial', document)
        self.assertEqual(result['classification'], 'FAILED_OR_PARTIAL_NOT_QUALIFIED')
        self.assertFalse(result['c3'] or result['group_completion_reconciled'])

    def test_missing_direct_frame_is_not_an_observed_parent(self):
        def mutate(tree):
            for inv in self.inventories(tree):
                inv['actual_forward_frames'] = [x for x in inv['actual_forward_frames'] if x['entry_kind'] != 'direct_frame']
                for i,x in enumerate(inv['actual_forward_frames']): x['sequence']=i+1
        self.rejects(mutate, 'ACTUAL_MODULE_FRAME_COUNT_OR_EXPLICIT_GAP')

    def test_direct_frame_source_must_match_frozen_model_source(self):
        def mutate(tree):
            for inv in self.inventories(tree):
                for event in inv['actual_forward_frames']:
                    if event['entry_kind'] == 'direct_frame': event['source']['sha256']='0'*64
        self.rejects(mutate, 'DIRECT_FRAME_FROZEN_MODEL_SOURCE_PIN')

    def test_none_or_exception_unwind_is_not_successful_return(self):
        def mutate(tree):
            for inv in self.inventories(tree): inv['actual_forward_frames'][0]['nonnull_return']=False
        self.rejects(mutate, 'ACTUAL_BOUND_FRAME_RETURN_AND_HELD_OBJECT')

    def test_duplicate_frame_sequence(self):
        def mutate(tree):
            for inv in self.inventories(tree): inv['actual_forward_frames'][-1]['sequence']=1
        self.rejects(mutate, 'COMPLETE_NORMAL_FRAME_RETURNS')

    def test_missing_delegate_frame_cannot_be_replaced_by_count(self):
        def mutate(tree):
            for inv in self.inventories(tree):
                parent=inv['bounded_noncall_proofs'][0]['parent_identity']
                idx=next(i for i,x in enumerate(inv['actual_forward_frames']) if x['module_object_identity']==parent and x['attribute']=='_forward_method')
                inv['actual_forward_frames'].pop(idx)
                for i,x in enumerate(inv['actual_forward_frames']): x['sequence']=i+1
        self.rejects(mutate, 'ROTARY_EACH_PARENT_FRAME_HAS_ONE_ACTUAL_API_RETURN')

    def test_missing_actual_api_return(self):
        def mutate(tree):
            for inv in self.inventories(tree):
                event=next(x for x in inv['actual_forward_frames'] if x['actual_rotary_returns'])
                event['actual_rotary_returns']=0
        self.rejects(mutate, 'ROTARY_EACH_PARENT_FRAME_HAS_ONE_ACTUAL_API_RETURN')

    def test_called_helper_must_not_be_classified_noncall(self):
        def mutate(tree):
            for inv in self.inventories(tree):
                proof=inv['bounded_noncall_proofs'][0]
                inv['modules'].append(dict(path=proof['child_aliases'][0],object_identity=proof['child_identity'],calls=1))
        self.rejects(mutate, 'ROTARY_HELPER_ACTUALLY_UNCALLED')

    def test_full_shared_alias_set_required(self):
        def mutate(tree):
            for inv in self.inventories(tree): inv['bounded_noncall_proofs'][0]['child_aliases'].pop()
        self.rejects(mutate, 'ROTARY_FULL_PARENT_CHILD_ALIAS_SETS')

    def test_other_delegate_or_source_refused(self):
        for field,value in [('function','AscendRotaryEmbedding.different_branch'),('sha256','0'*64)]:
            def mutate(tree):
                for inv in self.inventories(tree):
                    proof=inv['bounded_noncall_proofs'][0]; parent=proof['parent_identity']
                    proof['delegate_binding']['source'][field]=value
                    for event in inv['actual_forward_frames']:
                        if event['module_object_identity']==parent and event['attribute']=='_forward_method': event['source'][field]=value
            self.rejects(mutate, 'ROTARY_REVIEWED_SOURCE_DELEGATE_AND_API')

    def test_actual_parent_delegate_gap_stays_unproven(self):
        def mutate(tree):
            for inv in self.inventories(tree):
                binding=inv['bounded_noncall_proofs'][0]['delegate_binding']
                inv['frame_binding_gaps'].append(dict(path=binding['path'],module_object_identity=binding['module_object_identity'],attribute='_forward_method',reason='unverified'))
        self.rejects(mutate, 'ROTARY_REVIEWED_SOURCE_DELEGATE_AND_API')

    def test_reason_string_without_frames_is_not_a_proof(self):
        def mutate(tree):
            for inv in self.inventories(tree): inv['bounded_noncall_proofs']=[]
        self.rejects(mutate, 'NO_INVENTED_ROTARY_NONCALL_ROW')

    def test_complete_api_count_cannot_hide_missing_operator(self):
        def mutate(tree):
            for inv in self.inventories(tree): inv['operator_counts'][r.FORWARD.ROTARY_OPERATOR]=0
        self.rejects(mutate, 'ROTARY_RETURNS_WITHIN_COMPLETE_API_COUNT')

    def test_api_schema_and_real_owner_are_not_freeform_labels(self):
        def schema(tree):
            for inv in self.inventories(tree):
                inv['bounded_noncall_proofs'][0]['operator_schema']='vllm::another_op(Tensor x) -> Tensor'
        self.rejects(schema, 'ROTARY_REVIEWED_SOURCE_DELEGATE_AND_API')
        def owner(tree):
            for inv in self.inventories(tree): inv['frame_observer_scope']['owner']+=1
        self.rejects(owner, 'ACTUAL_FRAME_WORKER_OWNER_PHASE')

    def test_live_consumption_must_refer_to_exact_original_object_and_scope(self):
        for field in ('object_binding', 'scope'):
            groups=deepcopy(self.cases['partial']['final']['finished_groups'])
            used=groups['c3']['coverage']['live_c2_proof']['consumed']
            if field=='object_binding':used['object_binding']['python_object_id']+=1
            else:used['consumer']['scope']['implementation_sha256']='0'*64
            audit=r.C.Audit();audit.section('proof',r.check_live_proof,audit,groups)
            self.assertFalse(audit.report()['consistent'])
            self.assertIn('ORIGINAL_LIVE_C2_OBJECT_CONSUMED_BY_C3',[x['code'] for x in audit.issues])

    def test_cpu_rotary_profile_never_becomes_natural(self):
        result=self.frames(natural=True)
        self.assertFalse(result['consistent'])
        self.assertIn('NATURAL_REJECTS_CPU_ROTARY_PROFILE',[x['code'] for x in result['issues']])

    def test_current_source_pin_set_includes_observer_and_complete_group_identity(self):
        from experiments.c2c3_postwrite_20261117 import natural39_worker_core_probe as probe
        _,p=r.OLD.helpers(r.OLD.CUSTOM_PROFILE)
        self.assertEqual(set(r.OLD.MODULES+r.EXTRA_MODULES+p.custom.EXTRA_MODULES),{x[1] for x in probe.MODULES})
        data=self.cases['complete']; source=Path(__file__).parents[1]
        document=dict(sources={name:r.C.file_sha(source/name) for name in probe.SOURCES},native_sha256='a'*64)
        final=deepcopy(data['final'])
        # Source pin logic is tested separately from the explicitly unavailable
        # NPU DSO/external-weight provenance of this CPU protocol fixture.
        with patch.object(r.OLD,'provenance',return_value={'consistent':True}):
            result=r.provenance(document,{},final,source,'b'*64,'c'*40,source/'absent',profile=r.OLD.CUSTOM_PROFILE)
        source_issues=[x for x in result['group_source_review']['issues'] if x['code']=='GROUP_IMPLEMENTATION_BINDS_ACTUAL_LOADED_SOURCE']
        self.assertEqual(source_issues,[])
        for group in final['finished_groups'].values(): group['source_pins'].pop('serving39_output_discovery.py')
        with patch.object(r.OLD,'provenance',return_value={'consistent':True}):
            result=r.provenance(document,{},final,source,'b'*64,'c'*40,source/'absent',profile=r.OLD.CUSTOM_PROFILE)
        self.assertTrue(any(x['code']=='GROUP_IMPLEMENTATION_BINDS_ACTUAL_LOADED_SOURCE' for x in result['group_source_review']['issues']))


class FailedNativeSchemaTests(unittest.TestCase):
    """Small declared CPU schema fixture, never an actual Native receipt."""
    def fixture(self):
        target=dict(module_object_identity=101, operator_api='vllm.unquantized_gemm.default')
        key=[101,target['operator_api'],0]
        geometry=dict(shape=[2,2],dtype='torch.bfloat16',device='cpu',logical_bytes=8,storage_bytes=8)
        events=[dict(sequence=i+1,operation=op,handle=11,owner=9,generation=7,returned=True,
            return_code=4 if op=='apply' else 0,error=None) for i,op in enumerate(('open','apply','restore','release','live'))]
        events[1].update(mode=1,indices=[0],changed_elements=0);events[4]['native_live']=False
        receipt=dict(schema='demoncat.postwrite.lease_receipt.v1',owner=9,generation=7,thread=8,handle=11,
            element_offset=0,elements=4,geometry=geometry,events=events,open=events[0],applications=[events[1]],
            restore=events[2],release=events[3],live_readbacks=[events[4]],first_failure=events[1],closed=True)
        document=dict(error="PostwriteRefused('POSTWRITE_APPLY_4')",owned_request_abort=['failed-request'],
            requests=[dict(phase='c3_fault',mapping=dict(internal_request_id='failed-request'))])
        final=dict(finished_groups=dict(c3=dict(failed_attempt='CPU_SCHEMA_FAILURE',native_owner=9,
            coverage=dict(recovery=dict(targets=[dict(lease=dict(target=target))])),
            native_lease_receipts=[dict(actual_request_id='failed-request',target_key=key,target_id=r.C.digest(target),receipt=receipt)])),
            failures=[dict(phase='c3_fault',partial_inventory=dict(frame_observer_scope=dict(owner=9,native_generation=7,thread=8),
                calls=[dict(key=key,module_object_identity=101,operator=target['operator_api'],original_calls=1,output_geometry=geometry)]))])
        return document,final

    def audit(self,document,final):
        a=r.C.Audit();value=a.section('failed',r.check_failed_native,a,document,final)
        return a.report(),value

    def test_recorded_rejected_apply_and_release_never_means_hit(self):
        report,value=self.audit(*self.fixture())
        self.assertTrue(report['consistent'],report['issues'])
        self.assertEqual(value['unique_rejected_native_receipts'],1)
        self.assertEqual(value['receipts'][0]['native_apply_return_code'],4)
        self.assertFalse(value['receipts'][0]['numeric_original_value_observed'])
        self.assertFalse(report['c2'] or report['c3'] or report['e5'])

    def test_wrong_request_owner_target_or_missing_release_refused(self):
        for kind in ('request','owner','target','live'):
            d,f=self.fixture();record=f['finished_groups']['c3']['native_lease_receipts'][0]
            if kind=='request':record['actual_request_id']='foreign'
            elif kind=='owner':record['receipt']['owner']+=1
            elif kind=='target':record['target_id']='0'*64
            else:record['receipt']['events'][4]['native_live']=True
            self.assertFalse(self.audit(d,f)[0]['consistent'],kind)

    def test_error_string_without_nonzero_original_return_is_insufficient(self):
        d,f=self.fixture();receipt=f['finished_groups']['c3']['native_lease_receipts'][0]['receipt']
        receipt['events'][1]['return_code']=0
        report,_=self.audit(d,f)
        self.assertFalse(report['consistent'])
        self.assertIn('NATIVE_APPLY_ERROR_REQUIRES_ORIGINAL_NONZERO_RETURN',[x['code'] for x in report['issues']])


if __name__ == '__main__':
    unittest.main()
