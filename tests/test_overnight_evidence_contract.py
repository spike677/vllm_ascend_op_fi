import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import unittest
import subprocess

ROOT=Path(__file__).resolve().parents[1]


def module(name):
    import sys
    spec=importlib.util.spec_from_file_location(name,ROOT/'experiments/overnight_20261116'/(name+'.py'))
    result=importlib.util.module_from_spec(spec);sys.modules[name]=result;spec.loader.exec_module(result)
    return result


ec=module('evidence_contract')
design=module('collective25_design')


class EvidenceContractTests(unittest.TestCase):
    def scope(self):
        return dict(entrypoint='shared_worker',operator='aclnnMatmul',cann='actual-install',soc='observed-chip',
                    worker_start_id='pid-and-start',model_instance_id='model-object',build_sha256='a'*64,rank=0)

    def statistics(self):
        return dict(task='statistics',level='mix',summary_mode='statistics',rtol=1e-5,atol=1e-8,
                    reference='baseline',native_logs_read=False,private_tensors_read=False,aligned=True,
                    baseline_steps=[0],fault_steps=[0],steps=[dict(step=0,status='DIFFERENCE',unobservable_count=5,incomparable_count=0)])

    def record(self):
        scope=self.scope()
        gates={g:dict(status='PASS',scope_sha256=ec.scope_hash(scope)) for g in ('G5','G6','G7','G8')}
        gates['G6'].update(actual_calls=1,actual_execution_verified=True)
        gates['G8'].update(drained=True,off_verified=True,pending=0)
        return dict(mechanism='39.matmul_range',execution_kind='npu',scope=scope,gates=gates,
                    pretrained_model=True,request_mapping_verified=True,independent_statistics=self.statistics())

    def test_consistent_record_is_not_authentication_but_can_be_classified(self):
        self.assertEqual(ec.validate_execution(self.record(),self.scope(),require_model=True),'MODEL_CLOSED')

    def test_cpu_or_historical_entry_cannot_promote_shared_worker(self):
        for change in ('cpu','mock'):
            record=self.record();record['execution_kind']=change
            with self.assertRaisesRegex(ec.EvidenceError,'CPU_OR_MOCK'):ec.validate_execution(record,self.scope())
        record=self.record();record['scope']['entrypoint']='legacy_standalone'
        with self.assertRaisesRegex(ec.EvidenceError,'SCOPE_MISMATCH'):ec.validate_execution(record,self.scope())

    def test_actual_hit_count_without_execution_is_insufficient(self):
        record=self.record();record['gates']['G6']['actual_execution_verified']=False
        with self.assertRaisesRegex(ec.EvidenceError,'HIT_COUNT'):ec.validate_execution(record,self.scope())

    def test_foreign_lossless_receipt_and_pending_recovery_refuse(self):
        record=self.record();record['gates']['G5']['scope_sha256']='b'*64
        with self.assertRaisesRegex(ec.EvidenceError,'G5'):ec.validate_execution(record,self.scope())
        record=self.record();record['gates']['G8']['pending']=1
        with self.assertRaisesRegex(ec.EvidenceError,'RECOVERY'):ec.validate_execution(record,self.scope())

    def test_constructed_module_does_not_prove_pretrained_model(self):
        record=self.record();record['pretrained_model']=False
        self.assertEqual(ec.validate_execution(record,self.scope()),'NATIVE_VERIFIED')
        with self.assertRaisesRegex(ec.EvidenceError,'PRETRAINED'):ec.validate_execution(record,self.scope(),require_model=True)

    def test_nan_blocks_model_statistical_localization_but_null_is_retained(self):
        report=self.statistics();report['steps'][0].update(status='INCOMPARABLE',incomparable_count=1)
        self.assertFalse(ec.validate_statistics(report)['numerical_localization_complete'])
        record=self.record();record['independent_statistics']=report
        with self.assertRaisesRegex(ec.EvidenceError,'LOCALIZATION_INCOMPLETE'):ec.validate_execution(record,self.scope(),require_model=True)
        report['steps'][0]['status']='NO_OBSERVABLE_DIFFERENCE'
        with self.assertRaisesRegex(ec.EvidenceError,'NAN_CANNOT'):ec.validate_statistics(report)
        self.assertEqual(ec.validate_statistics(self.statistics())['nulls_preserved'],5)

    def test_native_logs_threshold_or_missing_step_cannot_select_first_diff(self):
        for key,value in [('native_logs_read',True),('rtol',1e-8),('fault_steps',[0,1]),('aligned',False)]:
            report=self.statistics();report[key]=value
            with self.assertRaises(ec.EvidenceError):ec.validate_statistics(report)

    def test_no_numeric_change_is_retained(self):
        report=self.statistics();report['steps'][0]['status']='NO_OBSERVABLE_DIFFERENCE'
        self.assertFalse(ec.validate_statistics(report)['observed_change'])

    def test_zero_or_injection_defined_denominator_not_coverage(self):
        report=dict(eligible_calls=3,executed_calls=3,completed_calls=3,restored_calls=3,
                    denominator_source='independent_observer',discovery_completed=True,
                    group_recovery_verified=True,resource_budget_verified=True,rank_isolation_verified=True)
        self.assertTrue(ec.validate_coverage(report).startswith('COVERAGE_VERIFIED'))
        for change in ({'eligible_calls':0,'executed_calls':0,'completed_calls':0,'restored_calls':0},
                       {'restored_calls':2},{'denominator_source':'mutation_hits'},{'group_recovery_verified':False}):
            with self.assertRaises(ec.EvidenceError):ec.validate_coverage(dict(report,**change))


class Collective25DesignTests(unittest.TestCase):
    def test_nonconsecutive_global_ranks_map_by_group_membership(self):
        plan=design.GroupPlan((2,5,7),9,2)
        self.assertEqual(plan.shard(5),(2,4));self.assertEqual(plan.send_count,6)
        data={r:[float(r)]*6 for r in plan.members}
        result=design.simulate(plan,data,(7,2,5))
        self.assertEqual(result,{2:[14.,14.],5:[14.,14.],7:[14.,14.]})

    def test_association_changes_real_cpu_fp32_sum_without_dropping_contribution(self):
        plan=design.GroupPlan((0,1,2),0,1)
        data={0:[1e8]*3,1:[1.]*3,2:[-1e8]*3}
        self.assertEqual(design.simulate(plan,data,(0,1,2)),{0:[0.],1:[0.],2:[0.]})
        self.assertEqual(design.simulate(plan,data,(0,2,1)),{0:[1.],1:[1.],2:[1.]})

    def test_invalid_or_duplicate_contributions_do_not_fake_rounding_mechanism(self):
        plan=design.GroupPlan((0,1,2),0,1);data={r:[float(r)]*3 for r in plan.members}
        for order in ((0,1),(0,1,1),(0,1,9)):
            with self.assertRaises(ValueError):design.simulate(plan,data,order)
        with self.assertRaises(ValueError):design.simulate(plan,{0:[1],1:[2],2:[3]},(0,1,2))

    def test_cpu_result_cannot_be_replayed_as_native_evidence(self):
        with self.assertRaises(ValueError):design.require_measured_delta({'source':'CPU_DESIGN_ONLY'}, {})
        identity={k:'observed' for k in ('process_group','members','sequence','operator','input_sha256',
            'layout_sha256','loaded_build_sha256','soc','dtype','reduce_op','send_count','receive_count')}
        sample=dict(source='OBSERVED_NATIVE_SAME_INPUT_PAIR',identity=identity,
                    native_execution_verified=True,input_equal_verified=True)
        with self.assertRaises(NotImplementedError):design.require_measured_delta(sample,identity)

    def test_unknown_sum_dtype_or_duplicate_rank_refuses(self):
        for args in (((0,0),0,1),((0,1),-1,1),((0,1),0,0)):
            with self.assertRaises(ValueError):design.GroupPlan(*args)
        with self.assertRaises(ValueError):design.GroupPlan((0,1),0,1,operation='max')


class CapabilityAuditTests(unittest.TestCase):
    def test_seventeen_distinct_semantics_and_no_inherited_execution_authority(self):
        from fault_injection.catalog import list_faults
        paths=list((ROOT/'profiles/capabilities').glob('*.json'))
        cards=[json.loads(p.read_text(encoding='utf-8')) for p in paths]
        self.assertEqual({r['mechanism'] for r in cards},{r['name'] for r in list_faults()})
        self.assertEqual(len(cards),17)
        for card in cards:
            self.assertFalse(card['historical_validation']['inherited_by_new_worker'])
            self.assertEqual(set(card['gates']),{'G'+str(i) for i in range(10)})
            self.assertFalse(any(g['accepted_for_current_worker'] for g in card['gates'].values()))
            self.assertTrue(card['next_action'])
            for ref in card['source_manifest']:
                self.assertTrue((ROOT/ref['path']).is_file(),ref['path'])

    def test_source_hashes_bind_actual_baseline_not_mutable_working_tree(self):
        checked={}
        for path in (ROOT/'profiles/capabilities').glob('*.json'):
            card=json.loads(path.read_text(encoding='utf-8'))
            for ref in card['source_manifest']:
                key=(ref['baseline_commit'],ref['path'])
                if key not in checked:
                    if key[0] is None:raw=(ROOT/key[1]).read_bytes()
                    else:raw=subprocess.check_output(['git','-C',str(ROOT),'show',key[0]+':'+key[1]])
                    checked[key]=(len(raw),hashlib.sha256(raw).hexdigest())
                self.assertEqual(checked[key],(ref['bytes'],ref['sha256']))


if __name__=='__main__':unittest.main()
