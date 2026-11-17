"""Actual CPU forwards exercise identity plumbing, not NPU qualification."""
import copy
from dataclasses import asdict, replace
import os
from types import SimpleNamespace
import unittest

import torch

from fault_injection.coverage import (
    BoundaryTargetIdentity, PythonHookBoundary, TensorAliasBoundary, OperatorApiBoundary,
    KernelLaunchBoundary, CollectiveBoundary, Candidate, CandidateSet, Eligibility,
    EligibilityResult, ObservedCall, Phase, TargetLease, IndependentObservation,
    CallOutcome, Outcome, CoverageReport, TargetIdentity)
from fault_injection.core import UnsupportedCall
from fault_injection.model_discovery import inspect_structure
from fault_injection.module_faults import ModuleFaultSession
from fault_injection.runtime_discovery import observed_python_candidate_set


class NormModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.unrelated_name = torch.nn.RMSNorm(8)

    def forward(self, value):
        return self.unrelated_name(value) + 1


def reports(model):
    observations = []
    for _ in range(2):
        with ModuleFaultSession(model, '18.api_half', arm='baseline') as session:
            with torch.inference_mode():
                model(torch.arange(8.).reshape(1, 8))
        observations.append(session.report())
    return observations


class TypedBoundaryTests(unittest.TestCase):
    def target(self, boundary=None):
        return BoundaryTargetIdentity(process_id=os.getpid(), process_start_identity='test-process-start',
            worker_id='controlled-model-test', scope_kind='MODEL_LOCAL', rank=0, device_type='cpu', logical_device=None,
            model_instance_id='test-model-load', model_object_identity=10, module_object_identity=11,
            module_roles=('ATTENTION_NORM',), operator_api='torch.nn.RMSNorm.forward', tensor_signature='a'*64,
            boundary=boundary or PythonHookBoundary('actual.forward', 'b'*64, 0),
            provider_identity='18.api_half:1')

    def test_typed_boundaries_do_not_invent_opcache_fields(self):
        boundaries = [PythonHookBoundary('actual.forward', 'a'*64, 1),
            TensorAliasBoundary(1,2,1024,'real-allocation-life',17,(8,8),(16,1)),
            OperatorApiBoundary('a'*64,'b'*64,'actual-call-token','torch_dispatch'),
            KernelLaunchBoundary('actual_kernel',101,'a'*64,'b'*64,'launch-3',4096),
            CollectiveBoundary('HCCL','actual-group',(0,2,5),7,'SUM','a'*64)]
        for boundary in boundaries:
            with self.subTest(boundary=boundary.kind):
                target=self.target(boundary)
                serialized=asdict(target)
                self.assertEqual(serialized['boundary']['kind'],boundary.kind)
                self.assertNotIn('executor_identity',serialized)
                self.assertNotIn('cache_lifecycle',serialized)
                self.assertNotIn('native_build_fingerprint',serialized)
        with self.assertRaises(ValueError):
            self.target(CollectiveBoundary('HCCL','group',(2,5),7,'SUM','a'*64))
        with self.assertRaises(ValueError):
            KernelLaunchBoundary('actual_kernel',True,'a'*64,'b'*64,'launch-3',0)
        with self.assertRaises(ValueError):
            replace(self.target(),logical_device=0)
        with self.assertRaises(ValueError):
            replace(self.target(),boundary={'kind':'python_hook'})

    def test_native_target_retains_original_private_identity_gate(self):
        with self.assertRaises(ValueError):
            TargetIdentity(123,'worker',0,0,'model',1,2,(),'aclnnMatmul','op_exec_cache_run',
                           '', '', 'BF16', 'a'*64)
        with self.assertRaises(ValueError):
            Candidate('fake', EligibilityResult(Eligibility.SUPPORTED), SimpleNamespace(worker='fake'))

    def test_shared_coverage_keeps_denominators_and_scope_separate(self):
        target=self.target()
        candidates=CandidateSet((Candidate('real',EligibilityResult(Eligibility.SUPPORTED),target),),True)
        lease=TargetLease(target,'owner',1,'lease',32)
        call=ObservedCall(target,'actual-1','forward-1',None,Phase.PRE_FAULT_PROPAGATION)
        independent=IndependentObservation('independent_workload_observer',(call,),True)
        report=CoverageReport.reconcile((target,),candidates,(lease,),independent,
            (CallOutcome(call,lease,Outcome.HIT,'c'*64),))
        self.assertEqual((report.planned_targets_hit,report.observed_calls_hit),(1,1))
        self.assertEqual(report.request_mapping_unproven,1)
        self.assertFalse(candidates.report()['execution_authorized'])
        other=replace(target,process_start_identity='another-process-lifetime')
        self.assertNotEqual(target.worker,other.worker)


class RealPythonObservationTests(unittest.TestCase):
    def test_same_actual_module_and_calls_choose_unique_without_layer_argument(self):
        model=NormModel().eval()
        actual=reports(model)
        candidates=observed_python_candidate_set(model,'18.api_half',reports=actual)
        target,=candidates.select()
        self.assertEqual(target.module_object_identity,id(model.unrelated_name))
        self.assertEqual(target.scope_kind,'MODEL_LOCAL')
        self.assertEqual(target.device_type,'cpu')
        self.assertIsNone(target.logical_device)
        self.assertEqual(target.operator_api,'torch.nn.RMSNorm.forward')
        self.assertEqual(target.boundary.call_ordinal,0)
        again=observed_python_candidate_set(model,'18.api_half',reports=actual)
        self.assertEqual(again.eligible,candidates.eligible)
        self.assertFalse(candidates.report()['execution_authorized'])

    def test_failed_missing_or_foreign_observation_cannot_select(self):
        model=NormModel().eval(); actual=reports(model)
        for mutate in ('foreign_object','changed_callable','changed_geometry','missing_call','failed_root'):
            pair=copy.deepcopy(actual)
            if mutate=='foreign_object':pair[1]['events'][0]['module_object_id']+=1
            if mutate=='changed_callable':pair[1]['events'][0]['module_forward_identity']+=1
            if mutate=='changed_geometry':pair[1]['events'][0]['output_shape']=[2,8]
            if mutate=='missing_call':pair[1]['events']=[]
            if mutate=='failed_root':pair[1]['forwards'][0]['status']='failed'
            candidates=observed_python_candidate_set(model,'18.api_half',reports=pair)
            with self.subTest(mutate=mutate), self.assertRaises(UnsupportedCall):candidates.select()
        with self.assertRaisesRegex(UnsupportedCall,'WORKER_IDENTITY_MISMATCH'):
            observed_python_candidate_set(model,'18.api_half',reports=actual,
                worker_identity={'pid':os.getpid()+1})

    def test_observed_two_modules_remain_ambiguous(self):
        model=torch.nn.Sequential(torch.nn.RMSNorm(8),torch.nn.RMSNorm(8)).eval()
        candidates=observed_python_candidate_set(model,'18.api_half',reports=reports(model))
        self.assertEqual(len(candidates.eligible),2)
        with self.assertRaises(UnsupportedCall):candidates.select()

    def test_shared_logical_vocabulary_preserves_legacy_native_roles(self):
        model=torch.nn.Module()
        model.config=SimpleNamespace(model_type='unknown')
        model.any_name=torch.nn.Module()
        model.any_name.experts=torch.nn.ModuleList([torch.nn.Linear(8,8),torch.nn.Linear(8,8)])
        model.any_name.router=torch.nn.Linear(8,2)
        rows={row['path']:row for row in inspect_structure(model)['modules']}
        router=rows['any_name.router']
        self.assertEqual(router['canonical_roles'],['MLP_GATE'])
        self.assertEqual(router['logical_roles'],['MOE_ROUTER'])
        self.assertEqual(rows['any_name.experts.0']['logical_roles'],['MOE_EXPERT'])
        self.assertFalse(router['actual_operator_observed'])
        model.any_name.router=torch.nn.Linear(8,3)
        rows={row['path']:row for row in inspect_structure(model)['modules']}
        self.assertEqual(rows['any_name.router']['logical_roles'],[])


if __name__=='__main__':unittest.main()
