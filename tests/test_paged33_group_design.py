"""CPU coordinator design contracts. Synthetic outcomes are never NPU proof."""
from dataclasses import replace
import os
import unittest
import uuid

from fault_injection.core import UnsupportedCall
from fault_injection.coverage import (BoundaryTargetIdentity,Candidate,CandidateSet,CoverageCapability,
    Eligibility,EligibilityResult,ObservedCall,OperatorApiBoundary,Outcome,Phase,QualifiedTargetSet,
    RecoveryReport,TargetLease,TargetLosslessProof)
from fault_injection.coverage_session import (CoverageScope,CoverageSession,GroupRecoveryHooks,
    IndependentCallObserver,SessionBudget,WorkerHealthProof)

A,B,C='a'*64,'b'*64,'c'*64


class CpuGroup:
    def __init__(self):
        self.worker,self.model=object(),object();self.modules=[object() for _ in range(28)]
        self.owner=uuid.uuid4().hex;self.start='cpu-design:'+uuid.uuid4().hex
        self.decode=tuple(self.target(i,'decode') for i in range(28))
        self.prefill=tuple(self.target(i,'prefill') for i in range(28))
        self.candidates=CandidateSet(tuple(
            [Candidate('decode:'+str(i),EligibilityResult(Eligibility.SUPPORTED),target) for i,target in enumerate(self.decode)]
            +[Candidate('prefill:'+str(i),EligibilityResult(Eligibility.UNSUPPORTED,('CPU_Q1_SCOPE_EXCLUDES_PREFILL',)),target)
              for i,target in enumerate(self.prefill)]),True)
        self.scope=CoverageScope('33.paged_decode',A,B,C,evidence_kind='cpu_contract')
        self.drain_ok=True;self.health_ok=True

    def target(self,index,role):
        return BoundaryTargetIdentity(os.getpid(),self.start,str(id(self.worker)),'INITIALIZED_WORKER',0,'cpu',None,
            str(id(self.model)),id(self.model),id(self.modules[index]),('attention',),'cpu_fixture_original',
            'CPU_ONLY:'+role,OperatorApiBoundary(A,B,'cpu-boundary:'+role+':'+str(index),'torch_dispatch'),
            '33:cpu-group-design',(('implementation_sha256',C),))

    def session(self,generation=1,trigger='once',proof=None,candidates=None,scope=None):
        candidates=candidates or self.candidates
        observer=IndependentCallObserver(source='independent_workload_observer',observer_id=uuid.uuid4().hex,
            implementation_sha256=C,workload_sha256=A,max_calls=4096)
        observer.seal_discovery(candidates,observed_targets=(*self.prefill,*self.decode),artifact_sha256=A,
            expected_workers=(self.decode[0].worker,),observed_workers=(self.decode[0].worker,))
        leases=tuple(TargetLease(t,self.owner,generation,'cpu:'+str(i),0,0) for i,t in enumerate(self.decode))
        proofs=tuple(TargetLosslessProof(lease,'cpu_normal_control',A,(B,),(B,),C,C,1,True,True) for lease in leases)
        cap=CoverageCapability(single_target=True,multi_target=True,every_match=True,max_concurrency=1,
            max_resources=0,max_host_bytes=0,group_recovery=True,recovery_scope='target_set',
            stop_policy='bounded_session',lossless_modes=('cpu_normal_control',))
        group=QualifiedTargetSet(candidates,leases,proofs,cap,selection='all_eligible',trigger=trigger)
        hooks=GroupRecoveryHooks(lambda lease:None,lambda lease,remaining:self.drain_ok,
            lambda lease:RecoveryReport(lease,True,True,True,proof_sha256=A,off_output_sha256=C),
            lambda worker,owner,gen:WorkerHealthProof(worker,owner,gen,A,C,C,0,0,0,0,0 if self.health_ok else 1))
        return CoverageSession(group=group,scope=scope or self.scope,observer=observer,
            budget=SessionBudget(4096,30),recovery=hooks,c2_proof=proof)

    def request(self,session,index,*,off=False,hold=False):
        held=[];admitted=[];reasons=[]
        for role,targets in (('prefill',self.prefill),('decode',self.decode)):
            for ordinal,target in enumerate(targets):
                phase=(Phase.OFF_RECOVERY if off else Phase.POST_FIRST_ANOMALY
                       if role=='decode' and admitted else Phase.PRE_FAULT_PROPAGATION)
                ticket=session.observer.observe(ObservedCall(target,uuid.uuid4().hex,'cpu-forward:'+str(index)+':'+role,
                    ('cpu-request:'+str(index),),phase),eligible=role=='decode')
                decision=session.begin(ticket);reasons.append(decision.reason)
                if decision.admitted:
                    admitted.append(ordinal)
                    if hold:held.append(decision);continue
                self.finish(session,decision)
        # This is the deliberate request-return barrier, not an API-return
        # completion. Actual effects and tensor binding are outside this fixture.
        for decision in held:self.finish(session,decision)
        return admitted,reasons

    @staticmethod
    def finish(session,decision):
        session.finish(decision,Outcome.HIT if decision.admitted else Outcome.PASSTHROUGH,
            execution_sha256=A,effect_sha256=B,native_applied=decision.admitted,original_calls=1)

    def c2(self):
        session=self.session()
        for index in range(28):
            admitted,_=self.request(session,2*index,hold=True)
            if admitted!=[index]:raise AssertionError(admitted)
            self.request(session,2*index+1,off=True)
        session.observer.seal(artifact_sha256=C)
        session.disable()
        return session


class Paged33GroupDesignContracts(unittest.TestCase):
    def test_c2_keeps_all_56_calls_and_holds_admission_until_request_return(self):
        f=CpuGroup();session=f.c2();report=session.report()
        self.assertTrue(report['contract_satisfied'],report['rejection_reasons'])
        self.assertEqual((report['planned_targets'],report['hits'],report['observed_calls']), (28,28,3136))
        self.assertEqual(report['observed_eligible_calls'],28*28)
        self.assertEqual(sum(row['decision']['reason']=='CONCURRENCY_BUDGET_EXHAUSTED' for row in report['calls']),sum(range(28)))
        self.assertFalse(report['public_qualification_granted'] or report['real_worker_evidence'])
        self.assertEqual(len(session.c2_proof().target_ids),28)

    def test_finishing_at_api_return_does_not_provide_isolated_c2(self):
        f=CpuGroup();session=f.session()
        admitted,_=f.request(session,0,hold=False)
        self.assertEqual(admitted,list(range(28)))  # Why the provider needs the request barrier.
        session.observer.seal(artifact_sha256=C);session.disable()

    def test_unresolved_target_is_not_dropped_from_the_denominator(self):
        f=CpuGroup();rows=list(f.candidates.candidates)
        rows[-1]=replace(rows[-1],eligibility=EligibilityResult(Eligibility.UNVERIFIED,('CPU_MISSING_ACTUAL_BOUNDARY',)))
        with self.assertRaisesRegex(UnsupportedCall,'UNRESOLVED_CANDIDATE_DENOMINATOR'):
            f.session(candidates=CandidateSet(tuple(rows),True))

    def test_failed_drain_keeps_all_leases_and_cannot_issue_c2(self):
        f=CpuGroup();session=f.session()
        for index in range(28):f.request(session,index,hold=True)
        session.observer.seal(artifact_sha256=C);f.drain_ok=False
        failed=session.disable(timeout_seconds=.1)
        self.assertFalse(failed['complete']);self.assertEqual(len(failed['held_leases']),28)
        with self.assertRaisesRegex(UnsupportedCall,'C2_CONTRACT_NOT_COMPLETE'):session.c2_proof()
        f.drain_ok=True;self.assertTrue(session.disable()['complete'])
        # The provider must archive this original receipt. The coordinator's
        # current recovery errors are intentionally retired after a good retry.
        self.assertTrue(failed['errors']);self.assertFalse(session.recovery_report()['errors'])

    def test_c3_cannot_change_the_build_scope_owner_or_generation(self):
        f=CpuGroup();proof=f.c2().c2_proof()
        for kwargs in (dict(generation=1),dict(scope=replace(f.scope,implementation_sha256='d'*64))):
            values=dict(generation=2,trigger='every_match',proof=proof);values.update(kwargs)
            with self.assertRaisesRegex(UnsupportedCall,'C2_SCOPE_OR_LIFETIME_MISMATCH'):f.session(**values)
        f.owner=uuid.uuid4().hex
        with self.assertRaisesRegex(UnsupportedCall,'C2_SCOPE_OR_LIFETIME_MISMATCH'):
            f.session(2,'every_match',proof)

    def test_only_original_c2_object_can_open_two_every_match_rounds(self):
        f=CpuGroup();c2=f.c2();proof=c2.c2_proof()
        for fake in (proof.report(),replace(proof)):
            with self.assertRaisesRegex(UnsupportedCall,'MATCHING_C2_PROOF_REQUIRED'):
                f.session(2,'every_match',fake)
        c3=f.session(2,'every_match',proof)
        for index in range(2):self.assertEqual(f.request(c3,index)[0],list(range(28)))
        f.request(c3,2,off=True);c3.observer.seal(artifact_sha256=C);c3.disable()
        report=c3.report();self.assertTrue(report['contract_satisfied'],report['rejection_reasons'])
        self.assertEqual((report['hits'],report['observed_eligible_calls'],report['observed_calls']), (56,56,168))
        issued=c2.report()['live_c2_proof']['issued'][0]
        used=report['live_c2_proof']['consumed']
        self.assertEqual(issued['object_binding'],used['object_binding'])
        self.assertEqual(used['object_binding']['python_object_id'],id(proof))
        self.assertEqual(used['consumer']['generation'],2)
        self.assertFalse(report['public_qualification_granted'] or report['real_worker_evidence'])


if __name__=='__main__':unittest.main()
