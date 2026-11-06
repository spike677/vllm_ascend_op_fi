"""CPU mocks exercise the proposed function and real main-finally AST.

No controller module import, subprocess execution, Docker access, SSH or NPU.
"""
import ast
import copy
import hashlib
import json
from pathlib import Path
import re
import subprocess
import traceback
import unittest

import controller_recovery_proposal as proposal

ID='a'*64
TOKEN='b'*32
NAME='dcat-f33-paged-cpu_mock'
IMAGE='sha256:'+'c'*64
TASK='fault33-paged-tiling-20261105'


class Clock:
    def __init__(self):self.now=0.0;self.sleeps=[]
    def monotonic(self):return self.now
    def time(self):return 1700000000.0+self.now
    def sleep(self,seconds):
        if seconds<=0:raise AssertionError('Unbounded/busy recovery polling')
        self.sleeps.append(seconds);self.now+=seconds


class Backend:
    def __init__(self,clock,events=(),running=False):
        self.clock=clock;self.events=list(events);self.calls=[];self.exists=False
        self.inspect_events=[]
        self.obj={'Id':ID,'Image':IMAGE,'Name':'/'+NAME,
            'Config':{'Labels':{'demoncat.controller_token':TOKEN,'demoncat.task':TASK}},
            'State':{'Running':running}}
    def __call__(self,argv,timeout=25):
        self.calls.append((list(argv),timeout))
        if argv[:2]==['docker','ps']:
            if '--filter' not in argv:return ID if self.exists else ''
            if 'label=demoncat.controller_token='+TOKEN not in argv or 'label=demoncat.task='+TASK not in argv:
                raise AssertionError('Recovery did not query the exact token/task')
            value=self.events.pop(0) if self.events else []
            if value=='STALL':
                self.clock.now+=timeout
                raise subprocess.TimeoutExpired(argv,timeout)
            if isinstance(value,Exception):raise value
            if value:self.exists=True
            return ' '.join(value)
        if argv[:2]==['docker','inspect']:
            if argv[2]!=ID:raise AssertionError('Unexpected inspection target')
            if self.inspect_events:
                event=self.inspect_events.pop(0)
                if event=='STALL':
                    self.clock.now+=timeout
                    raise subprocess.TimeoutExpired(argv,timeout)
                if event=='LATE_VALID':
                    self.clock.now+=2.1
                    return json.dumps([self.obj])
                if isinstance(event,Exception):raise event
                return event
            return json.dumps([self.obj])
        if argv[:2] in (['docker','stop'],['docker','rm']):
            if argv[-1]!=ID:raise AssertionError('Mutation attempted against another ID')
            self.exists=False
            return ''
        raise AssertionError('Unexpected command; all hardware/process calls are forbidden: '+repr(argv))
    @property
    def mutations(self):return [a for a,_ in self.calls if a[:2] in (['docker','stop'],['docker','rm'])]
    @property
    def queries(self):return [a for a,_ in self.calls if a[:2]==['docker','ps'] and '--filter' in a]


def require(value,message):
    if not value:raise RuntimeError(message)


def runtime(source,clock,backend,saved):
    tree=ast.parse(source)
    names={'verify_owned','recover_owned_id'}
    functions=[copy.deepcopy(n) for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names]
    main=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='main')
    cleanup=next(n for n in main.body if isinstance(n,ast.Try) and n.finalbody)
    args=ast.arguments(posonlyargs=[],args=[ast.arg(arg=n) for n in
        ('token','name','evidence','create_attempted','owned','child','lock','lock_acquired','result')],
        vararg=None,kwonlyargs=[],kw_defaults=[],kwarg=None,defaults=[])
    wrapper=ast.FunctionDef(name='exercise_cleanup',args=args,
        body=copy.deepcopy(cleanup.finalbody)+[ast.Return(value=ast.Name(id='result',ctx=ast.Load()))],decorator_list=[])
    module=ast.fix_missing_locations(ast.Module(body=functions+[wrapper],type_ignores=[]))
    def save(path,value):saved[Path(path).name]=json.loads(json.dumps(value))
    namespace={'re':re,'json':json,'subprocess':subprocess,'traceback':traceback,'time':clock,
        'run':backend,'require':require,'save':save,'IMAGE':IMAGE,'TASK':TASK,
        # Scaled virtual time keeps the same deadline math; no real sleeps.
        'UNCERTAIN_CREATE_RECOVERY_SECONDS':2.0,
        'UNCERTAIN_CREATE_RECOVERY_POLL_SECONDS':0.5,
        'UNCERTAIN_CREATE_RECOVERY_QUERY_SECONDS':0.75}
    exec(compile(module,'<proposed-controller-cleanup>','exec'),namespace)
    return namespace


class RecoveryChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.original_bytes=proposal.CONTROLLER.read_bytes()
        cls.before=cls.original_bytes.decode('utf-8')
        receipt=json.loads((proposal.HERE/'controller_recovery_proposal.json').read_text())
        actual=hashlib.sha256(cls.original_bytes).hexdigest()
        if actual==receipt['base_sha256']:
            cls.after=proposal.candidate(cls.before)
            if proposal.diff(cls.before,cls.after).encode()!= (proposal.HERE/'controller_recovery.patch').read_bytes():
                raise AssertionError('Tested source differs from the proposed patch')
        elif actual==receipt['candidate_sha256']:
            cls.after=cls.before
        else:raise AssertionError('Controller is neither the reviewed base nor the exact candidate')
        if hashlib.sha256(cls.after.encode()).hexdigest()!=receipt['candidate_sha256']:
            raise AssertionError('Candidate digest mismatch')

    def cleanup(self,events=(),running=False,owned=None,create=True,status='FAIL',mutate=None,inspect=(),source=None):
        clock=Clock();backend=Backend(clock,events,running)
        if owned:backend.exists=True
        if mutate:mutate(backend.obj)
        backend.inspect_events=list(inspect)
        saved={};ns=runtime(source or self.after,clock,backend,saved)
        result={'status':status,'cleanup_errors':[],'cases':[]}
        result=ns['exercise_cleanup'](TOKEN,NAME,Path('/mock/evidence'),create,owned,None,None,False,result)
        return result,saved,backend,clock

    def test_delayed_created_after_multiple_empty_snapshots_is_removed(self):
        result,saved,backend,clock=self.cleanup([[],[],[ID]])
        self.assertEqual(result['status'],'FAIL')  # Failed create must never become a successful capture.
        self.assertTrue(result['owned_container_removed'])
        self.assertEqual(backend.mutations,[['docker','rm',ID]])
        rec=saved['uncertain_create_recovery.json']
        self.assertEqual(rec['status'],'RECOVERED_EXACT_OWNER')
        self.assertEqual([x['status'] for x in rec['attempts']],['NO_MATCH_OBSERVED','NO_MATCH_OBSERVED','EXACT_OWNER_VERIFIED'])
        self.assertTrue(rec['ownership_verified']);self.assertFalse(rec['cleanup_claimed'])
        self.assertEqual(clock.now,1.0)

    def test_late_running_owner_uses_existing_stop_cleanup(self):
        result,saved,backend,_=self.cleanup([[],[ID]],running=True)
        self.assertTrue(result['owned_container_removed'])
        self.assertEqual(backend.mutations,[['docker','stop','--time','5',ID]])
        self.assertEqual(result['cleanup_errors'],[])

    def test_no_match_window_exhaustion_is_fail_cleanup_with_uncertainty_receipt(self):
        result,saved,backend,clock=self.cleanup()
        self.assertEqual(result['status'],'FAIL_CLEANUP')
        self.assertFalse(result['owned_container_removed'])
        self.assertIsNone(result['owned_container_id'])
        self.assertEqual(backend.mutations,[])
        rec=saved['uncertain_create_recovery.json']
        self.assertEqual(rec['status'],'UNRESOLVED_CREATE_OUTCOME')
        self.assertFalse(rec['ownership_verified'])
        self.assertFalse(rec['cleanup_claimed'])
        self.assertEqual(len(backend.queries),4)
        self.assertEqual(clock.now,2.0)
        self.assertEqual(saved['completion.json']['status'],'FAIL_CLEANUP')

    def test_query_timeouts_share_the_total_deadline(self):
        result,saved,backend,clock=self.cleanup(['STALL','STALL','STALL'])
        self.assertEqual(result['status'],'FAIL_CLEANUP')
        self.assertEqual(clock.now,2.0)
        self.assertTrue(all(timeout<=0.75 for argv,timeout in backend.calls if '--filter' in argv))
        self.assertEqual(backend.mutations,[])

    def test_inspect_error_is_retried_before_exact_owner_acceptance(self):
        result,saved,backend,_=self.cleanup([[ID],[ID]],inspect=[RuntimeError('daemon temporarily unavailable')])
        self.assertTrue(result['owned_container_removed'])
        self.assertEqual(saved['uncertain_create_recovery.json']['attempts'][0]['status'],'INSPECT_ERROR')
        self.assertEqual(backend.mutations,[['docker','rm',ID]])

    def test_query_error_is_retried_not_misreported_as_absence(self):
        result,saved,backend,_=self.cleanup([RuntimeError('daemon temporarily unavailable'),[ID]])
        self.assertTrue(result['owned_container_removed'])
        self.assertEqual(saved['uncertain_create_recovery.json']['attempts'][0]['status'],'QUERY_ERROR')

    def test_valid_inspect_returning_after_total_deadline_is_not_accepted(self):
        result,saved,backend,clock=self.cleanup([[ID]],inspect=['LATE_VALID'])
        self.assertEqual(result['status'],'FAIL_CLEANUP')
        self.assertFalse(result['owned_container_removed'])
        self.assertEqual(backend.mutations,[])
        self.assertEqual(saved['uncertain_create_recovery.json']['status'],'UNRESOLVED_CREATE_OUTCOME')
        self.assertEqual(saved['uncertain_create_recovery.json']['attempts'][0]['status'],'VERIFIED_AFTER_DEADLINE')

    def test_malformed_inspect_does_not_bypass_exact_owner_verification(self):
        result,saved,backend,_=self.cleanup([[ID],[ID]],inspect=['not-json'])
        self.assertTrue(result['owned_container_removed'])
        self.assertEqual(saved['uncertain_create_recovery.json']['attempts'][0]['status'],'INSPECT_ERROR')
        self.assertTrue(saved['uncertain_create_recovery.json']['ownership_verified'])

    def test_exact_identity_mismatch_never_reaches_mutation(self):
        mutations=[lambda o:o.update(Id='d'*64),lambda o:o.update(Image='sha256:'+'d'*64),
            lambda o:o.update(Name='/another-container'),
            lambda o:o['Config']['Labels'].update({'demoncat.controller_token':'d'*32}),
            lambda o:o['Config']['Labels'].update({'demoncat.task':'another-task'})]
        for mutate in mutations:
            with self.subTest(mutate=mutate):
                result,saved,backend,_=self.cleanup([[ID]],mutate=mutate)
                self.assertEqual(result['status'],'FAIL_CLEANUP')
                self.assertFalse(result['owned_container_removed'])
                self.assertEqual(backend.mutations,[])
                self.assertEqual(saved['uncertain_create_recovery.json']['status'],'OWNERSHIP_REJECTED')

    def test_ambiguous_or_invalid_candidate_ids_fail_closed(self):
        for ids in ([ID,'d'*64],['short-id']):
            with self.subTest(ids=ids):
                result,saved,backend,_=self.cleanup([ids])
                self.assertEqual(result['status'],'FAIL_CLEANUP')
                self.assertEqual(backend.mutations,[])
                self.assertFalse(any(argv[:2]==['docker','inspect'] for argv,_ in backend.calls))

    def test_ambiguous_inspect_fails_closed(self):
        result,saved,backend,_=self.cleanup([[ID]],inspect=['[]'])
        self.assertEqual(result['status'],'FAIL_CLEANUP')
        self.assertEqual(backend.mutations,[])
        self.assertEqual(saved['uncertain_create_recovery.json']['status'],'AMBIGUOUS_INSPECT_RESULT')

    def test_normal_successful_create_path_is_unchanged(self):
        result,saved,backend,clock=self.cleanup(owned=ID,running=True,status='CAPTURE_COMPLETED')
        self.assertEqual(result['status'],'CAPTURE_COMPLETED')
        self.assertTrue(result['owned_container_removed'])
        self.assertEqual(backend.queries,[])
        self.assertNotIn('uncertain_create_recovery.json',saved)
        self.assertEqual(clock.sleeps,[])
        self.assertEqual(backend.mutations,[['docker','stop','--time','5',ID]])

    def test_failure_before_create_does_not_start_recovery(self):
        result,saved,backend,_=self.cleanup(create=False)
        self.assertEqual(result['status'],'FAIL')
        self.assertEqual(backend.calls,[])
        self.assertNotIn('uncertain_create_recovery.json',saved)

    def test_old_once_empty_branch_demonstrates_the_actual_regression(self):
        if 'def recover_owned_id(token, name):' not in self.before:
            self.skipTest('Base was already replaced by the exact reviewed candidate')
        result,saved,backend,clock=self.cleanup([[],[ID]],source=self.before)
        self.assertEqual(result['status'],'FAIL')
        self.assertEqual(result['cleanup_errors'],[])
        self.assertEqual(len(backend.queries),1)
        self.assertEqual(clock.sleeps,[])
        self.assertIsNone(saved['uncertain_create_recovery.json']['container_id'])
        self.assertTrue(saved['uncertain_create_recovery.json']['token_query_completed'])


if __name__=='__main__':
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(RecoveryChecks))
    unchanged=proposal.CONTROLLER.read_bytes()==RecoveryChecks.original_bytes
    report={'schema':'fault33.controller_recovery.cpu_mock.v1','status':'PASS_CPU_MOCKS' if result.wasSuccessful() and unchanged else 'FAIL_CPU_MOCKS',
        'tests':result.testsRun,'failures':len(result.failures),'errors':len(result.errors),'skipped':len(result.skipped),
        'controller_unchanged':unchanged,'controller_sha256':hashlib.sha256(proposal.CONTROLLER.read_bytes()).hexdigest(),
        'patch_sha256':hashlib.sha256((proposal.HERE/'controller_recovery.patch').read_bytes()).hexdigest(),
        'test_scope':'Exact proposed recovery function and main-finally AST with fake clock and Docker backend',
        'docker_or_hardware_commands_executed':False,'ssh_executed':False,'npu_executed':False,'patch_applied':False}
    (proposal.HERE/'controller_recovery_cpu_report.json').write_bytes((json.dumps(report,indent=2)+'\n').encode('utf-8'))
    print(json.dumps(report))
    raise SystemExit(0 if report['status']=='PASS_CPU_MOCKS' else 1)
