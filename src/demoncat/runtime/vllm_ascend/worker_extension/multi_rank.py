"""Fail-closed group lifecycle over existing per-rank Worker leases.

No model/scheduler execution, new workers, rank user options, deployment grant,
or NPU qualification. TP2 providers must independently implement group guards.
"""
from copy import deepcopy
import hashlib
import json
import threading
import uuid
from .config import FaultConfig
from .host import HostAdapter, WorkerFaultClient, HostContractError, _GroupIdentity
from fault_injection.request_plan import request_plan
from fault_injection.continuous_contract import continuous_request_plan, MECHANISMS

RESTORE = {'recover', 'abort', 'disable', 'unregister', 'detach', 'clean'}
SUPPORTED = {'18.api_half', '39.output_proxy', '22.wrong_row', 'quant.batch_shared_scale', '41.tail_column'}


class MultiRankTaskClient:
    def __init__(self, executor, config, *, prepare_binding, commit_roles=None, timeout=30,
                 continuous=False, world_size=2, batch_size=None):
        # Trusted host constructor only, never a public FaultConfig bypass.
        if type(continuous) is not bool or type(world_size) is not int or world_size < 1:
            raise ValueError('INVALID_GROUP_POLICY')
        if not continuous and world_size != 2:
            raise ValueError('LEGACY_GROUP_IS_TP2_ONLY')
        if not isinstance(config, FaultConfig) or config.mechanism not in (MECHANISMS if continuous else SUPPORTED):
            raise ValueError('MULTI_RANK_FIRST_BATCH_ONLY')
        self.continuous=continuous; self.world_size=world_size
        if not callable(prepare_binding):raise TypeError('TRUSTED_GROUP_BINDING_REQUIRED')
        if commit_roles is not None and not callable(commit_roles):
            raise TypeError('TRUSTED_ROLE_COMMIT_CALLBACK_REQUIRED')
        self.commit_roles=commit_roles;self._role_decision=None
        self.host=HostAdapter(executor, timeout=timeout)
        self.config=config;self.binding=prepare_binding
        self.identity=_GroupIdentity('worker-group-'+uuid.uuid4().hex,
                                     'worker-group-client-'+uuid.uuid4().hex, 1)
        self.members=[];self.clients={};self.failed=None;self.history=[]
        self.started=False;self._recovering=False;self._last_request_id=None;self._normal_request_id=None;self._lock=threading.RLock()
        self.plan=(continuous_request_plan(config.mechanism, batch_size=batch_size if batch_size is not None else 1)
                   if continuous else request_plan(config.mechanism))

    def _census(self, *, allow_quarantined=False):
        rows=self.host.discover_workers()
        if not allow_quarantined and any(row.get('quarantined') is not False for row in rows):
            raise HostContractError('GROUP_WORKER_RECOVERY_REQUIRED')
        members=sorted((row['worker'] for row in rows),key=lambda x:x['rank'])
        if [row['rank'] for row in members]!=list(range(self.world_size)):
            raise HostContractError('EXACT_GROUP_MEMBERS_REQUIRED' if self.continuous else 'EXACT_TP2_MEMBERS_REQUIRED')
        if self.members and self.members!=members:
            raise HostContractError('GROUP_WORKER_OR_MODEL_CHANGED')
        return members

    def contract(self):
        return dict(owner=self.identity.owner,group_generation=self.identity.generation,
            members=deepcopy(self.members),members_sha256=hashlib.sha256(
                json.dumps(self.members,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest(),
            mechanism=self.config.mechanism,tp=self.world_size,
            injection_policy='ALL_RANK_EVERY_ELIGIBLE_CALL' if self.continuous else 'ONE_AUTOMATIC_RANK',
            qualification_granted=False)

    def _init_clients(self):
        self.members=self._census()
        for member in self.members:
            def bind(actual,config):
                return self.binding(actual,config,self.contract())
            host=HostAdapter(self.host.executor,select_worker=lambda config,rows,m=member:deepcopy(m),
                             prepare_binding=bind,timeout=self.host.timeout)
            self.clients[member['rank']]=WorkerFaultClient(host,self.config,_group_identity=self.identity)

    def _result(self,action,rows,status):
        states={rank:row.get('response',{}).get('provider_state') for rank,row in rows.items()}
        phase=None;complete=False
        if any(row.get('response',{}).get('recovery_required') is True for row in rows.values()):
            self.failed=self.failed or 'RANK_REQUIRES_RECOVERY'
        observed=[state for state in states.values() if isinstance(state,dict)]
        if len(observed)==self.world_size:
            phases=[s.get('next_phase',(s.get('serving') or {}).get('next_phase')) for s in observed]
            plans=[s.get('request_plan') for s in observed]
            if (all(p==self.plan for p in plans) and len(set(phases))==1
                    and phases[0] in [p['phase'] for p in self.plan['phases']]+['complete']):
                phase=phases[0];complete=phase=='complete' and all(s.get('complete') is True for s in observed)
            elif action in ('query','advance','advance_batch') and status in ('QUERY_OK','ACTION_COMPLETED'):
                self.failed=self.failed or 'GROUP_PHASE_OR_REQUEST_PLAN_MISMATCH'
                status='GROUP_ACTION_FAILED'
        if len(observed)!=self.world_size and action in ('query','advance','advance_batch') and status in ('QUERY_OK','ACTION_COMPLETED'):
            self.failed=self.failed or 'ALL_RANK_PROVIDER_STATE_REQUIRED'
            status='GROUP_ACTION_FAILED'
        result=dict(status=status,group=self.contract(),action=action,
            response=dict(provider_state=dict(request_plan=deepcopy(self.plan),next_phase=phase,
                complete=complete and self.failed is None,rank_states=deepcopy(states),
                failure=self.failed,qualification_granted=False)),ranks=deepcopy(rows))
        self.history.append(deepcopy(result));return result

    def _step(self,action,*args):
        with self._lock:
            if self.failed and action not in RESTORE|{'query'} and not (action in ('advance','advance_batch') and self._recovering):
                return self._result(action,{},'GROUP_RECOVERY_REQUIRED')
            rows={}
            try:
                if action=='prepare':
                    if self.started:raise HostContractError('GROUP_PREPARE_NOT_RETRIED')
                    self.started=True;self._init_clients()
                if not self.clients:raise HostContractError('GROUP_PREPARE_REQUIRED')
                if action not in RESTORE:self._census(allow_quarantined=action=='query' or self._recovering)
                expected='QUERY_OK' if action=='query' else 'DETACHED' if action=='detach' else 'ACTION_COMPLETED'
                for rank,client in self.clients.items():
                    try:
                        if action=='recover' and client.snapshot()['ticket'] is None:
                            row=client.detach() if client.snapshot()['attachment'] else {'status':'NO_OWNED_ATTACHMENT'}
                            good=row['status'] in ('DETACHED','NO_OWNED_ATTACHMENT')
                        else:
                            row=getattr(client,action)(*args);good=row['status']==expected
                        rows[rank]=row
                        if not good:
                            self.failed=self.failed or 'RANK_'+str(rank)+'_'+action.upper()+'_FAILED'
                            if action not in RESTORE:break
                    except Exception as error:
                        rows[rank]=dict(status='RPC_UNCERTAIN',error=repr(error))
                        self.failed=self.failed or 'RANK_'+str(rank)+'_'+action.upper()+'_UNCERTAIN'
                        if action not in RESTORE:break
                success=len(rows)==self.world_size and all(row['status'] in (expected,'NO_OWNED_ATTACHMENT','DETACHED')
                    if action=='recover' else row['status']==expected for row in rows.values())
                return self._result(action,rows,expected if success else 'GROUP_ACTION_FAILED')
            except Exception as error:
                self.failed=self.failed or str(error)
                return self._result(action,rows,'GROUP_ACTION_FAILED')

    def _guard_fault_roles(self,state, *, require_committed=True):
        ranks=state['response']['provider_state']['rank_states']
        if len(ranks)!=self.world_size:raise HostContractError('ALL_RANK_NORMAL_EVIDENCE_REQUIRED')
        candidates=[];fingerprints=[]
        for rank,row in ranks.items():
            proof=row.get('multi_rank') or {}
            if (proof.get('owner')!=self.identity.owner or proof.get('group_generation')!=1
                    or proof.get('actual_normal_request_observed') is not True
                    or proof.get('normal_request_id')!=self._normal_request_id
                    or proof.get('members_sha256')!=self.contract()['members_sha256']
                    or proof.get('worker')!=self.members[rank]):
                raise HostContractError('GROUP_NATIVE_ROLE_ADAPTER_NOT_READY')
            fingerprint=proof.get('request_fingerprint')
            if (not isinstance(fingerprint,str) or len(fingerprint)!=64
                    or any(c not in '0123456789abcdef' for c in fingerprint)):
                raise HostContractError('ACTUAL_REQUEST_FINGERPRINT_REQUIRED')
            fingerprints.append(fingerprint)
            if proof.get('eligible_target') is True:candidates.append(rank)
        if len(set(fingerprints))!=1:
            raise HostContractError('RANK_REQUEST_INPUTS_NOT_ALIGNED')
        if self.continuous:
            # No baseline candidate list constrains later dynamic EP routing.
            # Every rank arms its own sites; absent local calls generate no writes.
            decision=dict(group=self.contract(),selected_rank=None,normal_request_id=self._normal_request_id,
                request_fingerprint=fingerprints[0],roles={rank:'injector' for rank in ranks})
            if require_committed:
                for row in ranks.values():
                    proof=row['multi_rank']
                    if (proof.get('injection_policy')!='ALL_RANK_EVERY_ELIGIBLE_CALL'
                            or proof.get('role')!='injector' or proof.get('selected_rank') is not None):
                        raise HostContractError('CONTINUOUS_ROLE_READBACK_REQUIRED')
            return decision
        selected=0 if self.config.mechanism=='18.api_half' else min(candidates,default=None)
        if selected not in candidates:raise HostContractError('NO_ACTUAL_AUTOMATIC_TARGET_RANK')
        decision=dict(group=self.contract(),selected_rank=selected,normal_request_id=self._normal_request_id,
                      request_fingerprint=fingerprints[0],roles={rank:('injector' if rank==selected else 'guard') for rank in ranks})
        if require_committed:
            for rank,row in ranks.items():
                proof=row['multi_rank']
                if proof.get('selected_rank')!=selected or proof.get('role')!=decision['roles'][rank]:
                    raise HostContractError('ALL_RANK_ROLE_AGREEMENT_REQUIRED')
        return decision

    def advance(self,request_id):
        if self.plan['unit']!='request':raise ValueError('BATCH_REQUIRES_ADVANCE_BATCH')
        return self._advance(request_id,'advance')

    def advance_batch(self,request_ids):
        count=self.plan['phases'][0]['request_count']
        if (self.plan['unit']!='batch' or not isinstance(request_ids,list) or len(request_ids)!=count
                or any(not isinstance(x,str) or not x.strip() for x in request_ids)
                or len(set(request_ids))!=count):raise ValueError('PLANNED_ACTUAL_INTERNAL_IDS_REQUIRED' if self.continuous else 'EIGHT_ACTUAL_INTERNAL_IDS_REQUIRED')
        return self._advance(list(request_ids),'advance_batch')

    def _advance(self,request_id,action):
        with self._lock:
            if action=='advance' and (not isinstance(request_id,str) or not request_id.strip()):
                raise ValueError('ACTUAL_INTERNAL_REQUEST_ID_REQUIRED')
            observed=self.query()
            if observed['status']!='QUERY_OK':return observed
            phase=observed['response']['provider_state']['next_phase']
            if self.failed and (not self._recovering or phase not in ('off','health')):
                return self._result('advance',{},'GROUP_RECOVERY_REQUIRED')
            if phase=='complete':return self._result('advance',{},'GROUP_PHASES_EXHAUSTED')
            if phase not in [row['phase'] for row in self.plan['phases']]:
                self.failed=self.failed or 'GROUP_PHASE_UNKNOWN'
                return self._result('advance',{},'GROUP_ACTION_FAILED')
            if phase=='fault':
                try:
                    decision=self._guard_fault_roles(observed,require_committed=False)
                    if self._role_decision is not None and self._role_decision!=decision:
                        raise HostContractError('ROLE_DECISION_CHANGED')
                    if self._role_decision is None:
                        # Latch before callback: a lost acknowledgement cannot replay selection.
                        self._role_decision=deepcopy(decision)
                        if self.commit_roles is not None:
                            self.commit_roles(deepcopy(decision))
                            observed=self.query()
                            if observed['status']!='QUERY_OK':raise HostContractError('ROLE_COMMIT_READBACK_FAILED')
                    if self._guard_fault_roles(observed)!=decision:
                        raise HostContractError('ROLE_COMMIT_EVIDENCE_CHANGED')
                except Exception as error:
                    self.failed=self.failed or str(error)
                    return self._result('advance',{},'GROUP_ACTION_FAILED')
            result=self._step(action,request_id)
            if result['status']=='ACTION_COMPLETED':
                self._last_request_id=deepcopy(request_id)
                if phase=='normal':self._normal_request_id=deepcopy(request_id)
            return result

    def snapshot(self):
        with self._lock:
            ranks={rank:client.snapshot() for rank,client in self.clients.items()}
            return dict(group=self.contract(),failure=self.failed,ranks=ranks,
                local_generations={rank:(row['ticket'] or {}).get('generation') for rank,row in ranks.items()},
                ownership_retained=any(row['ownership_retained'] for row in ranks.values()),
                history=deepcopy(self.history),role_decision=deepcopy(self._role_decision),qualification_granted=False)

    def prepare(self):return self._step('prepare')
    def register(self):return self._step('register')
    def check(self):return self._step('check')
    def enable(self):return self._step('enable')
    def query(self):return self._step('query')
    def finish(self):return self._step('finish')
    def recover(self):
        result=self._step('recover')
        self._recovering=result['status']=='ACTION_COMPLETED'
        return result
    def disable(self):return self._step('disable')
    def abort(self):return self._step('abort')
    def unregister(self):return self._step('unregister')
    def detach(self):return self._step('detach')
    def clean(self):return self._step('clean')
