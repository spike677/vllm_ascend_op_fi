"""Bounded resumable case runner; no candidate search or foreign mutations."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

import controller
from case_factory import create

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'analysis'))
from compare_statistics import compare
from accept_statistics import accept
from audit_runtime import audit
from review_formal import review,values


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def write(p,value):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x',encoding='utf-8') as f:json.dump(value,f,indent=2,allow_nan=False);f.write('\n')


class ResourceBlocked(RuntimeError):pass


class Search:
    def __init__(self,max_idle_seconds):
        self.contract=json.loads((ROOT/'preregister.json').read_bytes())
        self.state_file=ROOT/'search_state.json'
        self.state=json.loads(self.state_file.read_bytes()) if self.state_file.exists() else {
            'schema':'fault28.visible.search_state.v1','status':'PREPARED','card':None,
            'preregister_sha256':sha(ROOT/'preregister.json'),'cases':{},'candidates':{},'finalists':{}}
        assert self.state['preregister_sha256']==sha(ROOT/'preregister.json')
        self.max_idle_seconds=max_idle_seconds;self.last_wait_message=None

    def persist(self):
        tmp=self.state_file.with_suffix('.json.tmp')
        tmp.write_text(json.dumps(self.state,indent=2,allow_nan=False)+'\n',encoding='utf-8')
        tmp.replace(self.state_file)

    def verify_freeze(self):
        for name,digest in self.contract['source_files'].items():
            assert sha(ROOT/name)==digest,'Pre-registered execution/comparison source changed: '+name
        assert sha(ROOT/'vendor_visible_a1/vendor_manifest.json')==self.contract['vendor_manifest_sha256']
        assert sha(ROOT/'private/input_bank/manifest.json')==self.contract['input_bank_sha256']
        fixture=ROOT/'private/analytic_fixture/qkv_manifest.json'
        assert sha(fixture)==self.contract['analytic_fixture_sha256']
        for kind in ('query','key','value'):
            row=json.loads(fixture.read_bytes())[kind]
            assert sha(fixture.parent/row['file'])==row['sha256']
        bank=json.loads((ROOT/'private/input_bank/manifest.json').read_bytes())
        for row in bank['inputs']:
            assert sha(ROOT/'private/input_bank'/row['file'])==row['sha256'],'Frozen input content changed'

    def idle_card(self):
        began=time.monotonic()
        while True:
            self.verify_freeze()
            cards=[self.state['card']] if self.state['card'] is not None else [2,5]
            status={}
            for card in cards:
                try:
                    report=controller.resources(card,strict=False,require_idle=True)
                    status[str(card)]={'eligible':report['eligible'],
                        'foreign_pids':[p['host_pid'] for p in report['selected_card_foreign_npu_processes']],
                        'table_idle':report['selected_card_table_idle']}
                    if report['eligible']:
                        self.state.update(card=card,status='RUNNING');self.persist();return card
                except Exception as e:status[str(card)]={'error':str(e)}
            self.state['status']='WAITING_FOR_IDLE_DEVICE';self.state['last_gate']=status;self.persist()
            signature=json.dumps({k:v.get('eligible',False) for k,v in status.items()},sort_keys=True)
            if signature!=self.last_wait_message:
                print(json.dumps({'status':'WAITING_FOR_IDLE_DEVICE','gate':status,'npu_calls_while_blocked':0}),flush=True)
                self.last_wait_message=signature
            if time.monotonic()-began>=self.max_idle_seconds:
                raise ResourceBlocked('Idle-device wait limit reached; no external task was stopped')
            time.sleep(20)

    def run_case(self,path):
        case=json.loads(Path(path).read_bytes());name=case['case_id']
        previous=self.state['cases'].get(name)
        if previous and previous['status']=='CAPTURE_COMPLETED':
            folder=ROOT/previous['folder']
            assert sha(folder/'result.json')==previous['result_sha256']
            return folder
        if previous and previous['status']=='PROBE_FAILED':return None
        for retry in range(3):
            card=self.idle_card();self.verify_freeze()
            number=1
            while (ROOT/'outputs'/f'v_{name}_{number}').exists():number+=1
            attempt=f'v_{name}_{number}';assert len(attempt)<=40
            full=case['capture_scope']=='model'
            args=[sys.executable,'-B',str(ROOT/'controller.py'),'--card',str(card),'--attempt',attempt,
                  '--max-seconds','2100' if full else '900','--case-seconds','1800' if full else '600',
                  '--probe','eager_probe.py','--case-file',str(path),'--capture','native-logs','--max-log-mib','4096']
            print(json.dumps({'status':'STARTING_CASE','case_id':name,'attempt':attempt,'physical_card':card,'scope':case['capture_scope']}),flush=True)
            rc=subprocess.run(args,cwd=ROOT).returncode
            completion=ROOT/'outputs'/attempt/'completion.json'
            data=json.loads(completion.read_bytes()) if completion.exists() else {'status':'MISSING_COMPLETION'}
            folder=completion.parent/name
            if rc==0 and data['status']=='CAPTURE_COMPLETED':
                self.state['cases'][name]={'status':'CAPTURE_COMPLETED','folder':folder.relative_to(ROOT).as_posix(),
                    'result_sha256':sha(folder/'result.json'),'runtime_audit':'DEFERRED_UNTIL_AFTER_STATISTICAL_LOCALIZATION'}
                self.persist();return folder
            text=data.get('traceback','')
            last_rc=(data.get('cases') or [{}])[-1].get('rc')
            incomplete=not (folder/'result.json').is_file() or last_rc in (137,143)
            if incomplete or 'Resource gate rejected' in text or 'BlockingIOError' in text or 'missing-PID' in text:
                self.state['cases'][name]={'status':'RESOURCE_INTERRUPTED','attempt':attempt}
                self.persist();continue
            self.state['cases'][name]={'status':'PROBE_FAILED','attempt':attempt,
                'completion':completion.relative_to(ROOT).as_posix(),'error':text[-2000:]}
            self.persist();return None
        raise ResourceBlocked('Three device-gate interruptions; retained attempts, no foreign cleanup')

    def harvest(self,input_id):
        path=create('h_'+input_id,input_id,'harvest','coupled_m128',capture=False)
        folder=self.run_case(path)
        if folder is None:raise RuntimeError('Required native-QKV harvest failed for '+input_id)
        return folder/'qkv_manifest.json'

