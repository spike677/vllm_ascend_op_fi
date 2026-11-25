"""Row12 acceptance harness derived from frozen v2 example; separate source."""
import hashlib,json,os,subprocess,sys,time,traceback,urllib.request
from pathlib import Path

def main(extension='external_persistent_product_probe.PersistentExampleExtension',variant='weight_scale_half'):
    from product_http_probe import HTTPExecutor
    from demoncat.runtime.vllm_ascend.worker_extension.persistent_product_host import PersistentProductClient
    from demoncat.runtime.vllm_ascend.worker_extension.persistent_ingress import PersistentIngress,IngressRefused
    from demoncat.runtime.vllm_ascend.worker_extension.external_persistent_runtime import PersistentFaultConfig
    from demoncat.runtime.vllm_ascend.worker_extension.product_errors import product_control_reply
    root=Path('/evidence');server=None;report=dict(status='STARTED',mode='persistent_'+variant+'_example',scopes=[])
    def save():(root/'persistent_product.json').write_text(json.dumps(report,indent=2)+'\n')
    try:
        cmd=[sys.executable,'-m','vllm.entrypoints.openai.api_server','--model','/model','--served-model-name','acceptance',
            '--host','127.0.0.1','--port','18937','--dtype','bfloat16','--quantization','ascend',
            '--tensor-parallel-size','2','--distributed-executor-backend','mp','--enforce-eager',
            '--no-enable-prefix-caching','--no-async-scheduling','--enable-chunked-prefill',
            '--max-num-batched-tokens','1024','--max-model-len','1024','--max-num-seqs','1',
            '--gpu-memory-utilization','0.8','--seed','42','--trust-remote-code',
            '--worker-extension-cls',extension,
            '--limit-mm-per-prompt','{"image":0,"video":0}']
        server=subprocess.Popen(cmd,env=dict(os.environ,VLLM_WORKER_MULTIPROC_METHOD='spawn'),
            stdout=(root/'http-server.log').open('w'),stderr=subprocess.STDOUT)
        ex=HTTPExecutor('http://127.0.0.1:18937')
        for unused in range(360):
            if server.poll() is not None:raise RuntimeError('SERVER_START_FAILED')
            try:
                with urllib.request.urlopen(ex.base+'/health',timeout=2):break
            except Exception:time.sleep(1)
        else:raise RuntimeError('SERVER_TIMEOUT')
        def rpc(action,options=None):return ex.collective_rpc('demoncat_persistent_example',args=(action,options or {}))
        prompt='Explain why the sky appears blue.'
        submissions={'pending':0,'uncertain':0,'completed':[]}
        def send(text):
            submissions['pending']+=1
            try:
                value=ex.post('/v1/completions',dict(model='acceptance',prompt=text,temperature=0,
                    max_tokens=4,min_tokens=4,ignore_eos=True,seed=42,logprobs=3))
                if (not value.get('id') or not value.get('choices')
                        or any(c.get('finish_reason')!='length' for c in value['choices'])
                        or value.get('usage',{}).get('completion_tokens')!=4):
                    raise RuntimeError('V2_BACKEND_COMPLETION_UNPROVEN')
                submissions['completed'].append(value['id']);return value
            except BaseException:
                # An HTTP failure is not a completed abort. Keep ingress closed
                # until explicit engine recovery; this example never guesses.
                submissions['uncertain']+=1;raise
            finally:submissions['pending']-=1
        ex.collective_rpc('demoncat_persistent_census',args=('start',))
        send(prompt);send(prompt)
        census=ex.collective_rpc('demoncat_persistent_census',args=('stop',))
        # Use rank-local observed census, no layer selected by the user.
        # Each rank has the same names; configure verifies its actual objects.
        report['configure']=rpc('configure',dict(censuses=census))
        def backend_drain(**kw):
            if not submissions['completed']:raise RuntimeError('V2_BACKEND_NO_COMPLETION_OBSERVATION')
            return dict(inflight=submissions['pending']+submissions['uncertain'],
                pending_submissions=submissions['pending'],aborts_completed=submissions['uncertain']==0)
        gate=PersistentIngress(validate_request=lambda r:None if r=={'tokens':4} else 'V2_EXAMPLE_RANGE',
            drain_backend=backend_drain)
        # Network-none, one harness thread, no async submissions or cancellations
        # in the positive sequence. Cancellation is a separate required gate.
        def canary():
            rpc('canary_start');send(prompt);return rpc('canary_end')
        def create():return PersistentProductClient(ex,PersistentFaultConfig('external',variant),
            operator='v2-example',ingress=gate,run_canary=canary,canary_request={'tokens':4},
            control_boundary=lambda owner,snapshot:rpc('boundary',dict(owner=owner,snapshot=snapshot)))
        source_sha=hashlib.sha256(Path('/source/source_manifest.json').read_bytes()).hexdigest()
        negative=create();negative.attach();report['missing_permit']=product_control_reply(negative.enable);save()
        require_negative=report['missing_permit']['exit_code']==4
        if not require_negative:raise RuntimeError('MISSING_PERMIT_NOT_STRUCTURED')
        for index in range(2):
            rpc('permit',dict(attempt='v2-weight-'+str(index),source_sha=source_sha))
            client=create();row=dict(owner=client.identity.owner);report['scopes'].append(row)
            row['attach']=client.attach();row['enable']=client.enable();save()
            row['duplicate_enable']=product_control_reply(client.enable)
            if row['duplicate_enable']['exit_code']!=4:raise RuntimeError('DUPLICATE_ENABLE_NOT_STRUCTURED')
            if variant in ('kv_value_cell','row12_lm_head_collapse','row12_lm_head_swap'):
                def outside_range():
                    with gate.request({'tokens':5}):raise RuntimeError('RANGE_GATE_DID_NOT_REFUSE')
                row['range_negative']=product_control_reply(outside_range)
                if row['range_negative']['exit_code']!=4:raise RuntimeError('RANGE_NEGATIVE_NOT_STRUCTURED')
            rpc('canary_start')
            with gate.request({'tokens':4}):row['fault_http']=send(prompt)
            row['fault_hidden']=rpc('canary_end')
            row['fault_hidden_changed']=row['fault_hidden']!=client.baseline
            row['status']=client.status();save()
            if variant=='kv_value_cell':row['page_audit']=ex.collective_rpc('demoncat_kv_cancel_audit',args=())
            row['disable']=client.disable();row['detach']=client.detach();row['cleanup']=rpc('cleanup');save()
            if not all(r['leases']==r['owners']==0 for r in row['cleanup']):raise RuntimeError('RESOURCES_NOT_ZERO')
        if variant=='kv_value_cell':
            rpc('permit',dict(attempt='v2-kv-http-cancel',source_sha=source_sha))
            client=create();cancel=dict(owner=client.identity.owner);report['cancel_scope']=cancel
            cancel['attach']=client.attach();cancel['enable']=client.enable();save()
            with gate.request({'tokens':4}):
                req=urllib.request.Request(ex.base+'/v1/completions',data=json.dumps(dict(model='acceptance',
                    prompt=prompt,temperature=0,max_tokens=4,min_tokens=4,ignore_eos=True,seed=42,stream=True)).encode(),
                    headers={'Content-Type':'application/json'})
                submissions['pending']+=1
                try:
                    with urllib.request.urlopen(req,timeout=120) as response:
                        for line in response:
                            if line.startswith(b'data: ') and b'[DONE]' not in line:
                                chunk=json.loads(line[6:])
                                if chunk.get('choices') and chunk['choices'][0].get('text'):
                                    cancel['http_id']=chunk['id'];break
                    if not cancel.get('http_id'):raise RuntimeError('CANCEL_STREAM_NO_TOKEN')
                    for _ in range(60):
                        audit=ex.collective_rpc('demoncat_kv_cancel_audit',args=())
                        endings=[[e for e in r['events'] if e['action']=='before_free'
                                  and e['request_id'].startswith(cancel['http_id'])] for r in audit]
                        if len(audit)==2 and all(endings):break
                        time.sleep(.25)
                    else:raise RuntimeError('CANCEL_ENGINE_RELEASE_NOT_OBSERVED')
                    if not all(len(e)==1 and e[0].get('request_status')=='FINISHED_ABORTED' for e in endings):
                        raise RuntimeError('HTTP_CLOSE_WAS_NOT_ACTUAL_ENGINE_ABORT')
                    if any(r['tickets'] or r['leases'] for r in audit):raise RuntimeError('CANCEL_RESOURCES_REMAIN')
                    cancel['audit']=audit;cancel['actual_abort_proven']=True
                except BaseException:submissions['uncertain']+=1;raise
                finally:submissions['pending']-=1
            cancel['status']=client.status();cancel['disable']=client.disable();cancel['detach']=client.detach()
            cancel['cleanup']=rpc('cleanup');save()
        report['observer_cleanup']=rpc('finish')
        label='WEIGHT' if variant=='weight_scale_half' else variant.upper()
        report['status']='TWO_OWNER_'+label+'_EXAMPLE_COMPLETE_NOT_RELEASE_QUALIFICATION'
    except Exception:
        report.update(status='FAILED',error=traceback.format_exc());raise
    finally:
        save()
        if server is not None:
            server.terminate()
            try:server.wait(timeout=45)
            except subprocess.TimeoutExpired:server.kill();server.wait(timeout=15)


if __name__=='__main__':main()
