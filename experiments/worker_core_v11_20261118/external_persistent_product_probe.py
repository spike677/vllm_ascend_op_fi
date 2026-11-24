"""Isolated network-none HTTP example, v2 weight scale through public facade.

The harness is the sole network-namespace client. All its real HTTP inference
submissions pass through PersistentIngress. Never use this acceptance RPC as a
public deployment endpoint. KV is not covered by this weight example.
"""
import hashlib,json,os,subprocess,sys,time,traceback,urllib.request
from pathlib import Path
from external_persistent_census import CensusExtension


class PersistentExampleExtension(CensusExtension):
    def demoncat_persistent_example(self,action,options):
        import torch
        from demoncat.runtime.vllm_ascend.worker_extension.deployment_authorization import (
            configure_deployment_acceptance,authority_for,install_deployment_acceptance_permit,remove_deployment_acceptance_permit)
        from demoncat.runtime.vllm_ascend.worker_extension.serving_context import require
        model=self.model_runner.model
        if action=='configure':
            from external_weight_scale_v2_example import WeightScaleProvider
            provider_factory=WeightScaleProvider;region_kind='persistent_weight';variant='weight_scale_half'
            if os.environ.get('DCAT_V2_KV_PRODUCT')=='1':
                from external_kv_v2_example import KVProvider
                provider_factory=KVProvider;region_kind='persistent_kv';variant='kv_value_cell'
            from demoncat.runtime.vllm_ascend.worker_extension.external_persistent_runtime import install_persistent_runtime,register_persistent_site,PersistentSite
            from demoncat.runtime.vllm_ascend.worker_extension.external_provider import factory_sha256
            from demoncat.runtime.vllm_ascend.worker_extension.deployment_acceptance import observe_deployment
            require(not hasattr(self,'_v2_weight_example_context'),'V2_EXAMPLE_ALREADY_CONFIGURED')
            authority=configure_deployment_acceptance(self,source='BLUE_COMBINATION',
                authorize_operator=lambda principal,operation:principal=='v2-example')
            self._v2_example_boundary=dict(dedicated_instance=True,accepting=False,inflight=0)
            census=next((r for r in options['censuses'] if r['rank']==self.rank),None)
            require(census is not None,'V2_RANK_LOCAL_CENSUS_REQUIRED')
            self._v2_weight_example_context=dict(census=census,boundary=lambda:dict(self._v2_example_boundary))
            install_persistent_runtime(self,authority=authority,operator='v2-example')
            actual=observe_deployment(self,model)
            require(actual['tp']==2 and actual['eager'] is True and actual['max_num_seqs']==1,'V2_EXAMPLE_DEPLOYMENT_RANGE')
            self._v2_example_mechanism='external.'+variant
            register_persistent_site(self,PersistentSite(self._v2_example_mechanism,region_kind,provider_factory,
                factory_sha256(provider_factory),actual),authority=authority,operator='v2-example')
            self._v2_canary_rows=[]
            def observe(module,args,output):
                require(isinstance(output,torch.Tensor),'V2_EXAMPLE_HIDDEN_TENSOR_REQUIRED')
                raw=output.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()
                self._v2_canary_rows.append(dict(sha256=hashlib.sha256(raw).hexdigest(),shape=list(output.shape),
                    dtype=str(output.dtype),finite=bool(torch.isfinite(output).all().item())))
            self._v2_canary_hook=model.register_forward_hook(observe)
            return dict(rank=self.rank,status='CONFIGURED_NOT_QUALIFIED',deployment=actual)
        authority=authority_for(self)
        require(authority is not None and authority.authorize_operator('v2-example','example') is True,'TRUSTED_EXAMPLE_REQUIRED')
        if action=='boundary':
            snapshot=options['snapshot']
            require(snapshot['accepting'] is False and snapshot['inflight']==0 and snapshot['backend_drained'] is True,
                    'V2_EXAMPLE_HOST_DRAIN_REQUIRED')
            torch.npu.current_stream().synchronize()
            self._v2_example_boundary.update(accepting=False,inflight=0)
            return dict(rank=self.rank,owner=options['owner'],inflight=0,device_readers_complete=True)
        if action=='canary_start':self._v2_canary_rows=[];return dict(rank=self.rank,status='ARMED')
        if action=='canary_end':
            torch.npu.current_stream().synchronize();rows=self._v2_canary_rows
            require(len(rows)==4,'V2_EXAMPLE_FOUR_REAL_MODEL_STEPS_REQUIRED')
            return dict(rank=self.rank,all_finite=all(r['finite'] for r in rows),output={'hidden':rows})
        if action=='permit':
            return install_deployment_acceptance_permit(self,authority=authority,operator='v2-example',
                mechanisms=[self._v2_example_mechanism],attempt=options['attempt'],source_manifest_sha256=options['source_sha'])
        if action=='cleanup':
            remove_deployment_acceptance_permit(self,authority=authority,operator='v2-example')
            r=self._demoncat_worker_fault_runtime
            return dict(rank=self.rank,leases=len(r._leases),owners=len(r._tenants))
        if action=='finish':
            self._v2_canary_hook.remove();del self._v2_canary_hook
            return dict(rank=self.rank,status='EXAMPLE_OBSERVER_REMOVED')
        raise ValueError('EXAMPLE_ACTION_UNKNOWN')


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
            if variant=='kv_value_cell':
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
