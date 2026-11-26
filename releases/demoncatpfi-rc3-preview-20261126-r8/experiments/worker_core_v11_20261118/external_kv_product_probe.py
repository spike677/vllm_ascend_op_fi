"""Private-network KV persistent example; actual engine page release barrier."""
import os
from pathlib import Path
from external_kv_page_probe import KVPageExtension
from external_persistent_product_probe import PersistentExampleExtension,main


class KVProductExtension(PersistentExampleExtension,KVPageExtension):
    def demoncat_kv_cancel_audit(self):
        import torch
        torch.npu.current_stream().synchronize()
        p=getattr(self,'_v2_active_kv_provider',None)
        return dict(rank=self.rank,events=getattr(self,'_v2_page_evidence',[]),
                    tickets=len(getattr(self,'_v2_page_tickets',{})),
                    restore_proofs=getattr(self,'_v2_free_proofs',[]),
                    leases=p.regions.leases.count() if p is not None else 0)

    def demoncat_kv_page_evidence(self,action,payload):
        import torch
        from demoncat.runtime.vllm_ascend.worker_extension.serving_context import require
        require(action in ('before_execute','before_free'),'V2_PAGE_ACTION_UNKNOWN')
        require(payload['sequence']>getattr(self,'_v2_page_sequence',0),'V2_PAGE_STALE_REQUEST')
        if action=='before_free':
            require(getattr(self,'_v2_page_tickets',{}).get(payload['request_id'])==payload['request_generation'],
                    'KV_FREE_GENERATION_MISMATCH')
        self._v2_kv_notice=dict(payload,before_free=action=='before_free')
        provider=getattr(self,'_v2_active_kv_provider',None)
        proof=None
        if provider is not None and action=='before_free':
            torch.npu.current_stream().synchronize();proof=provider.before_free()
        result=super().demoncat_kv_page_evidence(action,payload)
        if proof is not None:
            result.update(proof,read_only_no_injection=False)
            if not hasattr(self,'_v2_free_proofs'):self._v2_free_proofs=[]
            self._v2_free_proofs.append(dict(payload=payload,result=result))
        return result


if __name__=='__main__':
    root=Path('/evidence/owned-plugin');meta=root/'dcat_page_probe-0.0.0.dist-info';meta.mkdir(parents=True)
    (meta/'METADATA').write_text('Metadata-Version: 2.1\nName: dcat-page-probe\nVersion: 0.0.0\n')
    (meta/'entry_points.txt').write_text('[vllm.general_plugins]\ndcat_page_probe = external_kv_engine_plugin:register\n')
    os.environ['PYTHONPATH']=str(root)+os.pathsep+os.environ.get('PYTHONPATH','')
    os.environ['DCAT_V2_KV_PAGE_PROBE']='1';os.environ['DCAT_V2_KV_PRODUCT']='1'
    main(extension='external_kv_product_probe.KVProductExtension',variant='kv_value_cell')
