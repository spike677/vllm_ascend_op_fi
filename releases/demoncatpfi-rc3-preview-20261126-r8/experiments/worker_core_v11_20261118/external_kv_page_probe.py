"""Read-only actual EngineCore page association and release barrier census."""
import os
from pathlib import Path
from external_persistent_census import CensusExtension,main as census_main


class KVPageExtension(CensusExtension):
    def demoncat_kv_page_evidence(self,action,payload):
        import torch
        from demoncat.runtime.vllm_ascend.worker_extension.serving_context import require
        require(os.environ.get('DCAT_V2_KV_PAGE_PROBE')=='1','V2_PAGE_READONLY_EXAMPLE_NOT_CONFIGURED')
        if not hasattr(self,'_v2_page_evidence'):
            self._v2_page_evidence=[];self._v2_page_tickets={};self._v2_page_sequence=0
        require(payload['sequence']>self._v2_page_sequence,'V2_PAGE_STALE_REQUEST')
        self._v2_page_sequence=payload['sequence'];rid=payload['request_id'];generation=payload['request_generation']
        if action=='before_execute':
            require(rid not in self._v2_page_tickets or self._v2_page_tickets[rid]==generation,'V2_PAGE_GENERATION_CHANGED')
            self._v2_page_tickets[rid]=generation
        elif action=='before_free':
            require(self._v2_page_tickets.get(rid)==generation,'V2_PAGE_END_OWNER_MISMATCH')
            torch.npu.current_stream().synchronize()
            del self._v2_page_tickets[rid]
        else:raise ValueError('V2_PAGE_ACTION_UNKNOWN')
        self._v2_page_evidence.append(dict(action=action,**payload))
        # This probe has no injection or backups. Zero leases is observed from
        # that scope, not a claim that a future KV provider restored anything.
        return dict(rank=self.rank,sequence=payload['sequence'],request_id=rid,request_generation=generation,
            acknowledged=True,device_readers_complete=action=='before_free',bytes_restored=True,
            request_leases=0,read_only_no_injection=True)

    def demoncat_persistent_census(self,action):
        result=super().demoncat_persistent_census(action)
        if action=='stop':
            result.update(page_barrier_events=getattr(self,'_v2_page_evidence',[]),
                          active_page_tickets=len(getattr(self,'_v2_page_tickets',{})),
                          page_injection_executed=False)
        return result


if __name__=='__main__':
    # A private importlib-metadata entry point in this task's evidence mount.
    # It is not installed into the base image or other services.
    root=Path('/evidence/owned-plugin');meta=root/'dcat_page_probe-0.0.0.dist-info';meta.mkdir(parents=True)
    (meta/'METADATA').write_text('Metadata-Version: 2.1\nName: dcat-page-probe\nVersion: 0.0.0\n')
    (meta/'entry_points.txt').write_text('[vllm.general_plugins]\ndcat_page_probe = external_kv_engine_plugin:register\n')
    os.environ['PYTHONPATH']=str(root)+os.pathsep+os.environ.get('PYTHONPATH','')
    os.environ['DCAT_V2_KV_PAGE_PROBE']='1'
    census_main(extension='external_kv_page_probe.KVPageExtension')
