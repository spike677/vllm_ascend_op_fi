"""Additive, trusted EngineCore page-release barrier for persistent KV v2.

Resource tickets are not a task lifecycle. Runtime still owns provider scopes.
No collective runs at a conditional attention hit: notices run before the common
execute boundary, and restoration runs before the scheduler returns pages.
This adapter is not registered by default and grants no qualification.
"""
import inspect
import hashlib
import uuid
from pathlib import Path
from .serving_context import require

SOURCE_SHA={
    'step':'3ae1381a6af841e21058c825702382dc66faae45c950ac5acb8495d2d3d05aad',
    'free_request':'2ed2a550b6558b2495eda845a97ae38bcf0225027b9e25fbf00fc3880c1d3941',
    'free':'3f4af8d247f3fe9570b0132818b832b66ae6a2ac12942588828f899f6ff77ccf',
}


def verify_sources(engine):
    methods={'step':engine.step,'free_request':engine.scheduler._free_request_blocks,
             'free':engine.scheduler.kv_cache_manager.free}
    actual={}
    for name,method in methods.items():
        path=inspect.getsourcefile(inspect.unwrap(method))
        require(path is not None,'V2_KV_ENGINE_SOURCE_UNAVAILABLE')
        actual[name]=hashlib.sha256(Path(path).read_bytes()).hexdigest()
    require(actual==SOURCE_SHA,'V2_KV_ENGINE_SOURCE_CHANGED')
    return actual


class KVReleaseBarrier:
    """Scoped wrappers on one dedicated EngineCore and its own KV manager.

    notify is a trusted all-rank Worker RPC transport. It must return the exact
    sequence and request-generation on every rank, never stale cached replies.
    group_ids comes from actual full-attention cache layout discovery.
    """
    def __init__(self,engine,*,notify,world_size,group_ids,source_check=verify_sources):
        require(callable(notify) and type(world_size) is int and world_size>0,'V2_KV_RPC_REQUIRED')
        require(type(group_ids) is tuple and group_ids and len(set(group_ids))==len(group_ids)
                and all(type(i) is int and i>=0 for i in group_ids),'V2_KV_FULL_ATTENTION_GROUPS_REQUIRED')
        require(engine.async_scheduling is False and engine.batch_queue is None,
                'V2_KV_SYNCHRONOUS_ENGINE_ONLY')
        cfg=engine.vllm_config
        require(cfg.cache_config.enable_prefix_caching is False and cfg.kv_transfer_config is None
                and cfg.ec_transfer_config is None and cfg.speculative_config is None
                and cfg.parallel_config.pipeline_parallel_size==1,
                'V2_KV_PRIVATE_NONCONNECTOR_NON_SPECULATIVE_ONLY')
        self.sources=source_check(engine);self.engine=engine;self.notify=notify
        self.world_size=world_size;self.group_ids=group_ids;self.tickets={};self.patches=[];self.events=[]
        self.sequence=0

    def _pages(self,request_id):
        groups=self.engine.scheduler.kv_cache_manager.get_blocks(request_id).blocks
        require(all(i<len(groups) for i in self.group_ids),'V2_KV_GROUP_LAYOUT_CHANGED')
        pages=[]
        for group in self.group_ids:
            seen=set()
            for block in groups[group]:
                require(getattr(block,'is_null',None) is False and type(block.ref_cnt) is int
                        and block.ref_cnt==1,'V2_KV_PAGE_NOT_EXCLUSIVE')
                require(type(block.block_id) is int and block.block_id>=0 and block.block_id not in seen,
                        'V2_KV_PAGE_ID_INVALID')
                seen.add(block.block_id);pages.append([group,block.block_id])
        require(pages,'V2_KV_NO_OWNED_FULL_ATTENTION_PAGES')
        return pages

    def _rpc(self,action,ticket,pages):
        self.sequence+=1;sequence=self.sequence
        payload=dict(sequence=sequence,request_id=ticket['request_id'],request_generation=ticket['generation'],
                     pages=pages,world_size=self.world_size)
        rows=self.notify(action,payload)
        require(type(rows) is list and len(rows)==self.world_size
                and all(type(r.get('rank')) is int for r in rows)
                and sorted(r['rank'] for r in rows)==list(range(self.world_size)),
                'V2_KV_ALL_RANK_ACK_REQUIRED')
        require(all(r.get('sequence')==sequence and r.get('request_generation')==ticket['generation']
                    and r.get('request_id')==ticket['request_id'] and r.get('acknowledged') is True for r in rows),
                'V2_KV_STALE_OR_REFUSED_ACK')
        if action=='before_free':
            require(all(r.get('device_readers_complete') is True and r.get('bytes_restored') is True
                        and type(r.get('request_leases')) is int and r['request_leases']==0 for r in rows),
                    'V2_KV_RELEASE_RESTORE_NOT_PROVEN')
        self.events.append(dict(action=action,**payload,ranks=rows))

    def before_execute(self,scheduler_output):
        for rid in scheduler_output.num_scheduled_tokens:
            request=self.engine.scheduler.requests.get(rid)
            require(request is not None,'V2_KV_ACTUAL_REQUEST_MISSING')
            ticket=self.tickets.get(rid)
            if ticket is None:
                ticket=dict(request=request,request_id=rid,generation=uuid.uuid4().hex)
                self.tickets[rid]=ticket
            require(ticket['request'] is request,'V2_KV_REQUEST_ID_REUSED')
            pages=self._pages(rid)
            previous={tuple(p) for p in ticket.get('pages',())}
            require(previous<={tuple(p) for p in pages},'V2_KV_PAGE_REMOVED_WITHOUT_RELEASE_BARRIER')
            self._rpc('before_execute',ticket,pages)
            ticket['pages']=pages

    def before_free(self,request):
        rid=request.request_id;ticket=self.tickets.get(rid)
        if ticket is None:return # never executed/never exposed to a v2 writer
        require(ticket['request'] is request,'V2_KV_REQUEST_ID_REUSED')
        pages=self._pages(rid)
        require(pages==ticket.get('pages'),'V2_KV_PAGE_SET_CHANGED_BEFORE_RELEASE')
        self._rpc('before_free',ticket,pages)

    def install(self):
        require(not self.patches,'V2_KV_RELEASE_ALREADY_INSTALLED')
        executor=self.engine.model_executor;manager=self.engine.scheduler.kv_cache_manager
        for obj,name in ((executor,'execute_model'),(manager,'free'),(manager,'pop_blocks_for_free')):
            require(name not in vars(obj),'V2_KV_EXISTING_WRAPPER:'+name)
        original=executor.execute_model
        def execute(output,*args,**kwargs):
            self.before_execute(output);return original(output,*args,**kwargs)
        executor.execute_model=execute;self.patches.append((executor,'execute_model',execute))
        for name in ('free','pop_blocks_for_free'):
            method=getattr(manager,name)
            def release(request,*args,_original=method,**kwargs):
                self.before_free(request);result=_original(request,*args,**kwargs)
                self.tickets.pop(request.request_id,None)
                return result
            setattr(manager,name,release);self.patches.append((manager,name,release))

    def close(self):
        require(not self.tickets,'V2_KV_ACTIVE_REQUEST_TICKETS_RETAINED')
        require(all(vars(obj).get(name) is method for obj,name,method in self.patches),'V2_KV_WRAPPER_OWNER_CHANGED')
        for obj,name,method in reversed(self.patches):delattr(obj,name)
        self.patches=[]
