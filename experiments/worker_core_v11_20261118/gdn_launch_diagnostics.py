"""Private launch controls; preserve outputs and allocations, never grant admission."""
import hashlib,json,pathlib
class KernelAudit:
    def __init__(self,base,monitor,name,control):self.base=base;self.monitor=monitor;self.name=name;self.control=control;self.calls=0
    def __getattr__(self,name):return getattr(self.base,name)
    def __getitem__(self,grid):
        def run(*args,**kwargs):
            import torch
            self.calls+=1;first=self.calls==1
            row=dict(kind='kernel_launch',name=self.name,control=self.control,grid=list(grid),call=self.calls,arguments={})
            for k,v in kwargs.items():
                if isinstance(v,torch.Tensor):
                    row['arguments'][k]=dict(shape=list(v.shape),stride=list(v.stride()),offset=v.storage_offset(),dtype=str(v.dtype),pointer=v.data_ptr(),alignment512=v.data_ptr()%512)
                    if first and k in ('cu_seqlens','chunk_indices'):row['arguments'][k]['values']=v.cpu().tolist()
                elif isinstance(v,(int,float,bool,str,type(None))):row['arguments'][k]=v
            if self.control=='clone_meta':
                kwargs=dict(kwargs)
                for k in ('cu_seqlens','chunk_indices'):
                    if isinstance(kwargs.get(k),torch.Tensor):kwargs[k]=kwargs[k].clone()
            if self.control=='compile_first':self.base.run(*args,grid=grid,warmup=True,**kwargs)
            kernel=self.base[grid](*args,**kwargs)
            if first:
                row['kernel_hash']=getattr(kernel,'hash',None)
                row['metadata']=getattr(getattr(kernel,'metadata',None),'_asdict',lambda:{})()
                artifacts={}
                for k,v in getattr(kernel,'asm',{}).items():
                    if isinstance(v,(str,bytes)):
                        raw=v.encode() if isinstance(v,str) else v
                        if len(raw)<=8*1024*1024:
                            target=self.monitor.file.with_name(self.monitor.file.stem+'-'+self.name+'.'+k)
                            target.write_bytes(raw);artifacts[k]=dict(bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())
                row['artifacts']=artifacts
                with self.monitor.file.open('a') as f:f.write(json.dumps(row,default=str)+'\n')
            return kernel
        return run

def install(monitor,control):
    import vllm_ascend.ops.triton.fla.wy_fast as wy
    import vllm_ascend.device.device_op as device
    for obj,name in ((wy,'recompute_w_u_fwd_kernel'),(device,'chunk_scaled_dot_kkt_fwd_kernel')):
        original=getattr(obj,name);wrapper=KernelAudit(original,monitor,name,control)
        setattr(obj,name,wrapper);monitor.patches.append((obj,name,wrapper,True,original))
