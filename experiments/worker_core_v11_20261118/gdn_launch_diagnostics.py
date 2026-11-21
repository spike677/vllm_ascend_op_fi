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
            debug=None
            if self.control=='program_meta':
                kwargs=dict(kwargs)
                count=kwargs.get('task_num',grid[0])
                debug=torch.full((count,16),-777,dtype=torch.int32,device=kwargs['k'].device)
                kwargs['Debug']=debug
            kernel=self.base[grid](*args,**kwargs)
            if debug is not None and first:row['program_metadata']=debug.cpu()[:,:4].tolist()
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
        original=getattr(obj,name)
        base=instrument_metadata(original,monitor,name) if control=='program_meta' else original
        wrapper=KernelAudit(base,monitor,name,control)
        setattr(obj,name,wrapper);monitor.patches.append((obj,name,wrapper,True,original))


def instrument_metadata(original,monitor,name):
    """Read the indices consumed inside each real kernel; no output repair."""
    import importlib.util,inspect,sys
    jit=original
    while not hasattr(jit,'src'):
        jit=jit.fn
    source_file=pathlib.Path(inspect.getsourcefile(jit.fn))
    source=source_file.read_text()
    source=metadata_source(source,name)
    target=monitor.file.with_name(monitor.file.stem+'-'+name+'-instrumented.py')
    target.write_text(source)
    module_name='vllm_ascend.ops.triton.fla._dcat_'+name+'_'+monitor.file.stem.replace('-','_')
    spec=importlib.util.spec_from_file_location(module_name,target)
    module=importlib.util.module_from_spec(spec);sys.modules[module_name]=module
    spec.loader.exec_module(module)
    return getattr(module,name)

def metadata_source(source,name):
    marker='def '+name+'(\n'
    assert source.count(marker)==1
    source=source.replace(marker,marker+'    Debug,\n',1)
    anchor='            T = eos - bos' if 'kkt' in name else '            T = eos - bos'
    # Both installed kernels read per-sequence bounds within their program loop.
    assert source.count(anchor)==1,'UNREVIEWED_METADATA_KERNEL'
    slot='task_id' if 'kkt' in name else 'i_t_o'
    statements='\n'.join('            tl.store(Debug + '+slot+' * 16 + '+str(i)+', '+v+')' for i,v in enumerate(('i_n','i_t','bos','eos')))
    source=source.replace(anchor,anchor+'\n'+statements)
    return source
