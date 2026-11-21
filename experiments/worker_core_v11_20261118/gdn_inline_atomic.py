"""Private C220 compiler diagnostic: atomic-none at real GDN kernel entry.
No output initialization or mathematical changes; not public deployment admission.
"""
import hashlib, pathlib, re, subprocess, types

HELPER = '/inline-reset/reset-cce-hiipu64-hisilicon-cce-dav-c220-cube.bc'
TARGETS = ('recompute_w_u_fwd_kernel_atomic_none', 'chunk_scaled_dot_kkt_fwd_kernel_atomic_none')

def rewrite_ir(source):
    lines=source.splitlines(); output=[]; count=0
    declarations=[]
    for kind in ('aic','aiv'):
        declarations.append('  func.func private @dcat_gdn_atomic_none_'+kind+'.cube() attributes {hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<'+kind.upper()+'>, hivm.part_of_mix}')
    for line in lines:
        output.append(line)
        if line.startswith('module ') and line.rstrip().endswith('{'):
            output.extend(declarations)
        if 'func.func @' in line and 'hacc.entry' in line:
            assert any('@'+t+'_mix_' in line for t in TARGETS), 'UNREVIEWED_KERNEL_ENTRY'
            kind='aic' if 'hivm.func_core_type<AIC>' in line else 'aiv'
            assert 'hivm.func_core_type<'+kind.upper()+'>' in line
            output.append('    func.call @dcat_gdn_atomic_none_'+kind+'.cube() : () -> ()')
            count+=1
    assert count==2, ('UNREVIEWED_ENTRY_COUNT',count)
    return '\n'.join(output)+'\n'

class CompilerProcess:
    def __init__(self, original, monitor): self.original=original;self.monitor=monitor
    def __getattr__(self,name):return getattr(self.original,name)
    def run(self,args,*a,**kw):
        if not (isinstance(args,list) and str(args[0]).endswith('bishengir-compile')):
            return self.original.run(args,*a,**kw)
        src=pathlib.Path(args[1]); source=src.read_text()
        if not any('@'+t+'(' in source for t in TARGETS):
            return self.original.run(args,*a,**kw)
        out=pathlib.Path(args[args.index('-o')+1]); temps=out.parent/'atomic-temps'
        result=self.original.run(args+['--save-temps='+str(temps)],*a,**kw)
        ir=temps/'module.hivm.opt.mlir'; fixed=rewrite_ir(ir.read_text()); patched=out.parent/'atomic-fixed.mlir';patched.write_text(fixed)
        command=['/usr/local/Ascend/cann-9.1.0/bin/hivmc',str(patched),'--enable-debug-info=false','--enable-static-bare-ptr=true','--enable-bin-relocation=true','--link-aicore-bitcode='+HELPER,'-o',str(out)]
        self.original.run(command,*a,**kw)
        artifact=out.with_suffix('.o'); assert artifact.is_file()
        import json
        record=dict(kind='inline_atomic_recompile',targets=[t for t in TARGETS if '@'+t+'(' in source],original_ir_sha256=hashlib.sha256(ir.read_bytes()).hexdigest(),patched_ir_sha256=hashlib.sha256(fixed.encode()).hexdigest(),binary_sha256=hashlib.sha256(artifact.read_bytes()).hexdigest(),helper_sha256=hashlib.sha256(pathlib.Path(HELPER).read_bytes()).hexdigest(),entry_calls=2)
        with self.monitor.file.open('a') as f:f.write(json.dumps(record)+'\n')
        self.monitor.file.with_name(self.monitor.file.stem+'-'+record['targets'][0]+'-inline.mlir').write_text(fixed)
        return result

def install(monitor):
    import torch
    import triton.backends.ascend.compiler as compiler
    assert torch.npu.get_device_name()=='Ascend910B4'
    assert hashlib.sha256(pathlib.Path(HELPER).read_bytes()).hexdigest()=='77e72155b2740f655ff5e66c085a4e4bfdf65cb72c63f4ebac00510a2e23d6dd', 'UNREVIEWED_HELPER_BUILD'
    original=compiler.subprocess; wrapper=CompilerProcess(original,monitor)
    compiler.subprocess=wrapper;monitor.patches.append((compiler,'subprocess',wrapper,True,original))
