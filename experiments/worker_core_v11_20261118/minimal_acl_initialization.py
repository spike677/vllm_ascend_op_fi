"""Minimal isolated ACL lifecycle; no model, tensor allocation or operator."""
import ctypes as C
import datetime,hashlib,json,os,pathlib,subprocess,sys
r=dict(status='STARTED',started=datetime.datetime.now(datetime.timezone.utc).isoformat(),pid=os.getpid(),calls=[],model_loaded=False,operator_calls=0,device_memory_allocations=0)
out=pathlib.Path('/evidence/acl-init.json');initialized=False;selected=False;context=C.c_void_p();lib=None

def call(name,*args):
    code=int(getattr(lib,name)(*args));r['calls'].append(dict(api=name,status=code))
    if code:raise RuntimeError(name+':'+str(code))
try:
    p=subprocess.run(['/usr/local/bin/npu-smi','info'],capture_output=True,text=True,timeout=30)
    r['npu_smi']=dict(exit_code=p.returncode,stdout=p.stdout,stderr=p.stderr)
    if p.returncode:raise RuntimeError('NPU_SMI_FAILED')
    lib=C.CDLL('libascendcl.so')
    signatures={'aclInit':[C.c_char_p],'aclrtGetDeviceCount':[C.POINTER(C.c_uint32)],'aclrtSetDevice':[C.c_int32],
      'aclrtCreateContext':[C.POINTER(C.c_void_p),C.c_int32],'aclrtDestroyContext':[C.c_void_p],
      'aclrtResetDevice':[C.c_int32],'aclFinalize':[]}
    for name,args in signatures.items():getattr(lib,name).argtypes=args;getattr(lib,name).restype=C.c_int
    call('aclInit',None);initialized=True
    count=C.c_uint32();call('aclrtGetDeviceCount',C.byref(count));r['visible_device_count']=count.value
    if count.value!=1:raise RuntimeError('EXACTLY_ONE_VISIBLE_DEVICE_REQUIRED')
    call('aclrtSetDevice',0);selected=True
    call('aclrtCreateContext',C.byref(context),0)
    if not context.value:raise RuntimeError('NULL_CONTEXT')
    r['status']='CONTEXT_CREATED_PENDING_CLEANUP'
except BaseException as e:r['error']=repr(e);r['status']='FAILED'
finally:
    for name,args,needed in [('aclrtDestroyContext',(context,),bool(context.value)),('aclrtResetDevice',(0,),selected),('aclFinalize',(),initialized)]:
        if needed:
            try:call(name,*args)
            except BaseException as e:r.setdefault('cleanup_errors',[]).append(repr(e))
    r['cleanup_complete']=not r.get('cleanup_errors')
    if r['status']=='CONTEXT_CREATED_PENDING_CLEANUP' and r['cleanup_complete']:r['status']='ACL_INITIALIZATION_AND_CLEANUP_PASSED'
    r['ended']=datetime.datetime.now(datetime.timezone.utc).isoformat()
    out.write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r));
sys.exit(0 if r['status']=='ACL_INITIALIZATION_AND_CLEANUP_PASSED' else 2)
