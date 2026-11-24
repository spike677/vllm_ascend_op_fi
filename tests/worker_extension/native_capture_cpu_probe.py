"""Linux CPU-only interposer test using a fake ACL ABI; never an NPU/ABI acceptance."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT=Path(__file__).resolve().parents[2]
SOURCE=ROOT/'experiments/worker_core_v11_20261118/paged33_launch_capture_v2.cpp'
HEADER='''#pragma once
#include <cstdint>
#include <cstddef>
typedef int aclError;
typedef void* aclrtBinHandle;
typedef void* aclrtFuncHandle;
typedef void* aclrtStream;
struct aclrtLaunchKernelCfg { int unused; };
struct aclrtPlaceHolderInfo { uint64_t a,b; };
#define ACL_SUCCESS 0
extern "C" aclError aclrtBinaryGetFunctionByEntry(aclrtBinHandle,uint64_t,aclrtFuncHandle*);
extern "C" aclError aclrtBinaryGetFunction(aclrtBinHandle,const char*,aclrtFuncHandle*);
extern "C" aclError aclrtLaunchKernelWithHostArgs(aclrtFuncHandle,uint32_t,aclrtStream,aclrtLaunchKernelCfg*,void*,size_t,aclrtPlaceHolderInfo*,size_t);
'''
FAKE='''#include <acl/acl.h>
#include <cstring>
extern "C" aclError aclrtBinaryGetFunction(aclrtBinHandle,const char* name,aclrtFuncHandle* f){
 *f=(void*)101;return !strcmp(name,"fail")?7:0;}
extern "C" aclError aclrtBinaryGetFunctionByEntry(aclrtBinHandle,uint64_t e,aclrtFuncHandle* f){*f=(void*)(uintptr_t)(e+1);return 0;}
extern "C" aclError aclrtLaunchKernelWithHostArgs(aclrtFuncHandle f,uint32_t b,aclrtStream,aclrtLaunchKernelCfg*,void* p,size_t n,aclrtPlaceHolderInfo*,size_t){
 return f && b==3 && n>=8 && *(uint64_t*)p==0x12345678 ? 0:5;
}
'''
MAIN='''#include <acl/acl.h>
#include <cstdlib>
#include <cstdio>
#include <unistd.h>
#include <dlfcn.h>
#include <string>
int main(int argc,char** argv){aclrtFuncHandle f=nullptr;uint64_t args[8193]={0x12345678};
 auto flush=(int(*)())dlsym(RTLD_DEFAULT,"dcat33_capture_flush");if(!flush)return 13;
 std::string marker=std::string(getenv("DCAT33_CAPTURE_DIR"))+"/ARMED";
 if(access(marker.c_str(),F_OK)==0 && flush()!=1)return 14;
 const char* mode=getenv("BINDING_TEST");
 if(mode){
  std::string m(mode);
  if(m=="failed_lookup"){
   if(aclrtBinaryGetFunction(nullptr,"fail",&f)!=7)return 20;
   if(aclrtLaunchKernelWithHostArgs(f,3,nullptr,nullptr,args,8,nullptr,0))return 21;
  }else{
   if(m=="entry_first" && aclrtBinaryGetFunctionByEntry(nullptr,100,&f))return 22;
   if(aclrtBinaryGetFunction(nullptr,"named_kernel",&f))return 23;
   if(aclrtLaunchKernelWithHostArgs(f,3,nullptr,nullptr,args,8,nullptr,0))return 24;
   if(m=="promote"){
    if(aclrtBinaryGetFunctionByEntry(nullptr,100,&f))return 25;
    if(aclrtLaunchKernelWithHostArgs(f,3,nullptr,nullptr,args,8,nullptr,0))return 26;
    if(aclrtBinaryGetFunction(nullptr,"named_kernel",&f))return 27;
    if(aclrtLaunchKernelWithHostArgs(f,3,nullptr,nullptr,args,8,nullptr,0))return 28;
   }
  }
 }else for(int i=0;i<9000;i++){if(aclrtBinaryGetFunctionByEntry(nullptr,100+i,&f))return 10;
  if(aclrtLaunchKernelWithHostArgs(f,3,nullptr,nullptr,args,argc>1?sizeof(args):8,nullptr,0))return 11;
  if(args[0]!=0x12345678)return 12;}
 unlink(marker.c_str());if(flush()!=0 || flush()!=0)return 15;
 printf("%d",getpid());return 0;}
'''

def main():
    spec=importlib.util.spec_from_file_location('capture_review',ROOT/'tools/review_launch_capture_v2.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    result=dict(schema='dcat.launch.capture.cpu-probe.v1',npu_calls=0,
                abi='FAKE_CPU_ONLY_NOT_CANN_ABI_ACCEPTANCE',
                source_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
                compiler=subprocess.check_output(['g++','--version'],text=True).splitlines()[0],cases=[])
    with tempfile.TemporaryDirectory(prefix='dcat33-cpu-') as temp:
        p=Path(temp);(p/'acl').mkdir();(p/'acl/acl.h').write_text(HEADER)
        (p/'fake.cpp').write_text(FAKE);(p/'main.cpp').write_text(MAIN)
        subprocess.run(['g++','-std=c++17','-shared','-fPIC','-I',str(p),str(p/'fake.cpp'),'-o',str(p/'libfake.so')],check=True)
        subprocess.run(['g++','-std=c++17','-shared','-fPIC','-I',str(p),str(SOURCE),'-ldl','-pthread','-o',str(p/'capture.so')],check=True)
        subprocess.run(['g++','-std=c++17','-I',str(p),str(p/'main.cpp'),'-L',str(p),'-lfake','-ldl','-Wl,-rpath,'+str(p),'-o',str(p/'main')],check=True)
        for name,limit,armed,oversize in [('full_9000',10000,True,False),('capacity',16,True,False),('oversize',10000,True,True),('unarmed',10000,False,False)]:
            out=p/name;out.mkdir()
            if armed:(out/'ARMED').touch()
            env=dict(os.environ,LD_PRELOAD=str(p/'capture.so'),DCAT33_CAPTURE_DIR=str(out),
                     DCAT33_CAPTURE_ALL='1',DCAT33_MAX_RECORDS=str(limit))
            pid=int(subprocess.check_output([str(p/'main')]+(['oversize'] if oversize else []),env=env,text=True))
            reviewed=module.review(out,pid);c=reviewed['census']
            assert reviewed['complete']==(name=='full_9000'), reviewed
            if name=='full_9000':assert c['written']==9000 and len(reviewed['records'])==9000
            elif name=='capacity':assert c['capacity_drop']==8984 and c['written']==16
            elif name=='oversize':assert c['oversize']==9000
            else:assert c['unarmed']==9000 and c['written']==0
            result['cases'].append(dict(case=name,passed=True,census=c,complete=reviewed['complete'],issues=reviewed['issues']))
        for name in ('name_only','promote','entry_first','failed_lookup','name_filtered'):
            out=p/name;out.mkdir();(out/'ARMED').touch()
            env=dict(os.environ,LD_PRELOAD=str(p/'capture.so'),DCAT33_CAPTURE_DIR=str(out),
                     DCAT33_CAPTURE_ALL='0' if name=='name_filtered' else '1',
                     DCAT33_CAPTURE_KEYS='0',BINDING_TEST=name)
            pid=int(subprocess.check_output([str(p/'main')],env=env,text=True))
            reviewed=module.review(out,pid);c=reviewed['census']
            assert reviewed['complete']==(name not in ('failed_lookup','name_filtered')), reviewed
            records=sorted(reviewed['records'],key=lambda r:r['serial'])
            if name=='name_only':
                assert c['unknown']==0 and records[0]['key'] is None and records[0]['key_source']=='name_only'
            elif name=='promote':
                assert [r['key'] for r in records]==[None,100,100]
            elif name=='entry_first':assert records[0]['key']==100
            elif name=='failed_lookup':assert c['unknown']==1 and not records
            else:assert c['filtered']==1 and not records
            result['cases'].append(dict(case=name,passed=True,census=c,records=records,
                                      complete=reviewed['complete'],issues=reviewed['issues']))
    result['passed']=all(c['passed'] for c in result['cases'])
    Path(sys.argv[1]).write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(passed=result['passed'],cases=len(result['cases']),npu_calls=0)))

if __name__=='__main__':main()
