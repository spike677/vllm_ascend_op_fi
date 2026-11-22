// Read-only host launch capture, pinned to public CANN 9.1 ACL signatures.
// Does not read device memory, alter arguments, launch extra kernels or grant access.
#include <acl/acl.h>
#include <dlfcn.h>
#include <unistd.h>
#include <fcntl.h>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <mutex>
#include <unordered_map>
#include <vector>
#include <cstring>
static std::mutex guard;
static std::unordered_map<void*,uint64_t> entries;
static uint64_t serial=0;
extern "C" aclError aclrtBinaryGetFunctionByEntry(aclrtBinHandle b,uint64_t e,aclrtFuncHandle* f){
 using Fn=aclError(*)(aclrtBinHandle,uint64_t,aclrtFuncHandle*);
 static Fn real=reinterpret_cast<Fn>(dlsym(RTLD_NEXT,"aclrtBinaryGetFunctionByEntry"));
 if(!real)std::abort();
 auto rc=real(b,e,f);
 if(rc==ACL_SUCCESS && f && (e==5100000000010200203ULL || e==5000000000000200203ULL)){
  std::lock_guard<std::mutex> lock(guard);entries[*f]=e;
 }
 return rc;
}
extern "C" aclError aclrtLaunchKernelWithHostArgs(aclrtFuncHandle f,uint32_t blocks,aclrtStream stream,
 aclrtLaunchKernelCfg* cfg,void* args,size_t size,aclrtPlaceHolderInfo* holders,size_t count){
 using Fn=aclError(*)(aclrtFuncHandle,uint32_t,aclrtStream,aclrtLaunchKernelCfg*,void*,size_t,aclrtPlaceHolderInfo*,size_t);
 static Fn real=reinterpret_cast<Fn>(dlsym(RTLD_NEXT,"aclrtLaunchKernelWithHostArgs"));
 if(!real)std::abort();
 uint64_t key=0,id=0;
 {std::lock_guard<std::mutex> lock(guard);auto it=entries.find(f);if(it!=entries.end() && serial<8192){key=it->second;id=serial++;}}
 std::vector<unsigned char> copy,ph;
 if(key && args && size<=65536 && count<=128){
  copy.resize(size);std::memcpy(copy.data(),args,size);
  if(holders && count){ph.resize(count*sizeof(aclrtPlaceHolderInfo));std::memcpy(ph.data(),holders,ph.size());}
 }
 auto rc=real(f,blocks,stream,cfg,args,size,holders,count);
 if(!copy.empty()){
  const char* dir=std::getenv("DCAT_LAUNCH_CAPTURE_DIR");
  if(dir){char name[1024];std::snprintf(name,sizeof(name),"%s/launch-%d-%llu.bin",dir,getpid(),(unsigned long long)id);
   int fd=open(name,O_WRONLY|O_CREAT|O_EXCL,0600);
   if(fd>=0){uint64_t h[10]={0x444341544c415531ULL,(uint64_t)getpid(),id,key,blocks,size,count,sizeof(aclrtPlaceHolderInfo),(uint64_t)rc,ph.size()};
    auto emit=[&](const void* p,size_t n){auto* c=static_cast<const char*>(p);while(n){ssize_t w=write(fd,c,n);if(w<=0)break;c+=w;n-=w;}};
    emit(h,sizeof(h));emit(copy.data(),copy.size());if(!ph.empty())emit(ph.data(),ph.size());close(fd);
   }
  }
 }
 return rc;
}
