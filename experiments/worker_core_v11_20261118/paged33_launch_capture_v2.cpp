// Read-only capture. ABI must be checked against the loaded CANN 9.1 libraries.
// No device reads, changed arguments, extra launches, or qualification grant.
#include <acl/acl.h>
#include <dlfcn.h>
#include <unistd.h>
#include <fcntl.h>
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <mutex>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <string>

namespace {
struct Binding { uint64_t key=0; bool entry=false; std::string name; };
struct State {
 std::mutex lock;
 std::unordered_map<void*,Binding> entries;
 std::unordered_set<uint64_t> keys;
 std::string dir;
 bool all=false,valid=true,finalized=false;
 uint64_t pending=0;
 uint64_t limit=65536,handle_limit=131072,byte_limit=1073741824ULL;
 uint64_t seen=0,unarmed=0,unknown=0,filtered=0,selected=0,written=0;
 uint64_t oversize=0,capacity_drop=0,open_error=0,short_write=0,close_error=0;
 uint64_t map_drop=0,copy_error=0,reserved_bytes=0,launch_error=0;
 State(){
  const char* p=getenv("DCAT33_CAPTURE_DIR");if(p)dir=p;
  all=getenv("DCAT33_CAPTURE_ALL") && !strcmp(getenv("DCAT33_CAPTURE_ALL"),"1");
  const char* raw=getenv("DCAT33_CAPTURE_KEYS");
  if(raw){std::string s(raw);size_t off=0;
   while(off<s.size()){size_t end=s.find(',',off);auto part=s.substr(off,end-off);char* tail=nullptr;errno=0;
    auto n=strtoull(part.c_str(),&tail,10);
    if(part.empty() || part[0]<'0' || part[0]>'9' || errno || !tail || *tail)valid=false;
    else keys.insert(n);
    if(end==std::string::npos)break;off=end+1;if(off==s.size())valid=false;
   }
  }
  valid=valid && !dir.empty() && (all || !keys.empty());
  if(const char* raw_limit=getenv("DCAT33_MAX_RECORDS")){
   char* tail=nullptr;errno=0;auto n=strtoull(raw_limit,&tail,10);
   if(errno || !tail || *tail || n<1 || n>1000000)valid=false;else limit=n;
  }
 }
};
State& state(){static State* s=new State;return *s;}
bool emit(int fd,const void* data,size_t size){
 auto p=static_cast<const char*>(data);
 while(size){ssize_t n=write(fd,p,size);if(n<0 && errno==EINTR)continue;
  if(n<=0)return false;p+=n;size-=n;}return true;
}
std::string quote(const std::string& value){
 std::string out="\"";char escaped[7];
 for(unsigned char c:value){
  if(c=='"' || c=='\\'){out+='\\';out+=c;}
  else if(c<32 || c>=127){snprintf(escaped,sizeof(escaped),"\\u%04x",c);out+=escaped;}
  else out+=c;
 }
 return out+'"';
}
}

extern "C" aclError aclrtBinaryGetFunction(aclrtBinHandle b,const char* name,aclrtFuncHandle* f){
 using Fn=aclError(*)(aclrtBinHandle,const char*,aclrtFuncHandle*);
 static Fn real=reinterpret_cast<Fn>(dlsym(RTLD_NEXT,"aclrtBinaryGetFunction"));
 if(!real)std::abort();auto rc=real(b,name,f);auto& s=state();
 if(rc==ACL_SUCCESS && f && *f && name){std::lock_guard<std::mutex> g(s.lock);
  size_t length=strnlen(name,513);
  if(length>512){++s.map_drop;}
  else if(s.entries.count(*f) || s.entries.size()<s.handle_limit){
   // A name is not a tiling key. Zero is only a placeholder, explicitly
   // identified in each record's binding sidecar. Never downgrade a real entry.
   auto& binding=s.entries[*f];binding.name.assign(name,length);
  }else ++s.map_drop;
 }
 return rc;
}

extern "C" aclError aclrtBinaryGetFunctionByEntry(aclrtBinHandle b,uint64_t e,aclrtFuncHandle* f){
 using Fn=aclError(*)(aclrtBinHandle,uint64_t,aclrtFuncHandle*);
 static Fn real=reinterpret_cast<Fn>(dlsym(RTLD_NEXT,"aclrtBinaryGetFunctionByEntry"));
 if(!real)std::abort();auto rc=real(b,e,f);auto& s=state();
 if(rc==ACL_SUCCESS && f && *f){std::lock_guard<std::mutex> g(s.lock);
  if(s.entries.count(*f) || s.entries.size()<s.handle_limit){
   auto& binding=s.entries[*f];binding.key=e;binding.entry=true;
  }else ++s.map_drop;}
 return rc;
}

extern "C" aclError aclrtLaunchKernelWithHostArgs(aclrtFuncHandle f,uint32_t blocks,aclrtStream stream,
 aclrtLaunchKernelCfg* cfg,void* args,size_t size,aclrtPlaceHolderInfo* holders,size_t count){
 using Fn=aclError(*)(aclrtFuncHandle,uint32_t,aclrtStream,aclrtLaunchKernelCfg*,void*,size_t,aclrtPlaceHolderInfo*,size_t);
 static Fn real=reinterpret_cast<Fn>(dlsym(RTLD_NEXT,"aclrtLaunchKernelWithHostArgs"));
 if(!real)std::abort();auto& s=state();uint64_t key=0,id=0;bool capture=false;Binding binding;
 {
  std::lock_guard<std::mutex> g(s.lock);
  // Deployment creates/removes this file for a short, explicitly bounded window.
  if(s.finalized){} // Explicit finalization seals this process window.
  else if(!s.valid || access((s.dir+"/ARMED").c_str(),F_OK)!=0)++s.unarmed;
  else {
   ++s.seen;auto it=s.entries.find(f);
   if(it==s.entries.end())++s.unknown;
   else if(!s.all && (!it->second.entry || !s.keys.count(it->second.key)))++s.filtered;
   else {
    binding=it->second;key=binding.key;id=s.selected++;
    uint64_t bytes=80+size+count*sizeof(aclrtPlaceHolderInfo)+4096;
    if(!args || size==0 || size>65536 || count>128 || (count && !holders))++s.oversize;
    else if(id>=s.limit || bytes>s.byte_limit-s.reserved_bytes)++s.capacity_drop;
    else {s.reserved_bytes+=bytes;++s.pending;capture=true;}
   }
  }
 }
 std::vector<unsigned char> copy,ph;
 if(capture){try{copy.resize(size);std::memcpy(copy.data(),args,size);
  if(count){ph.resize(count*sizeof(aclrtPlaceHolderInfo));std::memcpy(ph.data(),holders,ph.size());}}
  catch(...){std::lock_guard<std::mutex> g(s.lock);++s.copy_error;--s.pending;capture=false;}}
 auto rc=real(f,blocks,stream,cfg,args,size,holders,count);
 if(capture){
  std::string path=s.dir+"/launch-"+std::to_string(getpid())+"-"+std::to_string(id)+".bin";
  int fd=open(path.c_str(),O_WRONLY|O_CREAT|O_EXCL,0600);
  uint64_t h[10]={0x444341544c415531ULL,(uint64_t)getpid(),id,key,blocks,size,count,sizeof(aclrtPlaceHolderInfo),(uint64_t)rc,ph.size()};
  bool ok=false;int closed=0;
  if(fd>=0){ok=emit(fd,h,sizeof(h)) && emit(fd,copy.data(),copy.size()) && emit(fd,ph.data(),ph.size());closed=close(fd);}
  // Provenance is frozen at launch time. A later ByEntry lookup does not
  // retroactively manufacture a key for earlier name-only launches.
  if(fd>=0 && ok && !closed){
   std::string metadata="{\"schema\":\"dcat.launch.binding.v1\",\"pid\":"+std::to_string(getpid())+
    ",\"serial\":"+std::to_string(id)+",\"key_source\":"+quote(binding.entry?"entry":"name_only")+
    ",\"entry\":"+(binding.entry?std::to_string(key):"null")+",\"function_name\":"+quote(binding.name)+"}\n";
   int meta=open((path+".binding.json").c_str(),O_WRONLY|O_CREAT|O_EXCL,0600);
   if(meta<0)fd=-1;else {ok=emit(meta,metadata.data(),metadata.size());closed=close(meta);}
  }
  std::lock_guard<std::mutex> g(s.lock);
  if(rc!=ACL_SUCCESS)++s.launch_error;
  if(fd<0)++s.open_error;else if(!ok)++s.short_write;else if(closed)++s.close_error;else ++s.written;
  --s.pending;
 }
 return rc;
}

extern "C" int dcat33_capture_flush(){
 auto& s=state();std::lock_guard<std::mutex> g(s.lock);
 if(s.finalized)return 0;
 if(s.dir.empty() || s.pending || access((s.dir+"/ARMED").c_str(),F_OK)==0)return 1;
 std::string path=s.dir+"/capture-"+std::to_string(getpid())+".json";
 int fd=open(path.c_str(),O_WRONLY|O_CREAT|O_EXCL,0600);if(fd<0)return 2;
 char out[2048];int n=snprintf(out,sizeof(out),
  "{\"schema\":\"dcat.launch.capture.v2\",\"binding_metadata\":true,\"finalized\":true,\"pending_writes\":0,\"pid\":%d,\"config_valid\":%s,"
  "\"seen\":%llu,\"unarmed\":%llu,\"unknown\":%llu,\"filtered\":%llu,\"selected\":%llu,\"written\":%llu,"
  "\"oversize\":%llu,\"capacity_drop\":%llu,\"open_error\":%llu,\"short_write\":%llu,\"close_error\":%llu,"
  "\"map_drop\":%llu,\"copy_error\":%llu,\"reserved_bytes\":%llu,\"launch_error\":%llu}\n",
  getpid(),s.valid?"true":"false",(unsigned long long)s.seen,(unsigned long long)s.unarmed,
  (unsigned long long)s.unknown,(unsigned long long)s.filtered,(unsigned long long)s.selected,(unsigned long long)s.written,
  (unsigned long long)s.oversize,(unsigned long long)s.capacity_drop,(unsigned long long)s.open_error,
  (unsigned long long)s.short_write,(unsigned long long)s.close_error,(unsigned long long)s.map_drop,
  (unsigned long long)s.copy_error,(unsigned long long)s.reserved_bytes,(unsigned long long)s.launch_error);
 bool ok=n>0 && (size_t)n<sizeof(out) && emit(fd,out,n);
 if(fsync(fd)!=0)ok=false;if(close(fd)!=0)ok=false;
 s.finalized=ok;return ok?0:2;
}
__attribute__((destructor)) static void finish_capture(){dcat33_capture_flush();}
