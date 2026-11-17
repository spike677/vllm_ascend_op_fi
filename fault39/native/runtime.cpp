// The four private ABI interpositions and metadata ordering proof below derive
// from l4_buffer_substitute_allhit_v1, validated historically with CANN 8.5.1.
// There are no other fault routes in this library.
#include "fault39_core.h"
#include "fault39_build_profile.h"
#include <acl/acl.h>
#include <aclnn/opdev/op_cache.h>
#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdlib>
#include <dlfcn.h>
#include <limits.h>
#include <map>
#include <mutex>
#include <set>
#include <thread>
#include <vector>
#include <unistd.h>
#if !defined(__aarch64__) || !defined(_GLIBCXX_USE_CXX11_ABI) || _GLIBCXX_USE_CXX11_ABI!=0
#error "Fault39 native ABI requires aarch64 and _GLIBCXX_USE_CXX11_ABI=0"
#endif
static_assert(ACL_BF16==fault39::kBf16 && ACL_FORMAT_ND==fault39::kNd,"ACL enum ABI");
namespace {
using Cache=op::internal::OpExecCache;
using Run=aclnnStatus(*)(Cache*,void*,aclrtStream,const std::vector<void*>&);
using Add=bool(*)(Cache*); using ParamC=void(*)(const aclTensor*); using Param=void(*)(aclTensor*);
using ApiInfo=std::string(*)();
using Device=aclError(*)(int32_t*); using Alloc=aclError(*)(void**,size_t,aclrtMemMallocPolicy);
using Free=aclError(*)(void*); using HostAlloc=aclError(*)(void**,size_t);
using Copy=aclError(*)(void*,size_t,const void*,size_t,aclrtMemcpyKind); using Sync=aclError(*)(aclrtStream);
using GetContext=aclError(*)(aclrtContext*);
Run real_run=nullptr; Add real_add=nullptr; ParamC real_paramc=nullptr; Param real_param=nullptr;
ApiInfo api_info=nullptr; Device get_device=nullptr; Alloc alloc=nullptr; Free release=nullptr,host_release=nullptr;
HostAlloc host_alloc=nullptr; Copy copy=nullptr; Sync sync_stream=nullptr;
GetContext get_context=nullptr;
std::mutex resolve_mutex;
std::atomic<int32_t> worker_rank{-1};
bool runtime_ok=false;
std::string runtime_path;
template<class T> void resolve_one(T& fn,const char* symbol,void* fallback=nullptr) {
 if(!fn)fn=reinterpret_cast<T>(dlsym(RTLD_NEXT,symbol));
 if(!fn && fallback)fn=reinterpret_cast<T>(dlsym(fallback,symbol));
}
template<class T> void resolve_public(T& fn,const char* symbol,void* verified_library) {
 // CANN also exports these names from libacl_rt.so. RTLD_NEXT is load-order
 // dependent; bind the exact public facade profiled at build time. Do not
 // load a missing runtime or accept another DSO merely for sharing a symbol.
 if(!fn && verified_library)fn=reinterpret_cast<T>(dlsym(verified_library,symbol));
}
bool resolve() {
 std::lock_guard<std::mutex> lock(resolve_mutex);
 void* nn=dlopen("libnnopbase.so",RTLD_LAZY|RTLD_NOLOAD);
 void* acl=dlopen("libascendcl.so",RTLD_LAZY|RTLD_NOLOAD);
 resolve_one(real_run,"_ZN2op8internal11OpExecCache3RunEPvS2_RKSt6vectorIS2_SaIS2_EE",nn);
 resolve_one(real_add,"_ZN2op8internal14AddOpExecCacheEPNS0_11OpExecCacheE",nn);
 resolve_one(real_paramc,"_ZN2op8internal13AddParamToBufEPK9aclTensor",nn);
 resolve_one(real_param,"_ZN2op8internal13AddParamToBufEP9aclTensor",nn);
 resolve_one(api_info,"_ZN2op8internal13GetLogApiInfoEv",nn);
 resolve_public(get_device,"aclrtGetDevice",acl); resolve_public(alloc,"aclrtMalloc",acl);
 resolve_public(release,"aclrtFree",acl); resolve_public(host_alloc,"aclrtMallocHost",acl);
 resolve_public(host_release,"aclrtFreeHost",acl); resolve_public(copy,"aclrtMemcpy",acl);
 resolve_public(sync_stream,"aclrtSynchronizeStream",acl);
 resolve_public(get_context,"aclrtGetCurrentContext",acl);
 Dl_info info{}; char path[PATH_MAX]{};
 if(runtime_path.empty() && real_run && dladdr(reinterpret_cast<void*>(real_run),&info) && info.dli_fname && realpath(info.dli_fname,path))runtime_path=path;
 // Match the exact build's loaded code and files, not a version in a pathname.
 // Successful identities are pinned/cached; each new worker still needs real
 // discovery and lossless validation before its first injection.
 runtime_ok=true;
 for(void* fn : {reinterpret_cast<void*>(real_run),reinterpret_cast<void*>(real_add),
     reinterpret_cast<void*>(real_paramc),reinterpret_cast<void*>(real_param),reinterpret_cast<void*>(api_info)})
  runtime_ok=f39identity::matches(fn,f39profile::nn) && runtime_ok;
 for(void* fn : {reinterpret_cast<void*>(get_device),reinterpret_cast<void*>(alloc),
     reinterpret_cast<void*>(release),reinterpret_cast<void*>(host_alloc),reinterpret_cast<void*>(host_release),
     reinterpret_cast<void*>(copy),reinterpret_cast<void*>(sync_stream),reinterpret_cast<void*>(get_context)})
  runtime_ok=f39identity::matches(fn,f39profile::acl) && runtime_ok;
 if(nn)dlclose(nn); if(acl)dlclose(acl); return runtime_ok;
}
int actual_rank() {
 return worker_rank.load(std::memory_order_acquire);
}
struct Capture { bool active=false; uint32_t adds=0,params=0; Cache* cache=nullptr; uint64_t tensors[3]{}; };
thread_local Capture capture;
thread_local bool forwarding_param=false;
// BEGIN CPU_TESTABLE_SERVING_INPUT_GUARD
// The owner/request context and the live model input evidence have one TLS
// lifetime. A malformed owner-scoped binding poisons this context instead of
// silently reverting to the legacy (unguarded) admission path.
constexpr size_t kServingInputCount=13;
struct ServingInputs {
 bool requested=false,valid=false;
 uint64_t values[kServingInputCount]{};
};
struct Context { std::string owner,ids; ServingInputs serving; };
thread_local Context context;
bool serving_values_valid(const uint64_t* v,size_t count) {
 if(!v || count!=kServingInputCount)return false;
 for(size_t i=0;i<count;++i)
  if(i>=4 && v[i]>uint64_t(INT64_MAX))return false;
 for(size_t i:{size_t(0),size_t(1),size_t(2),size_t(3),size_t(4),size_t(5),size_t(6),
               size_t(9),size_t(10),size_t(11),size_t(12)})if(!v[i])return false;
 if((v[1]&1) || (v[3]&1) || v[1]>uint64_t(SIZE_MAX) || v[3]>uint64_t(SIZE_MAX) ||
    !fault39::disjoint(v[0],v[1],v[2],v[3]))return false;
 // MxK self and KxN mat2, including any nonzero storage offset and padding.
 const uint64_t rows[]={v[4],v[5]},cols[]={v[5],v[6]},offsets[]={v[7],v[8]},
                s0[]={v[9],v[11]},s1[]={v[10],v[12]},bytes[]={v[1],v[3]};
 for(int i=0;i<2;++i) {
  uint64_t last=offsets[i],step=0;
  if(!fault39::mul(rows[i]-1,s0[i],step) || !fault39::add(last,step,last) ||
     !fault39::mul(cols[i]-1,s1[i],step) || !fault39::add(last,step,last) ||
     last>=bytes[i]/2)return false;
 }
 return true;
}
int bind_serving_inputs(Context& ctx,const char* owner,const uint64_t* values,size_t count) {
 if(!owner || !*owner || ctx.owner.empty() || ctx.ids.empty() || ctx.ids=="[]")return F39_INVALID_CONFIG;
 if(ctx.owner!=owner)return F39_OWNER_MISMATCH;
 const bool already_bound=ctx.serving.requested;
 ctx.serving={};ctx.serving.requested=true;
 if(already_bound)return F39_BUSY;
 if(!serving_values_valid(values,count))return F39_INVALID_CONFIG;
 std::copy(values,values+kServingInputCount,ctx.serving.values);
 ctx.serving.valid=true;return F39_OK;
}
bool serving_inputs_match(const Context& ctx,const Fault39MatmulMetadata& m,
                          const std::vector<void*>& addresses) {
 if(!ctx.serving.requested)return true;  // Explicit legacy context only.
 if(!ctx.serving.valid || addresses.size()!=3)return false;
 if(m.mat2.view_shape[0]!=m.self.view_shape[1] ||
    m.out.view_shape[0]!=m.self.view_shape[0] || m.out.view_shape[1]!=m.mat2.view_shape[1])return false;
 const uint64_t actual[kServingInputCount]={
  reinterpret_cast<uintptr_t>(addresses[0]),m.self.storage_bytes,
  reinterpret_cast<uintptr_t>(addresses[1]),m.mat2.storage_bytes,
  uint64_t(m.self.view_shape[0]),uint64_t(m.self.view_shape[1]),uint64_t(m.mat2.view_shape[1]),
  uint64_t(m.self.view_offset),uint64_t(m.mat2.view_offset),
  uint64_t(m.self.strides[0]),uint64_t(m.self.strides[1]),
  uint64_t(m.mat2.strides[0]),uint64_t(m.mat2.strides[1])};
 return std::equal(actual,actual+kServingInputCount,ctx.serving.values);
}
// END CPU_TESTABLE_SERVING_INPUT_GUARD
struct Registered { Fault39MatmulMetadata metadata{}; int device=-1,rank=-1; uint64_t tensors[3]{},cache_epoch=0; };
struct Candidate { Fault39MatmulMetadata metadata{}; int rank=-1,device=-1; std::set<std::pair<Cache*,uint64_t>> caches; std::map<std::pair<Cache*,uint64_t>,uint64_t> cache_observations; uint64_t observations=0; };
struct Lease {
 void* device[2]{}; void* host[3]{}; aclrtStream stream=nullptr; uint64_t generation=0;
 aclrtContext context=nullptr; int32_t device_id=-1; std::thread::id thread;
};
struct State {
 std::mutex mutex; std::condition_variable cv;
 Fault39ConfigV1 config{}; uint64_t generation=0,inflight=0,hits=0,run_calls=0,metadata_calls=0;
 uint64_t substituted=0,rejected=0,completed=0,serving_input_rejections=0;
 bool enabled=false; int error=0; std::string status="DISABLED",actual_ids="[]",last_run_api;
 std::map<Cache*,Registered> registry; std::map<std::string,Candidate> candidates;
 // Read-only denominator: independent of enabled/owner/mode/target/max_hits.
 // A bounded full counter map never silently evicts an observed cache lifetime.
 std::map<std::string,Candidate> independent_candidates;
 uint64_t independent_runs=0,independent_cache_count=0;bool independent_complete=true;
 std::map<Cache*,uint64_t> cache_epochs;uint64_t next_cache_epoch=0;
 Cache* selected_cache=nullptr;uint64_t selected_epoch=0,observation_cache=0,observation_epoch=0;
 std::vector<Lease> retained; fault39::Counts counts{};
 fault39::Counts original[2]{},temporary[2]{};
 uint64_t changed[2]{},unchanged[2]{};
 bool self_verified=false,columns_verified=false,serving_live_inputs_verified=false,input_device_readback_verified=false;
 std::string proof_json="{}";
 std::string storage_before[2],storage_after[2],output_sha256;
 uint64_t observation_generation=0,observation_hit=0;std::string observation_owner,observation_ids="[]";
 int pre_sync=-2147483647,run_status=-2147483647,post_sync=-2147483647,free_status=-2147483647,copy_status=-2147483647;
};
State& state(){static State* s=new State;return *s;} // Retained leases survive shutdown; never auto-freed.
const char* error_name(int rc) {
 switch(rc){case 0:return "OK";case -1:return "INVALID_CONFIG";case -2:return "ABI_MISMATCH";
 case -3:return "BUSY";case -4:return "OWNER_MISMATCH";case -5:return "GENERATION_MISMATCH";
 case -6:return "AMBIGUOUS_TARGET";case -7:return "TARGET_NOT_FOUND";case -8:return "UNSUPPORTED_RUNTIME";
 case -9:return "RECOVERY_FAILED";case -10:return "DRAIN_TIMEOUT";case -11:return "NO_OVERFLOW_SOLUTION";}return "UNKNOWN";
}
int fail(State& s,int rc){s.error=rc;s.status=error_name(rc);return rc;}
std::string quote(const std::string& v) {
 std::ostringstream out;out<<'"';
 const char* hex="0123456789abcdef";
 for(unsigned char c:v){if(c=='"' || c=='\\')out<<'\\'<<char(c);else if(c<32)out<<"\\u00"<<hex[c>>4]<<hex[c&15];else out<<char(c);}
 out<<'"';return out.str();
}
std::string counts_json(const fault39::Counts& c) {
 std::ostringstream j;j.precision(17);
 j<<"{\"total\":"<<c.total<<",\"finite\":"<<c.finite<<",\"zero\":"<<c.zero<<",\"positive\":"<<c.positive
  <<",\"negative\":"<<c.negative<<",\"nan\":"<<c.nan<<",\"posinf\":"<<c.posinf<<",\"neginf\":"<<c.neginf<<",\"min\":";
 if(c.finite)j<<c.minimum;else j<<"null";j<<",\"max\":";if(c.finite)j<<c.maximum;else j<<"null";
 j<<",\"max_abs\":";if(c.finite)j<<c.max_abs;else j<<"null";
 j<<",\"mean_abs\":";if(c.finite)j<<c.sum_abs/c.finite;else j<<"null";j<<'}';return j.str();
}
bool proof(const Registered& r,const std::vector<void*>& addresses,uint64_t cap) {
 if(addresses.size()!=3 || !fault39::matmul_valid(r.metadata,cap))return false;
 const Fault39TensorMetadata* ts[]={&r.metadata.self,&r.metadata.mat2,&r.metadata.out};
 for(int i=0;i<3;++i) {
  // Required matmul tensors: no optional/null compatibility from QMM.
  if(!addresses[i] || r.tensors[i]!=ts[i]->tensor_ptr)return false;
  for(int j=i+1;j<3;++j)if(!fault39::disjoint(reinterpret_cast<uintptr_t>(addresses[i]),ts[i]->storage_bytes,
      reinterpret_cast<uintptr_t>(addresses[j]),ts[j]->storage_bytes))return false;
 }
 return true;
}
bool free_lease(Lease& l) {
 bool ok=true;
 for(auto& p:l.device)if(p){if(release(p)==ACL_SUCCESS)p=nullptr;else ok=false;}
 for(auto& p:l.host)if(p){if(host_release(p)==ACL_SUCCESS)p=nullptr;else ok=false;}
 return ok;
}
struct Flight {
 State& s; uint64_t generation;
 ~Flight(){std::lock_guard<std::mutex> l(s.mutex);--s.inflight;s.cv.notify_all();}
};
// Single in-flight injection reserves retained capacity before any allocation.
// Any C++ exception after allocation preserves the complete lease for explicit recovery.
struct LeaseGuard {
 State& s;Lease& lease;bool transferred=false;
 ~LeaseGuard() noexcept {
  if(transferred)return;
  bool live=false;for(auto p:lease.device)live|=p!=nullptr;for(auto p:lease.host)live|=p!=nullptr;
  if(live){std::lock_guard<std::mutex> lock(s.mutex);s.retained.push_back(lease);s.enabled=false;s.error=F39_RECOVERY_FAILED;s.status="RECOVERY_FAILED";}
 }
};
aclnnStatus run_selected(Cache* cache,void* workspace,aclrtStream stream,const std::vector<void*>& addresses,
 const Registered& registered,const Fault39ConfigV1& cfg,uint64_t generation) {
 State& s=state(); Flight flight{s,generation}; Lease lease{};lease.stream=stream;lease.generation=generation;
 lease.device_id=cfg.device;lease.thread=std::this_thread::get_id();(void)get_context(&lease.context);
 LeaseGuard lease_guard{s,lease};
 auto recovery=[&]() -> aclnnStatus {
  std::lock_guard<std::mutex> lock(s.mutex); s.retained.push_back(lease);lease_guard.transferred=true;s.enabled=false;fail(s,F39_RECOVERY_FAILED);return 1;
 };
 auto record=[&](int State::*field,int code){std::lock_guard<std::mutex> lock(s.mutex);s.*field=code;return code;};
 if(record(&State::pre_sync,sync_stream(stream))!=ACL_SUCCESS)return recovery();
 const Fault39TensorMetadata* ts[]={&registered.metadata.self,&registered.metadata.mat2,&registered.metadata.out};
 std::vector<void*> replaced;
 try{replaced=addresses;}catch(...){std::lock_guard<std::mutex> lock(s.mutex);s.enabled=false;fail(s,F39_BUSY);return 1;}
 {
  for(int i=0;i<2;++i) {
   const bool substitute=cfg.mode!=F39_OBSERVE && (cfg.input_mask&(1U<<i));
   if(host_alloc(&lease.host[i],ts[i]->storage_bytes)!=ACL_SUCCESS ||
      (substitute && alloc(&lease.device[i],ts[i]->storage_bytes,ACL_MEM_MALLOC_HUGE_FIRST)!=ACL_SUCCESS)) {
    if(!free_lease(lease))return recovery();
    std::lock_guard<std::mutex> lock(s.mutex);s.enabled=false;fail(s,F39_BUSY);return 1;
   }
   if(!lease.host[i])return recovery();
   if(substitute) {
    if(!lease.device[i])return recovery();
    for(int j=0;j<3;++j)if(!fault39::disjoint(reinterpret_cast<uintptr_t>(lease.device[i]),ts[i]->storage_bytes,
        reinterpret_cast<uintptr_t>(addresses[j]),ts[j]->storage_bytes))return recovery();
    if(i==1 && lease.device[0] && !fault39::disjoint(reinterpret_cast<uintptr_t>(lease.device[0]),ts[0]->storage_bytes,
       reinterpret_cast<uintptr_t>(lease.device[1]),ts[1]->storage_bytes))return recovery();
   }
   if(record(&State::copy_status,copy(lease.host[i],ts[i]->storage_bytes,addresses[i],ts[i]->storage_bytes,ACL_MEMCPY_DEVICE_TO_HOST))!=ACL_SUCCESS)return recovery();
   {std::lock_guard<std::mutex> lock(s.mutex);s.storage_before[i]=f39identity::digest(lease.host[i],ts[i]->storage_bytes);}
   const auto original=fault39::observe(lease.host[i],*ts[i]);
   std::vector<uint16_t> before;
   if(cfg.mode==F39_INJECT) {
    try { auto* p=static_cast<uint16_t*>(lease.host[i]);before.assign(p,p+ts[i]->storage_elements); }
    catch(...) {if(!free_lease(lease))return recovery();std::lock_guard<std::mutex> lock(s.mutex);s.enabled=false;fail(s,F39_BUSY);return 1;}
   }
   if(cfg.mode==F39_INJECT)fault39::fill(lease.host[i],*ts[i],cfg.exponent,cfg.pattern,i==1);
   const auto temporary=fault39::observe(lease.host[i],*ts[i]);uint64_t changed=0;
   if(!before.empty()) {
    auto* p=static_cast<uint16_t*>(lease.host[i]);
    for(int64_t row=0;row<ts[i]->view_shape[0];++row)for(int64_t col=0;col<ts[i]->view_shape[1];++col){
     const auto offset=uint64_t(ts[i]->view_offset)+uint64_t(row)*ts[i]->strides[0]+uint64_t(col)*ts[i]->strides[1];
     changed+=before[offset]!=p[offset];
    }
   }
   const bool checked=cfg.mode==F39_INJECT && (i==0?fault39::verify_self(lease.host[i],*ts[i],cfg.exponent):fault39::verify_columns(lease.host[i],*ts[i],cfg.exponent));
   {std::lock_guard<std::mutex> lock(s.mutex);s.original[i]=original;s.temporary[i]=temporary;s.changed[i]=changed;s.unchanged[i]=uint64_t(ts[i]->numel)-changed;
    if(i==0)s.self_verified=checked;else s.columns_verified=checked;}
   if(cfg.mode==F39_INJECT && !checked)return recovery();
   if(substitute) {
    if(record(&State::copy_status,copy(lease.device[i],ts[i]->storage_bytes,lease.host[i],ts[i]->storage_bytes,ACL_MEMCPY_HOST_TO_DEVICE))!=ACL_SUCCESS)return recovery();
    if(s.selected_cache) {
     const auto expected=f39identity::digest(lease.host[i],ts[i]->storage_bytes);
     if(record(&State::copy_status,copy(lease.host[i],ts[i]->storage_bytes,lease.device[i],ts[i]->storage_bytes,ACL_MEMCPY_DEVICE_TO_HOST))!=ACL_SUCCESS)return recovery();
     if(f39identity::digest(lease.host[i],ts[i]->storage_bytes)!=expected)return recovery();
    }
    replaced[i]=lease.device[i];
   }
  }
  if(cfg.mode!=F39_OBSERVE){std::lock_guard<std::mutex> lock(s.mutex);++s.substituted;s.input_device_readback_verified=s.selected_cache!=nullptr;}
 }
 {
  std::ostringstream p;p<<"{\"api\":\"aclnnMatmul\",\"generation\":"<<generation<<",\"hit\":"<<s.observation_hit<<",\"formal_order_proven\":true,\"slots\":[0,1,2],\"input_mask\":"<<cfg.input_mask<<",\"tensors\":[";
  for(int i=0;i<3;++i){if(i)p<<',';p<<"{\"shape\":["<<ts[i]->view_shape[0]<<','<<ts[i]->view_shape[1]<<"],\"strides\":["<<ts[i]->strides[0]<<','<<ts[i]->strides[1]
   <<"],\"offset\":"<<ts[i]->view_offset<<",\"storage_bytes\":"<<ts[i]->storage_bytes<<",\"original_address\":"<<reinterpret_cast<uintptr_t>(addresses[i])
   <<",\"run_address\":"<<reinterpret_cast<uintptr_t>(replaced[i])<<'}';}p<<"]}";
  std::lock_guard<std::mutex> lock(s.mutex);s.proof_json=p.str();
 }
 const aclnnStatus result=real_run(cache,workspace,stream,replaced);
 record(&State::run_status,result);
 if(record(&State::post_sync,sync_stream(stream))!=ACL_SUCCESS)return recovery();
 // Read the entire ORIGINAL input allocations again after the real Run. The
 // temporary allocations never stand in for protection of the live inputs.
 for(int i=0;i<2;++i) {
  if(record(&State::copy_status,copy(lease.host[i],ts[i]->storage_bytes,addresses[i],ts[i]->storage_bytes,ACL_MEMCPY_DEVICE_TO_HOST))!=ACL_SUCCESS)return recovery();
  const auto hash=f39identity::digest(lease.host[i],ts[i]->storage_bytes);
  bool unchanged;
  {std::lock_guard<std::mutex> lock(s.mutex);s.storage_after[i]=hash;unchanged=hash==s.storage_before[i];}
  if(!unchanged)return recovery();
 }
 if(result==ACL_SUCCESS) {
  if(host_alloc(&lease.host[2],ts[2]->storage_bytes)!=ACL_SUCCESS) {
   if(!free_lease(lease))return recovery();
   std::lock_guard<std::mutex> lock(s.mutex);s.enabled=false;fail(s,F39_BUSY);return 1;
  }
  if(record(&State::copy_status,copy(lease.host[2],ts[2]->storage_bytes,addresses[2],ts[2]->storage_bytes,ACL_MEMCPY_DEVICE_TO_HOST))!=ACL_SUCCESS)return recovery();
  const auto counts=fault39::observe(lease.host[2],*ts[2]);
  const auto output_hash=f39identity::digest(lease.host[2],ts[2]->storage_bytes);
  std::lock_guard<std::mutex> lock(s.mutex);s.counts=counts;s.output_sha256=output_hash;++s.completed;s.status="COMPLETED";
  if(counts.nan || counts.posinf || counts.neginf)s.enabled=false;
 } else {std::lock_guard<std::mutex> lock(s.mutex);s.enabled=false;s.status="RUN_FAILED";}
 if(record(&State::free_status,free_lease(lease)?0:-1)!=0)return recovery();
 return result;
}
aclnnStatus run(Cache* cache,void* workspace,aclrtStream stream,const std::vector<void*>& addresses) {
 const bool ready=resolve(); State& s=state();Registered r{};Fault39ConfigV1 cfg{};uint64_t generation=0;bool selected=false;
 {
  std::lock_guard<std::mutex> lock(s.mutex);++s.run_calls;
  if(ready && api_info)s.last_run_api=api_info();
  if(ready && s.last_run_api.compare(0,12,"aclnnMatmul_")==0) {
   const auto entry=s.registry.find(cache);int device=-1;const int rank=actual_rank();
   if(entry!=s.registry.end() && entry->second.cache_epoch &&
      s.cache_epochs[cache]==entry->second.cache_epoch && get_device(&device)==ACL_SUCCESS &&
      rank==entry->second.rank && device==entry->second.device && proof(entry->second,addresses,512ull*1024*1024)) {
    ++s.independent_runs;const auto& actual=entry->second;
    const auto signature=fault39::signature(actual.metadata,rank,device);
    const auto identity=std::make_pair(cache,actual.cache_epoch);
    auto found=s.independent_candidates.find(signature);
    const bool known=found!=s.independent_candidates.end() && found->second.caches.count(identity);
    if(known || s.independent_cache_count<8192) {
     auto& observed=s.independent_candidates[signature];observed.metadata=actual.metadata;observed.rank=rank;observed.device=device;
     if(observed.caches.insert(identity).second)++s.independent_cache_count;
     ++observed.observations;++observed.cache_observations[identity];
    } else s.independent_complete=false;
   }
  }
  if(s.enabled && ready) {
   auto it=s.registry.find(cache);int device=-1;const int rank=actual_rank();
   const bool correct_api=s.last_run_api.compare(0,12,"aclnnMatmul_")==0;
   const bool serving_match=it!=s.registry.end() && serving_inputs_match(context,it->second.metadata,addresses);
   if(correct_api && context.owner==s.config.owner && context.serving.requested && !serving_match) {
    ++s.rejected;++s.serving_input_rejections;s.status="SERVING_INPUTS_MISMATCH";
   }
   if(correct_api && context.owner==s.config.owner && !context.ids.empty() && context.ids!="[]" &&
      it!=s.registry.end() && it->second.cache_epoch && s.cache_epochs[cache]==it->second.cache_epoch &&
      get_device(&device)==ACL_SUCCESS && device==s.config.device && rank==s.config.rank &&
      it->second.device==device && it->second.rank==rank && serving_match &&
      proof(it->second,addresses,s.config.max_bytes)) {
    r=it->second;const std::string sig=fault39::signature(r.metadata,rank,device);
    if(sig.size()<sizeof(s.config.target_signature)) {
     auto& candidate=s.candidates[sig];candidate.metadata=r.metadata;candidate.rank=rank;candidate.device=device;
     candidate.caches.insert({cache,r.cache_epoch});++candidate.cache_observations[{cache,r.cache_epoch}];++candidate.observations;
     if(s.config.mode!=F39_DISCOVER && sig==s.config.target_signature) {
      if(s.selected_cache && (cache!=s.selected_cache || r.cache_epoch!=s.selected_epoch)) {
       ++s.rejected;s.status="EXACT_CACHE_NOT_SELECTED";
      }
      else if(!s.selected_cache && candidate.caches.size()!=1){s.enabled=false;fail(s,F39_AMBIGUOUS_TARGET);}
      else if(context.owner!=s.config.owner || context.ids.empty() || context.ids=="[]") {
       ++s.rejected;s.status="UNPROVEN_REQUEST_MAPPING";
      } else if(s.inflight) {++s.rejected;}
      else if(!s.config.max_hits || s.hits<s.config.max_hits) {
       cfg=s.config;generation=s.generation;s.actual_ids=context.ids;
       // A new hit must never inherit successful evidence from an older Run.
       s.counts={};s.original[0]={};s.original[1]={};s.temporary[0]={};s.temporary[1]={};
       s.changed[0]=s.changed[1]=s.unchanged[0]=s.unchanged[1]=0;
       s.self_verified=s.columns_verified=s.input_device_readback_verified=false;s.proof_json="{}";
       s.serving_live_inputs_verified=context.serving.requested && context.serving.valid && serving_match;
       s.storage_before[0].clear();s.storage_before[1].clear();s.storage_after[0].clear();s.storage_after[1].clear();s.output_sha256.clear();
       s.pre_sync=s.run_status=s.post_sync=s.free_status=s.copy_status=-2147483647;
       s.error=0;s.status="INFLIGHT";
       s.observation_owner=cfg.owner;s.observation_ids=context.ids;s.observation_generation=generation;s.observation_hit=s.hits+1;
       s.observation_cache=reinterpret_cast<uintptr_t>(cache);s.observation_epoch=r.cache_epoch;
       selected=true;++s.inflight;++s.hits;
       if(s.config.max_hits && s.hits>=s.config.max_hits)s.enabled=false;
      }
     }
    }
   }
  }
 }
 if(selected)return run_selected(cache,workspace,stream,addresses,r,cfg,generation);
 return real_run?real_run(cache,workspace,stream,addresses):1;
}
int stop(const char* owner,uint64_t generation,uint32_t timeout,bool clean) {
 State& s=state();std::unique_lock<std::mutex> lock(s.mutex);
 if(!owner || s.config.owner!=std::string(owner))return F39_OWNER_MISMATCH;
 if(s.generation!=generation)return F39_GENERATION_MISMATCH;
 s.enabled=false;
 if(!s.cv.wait_for(lock,std::chrono::milliseconds(timeout),[&]{return s.inflight==0;}))return fail(s,F39_DRAIN_TIMEOUT);
 if(!s.retained.empty()) {
  // Explicit clean is the only operation allowed to retry a failed lease recovery.
  if(!clean)return fail(s,F39_RECOVERY_FAILED);
  for(auto& lease:s.retained) {
   int32_t device=-1;aclrtContext current=nullptr;
   if(std::this_thread::get_id()!=lease.thread || get_device(&device)!=ACL_SUCCESS || device!=lease.device_id ||
      get_context(&current)!=ACL_SUCCESS || current!=lease.context || sync_stream(lease.stream)!=ACL_SUCCESS || !free_lease(lease))return fail(s,F39_RECOVERY_FAILED);
  }
  s.retained.clear();
 }
 s.error=0;s.status=clean?"CLEAN":"DISABLED";
 if(clean){s.config={};s.candidates.clear();s.hits=0;s.counts={};s.actual_ids="[]";
  s.selected_cache=nullptr;s.selected_epoch=s.observation_cache=s.observation_epoch=0;
  for(int i=0;i<2;++i){s.original[i]={};s.temporary[i]={};s.changed[i]=s.unchanged[i]=0;}
  s.proof_json="{}";s.observation_ids="[]";s.observation_generation=0;s.observation_hit=0;s.observation_owner.clear();
  s.self_verified=s.columns_verified=false;
  s.serving_live_inputs_verified=s.input_device_readback_verified=false;
  s.storage_before[0].clear();s.storage_before[1].clear();s.storage_after[0].clear();s.storage_after[1].clear();s.output_sha256.clear();
  s.pre_sync=s.copy_status=s.run_status=s.post_sync=s.free_status=-2147483647;
 }
 return 0;
}
}

extern "C" uint32_t fault39_abi_version_v1(){return 1;}
extern "C" const char* fault39_build_profile_v1(){return f39profile::id;}
extern "C" int fault39_bind_worker_rank_v1(int32_t rank) {
 if(rank<0)return F39_INVALID_CONFIG;int32_t expected=-1;
 if(worker_rank.compare_exchange_strong(expected,rank)) {
  // vLLM initializes its model before its first extension RPC. Metadata
  // captured during disabled warmup belongs to this same process/device;
  // attach the now verified immutable worker rank to those unbound records.
  State& s=state();std::lock_guard<std::mutex> lock(s.mutex);
  for(auto& item:s.registry)if(item.second.rank<0)item.second.rank=rank;
  return 0;
 }
 return expected==rank?0:F39_OWNER_MISMATCH;
}
extern "C" int fault39_set_context_v1(const char* owner,const char* ids) {
 context={};
 if(!owner || !*owner || strnlen(owner,64)>=64 || !ids)return F39_INVALID_CONFIG;
 const size_t n=strnlen(ids,4097);if(n<4 || n>4096 || ids[0]!='[' || ids[n-1]!=']')return F39_INVALID_CONFIG;
 // Accept only a nonempty JSON array of nonempty strings. No object, null or [] can assert mapping.
 size_t at=1,count=0;auto ws=[&](){while(at<n && (ids[at]==' ' || ids[at]=='\t' || ids[at]=='\n' || ids[at]=='\r'))++at;};
 while(at<n-1) {
  ws();if(ids[at++]!='"')return F39_INVALID_CONFIG;size_t chars=0;bool closed=false;
  while(at<n-1){unsigned char c=ids[at++];if(c=='"'){closed=true;break;}if(c<32)return F39_INVALID_CONFIG;
   if(c=='\\'){if(at>=n-1)return F39_INVALID_CONFIG;const char e=ids[at++];
    if(e=='u'){for(int i=0;i<4;++i){if(at>=n-1)return F39_INVALID_CONFIG;const char h=ids[at++];if(!((h>='0'&&h<='9')||(h>='a'&&h<='f')||(h>='A'&&h<='F')))return F39_INVALID_CONFIG;}}
    else if(!std::strchr("\"\\/bfnrt",e))return F39_INVALID_CONFIG;
   }++chars;
  }
  if(!closed || !chars)return F39_INVALID_CONFIG;++count;ws();if(at==n-1)break;
  if(ids[at++]!=',')return F39_INVALID_CONFIG;ws();if(at>=n-1)return F39_INVALID_CONFIG;
 }
 if(!count || at!=n-1)return F39_INVALID_CONFIG;
 try{context.owner=owner;context.ids=ids;return 0;}catch(...){context={};return F39_BUSY;}
}
extern "C" void fault39_clear_context_v1(){context={};}
extern "C" int fault39_set_serving_inputs_v1(const char* owner,const uint64_t* values,size_t count) {
 return bind_serving_inputs(context,owner,values,count);
}
int configure_cache(const Fault39ConfigV1* cfg,uint64_t* generation,uint64_t cache_address,uint64_t cache_epoch) {
 if(!cfg || !generation)return F39_INVALID_CONFIG;
 if(cfg->abi_version!=1 || cfg->struct_size!=sizeof(*cfg))return F39_ABI_MISMATCH;
 if(cfg->mode<1 || cfg->mode>4 || cfg->pattern>4 || (cfg->exponent!=-1 && cfg->exponent!=58) ||
    cfg->rank<0 || cfg->device<0 || !cfg->max_hits || !cfg->max_bytes || cfg->max_bytes>uint64_t(SIZE_MAX) || cfg->reserved ||
    cfg->input_mask<1 || cfg->input_mask>3 || (cfg->mode!=F39_LOSSLESS && cfg->input_mask!=3) ||
    !cfg->owner[0] || !std::memchr(cfg->owner,0,sizeof(cfg->owner)) ||
    !std::memchr(cfg->target_signature,0,sizeof(cfg->target_signature)))return F39_INVALID_CONFIG;
 if(!resolve())return F39_UNSUPPORTED_RUNTIME;
 const char* dev=std::getenv("VLLM_SERVER_DEV_MODE");
 if(!dev || std::strcmp(dev,"1") || actual_rank()<0)return F39_UNSUPPORTED_RUNTIME;
 const bool exact=cache_address!=0 || cache_epoch!=0;
 if(exact && (!cache_address || !cache_epoch || cfg->mode==F39_DISCOVER || cfg->max_hits!=1 ||
    context.owner!=cfg->owner || context.ids.empty() || context.ids=="[]" ||
    !context.serving.requested || !context.serving.valid))return F39_INVALID_CONFIG;
 State& s=state();std::lock_guard<std::mutex> lock(s.mutex);
 if(s.config.owner[0] && std::strcmp(cfg->owner,s.config.owner))return F39_OWNER_MISMATCH;
 if(s.enabled || s.inflight)return F39_BUSY;
 if(!s.retained.empty())return fail(s,F39_RECOVERY_FAILED);
 Fault39ConfigV1 next=*cfg;
 if(cfg->mode==F39_DISCOVER)s.candidates.clear();
 if(cfg->mode!=F39_DISCOVER) {
  auto it=s.candidates.find(cfg->target_signature);
  if(it==s.candidates.end())return fail(s,F39_TARGET_NOT_FOUND);
  if(!exact && (it->second.caches.size()!=1 || it->second.observations<2))return fail(s,F39_AMBIGUOUS_TARGET);
  const auto cache_identity=exact?std::make_pair(reinterpret_cast<Cache*>(cache_address),cache_epoch):*it->second.caches.begin();
  const auto seen=it->second.cache_observations.find(cache_identity);
  if(exact && (seen==it->second.cache_observations.end() || seen->second<2))return fail(s,F39_TARGET_NOT_FOUND);
  if(s.cache_epochs[cache_identity.first]!=cache_identity.second)return fail(s,F39_TARGET_NOT_FOUND);
  const auto registered=s.registry.find(cache_identity.first);
  if(registered==s.registry.end() || registered->second.cache_epoch!=cache_identity.second ||
     fault39::signature(registered->second.metadata,registered->second.rank,registered->second.device)!=cfg->target_signature)
   return fail(s,F39_TARGET_NOT_FOUND);
  if(it->second.rank!=cfg->rank || it->second.device!=cfg->device)return fail(s,F39_TARGET_NOT_FOUND);
  if(!fault39::matmul_valid(it->second.metadata,cfg->max_bytes))return fail(s,F39_INVALID_CONFIG);
  if(next.exponent==-1){next.exponent=fault39::auto_exponent(it->second.metadata.self.view_shape[1]);if(next.exponent<0)return fail(s,F39_NO_OVERFLOW_SOLUTION);}
 }
 try{s.retained.reserve(1);}catch(...){return fail(s,F39_BUSY);}
 s.config=next;++s.generation;*generation=s.generation;s.enabled=true;s.hits=0;s.error=0;s.counts={};s.status="ARMED";
 s.selected_cache=exact?reinterpret_cast<Cache*>(cache_address):nullptr;s.selected_epoch=cache_epoch;
 s.observation_cache=s.observation_epoch=0;
 for(int i=0;i<2;++i){s.original[i]={};s.temporary[i]={};s.changed[i]=s.unchanged[i]=0;}
 s.proof_json="{}";s.actual_ids="[]";s.observation_ids="[]";s.observation_generation=0;s.observation_hit=0;s.observation_owner.clear();
 s.self_verified=s.columns_verified=s.input_device_readback_verified=false;
 s.serving_live_inputs_verified=false;
 s.storage_before[0].clear();s.storage_before[1].clear();s.storage_after[0].clear();s.storage_after[1].clear();s.output_sha256.clear();
 s.pre_sync=s.copy_status=s.run_status=s.post_sync=s.free_status=-2147483647;return 0;
}
extern "C" int fault39_configure_v1(const Fault39ConfigV1* cfg,uint64_t* generation) {
 return configure_cache(cfg,generation,0,0);
}
// Additive exact identity API. The v1 ABI and its unique-signature gate remain.
// Address is never dereferenced; a real typed cache registration/epoch is required.
extern "C" int fault39_configure_cache_v2(const Fault39ConfigV1* cfg,uint64_t cache_address,uint64_t cache_epoch,uint64_t* generation) {
 if(!cache_address || !cache_epoch)return F39_INVALID_CONFIG;
 return configure_cache(cfg,generation,cache_address,cache_epoch);
}
extern "C" int fault39_disable_v1(const char* owner,uint64_t generation,uint32_t timeout){return stop(owner,generation,timeout,false);}
extern "C" int fault39_clean_v1(const char* owner,uint64_t generation,uint32_t timeout){return stop(owner,generation,timeout,true);}
// An explicitly scoped serving continuation may pause discovery between real
// scheduler requests. Resume the SAME disabled discovery generation; configure
// still always clears inventory and injection cannot use this entry point.
extern "C" int fault39_resume_discovery_v1(const char* owner,uint64_t generation) {
 if(!owner || !resolve())return F39_UNSUPPORTED_RUNTIME;
 State& s=state();std::lock_guard<std::mutex> lock(s.mutex);
 if(s.config.owner!=std::string(owner))return F39_OWNER_MISMATCH;
 if(s.generation!=generation)return F39_GENERATION_MISMATCH;
 if(s.config.mode!=F39_DISCOVER || s.enabled || s.inflight || !s.retained.empty())return F39_BUSY;
 if(context.owner!=owner || context.ids.empty() || context.ids=="[]")return F39_INVALID_CONFIG;
 s.enabled=true;s.status="DISCOVERY_RESUMED";return 0;
}
extern "C" int fault39_snapshot_v1(char* out,size_t capacity) {
 try {
  const bool ready=resolve();State& s=state();std::lock_guard<std::mutex> lock(s.mutex);std::ostringstream j;
  j<<"{\"abi_version\":1,\"runtime_loaded\":true,\"runtime_ready\":"<<(ready?"true":"false")<<",\"runtime_path\":"<<quote(runtime_path)
   <<",\"build_profile_sha256\":"<<quote(f39profile::id)<<",\"cann_build_version\":"<<quote(f39profile::version)
   <<",\"runtime_identity\":"<<quote(ready?"MATCHED_BUILD":"MISSING_OR_MISMATCHED_LIBRARIES")
   <<",\"owner\":"<<quote(s.config.owner)<<",\"generation\":"<<s.generation<<",\"enabled\":"<<(s.enabled?"true":"false")
   <<",\"mode\":"<<s.config.mode<<",\"exponent\":"<<s.config.exponent<<",\"pattern\":"<<s.config.pattern<<",\"input_mask\":"<<s.config.input_mask
   <<",\"rank\":"<<s.config.rank<<",\"device\":"<<s.config.device<<",\"inflight\":"<<s.inflight<<",\"leases\":"<<(s.retained.size()+s.inflight)
   <<",\"status\":"<<quote(s.status)<<",\"last_error\":"<<s.error<<",\"target_signature\":"<<quote(s.config.target_signature)
   <<",\"interposed_run_calls\":"<<s.run_calls<<",\"last_run_api\":"<<quote(s.last_run_api)<<",\"typed_metadata_calls\":"<<s.metadata_calls<<",\"hits\":"<<s.hits
   <<",\"substituted\":"<<s.substituted<<",\"completed\":"<<s.completed<<",\"rejected\":"<<s.rejected
   <<",\"serving_input_rejections\":"<<s.serving_input_rejections
   <<",\"exact_cache_api_version\":2,\"selected_cache_address\":"<<reinterpret_cast<uintptr_t>(s.selected_cache)<<",\"selected_cache_epoch\":"<<s.selected_epoch
   <<",\"independent_eligible_runs\":"<<s.independent_runs<<",\"independent_observer_complete\":"<<(s.independent_complete?"true":"false")
   <<",\"independent_observer_max_bytes\":536870912,\"independent_observer_max_caches\":8192"
   <<",\"actual_request_ids_json\":"<<quote(s.actual_ids)<<",\"worker_rank\":"<<actual_rank()<<",\"observer\":"<<counts_json(s.counts)
   <<",\"last_observation\":{\"generation\":"<<s.observation_generation<<",\"owner\":"<<quote(s.observation_owner)<<",\"hit\":"<<s.observation_hit
   <<",\"cache_address\":"<<s.observation_cache<<",\"cache_epoch\":"<<s.observation_epoch
   <<",\"exponent\":"<<s.config.exponent<<",\"pattern\":"<<s.config.pattern
   <<",\"self_constant_positive_verified\":"<<(s.self_verified?"true":"false")<<",\"per_column_sign_counts_verified\":"<<(s.columns_verified?"true":"false")
   <<",\"serving_live_inputs_verified\":"<<(s.serving_live_inputs_verified?"true":"false")
   <<",\"amplitude_equal\":"<<((s.config.mode==F39_INJECT && s.temporary[0].total && s.temporary[1].total &&
       s.temporary[0].finite==s.temporary[0].total && s.temporary[1].finite==s.temporary[1].total &&
       !s.temporary[0].zero && !s.temporary[1].zero && s.temporary[0].minimum==s.temporary[0].maximum &&
       s.temporary[0].maximum==s.temporary[1].maximum && s.temporary[1].minimum==-s.temporary[1].maximum)?"true":"false")
   <<",\"theoretical_output_exact_zero\":"<<((s.config.mode==F39_INJECT && s.self_verified && s.columns_verified && s.temporary[0].total && s.temporary[1].total &&
       s.temporary[0].positive==s.temporary[0].total && s.temporary[1].positive==s.temporary[1].negative &&
       s.temporary[1].positive+s.temporary[1].negative==s.temporary[1].total &&
       s.temporary[0].minimum==s.temporary[0].maximum && s.temporary[0].maximum==s.temporary[1].maximum &&
       s.temporary[1].minimum==-s.temporary[1].maximum)?"true":"false")
   <<",\"actual_request_ids_json\":"<<quote(s.observation_ids)<<",\"temporary_source\":\"host_constructed\",\"output_source\":\"post_run_device_readback\",\"input_device_readback_verified\":"<<(s.input_device_readback_verified?"true":"false")
   <<",\"original_self\":"<<counts_json(s.original[0])<<",\"original_mat2\":"<<counts_json(s.original[1])
   <<",\"temporary_self\":"<<counts_json(s.temporary[0])<<",\"temporary_mat2\":"<<counts_json(s.temporary[1])<<",\"output\":"<<counts_json(s.counts)
   <<",\"original_storage_sha256_before\":["<<quote(s.storage_before[0])<<','<<quote(s.storage_before[1])<<"]"
   <<",\"original_storage_sha256_after\":["<<quote(s.storage_after[0])<<','<<quote(s.storage_after[1])<<"]"
   <<",\"output_storage_sha256\":"<<quote(s.output_sha256)
   <<",\"changed\":["<<s.changed[0]<<','<<s.changed[1]<<"],\"unchanged\":["<<s.unchanged[0]<<','<<s.unchanged[1]<<"],\"proof\":"<<s.proof_json
   <<",\"pre_sync_status\":"<<s.pre_sync<<",\"copy_status\":"<<s.copy_status<<",\"run_status\":"<<s.run_status<<",\"post_sync_status\":"<<s.post_sync<<",\"free_status\":"<<s.free_status<<"},\"candidates\":[";
  bool first=true;for(const auto& entry:s.candidates){if(!first)j<<',';first=false;const auto& c=entry.second;
   j<<"{\"signature\":"<<quote(entry.first)<<",\"candidate_count\":"<<c.caches.size()<<",\"observations\":"<<c.observations
    <<",\"M\":"<<c.metadata.self.view_shape[0]<<",\"K\":"<<c.metadata.self.view_shape[1]<<",\"N\":"<<c.metadata.mat2.view_shape[1]
    <<",\"rank\":"<<c.rank<<",\"device\":"<<c.device<<",\"api\":\"aclnnMatmul\",\"dtype\":\"BF16\",\"format\":\"ND\",\"cache_identities\":[";
   bool first_cache=true;for(const auto& identity:c.caches){if(!first_cache)j<<',';first_cache=false;
    const auto current=s.cache_epochs.find(identity.first);
    const auto registered=s.registry.find(identity.first);
    const bool live=current!=s.cache_epochs.end() && current->second==identity.second &&
        registered!=s.registry.end() && registered->second.cache_epoch==identity.second;
    const auto seen=c.cache_observations.find(identity);
    j<<"{\"address\":"<<reinterpret_cast<uintptr_t>(identity.first)<<",\"epoch\":"<<identity.second
     <<",\"observations\":"<<(seen==c.cache_observations.end()?0:seen->second)
     <<",\"registration_live\":"<<(live?"true":"false")<<'}';}
   j<<"]}";}
  j<<"],\"independent_candidates\":[";first=true;
  for(const auto& entry:s.independent_candidates){if(!first)j<<',';first=false;const auto& c=entry.second;
   j<<"{\"signature\":"<<quote(entry.first)<<",\"rank\":"<<c.rank<<",\"device\":"<<c.device<<",\"cache_identities\":[";
   bool first_cache=true;for(const auto& identity:c.caches){if(!first_cache)j<<',';first_cache=false;
    const auto current=s.cache_epochs.find(identity.first);const auto registered=s.registry.find(identity.first);
    const bool live=current!=s.cache_epochs.end() && current->second==identity.second && registered!=s.registry.end() && registered->second.cache_epoch==identity.second;
    j<<"{\"address\":"<<reinterpret_cast<uintptr_t>(identity.first)<<",\"epoch\":"<<identity.second
     <<",\"observations\":"<<c.cache_observations.at(identity)<<",\"registration_live\":"<<(live?"true":"false")<<'}';}
   j<<"]}";}
  j<<"]}";const auto text=j.str();if(text.size()>=INT32_MAX)return F39_BUSY;
  const size_t need=text.size()+1;if(out && capacity>=need)std::memcpy(out,text.c_str(),need);return int(need);
 }catch(...){return F39_BUSY;}
}
extern "C" void fault39_gws_begin_v1(){capture={};capture.active=true;}
extern "C" void fault39_gws_abort_v1(){capture={};}
extern "C" int fault39_gws_commit_v1(const Fault39MetadataEnvelopeV1* envelope) {
 const Capture c=capture;capture={};State& s=state();std::lock_guard<std::mutex> lock(s.mutex);++s.metadata_calls;
 if(!envelope || envelope->magic!=FAULT39_METADATA_MAGIC || envelope->version!=1 || envelope->size!=sizeof(*envelope) ||
  envelope->payload_size!=sizeof(Fault39MatmulMetadata) || !envelope->payload)return F39_ABI_MISMATCH;
 const auto& m=*envelope->payload;
 if(!c.active || c.adds!=1 || !c.cache || c.params!=3 || !m.all_present || m.any_empty_storage || m.any_shared_storage ||
  c.tensors[0]!=m.self.tensor_ptr || c.tensors[1]!=m.mat2.tensor_ptr || c.tensors[2]!=m.out.tensor_ptr)return F39_INVALID_CONFIG;
 int device=-1;if(!get_device || get_device(&device)!=ACL_SUCCESS)return F39_UNSUPPORTED_RUNTIME;
 Registered r{};r.metadata=m;r.device=device;r.rank=actual_rank();r.cache_epoch=s.cache_epochs[c.cache];
 if(!r.cache_epoch)return F39_INVALID_CONFIG;
 std::copy(c.tensors,c.tensors+3,r.tensors);s.registry[c.cache]=r;return 0;
}
namespace op::internal {
void AddParamToBuf(const aclTensor* tensor){resolve();const bool nested=forwarding_param;forwarding_param=true;if(real_paramc)real_paramc(tensor);forwarding_param=nested;if(!nested && capture.active){const auto i=capture.params++;if(i<3)capture.tensors[i]=reinterpret_cast<uintptr_t>(tensor);}}
void AddParamToBuf(aclTensor* tensor){resolve();const bool nested=forwarding_param;forwarding_param=true;if(real_param)real_param(tensor);forwarding_param=nested;if(!nested && capture.active){const auto i=capture.params++;if(i<3)capture.tensors[i]=reinterpret_cast<uintptr_t>(tensor);}}
bool AddOpExecCache(OpExecCache* cache){resolve();const bool result=real_add?real_add(cache):false;{auto& s=state();std::lock_guard<std::mutex> lock(s.mutex);s.registry.erase(cache);s.cache_epochs[cache]=++s.next_cache_epoch;}if(capture.active){++capture.adds;capture.cache=cache;}return result;}
aclnnStatus OpExecCache::Run(void* workspace,const aclrtStream stream,const std::vector<void*>& addresses){try{return run(this,workspace,stream,addresses);}catch(...){auto& s=state();std::lock_guard<std::mutex> lock(s.mutex);s.enabled=false;s.error=F39_RECOVERY_FAILED;s.status="RECOVERY_FAILED";return 1;}}
}
