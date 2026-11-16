// Included only by an explicitly opted-in build of the original v17 runtime.
// The original transform, semantic-slot checks, allocation and recovery paths
// remain in run_and_maybe_substitute. No model name or layer path is used.
#include <atomic>
#include <thread>
namespace {
struct Dcat20Call {
 uint32_t version=1, size=sizeof(Dcat20Call);
 uint64_t token=0;
 int32_t device=-1,dtype=-1;
 int64_t m=0,k=0;
 uint64_t bytes=0,input=0,weight=0,output=0;
 uint32_t metadata_abi=17,slot_proof=0;
};
using Dcat20Callback=uint32_t(*)(const Dcat20Call*);
std::mutex dcat20_owner_mutex;
Dcat20Callback dcat20_callback=nullptr;
std::thread::id dcat20_owner;
uint64_t dcat20_serial=0;
struct Dcat20Flight {
 Dcat20Call call;
 op::internal::OpExecCache* cache;
 void* workspace; aclrtStream stream;
 const std::vector<void*>* addresses;
 bool invoked=false;
};
thread_local Dcat20Flight* dcat20_flight=nullptr;
thread_local std::string dcat20_last="{}";

uint32_t dcat20_dispatch(op::internal::OpExecCache* cache,void* workspace,
                        aclrtStream stream,const std::vector<void*>& addresses) {
 pthread_once(&g_resolve_once,resolve_real);
 pthread_once(&g_control_once,initialize_control);
 Dcat20Callback callback=nullptr;
 {std::lock_guard<std::mutex> lock(dcat20_owner_mutex);
  if(dcat20_owner==std::this_thread::get_id())callback=dcat20_callback;}
 if(!callback || dcat20_flight || !g_abi_ok || !g_developer_mode_ok || !g_control)
  return run_and_maybe_substitute(cache,workspace,stream,addresses);
 RegisteredMatmulMetadata registered{}; bool found=false;
 {std::lock_guard<std::mutex> lock(registry_mutex());
  auto it=matmul_registry().find(cache);
  if(it!=matmul_registry().end() && it->second.api_kind==uint32_t(MatmulApiKind::kWeightNz))
   {registered=it->second;found=true;}}
 if(!found)return run_and_maybe_substitute(cache,workspace,stream,addresses);
 const auto proof=prove_matmul_storage_order(registered,addresses);
 const auto& x=registered.tensors.self;
 if(!proof.self_slot_proven || !proof.mat2_slot_proven || !proof.output_slot_proven ||
    x.format!=ACL_FORMAT_ND || (x.dtype!=ACL_BF16 && x.dtype!=ACL_FLOAT16) ||
    x.view_rank!=2 || x.storage_rank!=1 || x.stride_rank!=2 || !x.contiguous ||
    x.view_offset || x.strides[1]!=1 || x.strides[0]!=x.view_shape[1] ||
    x.storage_bytes==0 || x.storage_bytes>65536 || x.view_bytes!=x.storage_bytes ||
    registered.device<0 || registered.device>=64)
  return run_and_maybe_substitute(cache,workspace,stream,addresses);
 Dcat20Flight flight{};
 flight.call.token=++dcat20_serial;flight.call.device=registered.device;
 flight.call.dtype=x.dtype;flight.call.m=x.view_shape[0];flight.call.k=x.view_shape[1];
 flight.call.bytes=x.storage_bytes;flight.call.slot_proof=1;
 flight.call.input=reinterpret_cast<uintptr_t>(addresses[0]);
 flight.call.weight=reinterpret_cast<uintptr_t>(addresses[1]);
 flight.call.output=reinterpret_cast<uintptr_t>(addresses[2]);
 flight.cache=cache;flight.workspace=workspace;flight.stream=stream;flight.addresses=&addresses;
 dcat20_flight=&flight;
 uint32_t result=1;
 try{result=callback(&flight.call);}catch(...){result=1;}
 dcat20_flight=nullptr;
 // Never retry a callback failure; a call may already have launched work.
 if(!flight.invoked)return 1;
 return result;
}
}

extern "C" int dcat20_install_v1(Dcat20Callback callback) {
 if(!callback)return -1;
 pthread_once(&g_resolve_once,resolve_real);pthread_once(&g_control_once,initialize_control);
 std::lock_guard<std::mutex> lock(dcat20_owner_mutex);
 if(dcat20_callback || !g_control || !g_abi_ok || !g_developer_mode_ok ||
    __atomic_load_n(&g_control->enabled,__ATOMIC_ACQUIRE))return -2;
 dcat20_owner=std::this_thread::get_id();dcat20_callback=callback;return 0;
}
extern "C" int dcat20_remove_v1(Dcat20Callback callback) {
 std::lock_guard<std::mutex> lock(dcat20_owner_mutex);
 if(callback!=dcat20_callback || dcat20_owner!=std::this_thread::get_id() || dcat20_flight)return -1;
 if(g_control && __atomic_load_n(&g_control->enabled,__ATOMIC_ACQUIRE))return -2;
 dcat20_callback=nullptr;dcat20_owner=std::thread::id{};return 0;
}
extern "C" const char* dcat20_snapshot_v1(){return dcat20_last.c_str();}
extern "C" uint32_t dcat20_continue_v1(uint64_t token,uint32_t mode,uint32_t keep_bits) {
 auto* f=dcat20_flight;
 if(!f || f->call.token!=token || f->invoked || mode>2 || keep_bits>7)return 1;
 f->invoked=true;
 if(!mode)return run_and_maybe_substitute(f->cache,f->workspace,f->stream,*f->addresses);
 auto* c=g_control;
 if(!c || __atomic_load_n(&c->enabled,__ATOMIC_ACQUIRE))return 1;
 c->mode=mode==1?FAULT20_MODE_LOSSLESS_D2D:FAULT20_MODE_MANTISSA_TRUNCATE;
 c->target_api=FAULT20_TARGET_API_MATMUL_WEIGHT_NZ;c->target_device=f->call.device;
 c->target_m=f->call.m;c->target_k=f->call.k;c->target_dtype=f->call.dtype;
 c->target_call_index=1;c->max_hits=1;c->max_hits_per_device=1;c->max_hits_total=1;
 c->selection_mode=FAULT20_SELECTION_BOUNDED;c->hit_scope=FAULT20_HIT_SCOPE_ARM_LIFETIME;
 c->keep_mantissa_bits=keep_bits;c->max_bytes=f->call.bytes;
 c->target_device_mask=1ULL<<uint32_t(f->call.device);
 c->candidate_count=0;c->hit_count=0;c->last_status=0;
 for(size_t i=0;i<FAULT20_MAX_DEVICES;++i){c->candidate_count_per_device[i]=0;c->hit_count_per_device[i]=0;c->last_status_per_device[i]=0;}
 ++c->generation;
 __atomic_store_n(&c->enabled,1U,__ATOMIC_RELEASE);
 uint32_t result=1;
 try{result=run_and_maybe_substitute(f->cache,f->workspace,f->stream,*f->addresses);}
 catch(...){c->last_status_per_device[f->call.device]=-100;result=1;}
 __atomic_store_n(&c->enabled,0U,__ATOMIC_RELEASE);
 dcat20_last="{\"call_id\":\""+std::to_string(token)+"\",\"hits\":"+std::to_string(c->hit_count)+
  ",\"native_status\":"+std::to_string(c->last_status_per_device[f->call.device])+
  ",\"run_status\":"+std::to_string(result)+",\"enabled\":false,\"mode\":"+std::to_string(mode)+"}";
 return result;
}
