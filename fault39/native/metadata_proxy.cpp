// Minimal extraction of the CANN 8.5.1 historical lifecycle proxy.
#include "fault39_metadata.h"
#include <acl/acl.h>
#include <aclnn/acl_meta.h>
#include <aclnn/opdev/common_types.h>
#include <aclnnop/aclnn_matmul.h>
#include <dlfcn.h>
#include <limits>
#include <mutex>
#ifndef FAULT39_REAL_LIBRARY
#define FAULT39_REAL_LIBRARY "libopapi_real.so"
#endif

namespace {
uint64_t checked_bytes(int64_t elements,size_t size) {
 if(elements<0 || !size || uint64_t(elements)>UINT64_MAX/size) return 0;
 return uint64_t(elements)*size;
}
#include "metadata_capture.inc"
using Getws=aclnnStatus(*)(const aclTensor*,const aclTensor*,aclTensor*,int8_t,uint64_t*,aclOpExecutor**);
using Execute=aclnnStatus(*)(void*,uint64_t,aclOpExecutor*,aclrtStream);
using Begin=void(*)(); using Commit=int(*)(const Fault39MetadataEnvelopeV1*);
Getws getws=nullptr; Execute execute=nullptr; Begin begin=nullptr,abort_capture=nullptr; Commit commit=nullptr;
std::once_flag once;
void resolve() {
 // A split-opapi lookup can be handle-specific. Forward to this proxy's
 // explicitly renamed DT_NEEDED library, never another proxy in load order.
 void* real=dlopen(FAULT39_REAL_LIBRARY,RTLD_LAZY|RTLD_NOLOAD);
 if(real) {
  getws=reinterpret_cast<Getws>(dlsym(real,"aclnnMatmulGetWorkspaceSize"));
  execute=reinterpret_cast<Execute>(dlsym(real,"aclnnMatmul"));
  dlclose(real);
 }
 begin=reinterpret_cast<Begin>(dlsym(RTLD_DEFAULT,"fault39_gws_begin_v1"));
 abort_capture=reinterpret_cast<Begin>(dlsym(RTLD_DEFAULT,"fault39_gws_abort_v1"));
 commit=reinterpret_cast<Commit>(dlsym(RTLD_DEFAULT,"fault39_gws_commit_v1"));
}
}
extern "C" uint32_t fault39_metadata_proxy_abi_v1(){return 1;}
extern "C" aclnnStatus aclnnMatmulGetWorkspaceSize(const aclTensor* self,const aclTensor* mat2,
 aclTensor* out,int8_t cube_math_type,uint64_t* workspace_size,aclOpExecutor** executor) {
 std::call_once(once,resolve); if(!getws)return 1;
 const bool hooks=begin && commit && abort_capture;
 if(hooks)begin();
 const auto status=getws(self,mat2,out,cube_math_type,workspace_size,executor);
 if(!hooks)return status;
 Fault39MatmulMetadata m{};
 if(status!=ACL_SUCCESS || !capture_tensor_metadata(self,&m.self) ||
    !capture_tensor_metadata(mat2,&m.mat2) || !capture_tensor_metadata(out,&m.out)) {
  abort_capture(); return status;
 }
 m.cube_math_type=cube_math_type; m.all_present=1;
 const Fault39TensorMetadata* ts[]={&m.self,&m.mat2,&m.out};
 for(int i=0;i<3;++i) {
  if(!ts[i]->storage_ptr || !ts[i]->storage_addr || !ts[i]->storage_bytes)m.any_empty_storage=1;
  for(int j=i+1;j<3;++j)if(ts[i]->storage_ptr==ts[j]->storage_ptr)m.any_shared_storage=1;
 }
 const Fault39MetadataEnvelopeV1 envelope{FAULT39_METADATA_MAGIC,1,sizeof(Fault39MetadataEnvelopeV1),sizeof(m),0,&m};
 (void)commit(&envelope); return status;
}
extern "C" aclnnStatus aclnnMatmul(void* workspace,uint64_t workspace_size,aclOpExecutor* executor,aclrtStream stream) {
 std::call_once(once,resolve); return execute?execute(workspace,workspace_size,executor,stream):1;
}
