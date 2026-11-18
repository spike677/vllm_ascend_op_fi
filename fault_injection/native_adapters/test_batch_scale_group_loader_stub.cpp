// CPU ELF loader fixtures only. Compile against the explicit mock SDK header;
// these exported functions never load a CANN runtime or touch an NPU.
#include <acl/acl.h>
#include <aclnnop/aclnn_dynamic_quant_v2.h>
#include <cassert>
#include <cstring>

#if defined(DCAT_DQ_LOADER_NN)
namespace {
uint64_t tensor_kind(const aclTensor* tensor){return reinterpret_cast<uint64_t>(tensor);}
}
extern "C" aclnnStatus aclGetDataType(const aclTensor* tensor,aclDataType* out) {
    *out=tensor_kind(tensor)==1?ACL_BF16:tensor_kind(tensor)==2?ACL_INT8:ACL_FLOAT;return 0;
}
extern "C" aclnnStatus aclGetFormat(const aclTensor*,aclFormat* out){*out=ACL_FORMAT_ND;return 0;}
extern "C" aclnnStatus aclGetViewOffset(const aclTensor*,int64_t* out){*out=0;return 0;}
extern "C" aclnnStatus aclGetViewShape(const aclTensor* tensor,int64_t** out,uint64_t* count) {
    *count=tensor_kind(tensor)==3?1:2;*out=new int64_t[*count];(*out)[0]=2;if(*count==2)(*out)[1]=4;return 0;
}
extern "C" aclnnStatus aclGetStorageShape(const aclTensor* tensor,int64_t** out,uint64_t* count) {
    return aclGetViewShape(tensor,out,count);
}
extern "C" aclnnStatus aclGetViewStrides(const aclTensor* tensor,int64_t** out,uint64_t* count) {
    *count=tensor_kind(tensor)==3?1:2;*out=new int64_t[*count];(*out)[0]=*count==1?1:4;if(*count==2)(*out)[1]=1;return 0;
}
extern "C" aclnnStatus aclGetRawTensorAddr(const aclTensor* tensor,void** out) {
    *out=reinterpret_cast<void*>(tensor_kind(tensor)*4096);return 0;
}
extern "C" aclTensor* aclCreateTensor(const int64_t*,uint64_t,aclDataType,const int64_t*,int64_t,
    aclFormat,const int64_t*,uint64_t,void*){assert(false && "no group allocation in loader fixture");return nullptr;}
extern "C" aclnnStatus aclDestroyTensor(const aclTensor*){assert(false && "no group descriptor in loader fixture");return 1;}
#elif defined(DCAT_DQ_LOADER_ACL)
extern "C" aclError aclrtGetDevice(int32_t* out){*out=0;return 0;}
extern "C" aclError aclrtMalloc(void**,size_t,aclrtMemMallocPolicy){assert(false && "no device allocation");return 1;}
extern "C" aclError aclrtFree(void*){assert(false && "no device release");return 1;}
extern "C" aclError aclrtMemcpy(void*,size_t,const void*,size_t,aclrtMemcpyKind){assert(false && "no device copy");return 1;}
extern "C" aclError aclrtSynchronizeStream(aclrtStream){assert(false && "no device synchronization");return 1;}
#elif defined(DCAT_DQ_LOADER_FORWARDER)
extern "C" uint64_t fixture_prepare_calls();
extern "C" uint64_t fixture_dependency_anchor(){return fixture_prepare_calls();}
#elif defined(DCAT_DQ_LOADER_CACHE_ONLY)
extern "C" bool CanUsePTACache(const char*){return true;}
#else
namespace {uint64_t prepares{},executes{},cache_calls{};}
extern "C" uint64_t fixture_prepare_calls(){return prepares;}
extern "C" uint64_t fixture_execute_calls(){return executes;}
extern "C" uint64_t fixture_cache_calls(){return cache_calls;}
#if !defined(DCAT_DQ_LOADER_MISSING_PREPARE)
extern "C" aclnnStatus aclnnDynamicQuantV2GetWorkspaceSize(const aclTensor* x,const aclTensor* a,const aclTensor* b,
    int64_t dtype,const aclTensor* q,const aclTensor* s,const aclTensor* o,uint64_t* bytes,aclOpExecutor** executor) {
    assert(reinterpret_cast<uint64_t>(x)==1 && !a && !b && dtype==ACL_INT8
        && reinterpret_cast<uint64_t>(q)==2 && reinterpret_cast<uint64_t>(s)==3 && !o);
    ++prepares;*bytes=0;*executor=reinterpret_cast<aclOpExecutor*>(8192);return 0;
}
#endif
#if !defined(DCAT_DQ_LOADER_MISSING_EXECUTE)
extern "C" aclnnStatus aclnnDynamicQuantV2(void* workspace,uint64_t size,aclOpExecutor* executor,aclrtStream stream) {
    assert(!workspace && !size && reinterpret_cast<uint64_t>(executor)==8192 && reinterpret_cast<uint64_t>(stream)==77);
    ++executes;return 0;
}
#endif
#if defined(DCAT_DQ_LOADER_WITH_CACHE)
extern "C" bool CanUsePTACache(const char* name){assert(name && std::strcmp(name,"unrelated_api")==0);++cache_calls;return true;}
#endif
#endif
