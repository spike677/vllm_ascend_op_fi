// Real ELF/loader boundary, explicit host-memory SDK substitutes. The production
// group DSO is loaded intact; this test never assigns private api.* or ready.
#include "batch_scale_group_runtime.h"
#include <acl/acl.h>
#include <aclnnop/aclnn_dynamic_quant_v2.h>
#include <cassert>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <iostream>
#include <link.h>
#include <string>
using namespace dcat_dq_group;
namespace {
template<class T>T function(void* handle,const char* name) {
    auto address=dlsym(handle,name);assert(address);return reinterpret_cast<T>(address);
}
void* load(const std::string& path,int flags) {
    void* result=dlopen(path.c_str(),flags);if(!result)std::cerr<<dlerror()<<'\n';assert(result);return result;
}
void* object(void* address){Dl_info info{};assert(dladdr(address,&info) && info.dli_fbase);return info.dli_fbase;}
}
int main(int argc,char** argv) {
    assert(argc==4);const std::string group_path=argv[1],root=argv[2],mode=argv[3];
    const bool success=mode=="no_cache" || mode=="with_cache";
    assert(success || mode=="missing_prepare" || mode=="missing_execute" || mode=="self" || mode=="foreign_owner" || mode=="foreign_cache");
    void* nn=load(root+"/libnnopbase.so",RTLD_NOW|RTLD_LOCAL);
    void* acl=load(root+"/libascendcl.so",RTLD_NOW|RTLD_LOCAL);
    const std::string vendor=mode=="self"?group_path:root+"/libopapi-"+mode+".so";
    void* original=mode=="self"?nullptr:load(vendor,RTLD_NOW|RTLD_GLOBAL);
    assert(setenv("DCAT_REAL_OPAPI",vendor.c_str(),1)==0);
    unsetenv("DCAT_DQ_V2_ENABLE");unsetenv("DCAT_DQ_GROUP_V1_ALLOWED");unsetenv("VLLM_SERVER_DEV_MODE");
    void* group=load(group_path,RTLD_NOW|RTLD_LOCAL);
    const auto prepare=function<decltype(&aclnnDynamicQuantV2GetWorkspaceSize)>(group,"aclnnDynamicQuantV2GetWorkspaceSize");
    const auto execute=function<decltype(&aclnnDynamicQuantV2)>(group,"aclnnDynamicQuantV2");
    const auto cache=function<bool (*)(const char*)>(group,"CanUsePTACache");
    const auto state=function<int (*)(RuntimeState*)>(group,"dcat_dqg_state_v1");
    const auto raw=function<int (*)(uint64_t,RawRecord*)>(group,"dcat_dqg_raw_v1");
    const auto group_owner=object(reinterpret_cast<void*>(prepare));
    assert(object(reinterpret_cast<void*>(execute))==group_owner);
    assert(object(reinterpret_cast<void*>(state))==group_owner);
    if(success) {
        // The original is global FIRST. A preemptible &export in the group can
        // resolve here even though dlsym(group, export) names the group's body.
        assert(dlsym(RTLD_DEFAULT,"aclnnDynamicQuantV2GetWorkspaceSize")==dlsym(original,"aclnnDynamicQuantV2GetWorkspaceSize"));
        assert(object(dlsym(original,"aclnnDynamicQuantV2GetWorkspaceSize"))!=group_owner);
        assert((dlsym(original,"CanUsePTACache")!=nullptr)==(mode=="with_cache"));
    }
    if(mode=="foreign_owner" || mode=="foreign_cache") {
        link_map* loaded=nullptr;assert(dlinfo(original,RTLD_DI_LINKMAP,&loaded)==0 && loaded);
        const char* symbol=mode=="foreign_owner"?"aclnnDynamicQuantV2GetWorkspaceSize":"CanUsePTACache";
        assert(reinterpret_cast<uintptr_t>(object(dlsym(original,symbol)))!=static_cast<uintptr_t>(loaded->l_addr));
    }
    assert(!cache("aclnnDynamicQuantV2"));
    assert(cache("unrelated_api")==(mode=="with_cache"));
    uint64_t bytes=0;aclOpExecutor* executor=nullptr;
    const auto status=prepare(reinterpret_cast<const aclTensor*>(1),nullptr,nullptr,ACL_INT8,
        reinterpret_cast<const aclTensor*>(2),reinterpret_cast<const aclTensor*>(3),nullptr,&bytes,&executor);
    assert(status==(success?aclnnStatus(0):aclnnStatus(161001)));
    if(success) {
        assert(!execute(nullptr,bytes,executor,reinterpret_cast<aclrtStream>(77)));
        assert(function<uint64_t (*)()>(original,"fixture_prepare_calls")()==1);
        assert(function<uint64_t (*)()>(original,"fixture_execute_calls")()==1);
        assert(function<uint64_t (*)()>(original,"fixture_cache_calls")()==(mode=="with_cache"?1u:0u));
    } else if(original) {
        auto counter=reinterpret_cast<uint64_t (*)()>(dlsym(original,"fixture_prepare_calls"));
        if(counter)assert(counter()==0);
        counter=reinterpret_cast<uint64_t (*)()>(dlsym(original,"fixture_execute_calls"));
        if(counter)assert(counter()==0);
    }
    RuntimeState result{};assert(state(&result)==Ok);
    assert(!result.core.installed && !result.core.pending_parent && !result.core.device_bytes && !result.core.host_bytes);
    assert(result.public_gws_entries==1 && result.public_execute_entries==(success?1u:0u));
    assert(result.census.native_prepares==(success?1u:0u) && result.census.native_executes==(success?1u:0u));
    assert(result.last_sequence==(success?5u:1u) && result.rejected_calls==(success?0u:1u));
    assert(!result.census.errors && !result.census.pending && !result.census.preparing && !result.census.executing);
    RawRecord row{};assert(raw(1,&row)==Ok && row.kind==1 && row.parent.device==0);
    assert(row.parent.input.address==(success?4096u:0u));
    if(success) {
        assert(raw(3,&row)==Ok && row.kind==2 && row.native.original_calls==1 && !row.native.status);
        assert(raw(5,&row)==Ok && row.kind==2 && row.native.original_calls==1 && !row.native.status);
        assert(!result.quarantined && !result.abi_exceptions && !result.overlap_entries);
    } else assert(result.last_error==Identity);
    std::cout<<"{\"case\":\""<<mode<<"\",\"status\":\"PASS_REAL_ELF_LOADER_CPU_MOCK\",\"actual_npu_calls\":0,"
        "\"mocked_sdk_functions\":true,\"native_prepare_calls\":"<<result.census.native_prepares
        <<",\"native_execute_calls\":"<<result.census.native_executes<<"}\n";
    assert(dlclose(group)==0);if(original)assert(dlclose(original)==0);assert(dlclose(acl)==0);assert(dlclose(nn)==0);
}
