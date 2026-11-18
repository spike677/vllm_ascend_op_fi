// Actual production cache getter with SDK declarations, no operator execution.
#define FAULT26_CUSTOM_PROXY_ONLY 1
#include "../../fault20_fast_validation/source/fault26_readonly_trace_v1/fault26_opapi_trace_proxy.cpp"
#include <cassert>
#include <iostream>

static aclnnStatus cpu_execute(void*,uint64_t,aclOpExecutor*,aclrtStream) {
    assert(false);return 1;
}
static aclnnStatus cpu_gws(const aclTensor*,const aclTensor*,const aclTensor*,const aclTensor*,
    double,const aclTensor*,const aclTensor*,const aclTensor*,uint64_t*,aclOpExecutor**) {
    assert(false);return 1;
}
int main() {
    uint64_t address=999;
    assert(dcat26_custom_cached_address_v1(0,nullptr)==1);
    assert(dcat26_custom_cached_address_v1(0,&address)==3 && address==0);
    assert(g_real_add_rms_norm_bias==nullptr && g_real_add_rms_norm_bias_gws==nullptr);
    g_real_add_rms_norm_bias=cpu_execute;g_real_add_rms_norm_bias_gws=cpu_gws;
    assert(dcat26_custom_cached_address_v1(0,&address)==0 && address==reinterpret_cast<uintptr_t>(cpu_execute));
    assert(dcat26_custom_cached_address_v1(1,&address)==0 && address==reinterpret_cast<uintptr_t>(cpu_gws));
    assert(dcat26_custom_cached_address_v1(2,&address)==2 && address==0);
    assert(g_real_add_rms_norm_bias==cpu_execute && g_real_add_rms_norm_bias_gws==cpu_gws);
    std::cout<<"{\"status\":\"PASS_CPU_EXISTING_CUSTOM_CACHE_ONLY\",\"actual_npu_calls\":0}\n";
}
