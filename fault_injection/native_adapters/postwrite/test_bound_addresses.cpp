// Compile against SDK headers; the only ACL definitions are inert CPU stubs.
#include "postwrite.cpp"
#include <cassert>
#include <iostream>
extern "C" aclError aclrtGetDevice(int32_t*) {assert(false);return ACL_SUCCESS;}
extern "C" aclError aclrtSynchronizeDevice() {assert(false);return ACL_SUCCESS;}
extern "C" aclError aclrtMemcpy(void*,size_t,const void*,size_t,aclrtMemcpyKind) {assert(false);return ACL_SUCCESS;}
int main() {
    uint64_t address=999;
    assert(dcat_pw_linked_address_v1(0,nullptr)==1);
    assert(dcat_pw_linked_address_v1(0,&address)==0 && address==reinterpret_cast<uintptr_t>(&aclrtGetDevice));
    assert(dcat_pw_linked_address_v1(1,&address)==0 && address==reinterpret_cast<uintptr_t>(&aclrtSynchronizeDevice));
    assert(dcat_pw_linked_address_v1(2,&address)==0 && address==reinterpret_cast<uintptr_t>(&aclrtMemcpy));
    assert(dcat_pw_linked_address_v1(3,&address)==2 && address==0);
    assert(leases.empty() && serial==0 && live_span_bytes==0);
    std::cout<<"{\"status\":\"PASS_CPU_EXISTING_LINKED_ADDRESSES_ONLY\",\"actual_npu_calls\":0}\n";
}
