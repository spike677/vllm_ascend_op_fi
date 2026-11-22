// CPU only against real SDK declarations. No device calls are made.
#include "postwrite.cpp"
#include <cassert>
#include <iostream>
static uint64_t stream_syncs=0, device_syncs=0;
extern "C" aclError aclrtGetDevice(int32_t* d) {*d=0;return ACL_SUCCESS;}
extern "C" aclError aclrtSynchronizeDevice() {++device_syncs;return ACL_SUCCESS;}
extern "C" aclError aclrtSynchronizeStream(aclrtStream s) {assert(s);++stream_syncs;return ACL_SUCCESS;}
extern "C" aclError aclrtMemcpy(void* dst,size_t capacity,const void* src,size_t count,aclrtMemcpyKind) {
    assert(capacity>=count);std::memcpy(dst,src,count);return ACL_SUCCESS;
}
int main() {
    uint16_t a[]={1,2,3,4}, b[]={5,6,7,8}; uint64_t dst=0,src=0,changed=0;
    assert(dcat_pw_open_stream_v2(a,8,0,11,2,0,&dst)==1);
    assert(dcat_pw_open_stream_v2(a,8,0,11,2,7,&dst)==0);
    assert(dcat_pw_open_stream_v2(b,8,0,11,2,7,&src)==0);
    uint64_t overlap=0;assert(dcat_pw_open_stream_v2(a+1,2,0,11,2,7,&overlap)==14);
    assert(dcat_pw_copy_region_v2(dst,src,11,3,&changed)==2);
    assert(dcat_pw_copy_region_v2(dst,src,11,2,&changed)==0 && changed==4);
    assert(std::equal(a,a+4,b));
    assert(dcat_pw_release(dst,11,2)==2);
    // A subsequent legitimate write must never be overwritten by stale recovery.
    a[0]=9;assert(dcat_pw_restore(dst,11,2)==5 && a[0]==9);
    a[0]=5;assert(dcat_pw_restore(dst,11,2)==0 && a[0]==1);
    assert(dcat_pw_release(dst,11,2)==0 && dcat_pw_release(src,11,2)==0);
    assert(dcat_pw_open_stream_v2(a,8,0,11,2,7,&dst)==0);
    assert(dcat_pw_open_stream_v2(b,8,0,11,2,8,&src)==0);
    assert(dcat_pw_copy_region_v2(dst,src,11,2,&changed)==2);
    assert(dcat_pw_release(dst,11,2)==0 && dcat_pw_release(src,11,2)==0);
    assert(leases.empty() && live_span_bytes==0 && stream_syncs>0 && device_syncs==0);
    std::cout<<"{\"status\":\"PASS_CPU_STREAM_OBJECT_REGIONS\",\"actual_npu_calls\":0}\n";
}
