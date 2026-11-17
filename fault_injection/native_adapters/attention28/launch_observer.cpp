// Independent runtime-boundary observer. Never selects a key or edits arguments.
// Compile against this process's installed CANN headers; no copied ABI structs.
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include <runtime/kernel.h>
#include <runtime/dev.h>
#include <sys/uio.h>
#include <sys/syscall.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <dlfcn.h>
#include <chrono>
#include <cstring>
#include <cstdlib>
#include <cstdint>
#include <atomic>
#include <mutex>
#include <map>
#include <sstream>
#include <string>
#include <vector>
#include <stdexcept>

namespace {
std::mutex gate;
std::atomic<uint64_t> serial{0};
struct Binary { std::string file; uint64_t generation=0; };
std::map<void*, Binary> registered;
thread_local std::string token;
thread_local bool nested=false;
uint64_t now() { return std::chrono::duration_cast<std::chrono::nanoseconds>(
    std::chrono::steady_clock::now().time_since_epoch()).count(); }
bool safe(const std::string& value) {
    return value.size()==32 && value.find_first_not_of("0123456789abcdef")==std::string::npos;
}
std::string directory() {
    const char* p=std::getenv("DCAT_ATTN_OBSERVER_DIR");
    if(!p)return {};
    std::string s(p);
    if(s.rfind("/evidence/",0)!=0 || s.find("/../")!=std::string::npos ||
       s.find('"')!=std::string::npos || s.find('\\')!=std::string::npos || s.find('\n')!=std::string::npos)return {};
    struct stat st{};
    if(lstat(s.c_str(),&st) || !S_ISDIR(st.st_mode) || st.st_uid!=getuid())return {};
    return s;
}
std::string hex(const void* ptr, size_t size) {
    const auto* p=static_cast<const uint8_t*>(ptr);const char* a="0123456789abcdef";
    std::string out;out.reserve(size*2);
    for(size_t i=0;i<size;++i){out+=a[p[i]>>4];out+=a[p[i]&15];}return out;
}
bool read_self(const void* p, size_t n, std::vector<uint8_t>& bytes) {
    if(!p || !n || n>128*1024*1024)return false;
    bytes.resize(n);iovec local{bytes.data(),n},remote{const_cast<void*>(p),n};
    if(process_vm_readv(getpid(),&local,1,&remote,1,0)!=static_cast<ssize_t>(n)){bytes.clear();return false;}
    return true;
}
bool write_file(const std::string& path, const void* data, size_t n, bool append=false) {
    int fd=open(path.c_str(),O_WRONLY|O_CREAT|O_CLOEXEC|O_NOFOLLOW|(append?O_APPEND:O_EXCL),0600);
    if(fd<0)return false;
    const auto* p=static_cast<const char*>(data);size_t done=0;
    while(done<n){ssize_t rc=write(fd,p+done,n-done);if(rc<=0){close(fd);return false;}done+=rc;}
    return close(fd)==0;
}
bool emit(const std::string& body) {
    const auto dir=directory();if(dir.empty())return false;
    std::ostringstream line;line<<"{\"schema\":\"demoncat.attention.runtime_observer.v1\",\"pid\":"<<getpid()
        <<",\"tid\":"<<syscall(SYS_gettid)<<",\"monotonic_ns\":"<<now()<<",\"call_id\":\""<<token<<"\","<<body<<"}\n";
    const auto text=line.str();std::lock_guard<std::mutex> lock(gate);
    return write_file(dir+"/runtime.jsonl",text.data(),text.size(),true);
}
template<class T>T original(const char* name) {
    // A missing real entry point is an ABI error. Never pretend a launch succeeded.
    auto* p=dlsym(RTLD_NEXT,name);if(!p){const char* m="DCAT observer missing runtime symbol\n";write(2,m,strlen(m));_exit(126);}
    return reinterpret_cast<T>(p);
}
void register_binary(const rtDevBinary_t* input, void** handle, rtError_t result) {
    if(directory().empty() || result!=0 || !input || !handle || !*handle)return;
    std::vector<uint8_t> descriptor,bytes;
    if(!read_self(input,sizeof(*input),descriptor))return;
    rtDevBinary_t b{};std::memcpy(&b,descriptor.data(),sizeof(b));
    if(!read_self(b.data,b.length,bytes))return;
    // Raw registered bytes stay private on the experiment host. Python hashes
    // these exact bytes and compares them with the freshly built ELF manifest.
    const uint64_t id=++serial;
    const std::string name="registered-"+std::to_string(getpid())+"-"+std::to_string(id)+".bin";
    if(!write_file(directory()+"/"+name,bytes.data(),bytes.size()))return;
    {std::lock_guard<std::mutex> lock(gate);registered[*handle]={name,id};}
    std::ostringstream row;row<<"\"event\":\"register\",\"handle\":"<<reinterpret_cast<uintptr_t>(*handle)
        <<",\"generation\":"<<id<<",\"binary_file\":\""<<name<<"\",\"binary_bytes\":"<<bytes.size()
        <<",\"result\":"<<result;emit(row.str());
}
struct Snapshot {
    bool valid=false;rtArgsEx_t args{};std::vector<uint8_t> bytes,host;
    int schedule=-1;
};
Snapshot snapshot(const rtArgsEx_t* args,const rtTaskCfgInfo_t* config) {
    Snapshot s;std::vector<uint8_t> descriptor;
    if(!read_self(args,sizeof(*args),descriptor))return s;
    std::memcpy(&s.args,descriptor.data(),sizeof(s.args));
    if(s.args.argsSize>1024*1024 || !read_self(s.args.args,s.args.argsSize,s.bytes))return s;
    if(s.args.hostInputInfoNum>1024)return s;
    if(s.args.hostInputInfoNum && !read_self(s.args.hostInputInfoPtr,
       size_t(s.args.hostInputInfoNum)*sizeof(rtHostInputInfo_t),s.host))return s;
    if(config){
        if(!read_self(config,sizeof(*config),descriptor))return s;
        rtTaskCfgInfo_t cfg{};std::memcpy(&cfg,descriptor.data(),sizeof(cfg));s.schedule=cfg.schemMode;
    }
    s.valid=true;return s;
}
void launched(const char* api,void* handle,uint64_t key,uint32_t blocks,rtStream_t stream,
              const Snapshot& s,uint64_t begin,rtError_t result) {
    Binary binary;{std::lock_guard<std::mutex> lock(gate);auto i=registered.find(handle);if(i!=registered.end())binary=i->second;}
    std::ostringstream row;row<<"\"event\":\"launch\",\"api\":\""<<api<<"\",\"launch_begin_ns\":"<<begin
        <<",\"launch_end_ns\":"<<now()<<",\"handle\":"<<reinterpret_cast<uintptr_t>(handle)
        <<",\"generation\":"<<binary.generation<<",\"binary_file\":\""<<binary.file<<"\",\"tiling_key\":\""<<key
        <<"\",\"block_dim\":"<<blocks<<",\"schedule_mode\":"<<s.schedule
        <<",\"stream\":"<<reinterpret_cast<uintptr_t>(stream)<<",\"result\":"<<result
        <<",\"snapshot_valid\":"<<(s.valid?"true":"false")<<",\"args_size\":"<<s.args.argsSize
        <<",\"has_tiling\":"<<unsigned(s.args.hasTiling)<<",\"no_h2d_copy\":"<<unsigned(s.args.isNoNeedH2DCopy)
        <<",\"tiling_addr_offset\":"<<s.args.tilingAddrOffset<<",\"tiling_data_offset\":"<<s.args.tilingDataOffset
        <<",\"host_input_count\":"<<s.args.hostInputInfoNum<<",\"host_input_hex\":\""<<hex(s.host.data(),s.host.size())
        <<"\",\"args_hex\":\""<<hex(s.bytes.data(),s.bytes.size())<<"\",\"observer_mutates_arguments\":false";
    emit(row.str());
}
}

extern "C" int dcat_attention_observer_begin_v1(const char* id) {
    if(!id || !token.empty() || !safe(id) || directory().empty())return -1;
    token=id;if(!emit("\"event\":\"begin\"")){token.clear();return -2;}return 0;
}
extern "C" int dcat_attention_observer_end_v1() {
    if(token.empty())return -1;bool ok=emit("\"event\":\"end\"");token.clear();return ok?0:-2;
}
extern "C" rtError_t rtRegisterAllKernel(const rtDevBinary_t* bin,void** handle) {
    const auto rc=original<decltype(&rtRegisterAllKernel)>("rtRegisterAllKernel")(bin,handle);
    try{register_binary(bin,handle,rc);}catch(...){}return rc;
}
extern "C" rtError_t rtDevBinaryRegister(const rtDevBinary_t* bin,void** handle) {
    const auto rc=original<decltype(&rtDevBinaryRegister)>("rtDevBinaryRegister")(bin,handle);
    try{register_binary(bin,handle,rc);}catch(...){}return rc;
}
extern "C" rtError_t rtDevBinaryUnRegister(void* handle) {
    const auto rc=original<decltype(&rtDevBinaryUnRegister)>("rtDevBinaryUnRegister")(handle);
    if(rc==0){std::lock_guard<std::mutex> lock(gate);registered.erase(handle);}return rc;
}
extern "C" rtError_t rtKernelLaunchWithHandleV2(void* handle,const uint64_t key,uint32_t blocks,
    rtArgsEx_t* args,rtSmDesc_t* sm,rtStream_t stream,const rtTaskCfgInfo_t* config) {
    auto real=original<decltype(&rtKernelLaunchWithHandleV2)>("rtKernelLaunchWithHandleV2");
    if(token.empty() || nested)return real(handle,key,blocks,args,sm,stream,config);
    nested=true;Snapshot s;try{s=snapshot(args,config);}catch(...){}const auto begin=now();
    const auto rc=real(handle,key,blocks,args,sm,stream,config);
    try{launched("rtKernelLaunchWithHandleV2",handle,key,blocks,stream,s,begin,rc);}catch(...){}
    nested=false;return rc;
}
extern "C" rtError_t rtKernelLaunchWithHandle(void* handle,const uint64_t key,uint32_t blocks,
    rtArgsEx_t* args,rtSmDesc_t* sm,rtStream_t stream,const void* info) {
    auto real=original<decltype(&rtKernelLaunchWithHandle)>("rtKernelLaunchWithHandle");
    if(token.empty() || nested)return real(handle,key,blocks,args,sm,stream,info);
    nested=true;Snapshot s;try{s=snapshot(args,nullptr);}catch(...){}const auto begin=now();
    const auto rc=real(handle,key,blocks,args,sm,stream,info);
    try{launched("rtKernelLaunchWithHandle",handle,key,blocks,stream,s,begin,rc);}catch(...){}
    nested=false;return rc;
}
extern "C" rtError_t rtDeviceSynchronize() {
    auto rc=original<decltype(&rtDeviceSynchronize)>("rtDeviceSynchronize")();
    if(!token.empty())try{emit("\"event\":\"device_sync\",\"result\":"+std::to_string(rc));}catch(...){}return rc;
}
extern "C" rtError_t rtStreamSynchronize(rtStream_t stream) {
    auto rc=original<decltype(&rtStreamSynchronize)>("rtStreamSynchronize")(stream);
    if(!token.empty())try{emit("\"event\":\"stream_sync\",\"stream\":"+
        std::to_string(reinterpret_cast<uintptr_t>(stream))+",\"result\":"+std::to_string(rc));}catch(...){}return rc;
}
