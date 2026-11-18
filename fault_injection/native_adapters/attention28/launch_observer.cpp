// Independent runtime-boundary observer. Never selects a key or edits arguments.
// Compile against this process's installed CANN headers; no copied ABI structs.
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include <runtime/kernel.h>
#include <runtime/dev.h>
#include <acl/acl_rt.h>
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
#include <cerrno>

namespace {
std::mutex gate;
std::atomic<uint64_t> serial{0};
struct Binary { std::string file; uint64_t generation=0; };
std::map<void*, Binary> registered;
struct Function { void* binary=nullptr; uint64_t generation=0,key=0; std::string name; };
std::map<void*, Function> functions;
uint64_t captured_bytes=0,captured_count=0;
// A standard Worker registers hundreds of small runtime kernels before FIA.
// Keep byte limits unchanged; 128 registrations was exhausted before the first
// observed decode in the source45 stock/private bridge (333 registrations).
constexpr uint64_t max_binary_bytes=128ULL*1024*1024;
constexpr uint64_t max_capture_bytes=256ULL*1024*1024;
constexpr uint64_t max_capture_count=1024;
bool fits_capture_budget(uint64_t length,uint64_t count,uint64_t bytes) {
    return length>0 && length<=max_binary_bytes && count<max_capture_count &&
        bytes<=max_capture_bytes && length<=max_capture_bytes-bytes;
}
thread_local std::string token;
thread_local bool nested=false;
thread_local int memory_vm_errno=0,memory_proc_errno=0;
thread_local const char* memory_method="none";
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
    memory_vm_errno=0;memory_proc_errno=0;memory_method="none";
    if(process_vm_readv(getpid(),&local,1,&remote,1,0)==static_cast<ssize_t>(n)){
        memory_method="process_vm_readv_self";return true;
    }
    memory_vm_errno=errno;
    // Docker's default seccomp may deny process_vm_readv even for self.
    // A bounded read-only pread from this process's own memory retains kernel
    // pointer validation; never dereference an unchecked pointer or read a peer.
    int fd=open("/proc/self/mem",O_RDONLY|O_CLOEXEC);
    if(fd>=0){
        const auto got=pread(fd,bytes.data(),n,static_cast<off_t>(reinterpret_cast<uintptr_t>(p)));
        memory_proc_errno=got==static_cast<ssize_t>(n)?0:errno;close(fd);
        if(got==static_cast<ssize_t>(n)){memory_method="proc_self_mem_pread";return true;}
    }else memory_proc_errno=errno;
    bytes.clear();return false;
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
    bool overflow=false;uint64_t count=0,total=0;
    {std::lock_guard<std::mutex> lock(gate);
        overflow=!fits_capture_budget(b.length,captured_count,captured_bytes);
        if(!overflow){captured_bytes+=b.length;++captured_count;}
        count=captured_count;total=captured_bytes;}
    if(overflow){
        std::ostringstream row;row<<"\"event\":\"capture_overflow\",\"reason\":\"registered_binary_budget\""
            <<",\"binary_bytes\":"<<b.length<<",\"captured_count\":"<<count<<",\"captured_bytes\":"<<total
            <<",\"max_count\":"<<max_capture_count<<",\"max_total_bytes\":"<<max_capture_bytes;
        emit(row.str());return;
    }
    if(!read_self(b.data,b.length,bytes))return;
    // Raw registered bytes stay private on the experiment host. Python hashes
    // these exact bytes and compares them with the freshly built ELF manifest.
    const uint64_t id=++serial;
    const std::string name="registered-"+std::to_string(getpid())+"-"+std::to_string(id)+".bin";
    if(!write_file(directory()+"/"+name,bytes.data(),bytes.size()))return;
    {std::lock_guard<std::mutex> lock(gate);registered[*handle]={name,id};}
    std::ostringstream row;row<<"\"event\":\"register\",\"handle\":"<<reinterpret_cast<uintptr_t>(*handle)
        <<",\"generation\":"<<id<<",\"binary_file\":\""<<name<<"\",\"binary_bytes\":"<<bytes.size()
        <<",\"result\":"<<result<<",\"memory_read_method\":\""<<memory_method
        <<"\",\"process_vm_readv_errno\":"<<memory_vm_errno;emit(row.str());
}
struct Snapshot {
    bool valid=false;rtArgsEx_t args{};std::vector<uint8_t> bytes,host;
    bool aclHostArgs=false;std::vector<uint8_t> placeholders;
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
              const Snapshot& s,uint64_t begin,rtError_t result,void* function=nullptr) {
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
        <<"\",\"args_hex\":\""<<hex(s.bytes.data(),s.bytes.size())<<"\",\"observer_mutates_arguments\":false"
        <<",\"argument_layout\":\""<<(s.aclHostArgs?"aclrt_host_placeholders":"rtArgsEx")<<"\""
        <<",\"function_handle\":"<<reinterpret_cast<uintptr_t>(function)
        <<",\"placeholders_hex\":\""<<hex(s.placeholders.data(),s.placeholders.size())<<"\"";
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
extern "C" int dcat_attention_observer_memory_selftest_v1() {
    // Host-only budget boundary checks; no allocation, file or NPU operation.
    if(!fits_capture_budget(1,333,35521944) || !fits_capture_budget(1,1023,max_capture_bytes-1) ||
       fits_capture_budget(1,1024,0) || fits_capture_budget(max_binary_bytes+1,0,0) ||
       fits_capture_budget(1,0,max_capture_bytes) || fits_capture_budget(1,0,UINT64_MAX) ||
       fits_capture_budget(0,0,0))return -4;
    uint64_t expected[3]={0x4712a8ffdeadbeefULL,0,0x8a1f447766221199ULL};std::vector<uint8_t> bytes;
    if(!read_self(expected,sizeof(expected),bytes)||std::memcmp(bytes.data(),expected,sizeof(expected)))return -1;
    // Confirm inaccessible addresses are refused without a signal/crash.
    if(read_self(reinterpret_cast<void*>(1),16,bytes))return -2;
    return read_self(expected,sizeof(expected),bytes)&&!std::memcmp(bytes.data(),expected,sizeof(expected))?0:-3;
}
extern "C" const char* dcat_attention_observer_memory_status_v1() {
    static thread_local std::string value;
    value="{\"method\":\""+std::string(memory_method)+"\",\"process_vm_readv_errno\":"+
        std::to_string(memory_vm_errno)+",\"proc_self_mem_errno\":"+std::to_string(memory_proc_errno)+"}";
    return value.c_str();
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

// Modern CANN nnopbase uses these public ACL calls rather than rtArgsEx.
// Enabled only when the exact SDK headers declare all required interfaces.
#ifdef DCAT_OBSERVE_ACL_HOST_ARGS
namespace {
void acl_binary_bytes(const void* data,size_t length,aclrtBinHandle* handle,aclError result) {
    rtDevBinary_t bin{};bin.data=data;bin.length=length;
    register_binary(&bin,reinterpret_cast<void**>(handle),result);
}
void function_binding(aclrtBinHandle binary,aclrtFuncHandle* function,uint64_t entry,const char* name,aclError result) {
    if(result!=0 || !function || !*function)return;
    Binary b;{std::lock_guard<std::mutex> lock(gate);auto i=registered.find(binary);
        if(i!=registered.end())b=i->second;}
    if(b.generation==0){std::ostringstream row;row<<"\"event\":\"unbound_function\",\"handle\":"
        <<reinterpret_cast<uintptr_t>(binary)<<",\"function_handle\":"<<reinterpret_cast<uintptr_t>(*function)
        <<",\"tiling_key\":\""<<entry<<"\",\"result\":"<<result;emit(row.str());return;}
    {std::lock_guard<std::mutex> lock(gate);
        functions[*function]={binary,b.generation,entry,name?name:""};}
    // The runtime's returned function handle is the actual launch binding.
    std::ostringstream row;row<<"\"event\":\"function_bind\",\"handle\":"<<reinterpret_cast<uintptr_t>(binary)
        <<",\"generation\":"<<b.generation<<",\"function_handle\":"<<reinterpret_cast<uintptr_t>(*function)
        <<",\"tiling_key\":\""<<entry<<"\",\"result\":"<<result;
    emit(row.str());
}
Snapshot acl_snapshot(void* args,size_t size,aclrtPlaceHolderInfo* placeholders,size_t count,const aclrtLaunchKernelCfg* config) {
    Snapshot s;s.aclHostArgs=true;
    s.args.argsSize=size<=UINT32_MAX?static_cast<uint32_t>(size):0;
    if(size>1024*1024 || count>1024 || !read_self(args,size,s.bytes))return s;
    if(count && !read_self(placeholders,count*sizeof(*placeholders),s.placeholders))return s;
    if(!config)return s;
    std::vector<uint8_t> desc,attrs;
    if(!read_self(config,sizeof(*config),desc))return s;
    aclrtLaunchKernelCfg cfg{};std::memcpy(&cfg,desc.data(),sizeof(cfg));
    if(cfg.numAttrs>128 || !read_self(cfg.attrs,cfg.numAttrs*sizeof(*cfg.attrs),attrs))return s;
    for(size_t i=0;i<cfg.numAttrs;++i){aclrtLaunchKernelAttr attr{};
        std::memcpy(&attr,attrs.data()+i*sizeof(attr),sizeof(attr));
        if(attr.id==ACL_RT_LAUNCH_KERNEL_ATTR_SCHEM_MODE){if(s.schedule!=-1)return s;s.schedule=attr.value.schemMode;}}
    s.valid=true;return s;
}
}
extern "C" aclError aclrtBinaryLoadFromData(const void* data,size_t length,
    const aclrtBinaryLoadOptions* options,aclrtBinHandle* handle) {
    const auto rc=original<decltype(&aclrtBinaryLoadFromData)>("aclrtBinaryLoadFromData")(data,length,options,handle);
    try{acl_binary_bytes(data,length,handle,rc);}catch(...){}return rc;
}
extern "C" aclError aclrtBinaryLoadFromFile(const char* path,aclrtBinaryLoadOptions* options,aclrtBinHandle* handle) {
    std::vector<uint8_t> bytes;struct stat before{},after{};int fd=-1;bool captured=false;
    if(path && !directory().empty())try{
        fd=open(path,O_RDONLY|O_CLOEXEC|O_NOFOLLOW);
        if(fd>=0 && !fstat(fd,&before) && S_ISREG(before.st_mode) && before.st_size>0 && before.st_size<=128*1024*1024){
            bytes.resize(before.st_size);size_t done=0;
            while(done<bytes.size()){auto n=read(fd,bytes.data()+done,bytes.size()-done);if(n<=0)break;done+=n;}
            captured=done==bytes.size();}
    }catch(...){}
    const auto rc=original<decltype(&aclrtBinaryLoadFromFile)>("aclrtBinaryLoadFromFile")(path,options,handle);
    if(fd>=0){captured=captured && !fstat(fd,&after) && before.st_dev==after.st_dev && before.st_ino==after.st_ino
        &&before.st_size==after.st_size&&before.st_mtim.tv_sec==after.st_mtim.tv_sec&&before.st_mtim.tv_nsec==after.st_mtim.tv_nsec;
        close(fd);}
    if(captured)try{acl_binary_bytes(bytes.data(),bytes.size(),handle,rc);}catch(...){}return rc;
}
extern "C" aclError aclrtBinaryGetFunctionByEntry(aclrtBinHandle binary,uint64_t entry,aclrtFuncHandle* function) {
    const auto rc=original<decltype(&aclrtBinaryGetFunctionByEntry)>("aclrtBinaryGetFunctionByEntry")(binary,entry,function);
    try{function_binding(binary,function,entry,nullptr,rc);}catch(...){}return rc;
}
extern "C" aclError aclrtBinaryGetFunction(const aclrtBinHandle binary,const char* name,aclrtFuncHandle* function) {
    const auto rc=original<decltype(&aclrtBinaryGetFunction)>("aclrtBinaryGetFunction")(binary,name,function);
    // Named/static kernels are observed but are not admitted as dynamic FIA keys.
    try{function_binding(binary,function,0,name,rc);}catch(...){}return rc;
}
extern "C" aclError aclrtBinaryUnLoad(aclrtBinHandle binary) {
    const auto rc=original<decltype(&aclrtBinaryUnLoad)>("aclrtBinaryUnLoad")(binary);
    if(rc==0){std::lock_guard<std::mutex> lock(gate);registered.erase(binary);
        for(auto i=functions.begin();i!=functions.end();){if(i->second.binary==binary)i=functions.erase(i);else ++i;}}
    return rc;
}
extern "C" aclError aclrtLaunchKernelWithHostArgs(aclrtFuncHandle function,uint32_t blocks,
    aclrtStream stream,aclrtLaunchKernelCfg* cfg,void* args,size_t size,
    aclrtPlaceHolderInfo* placeholders,size_t count) {
    auto real=original<decltype(&aclrtLaunchKernelWithHostArgs)>("aclrtLaunchKernelWithHostArgs");
    if(token.empty()||nested)return real(function,blocks,stream,cfg,args,size,placeholders,count);
    nested=true;Snapshot s;try{s=acl_snapshot(args,size,placeholders,count,cfg);}catch(...){}
    Function f;{std::lock_guard<std::mutex> lock(gate);auto i=functions.find(function);if(i!=functions.end())f=i->second;}
    const auto begin=now();const auto rc=real(function,blocks,stream,cfg,args,size,placeholders,count);
    try{launched("aclrtLaunchKernelWithHostArgs",f.binary,f.key,blocks,stream,s,begin,rc,function);}catch(...){}
    nested=false;return rc;
}
#endif

extern "C" aclError aclrtSynchronizeDevice() {
    auto rc=original<decltype(&aclrtSynchronizeDevice)>("aclrtSynchronizeDevice")();
    if(!token.empty())try{emit("\"event\":\"device_sync\",\"api\":\"aclrtSynchronizeDevice\",\"result\":"+std::to_string(rc));}catch(...){}
    return rc;
}
extern "C" aclError aclrtSynchronizeStream(aclrtStream stream) {
    auto rc=original<decltype(&aclrtSynchronizeStream)>("aclrtSynchronizeStream")(stream);
    if(!token.empty())try{emit("\"event\":\"stream_sync\",\"api\":\"aclrtSynchronizeStream\",\"stream\":"+
        std::to_string(reinterpret_cast<uintptr_t>(stream))+",\"result\":"+std::to_string(rc));}catch(...){}return rc;
}
extern "C" aclError aclrtSynchronizeStreamWithTimeout(aclrtStream stream,int32_t timeout) {
    auto rc=original<decltype(&aclrtSynchronizeStreamWithTimeout)>("aclrtSynchronizeStreamWithTimeout")(stream,timeout);
    if(!token.empty())try{emit("\"event\":\"stream_sync\",\"api\":\"aclrtSynchronizeStreamWithTimeout\",\"stream\":"+
        std::to_string(reinterpret_cast<uintptr_t>(stream))+",\"result\":"+std::to_string(rc));}catch(...){}return rc;
}
extern "C" int dcat_attention_observer_synchronize_v1() {
    if(token.empty())return -1;
    // Explicit public ACL completion, also when torch resolves its sync through
    // a handle-specific dlsym which bypasses ELF interposition.
    return aclrtSynchronizeDevice();
}
