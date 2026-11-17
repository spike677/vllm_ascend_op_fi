// V2-only deployment derived from batch_scale_native.cpp. No arithmetic changes.
// V1/V3/V4 are deliberately not exported; a version number is not an ABI proof.
// Experimental backend descriptor fault. Does not modify a shared installation.
// Native DynamicQuant executes once, with one reduction row spanning the batch.
// This models an incorrect backend reduction domain, not modified device ISA.
#include <acl/acl.h>
#include <aclnnop/aclnn_dynamic_quant_v2.h>
#include <dlfcn.h>
#include <array>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <thread>
#include <sys/syscall.h>
#include <unistd.h>
#include "batch_scale_inventory.h"

namespace {
constexpr aclnnStatus invalid = 161001;
std::mutex guard;
struct Selection { void* ptr{}; int64_t rows{}, cols{}; bool fault{}; bool armed{}; };
thread_local Selection selection;
// prepare hits, execute hits, fault executions, API version, prepare errors,
// execution errors. Read only after synchronizing the selected operation.
thread_local std::array<uint64_t, 6> counters{};
struct Context {
    std::thread::id owner{std::this_thread::get_id()};
    bool fault{};
    int64_t rows{};
    void* scales{};
    aclTensor *x{}, *q{}, *s{};
    ~Context() { if (x) aclDestroyTensor(x); if (q) aclDestroyTensor(q); if (s) aclDestroyTensor(s); }
};
std::unordered_map<aclOpExecutor*, std::unique_ptr<Context>> pending;
bool owned{};
std::thread::id scope_owner;
uint64_t executing{};
dcat_dq_inventory::Inventory<32768> independent_inventory;

void* original(const char* name) {
    static void* handle = [] {
        const char* path = std::getenv("DCAT_REAL_OPAPI");
        if (!path) { std::fprintf(stderr, "Explicit matched real opapi required\n"); std::abort(); }
        void* h = dlopen(path, RTLD_NOW | RTLD_LOCAL);
        if (!h) { std::fprintf(stderr, "v2 backend cannot load original opapi: %s\n", dlerror()); std::abort(); }
        return h;
    }();
    void* fn = dlsym(handle, name);
    if (!fn) { std::fprintf(stderr, "v2 backend missing original symbol: %s\n", name); std::abort(); }
    return fn;
}

bool tensor_info(const aclTensor* t, aclDataType type, int64_t rows, int64_t cols, void** addr) {
    if (!t) return false;
    int64_t *dims{}, *strides{}, offset{};
    uint64_t n{}, ns{};
    aclDataType dtype{};
    aclFormat format{};
    if (aclGetViewShape(t, &dims, &n) || aclGetViewStrides(t, &strides, &ns) ||
        aclGetViewOffset(t, &offset) || aclGetDataType(t, &dtype) || aclGetFormat(t, &format) ||
        aclGetRawTensorAddr(t, addr)) return false;
    // ACL getters expose descriptor-owned arrays: no data or metadata is changed.
    if (dtype != type || format != ACL_FORMAT_ND || offset != 0 || !*addr) return false;
    if (cols == 0) return n == 1 && ns == 1 && dims[0] == rows && strides[0] == 1;
    return n == 2 && ns == 2 && dims[0] == rows && dims[1] == cols && strides[0] == cols && strides[1] == 1;
}

dcat_dq_inventory::Event observe_prepare(const aclTensor* x,const aclTensor* a,const aclTensor* b,
    int64_t dtype,const aclTensor* q,const aclTensor* s,const aclTensor* offset) {
    dcat_dq_inventory::Event event{};
    event.thread=static_cast<uint64_t>(syscall(SYS_gettid));
    (void)aclrtGetDevice(&event.device);
    const aclTensor* tensors[]{x,q,s};
    uint64_t* pointers[]{&event.input,&event.quant,&event.scale};
    int32_t* types[]{&event.input_dtype,&event.quant_dtype,&event.scale_dtype};
    int32_t* formats[]{&event.input_format,&event.quant_format,&event.scale_format};
    for(unsigned i=0;i<3;++i) {
        if(!tensors[i])continue;
        void* raw{};aclDataType actual_type{};aclFormat format{};
        if(!aclGetRawTensorAddr(tensors[i],&raw))*pointers[i]=reinterpret_cast<uint64_t>(raw);
        if(!aclGetDataType(tensors[i],&actual_type))*types[i]=static_cast<int32_t>(actual_type);
        if(!aclGetFormat(tensors[i],&format))*formats[i]=static_cast<int32_t>(format);
    }
    int64_t* shape{};uint64_t dimensions{};
    if(x && !aclGetViewShape(x,&shape,&dimensions) && dimensions==2) {
        event.rows=shape[0];event.columns=shape[1];
        void* raw{};
        if((event.input_dtype==ACL_BF16 || event.input_dtype==ACL_FLOAT16)
            && tensor_info(x,static_cast<aclDataType>(event.input_dtype),event.rows,event.columns,&raw))event.flags|=1;
        if(tensor_info(q,ACL_INT8,event.rows,event.columns,&raw))event.flags|=2;
        if(tensor_info(s,ACL_FLOAT,event.rows,0,&raw))event.flags|=4;
    }
    if(!a && !b && !offset)event.flags|=8;
    if(dtype==ACL_INT8)event.flags|=16;
    return independent_inventory.begin(event); // Before selection/owner/admission.
}

template<class Call>
aclnnStatus prepare(int api, const aclTensor* x, const aclTensor* smooth, const aclTensor* group,
                    int64_t dtype, bool symmetric, const char* mode, const aclTensor* q,
                    const aclTensor* s, const aclTensor* offset, uint64_t* ws,
                    aclOpExecutor** executor, Call call) {
    Selection chosen;
    {
        std::lock_guard<std::mutex> lock(guard);
        void* raw{};
        if (!selection.armed || !x || aclGetRawTensorAddr(x, &raw) || raw != selection.ptr)
            return call(x, q, s, ws, executor);
        chosen = selection;
        selection.armed = false; // one selected invocation only
        ++counters[0]; counters[3] = api;
    }
    auto ctx = std::make_unique<Context>();
    ctx->fault = chosen.fault; ctx->rows = chosen.rows;
    void *px{}, *pq{};
    aclDataType xtype{};
    const bool valid = !smooth && !group && !offset && dtype == ACL_INT8 && symmetric &&
        mode && std::strcmp(mode, "pertoken") == 0 && !aclGetDataType(x, &xtype) &&
        (xtype == ACL_BF16 || xtype == ACL_FLOAT16) &&
        tensor_info(x, xtype, chosen.rows, chosen.cols, &px) &&
        tensor_info(q, ACL_INT8, chosen.rows, chosen.cols, &pq) &&
        tensor_info(s, ACL_FLOAT, chosen.rows, 0, &ctx->scales);
    if (!valid) { std::lock_guard<std::mutex> lock(guard); ++counters[4]; return invalid; }
    const aclTensor *xx=x, *qq=q, *ss=s;
    if (chosen.fault) {
        int64_t dims[2]{1, chosen.rows * chosen.cols}, strides[2]{dims[1], 1}, one[1]{1};
        ctx->x = aclCreateTensor(dims, 2, xtype, strides, 0, ACL_FORMAT_ND, dims, 2, px);
        ctx->q = aclCreateTensor(dims, 2, ACL_INT8, strides, 0, ACL_FORMAT_ND, dims, 2, pq);
        ctx->s = aclCreateTensor(one, 1, ACL_FLOAT, one, 0, ACL_FORMAT_ND, one, 1, ctx->scales);
        if (!ctx->x || !ctx->q || !ctx->s) { std::lock_guard<std::mutex> lock(guard); ++counters[4]; return invalid; }
        xx=ctx->x; qq=ctx->q; ss=ctx->s;
    }
    auto rc = call(xx, qq, ss, ws, executor);
    std::lock_guard<std::mutex> lock(guard);
    if (rc) ++counters[4];
    else if (!pending.emplace(*executor, std::move(ctx)).second) std::abort();
    return rc;
}

aclnnStatus execute(const char* api, void* workspace, uint64_t size, aclOpExecutor* executor, aclrtStream stream,
    uint64_t* original_calls) {
    using Fn = aclnnStatus (*)(void*, uint64_t, aclOpExecutor*, aclrtStream);
    std::unique_ptr<Context> ctx;
    {
        std::lock_guard<std::mutex> lock(guard);
        auto it = pending.find(executor);
        if (it != pending.end()) {
            if(it->second->owner!=std::this_thread::get_id())return invalid;
            ctx=std::move(it->second); pending.erase(it);
            ++executing;
        }
    }
    ++*original_calls;
    auto rc = reinterpret_cast<Fn>(original(api))(workspace, size, executor, stream);
    if (!ctx) return rc;
    if (!rc && ctx->fault) {
        // Publish the scalar produced by this native quantization to all scale
        // rows, on the same stream, before returning from this backend boundary.
        for (int64_t row=1; row<ctx->rows && !rc; ++row)
            rc = aclrtMemcpyAsync(static_cast<char*>(ctx->scales)+row*sizeof(float), sizeof(float),
                                  ctx->scales, sizeof(float), ACL_MEMCPY_DEVICE_TO_DEVICE, stream);
    }
    // Keep alias metadata alive through completion and surface asynchronous errors.
    auto sync_rc = aclrtSynchronizeStream(stream);
    if (!rc) rc=sync_rc;
    std::lock_guard<std::mutex> lock(guard);
    --executing;
    ++counters[1];
    if (ctx->fault) ++counters[2];
    if (rc) ++counters[5];
    return rc;
}
} // namespace

// This capability is accepted only together with the approved source/DSO hashes.
extern "C" uint32_t dcat_dq_thread_scope_v1() { return 1; }
extern "C" int dcat_dq_inventory_state_v1(dcat_dq_inventory::State* output) {
    if(!output || output->version!=1 || output->size!=sizeof(dcat_dq_inventory::State))return -1;
    *output=independent_inventory.state();return 0;
}
extern "C" int dcat_dq_inventory_read_v1(uint64_t index,dcat_dq_inventory::Event* output) {
    if(!output || output->version!=1 || output->size!=sizeof(dcat_dq_inventory::Event))return -1;
    return independent_inventory.read(index,output)?0:-2;
}
extern "C" int dcat_dq_state_v1(uint64_t* out,size_t count) {
    std::lock_guard<std::mutex> lock(guard);
    if(!out || count!=7)return -1;
    const uint64_t values[7]{1,selection.armed,pending.size(),executing,
        owned && scope_owner==std::this_thread::get_id(),owned,counters[4]+counters[5]};
    std::copy(values,values+7,out);return 0;
}

extern "C" int dcat_dq_arm(void* pointer, int64_t rows, int64_t cols, int fault) {
    std::lock_guard<std::mutex> lock(guard);
    const char* enable = std::getenv("DCAT_DQ_V2_ENABLE");
    if (!enable || std::strcmp(enable, "1") || owned || executing || selection.armed || !pending.empty() || !pointer ||
        rows < 1 || rows > 64 || cols < 1 || cols > 65536 || rows*cols > 1048576 || (fault != 0 && fault != 1)) return -1;
    scope_owner=std::this_thread::get_id();owned=true;
    selection={pointer,rows,cols,fault != 0,true}; counters.fill(0); return 0;
}
extern "C" int dcat_dq_finish(uint64_t* output, size_t count) {
    std::lock_guard<std::mutex> lock(guard);
    if (!output || count != counters.size() || !owned || scope_owner!=std::this_thread::get_id()) return -1;
    std::copy(counters.begin(), counters.end(), output);
    bool valid = !selection.armed && pending.empty() && !executing && counters[0]==1 && counters[1]==1 && !counters[4] && !counters[5];
    selection.armed=false;
    if(pending.empty() && !executing)owned=false;
    return valid ? 0 : -1;
}
extern "C" bool CanUsePTACache(const char* name) {
    // A cached executor can bypass GetWorkspaceSize. Disable caching for this
    // API in both healthy and fault runs; leave unrelated operations unchanged.
    if (name && std::strcmp(name, "aclnnDynamicQuantV2")==0) return false;
    return reinterpret_cast<bool(*)(const char*)>(original("CanUsePTACache"))(name);
}

extern "C" aclnnStatus aclnnDynamicQuantV2GetWorkspaceSize(const aclTensor* x,const aclTensor* a,const aclTensor* b,int64_t d,const aclTensor* q,const aclTensor* s,const aclTensor* o,uint64_t* w,aclOpExecutor** e) {
    const auto observed=observe_prepare(x,a,b,d,q,s,o);
    uint64_t original_calls=0;
    auto fn=reinterpret_cast<decltype(&aclnnDynamicQuantV2GetWorkspaceSize)>(original("aclnnDynamicQuantV2GetWorkspaceSize"));
    const auto status=prepare(2,x,a,b,d,true,"pertoken",q,s,o,w,e,
        [&](const aclTensor* xx,const aclTensor* qq,const aclTensor* ss,uint64_t* ww,aclOpExecutor** ee){
            ++original_calls;return fn(xx,a,b,d,qq,ss,o,ww,ee);});
    independent_inventory.prepared(observed,!status && e?reinterpret_cast<uint64_t>(*e):0,status,original_calls);
    return status;
}
#define EXEC(NAME) extern "C" aclnnStatus NAME(void* w,uint64_t size,aclOpExecutor* e,aclrtStream stream) { \
    auto observed=independent_inventory.enter(reinterpret_cast<uint64_t>(e),static_cast<uint64_t>(syscall(SYS_gettid))); \
    uint64_t original_calls=0;auto status=execute(#NAME,w,size,e,stream,&original_calls); \
    independent_inventory.leave(observed,status,original_calls);return status; }
EXEC(aclnnDynamicQuantV2)
