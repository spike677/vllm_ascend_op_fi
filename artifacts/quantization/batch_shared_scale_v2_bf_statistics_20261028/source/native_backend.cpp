// Experimental backend descriptor fault. Does not modify a shared installation.
// Native DynamicQuant executes once, with one reduction row spanning the batch.
// This models an incorrect backend reduction domain, not modified device ISA.
#include <acl/acl.h>
#include <aclnnop/aclnn_dynamic_quant.h>
#include <aclnnop/aclnn_dynamic_quant_v2.h>
#include <aclnnop/aclnn_dynamic_quant_v3.h>
#include <aclnnop/aclnn_dynamic_quant_v4.h>
#include <dlfcn.h>
#include <array>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <mutex>
#include <unordered_map>

namespace {
constexpr aclnnStatus invalid = 161001;
std::mutex guard;
struct Selection { void* ptr{}; int64_t rows{}, cols{}; bool fault{}; bool armed{}; } selection;
// prepare hits, execute hits, fault executions, API version, prepare errors,
// execution errors. Read only after synchronizing the selected operation.
std::array<uint64_t, 6> counters{};
struct Context {
    bool fault{};
    int64_t rows{};
    void* scales{};
    aclTensor *x{}, *q{}, *s{};
    ~Context() { if (x) aclDestroyTensor(x); if (q) aclDestroyTensor(q); if (s) aclDestroyTensor(s); }
};
std::unordered_map<aclOpExecutor*, std::unique_ptr<Context>> pending;

void* original(const char* name) {
    static void* handle = [] {
        const char* path = std::getenv("DCAT_REAL_OPAPI");
        if (!path) path = "/usr/local/Ascend/cann-9.1.0/opp/built-in/op_impl/ai_core/tbe/op_api/lib/linux/aarch64/libopapi.so";
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

aclnnStatus execute(const char* api, void* workspace, uint64_t size, aclOpExecutor* executor, aclrtStream stream) {
    using Fn = aclnnStatus (*)(void*, uint64_t, aclOpExecutor*, aclrtStream);
    std::unique_ptr<Context> ctx;
    {
        std::lock_guard<std::mutex> lock(guard);
        auto it = pending.find(executor);
        if (it != pending.end()) { ctx=std::move(it->second); pending.erase(it); }
    }
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
    ++counters[1];
    if (ctx->fault) ++counters[2];
    if (rc) ++counters[5];
    return rc;
}
} // namespace

extern "C" int dcat_dq_arm(void* pointer, int64_t rows, int64_t cols, int fault) {
    std::lock_guard<std::mutex> lock(guard);
    const char* enable = std::getenv("DCAT_DQ_V2_ENABLE");
    if (!enable || std::strcmp(enable, "1") || selection.armed || !pending.empty() || !pointer ||
        rows < 1 || rows > 64 || cols < 1 || cols > 65536 || rows*cols > 1048576 || (fault != 0 && fault != 1)) return -1;
    selection={pointer,rows,cols,fault != 0,true}; counters.fill(0); return 0;
}
extern "C" int dcat_dq_finish(uint64_t* output, size_t count) {
    std::lock_guard<std::mutex> lock(guard);
    if (!output || count != counters.size()) return -1;
    std::copy(counters.begin(), counters.end(), output);
    bool valid = !selection.armed && pending.empty() && counters[0]==1 && counters[1]==1 && !counters[4] && !counters[5];
    selection.armed=false;
    return valid ? 0 : -1;
}
extern "C" bool CanUsePTACache(const char* name) {
    // A cached executor can bypass GetWorkspaceSize. Disable caching for this
    // API in both healthy and fault runs; leave unrelated operations unchanged.
    if (name && std::strncmp(name, "aclnnDynamicQuant", 17)==0) return false;
    return reinterpret_cast<bool(*)(const char*)>(original("CanUsePTACache"))(name);
}

extern "C" aclnnStatus aclnnDynamicQuantGetWorkspaceSize(const aclTensor* x,const aclTensor* a,const aclTensor* q,const aclTensor* s,uint64_t* w,aclOpExecutor** e) {
    auto fn=reinterpret_cast<decltype(&aclnnDynamicQuantGetWorkspaceSize)>(original("aclnnDynamicQuantGetWorkspaceSize"));
    return prepare(1,x,a,nullptr,ACL_INT8,true,"pertoken",q,s,nullptr,w,e,
        [&](const aclTensor* xx,const aclTensor* qq,const aclTensor* ss,uint64_t* ww,aclOpExecutor** ee){return fn(xx,a,qq,ss,ww,ee);});
}
extern "C" aclnnStatus aclnnDynamicQuantV2GetWorkspaceSize(const aclTensor* x,const aclTensor* a,const aclTensor* b,int64_t d,const aclTensor* q,const aclTensor* s,const aclTensor* o,uint64_t* w,aclOpExecutor** e) {
    auto fn=reinterpret_cast<decltype(&aclnnDynamicQuantV2GetWorkspaceSize)>(original("aclnnDynamicQuantV2GetWorkspaceSize"));
    return prepare(2,x,a,b,d,true,"pertoken",q,s,o,w,e,
        [&](const aclTensor* xx,const aclTensor* qq,const aclTensor* ss,uint64_t* ww,aclOpExecutor** ee){return fn(xx,a,b,d,qq,ss,o,ww,ee);});
}
extern "C" aclnnStatus aclnnDynamicQuantV3GetWorkspaceSize(const aclTensor* x,const aclTensor* a,const aclTensor* b,int64_t d,bool sym,const char* mode,const aclTensor* q,const aclTensor* s,const aclTensor* o,uint64_t* w,aclOpExecutor** e) {
    auto fn=reinterpret_cast<decltype(&aclnnDynamicQuantV3GetWorkspaceSize)>(original("aclnnDynamicQuantV3GetWorkspaceSize"));
    return prepare(3,x,a,b,d,sym,mode,q,s,o,w,e,
        [&](const aclTensor* xx,const aclTensor* qq,const aclTensor* ss,uint64_t* ww,aclOpExecutor** ee){return fn(xx,a,b,d,sym,mode,qq,ss,o,ww,ee);});
}
extern "C" aclnnStatus aclnnDynamicQuantV4GetWorkspaceSize(const aclTensor* x,const aclTensor* a,const aclTensor* b,int64_t d,bool sym,const char* mode,float max,const aclTensor* q,const aclTensor* s,const aclTensor* o,uint64_t* w,aclOpExecutor** e) {
    auto fn=reinterpret_cast<decltype(&aclnnDynamicQuantV4GetWorkspaceSize)>(original("aclnnDynamicQuantV4GetWorkspaceSize"));
    return prepare(4,x,a,b,d,sym,mode,q,s,o,w,e,
        [&](const aclTensor* xx,const aclTensor* qq,const aclTensor* ss,uint64_t* ww,aclOpExecutor** ee){return fn(xx,a,b,d,sym,mode,max,qq,ss,o,ww,ee);});
}
#define EXEC(NAME) extern "C" aclnnStatus NAME(void* w,uint64_t size,aclOpExecutor* e,aclrtStream stream) { return execute(#NAME,w,size,e,stream); }
EXEC(aclnnDynamicQuant)
EXEC(aclnnDynamicQuantV2)
EXEC(aclnnDynamicQuantV3)
EXEC(aclnnDynamicQuantV4)
