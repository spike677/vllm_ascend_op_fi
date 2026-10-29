#ifndef FAULT28_FA_POLICY_H
#define FAULT28_FA_POLICY_H
// Experiment-only typed host policy. No tensor input/output reads or changes.
#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <iomanip>
#include <initializer_list>
#include <sstream>
#include <stdexcept>
#include <string>
#include <sys/file.h>
#include <sys/syscall.h>
#include <unistd.h>
#include "exe_graph/runtime/tiling_context.h"
namespace fault28fa {
inline std::string Env(const char* name) { const char* p = std::getenv(name); return p ? p : ""; }
inline bool Decimal(const std::string& s) { return !s.empty() && s.size() <= 20 && std::all_of(s.begin(), s.end(), [](char c){return c >= '0' && c <= '9';}); }
inline bool Token(const std::string& s) { return !s.empty() && s.size() <= 128 && std::all_of(s.begin(), s.end(), [](char c){return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') || c == '_' || c == '.' || c == '-';}); }
inline std::string Quote(const std::string& s) { std::ostringstream o; o << '"'; for(unsigned char c:s){if(c=='"'||c=='\\')o << '\\' << c;else if(c<32)o << "\\u" << std::hex << std::setw(4) << std::setfill('0') << unsigned(c) << std::dec;else o << c;}o << '"';return o.str(); }
struct State {
    gert::TilingContext* context = nullptr;
    bool active = false, eligible = false, applied = false, typed = false, l1Known = false, l1Reuse = false;
    std::string mode, token, forward, path, typedJson;
    uint64_t workspace = 0;
};
inline thread_local State state;
inline void Begin(gert::TilingContext* context) noexcept {
    try {
        state = State{}; state.context = context;
        state.mode = Env("FAULT28_FA_MODE"); state.token = Env("FAULT28_FA_SCOPE_TOKEN");
        state.forward = Env("FAULT28_FA_SCOPE_FORWARD"); state.path = Env("FAULT28_FA_LOG_PATH");
        state.active = context && (state.mode == "off" || state.mode == "on") && Token(state.token) && Decimal(state.forward)
            && Env("FAULT28_FA_SCOPE_PID") == std::to_string(getpid())
            && Env("FAULT28_FA_SCOPE_TID") == std::to_string(syscall(SYS_gettid))
            && Env("FAULT28_FA_SCOPE_LAYER") == "1" && Env("FAULT28_FA_SCOPE_PHASE") == "prefill"
            && !state.path.empty() && state.path[0] == '/';
    } catch (...) { state.active = false; std::fprintf(stderr,"FAULT28_FA_BEGIN_ERROR\n"); }
}
inline bool Shape(const gert::StorageShape* p, std::initializer_list<int64_t> dims) {
    if (!p || p->GetStorageShape().GetDimNum() != dims.size()) return false;
    size_t i=0; for(auto d:dims)if(p->GetStorageShape().GetDim(i++) != d)return false;return true;
}
inline bool OptionalAbsent(gert::TilingContext* c, size_t i) { const auto* p=c->GetOptionalInputShape(i);return !p || p->GetStorageShape().GetShapeSize()==0; }
inline bool Eligible(gert::TilingContext* c) noexcept {
    try {
        if (!state.active || state.context != c) return false;
        const auto* a=c->GetAttrs(); if(!a || a->GetAttrNum()<8)return false;
        const auto *q=c->GetInputDesc(0), *k=c->GetInputDesc(1), *v=c->GetInputDesc(2), *mask=c->GetOptionalInputDesc(6);
        if(!q||!k||!v||!mask||q->GetDataType()!=ge::DT_FLOAT16||k->GetDataType()!=ge::DT_FLOAT16||v->GetDataType()!=ge::DT_FLOAT16||mask->GetDataType()!=ge::DT_BOOL)return false;
        if(!Shape(c->GetInputShape(0),{8,16,2048,128}) || !Shape(c->GetInputShape(1),{8,8,2048,128}) || !Shape(c->GetInputShape(2),{8,8,2048,128}) || !Shape(c->GetOptionalInputShape(6),{2048,2048}))return false;
        const auto *scale=a->GetAttrPointer<float>(0), *keep=a->GetAttrPointer<float>(1);
        const auto *pre=a->GetAttrPointer<int64_t>(2), *next=a->GetAttrPointer<int64_t>(3);
        const auto *heads=a->GetAttrPointer<uint32_t>(4); const auto *layout=a->GetAttrPointer<char>(5);
        const auto *precision=a->GetAttrPointer<uint8_t>(6); const auto *sparse=a->GetAttrPointer<int64_t>(7);
        if(!scale||!keep||!pre||!next||!heads||!layout||!precision||!sparse)return false;
        if(std::strcmp(layout,"BNSD") || *heads!=16 || *precision!=0 || *sparse!=2 || *keep!=1.0f || *pre!=2147483647LL || *next!=0 || !std::isfinite(*scale) || std::fabs(*scale-static_cast<float>(1.0/std::sqrt(128.0)))>1e-7f)return false;
        for(size_t i:{3U,4U,5U,7U,8U,9U,10U,11U,15U,16U,17U})if(!OptionalAbsent(c,i))return false;
        state.eligible=true;return true;
    } catch (...) { std::fprintf(stderr,"FAULT28_FA_ELIGIBILITY_ERROR\n"); return false; }
}
inline bool On(gert::TilingContext* c) noexcept { return Eligible(c) && state.mode == "on"; }
inline void Applied() noexcept { state.applied=true; }
inline void L1Reuse(bool enabled) noexcept { if(state.active){state.l1Known=true;state.l1Reuse=enabled;} }
inline void Typed(const std::string& json, uint64_t workspace) noexcept {
    try { if(state.active){state.typed=true;state.typedJson=json;state.workspace=workspace;} }catch(...){std::fprintf(stderr,"FAULT28_FA_TYPED_ERROR\n");}
}
inline void End(gert::TilingContext* c, uint32_t status) noexcept {
    try {
        if(!state.active || state.context!=c)return;
        const bool successful = status == 0;
        auto* raw=successful?c->GetRawTilingData():nullptr; const size_t n=raw?raw->GetDataSize():0;
        if(n>1024*1024 || (raw && n>raw->GetCapacity()))throw std::runtime_error("tiling raw exceeds capacity");
        const auto* data=raw?static_cast<const unsigned char*>(raw->GetData()):nullptr;
        std::ostringstream o; o << "{\"schema\":\"fault28.fa.typed_host.v1\",\"pid\":" << getpid() << ",\"tid\":" << syscall(SYS_gettid)
          << ",\"token\":" << Quote(state.token) << ",\"forward\":" << Quote(state.forward) << ",\"layer\":1,\"phase\":\"prefill\",\"mode\":" << Quote(state.mode)
          << ",\"eligible\":" << (state.eligible?"true":"false") << ",\"policy_applied\":" << (state.applied?"true":"false") << ",\"callback_status\":" << status;
        if(successful) o << ",\"tiling_key\":" << Quote(std::to_string(c->GetTilingKey())) << ",\"block_dim\":" << c->GetBlockDim() << ",\"schedule_mode\":" << c->GetScheduleMode() << ",\"workspace_count\":" << c->GetWorkspaceNum();
        else o << ",\"tiling_key\":null,\"block_dim\":null,\"schedule_mode\":null,\"workspace_count\":null";
        o << ",\"workspace_sizes\":"; if(successful && state.typed)o << '[' << state.workspace << ']';else o << "null";
        o << ",\"typed\":" << (successful && state.typed?state.typedJson:"null") << ",\"raw_size\":";
        if(successful)o << n;else o << "null";
        o << ",\"raw_hex\":"; if(successful)o << '"';else o << "null";
        if(n && !data)throw std::runtime_error("null tiling raw");
        for(size_t i=0;i<n;++i)o << std::hex << std::setw(2) << std::setfill('0') << unsigned(data[i]);
        if(successful)o << '"';o << "}\n"; const auto line=o.str();
        int fd=open(state.path.c_str(),O_CREAT|O_APPEND|O_WRONLY|O_CLOEXEC,0600);
        if(fd<0)throw std::runtime_error("cannot open host evidence");
        if(flock(fd,LOCK_EX)!=0){close(fd);throw std::runtime_error("cannot lock host evidence");}
        size_t pos=0; while(pos<line.size()){auto k=write(fd,line.data()+pos,line.size()-pos);if(k<0 && errno==EINTR)continue;if(k<=0)break;pos+=static_cast<size_t>(k);}
        flock(fd,LOCK_UN);close(fd);if(pos!=line.size())throw std::runtime_error("short host evidence write");
        state.active=false;
    }catch(...){state.active=false;std::fprintf(stderr,"FAULT28_FA_CAPTURE_ERROR\n");}
}
} // namespace fault28fa
#endif
