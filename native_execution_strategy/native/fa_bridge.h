#ifndef NES_FA_BRIDGE_H
#define NES_FA_BRIDGE_H
// Version-pinned typed strategy bridge. No tensor input/output reads or changes.
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
#include <vector>
#include <sys/file.h>
#include <sys/syscall.h>
#include <unistd.h>
#include "exe_graph/runtime/tiling_context.h"
#include "fa_plan.h"
namespace nesfa {
inline std::string Env(const char* name) { const char* p = std::getenv(name); return p ? p : ""; }
inline std::string Quote(const std::string& s) { std::ostringstream o; o << '"'; for(unsigned char c:s){if(c=='"'||c=='\\')o << '\\' << c;else if(c<32)o << "\\u" << std::hex << std::setw(4) << std::setfill('0') << unsigned(c) << std::dec;else o << c;}o << '"';return o.str(); }
struct State {
    gert::TilingContext* context=nullptr;
    Plan plan;
    bool active=false,eligible=false,typed=false,l1Known=false,l1Reuse=false;
    std::string path,typedJson;
    uint64_t workspace=0,eligibilityChecks=0;
    std::vector<std::string> guardReports;
    std::vector<FactorDecision> decisions;
};
inline thread_local State state;
inline void Begin(gert::TilingContext* context) noexcept {
    try {
        state=State{};state.context=context;
        const auto text=Env("NES_FA_PLAN_JSON");if(text.empty())return;
        state.plan=ParsePlan(text);state.path=Env("NES_FA_LOG_PATH");
        state.active=context && state.plan.pid==getpid() && state.plan.tid==syscall(SYS_gettid)
            && !state.path.empty() && state.path[0]=='/';
    } catch(const std::exception& e){state.active=false;std::fprintf(stderr,"NES_FA_PLAN_REJECTED: %s\n",e.what());}
      catch(...){state.active=false;std::fprintf(stderr,"NES_FA_PLAN_REJECTED\n");}
}
inline bool Shape(const gert::StorageShape* p, std::initializer_list<int64_t> dims) {
    if (!p || p->GetStorageShape().GetDimNum() != dims.size()) return false;
    size_t i=0; for(auto d:dims)if(p->GetStorageShape().GetDim(i++) != d)return false;return true;
}
// Only called for the fixed FA optional tensor inputs below. The pinned FA
// host treats rank-0 PSE/dropmask/sink and other tensor-option placeholders as
// absent (e.g. general.cpp:864,1568,1735,1775). This is NOT a generic rule for
// attributes or APIs where a scalar tensor is meaningful. Quantization-only
// scale option placeholders are checked only under this fixed FP16 scope;
// this helper does not establish absence semantics for FP8/quantized inputs.
inline bool NativeTensorOptionAbsent(const gert::StorageShape* p) {
    return !p || p->GetStorageShape().GetDimNum()==0 || p->GetStorageShape().GetShapeSize()==0;
}
inline bool OptionalAbsent(gert::TilingContext* c, size_t i) { return NativeTensorOptionAbsent(c->GetOptionalInputShape(i)); }
inline std::string ShapeJson(const gert::StorageShape* p) {
    if(!p)return "null";
    const auto& s=p->GetStorageShape(); const auto& o=p->GetOriginShape(); std::ostringstream x;
    x << "{\"rank\":" << s.GetDimNum() << ",\"elements\":" << s.GetShapeSize() << ",\"storage\":[";
    for(size_t i=0;i<s.GetDimNum();++i){if(i)x<<',';x<<s.GetDim(i);}x<<"],\"origin\":[";
    for(size_t i=0;i<o.GetDimNum();++i){if(i)x<<',';x<<o.GetDim(i);}x<<"]}";return x.str();
}
template<class T>inline std::string NumberJson(const T* p) { if(!p)return "null";std::ostringstream x;x<<std::setprecision(17)<<+*p;return x.str(); }
inline std::string PublicContextJson(gert::TilingContext* c) {
    std::ostringstream x;x<<"{\"ir_inputs\":[";
    for(size_t i=0;i<19;++i){
        if(i)x<<',';const auto* s=i<3?c->GetInputShape(i):c->GetOptionalInputShape(i);
        const auto* d=i<3?c->GetInputDesc(i):c->GetOptionalInputDesc(i);
        x<<"{\"ir_index\":"<<i<<",\"shape\":"<<ShapeJson(s)<<",\"dtype\":";
        if(d)x<<static_cast<int>(d->GetDataType());else x<<"null";x<<'}';
    }
    x<<"],\"attrs\":";const auto* a=c->GetAttrs();if(!a||a->GetAttrNum()<8){x<<"null}";return x.str();}
    const auto* layout=a->GetAttrPointer<char>(5);
    x<<"{\"scale\":"<<NumberJson(a->GetAttrPointer<float>(0))<<",\"keep_prob\":"<<NumberJson(a->GetAttrPointer<float>(1))
      <<",\"pre_tokens\":"<<NumberJson(a->GetAttrPointer<int64_t>(2))<<",\"next_tokens\":"<<NumberJson(a->GetAttrPointer<int64_t>(3))
      <<",\"heads\":"<<NumberJson(a->GetAttrPointer<uint32_t>(4))<<",\"layout\":"<<(layout?Quote(layout):"null")
      <<",\"precision\":"<<NumberJson(a->GetAttrPointer<uint8_t>(6))<<",\"sparse_mode\":"<<NumberJson(a->GetAttrPointer<int64_t>(7))<<"}}";
    return x.str();
}
inline bool Eligible(gert::TilingContext* c) noexcept {
    try {
        if (!state.active || state.context != c) return false;
        ++state.eligibilityChecks;std::vector<std::string> failures;
        auto check=[&](bool pass,const std::string& name){if(!pass)failures.push_back(name);};
        const auto* a=c->GetAttrs(); check(a && a->GetAttrNum()>=8,"attribute_count");
        const auto *q=c->GetInputDesc(0), *k=c->GetInputDesc(1), *v=c->GetInputDesc(2), *mask=c->GetOptionalInputDesc(6);
        check(q && q->GetDataType()==ge::DT_FLOAT16,"query_dtype");check(k && k->GetDataType()==ge::DT_FLOAT16,"key_dtype");
        check(v && v->GetDataType()==ge::DT_FLOAT16,"value_dtype");check(mask && mask->GetDataType()==ge::DT_BOOL,"mask_dtype");
        check(Shape(c->GetInputShape(0),{state.plan.requestCount,16,2048,128}),"query_shape");check(Shape(c->GetInputShape(1),{state.plan.requestCount,8,2048,128}),"key_shape");
        check(Shape(c->GetInputShape(2),{state.plan.requestCount,8,2048,128}),"value_shape");check(Shape(c->GetOptionalInputShape(6),{2048,2048}),"mask_shape");
        if(a && a->GetAttrNum()>=8){
            const auto *scale=a->GetAttrPointer<float>(0), *keep=a->GetAttrPointer<float>(1);
            const auto *pre=a->GetAttrPointer<int64_t>(2), *next=a->GetAttrPointer<int64_t>(3);
            const auto *heads=a->GetAttrPointer<uint32_t>(4); const auto *layout=a->GetAttrPointer<char>(5);
            const auto *precision=a->GetAttrPointer<uint8_t>(6); const auto *sparse=a->GetAttrPointer<int64_t>(7);
            check(layout && std::strcmp(layout,"BNSD")==0,"layout");check(heads && *heads==16,"head_count");
            check(precision && *precision==0,"precision");check(sparse && *sparse==2,"sparse_mode");check(keep && *keep==1.0f,"keep_probability");
            check(pre && *pre==2147483647LL,"pre_tokens");check(next && *next==0,"next_tokens");
            check(scale && std::isfinite(*scale) && *scale==static_cast<float>(1.0/std::sqrt(128.0)),"scale");
        }
        for(size_t i:{3U,4U,5U,7U,8U,9U,10U,11U,12U,13U,14U,15U,16U,17U,18U})check(OptionalAbsent(c,i),"optional_ir_"+std::to_string(i)+"_absent");
        std::ostringstream report;report<<"{\"check\":"<<state.eligibilityChecks<<",\"passed\":"<<(failures.empty()?"true":"false")<<",\"failed_guards\":[";
        for(size_t i=0;i<failures.size();++i){if(i)report<<',';report<<Quote(failures[i]);}
        report<<"],\"public_context\":"<<PublicContextJson(c)<<'}';state.guardReports.push_back(report.str());
        state.eligible=failures.empty();return state.eligible;
    } catch (...) { std::fprintf(stderr,"NES_FA_ELIGIBILITY_ERROR\n"); return false; }
}
inline void SelectFactors(gert::TilingContext* c,int64_t& m,int64_t& r) noexcept {
    try {
        if(!state.active || state.context!=c)return;
        const auto decision=DecideFactors(state.plan,Factors{m,r},Eligible(c));
        state.decisions.push_back(decision);
        if(decision.applied){m=decision.selected.m;r=decision.selected.r;}
    }catch(...){std::fprintf(stderr,"NES_FA_SELECT_REJECTED\n");}
}
inline void L1Reuse(bool enabled) noexcept { if(state.active){state.l1Known=true;state.l1Reuse=enabled;} }
inline void Typed(const std::string& json, uint64_t workspace) noexcept {
    try { if(state.active){state.typed=true;state.typedJson=json;state.workspace=workspace;} }catch(...){std::fprintf(stderr,"NES_FA_TYPED_ERROR\n");}
}
template<class Cube> inline std::string CubeJson(const Cube& c) {
    std::ostringstream x; x << "{";
    x << "\"usedCoreNum\":" << c.usedCoreNum;
    x << ",\"M\":" << c.M;
    x << ",\"N\":" << c.N;
    x << ",\"Ka\":" << c.Ka;
    x << ",\"Kb\":" << c.Kb;
    x << ",\"singleCoreM\":" << c.singleCoreM;
    x << ",\"singleCoreN\":" << c.singleCoreN;
    x << ",\"singleCoreK\":" << c.singleCoreK;
    x << ",\"baseM\":" << c.baseM;
    x << ",\"baseN\":" << c.baseN;
    x << ",\"baseK\":" << c.baseK;
    x << ",\"depthA1\":" << c.depthA1;
    x << ",\"depthB1\":" << c.depthB1;
    x << ",\"stepM\":" << c.stepM;
    x << ",\"stepN\":" << c.stepN;
    x << ",\"isBias\":" << c.isBias;
    x << ",\"transLength\":" << c.transLength;
    x << ",\"iterateOrder\":" << c.iterateOrder;
    x << ",\"shareMode\":" << c.shareMode;
    x << ",\"shareL1Size\":" << c.shareL1Size;
    x << ",\"shareL0CSize\":" << c.shareL0CSize;
    x << ",\"shareUbSize\":" << c.shareUbSize;
    x << ",\"batchM\":" << c.batchM;
    x << ",\"batchN\":" << c.batchN;
    x << ",\"singleBatchM\":" << c.singleBatchM;
    x << ",\"singleBatchN\":" << c.singleBatchN;
    x << ",\"stepKa\":" << c.stepKa;
    x << ",\"stepKb\":" << c.stepKb;
    x << ",\"depthAL1CacheUB\":" << c.depthAL1CacheUB;
    x << ",\"depthBL1CacheUB\":" << c.depthBL1CacheUB;
    x << ",\"dbL0A\":" << c.dbL0A;
    x << ",\"dbL0B\":" << c.dbL0B;
    x << ",\"dbL0C\":" << c.dbL0C;
    x << ",\"ALayoutInfoB\":" << c.ALayoutInfoB;
    x << ",\"ALayoutInfoS\":" << c.ALayoutInfoS;
    x << ",\"ALayoutInfoN\":" << c.ALayoutInfoN;
    x << ",\"ALayoutInfoG\":" << c.ALayoutInfoG;
    x << ",\"ALayoutInfoD\":" << c.ALayoutInfoD;
    x << ",\"BLayoutInfoB\":" << c.BLayoutInfoB;
    x << ",\"BLayoutInfoS\":" << c.BLayoutInfoS;
    x << ",\"BLayoutInfoN\":" << c.BLayoutInfoN;
    x << ",\"BLayoutInfoG\":" << c.BLayoutInfoG;
    x << ",\"BLayoutInfoD\":" << c.BLayoutInfoD;
    x << ",\"CLayoutInfoB\":" << c.CLayoutInfoB;
    x << ",\"CLayoutInfoS1\":" << c.CLayoutInfoS1;
    x << ",\"CLayoutInfoN\":" << c.CLayoutInfoN;
    x << ",\"CLayoutInfoG\":" << c.CLayoutInfoG;
    x << ",\"CLayoutInfoS2\":" << c.CLayoutInfoS2;
    x << ",\"BatchNum\":" << c.BatchNum;
    x << ",\"mxTypePara\":" << c.mxTypePara;
    x << "}";return x.str();
}
inline void End(gert::TilingContext* c, uint32_t status) noexcept {
    try {
        if(!state.active || state.context!=c)return;
        const bool successful = status == 0;
        auto* raw=successful?c->GetRawTilingData():nullptr; const size_t n=raw?raw->GetDataSize():0;
        if(n>1024*1024 || (raw && n>raw->GetCapacity()))throw std::runtime_error("tiling raw exceeds capacity");
        const auto* data=raw?static_cast<const unsigned char*>(raw->GetData()):nullptr;
        std::ostringstream hex;
        if(n && !data)throw std::runtime_error("null tiling raw");
        for(size_t i=0;i<n;++i)hex<<std::hex<<std::setw(2)<<std::setfill('0')<<unsigned(data[i]);
        Json o={{"schema","nes.fa.host_observation.v1"},{"pid",getpid()},{"tid",syscall(SYS_gettid)},
            {"token",state.plan.token},{"phase","operator_execution"},{"request_count",state.plan.requestCount},
            {"mode",state.plan.mode},{"plan",state.plan.document},{"eligible",state.eligible},
            {"callback_status",status},{"eligibility_checks",state.eligibilityChecks},
            {"guard_reports",Json::array()},{"selector_events",Json::array()},
            {"native_factors",nullptr},{"selected_factors",nullptr},{"selection_applied",false},
            {"selection_reason","selected_template_not_observed"},
            {"tiling_key",nullptr},{"block_dim",nullptr},{"schedule_mode",nullptr},{"workspace_count",nullptr},
            {"workspace_sizes",nullptr},{"typed",nullptr},{"raw_size",nullptr},{"raw_hex",nullptr},
            {"certificate_verified_by_host",false},{"new_build_runtime_validation_inherited",false}};
        for(const auto& report:state.guardReports)o["guard_reports"].push_back(Json::parse(report));
        for(const auto& d:state.decisions)o["selector_events"].push_back({{"native_factors",FactorsJson(d.native)},
            {"selected_factors",FactorsJson(d.selected)},{"selection_applied",d.applied},{"selection_reason",d.reason}});
        if(!state.decisions.empty()) {
            const auto& d=state.decisions.back();o["native_factors"]=FactorsJson(d.native);o["selected_factors"]=FactorsJson(d.selected);
            o["selection_applied"]=d.applied;o["selection_reason"]=d.reason;
        }
        if(successful){
            o["tiling_key"]=std::to_string(c->GetTilingKey());o["block_dim"]=c->GetBlockDim();o["schedule_mode"]=c->GetScheduleMode();
            o["workspace_count"]=c->GetWorkspaceNum();o["raw_size"]=n;o["raw_hex"]=hex.str();
            if(state.typed){o["typed"]=Json::parse(state.typedJson);o["workspace_sizes"]=Json::array({state.workspace});}
        }
        // Public v1 names; retain explicit aliases for existing raw/POD readers.
        o["context_guards"]=o["guard_reports"];
        o["reason"]=o["selection_reason"];
        o["key"]=o["tiling_key"];
        const auto line=o.dump()+"\n";
        int fd=open(state.path.c_str(),O_CREAT|O_APPEND|O_WRONLY|O_CLOEXEC,0600);
        if(fd<0)throw std::runtime_error("cannot open host evidence");
        if(flock(fd,LOCK_EX)!=0){close(fd);throw std::runtime_error("cannot lock host evidence");}
        size_t pos=0; while(pos<line.size()){auto k=write(fd,line.data()+pos,line.size()-pos);if(k<0 && errno==EINTR)continue;if(k<=0)break;pos+=static_cast<size_t>(k);}
        flock(fd,LOCK_UN);close(fd);if(pos!=line.size())throw std::runtime_error("short host evidence write");
        state.active=false;
    }catch(...){state.active=false;std::fprintf(stderr,"NES_FA_CAPTURE_ERROR\n");}
}
} // namespace nesfa
#endif
