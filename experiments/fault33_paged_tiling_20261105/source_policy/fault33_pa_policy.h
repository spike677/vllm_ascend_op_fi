#ifndef FAULT33_PA_POLICY_H
#define FAULT33_PA_POLICY_H
// Observation is through public host TilingContext, never executor memory.
#include "fault33_pa_selector.h"
#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <iomanip>
#include <initializer_list>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>
#include <sys/file.h>
#include <sys/syscall.h>
#include <time.h>
#include <unistd.h>
#include "exe_graph/runtime/tiling_context.h"
namespace fault33pa {
inline std::string Env(const char* name) { const char* p=std::getenv(name); return p?p:""; }
inline bool Decimal(const std::string& s) { return !s.empty() && s.size()<=9 && std::all_of(s.begin(),s.end(),[](char c){return c>='0'&&c<='9';}); }
inline bool Token(const std::string& s) { return !s.empty() && s.size()<=128 && std::all_of(s.begin(),s.end(),[](char c){return (c>='a'&&c<='z')||(c>='A'&&c<='Z')||(c>='0'&&c<='9')||c=='_'||c=='.'||c=='-';}); }
inline std::string Quote(const std::string& s) { std::ostringstream o;o<<'"';for(unsigned char c:s){if(c=='"'||c=='\\')o<<'\\'<<c;else if(c<32)o<<"\\u"<<std::hex<<std::setw(4)<<std::setfill('0')<<unsigned(c)<<std::dec;else o<<c;}o<<'"';return o.str(); }
inline uint64_t MonoNs() { timespec t{};clock_gettime(CLOCK_MONOTONIC,&t);return uint64_t(t.tv_sec)*1000000000ULL+uint64_t(t.tv_nsec); }
inline std::string Hex(const void* p,size_t n) { if(n&&!p)throw std::runtime_error("null tiling raw");std::ostringstream o;const auto* b=static_cast<const unsigned char*>(p);for(size_t i=0;i<n;++i)o<<std::hex<<std::setw(2)<<std::setfill('0')<<unsigned(b[i]);return o.str(); }
inline bool Shape(const gert::StorageShape* p,std::initializer_list<int64_t> dims) { if(!p||p->GetStorageShape().GetDimNum()!=dims.size())return false;size_t i=0;for(auto d:dims)if(p->GetStorageShape().GetDim(i++)!=d)return false;return true; }
inline bool TensorShape(const gert::Shape& s,std::initializer_list<int64_t> dims) { if(s.GetDimNum()!=dims.size())return false;size_t i=0;for(auto d:dims)if(s.GetDim(i++)!=d)return false;return true; }
// Native IFA ParseActualSeqLens uses the same public host tensor methods.
// Non-template wrapper avoids dependent GetData template lookup and rejects
// mixed lengths even when maxActualseq alone still equals the fixed length.
inline bool FixedActualLengths(const gert::Tensor* tensor,uint32_t batch,uint32_t length) {
    return tensor&&EqualLengths(tensor->GetData<int64_t>(),tensor->GetShapeSize(),batch,length);
}
struct State {
    gert::TilingContext* context=nullptr;
    bool active=false,eligible=false,applied=false,typed=false;
    uint32_t batch=0,length=0,block=0;
    uint64_t beginNs=0,workspace=0,checks=0,captures=0;
    std::string mode,token,call,path,typedJson,serializedHex,nativeChecksJson="{}",nativeValuesJson="{}";
    std::vector<std::string> reports;
};
inline thread_local State state;
inline void Begin(gert::TilingContext* c) noexcept {
    try {
        state=State{};state.context=c;state.beginNs=MonoNs();
        state.mode=Env("FAULT33_PA_MODE");state.token=Env("FAULT33_PA_SCOPE_TOKEN");
        state.call=Env("FAULT33_PA_CALL_ID");state.path=Env("FAULT33_PA_LOG_PATH");
        const auto b=Env("FAULT33_PA_BATCH"),l=Env("FAULT33_PA_LENGTH"),bs=Env("FAULT33_PA_BLOCK_SIZE");
        state.active=c&&ValidMode(state.mode)&&Token(state.token)&&Token(state.call)
            &&Env("FAULT33_PA_SCOPE_PID")==std::to_string(getpid())
            &&Env("FAULT33_PA_SCOPE_TID")==std::to_string(syscall(SYS_gettid))
            &&Env("FAULT33_PA_SCOPE_PHASE")=="decode"&&Env("FAULT33_PA_CALLER_DETERMINISTIC")=="1"
            &&Decimal(b)&&Decimal(l)&&Decimal(bs)&&!state.path.empty()&&state.path[0]=='/';
        if(state.active){state.batch=static_cast<uint32_t>(std::stoul(b));state.length=static_cast<uint32_t>(std::stoul(l));state.block=static_cast<uint32_t>(std::stoul(bs));
            state.active=state.batch==8&&state.length==16384&&state.block==128;}
    }catch(...){state.active=false;std::fprintf(stderr,"FAULT33_PA_BEGIN_ERROR\n");}
}
inline bool NativeScope(std::initializer_list<std::pair<const char*,bool>> checks,
                        std::initializer_list<std::pair<const char*,int64_t>> values) noexcept {
    try {
        bool passed=true;std::ostringstream c,v;c<<'{';v<<'{';bool comma=false;
        for(const auto& item:checks){passed=passed&&item.second;if(comma)c<<',';comma=true;c<<Quote(item.first)<<':'<<(item.second?"true":"false");}
        comma=false;for(const auto& item:values){if(comma)v<<',';comma=true;v<<Quote(item.first)<<':'<<item.second;}
        c<<'}';v<<'}';if(state.active){state.nativeChecksJson=c.str();state.nativeValuesJson=v.str();}return passed;
    }catch(...){std::fprintf(stderr,"FAULT33_PA_NATIVE_SCOPE_ERROR\n");return false;}
}
// These absence checks use the native IFA context's tensor pointers, not an
// invented generic convention that a rank-zero tensor is always absent.
template<class C> inline bool FixedInputs(const C& c) {
    const uint32_t count=(state.length+state.block-1)/state.block;
    bool ok=c.query.desc&&c.key.desc&&c.value.desc&&c.attenOut.desc
        &&c.query.desc->GetDataType()==ge::DT_FLOAT16&&c.key.desc->GetDataType()==ge::DT_FLOAT16
        &&c.value.desc->GetDataType()==ge::DT_FLOAT16&&c.attenOut.desc->GetDataType()==ge::DT_FLOAT16
        &&Shape(c.query.shape,{state.batch,16,1,128})
        &&Shape(c.key.shape,{int64_t(state.batch)*count,8,128,128})
        &&Shape(c.value.shape,{int64_t(state.batch)*count,8,128,128})
        &&Shape(c.attenOut.shape,{state.batch,16,1,128})
        &&c.blockTable.tensor&&TensorShape(c.blockTable.tensor->GetStorageShape(),{state.batch,count})
        &&c.blockTable.desc&&c.blockTable.desc->GetDataType()==ge::DT_INT32
        &&FixedActualLengths(c.actualSeqLengths.tensor,state.batch,state.length)
        &&c.layOut&&std::strcmp(c.layOut,"BNSD")==0
        &&c.numHeads&&*c.numHeads==16&&c.kvHeadNums&&*c.kvHeadNums==8
        &&c.blockSize&&*c.blockSize==128&&c.innerPrecise&&*c.innerPrecise==0
        &&c.scaleValue&&std::isfinite(*c.scaleValue)
        &&std::fabs(*c.scaleValue-static_cast<float>(1.0/std::sqrt(128.0)))<=1e-7f;
    for(const auto* tensor:{c.pseShift.tensor,c.attenMask.tensor,c.actualSeqLengthsQ.tensor,
        c.deqScale1.tensor,c.quantScale1.tensor,c.deqScale2.tensor,c.quantScale2.tensor,c.quantOffset2.tensor,
        c.antiquantScale.tensor,c.antiquantOffset.tensor,c.queryPaddingSize.tensor,c.kvPaddingSize.tensor,
        c.keyAntiquantScale.tensor,c.keyAntiquantOffset.tensor,c.valueAntiquantScale.tensor,c.valueAntiquantOffset.tensor,
        c.keySharedPrefix.tensor,c.valueSharedPrefix.tensor,c.actualSharedPrefixLen.tensor,
        c.queryRope.tensor,c.keyRope.tensor,c.keyRopeAntiquantScale.tensor,c.dequantScaleQuery.tensor,
        c.qStartIdx.tensor,c.kvStartIdx.tensor})ok=ok&&!tensor;
    return ok;
}
template<class C> inline uint32_t Select(uint32_t native,uint32_t seq,const C& c,bool nativeScope,
                                        uint32_t b,uint32_t actual,uint32_t bs,uint32_t perf) noexcept {
    try {
        if(!state.active)return native;
        ++state.checks;
        const bool fixed=FixedInputs(c);
        const bool equalLengths=FixedActualLengths(c.actualSeqLengths.tensor,state.batch,state.length);
        const bool eligible=fixed&&nativeScope&&b==state.batch&&actual==state.length&&bs==state.block;
        const uint32_t requested=RequestedInner(state.mode);
        const bool legal=state.mode=="off"||LegalInner(native,seq,bs,requested);
        state.eligible=eligible&&legal;
        const bool apply=state.eligible&&state.mode!="off";
        std::ostringstream o;o<<"{\"fixed_inputs\":"<<(fixed?"true":"false")<<",\"native_scope\":"<<(nativeScope?"true":"false")
          <<",\"native_checks\":"<<state.nativeChecksJson<<",\"native_values\":"<<state.nativeValuesJson
          <<",\"actual_lengths_all_equal\":"<<(equalLengths?"true":"false")
          <<",\"batch\":"<<b<<",\"actual_kv_length\":"<<actual<<",\"block_size\":"<<bs<<",\"perf_mode\":"<<perf
          <<",\"native_sinner\":"<<native<<",\"compute_seq_size\":"<<seq<<",\"requested_sinner\":"<<requested
          <<",\"legal_request\":"<<(legal?"true":"false")<<",\"eligible\":"<<(state.eligible?"true":"false")
          <<",\"applied\":"<<(apply?"true":"false")<<'}';state.reports.push_back(o.str());
        if(apply){state.applied=true;return requested;}return native;
    }catch(...){std::fprintf(stderr,"FAULT33_PA_SELECTOR_ERROR\n");return native;}
}
inline void Typed(const std::string& json,uint64_t workspace,gert::TilingContext& c) noexcept {
    try {if(!state.active)return;auto* raw=c.GetRawTilingData();if(!raw||raw->GetDataSize()>raw->GetCapacity()||raw->GetDataSize()>1024*1024)throw std::runtime_error("invalid serialized raw");
        state.typed=true;state.typedJson=json;state.workspace=workspace;++state.captures;state.serializedHex=Hex(raw->GetData(),raw->GetDataSize());
    }catch(...){state.typed=false;std::fprintf(stderr,"FAULT33_PA_TYPED_ERROR\n");}
}
inline void End(gert::TilingContext* c,uint32_t status) noexcept {
    try {
        if(!state.active||state.context!=c)return;
        const bool success=status==0;auto* raw=success?c->GetRawTilingData():nullptr;const size_t n=raw?raw->GetDataSize():0;
        if(n>1024*1024||(raw&&n>raw->GetCapacity()))throw std::runtime_error("invalid final raw");
        const auto hex=raw?Hex(raw->GetData(),n):"";
        std::ostringstream o;o<<"{\"schema\":\"fault33.pa.typed_host.v1\",\"observation\":\"final_public_tiling_context\",\"kernel_argument_raw_observed\":false"
          <<",\"pid\":"<<getpid()<<",\"tid\":"<<syscall(SYS_gettid)<<",\"token\":"<<Quote(state.token)<<",\"call_id\":"<<Quote(state.call)
          <<",\"phase\":\"decode\",\"mode\":"<<Quote(state.mode)<<",\"caller_declared_deterministic\":true,\"callback_begin_monotonic_ns\":"<<state.beginNs
          <<",\"callback_end_monotonic_ns\":"<<MonoNs()<<",\"callback_status\":"<<status
          <<",\"eligible\":"<<(state.eligible?"true":"false")<<",\"policy_applied\":"<<(state.applied?"true":"false")
          <<",\"selector_checks\":"<<state.checks<<",\"serialized_captures\":"<<state.captures<<",\"guard_reports\":[";
        for(size_t i=0;i<state.reports.size();++i){if(i)o<<',';o<<state.reports[i];}o<<']';
        if(success)o<<",\"tiling_key\":"<<Quote(std::to_string(c->GetTilingKey()))<<",\"block_dim\":"<<c->GetBlockDim()<<",\"schedule_mode\":"<<c->GetScheduleMode()<<",\"workspace_count\":"<<c->GetWorkspaceNum()<<",\"raw_size\":"<<n<<",\"raw_hex\":"<<Quote(hex);
        else o<<",\"tiling_key\":null,\"block_dim\":null,\"schedule_mode\":null,\"workspace_count\":null,\"raw_size\":null,\"raw_hex\":null";
        o<<",\"typed_matches_final_serialized_raw\":"<<(success&&state.typed&&state.serializedHex==hex?"true":"false")
          <<",\"workspace_sizes\":"<<(success&&state.typed?"["+std::to_string(state.workspace)+"]":"null")
          <<",\"typed\":"<<(success&&state.typed?state.typedJson:"null")<<"}\n";
        const auto line=o.str();int fd=open(state.path.c_str(),O_CREAT|O_APPEND|O_WRONLY|O_CLOEXEC,0600);if(fd<0)throw std::runtime_error("cannot open evidence");
        if(flock(fd,LOCK_EX)!=0){close(fd);throw std::runtime_error("cannot lock evidence");}
        size_t p=0;while(p<line.size()){auto k=write(fd,line.data()+p,line.size()-p);if(k<0&&errno==EINTR)continue;if(k<=0)break;p+=static_cast<size_t>(k);}flock(fd,LOCK_UN);close(fd);
        if(p!=line.size())throw std::runtime_error("short evidence write");state.active=false;
    }catch(...){state.active=false;std::fprintf(stderr,"FAULT33_PA_CAPTURE_ERROR\n");}
}
} // namespace fault33pa
#endif
