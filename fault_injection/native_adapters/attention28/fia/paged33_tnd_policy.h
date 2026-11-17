#ifndef DEMONCAT_PAGED33_TND_POLICY_H
#define DEMONCAT_PAGED33_TND_POLICY_H
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <unistd.h>
#include <sys/syscall.h>
#include <fcntl.h>
#include <sstream>
#include <iomanip>
#include <string>
#include <vector>
#include <stdexcept>

namespace paged33_tnd {
struct State {
    bool scoped=false,eligible=false,changed=false;
    uint32_t nativeS2=0,selectedS2=0;
    std::string mode,token,path;
};
inline std::string Env(const char* name){const char* p=std::getenv(name);return p?p:"";}
inline bool Token(const std::string& s){return s.size()==32&&s.find_first_not_of("0123456789abcdef")==std::string::npos;}
inline bool Shape(const gert::StorageShape* p,std::initializer_list<int64_t> dims){
    if(!p||p->GetStorageShape().GetDimNum()!=dims.size())return false;
    size_t i=0;for(auto d:dims)if(p->GetStorageShape().GetDim(i++)!=d)return false;return true;
}
inline std::string Quote(const std::string& s){
    std::ostringstream o;o<<'"';for(unsigned char c:s){if(c=='"'||c=='\\')o<<'\\'<<c;
        else if(c<32)o<<"\\u"<<std::hex<<std::setw(4)<<std::setfill('0')<<unsigned(c)<<std::dec;else o<<c;}o<<'"';return o.str();
}
inline std::string Hex(const void* data,size_t n){
    const auto* p=static_cast<const uint8_t*>(data);const char* a="0123456789abcdef";std::string out;
    for(size_t i=0;i<n;++i){out+=a[p[i]>>4];out+=a[p[i]&15];}return out;
}
inline State Select(const optiling::FiaTilingInfo& i,gert::TilingContext*,uint32_t realS2,uint32_t nativeS2){
    State s;s.nativeS2=s.selectedS2=nativeS2;s.mode=Env("DCAT33_TND_MODE");
    s.token=Env("DCAT33_TND_TOKEN");s.path=Env("DCAT33_TND_LOG");
    s.scoped=(s.mode=="off"||s.mode=="s2_256")&&Token(s.token)
        &&Env("DCAT33_TND_PID")==std::to_string(getpid())
        &&Env("DCAT33_TND_TID")==std::to_string(syscall(SYS_gettid))
        &&s.path.rfind("/evidence/",0)==0&&s.path.find("/../")==std::string::npos;
    if(!s.scoped)return s;
    const auto* q=i.opParamInfo.actualSeqLengthsQ.tensor;
    const auto* kv=i.opParamInfo.actualSeqLengths.tensor;
    const bool lengths=q&&kv&&q->GetShapeSize()==1&&kv->GetShapeSize()==1
        &&q->GetData<int64_t>()&&kv->GetData<int64_t>()&&q->GetData<int64_t>()[0]==1
        &&kv->GetData<int64_t>()[0]>=1025&&kv->GetData<int64_t>()[0]<=2048;
    s.eligible=i.socVersion==platform_ascendc::SocVersion::ASCEND910B
        &&i.bSize==1&&i.n1Size==16&&i.n2Size==8&&i.gSize==2
        &&i.qkHeadDim==128&&i.vHeadDim==128&&i.qTSize==1
        &&i.inputLayout==optiling::TilingKeyLayout::TND
        &&i.inputKvLayout==optiling::TilingKeyLayout::BSH_BSND
        &&i.pageAttentionFlag&&i.blockSize==16&&lengths
        &&Shape(i.opParamInfo.query.shape,{1,16,128})
        &&i.inputQType==ge::DT_FLOAT16&&i.inputKvType==ge::DT_FLOAT16&&i.outputType==ge::DT_FLOAT16
        &&i.innerPrecise==0&&i.sparseMode==3&&i.scaleValue==0.08838834613561630249f
        &&i.quantMode==optiling::FiaQuantMode::NO_QUANT&&!i.quantFlag&&!i.antiQuantFlag&&!i.isOutQuantEnable
        &&i.ropeMode==optiling::RopeMode::NO_ROPE&&!i.sysPrefixFlag&&!i.pseShiftFlag
        &&!i.kvPaddingSizeFlag&&!i.qPaddingSizeFlag&&!i.learnableSinkFlag&&!i.slidingFlag
        &&!i.isMaxWorkspace&&!i.emptyTensorFlag&&nativeS2==512&&realS2>=1025;
    if(s.eligible&&s.mode=="s2_256"){s.selectedS2=256;s.changed=true;}
    return s;
}
void AppendTyped(std::ostream&,optiling::FusedInferAttentionScoreTilingData&);
inline void Capture(const State& s,gert::TilingContext* c,optiling::FusedInferAttentionScoreTilingData& d,uint64_t workspace){
    if(!s.scoped)return;
    auto* raw=c->GetRawTilingData();
    if(!raw||raw->GetDataSize()!=d.GetDataSize()||raw->GetDataSize()>65536)throw std::runtime_error("TND33_SERIALIZER_SIZE");
    std::string bytes=Hex(raw->GetData(),raw->GetDataSize());
    std::vector<uint64_t> again((d.GetDataSize()+7)/8);d.SaveToBuffer(again.data(),again.size()*8);
    if(bytes!=Hex(again.data(),raw->GetDataSize()))throw std::runtime_error("TND33_SERIALIZER_ROUNDTRIP");
    std::ostringstream o;o<<std::boolalpha<<std::setprecision(17);
    o<<"{\"schema\":\"demoncat.paged33.tnd_host.v1\",\"pid\":"<<getpid()<<",\"tid\":"<<syscall(SYS_gettid)
        <<",\"token\":"<<Quote(s.token)<<",\"mode\":"<<Quote(s.mode)<<",\"eligible\":"<<s.eligible
        <<",\"policy_applied\":"<<s.changed<<",\"native_s2\":"<<s.nativeS2<<",\"selected_s2\":"<<s.selectedS2
        <<",\"tiling_key\":"<<Quote(std::to_string(c->GetTilingKey()))<<",\"block_dim\":"<<c->GetBlockDim()
        <<",\"schedule_mode\":"<<c->GetScheduleMode()<<",\"workspace\":"<<workspace
        <<",\"raw_size\":"<<raw->GetDataSize()<<",\"raw_hex\":"<<Quote(bytes)<<",\"typed\":";
    AppendTyped(o,d);o<<",\"kernel_argument_raw_observed\":false}\n";
    const auto text=o.str();int fd=open(s.path.c_str(),O_WRONLY|O_CREAT|O_APPEND|O_NOFOLLOW|O_CLOEXEC,0600);
    if(fd<0)throw std::runtime_error("TND33_HOST_LOG_OPEN");
    const auto n=write(fd,text.data(),text.size());close(fd);
    if(n!=static_cast<ssize_t>(text.size()))throw std::runtime_error("TND33_HOST_LOG_WRITE");
}
}
#include "paged33_tnd_typed.h"
#endif
