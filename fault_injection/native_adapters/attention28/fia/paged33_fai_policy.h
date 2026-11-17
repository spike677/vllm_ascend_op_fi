#ifndef DEMONCAT_PAGED33_FAI_POLICY_H
#define DEMONCAT_PAGED33_FAI_POLICY_H
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

namespace paged33_fai {
struct State {
    bool scoped=false,eligible=false,changed=false;
    uint32_t nativeS2=512,selectedS2=512,aic=0;
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
inline State Select(const optiling::FAInferContext& i,gert::TilingContext* c,uint32_t aic){
    State s;s.aic=aic;s.mode=Env("DCAT33_TND_MODE");s.token=Env("DCAT33_TND_TOKEN");s.path=Env("DCAT33_TND_LOG");
    s.scoped=(s.mode=="off"||s.mode=="s2_256")&&Token(s.token)
        &&Env("DCAT33_TND_PID")==std::to_string(getpid())
        &&Env("DCAT33_TND_TID")==std::to_string(syscall(SYS_gettid))
        &&s.path.rfind("/evidence/",0)==0&&s.path.find("/../")==std::string::npos;
    if(!s.scoped)return s;
    s.eligible=aic==16&&i.layout=="TND"&&i.pagedCacheFlag&&i.blockSize==16
        &&i.numHeads==16&&i.kvHeads==8&&i.embeddingSize==128&&i.embeddingSizeV==128
        &&i.batch==1&&i.maxQSeqlen==1&&i.maxKvSeqlen>=1025&&i.maxKvSeqlen<=2048
        &&i.qSeqlenList&&i.kvSeqlenList&&i.qSeqlenList[0]==1&&i.kvSeqlenList[0]==i.maxKvSeqlen
        &&Shape(c->GetInputShape(0),{1,16,128})
        &&c->GetInputDesc(0)->GetDataType()==ge::DT_FLOAT16
        &&c->GetInputDesc(1)->GetDataType()==ge::DT_FLOAT16
        &&c->GetInputDesc(2)->GetDataType()==ge::DT_FLOAT16
        &&i.dataType==optiling::DataType::FP16&&i.innerPrecise==0&&i.sparseMode==3
        &&i.scaleValue==0.08838834613561630249f
        &&(i.maskType==optiling::MaskType::NO_MASK||i.maskType==optiling::MaskType::MASK_SPEC)
        &&!i.lseFlag&&!i.learnableSinkFlag&&!i.isTilingSink&&!i.flashDecodeFlag&&!i.decodingFlag
        &&!i.kvcacheNzFlag&&i.pseQ==0&&i.pseKv==0;
    if(s.eligible&&s.mode=="s2_256"){s.selectedS2=256;s.changed=true;}
    return s;
}
void AppendTyped(std::ostream&,optiling::FAInferTilingData&);
inline void Capture(const State& s,gert::TilingContext* c,optiling::FAInferTilingData& d,uint64_t workspace){
    if(!s.scoped)return;
    auto* raw=c->GetRawTilingData();
    if(!raw||raw->GetDataSize()!=d.GetDataSize()||raw->GetDataSize()>65536)throw std::runtime_error("TND33_FAI_SERIALIZER_SIZE");
    std::string bytes=Hex(raw->GetData(),raw->GetDataSize());
    std::vector<uint64_t> again((d.GetDataSize()+7)/8);d.SaveToBuffer(again.data(),again.size()*8);
    if(bytes!=Hex(again.data(),raw->GetDataSize()))throw std::runtime_error("TND33_FAI_SERIALIZER_ROUNDTRIP");
    std::ostringstream o;o<<std::boolalpha<<std::setprecision(17);
    o<<"{\"schema\":\"demoncat.paged33.fai_host.v1\",\"pid\":"<<getpid()<<",\"tid\":"<<syscall(SYS_gettid)
        <<",\"token\":"<<Quote(s.token)<<",\"mode\":"<<Quote(s.mode)<<",\"eligible\":"<<s.eligible
        <<",\"policy_applied\":"<<s.changed<<",\"native_s2\":"<<s.nativeS2<<",\"selected_s2\":"<<s.selectedS2
        <<",\"aic\":"<<s.aic<<",\"route\":\"FAInferTiling\",\"native_route_or_decode_flags_changed\":false"
        <<",\"tiling_key\":"<<Quote(std::to_string(c->GetTilingKey()))<<",\"block_dim\":"<<c->GetBlockDim()
        <<",\"schedule_mode\":"<<c->GetScheduleMode()<<",\"workspace\":"<<workspace
        <<",\"raw_size\":"<<raw->GetDataSize()<<",\"raw_hex\":"<<Quote(bytes)<<",\"typed\":";
    AppendTyped(o,d);o<<",\"kernel_argument_raw_observed\":false}\n";
    const auto text=o.str();int fd=open(s.path.c_str(),O_WRONLY|O_CREAT|O_APPEND|O_NOFOLLOW|O_CLOEXEC,0600);
    if(fd<0)throw std::runtime_error("TND33_FAI_HOST_LOG_OPEN");
    const auto n=write(fd,text.data(),text.size());close(fd);
    if(n!=static_cast<ssize_t>(text.size()))throw std::runtime_error("TND33_FAI_HOST_LOG_WRITE");
}
}
#include "paged33_fai_typed.h"
#endif
