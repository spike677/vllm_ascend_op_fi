#ifndef DEMONCAT_PAGED33_ARCH22_BRIDGE_H
#define DEMONCAT_PAGED33_ARCH22_BRIDGE_H
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <iomanip>
#include <initializer_list>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
#include <unistd.h>
#include <sys/syscall.h>
#include "../../../fused_infer_attention_score/op_host/fused_infer_attention_score_tiling_index.h"

namespace paged33_arch22_bridge {
inline std::string Env(const char* name){const char* p=std::getenv(name);return p?p:"";}
inline bool Token(const std::string& s){return s.size()==32&&s.find_first_not_of("0123456789abcdef")==std::string::npos;}
inline bool Shape(const gert::StorageShape* p,std::initializer_list<int64_t> dims){
    if(!p||p->GetStorageShape().GetDimNum()!=dims.size())return false;
    size_t n=0;for(auto d:dims)if(p->GetStorageShape().GetDim(n++)!=d)return false;return true;
}
inline bool I64(gert::TilingContext* c,uint32_t index,int64_t value){
    const auto* p=c->GetAttrs()->GetAttrPointer<int64_t>(index);return p&&*p==value;
}
// A fixed normal implementation in a separate private package. Observation
// tokens never select its arithmetic: normal, OFF and post-unregister calls
// with these actual parameters use exactly the same unmodified native tiler.
inline bool Matches(gert::TilingContext* c){
    using namespace optiling;
    if(!c||!c->GetAttrs()||!c->GetPlatformInfo())return false;
    auto platform=platform_ascendc::PlatformAscendC(c->GetPlatformInfo());
    if(platform.GetSocVersion()!=platform_ascendc::SocVersion::ASCEND910B)return false;
    const auto* layout=c->GetAttrs()->GetAttrPointer<char>(ATTR_INPUT_LAYOUT_INDEX);
    const auto* scale=c->GetAttrs()->GetAttrPointer<float>(ATTR_SCALE_INDEX);
    const auto* lse=c->GetAttrs()->GetAttrPointer<bool>(ATTR_SOFTMAX_LSE_FLAG_INDEX);
    if(!layout||std::strcmp(layout,"TND")||!scale||*scale!=0.08838834613561630249f||!lse||*lse)return false;
    if(!I64(c,ATTR_N_INDEX,16)||!I64(c,ATTR_NUM_KV_HEADS_INDEX,8)||!I64(c,ATTR_BLOCK_SIZE_INDEX,16)
        ||!I64(c,ATTR_INNER_PRECISE_INDEX,0)||!I64(c,ATTR_SPARSE_MODE_INDEX,3))return false;
    if(!Shape(c->GetInputShape(QUERY_INDEX),{1,16,128}))return false;
    const auto* k=c->GetInputShape(KEY_INDEX);const auto* v=c->GetInputShape(VALUE_INDEX);
    if(!k||k->GetStorageShape().GetDimNum()!=3)return false;
    const auto blocks=k->GetStorageShape().GetDim(0);
    if(blocks<65||!Shape(k,{blocks,16,1024})||!Shape(v,{blocks,16,1024}))return false;
    for(uint32_t index:{QUERY_INDEX,KEY_INDEX,VALUE_INDEX}){
        const auto* desc=c->GetInputDesc(index);if(!desc||desc->GetDataType()!=ge::DT_FLOAT16)return false;
    }
    const auto* maskDesc=c->GetOptionalInputDesc(ATTEN_MASK_INDEX);
    const auto* tableDesc=c->GetOptionalInputDesc(BLOCK_TABLE_INDEX);
    const auto* table=c->GetOptionalInputShape(BLOCK_TABLE_INDEX);
    if(!Shape(c->GetOptionalInputShape(ATTEN_MASK_INDEX),{2048,2048})||!maskDesc||maskDesc->GetDataType()!=ge::DT_INT8
        ||!table||table->GetStorageShape().GetDimNum()!=2||table->GetStorageShape().GetDim(0)!=1
        ||!tableDesc||tableDesc->GetDataType()!=ge::DT_INT32)return false;
    const auto width=table->GetStorageShape().GetDim(1);
    if(width<65||width>128)return false;
    int64_t lengths[2]={0,0};size_t n=0;
    for(uint32_t index:{ACTUAL_SEQ_Q_INDEX,ACTUAL_SEQ_KV_INDEX}){
        const auto* tensor=c->GetOptionalInputTensor(index);const auto* desc=c->GetOptionalInputDesc(index);
        if(!tensor||!desc||desc->GetDataType()!=ge::DT_INT64||tensor->GetShapeSize()!=1||!tensor->GetData<int64_t>())return false;
        lengths[n++]=tensor->GetData<int64_t>()[0];
    }
    if(lengths[0]!=1||lengths[1]<1025||lengths[1]>2048||width<(lengths[1]+15)/16)return false;
    // The complete input index range is pinned to the actual c77 public host
    // declaration, not inferred from the stock runtime argument ABI.
    for(uint32_t index=PSE_SHIFT_INDEX;index<=KV_START_IDX_INDEX;++index){
        if(index==ATTEN_MASK_INDEX||index==ACTUAL_SEQ_Q_INDEX||index==ACTUAL_SEQ_KV_INDEX||index==BLOCK_TABLE_INDEX)continue;
        if(c->GetOptionalInputShape(index)||c->GetOptionalInputTensor(index))return false;
    }
    return true;
}
inline std::string Quote(const std::string& s){
    std::ostringstream o;o<<'"';for(unsigned char c:s){if(c=='"'||c=='\\')o<<'\\'<<c;
        else if(c<32)o<<"\\u"<<std::hex<<std::setw(4)<<std::setfill('0')<<unsigned(c)<<std::dec;else o<<c;}o<<'"';return o.str();
}
inline std::string Hex(const void* data,size_t n){
    const auto* p=static_cast<const uint8_t*>(data);const char* a="0123456789abcdef";std::string out;
    for(size_t i=0;i<n;++i){out+=a[p[i]>>4];out+=a[p[i]&15];}return out;
}
void AppendTyped(std::ostream&,optiling::FusedInferAttentionScoreTilingData&);
inline void Capture(const optiling::FiaTilingInfo& i,gert::TilingContext* c,
    optiling::FusedInferAttentionScoreTilingData& d,uint64_t workspace,uint32_t nativeS2){
    const auto mode=Env("DCAT33_TND_MODE"),token=Env("DCAT33_TND_TOKEN"),path=Env("DCAT33_TND_LOG");
    if(mode!="arch22_normal"||!Token(token)||Env("DCAT33_TND_PID")!=std::to_string(getpid())
        ||Env("DCAT33_TND_TID")!=std::to_string(syscall(SYS_gettid))
        ||path.rfind("/evidence/",0)!=0||path.find("/../")!=std::string::npos)return;
    if(!Matches(c)||i.isMaxWorkspace||i.emptyTensorFlag||nativeS2!=512)
        throw std::runtime_error("ARCH22_NORMAL_ACTUAL_CONTEXT_DIFFERS");
    auto* raw=c->GetRawTilingData();
    if(!raw||raw->GetDataSize()!=d.GetDataSize()||raw->GetDataSize()>65536)
        throw std::runtime_error("ARCH22_NORMAL_SERIALIZER_SIZE");
    const auto bytes=Hex(raw->GetData(),raw->GetDataSize());
    std::vector<uint64_t> again((d.GetDataSize()+7)/8);d.SaveToBuffer(again.data(),again.size()*8);
    if(bytes!=Hex(again.data(),raw->GetDataSize()))throw std::runtime_error("ARCH22_NORMAL_SERIALIZER_ROUNDTRIP");
    std::ostringstream o;o<<std::boolalpha<<std::setprecision(17);
    o<<"{\"schema\":\"demoncat.paged33.arch22_normal_host.v1\",\"pid\":"<<getpid()<<",\"tid\":"<<syscall(SYS_gettid)
        <<",\"token\":"<<Quote(token)<<",\"mode\":"<<Quote(mode)<<",\"route\":\"FiaTilingNonQuant\""
        <<",\"normal_route_preference_changed\":true,\"eligible\":true,\"policy_applied\":false"
        <<",\"native_s2\":"<<nativeS2<<",\"selected_s2\":"<<nativeS2
        <<",\"tiling_key\":"<<Quote(std::to_string(c->GetTilingKey()))<<",\"block_dim\":"<<c->GetBlockDim()
        <<",\"schedule_mode\":"<<c->GetScheduleMode()<<",\"workspace\":"<<workspace
        <<",\"raw_size\":"<<raw->GetDataSize()<<",\"raw_hex\":"<<Quote(bytes)<<",\"typed\":";
    AppendTyped(o,d);o<<",\"kernel_argument_raw_observed\":false}\n";
    const auto text=o.str();int fd=open(path.c_str(),O_WRONLY|O_CREAT|O_APPEND|O_NOFOLLOW|O_CLOEXEC,0600);
    if(fd<0)throw std::runtime_error("ARCH22_NORMAL_LOG_OPEN");
    const auto size=write(fd,text.data(),text.size());close(fd);
    if(size!=static_cast<ssize_t>(text.size()))throw std::runtime_error("ARCH22_NORMAL_LOG_WRITE");
}
}
#include "paged33_arch22_typed.h"
#endif
