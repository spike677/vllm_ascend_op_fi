#ifndef DEMONCAT_PAGED33_ARCH22_ALL_NORMAL_H
#define DEMONCAT_PAGED33_ARCH22_ALL_NORMAL_H
#include "paged33_arch22_bridge.h"

// Supplied only by the separately hashed LD_PRELOAD observer. A mode variable
// alone never selects a tile, even on the right PID/thread and geometry.
extern "C" int dcat_attention_observer_scope_matches_v1(const char*) __attribute__((weak));
namespace paged33_arch22_all_normal {
using paged33_arch22_bridge::Env;
using paged33_arch22_bridge::Token;
using paged33_arch22_bridge::Quote;
using paged33_arch22_bridge::Hex;
struct State {
    bool scoped=false,changed=false;
    uint32_t nativeS2=0,selectedS2=0,realS2=0,aic=0,aiv=0;
    int64_t actualKv=0;
    std::string mode,token,path,owner,generation,phase,target,targetSet,objectId,ordinal;
};
inline State Select(const optiling::FiaTilingInfo& i,gert::TilingContext* c,
    uint32_t realS2,uint32_t nativeS2,uint32_t aic,uint32_t aiv){
    State s;s.nativeS2=s.selectedS2=nativeS2;s.realS2=realS2;s.aic=aic;s.aiv=aiv;
    s.mode=Env("DCAT33_TND_MODE");s.token=Env("DCAT33_TND_TOKEN");s.path=Env("DCAT33_TND_LOG");
    s.owner=Env("DCAT33_TND_OWNER");s.generation=Env("DCAT33_TND_GENERATION");
    s.phase=Env("DCAT33_TND_PHASE");s.target=Env("DCAT33_TND_TARGET");
    s.targetSet=Env("DCAT33_TND_TARGET_SET");s.objectId=Env("DCAT33_TND_OBJECT");s.ordinal=Env("DCAT33_TND_ORDINAL");
    s.scoped=s.mode=="arch22_all_normal"&&Token(s.token)
        &&Token(s.owner)&&!s.generation.empty()&&s.generation.size()<=18&&s.generation[0]!='0'
        &&s.generation.find_first_not_of("0123456789")==std::string::npos
        &&!s.target.empty()&&s.target.size()<512
        &&s.target.find_first_not_of("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._")==std::string::npos
        &&s.targetSet.size()==64&&s.targetSet.find_first_not_of("0123456789abcdef")==std::string::npos
        &&!s.objectId.empty()&&s.objectId.size()<=20&&s.objectId[0]!='0'
        &&s.objectId.find_first_not_of("0123456789")==std::string::npos
        &&!s.ordinal.empty()&&s.ordinal.size()<=2&&s.ordinal.find_first_not_of("0123456789")==std::string::npos
        &&std::stoul(s.ordinal)<28&&std::to_string(std::stoul(s.ordinal))==s.ordinal
        &&(s.phase=="baseline"||s.phase=="off")
        &&Env("DCAT33_TND_PID")==std::to_string(getpid())
        &&Env("DCAT33_TND_TID")==std::to_string(syscall(SYS_gettid))
        &&s.path.rfind("/evidence/",0)==0&&s.path.find("/../")==std::string::npos;
    if(!s.scoped)return s;
    if(!dcat_attention_observer_scope_matches_v1||dcat_attention_observer_scope_matches_v1(s.token.c_str())!=1)
        throw std::runtime_error("ARCH22_ALL_NORMAL_ACTUAL_OBSERVER_SCOPE_REQUIRED");
    if(!paged33_arch22_bridge::Matches(c)||i.isMaxWorkspace||i.emptyTensorFlag
        ||i.quantMode!=optiling::FiaQuantMode::NO_QUANT||i.quantFlag||i.antiQuantFlag||i.isOutQuantEnable
        ||i.ropeMode!=optiling::RopeMode::NO_ROPE||i.sysPrefixFlag||i.pseShiftFlag
        ||i.kvPaddingSizeFlag||i.qPaddingSizeFlag||i.learnableSinkFlag||i.slidingFlag
        ||i.bSize!=1||i.n1Size!=16||i.n2Size!=8||i.gSize!=2||i.s1Size!=1||i.s2Size!=realS2
        ||i.qkHeadDim!=128||i.vHeadDim!=128||i.ropeHeadDim!=0
        ||i.inputLayout!=optiling::TilingKeyLayout::TND||!i.pageAttentionFlag||i.blockSize!=16
        ||!i.attenMaskFlag||i.sparseMode!=3||i.preToken!=2147483647||i.nextToken!=0
        ||i.softmaxLseFlag||i.isLegacyIfa||i.innerPrecise!=0
        ||i.inputQType!=ge::DT_FLOAT16||i.inputKvType!=ge::DT_FLOAT16||i.outputType!=ge::DT_FLOAT16
        ||nativeS2!=512||realS2<1025||realS2>2048||aic<16||aic>26||aiv!=2*aic)
        throw std::runtime_error("ARCH22_ALL_NORMAL_ACTUAL_CONTEXT_DIFFERS");
    const auto* seq=i.opParamInfo.actualSeqLengths.tensor;
    if(!seq||seq!=c->GetOptionalInputTensor(optiling::ACTUAL_SEQ_KV_INDEX)
        ||seq->GetShapeSize()!=1||!seq->GetData<int64_t>()||i.actualLenKvDims!=1||i.actualLenQDims!=1)
        throw std::runtime_error("ARCH22_ALL_NORMAL_ACTUAL_SEQUENCE_OBJECT_DIFFERS");
    s.actualKv=seq->GetData<int64_t>()[0];
    if(realS2!=static_cast<uint32_t>((s.actualKv+15)/16*16))
        throw std::runtime_error("ARCH22_ALL_NORMAL_ACTUAL_CAPACITY_DIFFERS");
    return s;
}
inline void Capture(const State& s,const optiling::FiaTilingInfo& i,gert::TilingContext* c,
    optiling::FusedInferAttentionScoreTilingData& d,uint64_t workspace,uint64_t libapi,
    uint32_t loops,uint32_t tail,uint32_t aligned,uint32_t m){
    if(!s.scoped)return;
    if(Env("DCAT33_TND_MODE")!=s.mode||Env("DCAT33_TND_TOKEN")!=s.token||Env("DCAT33_TND_LOG")!=s.path
        ||Env("DCAT33_TND_OWNER")!=s.owner||Env("DCAT33_TND_GENERATION")!=s.generation
        ||Env("DCAT33_TND_PHASE")!=s.phase||Env("DCAT33_TND_TARGET")!=s.target
        ||Env("DCAT33_TND_TARGET_SET")!=s.targetSet||Env("DCAT33_TND_OBJECT")!=s.objectId
        ||Env("DCAT33_TND_ORDINAL")!=s.ordinal
        ||Env("DCAT33_TND_PID")!=std::to_string(getpid())
        ||Env("DCAT33_TND_TID")!=std::to_string(syscall(SYS_gettid))
        ||!dcat_attention_observer_scope_matches_v1||dcat_attention_observer_scope_matches_v1(s.token.c_str())!=1
        ||!paged33_arch22_bridge::Matches(c)||i.isMaxWorkspace||i.emptyTensorFlag
        ||m!=512||d.innerSplitParams.get_mBaseSize()!=512
        ||d.innerSplitParams.get_s2BaseSize()!=s.selectedS2
        ||loops!=(s.realS2+s.selectedS2-1)/s.selectedS2
        ||tail!=s.realS2-(loops-1)*s.selectedS2||aligned!=s.selectedS2)
        throw std::runtime_error("ARCH22_ALL_NORMAL_FINAL_OWNERSHIP_OR_GEOMETRY_DIFFERS");
    auto* raw=c->GetRawTilingData();
    if(!raw||raw->GetDataSize()!=d.GetDataSize()||raw->GetDataSize()>65536)
        throw std::runtime_error("ARCH22_ALL_NORMAL_SERIALIZER_SIZE");
    const auto bytes=Hex(raw->GetData(),raw->GetDataSize());
    std::vector<uint64_t> again((d.GetDataSize()+7)/8);d.SaveToBuffer(again.data(),again.size()*8);
    if(bytes!=Hex(again.data(),raw->GetDataSize()))throw std::runtime_error("ARCH22_ALL_NORMAL_SERIALIZER_ROUNDTRIP");
    std::ostringstream o;o<<std::boolalpha<<std::setprecision(17);
    o<<"{\"schema\":\"demoncat.paged33.arch22_all_normal_host.v1\",\"pid\":"<<getpid()<<",\"tid\":"<<syscall(SYS_gettid)
        <<",\"token\":"<<Quote(s.token)<<",\"mode\":"<<Quote(s.mode)<<",\"route\":\"FiaTilingNonQuant\""
        <<",\"owner\":"<<Quote(s.owner)<<",\"generation\":"<<s.generation
        <<",\"phase\":"<<Quote(s.phase)<<",\"target_path\":"<<Quote(s.target)
        <<",\"target_set_sha256\":"<<Quote(s.targetSet)<<",\"module_object_id\":"<<s.objectId
        <<",\"target_ordinal\":"<<s.ordinal<<",\"normal_only\":true"
        <<",\"actual_observer_scope_matched\":true"
        <<",\"normal_route_preference_changed\":true,\"eligible\":true,\"policy_applied\":"<<s.changed
        <<",\"native_s2\":"<<s.nativeS2<<",\"selected_s2\":"<<s.selectedS2<<",\"actual_kv_length\":"<<s.actualKv
        <<",\"real_s2_capacity\":"<<s.realS2<<",\"s2_loops\":"<<loops<<",\"s2_tail\":"<<tail<<",\"s2_aligned\":"<<aligned
        <<",\"aic\":"<<s.aic<<",\"aiv\":"<<s.aiv<<",\"libapi_workspace\":"<<libapi
        <<",\"tiling_key\":"<<Quote(std::to_string(c->GetTilingKey()))<<",\"block_dim\":"<<c->GetBlockDim()
        <<",\"schedule_mode\":"<<c->GetScheduleMode()<<",\"workspace\":"<<workspace
        <<",\"raw_size\":"<<raw->GetDataSize()<<",\"raw_hex\":"<<Quote(bytes)<<",\"typed\":";
    paged33_arch22_bridge::AppendTyped(o,d);o<<",\"kernel_argument_raw_observed\":false}\n";
    const auto text=o.str();int fd=open(s.path.c_str(),O_WRONLY|O_CREAT|O_APPEND|O_NOFOLLOW|O_CLOEXEC,0600);
    if(fd<0)throw std::runtime_error("ARCH22_ALL_NORMAL_LOG_OPEN");
    const auto size=write(fd,text.data(),text.size());close(fd);
    if(size!=static_cast<ssize_t>(text.size()))throw std::runtime_error("ARCH22_ALL_NORMAL_LOG_WRITE");
}
}
#endif
