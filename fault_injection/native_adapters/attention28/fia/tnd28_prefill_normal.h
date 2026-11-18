#ifndef DEMONCAT_TND28_PREFILL_NORMAL_H
#define DEMONCAT_TND28_PREFILL_NORMAL_H
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <fstream>
#include <iomanip>
#include <initializer_list>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
#include <unistd.h>
#include <sys/syscall.h>
#include "tnd28_prefill_contract.h"
#include "../../../fused_infer_attention_score/op_host/fused_infer_attention_score_tiling_index.h"

extern "C" int dcat_attention_observer_scope_matches_v1(const char*) __attribute__((weak));
namespace tnd28_prefill_normal {
inline std::string Env(const char* name) { const char* p=std::getenv(name); return p ? p : ""; }
inline bool HexId(const std::string& value, size_t size) {
    return value.size()==size && value.find_first_not_of("0123456789abcdef")==std::string::npos;
}
inline bool Decimal(const std::string& value, size_t maxSize, bool zero=false) {
    return !value.empty() && value.size()<=maxSize
        && value.find_first_not_of("0123456789")==std::string::npos
        && (value[0]!='0' || (zero && value=="0"));
}
inline bool Shape(const gert::StorageShape* shape, std::initializer_list<int64_t> dims) {
    if (!shape || shape->GetStorageShape().GetDimNum()!=dims.size()) return false;
    size_t n=0;
    for (auto dim:dims) if (shape->GetStorageShape().GetDim(n++)!=dim) return false;
    return true;
}
template<class T> inline bool Attr(gert::TilingContext* c, uint32_t index, T value) {
    const auto* p=c->GetAttrs()->GetAttrPointer<T>(index);
    return p && *p==value;
}
inline bool Length(gert::TilingContext* c, uint32_t index) {
    const auto* tensor=c->GetOptionalInputTensor(index);
    const auto* desc=c->GetOptionalInputDesc(index);
    return tensor && desc && desc->GetDataType()==ge::DT_INT64
        && Shape(c->GetOptionalInputShape(index),{1})
        && tensor->GetShapeSize()==1 && tensor->GetData<int64_t>()
        && tensor->GetData<int64_t>()[0]==2048;
}
// This predicate sees prepared host shapes, never the public Tensor's view.
// Its fixed normal route is independent of all observation environment values.
// Original parser validation, arithmetic and kernel source remain untouched.
inline bool Matches(gert::TilingContext* c) {
    using namespace optiling;
    if (!c || !c->GetAttrs() || !c->GetPlatformInfo()) return false;
    auto platform=platform_ascendc::PlatformAscendC(c->GetPlatformInfo());
    const auto aic=platform.GetCoreNumAic(), aiv=platform.GetCoreNumAiv();
    if (platform.GetSocVersion()!=platform_ascendc::SocVersion::ASCEND910B
        || aic==0 || aic>26 || aiv!=2*aic) return false;
    const auto* layout=c->GetAttrs()->GetAttrPointer<char>(ATTR_INPUT_LAYOUT_INDEX);
    if (!layout || std::strcmp(layout,"TND")
        || !Attr<int32_t>(c,ATTR_N_INDEX,16) || !Attr<int32_t>(c,ATTR_NUM_KV_HEADS_INDEX,8)
        || !Attr<int32_t>(c,ATTR_BLOCK_SIZE_INDEX,128) || !Attr<int32_t>(c,ATTR_INNER_PRECISE_INDEX,0)
        || !Attr<int32_t>(c,ATTR_SPARSE_MODE_INDEX,3)
        || !Attr<float>(c,ATTR_SCALE_INDEX,0.08838834613561630249f)
        || !Attr<bool>(c,ATTR_SOFTMAX_LSE_FLAG_INDEX,false)) return false;
    // Raw pre/next are required and recorded; the pinned sparse3 parser sets
    // effective (INT_MAX,0). They are not replaced by Python schema defaults.
    if (!c->GetAttrs()->GetAttrPointer<int64_t>(ATTR_PRE_TOKEN_INDEX)
        || !c->GetAttrs()->GetAttrPointer<int64_t>(ATTR_NEXT_TOKEN_INDEX)) return false;
    for (uint32_t index:{ATTR_ANTIQUANT_MODE_INDEX,ATTR_KEY_ANTIQUANT_MODE_INDEX,
                         ATTR_VALUE_ANTIQUANT_MODE_INDEX,ATTR_QUERY_QUANT_MODE_INDEX})
        if (!Attr<int64_t>(c,index,0)) return false;
    if (!Shape(c->GetInputShape(QUERY_INDEX),{2048,16,128})
        || !Shape(c->GetInputShape(KEY_INDEX),{2048,8,128})
        || !Shape(c->GetInputShape(VALUE_INDEX),{2048,8,128})
        || !Shape(c->GetOutputShape(ATTENTION_OUT_INDEX),{2048,16,128})) return false;
    for (uint32_t index:{QUERY_INDEX,KEY_INDEX,VALUE_INDEX}) {
        const auto* desc=c->GetInputDesc(index);
        if (!desc || desc->GetDataType()!=ge::DT_FLOAT16) return false;
    }
    const auto* mask=c->GetOptionalInputDesc(ATTEN_MASK_INDEX);
    const auto* out=c->GetOutputDesc(ATTENTION_OUT_INDEX);
    if (!out || out->GetDataType()!=ge::DT_FLOAT16 || !mask || mask->GetDataType()!=ge::DT_INT8
        || !Shape(c->GetOptionalInputShape(ATTEN_MASK_INDEX),{2048,2048})
        || !Length(c,ACTUAL_SEQ_Q_INDEX) || !Length(c,ACTUAL_SEQ_KV_INDEX)) return false;
    for (uint32_t index=PSE_SHIFT_INDEX; index<=KV_START_IDX_INDEX; ++index) {
        if (index==ATTEN_MASK_INDEX || index==ACTUAL_SEQ_Q_INDEX || index==ACTUAL_SEQ_KV_INDEX) continue;
        if (c->GetOptionalInputShape(index) || c->GetOptionalInputTensor(index)
            || c->GetOptionalInputDesc(index)) return false;
    }
    return true;
}
inline std::string Quote(const std::string& s) {
    std::ostringstream o; o<<'"';
    for (unsigned char c:s) {
        if (c=='"' || c=='\\') o<<'\\'<<c;
        else if (c<32) o<<"\\u"<<std::hex<<std::setw(4)<<std::setfill('0')<<unsigned(c)<<std::dec;
        else o<<c;
    }
    o<<'"'; return o.str();
}
inline std::string Hex(const void* data, size_t size) {
    const auto* p=static_cast<const uint8_t*>(data);
    const char* digits="0123456789abcdef"; std::string out; out.reserve(size*2);
    for (size_t n=0;n<size;++n) { out+=digits[p[n]>>4]; out+=digits[p[n]&15]; }
    return out;
}
inline std::string ProcessStart() {
    std::ifstream input("/proc/self/stat"); std::string line, field;
    if (!std::getline(input,line)) throw std::runtime_error("TND28_PROCESS_START_UNOBSERVED");
    const auto end=line.rfind(')');
    if (end==std::string::npos) throw std::runtime_error("TND28_PROCESS_STAT_INVALID");
    std::istringstream fields(line.substr(end+1));
    for (unsigned n=3;n<=22;++n) if (!(fields>>field)) throw std::runtime_error("TND28_PROCESS_STAT_SHORT");
    if (!Decimal(field,20)) throw std::runtime_error("TND28_PROCESS_START_INVALID");
    return field;
}
struct Scope {
    std::string mode,token,owner,generation,phase,target,targetSet,objectId,ordinal,path,start;
};
inline Scope CurrentScope() {
    Scope s;
    s.mode=Env("DCAT28_TND_MODE"); s.token=Env("DCAT28_TND_TOKEN"); s.owner=Env("DCAT28_TND_OWNER");
    s.generation=Env("DCAT28_TND_GENERATION"); s.phase=Env("DCAT28_TND_PHASE");
    s.target=Env("DCAT28_TND_TARGET"); s.targetSet=Env("DCAT28_TND_TARGET_SET");
    s.objectId=Env("DCAT28_TND_OBJECT"); s.ordinal=Env("DCAT28_TND_ORDINAL");
    s.path=Env("DCAT28_TND_LOG"); s.start=ProcessStart();
    const bool phase=s.phase=="baseline" || s.phase=="baseline_repeat" || s.phase=="off" || s.phase=="recovery_off";
    if (s.mode!="prefill_normal" || !HexId(s.token,32) || !HexId(s.owner,32)
        || !Decimal(s.generation,18) || !phase || s.target.empty() || s.target.size()>511
        || s.target.find_first_not_of("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._")!=std::string::npos
        || !HexId(s.targetSet,64) || !Decimal(s.objectId,20) || !Decimal(s.ordinal,2,true)
        || std::stoul(s.ordinal)>=28 || s.path.rfind("/evidence/",0)!=0 || s.path.size()>4096
        || s.path.find("/../")!=std::string::npos || s.path.find("/./")!=std::string::npos
        || s.path.back()=='/' || s.path.find('\n')!=std::string::npos || s.path.find('\r')!=std::string::npos
        || Env("DCAT28_TND_PID")!=std::to_string(getpid())
        || Env("DCAT28_TND_TID")!=std::to_string(syscall(SYS_gettid))
        || Env("DCAT28_TND_PROCESS_START")!=s.start
        || !dcat_attention_observer_scope_matches_v1
        || dcat_attention_observer_scope_matches_v1(s.token.c_str())!=1)
        throw std::runtime_error("TND28_NORMAL_CURRENT_OWNER_SCOPE_REQUIRED");
    return s;
}
inline bool SameScope(const Scope& a, const Scope& b) {
    return a.mode==b.mode && a.token==b.token && a.owner==b.owner && a.generation==b.generation
        && a.phase==b.phase && a.target==b.target && a.targetSet==b.targetSet && a.objectId==b.objectId
        && a.ordinal==b.ordinal && a.path==b.path && a.start==b.start;
}
inline bool ParsedGeometry(const optiling::FiaTilingInfo& i, gert::TilingContext* c) {
    using namespace optiling;
    return Matches(c) && !i.isMaxWorkspace && !i.emptyTensorFlag
        && i.quantMode==FiaQuantMode::NO_QUANT && !i.quantFlag && !i.antiQuantFlag && !i.isOutQuantEnable
        && i.ropeMode==RopeMode::NO_ROPE && !i.sysPrefixFlag && !i.pseShiftFlag
        && !i.kvPaddingSizeFlag && !i.qPaddingSizeFlag && !i.learnableSinkFlag && !i.slidingFlag
        && i.bSize==1 && i.n1Size==16 && i.n2Size==8 && i.gSize==2 && i.s1Size==2048 && i.s2Size==2048
        && i.qkHeadDim==128 && i.vHeadDim==128 && i.ropeHeadDim==0
        && i.inputLayout==TilingKeyLayout::TND && i.inputKvLayout==TilingKeyLayout::TND
        && i.outputLayout==TilingKeyLayout::TND && !i.pageAttentionFlag
        // The pinned parser reads raw block_size only in the page-attention
        // branch. No-page keeps its initialized blockSize_==0, not raw 128.
        && i.kvStorageMode==KvStorageMode::BATCH_CONTINUOUS && i.blockSize==0
        && i.attenMaskFlag && i.sparseMode==3 && i.preToken==2147483647 && i.nextToken==0
        && !i.softmaxLseFlag && !i.isLegacyIfa && i.innerPrecise==0
        && i.inputQType==ge::DT_FLOAT16 && i.inputKvType==ge::DT_FLOAT16 && i.outputType==ge::DT_FLOAT16
        && i.scaleValue==0.08838834613561630249f && i.actualLenQDims==1 && i.actualLenKvDims==1
        && i.isAccumQSeq && i.isAccumKVSeq
        && i.opParamInfo.query.shape==c->GetInputShape(QUERY_INDEX)
        && i.opParamInfo.key.shape==c->GetInputShape(KEY_INDEX)
        && i.opParamInfo.value.shape==c->GetInputShape(VALUE_INDEX)
        && i.opParamInfo.attenOut.shape==c->GetOutputShape(ATTENTION_OUT_INDEX)
        && i.opParamInfo.actualSeqLengthsQ.tensor==c->GetOptionalInputTensor(ACTUAL_SEQ_Q_INDEX)
        && i.opParamInfo.actualSeqLengths.tensor==c->GetOptionalInputTensor(ACTUAL_SEQ_KV_INDEX);
}
void AppendTyped(std::ostream&,optiling::FusedInferAttentionScoreTilingData&);
inline void Capture(const optiling::FiaTilingInfo& i, gert::TilingContext* c,
    optiling::FusedInferAttentionScoreTilingData& d, uint64_t workspace, uint64_t libapi,
    uint32_t aic, uint32_t aiv, uint32_t m, uint32_t s2, uint32_t loops,
    uint32_t tail, uint32_t aligned, bool flashDecode) {
    if (Env("DCAT28_TND_MODE").empty()) return;
    const auto scope=CurrentScope();
    if (!ParsedGeometry(i,c)) throw std::runtime_error("TND28_NORMAL_PARSED_CONTEXT_DIFFERS");
    NormalTiling n;
    n.key=c->GetTilingKey(); n.s1=i.s1Size; n.s2=static_cast<uint32_t>(i.s2Size);
    n.groups=i.gSize; n.kvHeads=i.n2Size; n.m=m; n.s2Base=s2;
    n.loops=loops; n.tail=tail; n.aligned=aligned; n.aic=aic; n.aiv=aiv; n.blockDim=c->GetBlockDim();
    n.preToken=i.preToken; n.nextToken=i.nextToken;
    n.rawBlockSize=*c->GetAttrs()->GetAttrPointer<int32_t>(optiling::ATTR_BLOCK_SIZE_INDEX);
    n.parsedBlockSize=i.blockSize;
    n.cumulativeQ=i.isAccumQSeq; n.cumulativeKv=i.isAccumKVSeq;
    n.flashDecode=flashDecode; n.brcb=(m<=128);
    auto platform=platform_ascendc::PlatformAscendC(c->GetPlatformInfo());
    n.usedCores=d.baseParams.get_usedCoreNum();
    if (n.usedCores==0 || n.usedCores>aic) throw std::runtime_error("TND28_NORMAL_USED_CORES_DIFFERS");
    n.calculatedBlockDim=platform.CalcTschBlockDim(2*n.usedCores,n.usedCores,2*n.usedCores);
    if (!NormalPostcondition(n) || aic!=platform.GetCoreNumAic() || aiv!=platform.GetCoreNumAiv()
        || d.innerSplitParams.get_mBaseSize()!=m || d.innerSplitParams.get_s2BaseSize()!=s2
        || d.pageAttenParams.get_blockSize()!=0
        || d.fdParams.get_numOfFdHead()!=0 || d.baseParams.get_accumQSeqFlag()!=1
        || d.baseParams.get_accumKVSeqFlag()!=1 || d.maskParams.get_preToken()!=2147483647
        || d.maskParams.get_nextToken()!=0 || d.maskParams.get_sparseMode()!=3)
        throw std::runtime_error("TND28_NORMAL_FINAL_TILING_DIFFERS");
    auto* raw=c->GetRawTilingData();
    if (!raw || !raw->GetData() || raw->GetDataSize()!=d.GetDataSize()
        || raw->GetDataSize()==0 || raw->GetDataSize()>65536)
        throw std::runtime_error("TND28_NORMAL_SERIALIZER_SIZE");
    const auto bytes=Hex(raw->GetData(),raw->GetDataSize());
    std::vector<uint64_t> again((d.GetDataSize()+7)/8);
    d.SaveToBuffer(again.data(),again.size()*8);
    if (bytes!=Hex(again.data(),raw->GetDataSize())) throw std::runtime_error("TND28_NORMAL_SERIALIZER_ROUNDTRIP");
    std::ostringstream o; o<<std::boolalpha<<std::setprecision(17);
    o<<"{\"schema\":\"demoncat.attention28.tnd_prefill_normal_host.v1\",\"pid\":"<<getpid()
        <<",\"tid\":"<<syscall(SYS_gettid)<<",\"process_start_ticks\":"<<Quote(scope.start)
        <<",\"mode\":"<<Quote(scope.mode)<<",\"token\":"<<Quote(scope.token)<<",\"owner\":"<<Quote(scope.owner)
        <<",\"generation\":"<<scope.generation<<",\"phase\":"<<Quote(scope.phase)<<",\"target_path\":"<<Quote(scope.target)
        <<",\"target_set_sha256\":"<<Quote(scope.targetSet)<<",\"module_object_id\":"<<scope.objectId
        <<",\"target_ordinal\":"<<scope.ordinal<<",\"route\":\"FiaTilingNonQuant\",\"normal_only\":true"
        <<",\"actual_observer_scope_matched\":true,\"normal_route_preference_changed\":true,\"policy_applied\":false"
        <<",\"native_m\":"<<m<<",\"native_s2\":"<<s2<<",\"s2_loops\":"<<loops<<",\"s2_tail\":"<<tail
        <<",\"s2_aligned\":"<<aligned<<",\"aic\":"<<aic<<",\"aiv\":"<<aiv<<",\"flash_decode\":false"
        <<",\"used_cores\":"<<n.usedCores<<",\"calculated_block_dim\":"<<n.calculatedBlockDim
        <<",\"raw_pre_token\":"<<*c->GetAttrs()->GetAttrPointer<int64_t>(optiling::ATTR_PRE_TOKEN_INDEX)
        <<",\"raw_next_token\":"<<*c->GetAttrs()->GetAttrPointer<int64_t>(optiling::ATTR_NEXT_TOKEN_INDEX)
        <<",\"raw_block_size\":"<<n.rawBlockSize<<",\"parsed_no_page_block_size\":"<<n.parsedBlockSize
        <<",\"actual_query_length\":2048,\"actual_kv_length\":2048,\"cumulative_q\":true,\"cumulative_kv\":true"
        <<",\"tensor_v1\":"<<i.isTensorV1<<",\"key_stride_metadata_present\":"<<(i.keyStrides!=nullptr)
        <<",\"value_stride_metadata_present\":"<<(i.valueStrides!=nullptr)
        <<",\"public_prepared_pointer_stride_join\":false,\"prepared_stride_values_verified\":false"
        <<",\"mask_values_verified\":false,\"stock_private_exact_proven\":false,\"fault_execution_ready\":false"
        <<",\"tiling_key\":"<<Quote(std::to_string(n.key))<<",\"block_dim\":"<<n.blockDim
        <<",\"schedule_mode\":"<<c->GetScheduleMode()<<",\"workspace\":"<<workspace<<",\"libapi_workspace\":"<<libapi
        <<",\"raw_size\":"<<raw->GetDataSize()<<",\"raw_hex\":"<<Quote(bytes)<<",\"typed\":";
    AppendTyped(o,d); o<<",\"kernel_argument_raw_observed\":false}\n";
    const auto text=o.str();
    if (text.size()>524288 || !SameScope(scope,CurrentScope()))
        throw std::runtime_error("TND28_NORMAL_CAPTURE_BUDGET_OR_OWNER_CHANGED");
    const int fd=open(scope.path.c_str(),O_WRONLY|O_CREAT|O_APPEND|O_NOFOLLOW|O_CLOEXEC,0600);
    if (fd<0) throw std::runtime_error("TND28_NORMAL_LOG_OPEN");
    const auto written=write(fd,text.data(),text.size()); const int closed=close(fd);
    if (written!=static_cast<ssize_t>(text.size()) || closed!=0)
        throw std::runtime_error("TND28_NORMAL_LOG_WRITE_OR_CLOSE");
}
} // namespace tnd28_prefill_normal
#include "tnd28_prefill_typed.h"
#endif
