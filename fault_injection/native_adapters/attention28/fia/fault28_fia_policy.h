#ifndef DEMONCAT_FAULT28_FIA_POLICY_H
#define DEMONCAT_FAULT28_FIA_POLICY_H
#include <cerrno>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <fstream>
#include <iomanip>
#include <initializer_list>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <sys/syscall.h>
#include <unistd.h>
#include <utility>
#include <vector>

namespace fault28_fia {
struct State {
    bool scoped=false, eligible=false, prune=false, changed=false;
    uint32_t nativeM=0, selectedM=0;
    uint32_t ownerPid=0;
    std::string ownerStart;
    std::vector<std::pair<uint32_t,std::string>> ancestry;
    uint64_t preExtensionKey=0;
    std::string mode, token, log, node;
    std::vector<std::pair<std::string,bool>> checks;
};
inline std::string Env(const char* name) { const char* p=std::getenv(name); return p?p:""; }
inline std::string Quote(const std::string& s) {
    std::ostringstream o; o<<'"';
    for (unsigned char c:s) {
        if (c=='"'||c=='\\') o<<'\\'<<c;
        else if (c<32) o<<"\\u"<<std::hex<<std::setw(4)<<std::setfill('0')<<unsigned(c)<<std::dec;
        else o<<c;
    }
    o<<'"'; return o.str();
}
inline bool SafeToken(const std::string& s) {
    return s.size()==32 && s.find_first_not_of("0123456789abcdef")==std::string::npos;
}
inline void Append(const State& state,const std::string& line) {
    if (!state.scoped) return;
    static std::mutex lock;
    std::lock_guard<std::mutex> guard(lock);
    int fd=::open(state.log.c_str(),O_CREAT|O_WRONLY|O_APPEND|O_CLOEXEC,0600);
    if (fd<0) throw std::runtime_error("fault28 audit open failed");
    const std::string data=line+"\n";
    const ssize_t wrote=::write(fd,data.data(),data.size());
    const int saved=errno; ::close(fd);
    if (wrote!=static_cast<ssize_t>(data.size())) { errno=saved;throw std::runtime_error("fault28 audit short write"); }
}
inline bool Shape(const gert::StorageShape* p,std::initializer_list<int64_t> dims) {
    if (!p||p->GetStorageShape().GetDimNum()!=dims.size()) return false;
    size_t j=0;for(auto x:dims) if(p->GetStorageShape().GetDim(j++)!=x) return false;
    return true;
}
inline bool Lengths(const gert::Tensor* p,uint32_t batch) {
    if (!p) return false;
    const size_t n=p->GetShapeSize();
    if (n!=1&&n!=batch) return false;
    const int64_t* data=p->GetData<int64_t>();if(!data)return false;
    for(size_t j=0;j<n;++j)if(data[j]!=2048)return false;
    return true;
}
inline bool OwnerAncestry(State& s) {
    const std::string id=Env("FAULT28_FIA_GRAPH_PID");
    s.ownerStart=Env("FAULT28_FIA_GRAPH_OWNER_START");
    if(id.empty()||id.find_first_not_of("0123456789")!=std::string::npos||s.ownerStart.empty())return false;
    char* end=nullptr;const unsigned long root=std::strtoul(id.c_str(),&end,10);
    if(!end||*end||root==0||root>2147483647UL)return false;
    s.ownerPid=uint32_t(root);uint32_t current=uint32_t(::getpid());
    // Optional public-Worker scope: never inherit into a child process or a
    // foreign thread. Historical standalone callers without TID keep their
    // original ancestry contract; new Worker adapters require these bytes.
    const std::string ownerTid=Env("FAULT28_FIA_GRAPH_TID");
    if(!ownerTid.empty() && (current!=s.ownerPid ||
        ownerTid!=std::to_string(::syscall(SYS_gettid))))return false;
    for(unsigned depth=0;depth<32&&current;++depth) {
        std::ifstream file("/proc/"+std::to_string(current)+"/stat");std::string line;
        if(!std::getline(file,line))return false;
        const auto close=line.rfind(')');if(close==std::string::npos)return false;
        std::istringstream in(line.substr(close+1));std::vector<std::string> fields;std::string v;
        while(in>>v)fields.push_back(v);
        if(fields.size()<20)return false;
        s.ancestry.emplace_back(current,fields[19]);
        if(current==s.ownerPid)return fields[19]==s.ownerStart;
        const auto parent=std::strtoul(fields[1].c_str(),&end,10);
        if(!end||*end||parent==current||parent>2147483647UL)return false;
        current=uint32_t(parent);
    }
    return false;
}
inline bool Coupled(const State& s) {
    return s.eligible && s.mode.find("coupled") != std::string::npos;
}
inline bool Replay(const State& s) {
    return s.eligible && s.selectedM == 128 && s.mode.find("replay") != std::string::npos;
}
inline State Select(const optiling::FiaTilingInfo& i,gert::TilingContext* c,
                    uint32_t aic,uint32_t aiv,uint32_t nativeM,uint32_t inner) {
    State s; s.nativeM=s.selectedM=nativeM;
    s.mode=Env("FAULT28_FIA_GRAPH_MODE");s.token=Env("FAULT28_FIA_GRAPH_TOKEN");
    s.log=Env("FAULT28_FIA_GRAPH_LOG");s.node=c->GetNodeName();
    const std::vector<std::string> modes={"off","m128","m256","causal_base512",
        "causal_m128","causal_m256","causal_m64","causal_m32",
        "batch_causal_m128","batch_causal_m256","batch_causal_m64","batch_causal_m32",
        "coupled_base512","coupled_m128","coupled_m256","coupled_m64","coupled_m32",
        "batch_coupled_m128","batch_coupled_m256","batch_coupled_m64","batch_coupled_m32", "batch_coupled_replay_m128", "coupled_replay_m128"};
    bool mode=false;for(const auto& value:modes)mode=mode||s.mode==value;
    s.scoped=mode&&SafeToken(s.token)&&OwnerAncestry(s)&&
             s.log.rfind("/evidence/",0)==0&&s.log.find(".. ")==std::string::npos&&s.log.find("/../")==std::string::npos;
    if(!s.scoped)return s;
    auto add=[&](const char* name,bool ok){s.checks.emplace_back(name,ok);};
    add("soc_910b",i.socVersion==platform_ascendc::SocVersion::ASCEND910B);
    add("core_budget_16_32",aic==16&&aiv==32);
    add("batch_1_or_8",i.bSize==1||i.bSize==8);
    add("heads_16_8_g2",i.n1Size==16&&i.n2Size==8&&i.gSize==2);
    add("query_and_kv_length_2048",i.s1Size==2048&&i.s2Size==2048);
    add("head_dims_128",i.qkHeadDim==128&&i.vHeadDim==128);
    add("dtype_fp16",i.inputQType==ge::DT_FLOAT16&&i.inputKvType==ge::DT_FLOAT16&&i.outputType==ge::DT_FLOAT16);
    add("bnsd_layouts",i.inputLayout==optiling::TilingKeyLayout::BNSD&&i.inputKvLayout==optiling::TilingKeyLayout::BNSD);
    add("query_shape",Shape(i.opParamInfo.query.shape,{i.bSize,16,2048,128}));
    add("key_shape",Shape(i.opParamInfo.key.shape,{i.bSize,8,2048,128}));
    add("value_shape",Shape(i.opParamInfo.value.shape,{i.bSize,8,2048,128}));
    add("query_actual_lengths",Lengths(i.opParamInfo.actualSeqLengthsQ.tensor,i.bSize));
    add("kv_actual_lengths",Lengths(i.opParamInfo.actualSeqLengths.tensor,i.bSize));
    add("non_accumulated_lengths",!i.isAccumQSeq&&!i.isAccumKVSeq);
    add("high_precision_0",i.innerPrecise==0);
    add("causal_mask",i.attenMaskFlag&&i.sparseMode==2&&i.preToken==2147483647&&i.nextToken==0);
    add("scale",i.scaleValue==0.08838834613561630249f);
    add("no_quant",i.quantMode==optiling::FiaQuantMode::NO_QUANT&&!i.quantFlag&&!i.antiQuantFlag&&!i.isOutQuantEnable);
    add("no_rope",i.ropeMode==optiling::RopeMode::NO_ROPE);
    add("no_page_or_prefix",!i.pageAttentionFlag&&!i.sysPrefixFlag);
    add("no_padding_pse_sink",!i.kvPaddingSizeFlag&&!i.qPaddingSizeFlag&&!i.pseShiftFlag&&!i.learnableSinkFlag);
    add("runtime_tiling",!i.isMaxWorkspace&&!i.emptyTensorFlag);
    add("native_m512_inner512",nativeM==512&&inner==512);
    s.eligible=true;for(const auto& v:s.checks)s.eligible=s.eligible&&v.second;
    if (s.eligible) {
        const bool batch=s.mode.rfind("batch_",0)==0;
        if(!batch||i.bSize==8) {
            for(uint32_t candidate:{32U,64U,128U,256U}) {
                const std::string suffix="m"+std::to_string(candidate);
                if(s.mode==suffix || (s.mode.size()>suffix.size() &&
                    s.mode.compare(s.mode.size()-suffix.size(),suffix.size(),suffix)==0))s.selectedM=candidate;
            }
        }
        s.prune=s.mode.find("causal")!=std::string::npos||s.mode.find("coupled")!=std::string::npos;
        s.changed=s.selectedM!=s.nativeM||s.prune;
    }
    return s;
}
inline std::string Hex(const void* data,size_t n) {
    const char* alphabet="0123456789abcdef";std::string s;s.reserve(n*2);
    auto* p=static_cast<const unsigned char*>(data);
    for(size_t j=0;j<n;++j){s.push_back(alphabet[p[j]>>4]);s.push_back(alphabet[p[j]&15]);}return s;
}
// Generated against the pinned public declarations; includes typed getter
// values only. Independent serializer offsets are measured separately.
void AppendTyped(std::ostream&,optiling::FusedInferAttentionScoreTilingData&);
inline void Capture(const State& s,gert::TilingContext* c,
                    optiling::FusedInferAttentionScoreTilingData& d,
                    uint64_t baseTemplateKey,uint64_t workspace,uint32_t aic,uint32_t aiv) {
    if(!s.scoped)return;
    auto* raw=c->GetRawTilingData();
    if(!raw||raw->GetDataSize()!=d.GetDataSize()||raw->GetDataSize()>65536)throw std::runtime_error("fault28 invalid serializer size");
    const std::string before=Hex(raw->GetData(),raw->GetDataSize());
    std::ostringstream o;o<<std::setprecision(17)<<std::boolalpha;
    o<<"{\"schema\":\"fault28.fia_graph.typed_host.v1\",\"pid\":"<<::getpid()<<",\"tid\":"<<::syscall(SYS_gettid)
     <<",\"token\":"<<Quote(s.token)<<",\"node\":"<<Quote(s.node)<<",\"mode\":"<<Quote(s.mode)
     <<",\"owner_pid\":"<<s.ownerPid<<",\"owner_starttime_ticks\":"<<Quote(s.ownerStart)
     <<",\"scope_ancestry\":[";
    for(size_t j=0;j<s.ancestry.size();++j){if(j)o<<',';o<<"{\"pid\":"<<s.ancestry[j].first<<",\"starttime_ticks\":"<<Quote(s.ancestry[j].second)<<'}';}
    o<<"],\"eligible\":"<<s.eligible<<",\"intervened\":"<<s.changed<<",\"causal_tile_pruning\":"<<s.prune
     <<",\"native_m\":"<<s.nativeM<<",\"selected_m\":"<<s.selectedM<<",\"aic\":"<<aic<<",\"aiv\":"<<aiv
     <<",\"base_template_key\":"<<Quote(std::to_string(baseTemplateKey))<<",\"tiling_key\":"<<Quote(std::to_string(c->GetTilingKey()))
     <<",\"block_dim\":"<<c->GetBlockDim()<<",\"schedule_mode\":"<<c->GetScheduleMode()<<",\"workspace\":"<<workspace
     <<",\"raw_size\":"<<raw->GetDataSize()<<",\"raw_hex\":"<<Quote(before)<<",\"checks\":{";
    bool first=true;for(const auto& v:s.checks){if(!first)o<<',';first=false;o<<Quote(v.first)<<':'<<v.second;}
    o<<"},\"typed\":";AppendTyped(o,d);
    std::vector<uint64_t> again((d.GetDataSize()+7)/8);d.SaveToBuffer(again.data(),again.size()*8);
    o<<",\"serializer_roundtrip_matches\":"<<(before==Hex(again.data(),raw->GetDataSize()))
     <<",\"raw_unchanged_by_observer\":"<<(before==Hex(raw->GetData(),raw->GetDataSize()))
     <<",\"kernel_argument_raw_observed\":false}";
    Append(s,o.str());
}
}
#include "fault28_fia_typed.h"
#endif
