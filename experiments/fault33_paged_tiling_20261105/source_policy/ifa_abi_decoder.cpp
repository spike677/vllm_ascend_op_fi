#include <algorithm>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <vector>
#include "incre_flash_attention_tiling.h"
using Root = optiling::IncreFlashAttentionTilingDataV2;
struct Field { std::string name,type;size_t offset,width,count; };
static std::vector<unsigned char> Save(Root& root) {
    const auto n=root.GetDataSize();if(n==0||n>1024*1024)throw std::runtime_error("invalid public serializer size");
    std::vector<unsigned char> raw(n);root.SaveToBuffer(raw.data(),raw.size());return raw;
}
template<class T,class Setter> static Field Probe(Root& root,const std::string& name,const std::string& type,Setter set) {
    T zero{},ones;std::memset(&ones,0xff,sizeof ones);set(zero);const auto before=Save(root);set(ones);const auto after=Save(root);
    if(before.size()!=after.size())throw std::runtime_error("serializer size changed");
    size_t start=after.size(),changed=0;for(size_t i=0;i<after.size();++i)if(before[i]!=after[i]){start=std::min(start,i);++changed;}
    if(changed!=sizeof(T)||start+sizeof(T)>after.size())throw std::runtime_error("nonprimitive field span: "+name);
    for(size_t i=0;i<sizeof(T);++i)if(before[start+i]!=0||after[start+i]!=0xff)throw std::runtime_error("noncontiguous field encoding: "+name);
    const T marker=static_cast<T>(137.125);set(marker);const auto encoded=Save(root);T decoded{};std::memcpy(&decoded,encoded.data()+start,sizeof decoded);
    if(std::memcmp(&marker,&decoded,sizeof marker))throw std::runtime_error("primitive field roundtrip: "+name);set(zero);
    return Field{name,type,start,sizeof(T),1};
}
static std::vector<Field> Layout(Root& root) {
    std::vector<Field> fields;
    fields.push_back(Probe<uint32_t>(root,"baseParams.batchSize","uint32_t",[&](uint32_t x){root.tilingBase.baseParams.set_batchSize(x);}));
    fields.push_back(Probe<uint32_t>(root,"baseParams.seqSize","uint32_t",[&](uint32_t x){root.tilingBase.baseParams.set_seqSize(x);}));
    fields.push_back(Probe<uint32_t>(root,"baseParams.qSeqSize","uint32_t",[&](uint32_t x){root.tilingBase.baseParams.set_qSeqSize(x);}));
    fields.push_back(Probe<uint32_t>(root,"baseParams.headSize","uint32_t",[&](uint32_t x){root.tilingBase.baseParams.set_headSize(x);}));
    fields.push_back(Probe<uint32_t>(root,"baseParams.headSizeV","uint32_t",[&](uint32_t x){root.tilingBase.baseParams.set_headSizeV(x);}));
    fields.push_back(Probe<uint32_t>(root,"baseParams.blockSize","uint32_t",[&](uint32_t x){root.tilingBase.baseParams.set_blockSize(x);}));
    fields.push_back(Probe<uint32_t>(root,"baseParams.maxBlockNumPerBatch","uint32_t",[&](uint32_t x){root.tilingBase.baseParams.set_maxBlockNumPerBatch(x);}));
    fields.push_back(Probe<uint32_t>(root,"baseParams.maxBlockNumPerSeq","uint32_t",[&](uint32_t x){root.tilingBase.baseParams.set_maxBlockNumPerSeq(x);}));
    fields.push_back(Probe<float>(root,"baseParams.scaleValue","float",[&](float x){root.tilingBase.baseParams.set_scaleValue(x);}));
    fields.push_back(Probe<uint32_t>(root,"baseParams.kvHeadNum","uint32_t",[&](uint32_t x){root.tilingBase.baseParams.set_kvHeadNum(x);}));
    fields.push_back(Probe<uint32_t>(root,"baseParams.headNumRatio","uint32_t",[&](uint32_t x){root.tilingBase.baseParams.set_headNumRatio(x);}));
    fields.push_back(Probe<uint32_t>(root,"baseParams.qHeadNum","uint32_t",[&](uint32_t x){root.tilingBase.baseParams.set_qHeadNum(x);}));
    fields.push_back(Probe<uint32_t>(root,"baseParams.nNumOfQInOneGroup","uint32_t",[&](uint32_t x){root.tilingBase.baseParams.set_nNumOfQInOneGroup(x);}));
    fields.push_back(Probe<uint32_t>(root,"baseParams.batchContinuousFlag","uint32_t",[&](uint32_t x){root.tilingBase.baseParams.set_batchContinuousFlag(x);}));
    fields.push_back(Probe<uint32_t>(root,"baseParams.actualLenDims","uint32_t",[&](uint32_t x){root.tilingBase.baseParams.set_actualLenDims(x);}));
    fields.push_back(Probe<uint32_t>(root,"baseParams.attenMaskFlag","uint32_t",[&](uint32_t x){root.tilingBase.baseParams.set_attenMaskFlag(x);}));
    fields.push_back(Probe<uint32_t>(root,"baseParams.totalBlockNum","uint32_t",[&](uint32_t x){root.tilingBase.baseParams.set_totalBlockNum(x);}));
    fields.push_back(Probe<uint32_t>(root,"baseParams.paKvShapeType","uint32_t",[&](uint32_t x){root.tilingBase.baseParams.set_paKvShapeType(x);}));
    fields.push_back(Probe<uint32_t>(root,"baseParams.sparseMode","uint32_t",[&](uint32_t x){root.tilingBase.baseParams.set_sparseMode(x);}));
    fields.push_back(Probe<uint32_t>(root,"baseParams.slidingFlag","uint32_t",[&](uint32_t x){root.tilingBase.baseParams.set_slidingFlag(x);}));
    fields.push_back(Probe<uint32_t>(root,"splitKVParams.s2","uint32_t",[&](uint32_t x){root.tilingBase.splitKVParams.set_s2(x);}));
    fields.push_back(Probe<uint32_t>(root,"splitKVParams.sInnerLoopSize","uint32_t",[&](uint32_t x){root.tilingBase.splitKVParams.set_sInnerLoopSize(x);}));
    fields.push_back(Probe<uint32_t>(root,"splitKVParams.accumOutSize","uint32_t",[&](uint32_t x){root.tilingBase.splitKVParams.set_accumOutSize(x);}));
    fields.push_back(Probe<uint32_t>(root,"splitKVParams.logSumExpSize","uint32_t",[&](uint32_t x){root.tilingBase.splitKVParams.set_logSumExpSize(x);}));
    fields.push_back(Probe<uint32_t>(root,"increFlashAttentionSingleCoreParams.sInnerLoopTimes","uint32_t",[&](uint32_t x){root.tilingBase.increFlashAttentionSingleCoreParams.set_sInnerLoopTimes(x);}));
    fields.push_back(Probe<uint32_t>(root,"increFlashAttentionSingleCoreParams.singleProcessSInnerSize","uint32_t",[&](uint32_t x){root.tilingBase.increFlashAttentionSingleCoreParams.set_singleProcessSInnerSize(x);}));
    fields.push_back(Probe<uint32_t>(root,"increFlashAttentionSingleCoreParams.singleProcessSInnerSizeTail","uint32_t",[&](uint32_t x){root.tilingBase.increFlashAttentionSingleCoreParams.set_singleProcessSInnerSizeTail(x);}));
    fields.push_back(Probe<uint32_t>(root,"increFlashAttentionSingleCoreParams.usedCoreNum","uint32_t",[&](uint32_t x){root.tilingBase.increFlashAttentionSingleCoreParams.set_usedCoreNum(x);}));
    fields.push_back(Probe<uint32_t>(root,"increFlashAttentionSingleCoreParams.formerCoreNum","uint32_t",[&](uint32_t x){root.tilingBase.increFlashAttentionSingleCoreParams.set_formerCoreNum(x);}));
    fields.push_back(Probe<uint32_t>(root,"increFlashAttentionSingleCoreParams.blockSplitBn2Range","uint32_t",[&](uint32_t x){root.tilingBase.increFlashAttentionSingleCoreParams.set_blockSplitBn2Range(x);}));
    fields.push_back(Probe<uint32_t>(root,"increFlashAttentionSingleCoreParams.tailSplitedBatchRange","uint32_t",[&](uint32_t x){root.tilingBase.increFlashAttentionSingleCoreParams.set_tailSplitedBatchRange(x);}));
    fields.push_back(Probe<uint32_t>(root,"increFlashAttentionSingleCoreParams.groupSplitSize","uint32_t",[&](uint32_t x){root.tilingBase.increFlashAttentionSingleCoreParams.set_groupSplitSize(x);}));
    fields.push_back(Probe<uint32_t>(root,"increFlashAttentionSingleCoreParams.s1SplitSize","uint32_t",[&](uint32_t x){root.tilingBase.increFlashAttentionSingleCoreParams.set_s1SplitSize(x);}));
    fields.push_back(Probe<uint32_t>(root,"increFlashAttentionSingleCoreTensorSize.mmResUbSize","uint32_t",[&](uint32_t x){root.tilingBase.increFlashAttentionSingleCoreTensorSize.set_mmResUbSize(x);}));
    fields.push_back(Probe<uint32_t>(root,"increFlashAttentionSingleCoreTensorSize.bmm2ResUbSize","uint32_t",[&](uint32_t x){root.tilingBase.increFlashAttentionSingleCoreTensorSize.set_bmm2ResUbSize(x);}));
    fields.push_back(Probe<uint32_t>(root,"bmm1TilingData.M","uint32_t",[&](uint32_t x){root.tilingBase.bmm1TilingData.set_M(x);}));
    fields.push_back(Probe<uint32_t>(root,"bmm1TilingData.N","uint32_t",[&](uint32_t x){root.tilingBase.bmm1TilingData.set_N(x);}));
    fields.push_back(Probe<uint32_t>(root,"bmm1TilingData.Ka","uint32_t",[&](uint32_t x){root.tilingBase.bmm1TilingData.set_Ka(x);}));
    fields.push_back(Probe<uint32_t>(root,"bmm1TilingData.Kb","uint32_t",[&](uint32_t x){root.tilingBase.bmm1TilingData.set_Kb(x);}));
    fields.push_back(Probe<uint32_t>(root,"bmm1TilingData.singleCoreM","uint32_t",[&](uint32_t x){root.tilingBase.bmm1TilingData.set_singleCoreM(x);}));
    fields.push_back(Probe<uint32_t>(root,"bmm1TilingData.singleCoreN","uint32_t",[&](uint32_t x){root.tilingBase.bmm1TilingData.set_singleCoreN(x);}));
    fields.push_back(Probe<uint32_t>(root,"bmm1TilingData.singleCoreK","uint32_t",[&](uint32_t x){root.tilingBase.bmm1TilingData.set_singleCoreK(x);}));
    fields.push_back(Probe<uint32_t>(root,"bmm1TilingData.baseM","uint32_t",[&](uint32_t x){root.tilingBase.bmm1TilingData.set_baseM(x);}));
    fields.push_back(Probe<uint32_t>(root,"bmm1TilingData.baseN","uint32_t",[&](uint32_t x){root.tilingBase.bmm1TilingData.set_baseN(x);}));
    fields.push_back(Probe<uint32_t>(root,"bmm1TilingData.baseK","uint32_t",[&](uint32_t x){root.tilingBase.bmm1TilingData.set_baseK(x);}));
    fields.push_back(Probe<uint32_t>(root,"bmm2TilingData.M","uint32_t",[&](uint32_t x){root.tilingBase.bmm2TilingData.set_M(x);}));
    fields.push_back(Probe<uint32_t>(root,"bmm2TilingData.N","uint32_t",[&](uint32_t x){root.tilingBase.bmm2TilingData.set_N(x);}));
    fields.push_back(Probe<uint32_t>(root,"bmm2TilingData.Ka","uint32_t",[&](uint32_t x){root.tilingBase.bmm2TilingData.set_Ka(x);}));
    fields.push_back(Probe<uint32_t>(root,"bmm2TilingData.Kb","uint32_t",[&](uint32_t x){root.tilingBase.bmm2TilingData.set_Kb(x);}));
    fields.push_back(Probe<uint32_t>(root,"bmm2TilingData.singleCoreM","uint32_t",[&](uint32_t x){root.tilingBase.bmm2TilingData.set_singleCoreM(x);}));
    fields.push_back(Probe<uint32_t>(root,"bmm2TilingData.singleCoreN","uint32_t",[&](uint32_t x){root.tilingBase.bmm2TilingData.set_singleCoreN(x);}));
    fields.push_back(Probe<uint32_t>(root,"bmm2TilingData.singleCoreK","uint32_t",[&](uint32_t x){root.tilingBase.bmm2TilingData.set_singleCoreK(x);}));
    fields.push_back(Probe<uint32_t>(root,"bmm2TilingData.baseM","uint32_t",[&](uint32_t x){root.tilingBase.bmm2TilingData.set_baseM(x);}));
    fields.push_back(Probe<uint32_t>(root,"bmm2TilingData.baseN","uint32_t",[&](uint32_t x){root.tilingBase.bmm2TilingData.set_baseN(x);}));
    fields.push_back(Probe<uint32_t>(root,"bmm2TilingData.baseK","uint32_t",[&](uint32_t x){root.tilingBase.bmm2TilingData.set_baseK(x);}));
    fields.push_back(Probe<uint32_t>(root,"softmaxFlashTilingData.srcM","uint32_t",[&](uint32_t x){root.tilingBase.softmaxFlashTilingData.set_srcM(x);}));
    fields.push_back(Probe<uint32_t>(root,"softmaxFlashTilingData.srcK","uint32_t",[&](uint32_t x){root.tilingBase.softmaxFlashTilingData.set_srcK(x);}));
    fields.push_back(Probe<uint32_t>(root,"softmaxFlashTilingData.srcSize","uint32_t",[&](uint32_t x){root.tilingBase.softmaxFlashTilingData.set_srcSize(x);}));
    fields.push_back(Probe<uint32_t>(root,"softmaxFlashTilingData.outMaxM","uint32_t",[&](uint32_t x){root.tilingBase.softmaxFlashTilingData.set_outMaxM(x);}));
    fields.push_back(Probe<uint32_t>(root,"softmaxFlashTilingData.outMaxK","uint32_t",[&](uint32_t x){root.tilingBase.softmaxFlashTilingData.set_outMaxK(x);}));
    fields.push_back(Probe<uint32_t>(root,"softmaxFlashTilingData.outMaxSize","uint32_t",[&](uint32_t x){root.tilingBase.softmaxFlashTilingData.set_outMaxSize(x);}));
    fields.push_back(Probe<uint32_t>(root,"softmaxFlashTilingData.splitM","uint32_t",[&](uint32_t x){root.tilingBase.softmaxFlashTilingData.set_splitM(x);}));
    fields.push_back(Probe<uint32_t>(root,"softmaxFlashTilingData.splitK","uint32_t",[&](uint32_t x){root.tilingBase.softmaxFlashTilingData.set_splitK(x);}));
    fields.push_back(Probe<uint32_t>(root,"softmaxFlashTilingData.splitSize","uint32_t",[&](uint32_t x){root.tilingBase.softmaxFlashTilingData.set_splitSize(x);}));
    fields.push_back(Probe<uint32_t>(root,"softmaxFlashTilingData.reduceM","uint32_t",[&](uint32_t x){root.tilingBase.softmaxFlashTilingData.set_reduceM(x);}));
    fields.push_back(Probe<uint32_t>(root,"softmaxFlashTilingData.reduceK","uint32_t",[&](uint32_t x){root.tilingBase.softmaxFlashTilingData.set_reduceK(x);}));
    fields.push_back(Probe<uint32_t>(root,"softmaxFlashTilingData.reduceSize","uint32_t",[&](uint32_t x){root.tilingBase.softmaxFlashTilingData.set_reduceSize(x);}));
    fields.push_back(Probe<uint32_t>(root,"softmaxFlashTilingData.rangeM","uint32_t",[&](uint32_t x){root.tilingBase.softmaxFlashTilingData.set_rangeM(x);}));
    fields.push_back(Probe<uint32_t>(root,"softmaxFlashTilingData.tailM","uint32_t",[&](uint32_t x){root.tilingBase.softmaxFlashTilingData.set_tailM(x);}));
    fields.push_back(Probe<uint32_t>(root,"softmaxFlashTilingData.tailSplitSize","uint32_t",[&](uint32_t x){root.tilingBase.softmaxFlashTilingData.set_tailSplitSize(x);}));
    fields.push_back(Probe<uint32_t>(root,"softmaxFlashTilingData.tailReduceSize","uint32_t",[&](uint32_t x){root.tilingBase.softmaxFlashTilingData.set_tailReduceSize(x);}));
    for(uint32_t i=0;i<50;++i) {
        auto f=Probe<uint32_t>(root,"coreSidxEnd["+std::to_string(i)+"]","uint32_t",[&](uint32_t x){root.tilingBase.increFlashAttentionCoreParams.get_coreSidxEnd()[i]=x;});
        if(i==0){f.name="coreSidxEnd";f.count=50;fields.push_back(f);}
        else if(f.offset!=fields.back().offset+i*sizeof(uint32_t))throw std::runtime_error("noncontiguous public core array");
    }
    std::vector<bool> used(root.GetDataSize());for(const auto& f:fields)for(size_t i=0;i<f.width*f.count;++i){if(f.offset+i>=used.size()||used[f.offset+i])throw std::runtime_error("overlapping projected fields");used[f.offset+i]=true;}
    return fields;
}
static void Schema(size_t n,const std::vector<Field>& fields) {
    std::cout<<"{\"schema\":\"fault33.pa.public_serializer_layout.v1\",\"type\":\"IncreFlashAttentionTilingDataV2\",\"size\":"<<n
      <<",\"layout_source\":\"public_host_SaveToBuffer_single_field_roundtrip\",\"kernel_argument_raw_observed\":false,\"projection_only\":true,\"fields\":[";
    bool comma=false;for(const auto& f:fields){if(comma)std::cout<<',';comma=true;std::cout<<"{\"name\":\""<<f.name<<"\",\"type\":\""<<f.type<<"\",\"offset\":"<<f.offset<<",\"width\":"<<f.width<<",\"count\":"<<f.count<<'}';}std::cout<<"]}\n";
}
template<class T>static void Number(const unsigned char* p){T v;std::memcpy(&v,p,sizeof v);std::cout<<std::setprecision(17)<<+v;}
static void Values(const std::vector<unsigned char>& raw,const std::vector<Field>& fields) {
    std::cout<<"{\"schema\":\"fault33.pa.public_serializer_values.v1\",\"raw_size\":"<<raw.size()<<",\"fields\":{";bool comma=false;
    for(const auto& f:fields){if(comma)std::cout<<',';comma=true;std::cout<<'"'<<f.name<<"\":";if(f.count>1)std::cout<<'[';for(size_t i=0;i<f.count;++i){if(i)std::cout<<',';const auto* p=raw.data()+f.offset+i*f.width;
        if(f.type=="float")Number<float>(p);else if(f.type=="uint32_t")Number<uint32_t>(p);else if(f.type=="int32_t")Number<int32_t>(p);else if(f.type=="uint64_t")Number<uint64_t>(p);else if(f.type=="int64_t")Number<int64_t>(p);else throw std::runtime_error("unknown type");
    }if(f.count>1)std::cout<<']';}std::cout<<"}}\n";
}
int main(int argc,char** argv) {
    try {
        uint32_t endian=1;if(*reinterpret_cast<unsigned char*>(&endian)!=1)throw std::runtime_error("little endian required");
        Root root;const auto fields=Layout(root);const auto n=root.GetDataSize();
        if(argc==2&&std::string(argv[1])=="--schema"){Schema(n,fields);return 0;}
        if(argc==2&&std::string(argv[1])=="--self-test"){std::cout<<"{\"status\":\"PASS_PUBLIC_SERIALIZER_CPU_ROUNDTRIP\",\"size\":"<<n<<",\"projected_fields\":"<<fields.size()<<",\"stock_abi_validated\":false,\"kernel_argument_raw_observed\":false}\n";return 0;}
        if(argc!=2)throw std::runtime_error("usage: ifa_abi_decoder --schema | --self-test | raw.bin");
        std::ifstream f(argv[1],std::ios::binary);if(!f)throw std::runtime_error("cannot open raw");std::vector<unsigned char> raw((std::istreambuf_iterator<char>(f)),std::istreambuf_iterator<char>());
        if(raw.size()!=n)throw std::runtime_error("raw size differs from public serializer size");Values(raw,fields);return 0;
    }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
}
