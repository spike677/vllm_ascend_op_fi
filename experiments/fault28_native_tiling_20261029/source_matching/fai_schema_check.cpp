#include "fai_schema_projection.h"
#include <cstdint>
#include <cstring>
#include <iostream>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>
using optiling::FAInferTilingData;
struct Field { std::size_t offset,width,count; std::string dtype; };
using Layout=std::map<std::string,Field>;
static std::size_t width(const std::string& s) {
 static const std::map<std::string,std::size_t> m={{"bool",sizeof(bool)},{"uint8_t",1},{"int8_t",1},{"uint16_t",2},{"int16_t",2},{"uint32_t",4},{"int32_t",4},{"uint64_t",8},{"int64_t",8},{"float",4},{"double",8}};return m.at(s);
}
// Padding entries come from SDK GetFieldInfo; no guessed alignment or offsets.
static void fields(optiling::TilingDef& value,const std::string& prefix,std::size_t base,Layout& out,const std::map<std::string,optiling::TilingDef*>& children={}) {
 std::size_t cursor=0;
 for(const auto& f:value.GetFieldInfo()) {
  const std::string name=prefix+f.name_,kind=f.classType_;
  if(kind=="2") {auto* child=children.at(f.name_);if(child->GetDataSize()!=f.structSize_)throw std::runtime_error("nested size differs");fields(*child,name+".",base+cursor,out);cursor+=f.structSize_;}
  else {const auto n=kind=="1"?f.arrSize_:1,w=width(f.dtype_);out.emplace(name,Field{base+cursor,w,n,f.dtype_});cursor+=w*n;}
 }
 if(cursor!=value.GetDataSize())throw std::runtime_error("SDK metadata does not cover full serialized size");
}
static Layout layout(FAInferTilingData& v){Layout out;fields(v,"",0,out,{{"coreInfo",&v.coreInfo},{"splitInfo",&v.splitInfo}});return out;}
static void require_size(std::size_t a,std::size_t b){if(a!=b)throw std::runtime_error("raw tiling size differs");}
template<class T>static T read(const std::vector<std::uint64_t>& raw,const Layout& l,const std::string& name,std::size_t i=0){const auto& f=l.at(name);if(f.width!=sizeof(T)||i>=f.count||f.offset+(i+1)*f.width>raw.size()*8)throw std::runtime_error("field bounds differ");T x;std::memcpy(&x,reinterpret_cast<const std::uint8_t*>(raw.data())+f.offset+i*f.width,sizeof(T));return x;}
int main(){try{FAInferTilingData v,again;const auto size=v.GetDataSize();const auto l=layout(v);if(!size||size%8)throw std::runtime_error("size/alignment");
v.set_numHeads(101);
v.set_embeddingSize(202);
v.set_embeddingSizeV(303);
v.set_numBlocks(404);
v.set_blockSize(505);
v.set_maxQSeqlen(606);
v.set_maxKvSeqlen(707);
v.set_kvHeads(808);
v.set_batch(909);
v.set_maxNumBlocksPerBatch(1010);
v.set_firstBatchTaskNum(1111);
v.set_totalTaskNum(1212);
v.set_maskType(1313);
v.set_mm1OutSize(1414);
v.set_smOnlineOutSize(1515);
v.set_mm2OutSize(1616);
v.set_UpdateSize(1717);
v.set_workSpaceSize(1818);
v.set_scaleValue(19.125f);
v.set_pseQ(2020);
v.set_pseKv(2121);
v.set_padding3(2222);
v.set_preToken(2323);
v.set_nextToken(2424);
v.set_sparseMode(2525);
v.set_splitLseTotalSize(2626);
v.set_splitOTotalSize(2727);
v.set_totalSplitNodeNum(2828);
v.set_needCoreNum(2929);
v.set_mainLoopTaskNum(3030);
v.set_tailLoopTaskNum(3131);
v.set_tailStartBatch(3232);
v.set_tailStartN2(3333);
v.set_tailKvNBlockTile(3434);
for(std::uint32_t i=0;i<optiling::MAX_CORE_NUM_FD;++i)v.coreInfo.get_startBIdx()[i]=35000+i;
for(std::uint32_t i=0;i<optiling::MAX_CORE_NUM_FD;++i)v.coreInfo.get_startN1Idx()[i]=36000+i;
for(std::uint32_t i=0;i<optiling::MAX_CORE_NUM_FD;++i)v.coreInfo.get_startS1Idx()[i]=37000+i;
for(std::uint32_t i=0;i<optiling::MAX_CORE_NUM_FD;++i)v.coreInfo.get_startS2Idx()[i]=38000+i;
for(std::uint32_t i=0;i<optiling::MAX_CORE_NUM_FD;++i)v.coreInfo.get_endBIdx()[i]=39000+i;
for(std::uint32_t i=0;i<optiling::MAX_CORE_NUM_FD;++i)v.coreInfo.get_endN1Idx()[i]=40000+i;
for(std::uint32_t i=0;i<optiling::MAX_CORE_NUM_FD;++i)v.coreInfo.get_endS1Idx()[i]=41000+i;
for(std::uint32_t i=0;i<optiling::MAX_CORE_NUM_FD;++i)v.coreInfo.get_endS2Idx()[i]=42000+i;
for(std::uint32_t i=0;i<optiling::MAX_CORE_NUM_FD;++i)v.coreInfo.get_firstSplitKVTaskLseOffset()[i]=43000+i;
for(std::uint32_t i=0;i<optiling::MAX_CORE_NUM_FD;++i)v.coreInfo.get_firstSplitKVTaskOOffset()[i]=44000+i;
for(std::uint32_t i=0;i<optiling::MAX_CORE_NUM_FD;++i)v.splitInfo.get_batchIdx()[i]=45000+i;
for(std::uint32_t i=0;i<optiling::MAX_CORE_NUM_FD;++i)v.splitInfo.get_headStartIdx()[i]=46000+i;
for(std::uint32_t i=0;i<optiling::MAX_CORE_NUM_FD;++i)v.splitInfo.get_headEndIdx()[i]=47000+i;
for(std::uint32_t i=0;i<optiling::MAX_CORE_NUM_FD;++i)v.splitInfo.get_qStartIdx()[i]=48000+i;
for(std::uint32_t i=0;i<optiling::MAX_CORE_NUM_FD;++i)v.splitInfo.get_qEndIdx()[i]=49000+i;
for(std::uint32_t i=0;i<optiling::MAX_CORE_NUM_FD;++i)v.splitInfo.get_splitNum()[i]=50000+i;
for(std::uint32_t i=0;i<optiling::MAX_CORE_NUM_FD;++i)v.splitInfo.get_lseTaskOffset()[i]=51000+i;
for(std::uint32_t i=0;i<optiling::MAX_CORE_NUM_FD;++i)v.splitInfo.get_oTaskOffset()[i]=52000+i;
std::vector<std::uint64_t> raw(size/8),serialized(size/8);v.SaveToBuffer(raw.data(),size);require_size(raw.size()*8,size);
{auto x=read<uint32_t>(raw,l,"numHeads");if(x!=101)throw std::runtime_error("scalar numHeads");again.set_numHeads(x);}
{auto x=read<uint32_t>(raw,l,"embeddingSize");if(x!=202)throw std::runtime_error("scalar embeddingSize");again.set_embeddingSize(x);}
{auto x=read<uint32_t>(raw,l,"embeddingSizeV");if(x!=303)throw std::runtime_error("scalar embeddingSizeV");again.set_embeddingSizeV(x);}
{auto x=read<uint32_t>(raw,l,"numBlocks");if(x!=404)throw std::runtime_error("scalar numBlocks");again.set_numBlocks(x);}
{auto x=read<uint32_t>(raw,l,"blockSize");if(x!=505)throw std::runtime_error("scalar blockSize");again.set_blockSize(x);}
{auto x=read<uint32_t>(raw,l,"maxQSeqlen");if(x!=606)throw std::runtime_error("scalar maxQSeqlen");again.set_maxQSeqlen(x);}
{auto x=read<uint32_t>(raw,l,"maxKvSeqlen");if(x!=707)throw std::runtime_error("scalar maxKvSeqlen");again.set_maxKvSeqlen(x);}
{auto x=read<uint32_t>(raw,l,"kvHeads");if(x!=808)throw std::runtime_error("scalar kvHeads");again.set_kvHeads(x);}
{auto x=read<uint32_t>(raw,l,"batch");if(x!=909)throw std::runtime_error("scalar batch");again.set_batch(x);}
{auto x=read<uint32_t>(raw,l,"maxNumBlocksPerBatch");if(x!=1010)throw std::runtime_error("scalar maxNumBlocksPerBatch");again.set_maxNumBlocksPerBatch(x);}
{auto x=read<uint32_t>(raw,l,"firstBatchTaskNum");if(x!=1111)throw std::runtime_error("scalar firstBatchTaskNum");again.set_firstBatchTaskNum(x);}
{auto x=read<uint32_t>(raw,l,"totalTaskNum");if(x!=1212)throw std::runtime_error("scalar totalTaskNum");again.set_totalTaskNum(x);}
{auto x=read<uint32_t>(raw,l,"maskType");if(x!=1313)throw std::runtime_error("scalar maskType");again.set_maskType(x);}
{auto x=read<uint64_t>(raw,l,"mm1OutSize");if(x!=1414)throw std::runtime_error("scalar mm1OutSize");again.set_mm1OutSize(x);}
{auto x=read<uint64_t>(raw,l,"smOnlineOutSize");if(x!=1515)throw std::runtime_error("scalar smOnlineOutSize");again.set_smOnlineOutSize(x);}
{auto x=read<uint64_t>(raw,l,"mm2OutSize");if(x!=1616)throw std::runtime_error("scalar mm2OutSize");again.set_mm2OutSize(x);}
{auto x=read<uint64_t>(raw,l,"UpdateSize");if(x!=1717)throw std::runtime_error("scalar UpdateSize");again.set_UpdateSize(x);}
{auto x=read<uint64_t>(raw,l,"workSpaceSize");if(x!=1818)throw std::runtime_error("scalar workSpaceSize");again.set_workSpaceSize(x);}
{auto x=read<float>(raw,l,"scaleValue");if(x!=19.125f)throw std::runtime_error("scalar scaleValue");again.set_scaleValue(x);}
{auto x=read<uint64_t>(raw,l,"pseQ");if(x!=2020)throw std::runtime_error("scalar pseQ");again.set_pseQ(x);}
{auto x=read<uint64_t>(raw,l,"pseKv");if(x!=2121)throw std::runtime_error("scalar pseKv");again.set_pseKv(x);}
{auto x=read<uint32_t>(raw,l,"padding3");if(x!=2222)throw std::runtime_error("scalar padding3");again.set_padding3(x);}
{auto x=read<int64_t>(raw,l,"preToken");if(x!=2323)throw std::runtime_error("scalar preToken");again.set_preToken(x);}
{auto x=read<int64_t>(raw,l,"nextToken");if(x!=2424)throw std::runtime_error("scalar nextToken");again.set_nextToken(x);}
{auto x=read<uint32_t>(raw,l,"sparseMode");if(x!=2525)throw std::runtime_error("scalar sparseMode");again.set_sparseMode(x);}
{auto x=read<uint64_t>(raw,l,"splitLseTotalSize");if(x!=2626)throw std::runtime_error("scalar splitLseTotalSize");again.set_splitLseTotalSize(x);}
{auto x=read<uint64_t>(raw,l,"splitOTotalSize");if(x!=2727)throw std::runtime_error("scalar splitOTotalSize");again.set_splitOTotalSize(x);}
{auto x=read<uint32_t>(raw,l,"totalSplitNodeNum");if(x!=2828)throw std::runtime_error("scalar totalSplitNodeNum");again.set_totalSplitNodeNum(x);}
{auto x=read<uint32_t>(raw,l,"needCoreNum");if(x!=2929)throw std::runtime_error("scalar needCoreNum");again.set_needCoreNum(x);}
{auto x=read<uint32_t>(raw,l,"mainLoopTaskNum");if(x!=3030)throw std::runtime_error("scalar mainLoopTaskNum");again.set_mainLoopTaskNum(x);}
{auto x=read<uint32_t>(raw,l,"tailLoopTaskNum");if(x!=3131)throw std::runtime_error("scalar tailLoopTaskNum");again.set_tailLoopTaskNum(x);}
{auto x=read<uint32_t>(raw,l,"tailStartBatch");if(x!=3232)throw std::runtime_error("scalar tailStartBatch");again.set_tailStartBatch(x);}
{auto x=read<uint32_t>(raw,l,"tailStartN2");if(x!=3333)throw std::runtime_error("scalar tailStartN2");again.set_tailStartN2(x);}
{auto x=read<uint32_t>(raw,l,"tailKvNBlockTile");if(x!=3434)throw std::runtime_error("scalar tailKvNBlockTile");again.set_tailKvNBlockTile(x);}
for(std::uint32_t i=0;i<optiling::MAX_CORE_NUM_FD;++i){auto x=read<int32_t>(raw,l,"coreInfo.startBIdx",i);if(x!=35000+i)throw std::runtime_error("array coreInfo.startBIdx");again.coreInfo.get_startBIdx()[i]=x;}
for(std::uint32_t i=0;i<optiling::MAX_CORE_NUM_FD;++i){auto x=read<int32_t>(raw,l,"coreInfo.startN1Idx",i);if(x!=36000+i)throw std::runtime_error("array coreInfo.startN1Idx");again.coreInfo.get_startN1Idx()[i]=x;}
for(std::uint32_t i=0;i<optiling::MAX_CORE_NUM_FD;++i){auto x=read<int32_t>(raw,l,"coreInfo.startS1Idx",i);if(x!=37000+i)throw std::runtime_error("array coreInfo.startS1Idx");again.coreInfo.get_startS1Idx()[i]=x;}
for(std::uint32_t i=0;i<optiling::MAX_CORE_NUM_FD;++i){auto x=read<int32_t>(raw,l,"coreInfo.startS2Idx",i);if(x!=38000+i)throw std::runtime_error("array coreInfo.startS2Idx");again.coreInfo.get_startS2Idx()[i]=x;}
for(std::uint32_t i=0;i<optiling::MAX_CORE_NUM_FD;++i){auto x=read<int32_t>(raw,l,"coreInfo.endBIdx",i);if(x!=39000+i)throw std::runtime_error("array coreInfo.endBIdx");again.coreInfo.get_endBIdx()[i]=x;}
for(std::uint32_t i=0;i<optiling::MAX_CORE_NUM_FD;++i){auto x=read<int32_t>(raw,l,"coreInfo.endN1Idx",i);if(x!=40000+i)throw std::runtime_error("array coreInfo.endN1Idx");again.coreInfo.get_endN1Idx()[i]=x;}
for(std::uint32_t i=0;i<optiling::MAX_CORE_NUM_FD;++i){auto x=read<int32_t>(raw,l,"coreInfo.endS1Idx",i);if(x!=41000+i)throw std::runtime_error("array coreInfo.endS1Idx");again.coreInfo.get_endS1Idx()[i]=x;}
for(std::uint32_t i=0;i<optiling::MAX_CORE_NUM_FD;++i){auto x=read<int32_t>(raw,l,"coreInfo.endS2Idx",i);if(x!=42000+i)throw std::runtime_error("array coreInfo.endS2Idx");again.coreInfo.get_endS2Idx()[i]=x;}
for(std::uint32_t i=0;i<optiling::MAX_CORE_NUM_FD;++i){auto x=read<int64_t>(raw,l,"coreInfo.firstSplitKVTaskLseOffset",i);if(x!=43000+i)throw std::runtime_error("array coreInfo.firstSplitKVTaskLseOffset");again.coreInfo.get_firstSplitKVTaskLseOffset()[i]=x;}
for(std::uint32_t i=0;i<optiling::MAX_CORE_NUM_FD;++i){auto x=read<int64_t>(raw,l,"coreInfo.firstSplitKVTaskOOffset",i);if(x!=44000+i)throw std::runtime_error("array coreInfo.firstSplitKVTaskOOffset");again.coreInfo.get_firstSplitKVTaskOOffset()[i]=x;}
for(std::uint32_t i=0;i<optiling::MAX_CORE_NUM_FD;++i){auto x=read<int32_t>(raw,l,"splitInfo.batchIdx",i);if(x!=45000+i)throw std::runtime_error("array splitInfo.batchIdx");again.splitInfo.get_batchIdx()[i]=x;}
for(std::uint32_t i=0;i<optiling::MAX_CORE_NUM_FD;++i){auto x=read<int32_t>(raw,l,"splitInfo.headStartIdx",i);if(x!=46000+i)throw std::runtime_error("array splitInfo.headStartIdx");again.splitInfo.get_headStartIdx()[i]=x;}
for(std::uint32_t i=0;i<optiling::MAX_CORE_NUM_FD;++i){auto x=read<int32_t>(raw,l,"splitInfo.headEndIdx",i);if(x!=47000+i)throw std::runtime_error("array splitInfo.headEndIdx");again.splitInfo.get_headEndIdx()[i]=x;}
for(std::uint32_t i=0;i<optiling::MAX_CORE_NUM_FD;++i){auto x=read<int32_t>(raw,l,"splitInfo.qStartIdx",i);if(x!=48000+i)throw std::runtime_error("array splitInfo.qStartIdx");again.splitInfo.get_qStartIdx()[i]=x;}
for(std::uint32_t i=0;i<optiling::MAX_CORE_NUM_FD;++i){auto x=read<int32_t>(raw,l,"splitInfo.qEndIdx",i);if(x!=49000+i)throw std::runtime_error("array splitInfo.qEndIdx");again.splitInfo.get_qEndIdx()[i]=x;}
for(std::uint32_t i=0;i<optiling::MAX_CORE_NUM_FD;++i){auto x=read<int32_t>(raw,l,"splitInfo.splitNum",i);if(x!=50000+i)throw std::runtime_error("array splitInfo.splitNum");again.splitInfo.get_splitNum()[i]=x;}
for(std::uint32_t i=0;i<optiling::MAX_CORE_NUM_FD;++i){auto x=read<int64_t>(raw,l,"splitInfo.lseTaskOffset",i);if(x!=51000+i)throw std::runtime_error("array splitInfo.lseTaskOffset");again.splitInfo.get_lseTaskOffset()[i]=x;}
for(std::uint32_t i=0;i<optiling::MAX_CORE_NUM_FD;++i){auto x=read<int64_t>(raw,l,"splitInfo.oTaskOffset",i);if(x!=52000+i)throw std::runtime_error("array splitInfo.oTaskOffset");again.splitInfo.get_oTaskOffset()[i]=x;}
again.SaveToBuffer(serialized.data(),size);if(raw!=serialized)throw std::runtime_error("serialized roundtrip");int rejected=0;for(auto n:{size-1,size+1})try{require_size(n,size);}catch(const std::runtime_error&){++rejected;}if(rejected!=2)throw std::runtime_error("sizeguard");
 std::cout<<"{\"status\":\"PASS_CPU_FAINFER_ALL_FIELDS_SENTINEL\",\"schema\":\"FAInferTilingData\",\"scalar_fields_checked\":34,\"serialized_bytes\":"<<size<<",\"array_entries_checked\":"<<468<<",\"invalid_lengths_rejected\":2,\"stock_host_abi_verified\":false,\"npu_devices_required\":false,\"fields\":{";bool first=true;for(const auto& p:l){if(!first)std::cout<<',';first=false;std::cout<<'"'<<p.first<<"\":{\"offset\":"<<p.second.offset<<",\"width\":"<<p.second.width<<",\"count\":"<<p.second.count<<",\"dtype\":\""<<p.second.dtype<<"\"}";}std::cout<<"}}\n";return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
