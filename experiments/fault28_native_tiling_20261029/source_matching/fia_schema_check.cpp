#include "fia_schema_projection.h"
#include <cstdint>
#include <cstring>
#include <iostream>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>
using optiling::FusedInferAttentionScoreTilingData;
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
static Layout layout(FusedInferAttentionScoreTilingData& v) {
 Layout out;fields(v,"",0,out,{{"baseParams",&v.baseParams},{"pageAttenParams",&v.pageAttenParams},{"maskParams",&v.maskParams},{"workspaceParams",&v.workspaceParams},{"innerSplitParams",&v.innerSplitParams},{"outerSplitParams",&v.outerSplitParams},{"fdParams",&v.fdParams},{"prefixParams",&v.prefixParams},{"pseParams",&v.pseParams},{"leftPaddingParams",&v.leftPaddingParams},{"postquantParams",&v.postquantParams}});return out;
}
static void require_size(std::size_t a,std::size_t b){if(a!=b)throw std::runtime_error("raw tiling size differs");}
template<class T>static T read(const std::vector<std::uint64_t>& raw,const Layout& l,const std::string& name,std::size_t i=0){const auto& f=l.at(name);if(f.width!=sizeof(T)||i>=f.count||f.offset+(i+1)*f.width>raw.size()*8)throw std::runtime_error("field bounds differ");T x;std::memcpy(&x,reinterpret_cast<const std::uint8_t*>(raw.data())+f.offset+i*f.width,sizeof(T));return x;}
int main(){try{
 FusedInferAttentionScoreTilingData v,again;const auto size=v.GetDataSize();const auto l=layout(v);if(!size||size%8)throw std::runtime_error("unexpected size/alignment");
 v.baseParams.set_bSize(8);v.baseParams.set_s1Size(1024);v.baseParams.set_scaleValue(0.125f);v.maskParams.set_sparseMode(3);v.maskParams.set_nextToken(-7);v.innerSplitParams.set_mBaseSize(512);v.innerSplitParams.set_s2BaseSize(256);v.fdParams.set_numOfFdHead(0);
 std::uint32_t b[optiling::FIA_MAX_AIC_CORE_NUM]={},g[optiling::FIA_MAX_AIC_CORE_NUM]={},s[optiling::FIA_MAX_AIC_CORE_NUM]={};for(std::uint32_t i=0;i<optiling::FIA_MAX_AIC_CORE_NUM;++i){b[i]=i+1;g[i]=100+i;s[i]=1024+i;}v.outerSplitParams.set_bN2End(b);v.outerSplitParams.set_gS1End(g);v.outerSplitParams.set_s2End(s);
 std::vector<std::uint64_t> raw(size/8),serialized(size/8);v.SaveToBuffer(raw.data(),size);require_size(raw.size()*8,size);
#define ROUND_U(group,field,expected) do{auto x=read<std::uint32_t>(raw,l,#group "." #field);if(x!=(expected))throw std::runtime_error("scalar differs: " #field);again.group.set_##field(x);}while(false)
 ROUND_U(baseParams,bSize,8);ROUND_U(baseParams,s1Size,1024);ROUND_U(maskParams,sparseMode,3);ROUND_U(innerSplitParams,mBaseSize,512);ROUND_U(innerSplitParams,s2BaseSize,256);ROUND_U(fdParams,numOfFdHead,0);
#undef ROUND_U
 const auto scale=read<float>(raw,l,"baseParams.scaleValue");const auto next=read<std::int32_t>(raw,l,"maskParams.nextToken");if(scale!=0.125f||next!=-7)throw std::runtime_error("float/int differs");again.baseParams.set_scaleValue(scale);again.maskParams.set_nextToken(next);
 for(std::uint32_t i=0;i<optiling::FIA_MAX_AIC_CORE_NUM;++i)if(read<std::uint32_t>(raw,l,"outerSplitParams.bN2End",i)!=b[i]||read<std::uint32_t>(raw,l,"outerSplitParams.gS1End",i)!=g[i]||read<std::uint32_t>(raw,l,"outerSplitParams.s2End",i)!=s[i])throw std::runtime_error("array differs");
 again.outerSplitParams.set_bN2End(b);again.outerSplitParams.set_gS1End(g);again.outerSplitParams.set_s2End(s);again.SaveToBuffer(serialized.data(),size);if(raw!=serialized)throw std::runtime_error("serialized roundtrip differs");int rejected=0;for(auto n:{size-1,size+1})try{require_size(n,size);}catch(const std::runtime_error&){++rejected;}if(rejected!=2)throw std::runtime_error("size guard failed");
 std::cout<<"{\"status\":\"PASS_CPU_DERIVED_LAYOUT_SENTINEL\",\"serialized_bytes\":"<<size<<",\"array_entries_checked\":"<<3*optiling::FIA_MAX_AIC_CORE_NUM<<",\"invalid_lengths_rejected\":2,\"stock_host_abi_verified\":false,\"npu_devices_required\":false,\"fields\":{";bool first=true;for(const auto& p:l){if(!first)std::cout<<',';first=false;std::cout<<'"'<<p.first<<"\":{\"offset\":"<<p.second.offset<<",\"width\":"<<p.second.width<<",\"count\":"<<p.second.count<<",\"dtype\":\""<<p.second.dtype<<"\"}";}std::cout<<"}}\n";return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
