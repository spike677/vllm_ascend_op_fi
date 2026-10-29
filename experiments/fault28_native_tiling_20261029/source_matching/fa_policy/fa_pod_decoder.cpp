#include <cstddef>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>
#include "flash_attention_score_tiling.h"
using Root = FlashAttentionScoreGeneralTilingData;
static_assert(std::is_standard_layout<Root>::value,"SDK/public POD root must be standard layout");
static_assert(std::is_trivially_copyable<Root>::value,"SDK/public POD root must be trivially copyable");
struct Field{const char* name;const char* type;size_t offset,width,count;};
static const Field fields[]={
{"inputParams.bSize","int64_t",offsetof(Root,inputParams)+offsetof(InputParams,bSize),sizeof(int64_t),1},
{"inputParams.n2Size","int64_t",offsetof(Root,inputParams)+offsetof(InputParams,n2Size),sizeof(int64_t),1},
{"inputParams.gSize","int64_t",offsetof(Root,inputParams)+offsetof(InputParams,gSize),sizeof(int64_t),1},
{"inputParams.s1Size","int64_t",offsetof(Root,inputParams)+offsetof(InputParams,s1Size),sizeof(int64_t),1},
{"inputParams.s2Size","int64_t",offsetof(Root,inputParams)+offsetof(InputParams,s2Size),sizeof(int64_t),1},
{"inputParams.alignedS2","int64_t",offsetof(Root,inputParams)+offsetof(InputParams,alignedS2),sizeof(int64_t),1},
{"inputParams.dSize","int64_t",offsetof(Root,inputParams)+offsetof(InputParams,dSize),sizeof(int64_t),1},
{"inputParams.keepProb","float",offsetof(Root,inputParams)+offsetof(InputParams,keepProb),sizeof(float),1},
{"inputParams.scaleValue","float",offsetof(Root,inputParams)+offsetof(InputParams,scaleValue),sizeof(float),1},
{"inputParams.preTokens","int64_t",offsetof(Root,inputParams)+offsetof(InputParams,preTokens),sizeof(int64_t),1},
{"inputParams.nextTokens","int64_t",offsetof(Root,inputParams)+offsetof(InputParams,nextTokens),sizeof(int64_t),1},
{"inputParams.pseS1Size","int64_t",offsetof(Root,inputParams)+offsetof(InputParams,pseS1Size),sizeof(int64_t),1},
{"inputParams.pseS2Size","int64_t",offsetof(Root,inputParams)+offsetof(InputParams,pseS2Size),sizeof(int64_t),1},
{"inputParams.pseBSize","uint32_t",offsetof(Root,inputParams)+offsetof(InputParams,pseBSize),sizeof(uint32_t),1},
{"inputParams.bandIndex","uint32_t",offsetof(Root,inputParams)+offsetof(InputParams,bandIndex),sizeof(uint32_t),1},
{"inputParams.layoutType","uint8_t",offsetof(Root,inputParams)+offsetof(InputParams,layoutType),sizeof(uint8_t),1},
{"inputParams.pseShapeType","uint8_t",offsetof(Root,inputParams)+offsetof(InputParams,pseShapeType),sizeof(uint8_t),1},
{"inputParams.attenMaskShapeType","uint8_t",offsetof(Root,inputParams)+offsetof(InputParams,attenMaskShapeType),sizeof(uint8_t),1},
{"inputParams.attenMaskDataType","uint8_t",offsetof(Root,inputParams)+offsetof(InputParams,attenMaskDataType),sizeof(uint8_t),1},
{"inputParams.attenMaskCompressMode","uint8_t",offsetof(Root,inputParams)+offsetof(InputParams,attenMaskCompressMode),sizeof(uint8_t),1},
{"inputParams.implMode","uint8_t",offsetof(Root,inputParams)+offsetof(InputParams,implMode),sizeof(uint8_t),1},
{"inputParams.sparseType","uint8_t",offsetof(Root,inputParams)+offsetof(InputParams,sparseType),sizeof(uint8_t),1},
{"inputParams.needDropMaskOp","uint8_t",offsetof(Root,inputParams)+offsetof(InputParams,needDropMaskOp),sizeof(uint8_t),1},
{"inputParams.pseEncodeType","uint8_t",offsetof(Root,inputParams)+offsetof(InputParams,pseEncodeType),sizeof(uint8_t),1},
{"inputParams.rsv","uint8_t",offsetof(Root,inputParams)+offsetof(InputParams,rsv),sizeof(uint8_t),1},
{"inputParams.needL1Carry","uint8_t",offsetof(Root,inputParams)+offsetof(InputParams,needL1Carry),sizeof(uint8_t),1},
{"inputParams.remainPH","uint8_t",offsetof(Root,inputParams)+offsetof(InputParams,remainPH),sizeof(uint8_t),1},
{"inputParams.remain","uint16_t",offsetof(Root,inputParams)+offsetof(InputParams,remain),sizeof(uint16_t),1},
{"inputParams.attenMaskS2SizePH","uint8_t",offsetof(Root,inputParams)+offsetof(InputParams,attenMaskS2SizePH),sizeof(uint8_t),2},
{"inputParams.attenMaskS2Size","uint32_t",offsetof(Root,inputParams)+offsetof(InputParams,attenMaskS2Size),sizeof(uint32_t),1},
{"inputParams.pseType","uint32_t",offsetof(Root,inputParams)+offsetof(InputParams,pseType),sizeof(uint32_t),1},
{"inputParams.rsv1","uint32_t",offsetof(Root,inputParams)+offsetof(InputParams,rsv1),sizeof(uint32_t),1},
{"inputParams.qStartIdxPH","uint8_t",offsetof(Root,inputParams)+offsetof(InputParams,qStartIdxPH),sizeof(uint8_t),4},
{"inputParams.qStartIdx","int64_t",offsetof(Root,inputParams)+offsetof(InputParams,qStartIdx),sizeof(int64_t),1},
{"inputParams.kvStartIdx","int64_t",offsetof(Root,inputParams)+offsetof(InputParams,kvStartIdx),sizeof(int64_t),1},
{"inputParams.tndSoftmaxOut","uint8_t",offsetof(Root,inputParams)+offsetof(InputParams,tndSoftmaxOut),sizeof(uint8_t),1},
{"inputParams.needSinkOp","uint8_t",offsetof(Root,inputParams)+offsetof(InputParams,needSinkOp),sizeof(uint8_t),1},
{"inputParams.InputParamsPH","uint8_t",offsetof(Root,inputParams)+offsetof(InputParams,InputParamsPH),sizeof(uint8_t),6},
{"multiCoreParams.coreNum","int32_t",offsetof(Root,multiCoreParams)+offsetof(MultiCoreParams,coreNum),sizeof(int32_t),1},
{"multiCoreParams.reserve","int32_t",offsetof(Root,multiCoreParams)+offsetof(MultiCoreParams,reserve),sizeof(int32_t),1},
{"multiCoreParams.totalSize","int64_t",offsetof(Root,multiCoreParams)+offsetof(MultiCoreParams,totalSize),sizeof(int64_t),1},
{"multiCoreParams.splitFactorSize","int64_t",offsetof(Root,multiCoreParams)+offsetof(MultiCoreParams,splitFactorSize),sizeof(int64_t),1},
{"multiCoreParams.splitFactorTailSize","int64_t",offsetof(Root,multiCoreParams)+offsetof(MultiCoreParams,splitFactorTailSize),sizeof(int64_t),1},
{"multiCoreParams.sparseStartIdx","int64_t",offsetof(Root,multiCoreParams)+offsetof(MultiCoreParams,sparseStartIdx),sizeof(int64_t),48},
{"coreParams.s1BaseSize","int32_t",offsetof(Root,coreParams)+offsetof(CoreParams,s1BaseSize),sizeof(int32_t),1},
{"coreParams.s1BaseTailSize","int32_t",offsetof(Root,coreParams)+offsetof(CoreParams,s1BaseTailSize),sizeof(int32_t),1},
{"coreParams.s1OuterSize","int64_t",offsetof(Root,coreParams)+offsetof(CoreParams,s1OuterSize),sizeof(int64_t),1},
{"coreParams.s1Vec2BaseSize","int32_t",offsetof(Root,coreParams)+offsetof(CoreParams,s1Vec2BaseSize),sizeof(int32_t),1},
{"coreParams.s1Vec2BaseTailSize","int32_t",offsetof(Root,coreParams)+offsetof(CoreParams,s1Vec2BaseTailSize),sizeof(int32_t),1},
{"coreParams.s1Vec2OuterSize","int64_t",offsetof(Root,coreParams)+offsetof(CoreParams,s1Vec2OuterSize),sizeof(int64_t),1},
{"coreParams.s2BaseSize","int32_t",offsetof(Root,coreParams)+offsetof(CoreParams,s2BaseSize),sizeof(int32_t),1},
{"coreParams.s2BaseTailSize","int32_t",offsetof(Root,coreParams)+offsetof(CoreParams,s2BaseTailSize),sizeof(int32_t),1},
{"coreParams.s2OuterSize","int64_t",offsetof(Root,coreParams)+offsetof(CoreParams,s2OuterSize),sizeof(int64_t),1},
{"coreParams.dBaseSize","int32_t",offsetof(Root,coreParams)+offsetof(CoreParams,dBaseSize),sizeof(int32_t),1},
{"coreParams.dBaseTailSize","int32_t",offsetof(Root,coreParams)+offsetof(CoreParams,dBaseTailSize),sizeof(int32_t),1},
{"coreParams.dOuterSize","int64_t",offsetof(Root,coreParams)+offsetof(CoreParams,dOuterSize),sizeof(int64_t),1},
{"coreParams.bBaseSize","int32_t",offsetof(Root,coreParams)+offsetof(CoreParams,bBaseSize),sizeof(int32_t),1},
{"coreParams.bBaseTailSize","int32_t",offsetof(Root,coreParams)+offsetof(CoreParams,bBaseTailSize),sizeof(int32_t),1},
{"coreParams.bOuterSize","int64_t",offsetof(Root,coreParams)+offsetof(CoreParams,bOuterSize),sizeof(int64_t),1},
{"coreParams.n2BaseSize","int32_t",offsetof(Root,coreParams)+offsetof(CoreParams,n2BaseSize),sizeof(int32_t),1},
{"coreParams.n2BaseTailSize","int32_t",offsetof(Root,coreParams)+offsetof(CoreParams,n2BaseTailSize),sizeof(int32_t),1},
{"coreParams.n2OuterSize","int64_t",offsetof(Root,coreParams)+offsetof(CoreParams,n2OuterSize),sizeof(int64_t),1},
{"coreParams.gBaseSize","int32_t",offsetof(Root,coreParams)+offsetof(CoreParams,gBaseSize),sizeof(int32_t),1},
{"coreParams.gBaseTailSize","int32_t",offsetof(Root,coreParams)+offsetof(CoreParams,gBaseTailSize),sizeof(int32_t),1},
{"coreParams.gOuterSize","int64_t",offsetof(Root,coreParams)+offsetof(CoreParams,gOuterSize),sizeof(int64_t),1},
{"coreParams.nRatio","int32_t",offsetof(Root,coreParams)+offsetof(CoreParams,nRatio),sizeof(int32_t),1},
{"coreParams.dRopeSize","int32_t",offsetof(Root,coreParams)+offsetof(CoreParams,dRopeSize),sizeof(int32_t),1},
{"coreParams.d2Size","int32_t",offsetof(Root,coreParams)+offsetof(CoreParams,d2Size),sizeof(int32_t),1},
{"coreParams.s1SparseValidSizePH","uint8_t",offsetof(Root,coreParams)+offsetof(CoreParams,s1SparseValidSizePH),sizeof(uint8_t),4},
{"coreParams.s1SparseValidSize","int64_t",offsetof(Root,coreParams)+offsetof(CoreParams,s1SparseValidSize),sizeof(int64_t),1},
{"coreParams.s2SparseValidSize","int64_t",offsetof(Root,coreParams)+offsetof(CoreParams,s2SparseValidSize),sizeof(int64_t),1},
{"coreParams.pseAlibiBaseS1","int64_t",offsetof(Root,coreParams)+offsetof(CoreParams,pseAlibiBaseS1),sizeof(int64_t),1},
{"coreParams.pseAlibiBaseS2","int64_t",offsetof(Root,coreParams)+offsetof(CoreParams,pseAlibiBaseS2),sizeof(int64_t),1},
{"tensorSizeParams.bmm1ResUbSize","int32_t",offsetof(Root,tensorSizeParams)+offsetof(TensorSizeParams,bmm1ResUbSize),sizeof(int32_t),1},
{"tensorSizeParams.attenMaskUbSize","int32_t",offsetof(Root,tensorSizeParams)+offsetof(TensorSizeParams,attenMaskUbSize),sizeof(int32_t),1},
{"tensorSizeParams.pseUbSize","int32_t",offsetof(Root,tensorSizeParams)+offsetof(TensorSizeParams,pseUbSize),sizeof(int32_t),1},
{"tensorSizeParams.dropMaskUbSize","int32_t",offsetof(Root,tensorSizeParams)+offsetof(TensorSizeParams,dropMaskUbSize),sizeof(int32_t),1},
{"tensorSizeParams.castUbSize","int32_t",offsetof(Root,tensorSizeParams)+offsetof(TensorSizeParams,castUbSize),sizeof(int32_t),1},
{"tensorSizeParams.softmaxMaxUbSize","int32_t",offsetof(Root,tensorSizeParams)+offsetof(TensorSizeParams,softmaxMaxUbSize),sizeof(int32_t),1},
{"tensorSizeParams.softmaxSumUbSize","int32_t",offsetof(Root,tensorSizeParams)+offsetof(TensorSizeParams,softmaxSumUbSize),sizeof(int32_t),1},
{"tensorSizeParams.softmaxExpUbSize","int32_t",offsetof(Root,tensorSizeParams)+offsetof(TensorSizeParams,softmaxExpUbSize),sizeof(int32_t),1},
{"tensorSizeParams.apiTmpBufferBytes","int32_t",offsetof(Root,tensorSizeParams)+offsetof(TensorSizeParams,apiTmpBufferBytes),sizeof(int32_t),1},
{"tensorSizeParams.bmm2ResUbSize","int32_t",offsetof(Root,tensorSizeParams)+offsetof(TensorSizeParams,bmm2ResUbSize),sizeof(int32_t),1},
{"tensorSizeParams.inputQueBytes","int32_t",offsetof(Root,tensorSizeParams)+offsetof(TensorSizeParams,inputQueBytes),sizeof(int32_t),1},
{"tensorSizeParams.outputQueBytes","int32_t",offsetof(Root,tensorSizeParams)+offsetof(TensorSizeParams,outputQueBytes),sizeof(int32_t),1},
{"tensorSizeParams.tmpBufBytes","int32_t",offsetof(Root,tensorSizeParams)+offsetof(TensorSizeParams,tmpBufBytes),sizeof(int32_t),1},
{"tensorSizeParams.softmaxMaxOffsetBytes","int32_t",offsetof(Root,tensorSizeParams)+offsetof(TensorSizeParams,softmaxMaxOffsetBytes),sizeof(int32_t),1},
{"tensorSizeParams.softmaxSumOffsetBytes","int32_t",offsetof(Root,tensorSizeParams)+offsetof(TensorSizeParams,softmaxSumOffsetBytes),sizeof(int32_t),1},
{"tensorSizeParams.maxSumApiOffsetBytes","int32_t",offsetof(Root,tensorSizeParams)+offsetof(TensorSizeParams,maxSumApiOffsetBytes),sizeof(int32_t),1},
{"tensorSizeParams.customSoftmaxApiOffsetBytes","int32_t",offsetof(Root,tensorSizeParams)+offsetof(TensorSizeParams,customSoftmaxApiOffsetBytes),sizeof(int32_t),1},
{"tensorSizeParams.pseTbufOffsetBytes","int32_t",offsetof(Root,tensorSizeParams)+offsetof(TensorSizeParams,pseTbufOffsetBytes),sizeof(int32_t),1},
{"tensorSizeParams.dropoutApiOffsetBytes","int32_t",offsetof(Root,tensorSizeParams)+offsetof(TensorSizeParams,dropoutApiOffsetBytes),sizeof(int32_t),1},
{"tensorSizeParams.maxSumApiSize","int32_t",offsetof(Root,tensorSizeParams)+offsetof(TensorSizeParams,maxSumApiSize),sizeof(int32_t),1},
{"tensorSizeParams.customSoftmaxApiSize","int32_t",offsetof(Root,tensorSizeParams)+offsetof(TensorSizeParams,customSoftmaxApiSize),sizeof(int32_t),1},
{"tensorSizeParams.dropoutApiSize","int32_t",offsetof(Root,tensorSizeParams)+offsetof(TensorSizeParams,dropoutApiSize),sizeof(int32_t),1},
{"tensorSizeParams.attenMaskApiSize","int32_t",offsetof(Root,tensorSizeParams)+offsetof(TensorSizeParams,attenMaskApiSize),sizeof(int32_t),1},
{"tensorSizeParams.attenMaskApiOffsetBytes","int32_t",offsetof(Root,tensorSizeParams)+offsetof(TensorSizeParams,attenMaskApiOffsetBytes),sizeof(int32_t),1},
{"tensorSizeParams.bmm1ProcessTInStage2Size","int32_t",offsetof(Root,tensorSizeParams)+offsetof(TensorSizeParams,bmm1ProcessTInStage2Size),sizeof(int32_t),1},
{"tensorSizeParams.bmm1ProcessTInStage2OffsetBytes","int32_t",offsetof(Root,tensorSizeParams)+offsetof(TensorSizeParams,bmm1ProcessTInStage2OffsetBytes),sizeof(int32_t),1},
{"tensorSizeParams.wkspSection1OffsetBytes","int32_t",offsetof(Root,tensorSizeParams)+offsetof(TensorSizeParams,wkspSection1OffsetBytes),sizeof(int32_t),1},
{"tensorSizeParams.wkspSection2OffsetBytes","int32_t",offsetof(Root,tensorSizeParams)+offsetof(TensorSizeParams,wkspSection2OffsetBytes),sizeof(int32_t),1},
{"dropmaskParams.multiCoreFactorSize","int32_t",offsetof(Root,dropmaskParams)+offsetof(DropmaskParams,multiCoreFactorSize),sizeof(int32_t),1},
{"dropmaskParams.baseUbCalSize","int32_t",offsetof(Root,dropmaskParams)+offsetof(DropmaskParams,baseUbCalSize),sizeof(int32_t),1},
{"dropmaskParams.multiCoreTotalSize","int64_t",offsetof(Root,dropmaskParams)+offsetof(DropmaskParams,multiCoreTotalSize),sizeof(int64_t),1},
{"dropmaskParams.shapeTotalSize","int64_t",offsetof(Root,dropmaskParams)+offsetof(DropmaskParams,shapeTotalSize),sizeof(int64_t),1},
};
static void schema(){std::cout<<"{\"schema\":\"fault28.fa.public_pod_layout.v1\",\"type\":\"FlashAttentionScoreGeneralTilingData\",\"size\":"<<sizeof(Root)<<",\"alignment\":"<<alignof(Root)<<",\"fields\":[";bool comma=false;for(auto& f:fields){if(comma)std::cout<<',';comma=true;std::cout<<"{\"name\":\""<<f.name<<"\",\"type\":\""<<f.type<<"\",\"offset\":"<<f.offset<<",\"width\":"<<f.width<<",\"count\":"<<f.count<<'}';}std::cout<<"],\"sections\":[";
if(0)std::cout<<',';std::cout<<"{\"name\":\"inputParams\",\"offset\":"<<offsetof(Root,inputParams)<<",\"size\":"<<sizeof(InputParams)<<'}';
if(1)std::cout<<',';std::cout<<"{\"name\":\"multiCoreParams\",\"offset\":"<<offsetof(Root,multiCoreParams)<<",\"size\":"<<sizeof(MultiCoreParams)<<'}';
if(2)std::cout<<',';std::cout<<"{\"name\":\"coreParams\",\"offset\":"<<offsetof(Root,coreParams)<<",\"size\":"<<sizeof(CoreParams)<<'}';
if(3)std::cout<<',';std::cout<<"{\"name\":\"tensorSizeParams\",\"offset\":"<<offsetof(Root,tensorSizeParams)<<",\"size\":"<<sizeof(TensorSizeParams)<<'}';
if(4)std::cout<<',';std::cout<<"{\"name\":\"bmm1TilingData\",\"offset\":"<<offsetof(Root,bmm1TilingData)<<",\"size\":"<<sizeof(TCubeTiling)<<'}';
if(5)std::cout<<',';std::cout<<"{\"name\":\"bmm2TilingData\",\"offset\":"<<offsetof(Root,bmm2TilingData)<<",\"size\":"<<sizeof(TCubeTiling)<<'}';
if(6)std::cout<<',';std::cout<<"{\"name\":\"softmaxFlashTilingData\",\"offset\":"<<offsetof(Root,softmaxFlashTilingData)<<",\"size\":"<<sizeof(SoftMaxTiling)<<'}';
if(7)std::cout<<',';std::cout<<"{\"name\":\"transposeTilingData\",\"offset\":"<<offsetof(Root,transposeTilingData)<<",\"size\":"<<sizeof(CopyTransposeTiling)<<'}';
if(8)std::cout<<',';std::cout<<"{\"name\":\"transposeTilingDataTailCore\",\"offset\":"<<offsetof(Root,transposeTilingDataTailCore)<<",\"size\":"<<sizeof(CopyTransposeTiling)<<'}';
if(9)std::cout<<',';std::cout<<"{\"name\":\"dropmaskParams\",\"offset\":"<<offsetof(Root,dropmaskParams)<<",\"size\":"<<sizeof(DropmaskParams)<<'}';
std::cout<<"]}\n";}
template<class T>static void printvalue(const unsigned char* p){T v;std::memcpy(&v,p,sizeof v);std::cout<<std::setprecision(17)<<+v;}
static void values(const Root& data){const auto* ptr=reinterpret_cast<const unsigned char*>(&data);std::cout<<"{\"schema\":\"fault28.fa.public_pod_values.v1\",\"raw_size\":"<<sizeof(Root)<<",\"fields\":{";bool comma=false;for(const auto& f:fields){if(f.offset+f.width*f.count>sizeof(Root))throw std::runtime_error("field bounds");if(comma)std::cout<<',';comma=true;std::cout<<'"'<<f.name<<"\":";if(f.count>1)std::cout<<'[';for(size_t i=0;i<f.count;i++){if(i)std::cout<<',';const auto* p=ptr+f.offset+i*f.width;const std::string t=f.type;
if(t=="int64_t")printvalue<int64_t>(p);
else if(t=="uint64_t")printvalue<uint64_t>(p);
else if(t=="int32_t")printvalue<int32_t>(p);
else if(t=="uint32_t")printvalue<uint32_t>(p);
else if(t=="uint16_t")printvalue<uint16_t>(p);
else if(t=="uint8_t")printvalue<uint8_t>(p);
else if(t=="float")printvalue<float>(p);
else throw std::runtime_error("unknown primitive");}if(f.count>1)std::cout<<']';}std::cout<<"}}\n";}
static void selftest(){Root original{},copy{};std::memset(&original,0xa5,sizeof original);
original.inputParams.bSize=static_cast<int64_t>(2);
original.inputParams.n2Size=static_cast<int64_t>(3);
original.inputParams.gSize=static_cast<int64_t>(4);
original.inputParams.s1Size=static_cast<int64_t>(5);
original.inputParams.s2Size=static_cast<int64_t>(6);
original.inputParams.alignedS2=static_cast<int64_t>(7);
original.inputParams.dSize=static_cast<int64_t>(8);
original.inputParams.keepProb=static_cast<float>(9.125);
original.inputParams.scaleValue=static_cast<float>(10.125);
original.inputParams.preTokens=static_cast<int64_t>(11);
original.inputParams.nextTokens=static_cast<int64_t>(12);
original.inputParams.pseS1Size=static_cast<int64_t>(13);
original.inputParams.pseS2Size=static_cast<int64_t>(14);
original.inputParams.pseBSize=static_cast<uint32_t>(15);
original.inputParams.bandIndex=static_cast<uint32_t>(16);
original.inputParams.layoutType=static_cast<uint8_t>(17);
original.inputParams.pseShapeType=static_cast<uint8_t>(18);
original.inputParams.attenMaskShapeType=static_cast<uint8_t>(19);
original.inputParams.attenMaskDataType=static_cast<uint8_t>(20);
original.inputParams.attenMaskCompressMode=static_cast<uint8_t>(21);
original.inputParams.implMode=static_cast<uint8_t>(22);
original.inputParams.sparseType=static_cast<uint8_t>(23);
original.inputParams.needDropMaskOp=static_cast<uint8_t>(24);
original.inputParams.pseEncodeType=static_cast<uint8_t>(25);
original.inputParams.rsv=static_cast<uint8_t>(26);
original.inputParams.needL1Carry=static_cast<uint8_t>(27);
for(size_t i=0;i<1;++i)original.inputParams.remainPH[i]=static_cast<uint8_t>((27 + i)%100+1);
original.inputParams.remain=static_cast<uint16_t>(29);
for(size_t i=0;i<2;++i)original.inputParams.attenMaskS2SizePH[i]=static_cast<uint8_t>((29 + i)%100+1);
original.inputParams.attenMaskS2Size=static_cast<uint32_t>(31);
original.inputParams.pseType=static_cast<uint32_t>(32);
original.inputParams.rsv1=static_cast<uint32_t>(33);
for(size_t i=0;i<4;++i)original.inputParams.qStartIdxPH[i]=static_cast<uint8_t>((33 + i)%100+1);
original.inputParams.qStartIdx=static_cast<int64_t>(35);
original.inputParams.kvStartIdx=static_cast<int64_t>(36);
original.inputParams.tndSoftmaxOut=static_cast<uint8_t>(37);
original.inputParams.needSinkOp=static_cast<uint8_t>(38);
for(size_t i=0;i<6;++i)original.inputParams.InputParamsPH[i]=static_cast<uint8_t>((38 + i)%100+1);
original.multiCoreParams.coreNum=static_cast<int32_t>(40);
original.multiCoreParams.reserve=static_cast<int32_t>(41);
original.multiCoreParams.totalSize=static_cast<int64_t>(42);
original.multiCoreParams.splitFactorSize=static_cast<int64_t>(43);
original.multiCoreParams.splitFactorTailSize=static_cast<int64_t>(44);
for(size_t i=0;i<48;++i)original.multiCoreParams.sparseStartIdx[i]=static_cast<int64_t>((44 + i)%100+1);
original.coreParams.s1BaseSize=static_cast<int32_t>(46);
original.coreParams.s1BaseTailSize=static_cast<int32_t>(47);
original.coreParams.s1OuterSize=static_cast<int64_t>(48);
original.coreParams.s1Vec2BaseSize=static_cast<int32_t>(49);
original.coreParams.s1Vec2BaseTailSize=static_cast<int32_t>(50);
original.coreParams.s1Vec2OuterSize=static_cast<int64_t>(51);
original.coreParams.s2BaseSize=static_cast<int32_t>(52);
original.coreParams.s2BaseTailSize=static_cast<int32_t>(53);
original.coreParams.s2OuterSize=static_cast<int64_t>(54);
original.coreParams.dBaseSize=static_cast<int32_t>(55);
original.coreParams.dBaseTailSize=static_cast<int32_t>(56);
original.coreParams.dOuterSize=static_cast<int64_t>(57);
original.coreParams.bBaseSize=static_cast<int32_t>(58);
original.coreParams.bBaseTailSize=static_cast<int32_t>(59);
original.coreParams.bOuterSize=static_cast<int64_t>(60);
original.coreParams.n2BaseSize=static_cast<int32_t>(61);
original.coreParams.n2BaseTailSize=static_cast<int32_t>(62);
original.coreParams.n2OuterSize=static_cast<int64_t>(63);
original.coreParams.gBaseSize=static_cast<int32_t>(64);
original.coreParams.gBaseTailSize=static_cast<int32_t>(65);
original.coreParams.gOuterSize=static_cast<int64_t>(66);
original.coreParams.nRatio=static_cast<int32_t>(67);
original.coreParams.dRopeSize=static_cast<int32_t>(68);
original.coreParams.d2Size=static_cast<int32_t>(69);
for(size_t i=0;i<4;++i)original.coreParams.s1SparseValidSizePH[i]=static_cast<uint8_t>((69 + i)%100+1);
original.coreParams.s1SparseValidSize=static_cast<int64_t>(71);
original.coreParams.s2SparseValidSize=static_cast<int64_t>(72);
original.coreParams.pseAlibiBaseS1=static_cast<int64_t>(73);
original.coreParams.pseAlibiBaseS2=static_cast<int64_t>(74);
original.tensorSizeParams.bmm1ResUbSize=static_cast<int32_t>(75);
original.tensorSizeParams.attenMaskUbSize=static_cast<int32_t>(76);
original.tensorSizeParams.pseUbSize=static_cast<int32_t>(77);
original.tensorSizeParams.dropMaskUbSize=static_cast<int32_t>(78);
original.tensorSizeParams.castUbSize=static_cast<int32_t>(79);
original.tensorSizeParams.softmaxMaxUbSize=static_cast<int32_t>(80);
original.tensorSizeParams.softmaxSumUbSize=static_cast<int32_t>(81);
original.tensorSizeParams.softmaxExpUbSize=static_cast<int32_t>(82);
original.tensorSizeParams.apiTmpBufferBytes=static_cast<int32_t>(83);
original.tensorSizeParams.bmm2ResUbSize=static_cast<int32_t>(84);
original.tensorSizeParams.inputQueBytes=static_cast<int32_t>(85);
original.tensorSizeParams.outputQueBytes=static_cast<int32_t>(86);
original.tensorSizeParams.tmpBufBytes=static_cast<int32_t>(87);
original.tensorSizeParams.softmaxMaxOffsetBytes=static_cast<int32_t>(88);
original.tensorSizeParams.softmaxSumOffsetBytes=static_cast<int32_t>(89);
original.tensorSizeParams.maxSumApiOffsetBytes=static_cast<int32_t>(90);
original.tensorSizeParams.customSoftmaxApiOffsetBytes=static_cast<int32_t>(91);
original.tensorSizeParams.pseTbufOffsetBytes=static_cast<int32_t>(92);
original.tensorSizeParams.dropoutApiOffsetBytes=static_cast<int32_t>(93);
original.tensorSizeParams.maxSumApiSize=static_cast<int32_t>(94);
original.tensorSizeParams.customSoftmaxApiSize=static_cast<int32_t>(95);
original.tensorSizeParams.dropoutApiSize=static_cast<int32_t>(96);
original.tensorSizeParams.attenMaskApiSize=static_cast<int32_t>(97);
original.tensorSizeParams.attenMaskApiOffsetBytes=static_cast<int32_t>(98);
original.tensorSizeParams.bmm1ProcessTInStage2Size=static_cast<int32_t>(99);
original.tensorSizeParams.bmm1ProcessTInStage2OffsetBytes=static_cast<int32_t>(100);
original.tensorSizeParams.wkspSection1OffsetBytes=static_cast<int32_t>(1);
original.tensorSizeParams.wkspSection2OffsetBytes=static_cast<int32_t>(2);
original.dropmaskParams.multiCoreFactorSize=static_cast<int32_t>(3);
original.dropmaskParams.baseUbCalSize=static_cast<int32_t>(4);
original.dropmaskParams.multiCoreTotalSize=static_cast<int64_t>(5);
original.dropmaskParams.shapeTotalSize=static_cast<int64_t>(6);
std::vector<unsigned char> raw(sizeof(Root));std::memcpy(raw.data(),&original,sizeof original);std::memcpy(&copy,raw.data(),sizeof copy);if(std::memcmp(&copy,&original,sizeof(Root)))throw std::runtime_error("POD bytes roundtrip");
{int64_t value;std::memcpy(&value,raw.data()+offsetof(Root,inputParams)+offsetof(InputParams,bSize),sizeof value);if(value!=original.inputParams.bSize||value!=copy.inputParams.bSize)throw std::runtime_error("field roundtrip inputParams.bSize");}
{int64_t value;std::memcpy(&value,raw.data()+offsetof(Root,inputParams)+offsetof(InputParams,n2Size),sizeof value);if(value!=original.inputParams.n2Size||value!=copy.inputParams.n2Size)throw std::runtime_error("field roundtrip inputParams.n2Size");}
{int64_t value;std::memcpy(&value,raw.data()+offsetof(Root,inputParams)+offsetof(InputParams,gSize),sizeof value);if(value!=original.inputParams.gSize||value!=copy.inputParams.gSize)throw std::runtime_error("field roundtrip inputParams.gSize");}
{int64_t value;std::memcpy(&value,raw.data()+offsetof(Root,inputParams)+offsetof(InputParams,s1Size),sizeof value);if(value!=original.inputParams.s1Size||value!=copy.inputParams.s1Size)throw std::runtime_error("field roundtrip inputParams.s1Size");}
{int64_t value;std::memcpy(&value,raw.data()+offsetof(Root,inputParams)+offsetof(InputParams,s2Size),sizeof value);if(value!=original.inputParams.s2Size||value!=copy.inputParams.s2Size)throw std::runtime_error("field roundtrip inputParams.s2Size");}
{int64_t value;std::memcpy(&value,raw.data()+offsetof(Root,inputParams)+offsetof(InputParams,alignedS2),sizeof value);if(value!=original.inputParams.alignedS2||value!=copy.inputParams.alignedS2)throw std::runtime_error("field roundtrip inputParams.alignedS2");}
{int64_t value;std::memcpy(&value,raw.data()+offsetof(Root,inputParams)+offsetof(InputParams,dSize),sizeof value);if(value!=original.inputParams.dSize||value!=copy.inputParams.dSize)throw std::runtime_error("field roundtrip inputParams.dSize");}
{float value;std::memcpy(&value,raw.data()+offsetof(Root,inputParams)+offsetof(InputParams,keepProb),sizeof value);if(value!=original.inputParams.keepProb||value!=copy.inputParams.keepProb)throw std::runtime_error("field roundtrip inputParams.keepProb");}
{float value;std::memcpy(&value,raw.data()+offsetof(Root,inputParams)+offsetof(InputParams,scaleValue),sizeof value);if(value!=original.inputParams.scaleValue||value!=copy.inputParams.scaleValue)throw std::runtime_error("field roundtrip inputParams.scaleValue");}
{int64_t value;std::memcpy(&value,raw.data()+offsetof(Root,inputParams)+offsetof(InputParams,preTokens),sizeof value);if(value!=original.inputParams.preTokens||value!=copy.inputParams.preTokens)throw std::runtime_error("field roundtrip inputParams.preTokens");}
{int64_t value;std::memcpy(&value,raw.data()+offsetof(Root,inputParams)+offsetof(InputParams,nextTokens),sizeof value);if(value!=original.inputParams.nextTokens||value!=copy.inputParams.nextTokens)throw std::runtime_error("field roundtrip inputParams.nextTokens");}
{int64_t value;std::memcpy(&value,raw.data()+offsetof(Root,inputParams)+offsetof(InputParams,pseS1Size),sizeof value);if(value!=original.inputParams.pseS1Size||value!=copy.inputParams.pseS1Size)throw std::runtime_error("field roundtrip inputParams.pseS1Size");}
{int64_t value;std::memcpy(&value,raw.data()+offsetof(Root,inputParams)+offsetof(InputParams,pseS2Size),sizeof value);if(value!=original.inputParams.pseS2Size||value!=copy.inputParams.pseS2Size)throw std::runtime_error("field roundtrip inputParams.pseS2Size");}
{uint32_t value;std::memcpy(&value,raw.data()+offsetof(Root,inputParams)+offsetof(InputParams,pseBSize),sizeof value);if(value!=original.inputParams.pseBSize||value!=copy.inputParams.pseBSize)throw std::runtime_error("field roundtrip inputParams.pseBSize");}
{uint32_t value;std::memcpy(&value,raw.data()+offsetof(Root,inputParams)+offsetof(InputParams,bandIndex),sizeof value);if(value!=original.inputParams.bandIndex||value!=copy.inputParams.bandIndex)throw std::runtime_error("field roundtrip inputParams.bandIndex");}
{uint8_t value;std::memcpy(&value,raw.data()+offsetof(Root,inputParams)+offsetof(InputParams,layoutType),sizeof value);if(value!=original.inputParams.layoutType||value!=copy.inputParams.layoutType)throw std::runtime_error("field roundtrip inputParams.layoutType");}
{uint8_t value;std::memcpy(&value,raw.data()+offsetof(Root,inputParams)+offsetof(InputParams,pseShapeType),sizeof value);if(value!=original.inputParams.pseShapeType||value!=copy.inputParams.pseShapeType)throw std::runtime_error("field roundtrip inputParams.pseShapeType");}
{uint8_t value;std::memcpy(&value,raw.data()+offsetof(Root,inputParams)+offsetof(InputParams,attenMaskShapeType),sizeof value);if(value!=original.inputParams.attenMaskShapeType||value!=copy.inputParams.attenMaskShapeType)throw std::runtime_error("field roundtrip inputParams.attenMaskShapeType");}
{uint8_t value;std::memcpy(&value,raw.data()+offsetof(Root,inputParams)+offsetof(InputParams,attenMaskDataType),sizeof value);if(value!=original.inputParams.attenMaskDataType||value!=copy.inputParams.attenMaskDataType)throw std::runtime_error("field roundtrip inputParams.attenMaskDataType");}
{uint8_t value;std::memcpy(&value,raw.data()+offsetof(Root,inputParams)+offsetof(InputParams,attenMaskCompressMode),sizeof value);if(value!=original.inputParams.attenMaskCompressMode||value!=copy.inputParams.attenMaskCompressMode)throw std::runtime_error("field roundtrip inputParams.attenMaskCompressMode");}
{uint8_t value;std::memcpy(&value,raw.data()+offsetof(Root,inputParams)+offsetof(InputParams,implMode),sizeof value);if(value!=original.inputParams.implMode||value!=copy.inputParams.implMode)throw std::runtime_error("field roundtrip inputParams.implMode");}
{uint8_t value;std::memcpy(&value,raw.data()+offsetof(Root,inputParams)+offsetof(InputParams,sparseType),sizeof value);if(value!=original.inputParams.sparseType||value!=copy.inputParams.sparseType)throw std::runtime_error("field roundtrip inputParams.sparseType");}
{uint8_t value;std::memcpy(&value,raw.data()+offsetof(Root,inputParams)+offsetof(InputParams,needDropMaskOp),sizeof value);if(value!=original.inputParams.needDropMaskOp||value!=copy.inputParams.needDropMaskOp)throw std::runtime_error("field roundtrip inputParams.needDropMaskOp");}
{uint8_t value;std::memcpy(&value,raw.data()+offsetof(Root,inputParams)+offsetof(InputParams,pseEncodeType),sizeof value);if(value!=original.inputParams.pseEncodeType||value!=copy.inputParams.pseEncodeType)throw std::runtime_error("field roundtrip inputParams.pseEncodeType");}
{uint8_t value;std::memcpy(&value,raw.data()+offsetof(Root,inputParams)+offsetof(InputParams,rsv),sizeof value);if(value!=original.inputParams.rsv||value!=copy.inputParams.rsv)throw std::runtime_error("field roundtrip inputParams.rsv");}
{uint8_t value;std::memcpy(&value,raw.data()+offsetof(Root,inputParams)+offsetof(InputParams,needL1Carry),sizeof value);if(value!=original.inputParams.needL1Carry||value!=copy.inputParams.needL1Carry)throw std::runtime_error("field roundtrip inputParams.needL1Carry");}
for(size_t i=0;i<1;++i){uint8_t value;std::memcpy(&value,raw.data()+offsetof(Root,inputParams)+offsetof(InputParams,remainPH)+i*sizeof(uint8_t),sizeof value);if(value!=original.inputParams.remainPH[i]||value!=copy.inputParams.remainPH[i])throw std::runtime_error("field roundtrip inputParams.remainPH");}
{uint16_t value;std::memcpy(&value,raw.data()+offsetof(Root,inputParams)+offsetof(InputParams,remain),sizeof value);if(value!=original.inputParams.remain||value!=copy.inputParams.remain)throw std::runtime_error("field roundtrip inputParams.remain");}
for(size_t i=0;i<2;++i){uint8_t value;std::memcpy(&value,raw.data()+offsetof(Root,inputParams)+offsetof(InputParams,attenMaskS2SizePH)+i*sizeof(uint8_t),sizeof value);if(value!=original.inputParams.attenMaskS2SizePH[i]||value!=copy.inputParams.attenMaskS2SizePH[i])throw std::runtime_error("field roundtrip inputParams.attenMaskS2SizePH");}
{uint32_t value;std::memcpy(&value,raw.data()+offsetof(Root,inputParams)+offsetof(InputParams,attenMaskS2Size),sizeof value);if(value!=original.inputParams.attenMaskS2Size||value!=copy.inputParams.attenMaskS2Size)throw std::runtime_error("field roundtrip inputParams.attenMaskS2Size");}
{uint32_t value;std::memcpy(&value,raw.data()+offsetof(Root,inputParams)+offsetof(InputParams,pseType),sizeof value);if(value!=original.inputParams.pseType||value!=copy.inputParams.pseType)throw std::runtime_error("field roundtrip inputParams.pseType");}
{uint32_t value;std::memcpy(&value,raw.data()+offsetof(Root,inputParams)+offsetof(InputParams,rsv1),sizeof value);if(value!=original.inputParams.rsv1||value!=copy.inputParams.rsv1)throw std::runtime_error("field roundtrip inputParams.rsv1");}
for(size_t i=0;i<4;++i){uint8_t value;std::memcpy(&value,raw.data()+offsetof(Root,inputParams)+offsetof(InputParams,qStartIdxPH)+i*sizeof(uint8_t),sizeof value);if(value!=original.inputParams.qStartIdxPH[i]||value!=copy.inputParams.qStartIdxPH[i])throw std::runtime_error("field roundtrip inputParams.qStartIdxPH");}
{int64_t value;std::memcpy(&value,raw.data()+offsetof(Root,inputParams)+offsetof(InputParams,qStartIdx),sizeof value);if(value!=original.inputParams.qStartIdx||value!=copy.inputParams.qStartIdx)throw std::runtime_error("field roundtrip inputParams.qStartIdx");}
{int64_t value;std::memcpy(&value,raw.data()+offsetof(Root,inputParams)+offsetof(InputParams,kvStartIdx),sizeof value);if(value!=original.inputParams.kvStartIdx||value!=copy.inputParams.kvStartIdx)throw std::runtime_error("field roundtrip inputParams.kvStartIdx");}
{uint8_t value;std::memcpy(&value,raw.data()+offsetof(Root,inputParams)+offsetof(InputParams,tndSoftmaxOut),sizeof value);if(value!=original.inputParams.tndSoftmaxOut||value!=copy.inputParams.tndSoftmaxOut)throw std::runtime_error("field roundtrip inputParams.tndSoftmaxOut");}
{uint8_t value;std::memcpy(&value,raw.data()+offsetof(Root,inputParams)+offsetof(InputParams,needSinkOp),sizeof value);if(value!=original.inputParams.needSinkOp||value!=copy.inputParams.needSinkOp)throw std::runtime_error("field roundtrip inputParams.needSinkOp");}
for(size_t i=0;i<6;++i){uint8_t value;std::memcpy(&value,raw.data()+offsetof(Root,inputParams)+offsetof(InputParams,InputParamsPH)+i*sizeof(uint8_t),sizeof value);if(value!=original.inputParams.InputParamsPH[i]||value!=copy.inputParams.InputParamsPH[i])throw std::runtime_error("field roundtrip inputParams.InputParamsPH");}
{int32_t value;std::memcpy(&value,raw.data()+offsetof(Root,multiCoreParams)+offsetof(MultiCoreParams,coreNum),sizeof value);if(value!=original.multiCoreParams.coreNum||value!=copy.multiCoreParams.coreNum)throw std::runtime_error("field roundtrip multiCoreParams.coreNum");}
{int32_t value;std::memcpy(&value,raw.data()+offsetof(Root,multiCoreParams)+offsetof(MultiCoreParams,reserve),sizeof value);if(value!=original.multiCoreParams.reserve||value!=copy.multiCoreParams.reserve)throw std::runtime_error("field roundtrip multiCoreParams.reserve");}
{int64_t value;std::memcpy(&value,raw.data()+offsetof(Root,multiCoreParams)+offsetof(MultiCoreParams,totalSize),sizeof value);if(value!=original.multiCoreParams.totalSize||value!=copy.multiCoreParams.totalSize)throw std::runtime_error("field roundtrip multiCoreParams.totalSize");}
{int64_t value;std::memcpy(&value,raw.data()+offsetof(Root,multiCoreParams)+offsetof(MultiCoreParams,splitFactorSize),sizeof value);if(value!=original.multiCoreParams.splitFactorSize||value!=copy.multiCoreParams.splitFactorSize)throw std::runtime_error("field roundtrip multiCoreParams.splitFactorSize");}
{int64_t value;std::memcpy(&value,raw.data()+offsetof(Root,multiCoreParams)+offsetof(MultiCoreParams,splitFactorTailSize),sizeof value);if(value!=original.multiCoreParams.splitFactorTailSize||value!=copy.multiCoreParams.splitFactorTailSize)throw std::runtime_error("field roundtrip multiCoreParams.splitFactorTailSize");}
for(size_t i=0;i<48;++i){int64_t value;std::memcpy(&value,raw.data()+offsetof(Root,multiCoreParams)+offsetof(MultiCoreParams,sparseStartIdx)+i*sizeof(int64_t),sizeof value);if(value!=original.multiCoreParams.sparseStartIdx[i]||value!=copy.multiCoreParams.sparseStartIdx[i])throw std::runtime_error("field roundtrip multiCoreParams.sparseStartIdx");}
{int32_t value;std::memcpy(&value,raw.data()+offsetof(Root,coreParams)+offsetof(CoreParams,s1BaseSize),sizeof value);if(value!=original.coreParams.s1BaseSize||value!=copy.coreParams.s1BaseSize)throw std::runtime_error("field roundtrip coreParams.s1BaseSize");}
{int32_t value;std::memcpy(&value,raw.data()+offsetof(Root,coreParams)+offsetof(CoreParams,s1BaseTailSize),sizeof value);if(value!=original.coreParams.s1BaseTailSize||value!=copy.coreParams.s1BaseTailSize)throw std::runtime_error("field roundtrip coreParams.s1BaseTailSize");}
{int64_t value;std::memcpy(&value,raw.data()+offsetof(Root,coreParams)+offsetof(CoreParams,s1OuterSize),sizeof value);if(value!=original.coreParams.s1OuterSize||value!=copy.coreParams.s1OuterSize)throw std::runtime_error("field roundtrip coreParams.s1OuterSize");}
{int32_t value;std::memcpy(&value,raw.data()+offsetof(Root,coreParams)+offsetof(CoreParams,s1Vec2BaseSize),sizeof value);if(value!=original.coreParams.s1Vec2BaseSize||value!=copy.coreParams.s1Vec2BaseSize)throw std::runtime_error("field roundtrip coreParams.s1Vec2BaseSize");}
{int32_t value;std::memcpy(&value,raw.data()+offsetof(Root,coreParams)+offsetof(CoreParams,s1Vec2BaseTailSize),sizeof value);if(value!=original.coreParams.s1Vec2BaseTailSize||value!=copy.coreParams.s1Vec2BaseTailSize)throw std::runtime_error("field roundtrip coreParams.s1Vec2BaseTailSize");}
{int64_t value;std::memcpy(&value,raw.data()+offsetof(Root,coreParams)+offsetof(CoreParams,s1Vec2OuterSize),sizeof value);if(value!=original.coreParams.s1Vec2OuterSize||value!=copy.coreParams.s1Vec2OuterSize)throw std::runtime_error("field roundtrip coreParams.s1Vec2OuterSize");}
{int32_t value;std::memcpy(&value,raw.data()+offsetof(Root,coreParams)+offsetof(CoreParams,s2BaseSize),sizeof value);if(value!=original.coreParams.s2BaseSize||value!=copy.coreParams.s2BaseSize)throw std::runtime_error("field roundtrip coreParams.s2BaseSize");}
{int32_t value;std::memcpy(&value,raw.data()+offsetof(Root,coreParams)+offsetof(CoreParams,s2BaseTailSize),sizeof value);if(value!=original.coreParams.s2BaseTailSize||value!=copy.coreParams.s2BaseTailSize)throw std::runtime_error("field roundtrip coreParams.s2BaseTailSize");}
{int64_t value;std::memcpy(&value,raw.data()+offsetof(Root,coreParams)+offsetof(CoreParams,s2OuterSize),sizeof value);if(value!=original.coreParams.s2OuterSize||value!=copy.coreParams.s2OuterSize)throw std::runtime_error("field roundtrip coreParams.s2OuterSize");}
{int32_t value;std::memcpy(&value,raw.data()+offsetof(Root,coreParams)+offsetof(CoreParams,dBaseSize),sizeof value);if(value!=original.coreParams.dBaseSize||value!=copy.coreParams.dBaseSize)throw std::runtime_error("field roundtrip coreParams.dBaseSize");}
{int32_t value;std::memcpy(&value,raw.data()+offsetof(Root,coreParams)+offsetof(CoreParams,dBaseTailSize),sizeof value);if(value!=original.coreParams.dBaseTailSize||value!=copy.coreParams.dBaseTailSize)throw std::runtime_error("field roundtrip coreParams.dBaseTailSize");}
{int64_t value;std::memcpy(&value,raw.data()+offsetof(Root,coreParams)+offsetof(CoreParams,dOuterSize),sizeof value);if(value!=original.coreParams.dOuterSize||value!=copy.coreParams.dOuterSize)throw std::runtime_error("field roundtrip coreParams.dOuterSize");}
{int32_t value;std::memcpy(&value,raw.data()+offsetof(Root,coreParams)+offsetof(CoreParams,bBaseSize),sizeof value);if(value!=original.coreParams.bBaseSize||value!=copy.coreParams.bBaseSize)throw std::runtime_error("field roundtrip coreParams.bBaseSize");}
{int32_t value;std::memcpy(&value,raw.data()+offsetof(Root,coreParams)+offsetof(CoreParams,bBaseTailSize),sizeof value);if(value!=original.coreParams.bBaseTailSize||value!=copy.coreParams.bBaseTailSize)throw std::runtime_error("field roundtrip coreParams.bBaseTailSize");}
{int64_t value;std::memcpy(&value,raw.data()+offsetof(Root,coreParams)+offsetof(CoreParams,bOuterSize),sizeof value);if(value!=original.coreParams.bOuterSize||value!=copy.coreParams.bOuterSize)throw std::runtime_error("field roundtrip coreParams.bOuterSize");}
{int32_t value;std::memcpy(&value,raw.data()+offsetof(Root,coreParams)+offsetof(CoreParams,n2BaseSize),sizeof value);if(value!=original.coreParams.n2BaseSize||value!=copy.coreParams.n2BaseSize)throw std::runtime_error("field roundtrip coreParams.n2BaseSize");}
{int32_t value;std::memcpy(&value,raw.data()+offsetof(Root,coreParams)+offsetof(CoreParams,n2BaseTailSize),sizeof value);if(value!=original.coreParams.n2BaseTailSize||value!=copy.coreParams.n2BaseTailSize)throw std::runtime_error("field roundtrip coreParams.n2BaseTailSize");}
{int64_t value;std::memcpy(&value,raw.data()+offsetof(Root,coreParams)+offsetof(CoreParams,n2OuterSize),sizeof value);if(value!=original.coreParams.n2OuterSize||value!=copy.coreParams.n2OuterSize)throw std::runtime_error("field roundtrip coreParams.n2OuterSize");}
{int32_t value;std::memcpy(&value,raw.data()+offsetof(Root,coreParams)+offsetof(CoreParams,gBaseSize),sizeof value);if(value!=original.coreParams.gBaseSize||value!=copy.coreParams.gBaseSize)throw std::runtime_error("field roundtrip coreParams.gBaseSize");}
{int32_t value;std::memcpy(&value,raw.data()+offsetof(Root,coreParams)+offsetof(CoreParams,gBaseTailSize),sizeof value);if(value!=original.coreParams.gBaseTailSize||value!=copy.coreParams.gBaseTailSize)throw std::runtime_error("field roundtrip coreParams.gBaseTailSize");}
{int64_t value;std::memcpy(&value,raw.data()+offsetof(Root,coreParams)+offsetof(CoreParams,gOuterSize),sizeof value);if(value!=original.coreParams.gOuterSize||value!=copy.coreParams.gOuterSize)throw std::runtime_error("field roundtrip coreParams.gOuterSize");}
{int32_t value;std::memcpy(&value,raw.data()+offsetof(Root,coreParams)+offsetof(CoreParams,nRatio),sizeof value);if(value!=original.coreParams.nRatio||value!=copy.coreParams.nRatio)throw std::runtime_error("field roundtrip coreParams.nRatio");}
{int32_t value;std::memcpy(&value,raw.data()+offsetof(Root,coreParams)+offsetof(CoreParams,dRopeSize),sizeof value);if(value!=original.coreParams.dRopeSize||value!=copy.coreParams.dRopeSize)throw std::runtime_error("field roundtrip coreParams.dRopeSize");}
{int32_t value;std::memcpy(&value,raw.data()+offsetof(Root,coreParams)+offsetof(CoreParams,d2Size),sizeof value);if(value!=original.coreParams.d2Size||value!=copy.coreParams.d2Size)throw std::runtime_error("field roundtrip coreParams.d2Size");}
for(size_t i=0;i<4;++i){uint8_t value;std::memcpy(&value,raw.data()+offsetof(Root,coreParams)+offsetof(CoreParams,s1SparseValidSizePH)+i*sizeof(uint8_t),sizeof value);if(value!=original.coreParams.s1SparseValidSizePH[i]||value!=copy.coreParams.s1SparseValidSizePH[i])throw std::runtime_error("field roundtrip coreParams.s1SparseValidSizePH");}
{int64_t value;std::memcpy(&value,raw.data()+offsetof(Root,coreParams)+offsetof(CoreParams,s1SparseValidSize),sizeof value);if(value!=original.coreParams.s1SparseValidSize||value!=copy.coreParams.s1SparseValidSize)throw std::runtime_error("field roundtrip coreParams.s1SparseValidSize");}
{int64_t value;std::memcpy(&value,raw.data()+offsetof(Root,coreParams)+offsetof(CoreParams,s2SparseValidSize),sizeof value);if(value!=original.coreParams.s2SparseValidSize||value!=copy.coreParams.s2SparseValidSize)throw std::runtime_error("field roundtrip coreParams.s2SparseValidSize");}
{int64_t value;std::memcpy(&value,raw.data()+offsetof(Root,coreParams)+offsetof(CoreParams,pseAlibiBaseS1),sizeof value);if(value!=original.coreParams.pseAlibiBaseS1||value!=copy.coreParams.pseAlibiBaseS1)throw std::runtime_error("field roundtrip coreParams.pseAlibiBaseS1");}
{int64_t value;std::memcpy(&value,raw.data()+offsetof(Root,coreParams)+offsetof(CoreParams,pseAlibiBaseS2),sizeof value);if(value!=original.coreParams.pseAlibiBaseS2||value!=copy.coreParams.pseAlibiBaseS2)throw std::runtime_error("field roundtrip coreParams.pseAlibiBaseS2");}
{int32_t value;std::memcpy(&value,raw.data()+offsetof(Root,tensorSizeParams)+offsetof(TensorSizeParams,bmm1ResUbSize),sizeof value);if(value!=original.tensorSizeParams.bmm1ResUbSize||value!=copy.tensorSizeParams.bmm1ResUbSize)throw std::runtime_error("field roundtrip tensorSizeParams.bmm1ResUbSize");}
{int32_t value;std::memcpy(&value,raw.data()+offsetof(Root,tensorSizeParams)+offsetof(TensorSizeParams,attenMaskUbSize),sizeof value);if(value!=original.tensorSizeParams.attenMaskUbSize||value!=copy.tensorSizeParams.attenMaskUbSize)throw std::runtime_error("field roundtrip tensorSizeParams.attenMaskUbSize");}
{int32_t value;std::memcpy(&value,raw.data()+offsetof(Root,tensorSizeParams)+offsetof(TensorSizeParams,pseUbSize),sizeof value);if(value!=original.tensorSizeParams.pseUbSize||value!=copy.tensorSizeParams.pseUbSize)throw std::runtime_error("field roundtrip tensorSizeParams.pseUbSize");}
{int32_t value;std::memcpy(&value,raw.data()+offsetof(Root,tensorSizeParams)+offsetof(TensorSizeParams,dropMaskUbSize),sizeof value);if(value!=original.tensorSizeParams.dropMaskUbSize||value!=copy.tensorSizeParams.dropMaskUbSize)throw std::runtime_error("field roundtrip tensorSizeParams.dropMaskUbSize");}
{int32_t value;std::memcpy(&value,raw.data()+offsetof(Root,tensorSizeParams)+offsetof(TensorSizeParams,castUbSize),sizeof value);if(value!=original.tensorSizeParams.castUbSize||value!=copy.tensorSizeParams.castUbSize)throw std::runtime_error("field roundtrip tensorSizeParams.castUbSize");}
{int32_t value;std::memcpy(&value,raw.data()+offsetof(Root,tensorSizeParams)+offsetof(TensorSizeParams,softmaxMaxUbSize),sizeof value);if(value!=original.tensorSizeParams.softmaxMaxUbSize||value!=copy.tensorSizeParams.softmaxMaxUbSize)throw std::runtime_error("field roundtrip tensorSizeParams.softmaxMaxUbSize");}
{int32_t value;std::memcpy(&value,raw.data()+offsetof(Root,tensorSizeParams)+offsetof(TensorSizeParams,softmaxSumUbSize),sizeof value);if(value!=original.tensorSizeParams.softmaxSumUbSize||value!=copy.tensorSizeParams.softmaxSumUbSize)throw std::runtime_error("field roundtrip tensorSizeParams.softmaxSumUbSize");}
{int32_t value;std::memcpy(&value,raw.data()+offsetof(Root,tensorSizeParams)+offsetof(TensorSizeParams,softmaxExpUbSize),sizeof value);if(value!=original.tensorSizeParams.softmaxExpUbSize||value!=copy.tensorSizeParams.softmaxExpUbSize)throw std::runtime_error("field roundtrip tensorSizeParams.softmaxExpUbSize");}
{int32_t value;std::memcpy(&value,raw.data()+offsetof(Root,tensorSizeParams)+offsetof(TensorSizeParams,apiTmpBufferBytes),sizeof value);if(value!=original.tensorSizeParams.apiTmpBufferBytes||value!=copy.tensorSizeParams.apiTmpBufferBytes)throw std::runtime_error("field roundtrip tensorSizeParams.apiTmpBufferBytes");}
{int32_t value;std::memcpy(&value,raw.data()+offsetof(Root,tensorSizeParams)+offsetof(TensorSizeParams,bmm2ResUbSize),sizeof value);if(value!=original.tensorSizeParams.bmm2ResUbSize||value!=copy.tensorSizeParams.bmm2ResUbSize)throw std::runtime_error("field roundtrip tensorSizeParams.bmm2ResUbSize");}
{int32_t value;std::memcpy(&value,raw.data()+offsetof(Root,tensorSizeParams)+offsetof(TensorSizeParams,inputQueBytes),sizeof value);if(value!=original.tensorSizeParams.inputQueBytes||value!=copy.tensorSizeParams.inputQueBytes)throw std::runtime_error("field roundtrip tensorSizeParams.inputQueBytes");}
{int32_t value;std::memcpy(&value,raw.data()+offsetof(Root,tensorSizeParams)+offsetof(TensorSizeParams,outputQueBytes),sizeof value);if(value!=original.tensorSizeParams.outputQueBytes||value!=copy.tensorSizeParams.outputQueBytes)throw std::runtime_error("field roundtrip tensorSizeParams.outputQueBytes");}
{int32_t value;std::memcpy(&value,raw.data()+offsetof(Root,tensorSizeParams)+offsetof(TensorSizeParams,tmpBufBytes),sizeof value);if(value!=original.tensorSizeParams.tmpBufBytes||value!=copy.tensorSizeParams.tmpBufBytes)throw std::runtime_error("field roundtrip tensorSizeParams.tmpBufBytes");}
{int32_t value;std::memcpy(&value,raw.data()+offsetof(Root,tensorSizeParams)+offsetof(TensorSizeParams,softmaxMaxOffsetBytes),sizeof value);if(value!=original.tensorSizeParams.softmaxMaxOffsetBytes||value!=copy.tensorSizeParams.softmaxMaxOffsetBytes)throw std::runtime_error("field roundtrip tensorSizeParams.softmaxMaxOffsetBytes");}
{int32_t value;std::memcpy(&value,raw.data()+offsetof(Root,tensorSizeParams)+offsetof(TensorSizeParams,softmaxSumOffsetBytes),sizeof value);if(value!=original.tensorSizeParams.softmaxSumOffsetBytes||value!=copy.tensorSizeParams.softmaxSumOffsetBytes)throw std::runtime_error("field roundtrip tensorSizeParams.softmaxSumOffsetBytes");}
{int32_t value;std::memcpy(&value,raw.data()+offsetof(Root,tensorSizeParams)+offsetof(TensorSizeParams,maxSumApiOffsetBytes),sizeof value);if(value!=original.tensorSizeParams.maxSumApiOffsetBytes||value!=copy.tensorSizeParams.maxSumApiOffsetBytes)throw std::runtime_error("field roundtrip tensorSizeParams.maxSumApiOffsetBytes");}
{int32_t value;std::memcpy(&value,raw.data()+offsetof(Root,tensorSizeParams)+offsetof(TensorSizeParams,customSoftmaxApiOffsetBytes),sizeof value);if(value!=original.tensorSizeParams.customSoftmaxApiOffsetBytes||value!=copy.tensorSizeParams.customSoftmaxApiOffsetBytes)throw std::runtime_error("field roundtrip tensorSizeParams.customSoftmaxApiOffsetBytes");}
{int32_t value;std::memcpy(&value,raw.data()+offsetof(Root,tensorSizeParams)+offsetof(TensorSizeParams,pseTbufOffsetBytes),sizeof value);if(value!=original.tensorSizeParams.pseTbufOffsetBytes||value!=copy.tensorSizeParams.pseTbufOffsetBytes)throw std::runtime_error("field roundtrip tensorSizeParams.pseTbufOffsetBytes");}
{int32_t value;std::memcpy(&value,raw.data()+offsetof(Root,tensorSizeParams)+offsetof(TensorSizeParams,dropoutApiOffsetBytes),sizeof value);if(value!=original.tensorSizeParams.dropoutApiOffsetBytes||value!=copy.tensorSizeParams.dropoutApiOffsetBytes)throw std::runtime_error("field roundtrip tensorSizeParams.dropoutApiOffsetBytes");}
{int32_t value;std::memcpy(&value,raw.data()+offsetof(Root,tensorSizeParams)+offsetof(TensorSizeParams,maxSumApiSize),sizeof value);if(value!=original.tensorSizeParams.maxSumApiSize||value!=copy.tensorSizeParams.maxSumApiSize)throw std::runtime_error("field roundtrip tensorSizeParams.maxSumApiSize");}
{int32_t value;std::memcpy(&value,raw.data()+offsetof(Root,tensorSizeParams)+offsetof(TensorSizeParams,customSoftmaxApiSize),sizeof value);if(value!=original.tensorSizeParams.customSoftmaxApiSize||value!=copy.tensorSizeParams.customSoftmaxApiSize)throw std::runtime_error("field roundtrip tensorSizeParams.customSoftmaxApiSize");}
{int32_t value;std::memcpy(&value,raw.data()+offsetof(Root,tensorSizeParams)+offsetof(TensorSizeParams,dropoutApiSize),sizeof value);if(value!=original.tensorSizeParams.dropoutApiSize||value!=copy.tensorSizeParams.dropoutApiSize)throw std::runtime_error("field roundtrip tensorSizeParams.dropoutApiSize");}
{int32_t value;std::memcpy(&value,raw.data()+offsetof(Root,tensorSizeParams)+offsetof(TensorSizeParams,attenMaskApiSize),sizeof value);if(value!=original.tensorSizeParams.attenMaskApiSize||value!=copy.tensorSizeParams.attenMaskApiSize)throw std::runtime_error("field roundtrip tensorSizeParams.attenMaskApiSize");}
{int32_t value;std::memcpy(&value,raw.data()+offsetof(Root,tensorSizeParams)+offsetof(TensorSizeParams,attenMaskApiOffsetBytes),sizeof value);if(value!=original.tensorSizeParams.attenMaskApiOffsetBytes||value!=copy.tensorSizeParams.attenMaskApiOffsetBytes)throw std::runtime_error("field roundtrip tensorSizeParams.attenMaskApiOffsetBytes");}
{int32_t value;std::memcpy(&value,raw.data()+offsetof(Root,tensorSizeParams)+offsetof(TensorSizeParams,bmm1ProcessTInStage2Size),sizeof value);if(value!=original.tensorSizeParams.bmm1ProcessTInStage2Size||value!=copy.tensorSizeParams.bmm1ProcessTInStage2Size)throw std::runtime_error("field roundtrip tensorSizeParams.bmm1ProcessTInStage2Size");}
{int32_t value;std::memcpy(&value,raw.data()+offsetof(Root,tensorSizeParams)+offsetof(TensorSizeParams,bmm1ProcessTInStage2OffsetBytes),sizeof value);if(value!=original.tensorSizeParams.bmm1ProcessTInStage2OffsetBytes||value!=copy.tensorSizeParams.bmm1ProcessTInStage2OffsetBytes)throw std::runtime_error("field roundtrip tensorSizeParams.bmm1ProcessTInStage2OffsetBytes");}
{int32_t value;std::memcpy(&value,raw.data()+offsetof(Root,tensorSizeParams)+offsetof(TensorSizeParams,wkspSection1OffsetBytes),sizeof value);if(value!=original.tensorSizeParams.wkspSection1OffsetBytes||value!=copy.tensorSizeParams.wkspSection1OffsetBytes)throw std::runtime_error("field roundtrip tensorSizeParams.wkspSection1OffsetBytes");}
{int32_t value;std::memcpy(&value,raw.data()+offsetof(Root,tensorSizeParams)+offsetof(TensorSizeParams,wkspSection2OffsetBytes),sizeof value);if(value!=original.tensorSizeParams.wkspSection2OffsetBytes||value!=copy.tensorSizeParams.wkspSection2OffsetBytes)throw std::runtime_error("field roundtrip tensorSizeParams.wkspSection2OffsetBytes");}
{int32_t value;std::memcpy(&value,raw.data()+offsetof(Root,dropmaskParams)+offsetof(DropmaskParams,multiCoreFactorSize),sizeof value);if(value!=original.dropmaskParams.multiCoreFactorSize||value!=copy.dropmaskParams.multiCoreFactorSize)throw std::runtime_error("field roundtrip dropmaskParams.multiCoreFactorSize");}
{int32_t value;std::memcpy(&value,raw.data()+offsetof(Root,dropmaskParams)+offsetof(DropmaskParams,baseUbCalSize),sizeof value);if(value!=original.dropmaskParams.baseUbCalSize||value!=copy.dropmaskParams.baseUbCalSize)throw std::runtime_error("field roundtrip dropmaskParams.baseUbCalSize");}
{int64_t value;std::memcpy(&value,raw.data()+offsetof(Root,dropmaskParams)+offsetof(DropmaskParams,multiCoreTotalSize),sizeof value);if(value!=original.dropmaskParams.multiCoreTotalSize||value!=copy.dropmaskParams.multiCoreTotalSize)throw std::runtime_error("field roundtrip dropmaskParams.multiCoreTotalSize");}
{int64_t value;std::memcpy(&value,raw.data()+offsetof(Root,dropmaskParams)+offsetof(DropmaskParams,shapeTotalSize),sizeof value);if(value!=original.dropmaskParams.shapeTotalSize||value!=copy.dropmaskParams.shapeTotalSize)throw std::runtime_error("field roundtrip dropmaskParams.shapeTotalSize");}
std::cout<<"{\"status\":\"PASS_PUBLIC_POD_CPU_ROUNDTRIP\",\"size\":"<<sizeof(Root)<<",\"fields\":105,\"array_elements\":65,\"stock_abi_validated\":false}\n";}
int main(int argc,char** argv){try{uint32_t endian=1;if(*reinterpret_cast<unsigned char*>(&endian)!=1)throw std::runtime_error("little endian required");if(argc==2 && std::string(argv[1])=="--schema"){schema();return 0;}if(argc==2 && std::string(argv[1])=="--self-test"){selftest();return 0;}if(argc!=2)throw std::runtime_error("usage: fa_pod_decoder --schema | --self-test | raw.bin");std::ifstream f(argv[1],std::ios::binary);if(!f)throw std::runtime_error("cannot open raw");std::vector<unsigned char> raw((std::istreambuf_iterator<char>(f)),std::istreambuf_iterator<char>());if(raw.size()!=sizeof(Root))throw std::runtime_error("raw size differs from public POD sizeof");Root data{};std::memcpy(&data,raw.data(),sizeof data);values(data);return 0;}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
