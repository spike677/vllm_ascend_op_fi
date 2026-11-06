"""Generate a decoder using the pinned public host serializer, never guessed offsets.

This is a serializer-derived ABI projection, not a direct device argument read
or a claim that the SDK host class is a C++ POD. Full raw bytes remain available.
"""
from pathlib import Path
import re

import apply_policy

HEADER = r'''#include <algorithm>
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
'''
FOOTER = r'''    for(uint32_t i=0;i<50;++i) {
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
'''


def generate():
    # Extract exactly the projection's getter calls, keeping host observation
    # and the separately compiled raw decoder field names identical.
    fields = re.findall(r'base\.([A-Za-z0-9_]+)\.get_([A-Za-z0-9_]+)\(\)', apply_policy.typed_capture())
    result = [HEADER]
    for section, name in dict.fromkeys(fields):
        if name == "coreSidxEnd":
            continue
        typ = "float" if name == "scaleValue" else "uint32_t"
        result.append(f'    fields.push_back(Probe<{typ}>(root,"{section}.{name}","{typ}",[&]({typ} x){{root.tilingBase.{section}.set_{name}(x);}}));\n')
    result.append(FOOTER)
    return "".join(result)


if __name__ == "__main__":
    Path(__file__).with_name("ifa_abi_decoder.cpp").write_text(generate(), encoding="utf-8")
