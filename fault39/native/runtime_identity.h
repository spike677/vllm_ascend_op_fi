// Loaded-code and on-disk identity, independent of toolkit directory spelling.
// No CANN calls or late loading. Libraries must stay loaded and immutable.
#pragma once
#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <dlfcn.h>
#include <fstream>
#include <link.h>
#include <map>
#include <mutex>
#include <string>
#include <utility>

namespace f39identity {
class Sha256 {
 uint32_t h[8]={0x6a09e667,0xbb67ae85,0x3c6ef372,0xa54ff53a,0x510e527f,0x9b05688c,0x1f83d9ab,0x5be0cd19};
 uint8_t buf[64]{};size_t used=0;uint64_t bytes=0;
 static uint32_t ro(uint32_t x,unsigned n){return (x>>n)|(x<<(32-n));}
 void block(const uint8_t* p){
  static constexpr uint32_t k[64]={
   0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,
   0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,
   0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,
   0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967,
   0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,
   0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,0xd192e819,0xd6990624,0xf40e3585,0x106aa070,
   0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3,
   0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2};
  uint32_t w[64];for(int i=0;i<16;++i)w[i]=(uint32_t(p[i*4])<<24)|(uint32_t(p[i*4+1])<<16)|(uint32_t(p[i*4+2])<<8)|p[i*4+3];
  for(int i=16;i<64;++i){auto a=w[i-15],b=w[i-2];w[i]=w[i-16]+(ro(a,7)^ro(a,18)^(a>>3))+w[i-7]+(ro(b,17)^ro(b,19)^(b>>10));}
  uint32_t a=h[0],b=h[1],c=h[2],d=h[3],e=h[4],f=h[5],g=h[6],z=h[7];
  for(int i=0;i<64;++i){auto t=z+(ro(e,6)^ro(e,11)^ro(e,25))+((e&f)^(~e&g))+k[i]+w[i];auto u=(ro(a,2)^ro(a,13)^ro(a,22))+((a&b)^(a&c)^(b&c));z=g;g=f;f=e;e=d+t;d=c;c=b;b=a;a=t+u;}
  h[0]+=a;h[1]+=b;h[2]+=c;h[3]+=d;h[4]+=e;h[5]+=f;h[6]+=g;h[7]+=z;
 }
public:
 void add(const void* data,size_t n){auto p=static_cast<const uint8_t*>(data);bytes+=n;while(n){size_t c=std::min(n,64-used);std::memcpy(buf+used,p,c);used+=c;p+=c;n-=c;if(used==64){block(buf);used=0;}}}
 std::string finish(){uint64_t bits=bytes*8;uint8_t one=128,zero=0;add(&one,1);while(used!=56)add(&zero,1);uint8_t tail[8];for(int i=0;i<8;++i)tail[7-i]=uint8_t(bits>>(i*8));add(tail,8);std::string out;const char* hex="0123456789abcdef";for(auto v:h)for(int shift=28;shift>=0;shift-=4)out+=hex[(v>>shift)&15];return out;}
};
struct Segment {uint64_t vaddr,size;const char* sha256;};
struct Image {const char* file_sha256;uint64_t file_bytes;const Segment* segments;size_t count;};
inline std::string digest(const void* p,size_t n){Sha256 s;s.add(p,n);return s.finish();}
inline bool file_matches(const char* name,const Image& image){
 std::ifstream f(name,std::ios::binary);if(!f)return false;Sha256 s;std::array<char,65536> buf;uint64_t count=0;
 while(f){f.read(buf.data(),buf.size());auto n=f.gcount();if(n<0 || count+uint64_t(n)>image.file_bytes)return false;s.add(buf.data(),size_t(n));count+=uint64_t(n);}
 return f.eof() && count==image.file_bytes && s.finish()==image.file_sha256;
}
struct Search {uintptr_t base;const Image* expected;bool found=false,match=false;};
inline int visit(dl_phdr_info* info,size_t,void* data){
 auto& s=*static_cast<Search*>(data);if(info->dlpi_addr!=s.base)return 0;s.found=true;size_t index=0;
 for(size_t i=0;i<info->dlpi_phnum;++i){const auto& p=info->dlpi_phdr[i];if(p.p_type!=PT_LOAD || !(p.p_flags&PF_X))continue;
  if(index>=s.expected->count || !(p.p_flags&PF_R) || (p.p_flags&PF_W))return 1;
  const auto& e=s.expected->segments[index++];if(p.p_vaddr!=e.vaddr || p.p_memsz!=e.size || e.size>(1ULL<<30))return 1;
  if(digest(reinterpret_cast<const void*>(info->dlpi_addr+p.p_vaddr),p.p_memsz)!=e.sha256)return 1;
 }
 s.match=index>0 && index==s.expected->count;return 1;
}
inline bool matches(void* symbol,const Image& expected){
 if(!symbol)return false;Dl_info info{};if(!dladdr(symbol,&info) || !info.dli_fname || !info.dli_fbase)return false;
 // Cache only successful checks, scoped to a loaded base and immutable profile.
 static std::mutex mutex;static std::map<std::pair<void*,std::string>,void*> verified;
 std::lock_guard<std::mutex> lock(mutex);auto key=std::make_pair(info.dli_fbase,std::string(expected.file_sha256));
 if(verified.count(key))return true;
 Search s{reinterpret_cast<uintptr_t>(info.dli_fbase),&expected};dl_iterate_phdr(visit,&s);
 if(!s.found || !s.match || !file_matches(info.dli_fname,expected))return false;
 // Pin this existing mapping so a later dlclose/reload cannot reuse a cached base.
 void* pin=dlopen(info.dli_fname,RTLD_LAZY|RTLD_NOLOAD);
 if(!pin)return false;
 verified[key]=pin;return true;
}
}
