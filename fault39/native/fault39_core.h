#pragma once
#include "fault39_api.h"
#include "fault39_metadata.h"
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <numeric>
#include <sstream>
#include <string>
namespace fault39 {
// ACL values are verified against CANN headers in the runtime translation unit.
constexpr int kBf16=27, kNd=2;
inline bool mul(uint64_t a,uint64_t b,uint64_t& out) {
 if(b && a>UINT64_MAX/b) return false;
 out=a*b;
 return true;
}
inline bool add(uint64_t a,uint64_t b,uint64_t& out) {
 if(a>UINT64_MAX-b) return false;
 out=a+b;
 return true;
}
inline bool tensor_valid(const Fault39TensorMetadata& t,uint64_t cap) {
 if(!t.present || !t.tensor_ptr || !t.storage_ptr || !t.storage_addr ||
    t.dtype!=kBf16 || t.format!=kNd || t.element_size!=2 ||
    t.view_rank!=2 || t.stride_rank!=2 || !t.storage_rank || t.storage_rank>16 ||
    t.view_shape[0]<=0 || t.view_shape[1]<=0 || t.strides[0]<=0 || t.strides[1]<=0 ||
    t.view_offset<0 || t.storage_elements<=0 || !t.storage_bytes || t.storage_bytes>cap) return false;
 uint64_t elements=1,bytes=0,numel=0,last=uint64_t(t.view_offset),step=0;
 for(uint32_t i=0;i<t.storage_rank;++i) {
  if(t.storage_shape[i]<=0 || !mul(elements,uint64_t(t.storage_shape[i]),elements)) return false;
 }
 if(elements!=uint64_t(t.storage_elements) || !mul(elements,2,bytes) || bytes!=t.storage_bytes ||
    !mul(uint64_t(t.view_shape[0]),uint64_t(t.view_shape[1]),numel) || numel!=uint64_t(t.numel) ||
    !mul(numel,2,bytes) || bytes!=t.view_bytes) return false;
 for(int i=0;i<2;++i)
  if(!mul(uint64_t(t.view_shape[i]-1),uint64_t(t.strides[i]),step) || !add(last,step,last)) return false;
 if(last>=elements) return false;
 // Exact 2-D injectivity: smallest collision has deltas s1/gcd, -s0/gcd.
 const auto g=std::gcd(t.strides[0],t.strides[1]);
 return !(t.view_shape[0]>t.strides[1]/g && t.view_shape[1]>t.strides[0]/g);
}
inline bool disjoint(uint64_t a,uint64_t na,uint64_t b,uint64_t nb) {
 uint64_t ea=0,eb=0; return a && b && add(a,na,ea) && add(b,nb,eb) && (ea<=b || eb<=a);
}
inline bool matmul_valid(const Fault39MatmulMetadata& m,uint64_t cap) {
 const auto& a=m.self; const auto& b=m.mat2; const auto& c=m.out;
 if(!m.all_present || m.any_empty_storage || m.any_shared_storage ||
    !tensor_valid(a,cap) || !tensor_valid(b,cap) || !tensor_valid(c,cap) ||
    a.view_shape[1]!=b.view_shape[0] || a.view_shape[0]!=c.view_shape[0] ||
    b.view_shape[1]!=c.view_shape[1] || (a.view_shape[1]&1)) return false;
 const Fault39TensorMetadata* ts[]={&a,&b,&c};
 for(int i=0;i<3;++i) for(int j=i+1;j<3;++j)
  if(ts[i]->tensor_ptr==ts[j]->tensor_ptr || ts[i]->storage_ptr==ts[j]->storage_ptr ||
     !disjoint(ts[i]->storage_addr,ts[i]->storage_bytes,ts[j]->storage_addr,ts[j]->storage_bytes)) return false;
 return true;
}
inline int auto_exponent(uint64_t k) {
 if(k<2 || (k&1)) return -1;
 for(int e=0;e<=63;++e) {
  // FP32_MAX=(2^24-1)*2^104, all products with e<=63 are finite.
  // H > floor(FP32_MAX / 2^(2e)), exact integer comparison without rounding.
  const int shift=104-2*e;
  uint64_t threshold=0;
  if(shift>=0) {
   if(shift>40)continue;
   threshold=UINT64_C(16777215)<<shift;
  } else threshold=UINT64_C(16777215)>>(-shift);
  if(k/2>threshold)return e;
 }
 return -1;
}
inline bool positive(uint64_t k,uint64_t K,uint32_t pattern) {
 if(pattern==F39_HALF) return k<K/2;
 if(pattern==F39_ALTERNATING) return !(k&1);
 const uint64_t block=pattern==F39_BLOCK256?256:pattern==F39_BLOCK512?512:1024;
 const bool first=k<K/2, base=(((first?k:k-K/2)/block)&1)==0;
 return first?base:!base;
}
inline void fill(void* buffer,const Fault39TensorMetadata& t,int exponent,uint32_t pattern,bool weight) {
 auto* p=static_cast<uint16_t*>(buffer); const uint16_t bits=uint16_t((exponent+127)<<7);
 for(int64_t i=0;i<t.view_shape[0];++i) for(int64_t j=0;j<t.view_shape[1];++j) {
  const uint64_t index=uint64_t(t.view_offset)+uint64_t(i)*t.strides[0]+uint64_t(j)*t.strides[1];
  p[index]=bits | ((weight && !positive(uint64_t(i),uint64_t(t.view_shape[0]),pattern))?0x8000:0);
 }
}
inline bool verify_self(const void* buffer,const Fault39TensorMetadata& t,int exponent) {
 const auto* p=static_cast<const uint16_t*>(buffer);const auto expected=uint16_t((exponent+127)<<7);
 for(int64_t row=0;row<t.view_shape[0];++row)for(int64_t k=0;k<t.view_shape[1];++k)
  if(p[uint64_t(t.view_offset)+uint64_t(row)*t.strides[0]+uint64_t(k)*t.strides[1]]!=expected)return false;
 return true;
}
inline bool verify_columns(const void* buffer,const Fault39TensorMetadata& t,int exponent) {
 const auto* p=static_cast<const uint16_t*>(buffer);const auto expected=uint16_t((exponent+127)<<7);
 for(int64_t n=0;n<t.view_shape[1];++n) {
  uint64_t plus=0,minus=0;
  for(int64_t k=0;k<t.view_shape[0];++k) {
   const auto bits=p[uint64_t(t.view_offset)+uint64_t(k)*t.strides[0]+uint64_t(n)*t.strides[1]];
   if(bits==expected)++plus;else if(bits==uint16_t(expected|0x8000))++minus;else return false;
  }
  if(plus!=uint64_t(t.view_shape[0]/2) || minus!=plus)return false;
 }
 return true;
}
struct Counts {
 uint64_t total=0,finite=0,zero=0,positive=0,negative=0,nan=0,posinf=0,neginf=0;
 double minimum=0,maximum=0,max_abs=0,sum_abs=0;
};
inline Counts observe(const void* buffer,const Fault39TensorMetadata& t) {
 Counts c; auto* p=static_cast<const uint16_t*>(buffer);
 for(int64_t i=0;i<t.view_shape[0];++i) for(int64_t j=0;j<t.view_shape[1];++j) {
  const uint16_t v=p[uint64_t(t.view_offset)+uint64_t(i)*t.strides[0]+uint64_t(j)*t.strides[1]];
  ++c.total;
  if((v&0x7f80)==0x7f80) { if(v&0x7f)++c.nan; else if(v&0x8000)++c.neginf; else ++c.posinf; }
  else {
   uint32_t bits=uint32_t(v)<<16;float f;std::memcpy(&f,&bits,sizeof(f));
   if(!c.finite)c.minimum=c.maximum=f;else {c.minimum=std::fmin(c.minimum,f);c.maximum=std::fmax(c.maximum,f);}
   c.max_abs=std::fmax(c.max_abs,std::fabs(double(f)));c.sum_abs+=std::fabs(double(f));
   ++c.finite; if(!(v&0x7fff))++c.zero; else if(v&0x8000)++c.negative; else ++c.positive;
  }
 }
 return c;
}
inline std::string signature(const Fault39MatmulMetadata& m,int rank,int device) {
 // Canonical structural key, no ephemeral pointers. Multiple cache identities sharing it are ambiguous.
 std::ostringstream s; s<<"f39v1:aclnnMatmul:r"<<rank<<":d"<<device<<":c"<<m.cube_math_type;
 for(auto* t:{&m.self,&m.mat2,&m.out}) s<<":"<<t->dtype<<","<<t->format<<","<<t->view_shape[0]<<","<<t->view_shape[1]<<","<<t->strides[0]<<","<<t->strides[1]<<","<<t->view_offset<<","<<t->storage_bytes;
 return s.str();
}
}
