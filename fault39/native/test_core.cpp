#include "fault39_core.h"
#include <cassert>
#include <iostream>
#include <vector>
Fault39TensorMetadata tensor(int64_t m,int64_t n,int64_t s0,int64_t s1,int64_t offset=0) {
 Fault39TensorMetadata t{};t.dtype=27;t.format=2;t.present=1;t.tensor_ptr=1;t.storage_ptr=1;t.storage_addr=4096;
 t.element_size=2;t.view_rank=2;t.stride_rank=2;t.storage_rank=1;
 t.view_shape[0]=m;t.view_shape[1]=n;t.strides[0]=s0;t.strides[1]=s1;t.view_offset=offset;
 t.numel=m*n;t.storage_elements=offset+(m-1)*s0+(n-1)*s1+1;
 t.storage_shape[0]=t.storage_elements;t.storage_bytes=t.storage_elements*2;t.view_bytes=t.numel*2;return t;
}
int main() {
 using namespace fault39;
 for(auto k:{2,4,6})assert(auto_exponent(k)==-1);
 assert(auto_exponent(8)==63);assert(auto_exponent(12800)==58);assert(auto_exponent(65536)==57);
 assert(auto_exponent(7)==-1);
 for(uint64_t K:{2,6,8,514,2050,12800})for(uint32_t pattern=0;pattern<5;++pattern) {
  uint64_t plus=0;for(uint64_t k=0;k<K;++k)plus+=positive(k,K,pattern);
  assert(plus==K/2);
  if(pattern>=2)for(uint64_t k=0;k<K/2;++k)assert(positive(k,K,pattern)!=positive(k+K/2,K,pattern));
 }
 for(auto t:{tensor(3,8,8,1),tensor(8,3,1,8),tensor(3,8,20,2,5)})assert(tensor_valid(t,1<<20));
 auto t=tensor(3,8,8,1);t.strides[0]=1;assert(!tensor_valid(t,1<<20));
 t=tensor(3,8,8,1);t.strides[0]=0;assert(!tensor_valid(t,1<<20));
 t=tensor(3,8,8,1);--t.storage_elements;assert(!tensor_valid(t,1<<20));
 t=tensor(3,8,8,1);t.format=29;assert(!tensor_valid(t,1<<20));
 t=tensor(8,3,1,12,2);assert(tensor_valid(t,1<<20));
 std::vector<uint16_t> values(t.storage_elements,0x1234);fill(values.data(),t,63,F39_BLOCK256,true);
 auto c=observe(values.data(),t);assert(c.finite==24 && c.positive==12 && c.negative==12 && !c.zero);
 assert(verify_columns(values.data(),t,63));
 const auto two_columns=tensor(8,2,2,1);std::vector<uint16_t> adversarial(16);
 for(int row=0;row<8;++row){adversarial[row*2]=0x5c80;adversarial[row*2+1]=0xdc80;}
 const auto aggregate=observe(adversarial.data(),two_columns);
 assert(aggregate.positive==aggregate.negative && !verify_columns(adversarial.data(),two_columns,58));
 fill(adversarial.data(),two_columns,58,F39_HALF,false);assert(verify_self(adversarial.data(),two_columns,58));
 assert(values[0]==0x1234 && values[1]==0x1234 && values[10]==0x1234);
 auto o=tensor(1,8,8,1);uint16_t data[]={0,0x8000,0x3f80,0xbf80,0x7f80,0xff80,0x7fc0,1};
 c=observe(data,o);assert(c.total==8 && c.finite==5 && c.zero==2 && c.positive==2 && c.negative==1 && c.nan==1 && c.posinf==1 && c.neginf==1);
 assert(c.minimum==-1 && c.maximum==1 && c.max_abs==1);
 assert(disjoint(100,10,110,20));assert(!disjoint(100,11,110,20));assert(!disjoint(UINT64_MAX-2,4,8,1));
 Fault39MatmulMetadata m{};m.self=tensor(2,8,8,1);m.mat2=tensor(8,3,1,8);m.out=tensor(2,3,3,1);m.all_present=1;
 m.mat2.tensor_ptr=2;m.mat2.storage_ptr=2;m.mat2.storage_addr=8192;
 m.out.tensor_ptr=3;m.out.storage_ptr=3;m.out.storage_addr=12288;
 assert(matmul_valid(m,1<<20));m.out.storage_addr=4100;assert(!matmul_valid(m,1<<20));
 std::cout<<"fault39 core tests passed\n";
}
