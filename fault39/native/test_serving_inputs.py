"""Compile the production serving guard as a CANN-free CPU contract test.

No fake runtime, NPU call, successful injection or complete serving acceptance is
claimed. test_control.py additionally exercises this setter in the actual DSO.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile


CASES = r'''
#include <cassert>
#include <thread>
#include <iostream>
using Values=std::vector<uint64_t>;
Values expected(){return {0x1000,32,0x2000,48,2,8,3,0,0,8,1,1,8};}
Fault39MatmulMetadata metadata(const Values& v) {
 Fault39MatmulMetadata m{};
 m.self.storage_bytes=v[1];m.mat2.storage_bytes=v[3];
 m.self.view_shape[0]=v[4];m.self.view_shape[1]=v[5];
 m.mat2.view_shape[0]=v[5];m.mat2.view_shape[1]=v[6];
 m.out.view_shape[0]=v[4];m.out.view_shape[1]=v[6];
 m.self.view_offset=v[7];m.mat2.view_offset=v[8];
 m.self.strides[0]=v[9];m.self.strides[1]=v[10];
 m.mat2.strides[0]=v[11];m.mat2.strides[1]=v[12];return m;
}
std::vector<void*> addresses(const Values& v) {
 return {reinterpret_cast<void*>(v[0]),reinterpret_cast<void*>(v[2]),reinterpret_cast<void*>(0x4000)};
}
Context owned(){Context c;c.owner="owner";c.ids="[\"actual-request\"]";return c;}
int main() {
 const auto v=expected();const auto m=metadata(v);const auto a=addresses(v);
 Context legacy=owned();assert(serving_inputs_match(legacy,m,a));
 assert(!legacy.serving.requested && !legacy.serving.valid);
 Context empty;assert(bind_serving_inputs(empty,"owner",v.data(),v.size())==F39_INVALID_CONFIG);
 Context c=owned();
 assert(bind_serving_inputs(c,"other",v.data(),v.size())==F39_OWNER_MISMATCH);
 assert(!c.serving.requested);
 assert(bind_serving_inputs(c,"owner",v.data(),v.size())==F39_OK);
 assert(c.serving.requested && c.serving.valid && serving_inputs_match(c,m,a));
 // Every actual field is compared independently against the bound live input.
 for(size_t i=0;i<v.size();++i) {
  auto changed=v;++changed[i];
  assert(!serving_inputs_match(c,metadata(changed),addresses(changed)));
 }
 auto wrong=m;++wrong.mat2.view_shape[0];assert(!serving_inputs_match(c,wrong,a));
 wrong=m;++wrong.out.view_shape[1];assert(!serving_inputs_match(c,wrong,a));
 auto wrong_addresses=a;wrong_addresses.pop_back();assert(!serving_inputs_match(c,m,wrong_addresses));
 // A foreign owner cannot replace or poison another owner's valid guard.
 assert(bind_serving_inputs(c,"other",nullptr,0)==F39_OWNER_MISMATCH);
 assert(serving_inputs_match(c,m,a));
 // Same-owner rebind and bad input cannot silently return to legacy mode.
 assert(bind_serving_inputs(c,"owner",v.data(),v.size())==F39_BUSY);
 assert(c.serving.requested && !c.serving.valid && !serving_inputs_match(c,m,a));
 for(size_t bad_count:{size_t(0),size_t(12),size_t(14)}) {
  c=owned();assert(bind_serving_inputs(c,"owner",v.data(),bad_count)==F39_INVALID_CONFIG);
  assert(c.serving.requested && !serving_inputs_match(c,m,a));
 }
 c=owned();assert(bind_serving_inputs(c,"owner",nullptr,13)==F39_INVALID_CONFIG);
 for(auto bad:std::vector<std::pair<size_t,uint64_t>>{{0,0},{1,31},{1,30},{2,0x1000},
       {4,0},{5,UINT64_C(1)<<63},{7,UINT64_MAX},{8,24},{9,0},{12,UINT64_C(1)<<63}}) {
  auto changed=v;changed[bad.first]=bad.second;c=owned();
  assert(bind_serving_inputs(c,"owner",changed.data(),changed.size())==F39_INVALID_CONFIG);
  assert(!serving_inputs_match(c,m,a));
 }
 // Legal offsets, nontrivial strides and complete storage padding remain valid.
 auto padded=v;padded[7]=2;padded[8]=3;padded[9]=20;padded[10]=2;padded[12]=12;
 padded[1]=80;padded[3]=80;c=owned();
 assert(bind_serving_inputs(c,"owner",padded.data(),padded.size())==F39_OK);
 assert(serving_inputs_match(c,metadata(padded),addresses(padded)));
 // Explicit TLS binding cannot be inherited by another worker/model thread.
 context=owned();assert(bind_serving_inputs(context,"owner",v.data(),v.size())==F39_OK);
 int foreign=999;
 std::thread other([&]{foreign=bind_serving_inputs(context,"owner",v.data(),v.size());
   assert(!context.serving.requested);context=owned();
   assert(bind_serving_inputs(context,"owner",padded.data(),padded.size())==F39_OK);
   assert(serving_inputs_match(context,metadata(padded),addresses(padded)));});
 other.join();assert(foreign==F39_INVALID_CONFIG);
 assert(serving_inputs_match(context,m,a));context={};
 assert(!context.serving.requested && !context.serving.valid);
 assert(bind_serving_inputs(context,"owner",v.data(),v.size())==F39_INVALID_CONFIG);
 std::cout<<"serving input guard CPU contracts passed\n";
}
'''


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cxx', default=os.environ.get('CXX', 'g++'))
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    native = Path(__file__).resolve().parent
    runtime = (native / 'runtime.cpp').read_text(encoding='utf-8')
    first, last = '// BEGIN CPU_TESTABLE_SERVING_INPUT_GUARD', '// END CPU_TESTABLE_SERVING_INPUT_GUARD'
    assert runtime.count(first) == runtime.count(last) == 1
    production = runtime.split(first, 1)[1].split(last, 1)[0]
    compiler = shutil.which(args.cxx)
    receipt = dict(evidence_kind='CPU_COMPILED_PRODUCTION_GUARD_CONTRACT', actual_npu_calls=0,
                   production_guard_sha256=hashlib.sha256(production.encode()).hexdigest(),
                   full_runtime_or_serving_acceptance=False, compiler=compiler)
    if compiler is None:
        receipt['status'] = 'NOT_RUN_COMPILER_UNAVAILABLE'
    else:
        with tempfile.TemporaryDirectory(prefix='fault39-serving-guard-') as tmp:
            source = Path(tmp) / 'guard.cpp'
            exe = Path(tmp) / ('guard.exe' if os.name == 'nt' else 'guard')
            source.write_text('#include "fault39_core.h"\n#include <algorithm>\n#include <vector>\n'
                              + production + CASES, encoding='utf-8')
            subprocess.run([compiler, '-std=c++17', '-O2', '-Wall', '-Wextra', '-pedantic',
                            '-pthread', '-I', str(native), str(source), '-o', str(exe)], check=True)
            run = subprocess.run([str(exe)], check=True, capture_output=True, text=True)
            receipt.update(status='PASS', stdout=run.stdout.strip())
    text = json.dumps(receipt, indent=2)
    print(text)
    if args.output:
        args.output.write_text(text + '\n', encoding='utf-8')
    return 0 if receipt['status'] == 'PASS' else 2


if __name__ == '__main__':
    raise SystemExit(main())
