"""Inspect actual configured cache and generated ExternalProject URL metadata."""
import argparse
import hashlib
import json
from pathlib import Path
import re
from generation import layout


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--generation',required=True)
    args=parser.parse_args()
    paths=layout(Path('/audit'),args.generation)
    build=paths['source']/'build'
    cache={}
    for line in (build/'CMakeCache.txt').read_text().splitlines():
        if line.startswith(('#','//')) or ':' not in line or '=' not in line:
            continue
        name,rest=line.split(':',1);kind,value=rest.split('=',1);cache[name]=value
    expected={'CANN_3RD_LIB_PATH':'/deps','CANN_3RD_PKG_PATH':'/deps/pkg',
        'FETCHCONTENT_SOURCE_DIR_CANN-CMAKE':'/deps/cann-cmake','FETCHCONTENT_SOURCE_DIR_OPBASE':'/deps/opbase',
        'FETCHCONTENT_SOURCE_DIR_OPS-TENSOR':'/deps/ops-tensor',
        'CMAKE_PROJECT_INCLUDE':str(Path(__file__).resolve().with_name('offline_json.cmake')),'ENABLE_OPS_KERNEL':'OFF',
        'ENABLE_TEST':'OFF','BUILD_OPEN_PROJECT':'ON'}
    for key,value in expected.items():
        if cache.get(key)!=value:
            raise RuntimeError('Actual configure cache differs: '+key+'='+str(cache.get(key)))
    urls=[]
    for p in sorted(build.rglob('*')):
        if p.is_file() and (p.name.endswith('-urlinfo.txt') or p.name.endswith('-gitinfo.txt')):
            text=p.read_text()
            if re.search(r'https?://|ssh://|git@',text):
                raise RuntimeError('Actual ExternalProject still has a remote acquisition URL: '+str(p))
            urls.append({'path':p.relative_to(build).as_posix(),'text':text,
                'sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
    result={'schema':'fault28.fia_graph.actual_offline_configure.v1','status':'PASS_ACTUAL_CONFIGURE_OFFLINE_INPUTS',
        'generation':args.generation,'cache':{k:cache[k] for k in expected},'generated_acquisition_info':urls,
        'npu_executed':False,'network_enabled':False}
    (paths['evidence']/'actual_offline_configure.json').write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':
    main()
