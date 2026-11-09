"""Freeze the exact reviewed native dependency interfaces supplied by the root."""
import hashlib
import json
from pathlib import Path


def generate():
    here=Path(__file__).resolve().parent
    fixtures=here.parent/'source_matching'
    selected={
        'ops-transformer':('cmake_source',[
            'CMakeLists.txt','cmake/fetch_cann_cmake.cmake','cmake/third_party/opbase.cmake',
            'cmake/third_party/ops-tensor.cmake','cmake/config.cmake','cmake/variables.cmake',
            'cmake/custom_build.cmake','cmake/scripts/prepare.sh']),
        'cann-cmake':('cann_cmake_source',[
            'function/prepare.cmake','third_party/json.cmake','third_party/eigen.cmake',
            'third_party/abseil-cpp.cmake','third_party/protobuf.cmake','third_party/makeself-fetch.cmake',
            'third_party/protobuf_25.1_change_version.patch','third_party/protobuf-hide_absl_symbols.patch'])}
    files=[]
    for repository,(folder,paths) in selected.items():
        for name in paths:
            p=fixtures/folder/name
            files.append({'repository':repository,'path':name,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
    return {'schema':'fault28.fia_graph.reviewed_offline_cmake.v1','reviewed_files':files,
        'enabled_branches':[
            {'dependency':'cann-cmake','interface':'CANN_3RD_LIB_PATH/cann-cmake direct include; explicit FETCHCONTENT_SOURCE_DIR_CANN-CMAKE fallback', 'private_path':'/deps/cann-cmake'},
            {'dependency':'opbase','interface':'CANN_3RD_LIB_PATH/opbase native local-directory branch; local git checkout only','private_path':'/deps/opbase'},
            {'dependency':'ops-tensor','interface':'PROJECT_SOURCE_DIR/../ops-tensor first branch avoids git fetch; explicit FETCHCONTENT_SOURCE_DIR_OPS-TENSOR fallback','private_path':'generation-private sibling symlink to copied ops-tensor'},
            {'dependency':'json','interface':'CMAKE_PROJECT_INCLUDE creates frozen header-only json INTERFACE target with native namespace macro; native module TARGET guard returns','private_path':'/deps/json/single_include'},
            {'dependency':'eigen','interface':'native ExternalProject URL local-directory branch','private_path':'/deps/eigen'},
            {'dependency':'abseil-cpp','interface':'native ExternalProject URL first local-archive branch; official patch retained','private_path':'/deps/abseil-cpp/abseil-cpp-20230802.1.tar.gz'},
            {'dependency':'protobuf','interface':'native ExternalProject protobuf_src local archive; dependent builds use empty DOWNLOAD_COMMAND and official patches','private_path':'/deps/protobuf/protobuf-25.1.tar.gz'},
            {'dependency':'makeself','interface':'SDK packaging script tools copied and compared in separate read-only CPU container before freeze; no gitcode tar archive claim','private_path':'new source/third_party/makeself'}],
        'disabled_branches':[
            {'dependency':'gtest','reason':'ENABLE_TEST=OFF and all UT flags default OFF'},
            {'dependency':'cann_device','reason':'host-only open project; no device superbuild'},
            {'dependency':'extern_opbuild_gen_transformer','reason':'requires NOT BUILD_OPEN_PROJECT; configured BUILD_OPEN_PROJECT=ON'},
            {'dependency':'makeself FetchContent_MakeAvailable(${MAKESELF_NAME})','reason':'packager scripts prepopulated; custom host package uses SDK CPack script'},
            {'dependency':'other cann-cmake third_party modules','reason':'not called by the reviewed enabled root dependency chain; full inventory retained separately'}]}


if __name__=='__main__':
    Path(__file__).with_name('offline_cmake_contract.json').write_text(json.dumps(generate(),indent=2)+'\n',encoding='utf-8')
