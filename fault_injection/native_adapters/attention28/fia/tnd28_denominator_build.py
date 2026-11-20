"""Exact new variant hooks for the existing fresh FIA builder (CPU/SDK only)."""
import hashlib
import importlib.util
import json
from pathlib import Path


def recipe(here):
    spec=importlib.util.spec_from_file_location('tnd28_replay_recipe',here/'apply_tnd28_denominator.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


def configure(builder,source_in,base):
    r=recipe(base.HERE);require=base.require;sha=base.sha
    require(builder.sdk.name=='cann-8.5.1'
        and getattr(builder.args,'opdef_compat',None)=='ascend910b-cann851'
        and getattr(builder.args,'sdk_interface_manifest',None)
        and getattr(builder.args,'sdk_interface_sha256',None)==base.TND28_SDK_INTERFACE_SHA256
        and sha(builder.args.sdk_interface_manifest)==base.TND28_SDK_INTERFACE_SHA256,
        'TND28_REPLAY_EXACT_CANN851_INTERFACE_REQUIRED')
    raw={p:base.git_blob(source_in,p) for p in (*r.SOURCE_SHA,*r.EXTRA_UNCHANGED)}
    final,_,projection,fields=r.materials(raw)
    require(builder.receipt==r.receipt(raw,final,projection,fields),'TND28_REPLAY_RECONSTRUCTED_RECEIPT_DIFFERS')
    for name,value in final.items():
        p=source_in/name
        require(not p.is_symlink() and p.read_bytes()==value,'TND28_REPLAY_RECONSTRUCTED_BYTES_DIFFER:'+name)
    builder.variant=r.VARIANT;builder.vendor_name=r.VENDOR
    builder.report.update(variant=r.VARIANT,normal_only=False,**{name:False for name in base.TND28_DENIED})
    builder.report.update(tnd28_source_closure_reconstructed=True,
        tnd28_source_closure={p:hashlib.sha256(b).hexdigest() for p,b in final.items()},
        tnd28_projection_sha256=hashlib.sha256(projection.encode()).hexdigest(),
        tnd28_typed_fields=fields,tnd28_typed_field_count=len(fields),
        tnd28_replay_recipe_sha256=sha(base.HERE/'apply_tnd28_denominator.py'),
        tnd28_replay_build_helper_sha256=sha(Path(__file__)),
        tnd28_sdk_interface_sha256=base.TND28_SDK_INTERFACE_SHA256,
        tnd28_builder_sha256=sha(Path(base.__file__)),
        tnd28_binary_registration_sha256=sha(base.HERE/'binary_registration.py'))


def cpu_checks(builder,base):
    ev=builder.ev;here=base.HERE
    borrow=ev/'tnd28_execute_borrow_contract'
    builder.run('tnd28_execute_borrow_compile',['g++','-std=c++17','-Wall','-Wextra','-Werror','-pedantic',
        '-pthread',here/'test_tnd28_execute_borrow.cpp','-o',borrow],env=builder.env)
    builder.run('tnd28_execute_borrow_contract',[borrow],env=builder.env)
    base.require((ev/'tnd28_execute_borrow_contract.log').read_text().strip()==
        'PASS_CPU_TND28_EXECUTE_BORROW_NO_DEVICE_NO_F','TND28_EXECUTE_BORROW_CPU_CONTRACT')
    builder.report['tnd28_execute_borrow_header_sha256']=base.sha(here/'tnd28_execute_borrow.h')
    builder.report['tnd28_execute_borrow_cpu_log_sha256']=base.sha(ev/'tnd28_execute_borrow_contract.log')
    observer=ev/'tnd28_input_observation_contract'
    builder.run('tnd28_input_observation_compile',['g++','-std=c++17','-Wall','-Wextra','-Werror','-pedantic',
        '-Wno-misleading-indentation','-pthread',here/'test_tnd28_input_observation.cpp','-ldl','-o',observer],env=builder.env)
    builder.run('tnd28_input_observation_contract',[observer],env=builder.env)
    base.require((ev/'tnd28_input_observation_contract.log').read_text().strip()==
        'PASS_CPU_TND28_TYPED_INPUT_OBSERVATION_NO_DEVICE_NO_F','TND28_TYPED_OBSERVER_CPU_CONTRACT')
    serializer=builder.deps/'opbase/src/nnopbase/individual_op/executor/indv_args.cpp'
    base.require(base.sha(serializer)=='b54c905c4cda0d049892b3ade7edd0aa51dce3452caf2dcf233722e892288fa5',
        'TND28_ACTUAL_DYNAMIC_LIST_SERIALIZER_SOURCE_CHANGED')
    builder.report.update(tnd28_input_cpu_contract_log_sha256=base.sha(ev/'tnd28_input_observation_contract.log'),
        tnd28_input_serializer_sha256=base.sha(serializer))
    binary=ev/'tnd28_replay_contract';program=here/'test_tnd28_denominator_contract.cpp'
    builder.run('tnd28_replay_contract_compile',['g++','-std=c++17','-Wall','-Wextra','-Werror','-pedantic',
        '-pthread',program,'-o',binary],env=builder.env)
    builder.run('tnd28_replay_contract',[binary],env=builder.env)
    base.require((ev/'tnd28_replay_contract.log').read_text().strip()=='PASS_CPU_TND28_DENOMINATOR_CONTRACT_NOT_NPU',
                 'TND28_REPLAY_CPU_CONTRACT_REQUIRED')
    missing=ev/'tnd28_missing_input_contract'
    builder.run('tnd28_missing_input_compile',['g++','-std=c++17','-Wall','-Wextra','-Werror','-pedantic',
        '-pthread','-DDCAT_TND28_NO_INPUT_GATE',program,'-o',missing],env=builder.env)
    builder.run('tnd28_missing_input_contract',[missing],env=builder.env)
    base.require((ev/'tnd28_missing_input_contract.log').read_text().strip()=='PASS_CPU_TND28_MISSING_INPUT_SYMBOL_REFUSED',
                 'TND28_REPLAY_MISSING_INPUT_SYMBOL_MUST_REFUSE')
    builder.run('actual_tnd28_replay_split_matrix',[builder.args.python,here/'compile_tnd28_denominator_matrix.py',
        '--source',builder.source,'--out',ev/'tnd28_replay_matrix'],env=builder.env)
    matrix=json.loads((ev/'tnd28_replay_matrix/receipt.json').read_bytes())
    base.require(matrix.get('supported_core_profiles')==list(range(16,27)) and matrix.get('cases')==22
        and matrix.get('selected_scheduling_cores')==8
        and len(matrix.get('original_platform_schedule_refusals',[]))==11
        and matrix.get('source_execution')=='PINNED_ORIGINAL_SPLIT_CORE_CPU'
        and matrix.get('npu_executed') is False and matrix.get('qualification_granted') is False,
        'TND28_REPLAY_NO_COMPLETE_NATIVE_SPLIT_PROFILE')
    builder.report.update(tnd28_replay_matrix_sha256=base.sha(ev/'tnd28_replay_matrix/matrix.jsonl'),
        tnd28_replay_matrix_receipt_sha256=base.sha(ev/'tnd28_replay_matrix/receipt.json'),
        tnd28_replay_contract_source_sha256=base.sha(program),tnd28_replay_contract_executable_sha256=base.sha(binary),
        tnd28_replay_contract_log_sha256=base.sha(ev/'tnd28_replay_contract.log'),
        tnd28_missing_input_contract_log_sha256=base.sha(ev/'tnd28_missing_input_contract.log'),
        tnd28_missing_input_executable_sha256=base.sha(missing),
        supported_cpu_core_profiles=matrix['supported_core_profiles'],actual_worker_platform_observed=False)


def finish_check(builder,base):
    ev=builder.ev;report=builder.report
    private=builder.vendor/'op_api/lib/libcust_opapi.so'
    builder.run('tnd28_execute_borrow_exports',['nm','-D','--defined-only',private],env=builder.env)
    exports=(ev/'tnd28_execute_borrow_exports.log').read_text().splitlines()
    symbols=('dcat_tnd28_execute_borrow_query_v1','dcat_tnd28_public_plan_bind_v2',
        'dcat_tnd28_public_plan_clear_v2','dcat_tnd28_public_plan_query_v2')
    for symbol in symbols:
        query=[line.split() for line in exports if line.split() and line.split()[-1].split('@',1)[0]==symbol]
        base.require(len(query)==1 and len(query[0])==3 and query[0][1]=='T',
            'TND28_LIVE_BORROW_SYMBOL_NOT_EXPORTED:'+symbol)
    report['tnd28_execute_borrow_query']=dict(symbol='dcat_tnd28_execute_borrow_query_v1',
        library_sha256=base.sha(private),exports_sha256=base.sha(ev/'tnd28_execute_borrow_exports.log'),
        live_public_plan_symbols=list(symbols[1:]),
        read_only=True,fault_authorized=False)
    base.require(report['tnd28_execute_borrow_header_sha256']==base.sha(base.HERE/'tnd28_execute_borrow.h')
        and report['tnd28_execute_borrow_cpu_log_sha256']==base.sha(ev/'tnd28_execute_borrow_contract.log'),
        'TND28_EXECUTE_BORROW_BUILD_EVIDENCE_CHANGED')
    spec=importlib.util.spec_from_file_location('tnd28_actual_input_compile',base.HERE/'tnd28_input_compile.py')
    compiler=importlib.util.module_from_spec(spec);spec.loader.exec_module(compiler)
    report['tnd28_actual_input_compilation']=compiler.verify_compile_receipt(ev,builder.source,builder.sdk)
    observation=json.loads((ev/'tnd28_compiled_input_observer.json').read_bytes())
    base.require(observation['variant']==builder.variant and observation['observer_only'] is True
        and observation['header_sha256']==base.sha(base.HERE/'tnd28_input_observation.h')
        and observation['launcher_sha256']==base.sha(base.HERE/'tnd28_input_compile.py')
        and observation['patched_sha256']==base.sha(observation['unit'])
        and observation['header_sha256']==base.sha(observation['header'])
        and observation['execute_borrow_header_sha256']==base.sha(base.HERE/'tnd28_execute_borrow.h')
        and observation['execute_borrow_header_sha256']==base.sha(observation['execute_borrow_header'])
        and observation['fault_authorized'] is False,'TND28_COMPILED_INPUT_SOURCE_BINDING_REQUIRED')
    report['tnd28_compiled_input_observer']=observation
    report['tnd28_compiled_input_observer_receipt_sha256']=base.sha(ev/'tnd28_compiled_input_observer.json')
    base.require(all(report.get(name) is False for name in base.TND28_DENIED)
        and report.get('tnd28_source_closure_reconstructed') is True
        and report.get('tnd28_host_header_compiled_with_actual_sdk') is True
        and report.get('compiled_keys')==base.TND28_REPLAY_KEYS
        and report.get('tnd28_replay_matrix_receipt_sha256')==base.sha(ev/'tnd28_replay_matrix/receipt.json')
        and report.get('tnd28_replay_matrix_sha256')==base.sha(ev/'tnd28_replay_matrix/matrix.jsonl')
        and report.get('tnd28_replay_contract_log_sha256')==base.sha(ev/'tnd28_replay_contract.log')
        and report.get('tnd28_missing_input_contract_log_sha256')==base.sha(ev/'tnd28_missing_input_contract.log')
        and report.get('tnd28_public_schema_sha256')==base.sha(ev/'tnd28_schema/fia_abi_schema.json')
        and report.get('tnd28_schema_compile_sha256')==base.sha(ev/'tnd28_schema/compile_receipt.json')
        and report.get('tnd28_schema_selftest_sha256')==base.sha(ev/'tnd28_schema/fia_abi_selftest.json')
        and report.get('tnd28_macro_dependency_sha256')==base.sha(ev/'tnd28_schema/macro_dependency.json')
        and report.get('tnd28_macro_header_sha256')==base.TND28_MACRO_HEADER_SHA256
        and base.sha(report['tnd28_macro_header_path'])==base.TND28_MACRO_HEADER_SHA256,
        'TND28_REPLAY_COMPLETE_CPU_BUILD_EVIDENCE_REQUIRED')
    report.update(qualification_missing=['new package actual loaded-function and registered ELF identity',
        'current public/typed prepare/internal Slice/kernel input binding; source113 journal had zero events',
        'fresh stock/private native-normal AND M128 control normal comparison on the same actual operands',
        'actual host/POD/key/scoped launch equality and actual core profile',
        'single actual denominator replay numerical effect; OFF and same Worker health',
        'main-project active-request source integration'],
        numerical_effect_observed=False,actual_worker_platform_observed=False)
