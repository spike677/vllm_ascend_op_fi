"""Run exact pinned collector factory/start/stop code with CPU-only I/O seams.

Set DCAT_POSTWRITE_COLLECTOR_SOURCES to a read-only copy of api_registry.py,
api_register.py and service.py. Their original bytes are hash-checked before
compilation. This is not a full PrecisionDebugger dump or an NPU qualification.
The real factory/template/registry/service methods run; module-hook/file/device
services are explicit CPU doubles because the installed collector imports NPU.
"""
import ast
import builtins
from contextlib import contextmanager
import functools
import hashlib
import inspect
import os
from pathlib import Path
import sys
from types import FunctionType, ModuleType, SimpleNamespace as NS
import unittest
from unittest.mock import patch
from typing import Dict,Any,Optional,Callable,Union,List,Tuple

import torch

from demoncat.runtime.vllm_ascend.worker_extension import postwrite_collector as cleanup
from demoncat.runtime.vllm_ascend.worker_extension import postwrite41 as adapter


SERVICE_SHA='d51804b819a07453733e52eb81ec132f2cfa9934bfa5b243aaf7ceecd8ad97b3'


def source(root,name,sha):
    path=(root/name).resolve(strict=True);raw=path.read_bytes()
    if hashlib.sha256(raw).hexdigest()!=sha:raise RuntimeError('PINNED_COLLECTOR_BYTES_REQUIRED:'+name)
    return path,raw


def selected(module,path,raw,names):
    tree=ast.parse(raw,filename=str(path))
    nodes=[node for node in tree.body if isinstance(node,(ast.ClassDef,ast.FunctionDef)) and node.name in names]
    if {node.name for node in nodes}!=set(names):raise RuntimeError('PINNED_DEFINITIONS_REQUIRED')
    # Extract code from the full source compilation. Recompiling an AST subset
    # changes Python 3.12's import-aware LOAD_GLOBAL/LOAD_ATTR instructions.
    compiled=compile(raw,str(path),'exec',dont_inherit=True)
    for node in nodes:
        function=FunctionType(cleanup._nested(compiled,node.name),vars(module),node.name)
        if isinstance(node,ast.ClassDef):
            bases=tuple(vars(module)[base.id] for base in node.bases)
            value=builtins.__build_class__(function,node.name,*bases)
        else:value=function
        setattr(module,node.name,value)


@contextmanager
def production_collector_sources(root,api):
    rp,rr=source(root,'api_registry.py',cleanup.REGISTRY_SHA256)
    tp,tr=source(root,'api_register.py',cleanup.TEMPLATE_SHA256)
    sp,sr=source(root,'service.py',SERVICE_SHA)
    registry,template=ModuleType(cleanup.REGISTRY_MODULE),ModuleType(cleanup.TEMPLATE_MODULE)
    registry.__file__,template.__file__=str(rp),str(tp)
    const=NS(SEP='.',API_WITH_SELF_ARG={},API_ATTR_LIST=[],API_DATA_PREFIX={'pytorch':{'NPU':'NPU'}},
        SUPPORT_API_DICT_KEY_MAP={'pytorch':{'NPU':'npu'}},CPU_LOWERCASE='cpu',DIST_API_TYPE_PREFIX='Distributed',
        CUDA_LOWERCASE='cuda',API='API',TOOL_NAME='PINNED_COLLECTOR_CPU_CONTRACT')
    names=list(adapter.APIS)
    registry.__dict__.update(inspect=inspect,Const=const,load_yaml=lambda path:{'npu':names},
        logger=NS(warning=lambda *a:None),Dict=Dict,Any=Any,Optional=Optional,Callable=Callable,
        Union=Union,List=List,Tuple=Tuple)
    selected(registry,rp,rr,('_get_attr','ApiWrapper','ApiRegistry'))
    class CpuHookModule(torch.nn.Module):
        def __init__(self,hook_build_func):super().__init__()
    template.__dict__.update(Const=const,HOOKModule=CpuHookModule,torch_device_guard=lambda f:f,
        functools=functools,npu_custom_functions={},_cuda_func_mapping={})
    selected(template,tp,tr,('npu_module_forward','ApiTemplate'))
    template.forward_methods={'NPU':template.npu_module_forward}
    template.api_register=registry.ApiRegistry({'pytorch':{'NPU':((api,),(api,))}}, {},'unused-cpu.yml',template.ApiTemplate)
    runtime=NS(is_running=False)
    globals_=dict(Runtime=runtime,Const=const,functools=functools)
    compiled=compile(sr,str(sp),'exec',dont_inherit=True)
    methods={name:FunctionType(cleanup._nested(compiled,'BaseService',name),globals_,name)
             for name in ('start','stop','_register_api_hook')}
    service=NS(_process_iteration=lambda:None,_is_dump_enabled=True,_is_debug_level=False,
        _is_need_module_hook=False,_need_stop_service=lambda:False,_is_no_dump_rank=False,
        _is_no_dump_step=False,_is_l2_level=False,_is_need_api_hook=True,first_start=True,
        current_iter=0,should_stop_service=False,model=None,hooked_modules=[],
        _register_hook=lambda:None,_change_jit_switch=lambda value:None,create_dirs=lambda:None,
        _flush_dump_result=lambda:None,logger=NS(info=lambda *a:None),dump_iter_dir='CPU_NO_DUMP',
        data_collector=NS(data_processor=NS()),config=NS(task='statistics'),
        build_hook=lambda *a:None,api_register=template.api_register,ori_customer_func={})
    collector=NS(service=service)
    with patch.dict(sys.modules,{cleanup.REGISTRY_MODULE:registry,cleanup.TEMPLATE_MODULE:template}):
        try:yield registry,template,service,collector,methods,runtime
        finally:
            owner=adapter._ACTIVE
            if owner is not None and owner.api_module is api:owner.close()


@unittest.skipUnless(os.environ.get('DCAT_POSTWRITE_COLLECTOR_SOURCES'),'pinned collector source copy not supplied')
class PinnedCollectorLifecycleTests(unittest.TestCase):
    def provider(self,api):
        import threading
        p=adapter.Postwrite41WorkerProvider.__new__(adapter.Postwrite41WorkerProvider)
        p.model=torch.nn.Linear(2,2);p.candidates=[]
        p.thread=threading.get_ident();p.current=p.lease=None
        p.installed=False;p.handles=[];p.patches=[];p.collectors=[];p.cleanup_receipts=[]
        p.api_module=api
        class Device:
            @staticmethod
            def reshape_and_cache(**kwargs):return None
        p.device_operator=Device
        return p

    def test_actual_registry_service_cycles_reinstall_rebuild_and_retry(self):
        root=Path(os.environ['DCAT_POSTWRITE_COLLECTOR_SOURCES'])
        for rebuild in (False,True):
            with self.subTest(rebuild=rebuild):
                api=NS(**{name:torch.add for name in adapter.APIS})
                p=self.provider(api);p.install()
                try:
                    with production_collector_sources(root,api) as (registry,template,service,collector,methods,runtime):
                        p.collectors.append(collector)
                        methods['_register_api_hook'](service)
                        for _ in range(2):
                            methods['start'](service,model=None,token_range=None,rank_id=0)
                            self.assertTrue(runtime.is_running)
                            self.assertEqual(api.npu_incre_flash_attention(torch.tensor(2),3).item(),5)
                            methods['stop'](service);self.assertFalse(runtime.is_running)
                        p.close()
                        self.assertEqual((len(p.handles),len(p.patches)),(0,0))
                        kept=api.npu_incre_flash_attention
                        p.install()
                        if rebuild:methods['_register_api_hook'](service)
                        for _ in range(2):
                            methods['start'](service,model=None,token_range=None,rank_id=0)
                            self.assertEqual(api.npu_incre_flash_attention(torch.tensor(4),1).item(),5)
                            methods['stop'](service)
                        expected=api.npu_incre_flash_attention
                        foreign=lambda *a,**kw:expected(*a,**kw)
                        api.npu_incre_flash_attention=foreign
                        before=(tuple(p.handles),tuple(p.patches))
                        with self.assertRaisesRegex(RuntimeError,'WRAPPER_OWNERSHIP_CHANGED'):p.close()
                        self.assertTrue(p.installed)
                        self.assertEqual((tuple(p.handles),tuple(p.patches)),before)
                        self.assertIs(api.npu_incre_flash_attention,foreign)
                        api.npu_incre_flash_attention=expected
                        p.close()
                        self.assertEqual((len(p.handles),len(p.patches)),(0,0))
                        if not rebuild:self.assertIs(api.npu_incre_flash_attention,kept)
                        self.assertEqual(api.npu_incre_flash_attention(torch.tensor(3),2).item(),5)
                        # Real registry restore/register cannot resurrect our wrapper.
                        template.api_register.restore_all_api();template.api_register.register_all_api()
                        self.assertEqual(api.npu_incre_flash_attention(torch.tensor(1),4).item(),5)
                        p.close()
                finally:
                    if p.installed:p.close()


if __name__=='__main__':unittest.main()
