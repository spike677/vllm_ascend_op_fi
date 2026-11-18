"""CPU contracts for the separate normal-only route; no native qualification."""
import copy
import json
import os
from pathlib import Path
import tempfile
import threading
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

import torch
from demoncat.runtime.vllm_ascend.worker_extension import paged33_arch22_bridge as bridge
from demoncat.runtime.vllm_ascend.worker_extension.attention_native import OwnedLaunchScope
from fault_injection.native_adapters.attention28.launch import LaunchObserver
from fault_injection.native_adapters.attention28.fia import apply_tnd33_arch22_bridge as recipe
from fault_injection.native_adapters.attention28.fia import binary_registration, build_fresh
from test_paged33_tnd import fixture as paged_fixture


def fixture():
    actual, boundary = paged_fixture()
    actual['atten_mask'] = torch.ones(2048, 2048, dtype=torch.int8).triu(1)
    boundary.update(module_path='model.layers.0.self_attn.attn', module_object_id=123)
    return actual, boundary


def source_fixture():
    return {recipe.PATHS[0]: 'namespace optiling {\nclass X {};\n}',
        recipe.PATHS[1]: 'void arithmetic(){ CalcInnerSize(); SplitCore(); GenTilingKey(); }\n'
                        '    return ge::GRAPH_SUCCESS;\n}\n\n// unchanged',
        recipe.ROUTE: 'CheckRouteToFiaPrerequisites(context);\n'+recipe.ROUTE_ANCHOR+'\nreturn true;'}


def live_scope():
    observer = LaunchObserver.__new__(LaunchObserver)
    state = dict(active=False, token=None, end='ok', syncs=0)
    class Fn:
        def __init__(self, fn):self.fn=fn
        def __call__(self, *args):return self.fn(*args)
    def begin(token):state.update(active=True, token=token);return 0
    def end():
        if state['end']=='before':raise RuntimeError('before native end')
        state['active']=False
        if state['end']=='after':raise RuntimeError('after native end')
        return 0
    observer.lib=NS(dcat_attention_observer_scope_active_v1=Fn(lambda:int(state['active'])),
        dcat_attention_observer_scope_matches_v1=Fn(lambda token: int(state['active'] and token==state['token'])),
        dcat_attention_observer_begin_v1=Fn(begin),dcat_attention_observer_end_v1=Fn(end))
    observer.active=False
    observer.synchronize=lambda:None
    return OwnedLaunchScope(observer), state


class Arch22NormalContracts(unittest.TestCase):
    def test_patch_changes_only_route_preference_and_capture_not_arithmetic_or_kernel(self):
        original=source_fixture();after=recipe.patched(original)
        self.assertEqual(after[recipe.PATHS[0]].replace('#include "paged33_arch22_bridge.h"\n\n',''), original[recipe.PATHS[0]])
        self.assertEqual(after[recipe.PATHS[1]].replace(
            '    paged33_arch22_bridge::Capture(*fiaInfo_, context_, tilingData_, workspaceSize_, sInnerSize_);\n',''),
            original[recipe.PATHS[1]])
        self.assertEqual(after[recipe.ROUTE].replace(
            'CheckSpecConditions(context) && !paged33_arch22_bridge::Matches(context)', 'CheckSpecConditions(context)'),
            original[recipe.ROUTE])
        self.assertFalse(any('op_kernel/' in path for path in after))
        for path in original:
            broken=dict(original);broken[path]+=original[path]
            with self.assertRaisesRegex(ValueError,'ambiguous'):recipe.patched(broken)
        with self.assertRaisesRegex(ValueError,'anchor'):recipe.patched(after)

    def test_changed_pristine_source_refuses_before_patch(self):
        raw={name:b'not the actual source' for name in recipe.SOURCE_SHA}
        with self.assertRaisesRegex(RuntimeError,'PRISTINE_SOURCE_DIFFERS'):recipe.verify_source(raw)

    def test_normal_build_preserves_warmup_entries_and_has_no_fault_entries(self):
        self.assertEqual(set(recipe.KEYS), set(build_fresh.TND33_ARCH22_KEYS))
        self.assertEqual(set(recipe.KEYS), binary_registration.KEYS[recipe.VARIANT])
        self.assertIn(103000000010300003,recipe.KEYS)
        self.assertTrue(set(build_fresh.TND33_FAI_FD_KEYS[:6]).issubset(recipe.KEYS))
        self.assertFalse(set(build_fresh.TND33_FAI_FD_KEYS[6:]) & set(recipe.KEYS))
        with tempfile.TemporaryDirectory() as root:
            binary=Path(root)/'new.o';binary.write_bytes(b'cpu fixture')
            metadata=dict(binFileName=binary_registration.BINARY,binFileSuffix='.o',
                sha256=binary_registration.sha(binary),coreType='MIX',taskRation='tilingKey',
                kernelList=[dict(tilingKey=key) for key in recipe.KEYS],supportInfo=dict(
                    simplifiedKeyMode=0,simplifiedKey=['test'],optionalInputMode='gen_placeholder',
                    optionalOutputMode='gen_placeholder',tilingKey=list(recipe.KEYS)))
            binary_registration.validate_metadata(metadata,binary,recipe.VARIANT)
            metadata['kernelList'][0]['tilingKey']=6000000000010200103
            with self.assertRaisesRegex(RuntimeError,'KEY_SET_DIFFERS'):
                binary_registration.validate_metadata(metadata,binary,recipe.VARIANT)

    def test_actual_native_layout_identity_and_int8_mask_are_required(self):
        actual,boundary=fixture()
        before=actual['key'].untyped_storage()._cdata
        scope=bridge.validate_normal_call(actual,boundary,allow_cpu_test=True)
        self.assertEqual(scope['native_route'],'FiaTilingNonQuant')
        self.assertTrue(scope['normal_only'])
        self.assertEqual(scope['omitted_argument_defaults'],dict(pre_tokens=2147483647,next_tokens=2147483647))
        self.assertNotIn('pre_tokens',actual)
        self.assertEqual(before,actual['key'].untyped_storage()._cdata)
        for change in ('equal_copy','mask_dtype','mask_stride','optional','unknown','lse','cpu_on_native'):
            with self.subTest(change=change):
                modified=dict(actual)
                if change=='equal_copy':modified['key']=actual['key'].clone()
                if change=='mask_dtype':modified['atten_mask']=actual['atten_mask'].bool()
                if change=='mask_stride':modified['atten_mask']=actual['atten_mask'].T
                if change=='optional':modified['learnable_sink']=torch.zeros(1)
                if change=='unknown':modified['unreviewed_feature']=None
                if change=='lse':modified['softmax_lse_flag']=True
                with self.assertRaises(RuntimeError):
                    bridge.validate_normal_call(modified,boundary,allow_cpu_test=change!='cpu_on_native')

    def native_fixture(self):
        actual,boundary=fixture()
        typed={'baseParams.bSize':1,'baseParams.n2Size':8,'baseParams.gSize':2,'baseParams.headDim':128,
            'baseParams.headDimRope':0,'baseParams.softmaxLseFlag':0,'baseParams.actualSeqS1Dims':1,
            'baseParams.actualSeqS2Dims':1,'innerSplitParams.s2BaseSize':512,'pageAttenParams.blockSize':16,
            'pageAttenParams.maxBlockNumPerBatch':65,'maskParams.attenMaskFlag':1,'maskParams.sparseMode':3,
            'maskParams.preToken':2147483647,'maskParams.nextToken':2147483647,
            'leftPaddingParams.qPaddingFlag':0,'leftPaddingParams.kvPaddingFlag':0,
            'prefixParams.prefixFlag':0,'pseParams.pseShiftFlag':0,'baseParams.s1Size':1,'baseParams.s2Size':1025,
            'baseParams.scaleValue':0.0883883461356163}
        host=dict(schema='demoncat.paged33.arch22_normal_host.v1',route='FiaTilingNonQuant',
            mode='arch22_normal',normal_route_preference_changed=True,eligible=True,policy_applied=False,
            native_s2=512,selected_s2=512)
        launch=dict(tiling_key=103000000010300003,block_dim=16,schedule_mode=1,typed=typed)
        return actual,boundary,host,launch

    def test_actual_final_public_scalars_are_checked_and_stock_abi_is_not_invented(self):
        actual,boundary,host,launch=self.native_fixture()
        receipt=bridge.validate_native(launch,host,actual,boundary)
        self.assertFalse(receipt['stock_abi_decoded'])
        self.assertFalse(receipt['stock_private_exact_proven'])
        self.assertEqual(receipt['actual_next_tokens'],2147483647)
        for field in ('baseParams.headDim','innerSplitParams.s2BaseSize','maskParams.nextToken',
                      'pageAttenParams.maxBlockNumPerBatch','baseParams.scaleValue'):
            bad=copy.deepcopy(launch);bad['typed'][field]+=1
            with self.subTest(field=field),self.assertRaises(RuntimeError):
                bridge.validate_native(bad,host,actual,boundary)
        bad=dict(launch,tiling_key=5100000000010200103)
        with self.assertRaisesRegex(RuntimeError,'STOCK_ROUTE_ENTRY_DIFFERS'):
            bridge.validate_native(bad,host,actual,boundary)

    def driver(self, folder):
        native=bridge.Arch22NormalNativeCall.__new__(bridge.Arch22NormalNativeCall)
        native.scope,state=live_scope()
        native.thread=threading.get_ident();native.policy_prefix='DCAT33_TND_'
        native.folder=Path(folder);native.environment={};native.failures=[];native.calls=[]
        native.runtime_environment={};native.observer_sha256='a'*64
        native.fingerprint=dict(normal_only=True,kernel_sha256='b'*64);native.build=dict(vendor='cpu-contract')
        native.runtime_dir=Path(folder);native.schema={};native.metadata={};native.abi={}
        return native,state

    def test_f_refuses_before_original_call_or_native_scope(self):
        native=bridge.Arch22NormalNativeCall.__new__(bridge.Arch22NormalNativeCall)
        calls=[]
        with self.assertRaisesRegex(RuntimeError,'NEVER_AUTHORIZES_FAULT'):
            native.execute({}, {}, 'fault', lambda:calls.append('original'))
        self.assertEqual(calls,[])

    def test_real_scope_and_original_return_survive_normal_bridge_verification(self):
        actual,boundary,host,launch=self.native_fixture()
        output=object();calls=[]
        original_validate=bridge.validate_normal_call
        with tempfile.TemporaryDirectory() as folder, patch.dict(os.environ,{},clear=True), \
             patch.object(torch,'npu',NS(synchronize=lambda:None),create=True), \
             patch('demoncat.runtime.vllm_ascend.worker_extension.paged33_loading.validate_environment'), \
             patch.object(bridge,'validate_normal_call',side_effect=lambda a,b:original_validate(a,b,allow_cpu_test=True)):
            native,state=self.driver(folder)
            def original():
                calls.append(actual)
                self.assertTrue(state['active'])
                row=dict(host,token=os.environ['DCAT33_TND_TOKEN'])
                (Path(folder)/'host.jsonl').write_text(json.dumps(row)+'\n')
                return output
            with patch.object(bridge,'verify_launch',return_value=launch):
                returned,receipt=native.execute(actual,boundary,'baseline',original)
            self.assertIs(returned,output);self.assertEqual(len(calls),1)
            self.assertFalse(receipt['selection_applied'])
            self.assertEqual(native.state()['environment_owned'],0)
            self.assertFalse(state['active'])

    def test_before_end_failure_retains_scope_and_environment_until_retry(self):
        with tempfile.TemporaryDirectory() as folder,patch.dict(os.environ,{},clear=True), \
             patch.object(torch,'npu',NS(synchronize=lambda:None),create=True):
            native,state=self.driver(folder);state['end']='before'
            with self.assertRaisesRegex(RuntimeError,'before native end'):
                native._run('a'*32,{'DCAT33_TND_TOKEN':'a'*32},lambda:object(),lambda:{})
            self.assertTrue(native.scope.owned and state['active'])
            self.assertTrue(native.environment)
            self.assertIn('before native end',native.scope.first_error)
            state['end']='ok';native.stop()
            self.assertFalse(native.scope.owned or native.environment)
            self.assertIn('before native end',native.scope.first_error)

    def test_foreign_thread_and_foreign_token_cannot_clear_scope(self):
        with tempfile.TemporaryDirectory() as folder,patch.dict(os.environ,{},clear=True), \
             patch.object(torch,'npu',NS(synchronize=lambda:None),create=True):
            native,state=self.driver(folder);native.scope.owned=True;native.scope.token='a'*32
            state.update(active=True,token=('b'*32).encode())
            with self.assertRaisesRegex(RuntimeError,'TOKEN_OWNER_CHANGED'):native.stop()
            self.assertTrue(native.scope.owned and state['active'])
            state['token']=('a'*32).encode();native.thread=-1
            with self.assertRaisesRegex(RuntimeError,'THREAD_CHANGED'):native.stop()
            native.thread=threading.get_ident();native.stop()

    def test_after_end_exception_is_retired_by_actual_readback_and_keeps_first_error(self):
        with tempfile.TemporaryDirectory() as folder,patch.dict(os.environ,{},clear=True), \
             patch.object(torch,'npu',NS(synchronize=lambda:None),create=True):
            native,state=self.driver(folder);state['end']='after'
            with self.assertRaisesRegex(RuntimeError,'after native end'):
                native._run('a'*32,{'DCAT33_TND_TOKEN':'a'*32},lambda:object(),lambda:{})
            self.assertFalse(native.scope.owned or native.environment or state['active'])
            self.assertIn('after native end',native.scope.first_error)
            native.stop()


if __name__=='__main__':unittest.main()
