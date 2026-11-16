"""Architecture identity and installation inventory never imply native support."""
import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from fault_injection.environment import classify_soc, discover_installations, loaded_installation_links, _live_soc
from fault_injection.model_discovery import model_identity


class InstallationInventoryTests(unittest.TestCase):
    def create(self, root, version):
        base = root / ('cann-' + version) / 'aarch64-linux'
        base.mkdir(parents=True)
        (base / 'ascend_toolkit_install.info').write_text('version='+version+'\narch=aarch64\n')
        (base / 'lib64').mkdir()
        (base / 'lib64/libopapi.so').write_bytes(b'CPU fixture; not a vendor library')
        return base

    def test_all_installed_metadata_is_separate_from_loaded_identity(self):
        with tempfile.TemporaryDirectory() as tmp, patch('fault_injection.environment.platform.machine',return_value='aarch64'):
            root=Path(tmp)
            for version in ('8.5.1','9.1.0'):self.create(root,version)
            out=discover_installations(root,fingerprint_libraries=True)
        self.assertEqual([r['fields']['version'] for r in out['installations']],['8.5.1','9.1.0'])
        self.assertTrue(all(not r['native_loaded'] and r['compatibility']=='UNVERIFIED' for r in out['installations']))
        self.assertFalse(out['loaded_library_identity_proven'])
        self.assertTrue(all(r['libraries']['libopapi.so']['sha256'] for r in out['installations']))
        self.assertTrue(all(r['libraries']['libopapi.so']['elf_build_id'] is None for r in out['installations']))

    def test_proxy_and_mixed_installations_are_not_assumed_compatible(self):
        installations=[dict(toolkit='/actual/'+v, fields=dict(version=v), sha256=v) for v in ('8.5.1','9.1.0')]
        libraries=[dict(path='/actual/'+v+'/lib64/libnnopbase.so',name='libnnopbase.so',status='MAPPED_FILE_VERIFIED') for v in ('8.5.1','9.1.0')]
        libraries.append(dict(path='/proxy/cann-9.1.0/libopapi.so',name='libopapi.so',status='MAPPED_FILE_VERIFIED'))
        out=loaded_installation_links(libraries,installations)
        self.assertEqual(out['status'],'MULTIPLE_INSTALLATIONS_LOADED')
        self.assertEqual(out['libraries'][-1]['status'],'UNRESOLVED')
        self.assertFalse(out['abi_compatibility_proven'])

    def test_unverified_mapping_cannot_claim_an_installed_version(self):
        out=loaded_installation_links([dict(path='/actual/9/libopapi.so',name='libopapi.so',status='MAPPING_DELETED')],
            [dict(toolkit='/actual/9',fields=dict(version='9.0.0'),sha256='a')])
        self.assertEqual(out['libraries'][0]['matches'],[])

    def test_soc_query_never_initializes_an_unused_backend(self):
        calls=[]
        backend=SimpleNamespace(npu=SimpleNamespace(is_initialized=lambda:False,
            current_device=lambda:calls.append('device'),get_device_name=lambda _:calls.append('name')))
        with patch.dict('sys.modules',{'torch_npu':backend}):
            self.assertEqual(_live_soc(),(None,'BACKEND_NOT_INITIALIZED_NO_QUERY'))
        self.assertEqual(calls,[])

    def test_observed_soc_family_is_not_provider_certification(self):
        backend=SimpleNamespace(npu=SimpleNamespace(is_initialized=lambda:True,
            current_device=lambda:0,get_device_name=lambda _:'Ascend910B4'))
        with patch.dict('sys.modules',{'torch_npu':backend}):
            soc,source=_live_soc()
        self.assertEqual((soc['family'],soc['logical_device']),('A2',0))
        self.assertEqual(soc['provider_compatibility'],'UNVERIFIED')
        self.assertEqual(source,'INITIALIZED_TORCH_NPU_DEVICE_QUERY')
        self.assertEqual(classify_soc('Ascend910_9391')['family'],'A3')
        self.assertIsNone(classify_soc('user labeled A3')['family'])


class ModelIdentityTests(unittest.TestCase):
    def test_checkpoint_title_is_not_architecture_identity(self):
        from transformers import Qwen3Config,Qwen3ForCausalLM
        import torch
        with torch.device('meta'):
            model=Qwen3ForCausalLM(Qwen3Config(vocab_size=32,hidden_size=16,intermediate_size=32,
                num_hidden_layers=1,num_attention_heads=2,num_key_value_heads=1,head_dim=8))
        model.config._name_or_path='DeepSeek-R1-Distill-Qwen'
        out=model_identity(model,model.config)
        self.assertEqual(out['implementation_family'],'Qwen')
        self.assertFalse(out['native_provider_verified'])
        model.config.model_type='deepseek_v3'
        self.assertEqual(model_identity(model,model.config)['status'],'FAMILY_METADATA_CONFLICT')

    def test_unrecognized_custom_implementation_stays_unverified(self):
        out=model_identity(SimpleNamespace(),SimpleNamespace(model_type='qwen3'))
        self.assertEqual(out['configured_family'],'Qwen')
        self.assertIsNone(out['implementation_family'])
        self.assertEqual(out['status'],'CUSTOM_IMPLEMENTATION_UNVERIFIED')

    def test_actual_moe_and_mla_models_keep_roles_and_cpu_recovery(self):
        import torch
        torch.set_num_threads(1)
        path=Path(__file__).resolve().parents[1]/'experiments/overnight_20261116/model_matrix_probe.py'
        spec=importlib.util.spec_from_file_location('_overnight_models',path)
        probe=importlib.util.module_from_spec(spec);spec.loader.exec_module(probe)
        for name in ('qwen3_moe','deepseek_v3_mla_moe'):
            with self.subTest(name=name):
                out=probe.exercise(name)
                self.assertEqual(out['role_counts']['MLP_GATE'],2)
                self.assertEqual(out['role_counts']['MLP_EXPERTS'],2)
                self.assertTrue(out['transparent_exact']['cached_decode'])
                self.assertEqual(out['native_provider'],'UNVERIFIED')
                if name.startswith('deepseek'):
                    self.assertEqual(out['role_counts']['ATTENTION_KV_LATENT_PROJ'],2)
                    self.assertEqual(out['role_counts']['ATTENTION_Q_LATENT_PROJ'],2)
                    self.assertNotEqual(out['attention_calls'][0]['geometry']['qk_dim'],out['attention_calls'][0]['geometry']['value_dim'])


if __name__=='__main__':unittest.main()
