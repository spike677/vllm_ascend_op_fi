import hashlib,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch
from PIL import Image
from fault_injection.environment import _live_soc,classify_soc
from fault_injection.vision_input import load_fixed_image


class IdentityTests(unittest.TestCase):
    def test_a3_is_runtime_name_not_generic_smi_label(self):
        backend=NS(npu=NS(is_initialized=lambda:True,current_device=lambda:1,
            get_device_name=lambda index:'Ascend910_9382' if index==1 else 'wrong'))
        with patch.dict('sys.modules',{'torch_npu':backend}):soc,source=_live_soc()
        self.assertEqual(soc['observed_name'],'Ascend910_9382')
        self.assertEqual((soc['family'],soc['logical_device']),('A3',1))
        self.assertEqual(source,'INITIALIZED_TORCH_NPU_DEVICE_QUERY')
        self.assertIsNone(classify_soc('Ascend910')['family'])
        self.assertEqual(classify_soc('Ascend910B4')['family'],'A2')

    def test_fixed_image_expected_hash_optional_dimensions_required(self):
        with tempfile.TemporaryDirectory() as root:
            p=Path(root)/'image.png';Image.new('RGB',(64,64)).save(p)
            sha=hashlib.sha256(p.read_bytes()).hexdigest()
            image,record=load_fixed_image(p,sha)
            self.assertEqual(image.size,(64,64));self.assertEqual(record['file_sha256'],sha)
            _,unchecked=load_fixed_image(p)
            self.assertEqual(unchecked['file_sha256'],sha);self.assertFalse(unchecked['expected_sha256_check'])
            for invalid in ('29148cb0...a7e37','0'*64):
                with self.assertRaises(ValueError):load_fixed_image(p,invalid)
            Image.new('RGB',(32,64)).save(p)
            with self.assertRaisesRegex(ValueError,'64X64'):load_fixed_image(p,hashlib.sha256(p.read_bytes()).hexdigest())

    def test_native_builder_has_no_patchelf_subprocess_dependency(self):
        import ast,inspect
        from tools import build_native_regions
        tree=ast.parse(inspect.getsource(build_native_regions))
        commands=[node.value for node in ast.walk(tree) if isinstance(node,ast.Constant) and isinstance(node.value,str)]
        self.assertNotIn('patchelf',commands)
