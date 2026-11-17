"""Complete tensor evidence contracts; CPU fixtures grant no native admission."""
from pathlib import Path
import hashlib
import tempfile
import sys
from types import ModuleType
import unittest
from unittest.mock import patch

import torch
from demoncat.runtime.vllm_ascend.worker_extension import call_binding
from demoncat.runtime.vllm_ascend.worker_extension.paged33_tnd_probe import logical_cache_file, tensor_file, payload_identity, class_forward_binding


class CompleteNaturalPayloadTests(unittest.TestCase):
    def test_off_binding_retains_real_class_signature_after_collector_stop(self):
        class Module:
            def forward(self, lm_head, hidden_states):
                return hidden_states
        source=('def wrap_forward_with_hook_safety(module):\n'
                ' original_forward=module.forward\n'
                ' def wrapped_forward(*args,**kwargs):\n'
                '  if module is None: raise ValueError()\n'
                '  return original_forward(*args,**kwargs)\n'
                ' module.forward=wrapped_forward\n')
        folder=Path(self.enterContext(tempfile.TemporaryDirectory()))
        path=folder/'module_processor.py'
        path.write_text(source,encoding='utf-8')
        collector=ModuleType(call_binding.COLLECTOR_MODULE)
        collector.__file__=str(path)
        exec(compile(path.read_bytes(),str(path),'exec'),collector.__dict__)
        self.enterContext(patch.dict(sys.modules,{collector.__name__:collector}))
        self.enterContext(patch.object(call_binding,'REVIEWED_COLLECTOR_SHA256',
                                      hashlib.sha256(path.read_bytes()).hexdigest()))
        module=Module();head=object();hidden=object()
        collector.wrap_forward_with_hook_safety(module)
        wrapper=module.forward
        binding=class_forward_binding(module)
        actual=binding.bind(module,(head,hidden),{})
        self.assertIs(actual['hidden_states'],hidden)
        self.assertIs(module.forward,wrapper)
        self.assertIs(module.forward(head,hidden),hidden)
        module.forward=lambda *a,**k:hidden
        with self.assertRaises(RuntimeError):binding.bind(module,(head,hidden),{})

    def test_page_allocator_permutation_does_not_change_logical_evidence(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)
            logical=torch.arange(5*2*3,dtype=torch.float16).reshape(5,2,3)
            a=torch.full((4,2,2,3),-10,dtype=torch.float16)
            b=torch.full_like(a,99)
            for cache,pages in ((a,[0,2,3]),(b,[3,1,0])):
                for index,page in enumerate(pages):
                    chunk=logical[index*2:(index+1)*2]
                    cache[page,:len(chunk)]=chunk
            aa=logical_cache_file(a,[[0,2,3]],5,root/'a.bin')
            bb=logical_cache_file(b,[[3,1,0]],5,root/'b.bin')
            direct=tensor_file(logical,root/'logical.bin')
            self.assertEqual(aa['sha256'],bb['sha256'])
            self.assertEqual(aa['sha256'],direct['sha256'])
            self.assertNotEqual(aa['physical_pages'],bb['physical_pages'])
            self.assertFalse(aa['operator_input_repacked'])
            a[3,1]=float('nan')  # invalid tail is not a natural KV input
            self.assertEqual(logical_cache_file(a,[[0,2,3]],5,root/'tail.bin')['sha256'],aa['sha256'])
            a[2,0,0,0]+=1
            self.assertNotEqual(logical_cache_file(a,[[0,2,3]],5,root/'changed.bin')['sha256'],aa['sha256'])

    def test_complete_dtype_shape_and_nonfinite_evidence_is_preserved(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)
            value=torch.tensor([1.,float('nan')])
            row=tensor_file(value,root/'value.bin')
            self.assertFalse(row['finite'])
            a=dict(hidden=row,logits_hidden=row,logits=row,attention={'actual.layer':dict(output=row,key=row,value=row)})
            b=dict(a,logits=dict(row,shape=[1,2]))
            self.assertNotEqual(payload_identity(a),payload_identity(b))
            b=dict(a,attention={'actual.layer':dict(output=dict(row,dtype='torch.float16'),key=row,value=row)})
            self.assertNotEqual(payload_identity(a),payload_identity(b))

    def test_invalid_page_order_never_yields_a_complete_cache_hash(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)
            cache=torch.zeros(3,2,2,3)
            for index,pages in enumerate(([0,0],[0,3],[0],[0,1,2])):
                with self.subTest(pages=pages), self.assertRaisesRegex(RuntimeError,'PAGE_MAP_INVALID'):
                    logical_cache_file(cache,[pages],4,root/f'{index}.bin')


if __name__=='__main__':
    unittest.main()
