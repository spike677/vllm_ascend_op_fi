"""Real two-process CPU Gloo calibration plus strict measured-sample admission."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from types import SimpleNamespace

import torch

from fault_injection.collective25 import ReduceScatterSession, MeasuredCollectivePair


class Collective25(unittest.TestCase):
    def test_common_worker_binding_requires_real_calibration_before_install(self):
        from demoncat.runtime.vllm_ascend.worker_extension.providers.tensor_modules import install_tensor_replay
        worker=SimpleNamespace(model_runner=SimpleNamespace(model=torch.nn.Identity().eval()))
        with self.assertRaisesRegex(RuntimeError,'measured same-input'):
            install_tensor_replay(worker,'25.reduce_scatter',replay=lambda m,p:m(torch.ones(2)),
                synchronize=lambda:None,workload_kind='constructed_module',simulation=True)
        self.assertFalse(hasattr(worker,'_dcat_tensor_module_bindings'))

    def test_unissued_or_synthetic_pair_cannot_arm(self):
        model=torch.nn.Identity()
        with self.assertRaisesRegex(RuntimeError,'measured same-input'):
            ReduceScatterSession(model,mode='replay_delta',pair={'delta':[1.]},simulation=True)
        fake=MeasuredCollectivePair(object(),model,{},torch.ones(2),torch.ones(1),torch.zeros(1),(0,1),object())
        with self.assertRaisesRegex(RuntimeError,'not issued'):
            ReduceScatterSession(model,mode='replay_delta',pair=fake,simulation=True)

    def test_two_real_gloo_ranks_observe_no_delta_and_restore(self):
        root=Path(__file__).resolve().parents[1]
        script=root/'experiments/engineering_20261117/collective25_probe.py'
        with tempfile.TemporaryDirectory(prefix='dcat-collective25-') as folder:
            folder=Path(folder)
            children=[]
            try:
                for rank in range(2):
                    command=[sys.executable,str(script),'--cpu','--file-store',str(folder/'store'),
                             '--rank',str(rank),'--world-size','2','--out',str(folder/'result')]
                    children.append(subprocess.Popen(command,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,
                        text=True,cwd=root,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0)))
                for child in children:
                    output,_=child.communicate(timeout=45)
                    if child.returncode and 'makeDeviceForHostname(): unsupported gloo device' in output:
                        self.skipTest('This torch build has no usable Gloo device; no collective was executed')
                    self.assertEqual(child.returncode,0,output)
                for rank in range(2):
                    receipt=json.loads((folder/'result'/f'rank{rank}.json').read_text())
                    self.assertEqual(receipt['status'],'NO_REPLAY_SAMPLE')
                    self.assertEqual(receipt['reason'],'NO_NATIVE_SAME_INPUT_DIFFERENCE_OBSERVED')
                    self.assertEqual(receipt['original_collective_calls'],2)
                    self.assertTrue(receipt['repeat_exact'] and receipt['off_exact'])
                    self.assertEqual(receipt['observation']['hits'],0)
                    events=receipt['observation']['events']
                    self.assertTrue(all(e['signature']['backend']=='gloo' for e in events))
                    self.assertTrue(all(e['native_sequence_after']>e['native_sequence_before'] for e in events))
            finally:
                for child in children:
                    if child.poll() is None:
                        child.kill();child.communicate(timeout=10)
                    if child.stdout is not None and not child.stdout.closed:
                        child.stdout.close()


if __name__=='__main__':unittest.main()
