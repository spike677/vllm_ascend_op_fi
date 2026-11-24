import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch,Mock
from experiments.worker_core_v11_20261118 import fault33_capture_worker as helper


class WorkerEvidenceTests(unittest.TestCase):
    def test_inode_mismatch_never_claims_loaded_file(self):
        with tempfile.TemporaryDirectory() as td:
            path=Path(td)/'opapi.so';path.write_bytes(b'not-an-elf')
            with patch.object(helper.os,'major',return_value=1,create=True),patch.object(helper.os,'minor',return_value=2,create=True):
                row=helper._file_identity(str(path),{str(path):{('00:00',0)}})
            self.assertEqual(row['status'],'UNPROVEN')
            self.assertEqual(row['reason'],'MAPPED_INODE_MISMATCH')

    def test_unmapped_flush_never_loads_another_library(self):
        with patch.object(helper,'_mapped_files',return_value={}),patch.object(helper.ctypes,'CDLL') as load:
            row=helper.flush(SimpleNamespace(rank=1),library_path='/missing.so',expected_sha256='0'*64,sequence='s1')
            self.assertEqual(row['status'],'CAPTURE_FLUSH_REFUSED');load.assert_not_called()

    def test_flush_preserves_rank_sequence_and_uses_noload(self):
        if not hasattr(os,'RTLD_NOLOAD'):
            noload=4;local=0
        else:noload=os.RTLD_NOLOAD;local=os.RTLD_LOCAL
        lib=Mock();lib.dcat33_capture_flush.return_value=0
        path=str(Path('/capture-v2.so').resolve())
        with patch.object(helper,'_mapped_files',return_value={path:set()}), \
             patch.object(helper,'_file_identity',return_value={'status':'MAPPED_FILE_VERIFIED','sha256':'a'*64}), \
             patch.object(helper.os,'RTLD_NOLOAD',noload,create=True),patch.object(helper.os,'RTLD_LOCAL',local,create=True), \
             patch.object(helper.ctypes,'CDLL',return_value=lib) as load:
            row=helper.flush(SimpleNamespace(rank=1),library_path=path,expected_sha256='a'*64,sequence='s1')
            self.assertEqual((row['rank'],row['sequence'],row['status']),(1,'s1','CAPTURE_SEALED'))
            load.assert_called_once_with(path,mode=noload|local)
