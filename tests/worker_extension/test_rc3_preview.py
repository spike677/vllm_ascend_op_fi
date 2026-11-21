import unittest
import tempfile
from pathlib import Path
from tools.build_rc3_preview import write_json_bytes
from fault_injection.rc3_preview_profile import TEST_CANDIDATES,prepare_binding
from demoncat.runtime.vllm_ascend.worker_extension.config import FaultConfig


class PreviewTests(unittest.TestCase):
    def test_manifest_bytes_do_not_depend_on_host_newline(self):
        with tempfile.TemporaryDirectory() as root:
            p=Path(root)/'source-review.json';write_json_bytes(p,{'review':'source'})
            self.assertEqual(p.read_bytes(),b'{\n  "review": "source"\n}\n')
    def test_allow_list_is_not_a_live_grant(self):
        self.assertEqual(TEST_CANDIDATES,{'28.fia.denominator_replay','quant.batch_shared_scale'})
        group={'injection_policy':'ALL_RANK_EVERY_ELIGIBLE_CALL'}
        def refuse(*args):raise RuntimeError('SOURCE_OR_DEPLOYMENT_REJECTED')
        with self.assertRaisesRegex(RuntimeError,'SOURCE_OR_DEPLOYMENT'):
            prepare_binding(refuse)({},FaultConfig(28,'fia.denominator_replay'),group)
        calls=[];bind=prepare_binding(lambda *a:calls.append(a))
        with self.assertRaisesRegex(RuntimeError,'OUTSIDE_DELIVERY_SCOPE'):
            bind({},FaultConfig(18,'api_half'),group)
        with self.assertRaisesRegex(RuntimeError,'CONTINUOUS_GROUP'):
            bind({},FaultConfig(28,'fia.denominator_replay'),{})
        self.assertEqual(calls,[])
