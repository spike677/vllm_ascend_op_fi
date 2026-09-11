import asyncio
import copy
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'tools'))
from private_service_owner import PrivateTransport, PrivateServiceOwner
from ascend_fi.inspect.private_demo_auth import PrivateDemoAuth
from ascend_fi.inspect.client import InspectClient
from ascend_fi.inspect.target_resolver import resolve_targets
from ascend_fi.inspect.plan_builder import build_target_plan, verify_plan
from runtime_inspect.test_model_targeting import report_fixture, ARM, ROLES


class PrivateDeploymentTests(unittest.TestCase):
    def test_tp1_28_layers_not_hardcoded_64(self):
        d, m, e, s = report_fixture(layers=28)
        d['topology']['tensor_parallel_size'] = 1
        report = resolve_targets(d, m, e, s)
        plan = build_target_plan([report], roles=ROLES, rank_filter=[0], scale_multiplier=20,
            reference_arm=ARM, max_tokens=512, max_prompt_tokens=80)
        verify_plan(plan)
        self.assertEqual(len(plan['selected_modules']), 112)
        self.assertEqual(plan['request_window']['max_hits'], 112 * (512 + 80 + 16) * 2)

    def test_decoder_does_not_invent_token_ids(self):
        row = PrivateTransport._choice_row({'id': 'p', 'category': 'writing'}, {}, 200,
            {'choices': [{'text': 'abc', 'logprobs': {'tokens': ['a', 'bc']}, 'finish_reason': 'stop'}],
             'usage': {'completion_tokens': 2}}, None)
        self.assertIsNone(row['token_ids'])
        self.assertEqual(row['token_pieces'], ['a', 'bc'])
        self.assertEqual(row['comparison_basis'], 'token_pieces')

    def test_inspect_transport_choice_explicit(self):
        with self.assertRaises(ValueError):
            InspectClient('http://127.0.0.1:1', transport=object(), api_key_file='not-read')

    def test_private_rpc_and_generation_auth(self):
        async def app(scope, receive, send):
            await send({'type': 'http.response.start', 'status': 200})
        async def run(path, header):
            out = []
            async def send(v): out.append(v)
            async def receive(): return {}
            with patch.dict(os.environ, {'VLLM_API_KEY': 'x' * 40}):
                await PrivateDemoAuth(app)({'type': 'http', 'path': path, 'headers': header}, receive, send)
            return out[0]['status']
        for path in ('/collective_rpc', '/v1/chat/completions', '/tokenize'):
            self.assertEqual(asyncio.run(run(path, [])), 401)
            self.assertEqual(asyncio.run(run(path, [(b'authorization', b'Bearer ' + b'x' * 40)])), 200)
        self.assertEqual(asyncio.run(run('/health', [])), 200)

    def test_wrong_lease_rejected_before_any_request(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'access.token').write_text('x' * 40)
            (root / 'owner.json').write_text(json.dumps({'owner': 'OTHER'}))
            d = {'base_url': 'http://127.0.0.1:1234', 'tp_size': 1, 'device_ids': [5],
                'reference_path': str(root / 'missing.py'), 'source_root': str(root),
                'served_model_name': 'small', 'reference_arm': copy.deepcopy(ARM),
                'lock_owner_path': str(root / 'owner.json'), 'owner': 'OURS', 'lease_id': '123'}
            owner = PrivateServiceOwner(root, {}, d)
            with self.assertRaisesRegex(RuntimeError, 'EXCLUSIVE_LEASE_MISMATCH'):
                owner.guard()


if __name__ == '__main__':
    unittest.main()
