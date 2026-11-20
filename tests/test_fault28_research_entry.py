import unittest
from unittest.mock import patch
from types import SimpleNamespace
from fault_injection.research_entries import research_entry
from fault_injection.delivery_scope import describe_delivery
from fault_injection import attach,UnsupportedCall
from demoncat.runtime.vllm_ascend.worker_extension.registry import create_worker_provider,create_research_provider
from demoncat.runtime.vllm_ascend.worker_extension.config import FaultConfig

NAMES=('28.fa.legal_tiling','28.fia.legal_tiling','28.fia.denominator_replay')
class ResearchTests(unittest.TestCase):
    def test_public_routes_refuse_without_loading(self):
        for name in NAMES:
            config=FaultConfig(28,name[3:])
            with patch('fault_injection.resolve',side_effect=AssertionError('public fallback')):
                with self.assertRaisesRegex(UnsupportedCall,'EXPLICIT_RESEARCH_ENTRY_REQUIRED'):attach(object(),name)
            with self.assertRaisesRegex(RuntimeError,'HISTORICAL_ONLY|TND28_FAULT_OFF_RECOVERY_NOT_ACCEPTED'):
                create_worker_provider(config.mechanism,'1',SimpleNamespace(),object(),{})
    def test_explicit_research_dispatch_only_and_no_acceptance(self):
        for name in NAMES:
            with patch('demoncat.runtime.vllm_ascend.worker_extension.registry.importlib.import_module') as module:
                create_research_provider(name,'1',object(),object(),{})
                module.assert_called_once()
            record=research_entry(name)
            self.assertFalse(record['delivery_capability']);self.assertFalse(record['historical_worker_proof'])
            self.assertEqual(record['state'],describe_delivery(name)['state'])
            record['state']='modified';self.assertNotEqual(research_entry(name)['state'],'modified')
