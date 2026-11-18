"""Probe controls call only explicit collection; no inference or SDK imports."""
from types import SimpleNamespace as NS
import inspect
import sys
import unittest
from unittest.mock import Mock, patch

from experiments.c2c3_closure_20261117 import providers_quant_inventory_probe as probe


class QuantProbeCollectionTests(unittest.TestCase):
    def test_query_is_read_only_and_advance_collect_preserves_failure(self):
        session = NS(report=Mock(return_value={'complete': False}),
                     collect=Mock(side_effect=RuntimeError('ACTUAL_FORMALS_REFUSED')))
        worker = NS(_demoncat_shared_scale_inventory=session)
        with patch.object(probe, 'actual', return_value=worker):
            self.assertEqual(probe.query(worker), {'complete': False})
            session.collect.assert_not_called()
            with self.assertRaisesRegex(RuntimeError, 'ACTUAL_FORMALS_REFUSED'):
                probe.advance(worker, collect=True)
        session.collect.assert_called_once_with()

    def test_collection_does_not_accept_capture_or_request_mapping(self):
        session = NS(report=Mock(), collect=Mock())
        worker = NS(_demoncat_shared_scale_inventory=session)
        with patch.object(probe, 'actual', return_value=worker):
            for kwargs in ({'collect': True, 'capture': True}, {'collect': True, 'request_ids': ['x']}, {'collect': 1}):
                with self.subTest(kwargs=kwargs), self.assertRaisesRegex(RuntimeError, 'EXPLICIT_INVENTORY_COLLECTION'):
                    probe.advance(worker, **kwargs)
        session.collect.assert_not_called()

    def test_integer_route_capture_is_opt_in_and_rejected_without_group_before_import(self):
        self.assertIs(inspect.signature(probe.setup).parameters['capture_integer_routes'].default, False)
        with self.assertRaisesRegex(RuntimeError, 'READONLY_GROUP_CENSUS'):
            probe.setup(object(), [], '0' * 64, capture_integer_routes=True)
        with patch.object(sys, 'argv', ['probe', '--build-sha256', '0' * 64, '--capture-integer-routes']):
            with self.assertRaises(SystemExit) as caught:
                probe.main()
        self.assertEqual(caught.exception.code, 2)


if __name__ == '__main__':
    unittest.main()
