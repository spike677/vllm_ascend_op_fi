"""Bounded receipt transport over real CPU Host/Runtime lifecycle contracts."""

import hashlib
import io
import json
import unittest
from unittest.mock import Mock, patch

from demoncat.runtime.vllm_ascend.worker_extension import host_receipt_ledger as L
from demoncat.runtime.vllm_ascend.worker_extension.config import FaultConfig
from demoncat.runtime.vllm_ascend.worker_extension.host import (
    HostAdapter, HostContractError, WorkerFaultClient,
)

try:
    from . import test_core_host as fixtures
    from . import test_core_host_census as census_fixtures
except ImportError:
    import test_core_host as fixtures
    import test_core_host_census as census_fixtures


class TrackingSink(io.BytesIO):
    def __init__(self, *, short=None, fail_after=None, fail_flush=False, callback=None):
        super().__init__()
        self.calls = []
        self.short, self.fail_after = short, fail_after
        self.fail_flush, self.callback = fail_flush, callback

    def write(self, data):
        self.calls.append(len(data))
        if self.callback is not None:
            callback, self.callback = self.callback, None
            callback()
        if self.fail_after is not None and self.tell() >= self.fail_after:
            raise OSError('fixture sink failure')
        return super().write(data[:self.short] if self.short else data)

    def flush(self):
        if self.fail_flush:
            raise OSError('fixture flush failure')
        return super().flush()


class HostReceiptCursorTests(unittest.TestCase):
    setUp = fixtures.CoreHostTests.setUp
    _resolve = fixtures.CoreHostTests._resolve
    checked = fixtures.CoreHostTests.checked
    actions = fixtures.CoreHostTests.actions
    fail_once_after = fixtures.CoreHostTests.fail_once_after

    def client(self, *, receipt_mode='cursor', **host_options):
        return WorkerFaultClient(HostAdapter(self.executor, **host_options),
                                 FaultConfig(20, 'buffer_substitute'),
                                 receipt_mode=receipt_mode)

    def test_default_full_has_exact_legacy_fields_and_returns_isolated_history(self):
        client = WorkerFaultClient(HostAdapter(self.executor), FaultConfig(20, 'buffer_substitute'))
        row = self.checked(client)
        self.assertIsNone(client._receipt_ledger)
        self.assertEqual(set(row), {
            'schema', 'status', 'config', 'mechanism', 'worker', 'owner', 'tenant_id',
            'attachment', 'ticket', 'census_ticket', 'ownership_retained',
            'capability_enumeration', 'provider_check', 'assessments', 'response',
            'history', 'inner_framework_integration_verified',
        })
        self.assertEqual(row['schema'], 'demoncat.worker_core.host_receipt.v1')
        self.assertEqual(row, client.snapshot())
        row['history'][1]['response']['ticket']['generation'] = -1
        self.assertGreater(client.snapshot()['ticket']['generation'], 0)
        with self.assertRaisesRegex(L.ReceiptLedgerError, 'CURSOR_MODE_REQUIRED'):
            client.receipt_cursor()
        self.assertEqual(client.detach()['status'], 'DETACHED')

    def test_explicit_full_retains_the_same_legacy_schema(self):
        client = self.client(receipt_mode='full')
        self.assertEqual(client.prepare(), client.snapshot())
        self.assertIsNone(client._receipt_ledger)
        self.assertEqual(client.detach()['status'], 'DETACHED')
        for value in (None, False, '', 'delta', {}):
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.client(receipt_mode=value)

    def test_cursor_controls_never_materialize_a_full_history_snapshot(self):
        client = self.client()
        start = client.receipt_cursor()
        with patch.object(client, '_snapshot', side_effect=AssertionError('full history cloned')):
            for action in ('prepare', 'register', 'check', 'enable', 'query', 'finish',
                           'disable', 'unregister', 'detach'):
                result = getattr(client, action)()
                self.assertLessEqual(len(L.encoded(result)), L.CONTROL_BYTES)
                self.assertEqual(result['schema'], 'demoncat.worker_core.cursor_control.v1')
                self.assertFalse(result['history_included'])
                self.assertFalse(result['rpc_response_included'])
                self.assertNotIn('history', result)
                self.assertNotIn('response', result)
                self.assertTrue(result['control_complete'])
                before = len(self.executor.calls)
                self.assertEqual(client.status_receipt(), result)
                self.assertEqual(len(self.executor.calls), before)
        self.assertEqual(start['next_sequence'], 0)
        self.assertEqual(client.receipt_cursor()['next_sequence'], 10)  # implicit attach
        self.assertFalse(client.status_receipt()['ownership_retained'])
        self.assertEqual(client.status_receipt()['assessments']['C2'], 'NOT_ASSESSED')

    def test_prepare_batch_includes_implicit_attach_and_keeps_exact_runtime_ticket(self):
        client = self.client()
        start = client.receipt_cursor()
        row = client.prepare()
        batch = client.receipt_batch(start)
        self.assertEqual([entry['record']['action'] for entry in batch['records']], ['attach', 'prepare'])
        self.assertEqual([entry['record'] for entry in batch['records']], client.snapshot()['history'])
        self.assertEqual(batch['next_cursor'], row['cursor'])
        self.assertEqual(row['ticket'], client.snapshot()['response']['ticket'])
        self.assertTrue(batch['at_tail'])
        self.assertEqual(client.receipt_batch(batch['next_cursor'])['records'], [])
        self.assertEqual(client.detach()['status'], 'DETACHED')

    def test_cursor_scope_sequence_prefix_and_unknown_fields_reject_without_rpc(self):
        client, other = self.client(), self.client()
        start = client.receipt_cursor()
        client.prepare()
        before, state = len(self.executor.calls), client.snapshot()
        variants = [other.receipt_cursor(), None, [], dict(start, extra=True)]
        variants.extend(dict(start, **{key: value}) for key, value in (
            ('ledger_id', 'foreign'), ('owner', 'foreign'), ('tenant_id', 'foreign'),
            ('schema', 'foreign'), ('next_sequence', -1), ('next_sequence', 99),
            ('next_sequence', True), ('next_sequence', 1), ('prefix_sha256', '0' * 64),
        ))
        for value in variants:
            with self.subTest(cursor=value), self.assertRaises(L.ReceiptLedgerError):
                client.receipt_batch(value)
        for limit in (0, 5, True, None):
            with self.subTest(limit=limit), self.assertRaisesRegex(L.ReceiptLedgerError, 'BATCH_COUNT_BOUND'):
                client.receipt_batch(start, limit=limit)
        self.assertEqual(len(self.executor.calls), before)
        self.assertEqual(client.snapshot(), state)
        self.assertEqual(client.detach()['status'], 'DETACHED')

    def test_reprepare_still_refuses_and_old_cursor_cannot_migrate_to_new_client(self):
        client = self.client()
        start = client.receipt_cursor()
        client.prepare()
        refused = client.prepare()
        self.assertEqual(refused['response_header']['reason'], 'NEW_CLIENT_REQUIRED_AFTER_PREPARE_ATTEMPT')
        self.assertEqual(self.actions(), ['prepare'])
        client.detach()
        replacement = self.client()
        with self.assertRaisesRegex(L.ReceiptLedgerError, 'SCOPE_OR_PREFIX_DIFFERS'):
            replacement.receipt_batch(start)
        # A completed client retains its own read-only originals.
        self.assertEqual(client.receipt_batch(start)['records'][0]['record']['action'], 'attach')

    def test_control_batch_cursor_mutations_cannot_pollute_cache_or_live_state(self):
        client = self.client()
        start = client.receipt_cursor()
        row = client.prepare()
        expected = client.snapshot()
        batch = client.receipt_batch(start)
        row['ticket']['generation'] = -1
        row['worker']['rank'] = -1
        row['cursor']['owner'] = 'foreign'
        batch['records'][1]['record']['response']['ticket']['generation'] = -1
        batch['records'][0]['sha256'] = '0' * 64
        batch['next_cursor']['prefix_sha256'] = '0' * 64
        start['owner'] = 'foreign'
        self.assertEqual(client.snapshot(), expected)
        fresh_start = client._receipt_ledger.cursor(0)
        self.assertEqual([row['record'] for row in client.receipt_batch(fresh_start)['records']],
                         expected['history'])
        self.assertEqual(client.detach()['status'], 'DETACHED')

    def test_control_bound_preserves_full_binding_failure_and_allows_real_detach(self):
        reason = 'binding-refused-' + 'x' * L.CONTROL_BYTES
        binding = Mock(side_effect=HostContractError(reason))
        client = self.client(prepare_binding=binding)
        result = client.prepare()
        self.assertLessEqual(len(L.encoded(result)), L.CONTROL_BYTES)
        self.assertFalse(result['control_complete'])
        self.assertNotIn('ticket', result)
        self.assertTrue(result['ownership_retained'])
        self.assertEqual(result['first_evidence_error']['code'], 'RECEIPT_CONTROL_BYTE_BOUND')
        self.assertEqual(client.snapshot()['response']['reason'], reason)
        self.assertEqual(client.snapshot()['assessments']['Recovery']['status'], 'REQUIRED')
        self.assertEqual(self.actions(), [])
        self.assertEqual(client.detach()['status'], 'DETACHED')
        self.assertEqual(client.status_receipt()['first_evidence_error'], result['first_evidence_error'])

    def test_legacy_and_individual_streams_equal_original_snapshot_with_bounded_writes(self):
        client = self.client()
        start = client.receipt_cursor()
        self.checked(client)
        expected = client.snapshot()
        sink = TrackingSink(short=17)
        with patch.object(client, '_snapshot', side_effect=AssertionError('full history cloned')):
            export = client.stream_snapshot(sink, cursor=client.receipt_cursor())
        self.assertTrue(export['complete'])
        self.assertFalse(export['sink_readback_verified'])
        self.assertEqual(json.loads(sink.getvalue()), expected)
        self.assertEqual(export['sha256'], hashlib.sha256(sink.getvalue()).hexdigest())
        self.assertEqual(export['bytes'], len(sink.getvalue()))
        self.assertLessEqual(max(sink.calls), L.STREAM_CHUNK_BYTES)
        raw = TrackingSink()
        row_export = client.stream_receipt(start, raw)
        self.assertEqual(json.loads(raw.getvalue()), expected['history'][0])
        self.assertEqual(row_export['end_cursor']['next_sequence'], 1)
        self.assertEqual(row_export['sha256'], hashlib.sha256(raw.getvalue()).hexdigest())
        self.assertEqual(client.snapshot(), expected)
        client.detach()

    def test_stream_requires_current_tail_for_snapshot_but_allows_historical_record(self):
        client = self.client()
        start = client.receipt_cursor()
        client.prepare()
        sink = TrackingSink()
        with self.assertRaisesRegex(L.ReceiptLedgerError, 'CURRENT_TAIL_REQUIRED'):
            client.stream_snapshot(sink, cursor=start)
        self.assertEqual(sink.calls, [])
        self.assertTrue(client.stream_receipt(start, sink)['complete'])
        with self.assertRaisesRegex(L.ReceiptLedgerError, 'CURSOR_AT_TAIL'):
            client.stream_receipt(client.receipt_cursor(), TrackingSink())
        client.detach()

    def test_record_overflow_and_export_overflow_do_not_change_real_recovery(self):
        client = self.client()
        checked = self.checked(client)
        provider = self.providers[(self.worker, client.config.mechanism)]
        provider.state['evidence_blob'] = 'x' * 8192
        before_query = client.receipt_cursor()
        with patch.object(L, 'RECORD_BYTES', 4096):
            query = client.query()
            self.assertEqual(query['status'], 'QUERY_OK')
            self.assertEqual(query['ticket'], checked['ticket'])
            self.assertEqual(query['first_evidence_error']['code'], 'RECEIPT_RECORD_BATCH_BOUND')
            with self.assertRaisesRegex(L.ReceiptLedgerError, 'RECORD_BATCH_BOUND'):
                client.receipt_batch(before_query)
        self.fail_once_after('enable')
        uncertain = client.enable()
        self.assertEqual(uncertain['status'], 'RPC_UNCERTAIN')
        self.assertEqual(uncertain['ticket'], checked['ticket'])
        self.assertEqual(client.enable()['response_header']['reason'], 'RECOVERY_REQUIRED')
        snapshot = client.snapshot()
        sink = TrackingSink()
        with patch.object(L, 'EXPORT_BYTES', 1024), self.assertRaisesRegex(
                L.ReceiptLedgerError, 'EXPORT_BYTE_BOUND') as caught:
            client.stream_snapshot(sink, cursor=client.receipt_cursor())
        self.assertFalse(caught.exception.details['complete'])
        self.assertEqual(sink.calls, [])
        self.assertEqual(client.snapshot(), snapshot)
        self.assertTrue(client.status_receipt()['ownership_retained'])
        provider.callbacks['recover'] = Mock(side_effect=RuntimeError('drain failed first'))
        self.assertEqual(client.recover()['status'], 'PROVIDER_ACTION_FAILED')
        self.assertEqual(client.status_receipt()['ticket'], checked['ticket'])
        provider.callbacks.clear()
        self.assertEqual(client.recover()['status'], 'ACTION_COMPLETED')
        self.assertEqual(client.unregister()['status'], 'ACTION_COMPLETED')
        self.assertEqual(client.detach()['status'], 'DETACHED')
        self.assertEqual(self.actions().count('enable'), 1)
        self.assertFalse(client.status_receipt()['ownership_retained'])
        self.assertEqual(client.status_receipt()['first_evidence_error'], query['first_evidence_error'])
        history = client.snapshot()['history']
        self.assertTrue(any(row['response']['status'] == 'RPC_UNCERTAIN' for row in history))
        self.assertTrue(any(row['response']['status'] == 'PROVIDER_ACTION_FAILED' for row in history))
        final = TrackingSink()
        client.stream_snapshot(final, cursor=client.receipt_cursor())
        self.assertEqual(json.loads(final.getvalue()), client.snapshot())

    def test_uncertain_prepare_has_no_ticket_and_cursor_cannot_replace_detach(self):
        client = self.client()
        start = client.receipt_cursor()
        self.fail_once_after('prepare')
        result = client.prepare()
        self.assertEqual(result['status'], 'RPC_UNCERTAIN')
        self.assertIsNone(result['ticket'])
        self.assertTrue(result['ownership_retained'])
        self.assertEqual(client.query()['response_header']['reason'],
                         'LEASE_TICKET_REQUIRED_USE_DETACH_FOR_UNCERTAIN_PREPARE')
        self.assertEqual(client.detach()['status'], 'DETACHED')
        self.assertEqual(self.actions(), ['prepare', 'unregister'])
        self.assertEqual(client.receipt_batch(start)['records'][1]['record']['response']['status'],
                         'RPC_UNCERTAIN')

    def test_cursor_census_failed_cleanup_retains_owner_until_real_detach_retry(self):
        from demoncat.runtime.vllm_ascend.worker_extension import call_census

        client = self.client()
        start = client.receipt_cursor()
        captures = []

        def factory(*args, **kwargs):
            captures.append(census_fixtures.Readback(*args, **kwargs))
            return captures[-1]

        with patch.object(call_census, 'WorkerCallCensus', side_effect=factory):
            control = client.census_start(phase='prefill', expected={})
            self.assertEqual(control['status'], 'CENSUS_SNAPSHOT')
            census_ticket = control['census_ticket']
            captures[0].fail_cleanup = True
            failed = client.census_cleanup()
            self.assertEqual(failed['status'], 'CENSUS_RECOVERY_REQUIRED')
            self.assertEqual(failed['census_ticket'], census_ticket)
            self.assertTrue(failed['ownership_retained'])
            self.assertEqual(client.detach()['status'], 'RECOVERY_REQUIRED')
            captures[0].fail_cleanup = False
            self.assertEqual(client.detach()['status'], 'DETACHED')
        self.assertFalse(client.status_receipt()['ownership_retained'])
        self.assertIsNone(client.status_receipt()['census_ticket'])
        batch = client.receipt_batch(start)
        self.assertEqual([row['record'] for row in batch['records']], client.snapshot()['history'][:4])
        self.assertEqual(client.snapshot()['history'][2]['response']['status'], 'CENSUS_RECOVERY_REQUIRED')

    def test_stream_io_failure_preserves_originals_and_resets_guard_for_real_recovery(self):
        client = self.client()
        checked = self.checked(client)
        before = client.receipt_cursor()
        sink = TrackingSink(fail_after=1)
        with self.assertRaisesRegex(L.ReceiptLedgerError, 'STREAM_IO_FAILED') as caught:
            client.stream_snapshot(sink, cursor=before)
        self.assertFalse(caught.exception.details['complete'])
        self.assertGreater(caught.exception.details['confirmed_bytes'], 0)
        self.assertTrue(caught.exception.details['internal_originals_retained'])
        self.assertEqual(client.receipt_cursor(), before)
        self.assertEqual(client.status_receipt()['ticket'], checked['ticket'])
        self.assertEqual(client.recover()['status'], 'ACTION_COMPLETED')
        self.assertEqual(client.unregister()['status'], 'ACTION_COMPLETED')
        self.assertEqual(client.detach()['status'], 'DETACHED')
        fresh = TrackingSink()
        self.assertTrue(client.stream_snapshot(fresh, cursor=client.receipt_cursor())['complete'])
        self.assertEqual(json.loads(fresh.getvalue()), client.snapshot())

    def test_reentrant_sink_cannot_send_rpc_or_mutate_receipt_during_export(self):
        client = self.client()
        self.checked(client)
        before, calls = client.receipt_cursor(), len(self.executor.calls)
        for action in ('query', 'recover', 'prepare', 'detach', 'discover'):
            with self.subTest(action=action):
                sink = TrackingSink(callback=getattr(client, action))
                with self.assertRaisesRegex(L.ReceiptLedgerError, 'EXPORT_IN_PROGRESS'):
                    client.stream_snapshot(sink, cursor=before)
                self.assertEqual(client.receipt_cursor(), before)
                self.assertEqual(len(self.executor.calls), calls)
        self.assertEqual(client.detach()['status'], 'DETACHED')


class ReceiptLedgerBoundsTests(unittest.TestCase):
    def ledger(self, rows):
        ledger = L.ReceiptLedger('owner-a', 'tenant-a')
        for row in rows:
            ledger.append(row)
        return ledger

    def record_of_size(self, size):
        row = {'action': 'query', 'response': {'status': 'QUERY_OK', 'evidence': ''}}
        row['response']['evidence'] = 'x' * (size - len(L.encoded(row)))
        self.assertEqual(len(L.encoded(row)), size)
        return row

    def test_real_four_mib_record_and_eight_mib_envelope_are_distinct_bounds(self):
        self.assertEqual((L.CONTROL_BYTES, L.RECORD_BYTES, L.BATCH_BYTES, L.BATCH_RECORDS,
                          L.STREAM_CHUNK_BYTES, L.EXPORT_BYTES),
                         (128 * 1024, 4 * 1024**2, 8 * 1024**2, 4,
                          64 * 1024, 128 * 1024**2))
        rows = [self.record_of_size(L.RECORD_BYTES), self.record_of_size(L.RECORD_BYTES)]
        ledger = self.ledger(rows)
        first = ledger.batch(ledger.cursor(0), rows, 4)
        self.assertEqual(len(first['records']), 1)  # Two records plus envelope exceed 8 MiB.
        self.assertLessEqual(len(L.encoded(first)), L.BATCH_BYTES)
        second = ledger.batch(first['next_cursor'], rows, 4)
        self.assertTrue(second['at_tail'])
        self.assertEqual(second['records'][0]['bytes'], L.RECORD_BYTES)
        rows.append(self.record_of_size(L.RECORD_BYTES + 1))
        ledger.append(rows[-1])
        with self.assertRaisesRegex(L.ReceiptLedgerError, 'RECORD_BATCH_BOUND'):
            ledger.batch(ledger.cursor(2), rows, 1)
        # Oversized for batch does not discard the original or prevent bounded streaming.
        parts, size, metadata = ledger.record_parts(ledger.cursor(2), rows)
        sink = TrackingSink()
        self.assertTrue(ledger.write(sink, parts, size, metadata)['complete'])
        self.assertEqual(json.loads(sink.getvalue()), rows[-1])
        self.assertLessEqual(max(sink.calls), 64 * 1024)

    def test_batch_stops_before_oversized_row_and_never_skips_it(self):
        rows = [self.record_of_size(100), self.record_of_size(1000), self.record_of_size(100)]
        ledger = self.ledger(rows)
        with patch.object(L, 'RECORD_BYTES', 999):
            batch = ledger.batch(ledger.cursor(0), rows, 4)
            self.assertEqual(len(batch['records']), 1)
            self.assertEqual(batch['next_cursor'], ledger.cursor(1))
            with self.assertRaisesRegex(L.ReceiptLedgerError, 'RECORD_BATCH_BOUND'):
                ledger.batch(batch['next_cursor'], rows, 4)

    def test_four_record_limit_and_exact_envelope_boundary(self):
        rows = [self.record_of_size(100)] * 5
        ledger = self.ledger(rows)
        batch = ledger.batch(ledger.cursor(0), rows, 4)
        self.assertEqual(len(batch['records']), 4)
        exact = len(L.encoded(batch))
        with patch.object(L, 'BATCH_BYTES', exact):
            self.assertEqual(len(ledger.batch(ledger.cursor(0), rows, 4)['records']), 4)
        with patch.object(L, 'BATCH_BYTES', exact - 1):
            self.assertEqual(len(ledger.batch(ledger.cursor(0), rows, 4)['records']), 3)

    def test_export_exact_cap_and_cap_plus_one_preflight_no_write(self):
        rows = [self.record_of_size(100)]
        ledger = self.ledger(rows)
        parts, size, metadata = ledger.record_parts(ledger.cursor(0), rows)
        with patch.object(L, 'EXPORT_BYTES', size):
            self.assertTrue(ledger.write(TrackingSink(), parts, size, metadata)['complete'])
        parts, size, metadata = ledger.record_parts(ledger.cursor(0), rows)
        sink = TrackingSink()
        with patch.object(L, 'EXPORT_BYTES', size - 1), self.assertRaisesRegex(
                L.ReceiptLedgerError, 'EXPORT_BYTE_BOUND') as caught:
            ledger.write(sink, parts, size, metadata)
        self.assertEqual(sink.calls, [])
        self.assertFalse(caught.exception.details['complete'])
        self.assertEqual(ledger.batch(ledger.cursor(0), rows, 1)['records'][0]['record'], rows[0])
        with self.assertRaisesRegex(L.ReceiptLedgerError, 'EXPORT_BYTE_BOUND'):
            ledger.write(sink, iter(()), 128 * 1024**2 + 1, {})
        self.assertEqual(sink.calls, [])

    def test_nonempty_sink_flush_failure_and_zero_write_never_claim_complete(self):
        rows = [self.record_of_size(100)]
        ledger = self.ledger(rows)
        nonempty = io.BytesIO(b'preserve me')
        zero = TrackingSink()
        zero.write = Mock(return_value=0)
        for sink, code in ((nonempty, 'FRESH_BINARY_SINK_REQUIRED'),
                           (TrackingSink(fail_flush=True), 'STREAM_IO_FAILED'),
                           (zero, 'STREAM_SHORT_WRITE')):
            with self.subTest(code=code):
                parts, size, metadata = ledger.record_parts(ledger.cursor(0), rows)
                with self.assertRaisesRegex(L.ReceiptLedgerError, code) as caught:
                    ledger.write(sink, parts, size, metadata)
                self.assertFalse(caught.exception.details['complete'])
                self.assertTrue(caught.exception.details['internal_originals_retained'])
        self.assertEqual(nonempty.getvalue(), b'preserve me')
        self.assertEqual(ledger.first_error['code'], 'RECEIPT_FRESH_BINARY_SINK_REQUIRED')

    def test_internal_mutation_and_owner_replacement_are_detected_not_imported(self):
        rows = [self.record_of_size(100)]
        ledger = self.ledger(rows)
        for owner, tenant in (('owner-b', 'tenant-a'), ('owner-a', 'tenant-b')):
            with self.assertRaisesRegex(L.ReceiptLedgerError, 'LEDGER_OWNER_CHANGED'):
                ledger.check_scope(owner, tenant, rows)
        rows[0]['response']['evidence'] = 'tampered'
        with self.assertRaisesRegex(L.ReceiptLedgerError, 'INTERNAL_RECORD_CHANGED'):
            ledger.batch(ledger.cursor(0), rows, 1)
        with self.assertRaisesRegex(L.ReceiptLedgerError, 'INTERNAL_RECORD_CHANGED'):
            ledger.record_parts(ledger.cursor(0), rows)


if __name__ == '__main__':
    unittest.main()
