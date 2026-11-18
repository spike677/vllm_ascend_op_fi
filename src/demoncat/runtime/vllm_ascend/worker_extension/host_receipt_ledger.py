"""Process-local receipt index. It reads no files and grants no Worker authority."""

import hashlib
import json
import uuid


CONTROL_BYTES = 128 * 1024
RECORD_BYTES = 4 * 1024**2
BATCH_BYTES = 8 * 1024**2
BATCH_RECORDS = 4
STREAM_CHUNK_BYTES = 64 * 1024
EXPORT_BYTES = 128 * 1024**2
CURSOR_SCHEMA = 'demoncat.worker_core.receipt_cursor.v1'
_ENCODER = json.JSONEncoder(allow_nan=False, sort_keys=True, separators=(',', ':'))


class ReceiptLedgerError(ValueError):
    def __init__(self, code, **details):
        super().__init__(code)
        self.code = code
        self.details = details


def encoded_parts(value):
    for part in _ENCODER.iterencode(value):
        raw = part.encode('utf-8')
        for offset in range(0, len(raw), STREAM_CHUNK_BYTES):
            yield raw[offset:offset + STREAM_CHUNK_BYTES]


def encoded_size(value, maximum=None):
    size = 0
    for part in encoded_parts(value):
        size += len(part)
        if maximum is not None and size > maximum:
            return size
    return size


def encoded(value):
    return b''.join(encoded_parts(value))


def _need(ok, code, **details):
    if not ok:
        raise ReceiptLedgerError(code, **details)


class ReceiptLedger:
    """Append only from an actual Host record; never import an external ledger."""

    def __init__(self, owner, tenant):
        self.owner, self.tenant = owner, tenant
        self.ledger_id = uuid.uuid4().hex
        seed = dict(schema=CURSOR_SCHEMA, ledger_id=self.ledger_id,
                    owner=owner, tenant_id=tenant)
        self._prefixes = [hashlib.sha256(encoded(seed)).hexdigest()]
        self._records = []
        self.first_error = None

    def note_error(self, code, **details):
        if self.first_error is None:
            self.first_error = dict(code=code, **details)

    def append(self, record):
        raw = encoded(record)
        digest = hashlib.sha256(raw).hexdigest()
        sequence = len(self._records)
        link = dict(sequence=sequence, previous_sha256=self._prefixes[-1],
                    record_sha256=digest, bytes=len(raw))
        self._records.append((raw, digest))
        self._prefixes.append(hashlib.sha256(encoded(link)).hexdigest())
        if len(raw) > RECORD_BYTES:
            self.note_error('RECEIPT_RECORD_BATCH_BOUND', sequence=sequence, bytes=len(raw))

    def check_scope(self, owner, tenant, history):
        _need((owner, tenant) == (self.owner, self.tenant), 'RECEIPT_LEDGER_OWNER_CHANGED')
        _need(len(history) == len(self._records), 'RECEIPT_INTERNAL_HISTORY_CHANGED')

    def cursor(self, sequence=None):
        sequence = len(self._records) if sequence is None else sequence
        _need(type(sequence) is int and 0 <= sequence <= len(self._records),
              'RECEIPT_CURSOR_SEQUENCE_INVALID')
        return dict(schema=CURSOR_SCHEMA, ledger_id=self.ledger_id, owner=self.owner,
                    tenant_id=self.tenant, next_sequence=sequence,
                    prefix_sha256=self._prefixes[sequence])

    def validate(self, cursor):
        _need(type(cursor) is dict, 'RECEIPT_CURSOR_INVALID')
        _need(all(type(cursor.get(key)) is str for key in
                  ('schema', 'ledger_id', 'owner', 'tenant_id', 'prefix_sha256')),
              'RECEIPT_CURSOR_INVALID')
        sequence = cursor.get('next_sequence')
        _need(type(sequence) is int and 0 <= sequence <= len(self._records),
              'RECEIPT_CURSOR_SEQUENCE_INVALID')
        _need(cursor == self.cursor(sequence), 'RECEIPT_CURSOR_SCOPE_OR_PREFIX_DIFFERS')
        return sequence

    def metadata(self, sequence):
        raw, digest = self._records[sequence]
        return dict(sequence=sequence, bytes=len(raw), sha256=digest,
                    previous_prefix_sha256=self._prefixes[sequence],
                    prefix_sha256=self._prefixes[sequence + 1])

    def last(self):
        return self.metadata(len(self._records) - 1) if self._records else None

    def check_record(self, sequence, history):
        _need(encoded(history[sequence]) == self._records[sequence][0],
              'RECEIPT_INTERNAL_RECORD_CHANGED', sequence=sequence)

    def batch(self, cursor, history, limit):
        start = self.validate(cursor)
        _need(type(limit) is int and 1 <= limit <= BATCH_RECORDS, 'RECEIPT_BATCH_COUNT_BOUND')
        result = dict(schema='demoncat.worker_core.receipt_batch.v1',
                      start_cursor=self.cursor(start), next_cursor=self.cursor(start),
                      records=[], record_count_snapshot=len(self._records),
                      at_tail=start == len(self._records), qualification_granted=False)
        for sequence in range(start, min(start + limit, len(self._records))):
            raw = self._records[sequence][0]
            if len(raw) > RECORD_BYTES:
                if result['records']:
                    break
                raise ReceiptLedgerError('RECEIPT_RECORD_BATCH_BOUND', **self.metadata(sequence))
            self.check_record(sequence, history)
            row = dict(self.metadata(sequence), record=json.loads(raw))
            previous = result['next_cursor']
            result['records'].append(row)
            result['next_cursor'] = self.cursor(sequence + 1)
            result['at_tail'] = sequence + 1 == len(self._records)
            if encoded_size(result, BATCH_BYTES) > BATCH_BYTES:
                result['records'].pop()
                result['next_cursor'] = previous
                result['at_tail'] = False
                _need(bool(result['records']), 'RECEIPT_BATCH_BYTE_BOUND')
                break
        _need(encoded_size(result, BATCH_BYTES) <= BATCH_BYTES, 'RECEIPT_BATCH_BYTE_BOUND')
        return result

    def control(self, fields, *, record_error=False):
        selected = ('status', 'config', 'mechanism', 'worker', 'owner', 'tenant_id',
                    'attachment', 'ticket', 'census_ticket', 'ownership_retained',
                    'provider_check', 'assessments', 'inner_framework_integration_verified')
        value = {key: fields[key] for key in selected}
        value.update(schema='demoncat.worker_core.cursor_control.v1', cursor=self.cursor(),
                     last_record=self.last(), history_included=False, rpc_response_included=False,
                     control_complete=True, first_evidence_error=self.first_error,
                     response_header={key: fields['response'][key] for key in
                                      ('status', 'action', 'phase', 'recovery_required', 'reason', 'error_type')
                                      if key in fields['response']})
        if encoded_size(value, CONTROL_BYTES) > CONTROL_BYTES:
            error = dict(code='RECEIPT_CONTROL_BYTE_BOUND')
            if record_error:
                self.note_error(error['code'])
            # Do not truncate an identity/ticket and present it as usable state.
            status = fields['status']
            value = dict(schema='demoncat.worker_core.cursor_control.v1',
                         status=status if type(status) is str and len(status) <= 128 else 'CONTROL_BOUND_EXCEEDED',
                         cursor=self.cursor(), last_record=self.last(), owner=self.owner,
                         tenant_id=self.tenant, ownership_retained=fields['ownership_retained'],
                         history_included=False, rpc_response_included=False, control_complete=False,
                         omitted_fields=list(selected) + ['response_header'],
                         first_evidence_error=self.first_error or error)
        _need(encoded_size(value, CONTROL_BYTES) <= CONTROL_BYTES, 'RECEIPT_CONTROL_BYTE_BOUND')
        return json.loads(encoded(value))

    def record_parts(self, cursor, history):
        sequence = self.validate(cursor)
        _need(sequence < len(self._records), 'RECEIPT_CURSOR_AT_TAIL')
        self.check_record(sequence, history)
        raw, digest = self._records[sequence]
        return (raw[offset:offset + STREAM_CHUNK_BYTES] for offset in range(0, len(raw), STREAM_CHUNK_BYTES)), \
            len(raw), dict(kind='original_rpc', start_cursor=self.cursor(sequence),
                           end_cursor=self.cursor(sequence + 1), expected_sha256=digest)

    def snapshot_parts(self, cursor, fields, history):
        _need(self.validate(cursor) == len(self._records), 'RECEIPT_CURRENT_TAIL_REQUIRED')
        for sequence in range(len(self._records)):
            self.check_record(sequence, history)
        header = {key: encoded(value) for key, value in fields.items() if key != 'history'}
        keys = sorted([*header, 'history'])
        size = 2 + max(0, len(keys) - 1)
        for key in keys:
            size += len(encoded(key)) + 1
            size += (2 + max(0, len(self._records) - 1) + sum(len(row[0]) for row in self._records)
                     if key == 'history' else len(header[key]))

        def parts():
            yield b'{'
            for ordinal, key in enumerate(keys):
                if ordinal:
                    yield b','
                yield encoded(key) + b':'
                if key == 'history':
                    yield b'['
                    for index, (raw, _) in enumerate(self._records):
                        if index:
                            yield b','
                        for offset in range(0, len(raw), STREAM_CHUNK_BYTES):
                            yield raw[offset:offset + STREAM_CHUNK_BYTES]
                    yield b']'
                else:
                    raw = header[key]
                    for offset in range(0, len(raw), STREAM_CHUNK_BYTES):
                        yield raw[offset:offset + STREAM_CHUNK_BYTES]
            yield b'}'
        return parts(), size, dict(kind='legacy_host_receipt_v1', end_cursor=self.cursor())

    def write(self, sink, parts, expected_bytes, metadata):
        confirmed = 0
        checksum = hashlib.sha256()
        pending = 0
        try:
            _need(type(expected_bytes) is int and 0 <= expected_bytes <= EXPORT_BYTES,
                  'RECEIPT_EXPORT_BYTE_BOUND', expected_bytes=expected_bytes)
            _need(sink.seekable() and sink.tell() == 0, 'RECEIPT_FRESH_BINARY_SINK_REQUIRED')
            end = sink.seek(0, 2)
            sink.seek(0)
            _need(end == 0, 'RECEIPT_FRESH_BINARY_SINK_REQUIRED')
            for part in parts:
                _need(type(part) is bytes and len(part) <= STREAM_CHUNK_BYTES,
                      'RECEIPT_STREAM_CHUNK_BOUND')
                _need(confirmed + len(part) <= expected_bytes, 'RECEIPT_EXPORT_SIZE_CHANGED')
                while part:
                    pending = len(part)
                    written = sink.write(part)
                    _need(type(written) is int and 0 < written <= len(part), 'RECEIPT_STREAM_SHORT_WRITE')
                    checksum.update(part[:written])
                    confirmed += written
                    part = part[written:]
                    pending = 0
            _need(confirmed == expected_bytes, 'RECEIPT_EXPORT_SIZE_CHANGED')
            digest = checksum.hexdigest()
            _need(metadata.get('expected_sha256', digest) == digest, 'RECEIPT_EXPORT_HASH_CHANGED')
            sink.flush()
            return dict(schema='demoncat.worker_core.receipt_export.v1', complete=True,
                        bytes=confirmed, sha256=digest, **metadata,
                        sink_readback_verified=False, qualification_granted=False)
        except BaseException as error:
            code = error.code if isinstance(error, ReceiptLedgerError) else 'RECEIPT_STREAM_IO_FAILED'
            self.note_error(code, error_type=type(error).__name__)
            details = dict(complete=False, confirmed_bytes=confirmed, pending_chunk_bytes=pending,
                           prefix_sha256=checksum.hexdigest(), expected_bytes=expected_bytes,
                           error_type=type(error).__name__, internal_originals_retained=True)
            if not isinstance(error, Exception):
                raise
            raise ReceiptLedgerError(code, **details) from error
