"""Sequential capture of one immutable original native window per prefill call.

No tensor/operator/strategy computation. Original CANN raw decoding is delegated
unchanged to native_evidence.actual_launch. The cursor detects observed inode
replacement/truncation, but does not rehash consumed bytes on every call. A
same-length rewrite of an already consumed prefix must be detected by the final
independent whole-log acceptance, not claimed detected here.
"""
from __future__ import annotations
import copy
import hashlib
import os
from pathlib import Path
import stat
import tempfile

import native_evidence


IDENTITY_FIELDS = ('call_id', 'case_id', 'api', 'pid', 'thread_id', 'scope_token',
                   'phase', 'batch_size', 'step', 'mode', 'layer_index', 'model_stage', 'run_index')
PROCESS_FIELDS = ('case_id', 'api', 'pid', 'thread_id', 'batch_size')


def need(ok, message):
    if not ok:
        raise ValueError(message)


class NativeLogCursor:
    """Single process, single original log, monotonically ordered SDPA windows."""
    def __init__(self, log_path):
        self.path = Path(log_path)
        need(self.path.is_file() and not self.path.is_symlink(), 'Original native log must be a regular non-symlink file')
        self.resolved = self.path.resolve(strict=True)
        self.stream = self.path.open('rb')
        info = os.fstat(self.stream.fileno())
        need(stat.S_ISREG(info.st_mode), 'Original native log is not a regular file')
        self.identity = (info.st_dev, info.st_ino)
        self.maximum_observed_size = info.st_size
        self.offset, self.line, self.prefix = 0, 0, hashlib.sha256()
        self.loads, self.seen_ids = [], set()
        self.last_position = None
        self.process_identity = None
        self.failed, self.closed = False, False
        self.bytes_consumed = 0
        self.completed_prefill_windows = 0

    def close(self):
        if not self.closed:
            self.stream.close()
            self.closed = True

    def receipt(self):
        return {'schema': 'nes.model_native_stream.receipt.v1',
            'mode': 'sequential_original_log_prefix', 'source_log': str(self.path),
            'reader_source_sha256': native_evidence.sha(Path(__file__)),
            'raw_parser_source_sha256': native_evidence.sha(Path(native_evidence.__file__)),
            'opened_file_identity': {'device': self.identity[0], 'inode': self.identity[1]},
            'bytes_consumed': self.bytes_consumed, 'prefix_bytes': self.offset,
            'prefix_lines': self.line, 'prefix_sha256': self.prefix.hexdigest(),
            'completed_prefill_windows': self.completed_prefill_windows,
            'closed': self.closed, 'failed': self.failed,
            'full_log_rehashed': False, 'independent_acceptance_required': True,
            'scope': 'Through last captured prefill exit; intervening decode is consumed before the next prefill. Trailing decode requires the independent whole-log census.'}

    def _check_file(self):
        need(not self.closed and not self.failed, 'Native log cursor is closed or failed; cannot resume it')
        path_info, opened = self.path.stat(), os.fstat(self.stream.fileno())
        need(not self.path.is_symlink() and self.path.resolve(strict=True) == self.resolved
             and (path_info.st_dev, path_info.st_ino) == self.identity
             and (opened.st_dev, opened.st_ino) == self.identity, 'Original log inode/path was replaced')
        need(path_info.st_size >= self.maximum_observed_size and path_info.st_size >= self.offset,
             'Original native log was truncated')
        self.maximum_observed_size = path_info.st_size

    def _identity(self, event):
        need(all(key in event for key in IDENTITY_FIELDS), 'Incomplete native marker identity')
        need(type(event['call_id']) is str and bool(event['call_id'])
             and event['api'] == 'torch.nn.functional.scaled_dot_product_attention'
             and event['phase'] == 'operator_execution', 'Wrong original SDPA marker API/phase')
        need(all(type(event[x]) is int and event[x] > 0 for x in ('pid', 'thread_id', 'batch_size'))
             and type(event['step']) is int and event['step'] >= 0
             and type(event['layer_index']) is int and 0 <= event['layer_index'] < 28
             and type(event['run_index']) is int and event['run_index'] == event['step'] // 2,
             'Invalid native marker process/step/layer identity')
        expected_stage = 'prefill' if event['step'] % 2 == 0 else 'decode'
        need(event['model_stage'] == expected_stage, 'Native marker stage differs from step')
        current = tuple(event[x] for x in PROCESS_FIELDS)
        if self.process_identity is None:
            self.process_identity = current
        need(current == self.process_identity, 'Cross-case/PID/TID native markers in one cursor')
        return {key: event[key] for key in IDENTITY_FIELDS}

    def actual_launch(self, log_path, call, output):
        """Read only unseen log lines; preserve the original actual_launch schema."""
        output = Path(output)
        window = None
        try:
            self._check_file()
            need(Path(log_path).resolve(strict=True) == self.resolved, 'Cursor was used for a different original log')
            wanted = self._identity(call)
            need(call['model_stage'] == 'prefill' and call['scope_token'] == call['call_id'],
                 'Only individually scoped prefill calls request native raw capture')
            need(call['call_id'] not in self.seen_ids, 'Repeated native call requested')
            position = (call['step'], call['layer_index'])
            need(self.last_position is None or position > self.last_position, 'Out-of-order native call requested')
            need(not output.exists(), 'Never overwrite an original native window')
            active, selected = None, False
            start_offset = start_line = None
            while True:
                payload = self.stream.readline()
                need(payload and payload.endswith(b'\n'), 'Native exit marker missing or incomplete log line')
                before = self.offset
                self.offset += len(payload); self.bytes_consumed += len(payload); self.line += 1
                self.prefix.update(payload)
                was_selected = selected
                if selected:
                    window.write(payload)
                text = payload.decode('utf-8', 'surrogateescape').rstrip('\r\n')
                if 'LoadOppPackage' in text and 'load so, so_name is ' in text:
                    self.loads.append({'line': self.line,
                        'path': text.split('load so, so_name is ', 1)[1].strip().replace('//', '/'),
                        'line_sha256': hashlib.sha256(payload).hexdigest()})
                event = native_evidence.marker(text)
                need(event is not None or not any('"' + name + '"' in text for name in ('native_api_enter', 'native_api_exit')),
                     'Malformed/interleaved native call marker')
                if event is not None:
                    identity = self._identity(event)
                    if event['event'] == 'native_api_enter':
                        need(active is None, 'Cross-call or nested native entry before prior exit')
                        event_position = (event['step'], event['layer_index'])
                        need(event['call_id'] not in self.seen_ids
                             and (self.last_position is None or event_position > self.last_position),
                             'Repeated/out-of-order original native marker')
                        self.seen_ids.add(event['call_id'])
                        active = identity
                        selected = event['call_id'] == call['call_id']
                        if selected:
                            need(identity == wanted, 'Selected native marker PID/TID/token/context differs')
                            output.parent.mkdir(parents=True, exist_ok=True)
                            window = output.open('xb')
                            start_offset, start_line = before, self.line
                        else:
                            need(event['model_stage'] == 'decode' and event['scope_token'] is None
                                 and event_position < position,
                                 'Unrequested prefill or future/cross-call window before selected call')
                    else:
                        need(active is not None and identity == active and event.get('status') == 'returned',
                             'Unmatched native exit or mismatched PID/TID/token/status')
                        self.last_position = (event['step'], event['layer_index'])
                        if selected:
                            window.flush(); window.close(); window = None
                            self._check_file()
                            break
                        active = None
                if selected and not was_selected:
                    window.write(payload)
            # Reuse the unchanged exact-character raw parser on ORIGINAL bytes.
            # A failure preserves output (including partial evidence) and poisons
            # the cursor. The controller's full original log is never modified.
            with tempfile.TemporaryDirectory(prefix='.native_window_parse_', dir=output.parent) as temporary:
                parsed_path = Path(temporary) / 'parsed.log'
                parsed = native_evidence.actual_launch(output, call, parsed_path)
                need(parsed['window_byte_range'] == [0, output.stat().st_size]
                     and parsed['window_line_range'] == [1, self.line - start_line + 1]
                     and native_evidence.sha(parsed_path) == native_evidence.sha(output),
                     'Original window parser did not consume precisely the captured window')
            parsed.update(source_log=str(log_path), source_prefix_sha256=self.prefix.hexdigest(),
                source_prefix_bytes=self.offset, window_byte_range=[start_offset, self.offset],
                window_line_range=[start_line, self.line], window_file=str(output),
                loaded_libraries_before_exit=copy.deepcopy(self.loads))
            self.completed_prefill_windows += 1
            return parsed
        except BaseException:
            self.failed = True
            if window is not None:
                window.flush(); window.close()
            raise
