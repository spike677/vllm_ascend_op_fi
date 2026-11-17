"""Unmodified msprobe capture of actual model API/module boundaries.

Public stop/start pauses exclude the provider's owned diagnostics and proxy
implementation arithmetic. The actual model producer and returned module value
remain captured. No operator filters, synthesized tensors, rewritten records,
collector monkeypatches or tolerance changes are used.
"""
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import threading


REVIEWED_CONTROLS = {
    'core/dump/service.py': 'd51804b819a07453733e52eb81ec132f2cfa9934bfa5b243aaf7ceecd8ad97b3',
    'core/dump/data_dump/json_writer.py': '59b31ed3b0f75c2378966c7809a0f519efc11f3d77d41179019ab18931807a48',
    'pytorch/dump/module_dump/module_processor.py': '4876e89f7f726cd552fc1bf0e7b7745a898461deec2cb2abf33b02deba7329d7',
    'pytorch/dump/debugger/precision_debugger.py': '43bb9d834c736ebeb5c7c89e4770c23e1c6fc813000d12aeee8c2b72753a02fb'}


class TensorStatisticsCapture:
    def __init__(self, folder, synchronize):
        import msprobe
        from msprobe.pytorch import PrecisionDebugger
        self.folder = Path(folder); self.folder.mkdir(parents=True, exist_ok=False)
        self.thread = threading.get_ident()
        self.synchronize = synchronize
        self.active = False
        self.depth = 0
        self.model = None
        self.rows = []
        package = Path(msprobe.__file__).resolve().parent
        files = {p.relative_to(package).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in package.rglob('*') if p.is_file() and p.suffix in ('.py', '.yaml', '.json')}
        for name, expected in REVIEWED_CONTROLS.items():
            if files.get(name) != expected:
                raise RuntimeError('UNREVIEWED_COLLECTOR_CAPTURE_CONTROL:' + name)
        config = dict(task='statistics', level='mix', precision='high', rank=[], step=[0, 1, 2],
            dump_path=str(self.folder / 'observation'), async_dump=False,
            statistics=dict(scope=[], list=[], data_mode=['all'], summary_mode='statistics', tensor_list=[]))
        config_path = self.folder / 'config.json'
        config_path.write_text(json.dumps(config, indent=2)+'\n')
        (self.folder / 'collector_identity.json').write_text(json.dumps(dict(package=str(package),
            files=files, edited_by_experiment=False, raw_statistics_rewritten=False,
            controls='Public PrecisionDebugger.stop/start; verified same-step append-preserving writer',
            excludes='Owned provider diagnostics/proxy arithmetic only; actual module returns stay captured',
            model_api_filter=False, synthetic_first_difference_tensor=False), indent=2)+'\n')
        self.collector = PrecisionDebugger(config_path=str(config_path))

    def _owner(self):
        if threading.get_ident() != self.thread:
            raise RuntimeError('COLLECTOR_CAPTURE_OWNER_THREAD_CHANGED')

    @contextmanager
    def __call__(self, model, phase):
        if phase not in ('statistics_baseline', 'statistics_baseline_repeat', 'statistics_fault'):
            yield
            return
        self._owner()
        if self.active or self.depth:
            raise RuntimeError('COLLECTOR_CAPTURE_WINDOW_ALREADY_ACTIVE')
        self.model = model
        self.rows.append(dict(phase=phase, step=len(self.rows), pauses=0, completed=False))
        self.collector.start(model=model, rank_id=0)
        self.active = True
        try:
            yield
            self.synchronize()
            self.rows[-1]['completed'] = True
        finally:
            self.active = False
            self.collector.stop()
            self.collector.step()
            self.model = None
            self.save()

    @contextmanager
    def diagnostics(self):
        self._owner()
        outer = self.active and self.depth == 0
        self.depth += 1
        try:
            if outer:
                self.synchronize()
                self.collector.stop()
                self.rows[-1]['pauses'] += 1
            yield
        finally:
            self.depth -= 1
            if outer:
                self.synchronize()
                # No step/reset occurs here: reviewed JSON writer appends to
                # the same in-memory/raw capture and preserves API ordinals.
                self.collector.start(model=self.model, rank_id=0)

    def save(self):
        (self.folder / 'capture_windows.json').write_text(json.dumps(dict(phases=self.rows,
            diagnostics_depth=self.depth, active=self.active, raw_data_modified=False,
            comparator=dict(rtol=1e-5, atol=1e-8, null_equal=False, nan_equal=False)), indent=2)+'\n')
