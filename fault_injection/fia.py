"""FIA adapter: discover from calls, keep the native kernel's exact support guard.

No model-family imports. The caller uses the existing verified vendor launcher;
this module never loads an arbitrary .so or changes global SDK files.
"""
import json
import math
import os
from pathlib import Path
import threading
import uuid

from .core import Decision, UnsupportedCall

VENDOR_SHA = '0ce1bfe8e4722b965efb328c7242e0ca1aa9f69769a0a27cc28e80b9403e4e14'
_NATIVE_LOCK = threading.Lock()


class FiaBackend:
    """Bounded denominator-replay backend, shared by any compatible model.

    Native hit/launch evidence remains separate from Python routing coverage.
    The old verified launcher must run before importing torch/torch_npu.
    """
    def __init__(self, *, arm='fault', evidence_dir=None, mechanism='28.fia.denominator_replay'):
        if arm not in ('baseline', 'fault'):
            raise ValueError('arm must be baseline or fault')
        self.arm = arm
        if mechanism not in ('28.fia.denominator_replay', '28.fia.legal_tiling'):
            raise ValueError('Unrecognized FIA mechanism')
        self.mechanism = mechanism
        self.evidence_dir = Path(evidence_dir or os.environ.get('DEMONCAT_EVIDENCE_DIR', '/evidence/unified'))
        self.records = []

    def native_plan(self):
        if self.mechanism == '28.fia.legal_tiling':
            # The already built healthy coupled templates. No replay key is
            # selected in either arm; this is not the historical default path.
            return (('coupled_base512', 512, '109000000000000000') if self.arm == 'baseline'
                    else ('coupled_m128', 128, '109000000000400000'))
        return (('coupled_m128',128,'109000000000400000') if self.arm == 'baseline'
                else ('coupled_replay_m128',128,'119000000000400000'))

    def check(self, call):
        # The model has already applied RoPE and cache updates before this
        # boundary. Do not silently discard a semantic option from a new model.
        unknown = set(call.kwargs) - {'is_causal', 'sliding_window', 'output_attentions', 'position_ids', 'use_cache'}
        if unknown:
            return Decision(False, 'Unvalidated attention arguments: ' + ', '.join(sorted(unknown)))
        if call.kwargs.get('is_causal') is False:
            return Decision(False, 'Explicit non-causal override is unsupported')
        if 'use_cache' in call.kwargs and type(call.kwargs['use_cache']) is not bool:
            return Decision(False, 'Unresolved cache option')
        try:
            g = call.geometry()
        except UnsupportedCall as exc:
            return Decision(False, str(exc))
        if call.module.training or call.dropout != 0:
            return Decision(False, 'Only inference with dropout=0 is supported')
        if g['batch'] not in (1, 8) or any(g[k] != n for k, n in dict(
                query_heads=16, kv_heads=8, query_length=2048, kv_length=2048,
                qk_dim=128, value_dim=128).items()):
            return Decision(False, 'Pinned kernel supports B1/B8, Hq16/Hkv8, S2048, Dqk=Dv128 only; model family is irrelevant')
        if any(str(t.dtype) != 'torch.float16' or t.device.type != 'npu'
               for t in (call.query, call.key, call.value)):
            return Decision(False, 'Pinned kernel requires FP16 NPU tensors')
        if len({str(t.device) for t in (call.query, call.key, call.value)}) != 1:
            return Decision(False, 'Q/K/V must reside on one device')
        if call.scaling is None or not math.isclose(float(call.scaling), 128**-0.5, rel_tol=0, abs_tol=1e-8):
            return Decision(False, 'Native scale outside pinned contract')
        if not getattr(call.module, 'is_causal', False):
            return Decision(False, 'Only causal self attention is supported')
        for key in ('sliding_window', 'softcap', 'softmax_scale', 'attention_bias', 'alibi', 'head_mask'):
            if call.kwargs.get(key) is not None:
                return Decision(False, 'Unsupported attention semantic: ' + key)
        if call.kwargs.get('output_attentions', False):
            return Decision(False, 'FIA does not return attention weights')
        return Decision(True, 'Geometry matches pinned kernel; mask and runtime checked before execution')

    @staticmethod
    def validate_positions(call):
        # HF has already applied RoPE/updated cache. Only fresh, full prefill
        # positions are admitted by this pinned native adapter, even when the
        # caller requests construction of a cache for a subsequent decode.
        import torch
        positions = call.kwargs.get('position_ids')
        if positions is None:
            return
        b, _, s, _ = call.query.shape
        if (not isinstance(positions, torch.Tensor) or positions.ndim != 2
                or positions.shape[0] not in (1, b) or positions.shape[1] != s
                or positions.dtype not in (torch.int32, torch.int64)):
            raise UnsupportedCall('Position mapping is not a full prefill')
        expected = torch.arange(s, device=positions.device, dtype=positions.dtype).expand_as(positions)
        if not torch.equal(positions, expected):
            raise UnsupportedCall('Nonzero cache offset, padding positions or packed positions are unsupported')

    @staticmethod
    def causal_mask(call):
        import torch
        s = call.query.shape[2]
        blocked = torch.ones((s, s), dtype=torch.bool, device=call.query.device).triu(1)
        mask = call.attention_mask
        if mask is None:
            # The session preserves the original mask factory. A missing mask
            # is only acceptable for full causal prefill with equal Q/K lengths.
            return blocked
        if mask.device != call.query.device or mask.ndim not in (2, 4):
            raise UnsupportedCall('Mask must be a same-device 2D/4D full causal mask')
        if tuple(mask.shape[-2:]) != (s, s) or (mask.ndim == 4 and (
                mask.shape[0] not in (1, call.query.shape[0]) or mask.shape[1] not in (1, call.query.shape[1]))):
            raise UnsupportedCall('Padding/packed/partial mask is outside the pinned kernel contract')
        if mask.dtype == torch.bool:
            # HF eager/sdpa mask factories use True=allowed; FIA uses True=blocked.
            valid = torch.equal(mask, (~blocked).expand_as(mask))
        elif mask.is_floating_point():
            expected = blocked.expand_as(mask)
            # Admit only hard masking, never arbitrary additive bias.
            hard = torch.isneginf(mask) | (mask == torch.finfo(mask.dtype).min)
            valid = bool(torch.all(torch.where(expected, hard, mask == 0)).item())
        else:
            valid = False
        if not valid:
            raise UnsupportedCall('Non-causal, padding, sliding-window or biased mask rejected')
        return blocked

    def _runtime(self):
        import hashlib
        manifest = os.environ.get('DEMONCAT_FIA_VENDOR_MANIFEST')
        if not manifest or hashlib.sha256(Path(manifest).read_bytes()).hexdigest() != VENDOR_SHA:
            raise UnsupportedCall('Run the verified denominator-replay vendor launcher and set DEMONCAT_FIA_VENDOR_MANIFEST')
        # Manifest identity alone is insufficient: hash the on-disk vendor tree.
        # This is not verification of a loaded kernel or a real launch.
        data = json.loads(Path(manifest).read_bytes())
        roots = os.environ.get('ASCEND_CUSTOM_OPP_PATH', '').split(':')
        vendor = Path(manifest).parent / 'vendors' / 'fault28_fia_graph_transformer'
        if str(vendor) not in roots or os.environ.get('ACLNN_CACHE_LIMIT') != '0':
            raise UnsupportedCall('Fresh verified vendor process with ACLNN_CACHE_LIMIT=0 required')
        expected = set()
        for row in data['artifacts']:
            file = vendor / row['path']
            if not file.resolve().is_relative_to(vendor.resolve()) or not file.is_file():
                raise UnsupportedCall('Invalid vendor artifact path')
            if file.stat().st_size != row['bytes'] or hashlib.sha256(file.read_bytes()).hexdigest() != row['sha256']:
                raise UnsupportedCall('Vendor artifact differs from validated bundle: ' + row['path'])
            expected.add(row['path'])
        if {p.relative_to(vendor).as_posix() for p in vendor.rglob('*') if p.is_file()} != expected:
            raise UnsupportedCall('Vendor inventory mismatch')
        return VENDOR_SHA

    def _host_receipt(self, rows, token, owner_start):
        mode, selected_m, key = self.native_plan()
        matched = [r for r in rows if r.get('token') == token]
        if len(matched) != 1:
            raise UnsupportedCall('Expected exactly one scoped host decision; concurrent/cache calls are unsupported')
        r = matched[0]
        if not (r.get('owner_pid') == os.getpid() and r.get('owner_starttime_ticks') == owner_start
                and r.get('mode') == mode and r.get('eligible') is True
                and r.get('selected_m') == selected_m and str(r.get('tiling_key')) == key
                and r.get('serializer_roundtrip_matches') is True
                and isinstance(r.get('checks'), dict) and r['checks']
                and all(value is True for value in r['checks'].values())):
            raise UnsupportedCall('Native host decision differs from this call contract')
        return r

    def run(self, call):
        decision = self.check(call)
        if not decision.supported:
            raise UnsupportedCall(decision.reason)
        import torch
        import torch_npu
        if torch.compiler.is_compiling():
            raise UnsupportedCall('This adapter is eager only; graph integration requires separate acceptance')
        if threading.current_thread() is not threading.main_thread() or threading.active_count() != 1:
            raise UnsupportedCall('Use a dedicated single-request process; native policy is process scoped')
        self.validate_positions(call)
        mask = self.causal_mask(call)
        if not _NATIVE_LOCK.acquire(blocking=False):
            raise UnsupportedCall('Concurrent native fault scopes are unsupported')
        saved = {}
        try:
            if not getattr(self, '_verified', False):
                self._runtime()
                self._verified = True
            if not self.evidence_dir.is_absolute() or not self.evidence_dir.resolve().is_relative_to(Path('/evidence')):
                raise UnsupportedCall('Host policy requires an absolute /evidence directory')
            self.evidence_dir.mkdir(parents=True, exist_ok=True)
            token = uuid.uuid4().hex
            log = self.evidence_dir / (token + '.jsonl')
            stat = Path('/proc/self/stat').read_text().rsplit(')', 1)[1].split()
            env = dict(FAULT28_FIA_GRAPH_MODE=self.native_plan()[0],
                       FAULT28_FIA_GRAPH_TOKEN=token, FAULT28_FIA_GRAPH_PID=str(os.getpid()),
                       FAULT28_FIA_GRAPH_OWNER_START=stat[19], FAULT28_FIA_GRAPH_LOG=str(log))
            saved = {k: os.environ.get(k) for k in env}
            os.environ.update(env)
            q, k, v = (t.contiguous() for t in (call.query, call.key, call.value))
            # Resolve after msprobe hooks are installed; do not cache the API.
            out, _lse = torch_npu.npu_fused_infer_attention_score(q, k, v,
                atten_mask=mask, actual_seq_lengths=[q.shape[2]]*q.shape[0],
                actual_seq_lengths_kv=[k.shape[2]]*k.shape[0], num_heads=q.shape[1],
                num_key_value_heads=k.shape[1], scale=call.scaling, input_layout='BNSD',
                pre_tokens=2147483647, next_tokens=0, sparse_mode=2, inner_precise=0)
            torch.npu.synchronize()
            rows = [json.loads(line) for line in log.read_text().splitlines()] if log.exists() else []
            self._host_receipt(rows, token, stat[19])
            self.records.append(dict(module=call.path, mechanism=self.mechanism, geometry=call.geometry(), host_eligible=True,
                                     host_log=str(log), native_launch_verified=False))
            return out.transpose(1, 2), None
        finally:
            for name, value in saved.items():
                if value is None:
                    os.environ.pop(name, None)
                else:
                    os.environ[name] = value
            _NATIVE_LOCK.release()
