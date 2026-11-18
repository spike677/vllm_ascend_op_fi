"""Actual-call Native strategies for the four separate Attention mechanisms.

Trusted deployment code constructs these objects before RPC admission. No
callable, library path or policy is accepted from a request. Q/K/V, masks,
optional arguments and the original return object belong to the caller.
"""
from contextlib import contextmanager
import ctypes
import hashlib
import json
import os
from pathlib import Path
import threading
import uuid
import struct

from .serving_context import require
from .attention28_prefill import strategy_request, verify_sdk_binding
from fault_injection.native_adapters.attention28.launch import (
    LaunchObserver, bundle_kernel_abi, sha, verify_launch)

MECHANISMS = ('28.fa.legal_tiling', '28.fia.legal_tiling',
              '28.fia.denominator_replay', '33.paged_decode')


class OwnedLaunchScope:
    """Retain an uncertain stop until actual TLS readback confirms zero scope.

The readback is a new observer ABI. Old preloaded observers cannot silently
inherit it. Its return value is only ownership state, never launch evidence.
"""
    def __init__(self, observer):
        require(type(observer) is LaunchObserver, 'ATTENTION_REVIEWED_LAUNCH_OBSERVER_REQUIRED')
        self.observer, self.thread = observer, threading.get_ident()
        self.readback = observer.lib.dcat_attention_observer_scope_active_v1
        self.readback.argtypes, self.readback.restype = [], ctypes.c_int
        self.matches = observer.lib.dcat_attention_observer_scope_matches_v1
        self.matches.argtypes, self.matches.restype = [ctypes.c_char_p], ctypes.c_int
        self.owned = False
        self.token = None
        self.first_error = None
        require(self.readback() == 0, 'ATTENTION_FOREIGN_NATIVE_SCOPE')

    def state(self):
        require(threading.get_ident() == self.thread, 'ATTENTION_NATIVE_OWNER_THREAD_CHANGED')
        active = self.readback()
        require(active in (0, 1), 'ATTENTION_NATIVE_READBACK_INVALID')
        require(self.owned or active == 0, 'ATTENTION_FOREIGN_NATIVE_SCOPE')
        require(not active or self.token is not None and self.matches(self.token.encode()) == 1,
                'ATTENTION_NATIVE_TOKEN_OWNER_CHANGED')
        return dict(owned=self.owned, active=bool(active), first_error=self.first_error)

    def stop(self):
        state = self.state()
        if not self.owned:
            return state
        try:
            if state['active']:
                require(self.observer.lib.dcat_attention_observer_end_v1() == 0,
                        'ATTENTION_NATIVE_END_FAILED')
        except BaseException as error:
            if self.first_error is None:
                self.first_error = repr(error)
            raise
        finally:
            # A failed end may already have removed its own TLS token. Only
            # actual same-thread readback can retire our Python handle.
            if self.readback() == 0:
                self.owned = self.observer.active = False
                self.token = None
        require(not self.owned, 'ATTENTION_NATIVE_STOP_UNCONFIRMED')
        return self.state()

    @contextmanager
    def scope(self, token):
        require(not self.state()['active'] and not self.owned, 'ATTENTION_NATIVE_REENTRY')
        self.token = token
        self.owned = True  # Preserve owner before an uncertain foreign call.
        try:
            require(self.observer.lib.dcat_attention_observer_begin_v1(token.encode()) == 0,
                    'ATTENTION_NATIVE_BEGIN_FAILED')
            self.observer.active = True
            yield
        except BaseException as error:
            if self.first_error is None:
                self.first_error = repr(error)
            raise
        finally:
            try:
                self.stop()
            except BaseException as error:
                if self.first_error is None:self.first_error = repr(error)
                raise


class _NativeCall:
    simulation = False

    def _initialize(self, mechanism, observer_receipt, observer_receipt_sha256, evidence_dir):
        require(mechanism in MECHANISMS, 'ATTENTION_UNKNOWN_MECHANISM')
        require(sha(observer_receipt) == observer_receipt_sha256, 'ATTENTION_OBSERVER_RECEIPT_CHANGED')
        receipt = json.loads(Path(observer_receipt).read_bytes())
        require(receipt.get('observer_compiled') is True and receipt.get('actual_npu_calls') == 0
                and receipt.get('headers') and all(Path(x['path']).is_file() and sha(x['path']) == x['sha256']
                    for x in receipt['headers']), 'ATTENTION_OBSERVER_BUILD_OR_SDK_CHANGED')
        library = receipt['observer_library']
        observer = LaunchObserver(library['path'], library['sha256'])
        self.scope = OwnedLaunchScope(observer)
        self.mechanism, self.thread = mechanism, threading.get_ident()
        self.policy_prefix = 'NES_FA_' if mechanism == '28.fa.legal_tiling' else 'FAULT28_FIA_GRAPH_'
        self.folder = Path(evidence_dir).resolve()
        require(self.folder.is_relative_to('/evidence') and not self.folder.exists(),
                'ATTENTION_FRESH_EVIDENCE_DIRECTORY_REQUIRED')
        self.folder.mkdir(parents=True)
        self.environment, self.failures, self.calls = {}, [], []
        self.runtime_dir = Path(os.environ['DCAT_ATTN_OBSERVER_DIR']).resolve(strict=True)
        self.runtime_environment = {name: os.environ.get(name) for name in
            ('DCAT_ATTN_OBSERVER_DIR', 'ASCEND_CUSTOM_OPP_PATH', 'ACLNN_CACHE_LIMIT', 'TASK_QUEUE_ENABLE')}
        require(self.runtime_environment['ACLNN_CACHE_LIMIT'] == '0'
                and self.runtime_environment['TASK_QUEUE_ENABLE'] == '0',
                'ATTENTION_UNCACHED_SYNCHRONOUS_DEPLOYMENT_REQUIRED')
        self.observer_sha256 = library['sha256']

    def _check(self):
        require(threading.get_ident() == self.thread, 'ATTENTION_NATIVE_OWNER_THREAD_CHANGED')
        require(all(os.environ.get(k) == v for k, v in self.runtime_environment.items()),
                'ATTENTION_RUNTIME_ENVIRONMENT_CHANGED')
        require(all(name in self.environment and self.environment[name] == value for name, value in
            os.environ.items() if name.startswith(self.policy_prefix)), 'ATTENTION_FOREIGN_NATIVE_ENVIRONMENT')
        return self.scope.state()

    def state(self):
        scope = self._check()
        return dict(native_loaded=True, observer_sha256=self.observer_sha256,
            scope_active=scope['active'], scope_owned=scope['owned'],
            environment_owned=len(self.environment), calls=len(self.calls), failures=list(self.failures),
            first_error=scope['first_error'], native_launch_qualified=False, build=self.fingerprint, simulation=False)

    def stop(self):
        self._check()
        self.scope.stop()
        for name, value in list(self.environment.items())[::-1]:
            require(os.environ.get(name) in (None, value), 'ATTENTION_NATIVE_ENV_OWNER_CHANGED')
            if os.environ.get(name) == value:
                del os.environ[name]
            require(name not in os.environ, 'ATTENTION_NATIVE_ENV_RESTORE_UNCONFIRMED')
            del self.environment[name]
        import torch
        torch.npu.synchronize()
        return self.state()

    def _run(self, token, environment, invoke_once, verify):
        self._check()
        require(not self.environment and not any(k in os.environ for k in environment),
                'ATTENTION_FOREIGN_NATIVE_ENVIRONMENT')
        import torch
        torch.npu.synchronize()
        self.environment = dict(environment)  # Before the first environment write.
        try:
            os.environ.update(environment)
            with self.scope.scope(token):
                result = invoke_once()
                torch.npu.synchronize()
                self.scope.observer.synchronize()
            row = verify()
            self.calls.append(row)
            (self.folder / (token + '.json')).write_text(json.dumps(row, allow_nan=False, indent=2)+'\n')
            return result, row
        except BaseException as error:
            self.failures.append(dict(token=token, error=repr(error)))
            raise
        finally:
            self.stop()


class FaNativeCall(_NativeCall):
    """Fresh FA S1/S2 selector around an already occurring FA v3 call."""
    def __init__(self, *, build_root, completion_sha256, sdk_root, observer_receipt,
                 observer_receipt_sha256, evidence_dir, selection_target='query_tile'):
        from fault_injection.native_adapters.attention28.fa.launch import verify_bundle, verify_runtime_sdk
        require(selection_target in ('query_tile', 'key_group', 'joint'), 'FA_SELECTION_TARGET_INVALID')
        self.bundle = verify_bundle(build_root, completion_sha256)
        verify_runtime_sdk(self.bundle, sdk_root)
        require(os.environ.get('ASCEND_CUSTOM_OPP_PATH') == self.bundle['vendor'], 'FA_FRESH_VENDOR_NOT_SELECTED')
        self.fingerprint = dict(completion_sha256=completion_sha256,
            kernel_sha256=self.bundle['completion']['kernel_binary_sha256'], mechanism='28.fa.legal_tiling')
        self.selection_target, self.native = selection_target, {}
        self._initialize('28.fa.legal_tiling', observer_receipt, observer_receipt_sha256, evidence_dir)

    def execute(self, actual, boundary, phase, invoke_once):
        from fault_injection.native_adapters.attention28.fa.launch import verify_fa_launch
        require(boundary['actual_operator'] == 'torch_npu.npu_fusion_attention_v3'
            and boundary['input_contract_matches'], 'FA_ACTUAL_V3_BNSD_BOUNDARY_REQUIRED')
        token = uuid.uuid4().hex
        key = boundary['target_key']
        fault, off = phase == 'fault', phase in ('observe', 'observe_repeat', 'off', 'health', 'recovery_off')
        native = self.native.get(key)
        require(not fault or native is not None, 'FA_SAME_TARGET_NATIVE_FACTORS_REQUIRED')
        selected = dict(native) if fault else None
        if fault:
            if self.selection_target in ('query_tile', 'joint'):selected['m'] = 192-native['m']
            if self.selection_target in ('key_group', 'joint'):selected['n_ratio'] = 12-native['n_ratio']
        context = dict(query_shape=list(actual['query'].shape), key_shape=list(actual['key'].shape),
            value_shape=list(actual['value'].shape), mask_shape=list(actual['atten_mask'].shape),
            dtype='float16', mask_dtype='bool', layout='BNSD',
            scale=struct.unpack('<f', struct.pack('<f', actual['scale']))[0], keep_prob=1,
            pre_tokens=2147483647, next_tokens=0, heads=16, inner_precise=0, sparse_mode=2,
            optional_inputs_absent=True)
        data = self.bundle['completion']
        plan = dict(schema='nes.fa.typed_plan.v1', operator='FlashAttentionScore',
            mode='controlled' if fault else 'native', scope=dict(pid=os.getpid(), native_tid=threading.get_native_id(),
                token=token, phase='operator_execution', request_count=boundary['scheduler']['num_requests']),
            context=context, version=dict(source_commit=data['source_commit'], cann_build='V100R001C11SPC001B243',
                bridge_abi='nes.fa.cann9.1.s1s2.v1'), requested=dict(native=native, selected=selected),
            certificate=dict(id='actual-worker-experimental-only', source_manifest_sha256=data['patch_receipt_sha256'],
                build_manifest_sha256=self.bundle['completion_sha256'],
                validation_sha256=hashlib.sha256(json.dumps(boundary, sort_keys=True).encode()).hexdigest(),
                validation_state='experimental'))
        log = self.folder/(token+'-host.jsonl')
        environment = {} if off else dict(NES_FA_PLAN_JSON=json.dumps(plan, separators=(',', ':')),
                                          NES_FA_LOG_PATH=str(log))
        def verify():
            hosts = [] if off else [json.loads(x) for x in log.read_text().splitlines()]
            require(off or len(hosts) == 1 and hosts[0]['token'] == token and hosts[0]['plan'] == plan,
                    'FA_ACTUAL_HOST_PLAN_DIFFERS')
            launch = verify_fa_launch(self.runtime_dir, token, self.bundle, hosts[0] if hosts else None,
                                      self.folder/(token+'.bin'), unscoped_off=off)
            pod = launch['typed']
            factors = dict(m=pod['coreParams.s1BaseSize'], n_ratio=pod['coreParams.nRatio'])
            require(factors['m'] in (64, 128) and factors['n_ratio'] in (4, 8), 'FA_ACTUAL_FACTORS_UNADAPTED')
            require(not fault or factors == selected, 'FA_FINAL_SELECTION_DIFFERS')
            if native is not None and not fault:
                require(factors == native, 'FA_NATIVE_FACTORS_CHANGED')
            self.native[key] = native or factors
            return dict(token=token, phase=phase, target_key=key, host=hosts, native=launch,
                        original_calls=1, selection_applied=fault, mechanism=self.mechanism)
        return self._run(token, environment, invoke_once, verify)


class FiaNativeCall(_NativeCall):
    """Fresh FIA legal templates and replay are distinct verified build variants."""
    def __init__(self, *, mechanism, completion, completion_sha256, vendor, schema, policy_receipt,
                 schema_sha256, sdk_root, observer_receipt, observer_receipt_sha256, evidence_dir):
        require(mechanism in MECHANISMS[1:3], 'FIA_DISTINCT_MECHANISM_REQUIRED')
        from fault_injection.native_adapters.attention28 import launch
        contract = Path(launch.__file__).with_name('fia')/'build_contract.json'
        verify_sdk_binding(completion, completion_sha256, contract, sdk_root)
        data = json.loads(Path(completion).read_bytes())
        require(data['variant'] == mechanism.rsplit('.', 1)[1], 'FIA_BUILD_VARIANT_DIFFERS')
        keys = {103000000000000000, 109000000000000000, 109000000000400000}
        if data['variant'] == 'denominator_replay':keys.add(119000000000400000)
        require(set(data.get('compiled_keys', [])) == keys and len(data['compiled_keys']) == len(keys),
                'FIA_DISTINCT_COMPILED_ENTRY_SET_REQUIRED')
        require(sha(policy_receipt) == data['patch_receipt_sha256'], 'FIA_FRESH_POLICY_RECEIPT_CHANGED')
        policy = Path(contract).with_name('fault28_fia_policy.h').read_text(encoding='utf-8')
        if data['variant'] == 'legal_tiling':
            policy = policy.replace(', "batch_coupled_replay_m128", "coupled_replay_m128"', '')
        policy_hash = hashlib.sha256(policy.encode()).hexdigest()
        policies = [item for item in json.loads(Path(policy_receipt).read_bytes())['files']
                    if item['path'].endswith('/fault28_fia_policy.h')]
        require(len(policies) == 1 and policies[0]['after_sha256'] == policy_hash,
                'FIA_CURRENT_NATIVE_THREAD_OWNER_GUARD_REQUIRED')
        self.vendor = Path(vendor).resolve(strict=True)
        files = {}
        for row in data['artifacts']:
            relative = Path(row['path']); path = self.vendor/relative
            require(not relative.is_absolute() and '..' not in relative.parts and row['path'] not in files
                and path.resolve(strict=True).is_relative_to(self.vendor) and path.is_file()
                and path.stat().st_size == row['bytes'] and sha(path) == row['sha256'], 'FIA_FRESH_ARTIFACT_CHANGED')
            if path.is_symlink():
                require(row['path'] == 'op_impl/ai_core/tbe/op_tiling/liboptiling.so'
                    and os.readlink(path) == 'lib/linux/aarch64/libcust_opmaster_rt2.0.so', 'FIA_VENDOR_LINK_UNREVIEWED')
            files[row['path']] = row
        require({p.relative_to(self.vendor).as_posix() for p in self.vendor.rglob('*') if p.is_file()} == set(files),
                'FIA_FRESH_VENDOR_INVENTORY_CHANGED')
        require(os.environ.get('ASCEND_CUSTOM_OPP_PATH') == str(self.vendor), 'FIA_FRESH_VENDOR_NOT_SELECTED')
        require(sha(schema) == schema_sha256, 'FIA_FRESH_SCHEMA_CHANGED')
        self.schema = json.loads(Path(schema).read_bytes())
        require(self.schema.get('type') == 'FusedInferAttentionScoreTilingData'
            and self.schema.get('size') == 1576, 'FIA_PUBLIC_SERIALIZER_REQUIRED')
        self.metadata, self.abi = bundle_kernel_abi(self.vendor, files, data['kernel_binary_sha256'])
        self.fingerprint = dict(completion_sha256=completion_sha256,
            kernel_sha256=data['kernel_binary_sha256'], mechanism=mechanism)
        self._initialize(mechanism, observer_receipt, observer_receipt_sha256, evidence_dir)

    def execute(self, actual, boundary, phase, invoke_once):
        require(boundary['actual_operator'] == 'torch_npu.npu_fused_infer_attention_score'
                and boundary['input_contract_matches'], 'FIA_ACTUAL_BNSD_BOUNDARY_REQUIRED')
        token = uuid.uuid4().hex
        arm = 'fault' if phase == 'fault' else 'baseline' if phase.startswith('baseline') else 'off'
        plan = strategy_request(self.mechanism, arm)
        log = self.folder/(token+'-host.jsonl')
        start = Path('/proc/self/stat').read_text().rsplit(')', 1)[1].split()[19]
        environment = dict(FAULT28_FIA_GRAPH_MODE=plan['requested_mode'], FAULT28_FIA_GRAPH_TOKEN=token,
            FAULT28_FIA_GRAPH_PID=str(os.getpid()), FAULT28_FIA_GRAPH_OWNER_START=start,
            FAULT28_FIA_GRAPH_TID=str(threading.get_native_id()),
            FAULT28_FIA_GRAPH_LOG=str(log))
        def verify():
            rows = [json.loads(x) for x in log.read_text().splitlines()]
            require(len(rows) == 1 and rows[0]['token'] == token, 'FIA_EXACT_ONE_ACTUAL_HOST_CALLBACK_REQUIRED')
            host = rows[0]
            require(host['eligible'] is True and host['mode'] == plan['requested_mode']
                and host['selected_m'] == plan['requested_m'] and int(host['tiling_key']) == plan['requested_key']
                and host['owner_pid'] == os.getpid() and str(host['owner_starttime_ticks']) == start
                and host['pid'] == os.getpid() and host['tid'] == threading.get_native_id()
                and host['serializer_roundtrip_matches'] is True, 'FIA_ACTUAL_HOST_SELECTION_DIFFERS')
            launch = verify_launch(self.runtime_dir, token, kernel_sha256=self.fingerprint['kernel_sha256'],
                schema=self.schema, host=host, kernel_metadata=self.metadata, kernel_abi=self.abi)
            return dict(token=token, phase=phase, target_key=boundary['target_key'], host=host, native=launch,
                        original_calls=1, selection_applied=phase == 'fault', mechanism=self.mechanism)
        return self._run(token, environment, invoke_once, verify)


class PagedNativeCall:
    """Reuse the actual TND/page16 adapter, with recoverable observer ownership."""
    mechanism, simulation = '33.paged_decode', False

    def __init__(self, adapter):
        from .paged33_tnd import TNDNativeAdapter
        require(type(adapter) is TNDNativeAdapter and adapter.arm != 'fault', 'TND33_REVIEWED_INERT_ADAPTER_REQUIRED')
        self.adapter = adapter
        self.scope = OwnedLaunchScope(adapter.observer)
        require(sha(adapter.build['completion_path']) == adapter.build['completion_sha256'],
                'TND33_COMPLETION_HASH_DIFFERS')
        self.fingerprint = dict(mechanism=self.mechanism,
            completion_sha256=adapter.build['completion_sha256'],
            kernel_sha256=adapter.build['completion']['kernel_binary_sha256'])
        require('scope' not in vars(adapter.observer), 'TND33_OBSERVER_SCOPE_ALREADY_OWNED')
        # This is the adapter's private observer, not a framework/API hook.
        # All constructor checks finish before installing its recovery scope.
        adapter.observer.scope = self.scope.scope

    def state(self):
        scope = self.scope.state()
        return dict(native_loaded=True, scope_active=scope['active'], scope_owned=scope['owned'],
            environment_owned=sum(name.startswith('DCAT33_TND_') for name in os.environ),
            calls=len(self.adapter.calls), failures=list(self.adapter.failures), first_error=scope['first_error'],
            build=self.fingerprint, simulation=False)

    def execute(self, actual, boundary, phase, invoke_once):
        self.adapter.arm = 'fault' if phase == 'fault' else 'off'
        self.adapter.enabled = True
        result = self.adapter.execute(actual, boundary, invoke_once)
        row = dict(self.adapter.calls[-1], mechanism=self.mechanism, phase=phase,
                   original_calls=1, selection_applied=phase == 'fault', target_key=boundary['target_key'])
        return result, row

    def stop(self):
        self.scope.stop()
        self.adapter.disable()
        return self.state()
