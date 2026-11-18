"""Diagnostic reads at one real decode API boundary; never feeds copied inputs.

The stock trace deliberately has no assumed kernel ABI or private-bundle key.
This module grants no bridge, fault, C2 or C3 admission.
"""
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import re
import threading
import uuid

from .paged33_audit import _meta
from .serving_context import ContextRefused, require


FIRST_LAYER = 'model.layers.0.self_attn.attn'
MAX_TENSOR_BYTES = 128 * 1024**2


def sha(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024**2), b''):
            result.update(block)
    return result.hexdigest()


def tensor_version(value):
    try:
        return value._version
    except RuntimeError:
        return None  # inference tensors have no version counter


def tensor_snapshot(value, path=None):
    import torch
    require(isinstance(value, torch.Tensor) and value.layout == torch.strided
            and value.numel() * value.element_size() <= MAX_TENSOR_BYTES, 'BOUNDARY_TENSOR_LIMIT')
    metadata = dict(_meta(value), object_id=id(value))
    version = tensor_version(value)
    # Contiguity conversion happens on the CPU, never feeds the real operator.
    cpu = value.detach().cpu().contiguous()
    raw = cpu.reshape(-1).view(torch.uint8).numpy().tobytes()
    require(dict(_meta(value), object_id=id(value)) == metadata, 'BOUNDARY_STORAGE_CHANGED_DURING_READ')
    if path is not None:
        with Path(path).open('xb') as stream:
            stream.write(raw)
    return dict(metadata=metadata, version=version, bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest(),
                finite=bool(torch.isfinite(cpu).all()), path=str(path) if path is not None else None)


def logical_cache_snapshot(cache, pages, length):
    require(cache.ndim == 4 and len(pages) == 1 and type(length) is int and length > 0,
            'BOUNDARY_SINGLE_LOGICAL_CACHE_REQUIRED')
    block = int(cache.shape[1]); selected = pages[0]
    require(len(selected) == (length + block - 1) // block and len(set(selected)) == len(selected)
            and all(type(page) is int and 0 <= page < cache.shape[0] for page in selected),
            'BOUNDARY_ACTUAL_PAGE_ORDER_INVALID')
    require(length * math.prod(cache.shape[2:]) * cache.element_size() <= MAX_TENSOR_BYTES,
            'BOUNDARY_LOGICAL_CACHE_LIMIT')
    metadata = dict(_meta(cache), object_id=id(cache)); digest = hashlib.sha256(); left = length
    import torch
    finite = True; size = 0
    for page in selected:
        count = min(left, block)
        cpu = cache[page, :count].detach().cpu().contiguous()
        raw = cpu.reshape(-1).view(torch.uint8).numpy().tobytes()
        digest.update(raw); size += len(raw); left -= count
        finite = finite and bool(torch.isfinite(cpu).all())
    require(left == 0 and dict(_meta(cache), object_id=id(cache)) == metadata,
            'BOUNDARY_CACHE_STORAGE_CHANGED_DURING_READ')
    return dict(metadata=metadata, version=tensor_version(cache), shape=[length, *list(cache.shape[2:])], bytes=size,
                sha256=digest.hexdigest(), finite=finite, physical_pages=list(selected),
                readback='ACTUAL_VALID_LOGICAL_PAGE_ORDER', operator_input_repacked=False)


def scalar_snapshot(value):
    if value is None or type(value) in (bool, int, str):
        return value
    if type(value) is float:
        require(math.isfinite(value), 'BOUNDARY_NONFINITE_SCALAR')
        return value
    require(type(value) in (tuple, list) and len(value) <= 4096, 'BOUNDARY_UNKNOWN_ARGUMENT_TYPE')
    return [scalar_snapshot(item) for item in value]


def same_tensor(before, after):
    return all(before.get(key) == after.get(key) for key in ('metadata', 'bytes', 'sha256', 'version'))


def read_scoped_launches(directory, token, *, expected_pid=None, expected_native_tid=None,runtime_index=None):
    """Join raw registration/launch facts, without decoding a guessed stock ABI."""
    root = Path(directory).resolve(strict=True); raw = root/'runtime.jsonl'
    indexed=None
    if runtime_index is not None:
        from fault_injection.native_adapters.attention28.runtime_log_index import RuntimeLogIndex
        require(type(runtime_index) is RuntimeLogIndex,'BOUNDARY_EXACT_OWNED_RUNTIME_INDEX_REQUIRED')
    limit=runtime_index.maximum_bytes if runtime_index is not None else 128*1024**2
    require(raw.stat().st_size<=limit and re.fullmatch('[0-9a-f]{32}',token),'BOUNDARY_RAW_LAUNCH_LIMIT')
    if runtime_index is not None:
        indexed=runtime_index.scope(root,token)
    snapshot = raw.read_bytes() if indexed is None else None
    if indexed is None:require(len(snapshot) <= 128 * 1024**2, 'BOUNDARY_RAW_LAUNCH_LIMIT')
    rows = indexed['rows'] if indexed is not None else [json.loads(line) for line in snapshot.splitlines()]
    scoped = [row for row in rows if row.get('call_id') == token]
    begins = [row for row in scoped if row.get('event') == 'begin']
    ends = [row for row in scoped if row.get('event') == 'end']
    require(len(begins) == len(ends) == 1, 'BOUNDARY_SCOPE_BEGIN_END_AMBIGUOUS')
    begin, end = begins[0], ends[0]; owner = (begin['pid'], begin['tid'])
    require((expected_pid is None or type(expected_pid) is int and expected_pid == owner[0])
            and (expected_native_tid is None or type(expected_native_tid) is int and expected_native_tid == owner[1]),
            'BOUNDARY_CAPTURED_NATIVE_OWNER_DIFFERS')
    begin_index = rows.index(begin); end_index = rows.index(end)
    require(begin_index < end_index and begin['monotonic_ns'] < end['monotonic_ns'] and all(
        (row['pid'], row['tid']) == owner and row.get('schema') == 'demoncat.attention.runtime_observer.v1'
        and begin_index <= index <= end_index
        and begin['monotonic_ns'] <= row['monotonic_ns'] <= end['monotonic_ns']
        for index,row in enumerate(rows) if row.get('call_id') == token),
        'BOUNDARY_SCOPE_OWNER_OR_ORDER_CHANGED')
    launches = []; gaps = []
    if any(row.get('event') == 'capture_overflow' for row in rows):
        gaps.append('REGISTERED_BINARY_CAPTURE_OVERFLOW')
    for index, row in enumerate(rows):
        if row.get('call_id') != token or row.get('event') != 'launch':
            continue
        require(row['result'] == 0 and row.get('observer_mutates_arguments') is False
                and begin['monotonic_ns'] <= row['launch_begin_ns'] <= row['launch_end_ns'] <= row['monotonic_ns'],
                'BOUNDARY_LAUNCH_FAILED_OR_ORDER_CHANGED')
        registrations = [(i,item) for i,item in enumerate(rows) if item.get('event') == 'register'
            and (item['pid'],item['handle']) == (row['pid'],row['handle'])
            and item['monotonic_ns'] <= row['launch_begin_ns']]
        registered = [(i,item) for i,item in registrations if item['generation'] == row['generation']]
        require(len(registered) == 1 and registered[0][1]['result'] == 0, 'BOUNDARY_REGISTERED_BINARY_JOIN_MISSING')
        reg_index, registration = registered[0]; name = row['binary_file']
        latest_registration = max(item['monotonic_ns'] for _,item in registrations)
        require(registration['monotonic_ns'] == latest_registration
                and sum(item['monotonic_ns'] == latest_registration for _,item in registrations) == 1
                and reg_index < index, 'BOUNDARY_STALE_REGISTERED_HANDLE_GENERATION')
        require(name == registration['binary_file'] and re.fullmatch(r'registered-\d+-\d+\.bin', name),
                'BOUNDARY_REGISTERED_BINARY_NAME_CHANGED')
        binary = root/name
        require(not binary.is_symlink() and binary.resolve(strict=True).parent == root
                and binary.stat().st_size == registration['binary_bytes'] <= MAX_TENSOR_BYTES,
                'BOUNDARY_REGISTERED_BINARY_FILE_CHANGED')
        binding = None
        if row['api'] == 'aclrtLaunchKernelWithHostArgs' or row.get('function_handle'):
            require(type(row.get('function_handle')) is int and row['function_handle'] > 0,
                    'BOUNDARY_ACTUAL_FUNCTION_JOIN_MISSING')
            # Same current-scope/last-effective-bind rule as the independent
            # source53 reader. Examine all prior identities before selecting;
            # filtering by expected key/handle would hide conflicting history.
            history = [(i,item) for i,item in enumerate(rows) if item.get('event') == 'function_bind'
                and (item['pid'],item['function_handle']) == (row['pid'],row['function_handle'])
                and item['monotonic_ns'] <= row['launch_begin_ns']]
            require(history, 'BOUNDARY_ACTUAL_FUNCTION_JOIN_MISSING')
            identity = (row['handle'],row['generation'],str(row['tiling_key']))
            require(all(item['result'] == 0 and (item['handle'],item['generation'],str(item['tiling_key'])) == identity
                        for _,item in history), 'BOUNDARY_CONFLICTING_FUNCTION_HANDLE_KEY_OR_GENERATION_HISTORY')
            current = [(i,item) for i,item in history if item.get('call_id') == token]
            require(len(current) == 1, 'BOUNDARY_CURRENT_SCOPE_FUNCTION_BIND_NOT_UNIQUE')
            bind_index, bound = current[0]; newest = max(item['monotonic_ns'] for _,item in history)
            require(sum(item['monotonic_ns'] == newest for _,item in history) == 1
                and bound['monotonic_ns'] == newest and (bound['pid'],bound['tid']) == owner
                and reg_index < bind_index < index and begin_index < bind_index
                and begin['monotonic_ns'] <= bound['monotonic_ns'] <= row['launch_begin_ns'],
                'BOUNDARY_LATEST_EFFECTIVE_BIND_NOT_IN_CURRENT_SCOPE')
            binding = dict(rule='CURRENT_SCOPE_LAST_EFFECTIVE_BIND_WITH_ALL_PRIOR_FUNCTION_IDENTITIES_CONSISTENT',
                function_handle=row['function_handle'],raw_line=indexed['line_numbers'][bind_index] if indexed is not None else bind_index+1,repeated_identical_history=len(history)-1,
                current_scope_binds=1,handle=row['handle'],generation=row['generation'],tiling_key=str(row['tiling_key']))
        arguments = bytes.fromhex(row['args_hex'])
        require(len(arguments) == row['args_size'] <= 1024**2, 'BOUNDARY_ARGUMENT_SNAPSHOT_SIZE_CHANGED')
        if not row.get('snapshot_valid'):
            gaps.append('INVALID_RAW_ARGUMENT_SNAPSHOT')
        if row.get('schedule_mode', -1) < 0:
            gaps.append('SCHEDULE_MODE_NOT_OBSERVED')
        launches.append(dict(api=row['api'], tiling_key=str(row['tiling_key']), block_dim=row['block_dim'],
            schedule_mode=row['schedule_mode'], argument_layout=row['argument_layout'],
            args_bytes=len(arguments), args_sha256=hashlib.sha256(arguments).hexdigest(),
            registered_binary=dict(path=str(binary), bytes=binary.stat().st_size, sha256=sha(binary)),
            pid=row['pid'], tid=row['tid'], generation=row['generation'],
            function_binding=binding,
            kernel_argument_abi='NOT_DECODED_NO_ASSUMED_PRIVATE_ABI', api_library_owner='NOT_CAPTURED'))
    if not launches:
        gaps.append('NO_INTERCEPTED_RUNTIME_LAUNCH')
    last_launch = max((row['launch_end_ns'] for row in scoped if row.get('event') == 'launch'), default=0)
    require(any(row.get('event') in ('device_sync', 'stream_sync') and row.get('result') == 0
                and row['monotonic_ns'] >= last_launch for row in scoped), 'BOUNDARY_SCOPE_COMPLETION_UNOBSERVED')
    return dict(call_id=token, runtime_path=str(raw), runtime_prefix_bytes=indexed['prefix_bytes'] if indexed is not None else len(snapshot),
        runtime_prefix_sha256=indexed['prefix_sha256'] if indexed is not None else hashlib.sha256(snapshot).hexdigest(), runtime_file_can_append=True, launches=launches,
        intercepted_launch_count=len(launches), observation_gaps=sorted(set(gaps)),
        argument_abi_qualified=False, actual_api_library_owner_observed=False,
        all_runtime_entrypoints_covered=False, private_key_used_to_classify_stock=False)


class MappedLibraryObservationError(ContextRefused):
    def __init__(self, reason, evidence):
        super().__init__(reason)
        self.evidence = evidence


def mapped_api_libraries():
    """Keep exact failure evidence; presence still is not resolver selection."""
    names = {'libcust_opapi.so', 'libopapi.so', 'liboptiling.so', 'libcust_opmaster_rt2.0.so', 'libtorch_npu.so'}
    report = dict(schema='demoncat.paged33.mapped_library_diagnostic.v1',pid=os.getpid(),
        python_thread_ident=threading.get_ident(),native_tid=threading.get_native_id(),
        considered_basenames=sorted(names),map_rows=[],files=[],current_file=None,
        selected_api_owner_observed=False,complete=False)
    def stat_record(value):
        return dict(inode=value.st_ino,device=value.st_dev,device_major=os.major(value.st_dev),
            device_minor=os.minor(value.st_dev),bytes=value.st_size,mtime_ns=value.st_mtime_ns)
    try:
        maps = Path('/proc/self/maps'); raw = maps.read_bytes(); paths = {}
        report.update(maps_sha256=hashlib.sha256(raw).hexdigest(),maps_bytes=len(raw))
        require(len(raw) <= 16*1024**2, 'BOUNDARY_MAPPED_LIBRARY_MAPS_LIMIT')
        for line in raw.decode().splitlines():
            parts = line.split(None, 5)
            if len(parts) == 6 and Path(parts[5].removesuffix(' (deleted)')).name in names:
                row=dict(raw_line=line,raw_path=parts[5],address_range=parts[0],permissions=parts[1],
                    offset_hex=parts[2],device_hex=parts[3],inode=int(parts[4]))
                report['map_rows'].append(row)
        require(len(report['map_rows']) <= 512, 'BOUNDARY_MAPPED_LIBRARY_ROWS_LIMIT')
        for row in report['map_rows']:
            report['current_map_row']=row
            require(not row['raw_path'].endswith(' (deleted)'), 'BOUNDARY_MAPPED_LIBRARY_DELETED')
            path=Path(row['raw_path']).resolve(strict=True);row['resolved_path']=str(path)
            row['device_major'],row['device_minor']=(int(x,16) for x in row['device_hex'].split(':'))
            paths.setdefault(path,[]).append(row)
        report.pop('current_map_row',None)
        require(len(paths) <= 32, 'BOUNDARY_MAPPED_LIBRARY_LIMIT')
        for path, rows in sorted(paths.items()):
            item=dict(path=str(path),map_rows=rows);report['current_file']=item
            before=path.stat();item['stat_before']=stat_record(before)
            item['checks']=dict(inode_matches=all(before.st_ino == row['inode'] for row in rows),
                device_matches=all((os.major(before.st_dev),os.minor(before.st_dev)) ==
                    (row['device_major'],row['device_minor']) for row in rows),
                size_within_bound=before.st_size <= 512*1024**2)
            item['failed_checks']=[key for key,value in item['checks'].items() if not value]
            require(not item['failed_checks'], 'BOUNDARY_MAPPED_LIBRARY_IDENTITY_CHANGED')
            item['sha256']=sha(path);after=path.stat();item['stat_after']=stat_record(after)
            require((before.st_dev,before.st_ino,before.st_size,before.st_mtime_ns) ==
                (after.st_dev,after.st_ino,after.st_size,after.st_mtime_ns), 'BOUNDARY_MAPPED_LIBRARY_CHANGED_DURING_HASH')
            item['bytes']=before.st_size;report['files'].append(item)
        report.update(status='ACTUAL_MAPPED_FILE_FINGERPRINTS_ONLY',complete=True,current_file=None,
            limitation='Existing observer does not intercept ACLNN resolver selection')
        return report
    except Exception as error:
        report.update(status='UNOBSERVED_OR_INCOMPLETE',error=repr(error),complete=False)
        raise MappedLibraryObservationError(str(error),report) from error


class StockLaunchScope:
    """One read-only stock scope, sharing the reviewed observer, no host policy."""
    def __init__(self, observer, synchronize, runtime_dir):
        self.observer, self.synchronize, self.runtime_dir = observer, synchronize, Path(runtime_dir)
        self.runtime_env = os.environ.get('DCAT_ATTN_OBSERVER_DIR')
        self.thread, self.pid = threading.get_ident(), os.getpid()
        self.native_tid = threading.get_native_id()
        self.token = uuid.uuid4().hex
        self.active = self.inflight = self.used = False
        self.errors = []

    def _owner(self):
        require(threading.get_ident() == self.thread and threading.get_native_id() == self.native_tid
                and os.getpid() == self.pid
                and self.observer.owner == self.thread and os.environ.get('DCAT_ATTN_OBSERVER_DIR') == self.runtime_env,
                'BOUNDARY_SCOPE_OWNER_CHANGED')

    def _end(self):
        rc = self.observer.lib.dcat_attention_observer_end_v1()
        # Exact reviewed C++ end clears its token for both 0 and -2 (emit fail).
        if rc in (0, -2):
            self.active = False
            self.observer.active = False
        require(rc == 0, 'BOUNDARY_SCOPE_END_FAILED:' + str(rc))

    def invoke(self, invoke_once):
        self._owner()
        require(not self.used and not self.active and not self.observer.active, 'BOUNDARY_SCOPE_ALREADY_USED')
        require(not any(name.startswith('DCAT33_TND_') for name in os.environ), 'BOUNDARY_STOCK_FOREIGN_NATIVE_POLICY')
        self.used = self.inflight = True; primary = None; result = None
        try:
            self.synchronize()
            require(self.observer.lib.dcat_attention_observer_begin_v1(self.token.encode()) == 0,
                    'BOUNDARY_SCOPE_BEGIN_FAILED')
            self.active = True
            self.observer.active = True
            result = invoke_once()
            self.synchronize(); self.observer.synchronize()
        except BaseException as error:
            primary = error; self.errors.append(dict(stage='invoke', error=repr(error)))
        finally:
            if self.active:
                try:
                    self._end()
                except BaseException as error:
                    self.errors.append(dict(stage='end', error=repr(error)))
                    if primary is None:
                        primary = error
            self.inflight = False
        if primary is not None:
            raise primary
        return result

    def close(self):
        self._owner(); require(not self.inflight, 'BOUNDARY_SCOPE_STILL_INFLIGHT')
        self.synchronize()
        if self.active:
            self._end()
        return dict(scope_cleared=not self.active, local_streams_drained=True, errors=list(self.errors))


def stock_scope_from_receipt(path, expected_sha256):
    from fault_injection.native_adapters.attention28 import launch
    require(sha(path) == expected_sha256, 'BOUNDARY_OBSERVER_RECEIPT_CHANGED')
    receipt = json.loads(Path(path).read_bytes())
    require(receipt.get('observer_compiled') is True and receipt.get('actual_npu_calls') == 0
            and receipt.get('observer_memory_selftest', {}).get('exit_code') == 0
            and receipt['observer_source']['sha256'] == sha(Path(launch.__file__).with_name('launch_observer.cpp')),
            'BOUNDARY_REVIEWED_OBSERVER_SOURCE_REQUIRED')
    require(receipt.get('headers') and all(Path(row['path']).is_file() and sha(row['path']) == row['sha256']
            for row in receipt['headers']), 'BOUNDARY_OBSERVER_SDK_CHANGED')
    lib = receipt['observer_library']
    observer = launch.LaunchObserver(lib['path'], lib['sha256'])
    import torch
    return StockLaunchScope(observer, torch.npu.synchronize, os.environ['DCAT_ATTN_OBSERVER_DIR'])


class FirstDecodeBoundaryCapture:
    def __init__(self, worker, folder, synchronize, *, allow_cpu_test=False, target_scope=None, target_path=None):
        self.worker, self.runner, self.model = worker, worker.model_runner, worker.model_runner.model
        self.rank = worker.rank
        self.folder, self.synchronize = Path(folder), synchronize
        self.thread, self.pid = threading.get_ident(), os.getpid()
        self.native_tid = threading.get_native_id()
        self.allow_cpu_test, self.refs, self.result, self.data = allow_cpu_test, None, None, None
        self.calls = 0; self.inflight = False
        self.target_scope,self.target_path=target_scope,target_path
        if target_scope is not None:
            from .paged33_normal_targets import NormalTargetScope
            require(type(target_scope) is NormalTargetScope and type(target_path) is str,
                    'BOUNDARY_LIVE_NORMAL_TARGET_SCOPE_REQUIRED')

    def _owner(self):
        require(threading.get_ident() == self.thread and threading.get_native_id() == self.native_tid
                and os.getpid() == self.pid
                and self.worker.model_runner is self.runner and self.runner.model is self.model and self.worker.rank == self.rank,
                'BOUNDARY_WORKER_MODEL_OR_THREAD_CHANGED')

    def start(self, name, original, args, kwargs, boundary, module, ordinal):
        import torch
        self._owner()
        expected_path,expected_ordinal=FIRST_LAYER,0
        if self.target_scope is not None:
            expected_path=self.target_path
            expected_ordinal=self.target_scope.check(self.worker,expected_path,module,
                self.target_scope.owner,self.target_scope.generation)
        require(self.data is None and ordinal == expected_ordinal and name == 'npu_fused_infer_attention_score'
                and boundary['module_path'] == expected_path and boundary['module_object_id'] == id(module)
                and boundary.get('metadata_binding') == 'VERIFIED', 'BOUNDARY_FIRST_ACTUAL_LAYER_REQUIRED')
        if not self.allow_cpu_test:
            require(original is getattr(torch.ops.npu, name), 'BOUNDARY_ACTUAL_PUBLIC_PACKET_REQUIRED')
        require(len(args) <= 3, 'BOUNDARY_POSITIONAL_OPTIONALS_UNADAPTED')
        actual = dict(kwargs)
        for key, value in zip(('query', 'key', 'value'), args):
            require(key not in actual, 'BOUNDARY_DUPLICATE_ARGUMENT'); actual[key] = value
        require(str(actual['query'].device).startswith('npu') or (self.allow_cpu_test and str(actual['query'].device) == 'cpu'),
                'BOUNDARY_ACTUAL_NPU_REQUIRED')
        require(all(actual[key] is not None for key in ('query', 'key', 'value', 'atten_mask', 'block_table')),
                'BOUNDARY_ACTUAL_ARGUMENT_MISSING')
        for key, recorded in (('query', 'query'), ('key', 'key'), ('value', 'value'), ('block_table', 'page_table')):
            current = _meta(actual[key])
            require(all(current[field] == boundary[recorded][field] for field in
                ('storage_identity', 'storage_offset', 'shape', 'stride', 'dtype', 'device', 'storage_nbytes')),
                'BOUNDARY_ACTUAL_ARGUMENT_CHANGED_AFTER_AUDIT:' + key)
        self.folder.mkdir()
        self.refs = dict(actual=actual, original=original, module=module, key=module.impl.key_cache, value=module.impl.value_cache,
                         pages=[list(row) for row in boundary['used_pages']], length=boundary['sequence_lengths'][0])
        self.data = dict(schema='demoncat.paged33.first_decode_boundary.v1', state='CAPTURING_INPUT',
            worker_object_id=id(self.worker), model_object_id=id(self.model), rank=self.worker.rank,
            pid=self.pid, thread=self.thread, boundary=json.loads(json.dumps(boundary)),
            # `thread` remains the legacy Python-ident field. It is not gettid.
            thread_identity=dict(python_thread_ident=self.thread, native_tid=self.native_tid,
                pid=self.pid, source='CURRENT_THREAD_GET_IDENT_AND_GET_NATIVE_ID'),
            positional_count=len(args), supplied_argument_names=sorted(actual),
            original_callable=dict(object_id=id(original), type=type(original).__module__+'.'+type(original).__qualname__),
            inputs={}, outputs=[], errors=[], original_calls=0,
            input_substitutions=0, operator_arguments_modified=False, fault_enabled=False, c2=False, c3=False)
        self.data['execution'] = 'CPU_CONTRACT_ONLY' if self.allow_cpu_test else 'NATURAL_WORKER_DIAGNOSTIC'
        self.synchronize()
        for key, value in actual.items():
            if isinstance(value, torch.Tensor):
                # Cache contents use the proven actual logical page mapping.
                self.data['inputs'][key] = (dict(kind='paged_cache', metadata=dict(_meta(value), object_id=id(value)), version=tensor_version(value))
                    if key in ('key', 'value') else dict(kind='tensor', snapshot=tensor_snapshot(value,
                        self.folder/'query.bin' if key == 'query' else None)))
            else:
                self.data['inputs'][key] = dict(kind='scalar', value=scalar_snapshot(value))
        self.data['logical_kv_before'] = self._cache()
        self.data['block_table_values'] = actual['block_table'].detach().cpu().tolist()
        # Packet defaults are observations of its actual default overload schema,
        # not claims that a different internal dispatch route was selected.
        overload = getattr(original, 'default', None)
        schema = getattr(overload, '_schema', None)
        self.data['public_default_schema'] = str(schema) if schema is not None else None
        self.data['omitted_argument_defaults'] = ({arg.name: scalar_snapshot(arg.default_value)
            for arg in schema.arguments if arg.name not in actual and arg.has_default_value()}
            if schema is not None else None)
        self.data['default_schema_dispatch_observed'] = False
        self.data['state'] = 'INPUT_CAPTURED'

    def _cache(self):
        return {key: logical_cache_snapshot(self.refs[key], self.refs['pages'], self.refs['length']) for key in ('key', 'value')}

    def invoke_original(self, original, args, kwargs, *, call_once=None):
        import torch
        self._owner()
        require(self.data is not None and self.data['state'] == 'INPUT_CAPTURED' and self.calls == 0,
                'BOUNDARY_ORIGINAL_CALL_REENTRY')
        expected = self.refs['original']
        require(original is expected or (inspect.ismethod(original) and inspect.ismethod(expected)
                and original.__func__ is expected.__func__ and original.__self__ is expected.__self__),
                'BOUNDARY_ORIGINAL_CALLABLE_CHANGED')
        actual = dict(kwargs)
        actual.update(zip(('query', 'key', 'value'), args))
        require(set(actual) == set(self.refs['actual']) and all(actual[key] is value for key, value in self.refs['actual'].items()),
                'BOUNDARY_ORIGINAL_ARGUMENT_IDENTITY_CHANGED')
        self.calls += 1; self.data['original_calls'] = self.calls; self.inflight = True
        try:
            require(call_once is None or self.target_scope is not None, 'BOUNDARY_OWNED_CALL_ONCE_SCOPE_REQUIRED')
            result = original(*args, **kwargs) if call_once is None else call_once()
            self.result = result
            values = list(result) if isinstance(result, (tuple, list)) else [result]
            require(values and len(values) <= 16 and all(isinstance(value, torch.Tensor) for value in values),
                    'BOUNDARY_RETURN_STRUCTURE_UNADAPTED')
            self.synchronize()
            self.data['outputs'] = [tensor_snapshot(value, self.folder/('output-'+str(index)+'.bin'))
                                    for index, value in enumerate(values)]
            self.data['return_container_type'] = type(result).__name__
            self.data['state'] = 'ORIGINAL_RETURN_CAPTURED'
            return result  # the same tuple/list/Tensor object, no output replacement
        except BaseException as error:
            self.data['errors'].append(dict(stage='original_or_immediate_capture', error=repr(error)))
            self.data['state'] = 'FAILED'; raise
        finally:
            self.inflight = False

    def finalize(self):
        self._owner()
        require(not self.inflight and self.data is not None and self.data['state'] == 'ORIGINAL_RETURN_CAPTURED',
                'BOUNDARY_ORIGINAL_NOT_COMPLETED')
        self.synchronize(); checks = {}; after = {}
        for key, value in self.refs['actual'].items():
            before = self.data['inputs'][key]
            if before['kind'] == 'tensor':
                after[key] = tensor_snapshot(value); checks[key] = same_tensor(before['snapshot'], after[key])
            elif before['kind'] == 'paged_cache':
                checks[key] = before['metadata'] == dict(_meta(value), object_id=id(value)) and before['version'] == tensor_version(value)
            else:
                checks[key] = before['value'] == scalar_snapshot(value)
        cache = self._cache()
        for key in ('key', 'value'):
            checks['logical_'+key] = same_tensor(self.data['logical_kv_before'][key], cache[key])
            checks['impl_'+key+'_same_object'] = getattr(self.refs['module'].impl, key+'_cache') is self.refs[key]
        values = list(self.result) if isinstance(self.result, (tuple, list)) else [self.result]
        outputs = [tensor_snapshot(value) for value in values]
        for index, (before, current) in enumerate(zip(self.data['outputs'], outputs)):
            checks['output_'+str(index)] = same_tensor(before, current)
        require(len(outputs) == len(self.data['outputs']), 'BOUNDARY_RETURN_CONTAINER_CHANGED')
        self.data.update(state='FINALIZED', subsequent_checks=checks, all_checked_values_unchanged=all(checks.values()),
            inputs_after=after, logical_kv_after=cache, outputs_after=outputs,
            mutation_observation='CONTENT_AND_AVAILABLE_VERSION_AT_CHECKPOINTS; INTERVENING_WRITES_WITHOUT_VERSION_UNOBSERVED')
        self.write()
        return self.data

    def write(self):
        if self.data is not None:
            (self.folder/'boundary.json').write_text(json.dumps(self.data, indent=2, allow_nan=False)+'\n')

    def release(self):
        self._owner(); require(not self.inflight, 'BOUNDARY_INFLIGHT_RELEASE_REFUSED')
        self.synchronize()  # keep strong references if drain fails
        self.write(); self.refs = self.result = None
