"""Experimental native TND/page16 adapter, separate from historical BNSD IFA.

The original Worker API, live cache, mask, page table and return object are
preserved. Only the fresh source-built native host selector changes S2. This
module is not registered as a qualified common Provider: every new artifact
must first pass exact-image ABI, stock/OFF, final arguments and recovery gates.
"""
import hashlib
import json
import math
import os
from pathlib import Path
import threading
import uuid

from .paged33_audit import Paged33Audit, _meta
from .serving_context import require
from fault_injection.native_adapters.attention28.launch import LaunchObserver, verify_launch, sha, bundle_kernel_abi
from fault_injection.native_adapters.attention28.fia.fai_fd_geometry import KEYS as FD_KEYS, VARIANT as FD_VARIANT, validate_fd_geometry

COMMIT = 'c77c0829f1f0f0b8f882db1d9b051a758b605472'
KEYS = set(FD_KEYS)
PREFIX = 'DCAT33_TND_'


def verify_bundle(completion, completion_sha256, vendor):
    path, root = Path(completion).resolve(strict=True), Path(vendor).resolve(strict=True)
    require(sha(path) == completion_sha256, 'TND33_COMPLETION_HASH_DIFFERS')
    data = json.loads(path.read_bytes())
    require(data.get('schema') == 'demoncat.fia.fresh_build.v1'
            and data.get('status') == 'CPU_BUILD_ONLY_REQUIRES_NATIVE_QUALIFICATION'
            and data.get('source_commit') == COMMIT and data.get('variant') == FD_VARIANT
            and data.get('loading_allowed') is False and data.get('npu_executed') is False
            and data.get('previous_experiment_objects_used') is False
            and len(data.get('compiled_keys', [])) == len(KEYS)
            and set(data.get('compiled_keys', [])) == KEYS, 'TND33_INDEPENDENT_FRESH_BUILD_REQUIRED')
    files = {}
    for row in data['artifacts']:
        relative = Path(row['path'])
        require(not relative.is_absolute() and '..' not in relative.parts and row['path'] not in files,
                'TND33_ARTIFACT_PATH_INVALID')
        target = root/relative
        require(target.resolve(strict=True).is_relative_to(root), 'TND33_ARTIFACT_ESCAPES_VENDOR')
        if target.is_symlink():
            require(row['path'] == 'op_impl/ai_core/tbe/op_tiling/liboptiling.so'
                    and os.readlink(target) == 'lib/linux/aarch64/libcust_opmaster_rt2.0.so',
                    'TND33_UNREVIEWED_ARTIFACT_LINK')
        require(target.is_file() and sha(target) == row['sha256'] and target.stat().st_size == row['bytes'],
                'TND33_FRESH_ARTIFACT_BYTES_DIFFER')
        files[row['path']] = row
    observed = set()
    for target in root.rglob('*'):
        require(not (target.is_symlink() and target.is_dir()), 'TND33_DIRECTORY_LINK_REFUSED')
        if target.is_file():
            observed.add(target.relative_to(root).as_posix())
    require(observed == set(files), 'TND33_VENDOR_HAS_MISSING_OR_EXTRA_FILES')
    kernels = [row for row in files.values() if row['path'].endswith('.o')]
    require(len(kernels) == 1 and kernels[0]['sha256'] == data['kernel_binary_sha256'],
            'TND33_MATCHING_NEW_KERNEL_REQUIRED')
    from fault_injection.native_adapters.attention28.fia import fai_fd_geometry
    require(data.get('fai_fd_geometry_validator_sha256') == sha(fai_fd_geometry.__file__)
            and data.get('actual_worker_platform_observed') is False,
            'TND33_FRESH_FD_CPU_CONTRACT_REQUIRED')
    cpu_dir = path.parent/'fai_schema'
    require(sha(cpu_dir/'compile_receipt.json') == data.get('fai_cpu_geometry_receipt_sha256')
            and sha(cpu_dir/'fai_fd_geometry_matrix.json') == data.get('fai_fd_cpu_matrix_sha256'),
            'TND33_FRESH_FD_CPU_EVIDENCE_DIFFERS')
    cpu = json.loads((cpu_dir/'compile_receipt.json').read_bytes())['tools']['fai_host_geometry']['result']
    require(cpu.get('status') == 'PASS_CPU_ACTUAL_FAI_FD_COVERAGE_AND_REDUCTION'
            and cpu.get('cases') == 24576 and cpu.get('all_public_arrays_checked') is True
            and cpu.get('actual_worker_aic_observed') is False
            and cpu.get('matrix_sha256') == data['fai_fd_cpu_matrix_sha256'],
            'TND33_FRESH_FD_CPU_MATRIX_INCOMPLETE')
    return dict(completion=data, completion_path=str(path), completion_sha256=completion_sha256,
                vendor=str(root), artifacts=files)


def validate_tnd_call(actual, boundary, *, allow_cpu_test=False):
    """Narrow native selection scope plus the independent live-cache audit."""
    require(boundary.get('metadata_binding') == 'VERIFIED'
            and boundary.get('actual_operator') == 'torch_npu.npu_fused_infer_attention_score'
            and boundary.get('layout') == 'TND', 'TND33_BOUND_LIVE_WORKER_CALL_REQUIRED')
    require(boundary.get('qualification_scope') == ('CPU_CONTRACT_ONLY' if allow_cpu_test else 'NATURAL_CALL_METADATA_ONLY'),
            'TND33_AUDIT_SCOPE_DIFFERS')
    require(actual.get('input_layout') == 'TND' and actual.get('block_size') == 16
            and actual.get('sparse_mode') == 3, 'TND33_API_LAYOUT_PAGE_OR_SPARSE_DIFFERS')
    require(actual.get('num_heads') == 16 and actual.get('num_key_value_heads') == 8
            and math.isclose(float(actual.get('scale', 0)), 128**-.5, rel_tol=0, abs_tol=1e-12)
            and actual.get('inner_precise', 0) == 0, 'TND33_HEAD_SCALE_PRECISION_DIFFERS')
    require(list(actual['actual_seq_lengths']) == [1] and len(actual['actual_seq_lengths_kv']) == 1
            and 1025 <= actual['actual_seq_lengths_kv'][0] <= 2048, 'TND33_LONG_SINGLE_TOKEN_DECODE_REQUIRED')
    require(list(actual['actual_seq_lengths_kv']) == boundary['sequence_lengths'], 'TND33_LENGTHS_CHANGED_AFTER_AUDIT')
    query, key, value = (actual[name] for name in ('query', 'key', 'value'))
    require(list(query.shape) == [1, 16, 128] and query.is_contiguous(), 'TND33_QUERY_SHAPE_OR_STRIDE_DIFFERS')
    for name, tensor in (('query', query), ('key', key), ('value', value), ('page_table', actual['block_table'])):
        meta = _meta(tensor)
        require(all(meta[field] == boundary[name][field] for field in
                    ('storage_identity', 'storage_offset', 'shape', 'stride', 'dtype', 'device')),
                'TND33_ACTUAL_STORAGE_CHANGED_AFTER_AUDIT')
    require(str(query.dtype) == 'torch.float16' and
            (str(query.device).startswith('npu') or (allow_cpu_test and str(query.device) == 'cpu')),
            'TND33_NPU_FP16_REQUIRED')
    require(list(key.shape) == list(value.shape) and len(key.shape) == 3
            and list(key.shape)[1:] == [16, 1024], 'TND33_NATURAL_PAGE_VIEW_REQUIRED')
    # Optional feature inputs cannot be silently ignored by a new native route.
    for name in ('pse_shift', 'dequant_scale1', 'quant_scale1', 'dequant_scale2', 'quant_scale2',
                 'quant_offset2', 'antiquant_scale', 'antiquant_offset', 'key_antiquant_scale',
                 'key_antiquant_offset', 'value_antiquant_scale', 'value_antiquant_offset',
                 'key_shared_prefix', 'value_shared_prefix', 'query_rope', 'key_rope'):
        require(actual.get(name) is None, 'TND33_OPTIONAL_FEATURE_UNADAPTED:' + name)
    return dict(api='torch_npu.npu_fused_infer_attention_score', layout='TND', page_size=16,
                kv_length=actual['actual_seq_lengths_kv'][0], native_s2=512, fault_s2=256,
                native_route='TilingProcess4SplitFuse / FAInferTiling')


def validate_fai_launch(launch, host, *, arm, kv_length):
    """Check the *final observed* FAI ABI against matched kernel specialization.

    FAI encodes its KV tile in the compiled entry, not a mutable s2BaseSize
    field. The observed function key, fresh ELF, and dependent workspace slices
    must all agree. Public C2/C3 qualification remains separate.
    """
    if host.get('schema') == 'demoncat.paged33.fai_fd_host.v1':
        return validate_fd_geometry(launch['typed'], host, key=launch['tiling_key'],
            block_dim=launch['block_dim'], schedule_mode=launch['schedule_mode'], arm=arm, kv_length=kv_length)
    # Kept solely for explicit old non-FD CPU regressions. A live adapter below
    # requires the new FD build and FD host schema before this function runs.
    tile = 256 if arm == 'fault' else 512
    typed = launch['typed']
    require(host['route'] == 'FAInferTiling' and host['aic'] == 16
            and host['native_route_or_decode_flags_changed'] is False,
            'TND33_ACTUAL_FAI_ROUTE_OR_CORE_BUDGET_DIFFERS')
    require(all(typed[name] == value for name, value in dict(numHeads=16, kvHeads=8, batch=1,
        embeddingSize=128, embeddingSizeV=128, blockSize=16, maxQSeqlen=1,
        maxKvSeqlen=kv_length, sparseMode=3, firstBatchTaskNum=8, totalTaskNum=8,
        splitLseTotalSize=0, splitOTotalSize=0, totalSplitNodeNum=0).items()),
        'TND33_FINAL_FAI_GEOMETRY_DIFFERS')
    mask = typed['maskType']
    require(mask in (0, 1), 'TND33_FAI_MASK_UNADAPTED')
    expected_key = 5000000000010200100+(3 if mask == 1 else 0)+(900000000000000000 if arm == 'fault' else 0)
    require(launch['tiling_key'] == expected_key and launch['block_dim'] == 16 and launch['schedule_mode'] == 1,
            'TND33_FINAL_FAI_ENTRY_OR_LAUNCH_DIFFERS')
    unit = 16*128*tile*3
    require(all(typed[name] == unit*width for name, width in dict(mm1OutSize=4, smOnlineOutSize=2,
        mm2OutSize=4, UpdateSize=4, workSpaceSize=14).items())
        and host['workspace'] == 16*1024**2+16*128*tile*4*3*4,
        'TND33_FINAL_FAI_WORKSPACE_DIFFERS')
    return dict(selected_s2=tile, selection='EXACT_FRESH_COMPILED_ENTRY_AND_FINAL_WORKSPACE',
                route='FAInferTiling', host_device_geometry_agrees=True)


class TNDNativeAdapter:
    def __init__(self, *, completion, completion_sha256, vendor, schema, schema_sha256,
                 observer_receipt, observer_receipt_sha256, evidence_dir, arm,
                 experimental_qualification=False, max_calls=224):
        require(experimental_qualification is True, 'TND33_EXPLICIT_EXPERIMENTAL_QUALIFICATION_REQUIRED')
        require(arm in ('baseline', 'fault', 'off') and type(max_calls) is int and 1 <= max_calls <= 224,
                'TND33_BOUNDED_ARM_REQUIRED')
        self.build = verify_bundle(completion, completion_sha256, vendor)
        self.kernel_metadata, self.kernel_abi = bundle_kernel_abi(self.build['vendor'], self.build['artifacts'],
            self.build['completion']['kernel_binary_sha256'])
        from .paged33_loading import verify_runtime
        self.library_coexistence = verify_runtime(self.build)
        require(os.environ.get('ACLNN_CACHE_LIMIT') == '0' and os.environ.get('TASK_QUEUE_ENABLE') == '0',
                'TND33_PREIMPORT_PRIVATE_VENDOR_UNCACHED_SYNC_REQUIRED')
        require(sha(schema) == schema_sha256 == self.build['completion'].get('fai_public_schema_sha256'),
                'TND33_PUBLIC_SCHEMA_HASH_DIFFERS')
        self.schema = json.loads(Path(schema).read_bytes())
        require(self.schema.get('schema') == 'demoncat.paged33.fai_public_serializer.v1'
                and self.schema.get('type') == 'FAInferTilingData', 'TND33_ACTUAL_FAI_PUBLIC_SCHEMA_REQUIRED')
        require(sha(observer_receipt) == observer_receipt_sha256, 'TND33_OBSERVER_RECEIPT_HASH_DIFFERS')
        receipt = json.loads(Path(observer_receipt).read_bytes())
        require(receipt.get('observer_compiled') is True and receipt.get('actual_npu_calls') == 0,
                'TND33_EXACT_IMAGE_OBSERVER_BUILD_REQUIRED')
        require(receipt.get('headers') and all(Path(x['path']).is_file() and sha(x['path']) == x['sha256']
                for x in receipt['headers']), 'TND33_RUNTIME_SDK_DIFFERS_FROM_OBSERVER_BUILD')
        # The source builder's own reviewed SDK hashes remain required. A 9.1
        # vendor cannot be admitted into 8.5 merely because the observer compiles.
        from fault_injection.native_adapters.attention28 import launch
        contract = Path(launch.__file__).with_name('fia')/'build_contract.json'
        contract_data = json.loads(contract.read_bytes())
        require(sha(contract) == self.build['completion']['contract_sha256'], 'TND33_BUILD_CONTRACT_DIFFERS')
        sdk = Path(receipt['sdk'])
        interfaces = self.build['completion'].get('sdk_interface_files')
        require(interfaces and {x['path'] for x in interfaces} ==
                {x['path'] for x in contract_data['files'] if x['kind'] == 'sdk'}, 'TND33_BUILD_SDK_INTERFACE_SET_MISSING')
        require(all((sdk/x['path']).is_file() and sha(sdk/x['path']) == x['sha256']
                    for x in interfaces), 'TND33_FRESH_VENDOR_SDK_ABI_DIFFERS')
        observer = receipt['observer_library']
        self.observer = LaunchObserver(observer['path'], observer['sha256'])
        self.output = Path(evidence_dir).resolve()
        require(str(self.output).startswith('/evidence/') and not self.output.exists(), 'TND33_FRESH_EVIDENCE_REQUIRED')
        self.output.mkdir(parents=True)
        self.runtime_dir = Path(os.environ['DCAT_ATTN_OBSERVER_DIR'])
        self.arm, self.max_calls, self.owner = arm, max_calls, threading.get_ident()
        self.active, self.enabled, self.calls, self.failures = False, True, [], []

    def execute(self, actual, boundary, invoke_once):
        import torch
        from .paged33_loading import validate_environment
        validate_environment(self.build['vendor'],platform_ready=True)
        require(threading.get_ident() == self.owner and self.enabled and not self.active, 'TND33_SCOPE_OWNER_REENTRY_OR_DISABLED')
        require(len(self.calls) + len(self.failures) < self.max_calls, 'TND33_SESSION_BUDGET_REACHED')
        shape = validate_tnd_call(actual, boundary)
        require(not any(name.startswith(PREFIX) for name in os.environ), 'TND33_FOREIGN_NATIVE_SCOPE')
        call_id, log = uuid.uuid4().hex, self.output/'host.jsonl'
        mode = 's2_256' if self.arm == 'fault' else 'off'
        env = dict(DCAT33_TND_MODE=mode, DCAT33_TND_TOKEN=call_id,
                   DCAT33_TND_PID=str(os.getpid()), DCAT33_TND_TID=str(threading.get_native_id()),
                   DCAT33_TND_LOG=str(log))
        self.active = True
        try:
            torch.npu.synchronize()
            os.environ.update(env)
            with self.observer.scope(call_id):
                result = invoke_once()
                torch.npu.synchronize()
                self.observer.synchronize()
            rows = [json.loads(line) for line in log.read_text().splitlines()]
            matches = [row for row in rows if row.get('token') == call_id]
            require(len(matches) == 1, 'TND33_EXACT_ONE_NATIVE_HOST_CALLBACK_REQUIRED')
            host = matches[0]
            require(host.get('schema') == 'demoncat.paged33.fai_fd_host.v1'
                    and host['eligible'] is True and host['policy_applied'] == (self.arm == 'fault')
                    and host['mode'] == mode and host['native_s2'] == 512
                    and host['selected_s2'] == (256 if self.arm == 'fault' else 512),
                    'TND33_ACTUAL_HOST_STRATEGY_DIFFERS')
            launch = verify_launch(self.runtime_dir, call_id, schema=self.schema, host=host,
                                   kernel_sha256=self.build['completion']['kernel_binary_sha256'],
                                   kernel_metadata=self.kernel_metadata, kernel_abi=self.kernel_abi)
            require(launch['typed']['numBlocks'] == int(actual['key'].shape[0])
                    and launch['typed']['maxNumBlocksPerBatch'] == int(actual['block_table'].shape[1])
                    and launch['typed']['sparseMode'] == actual['sparse_mode']
                    and launch['typed']['maskType'] == 1, 'TND33_FINAL_NATIVE_API_BINDING_DIFFERS')
            geometry = validate_fai_launch(launch, host, arm=self.arm, kv_length=shape['kv_length'])
            row = dict(call_id=call_id, module_path=boundary['module_path'],
                       module_object_id=boundary['module_object_id'], virtual_engine=boundary['virtual_engine'],
                       scheduler=boundary['scheduler'], native=launch, scope=shape, geometry=geometry,
                       cache_repacked=False, api_arguments_changed=False, output_mutated=False)
            self.calls.append(row)
            (self.output/(call_id+'.json')).write_text(json.dumps(row, indent=2)+'\n')
            return result
        except BaseException as error:
            self.failures.append(dict(call_id=call_id, error=repr(error)))
            self.enabled = False
            raise
        finally:
            mismatches = [name for name, value in env.items() if os.environ.get(name) not in (None, value)]
            for name, value in env.items():
                if os.environ.get(name) == value:
                    del os.environ[name]
            self.active = False
            require(not mismatches, 'TND33_SCOPE_ENV_OWNERSHIP_CHANGED')

    def disable(self):
        require(threading.get_ident() == self.owner and not self.active, 'TND33_ACTIVE_OR_FOREIGN_DISABLE')
        import torch
        self.enabled = False
        torch.npu.synchronize()
        require(not any(name.startswith(PREFIX) for name in os.environ), 'TND33_NATIVE_SCOPE_LEAK')
        return dict(scope_cleared=True, local_streams_drained=True, calls=len(self.calls),
                    library_coexistence=self.library_coexistence,
                    failures=self.failures, public_worker_qualified=False, c2=False, c3=False,
                    group_recovery='NOT_ESTABLISHED_BY_SERIAL_ADAPTER')


class Paged33TNDProvider(Paged33Audit):
    """Reuse independent natural boundary checks; do not replace runner methods."""
    def __init__(self, worker, *, adapter):
        super().__init__(worker, max_seqs=1)
        self.adapter = adapter

    def _invoke_api(self, name, original, args, kwargs, record):
        if record is None:
            return original(*args, **kwargs)
        require(name == 'npu_fused_infer_attention_score' and record.get('metadata_binding') == 'VERIFIED',
                'TND33_UNADAPTED_ACTUAL_API_REFUSED')
        actual = dict(kwargs)
        for key, value in zip(('query', 'key', 'value'), args):
            actual[key] = value
        return self.adapter.execute(actual, record, lambda: original(*args, **kwargs))

    def close(self):
        state = self.adapter.disable()
        super().close()
        return state
