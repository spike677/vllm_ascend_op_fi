"""Independently inspect completed vision preparation; never run model/runtime code.

The only subprocess is read-only image inspection in the hash-pinned layer
reader. Inputs are immutable receipts plus existing files. Output is a fresh
summary, never an amendment to an original execution receipt.
"""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import stat


MODEL = 'Qwen/Qwen3-VL-2B-Instruct'
REVISION = '89644892e4d85e24eaac8bacfd4f463576704203'
METADATA_SHA = '2fcf3612194e166fae921a35892bce9f424b6095a4edf24fc1e5ebfbf5eb7395'
WEIGHT_SHA = '7de1838c87a5349b016c26a1c3f7d2bc400a3d485f95ef39a7059ffd734977a0'
WEIGHT_BYTES = 4255140312
PROCESSOR_SHA = 'f2a857c8f215aa66ee72f9b36e331dfb137ed7a00ea0b690969faa4cff9a3987'
IMAGE_READER_SHA = '6a6747921760a819e45e43ea0c289cad12620147f4377a089757884edaa0c55a'
NAMES = ('chat_template.json', 'config.json', 'generation_config.json', 'merges.txt',
         'model.safetensors', 'preprocessor_config.json', 'tokenizer.json',
         'tokenizer_config.json', 'video_preprocessor_config.json', 'vocab.json')
PINS = {
    'vision-remote-files-verified-a1.json': '854c618c173e073df60330e09b1fdd3c3a0816adcd4e5e2e87698ac9d9590faa',
    'vision-model-files-reverified-a2.json': '2f3dea862ec5acdc23823dad83986e3bad91931d3160dbdc60d872e83be2e409',
    'vision-download-a1-receipt.json': 'b4b6f6c8fdba3823e60230cb410ee3496c747f5b8723b6bbdb730e9aac18500a',
    'qwen3vl_official_metadata.json': METADATA_SHA,
    'verify_vision_remote_a1.py': '075063b03fbee714ed014b65d6a826f44fd879f357b1a2c7f28562d9b4521081',
}


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def snapshot(value):
    return tuple(getattr(value, key) for key in
                 ('st_dev', 'st_ino', 'st_size', 'st_mtime_ns', 'st_ctime_ns'))


def stable_file(path, capture=False):
    """Hash one open regular file and bind before/after path and fd identity."""
    path = Path(path)
    before = path.lstat()
    require(stat.S_ISREG(before.st_mode), 'REGULAR_FILE_REQUIRED:' + str(path))
    sha = hashlib.sha256()
    blob = hashlib.sha1(b'blob ' + str(before.st_size).encode() + b'\0')
    data = [] if capture else None
    with path.open('rb') as stream:
        require(snapshot(os.fstat(stream.fileno())) == snapshot(before), 'OPEN_IDENTITY_CHANGED')
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            sha.update(block)
            blob.update(block)
            if data is not None:
                data.append(block)
        require(snapshot(os.fstat(stream.fileno())) == snapshot(before), 'READ_IDENTITY_CHANGED')
    require(snapshot(path.lstat()) == snapshot(before), 'PATH_IDENTITY_CHANGED')
    receipt = dict(path=str(path), bytes=before.st_size, sha256=sha.hexdigest(),
                   git_blob_sha1=blob.hexdigest(), stable_identity=list(snapshot(before)))
    return receipt, b''.join(data) if data is not None else None


def load_pinned(path, expected=None):
    receipt, data = stable_file(path, capture=True)
    require(expected is None or receipt['sha256'] == expected, 'PINNED_FILE_MISMATCH:' + str(path))
    return receipt, json.loads(data)


def exact_files(entries):
    require(isinstance(entries, list), 'FILE_LIST_REQUIRED')
    names = [item['name'] for item in entries]
    require(len(names) == len(set(names)) and set(names) == set(NAMES), 'EXACT_TEN_FILES_REQUIRED')
    return {item['name']: item for item in entries}


def check_model_inventory(metadata, actual, inventories):
    require(metadata['id'] == MODEL and metadata['sha'] == REVISION and metadata['gated'] is False,
            'OFFICIAL_MODEL_REVISION_MISMATCH')
    siblings = {item['rfilename']: item for item in metadata['siblings']}
    require(set(actual) == set(NAMES), 'ACTUAL_FILE_SET_MISMATCH')
    for name in NAMES:
        file, official = actual[name], siblings[name]
        require(file['bytes'] == official['size'], 'OFFICIAL_SIZE_MISMATCH:' + name)
        for inventory in inventories:
            require(all(file[key] == inventory[name][key] for key in ('bytes', 'sha256')),
                    'RECORDED_FILE_MISMATCH:' + name)
        if name == 'model.safetensors':
            require(file['sha256'] == official['lfs']['sha256'] == WEIGHT_SHA and
                    file['bytes'] == official['lfs']['size'] == WEIGHT_BYTES,
                    'OFFICIAL_LFS_WEIGHT_MISMATCH')
        else:
            require(file['git_blob_sha1'] == official['blobId'], 'OFFICIAL_GIT_BLOB_MISMATCH:' + name)


def check_cpu_command(controller, root):
    command = controller['command']
    require(command[:2] == ['docker', 'run'], 'RECORDED_DOCKER_RUN_REQUIRED')
    require('--device' not in command and not any(v.startswith('--device=') for v in command) and
            '--privileged' not in command and controller['npu_devices'] == [], 'DEVICE_ISOLATION_REQUIRED')
    require(command[command.index('--runtime') + 1] == 'runc' and
            command[command.index('--network') + 1] == 'none', 'CPU_NETWORK_ISOLATION_REQUIRED')
    require('ASCEND_VISIBLE_DEVICES=void' in command and 'TORCH_DEVICE_BACKEND_AUTOLOAD=0' in command,
            'CPU_ENVIRONMENT_REQUIRED')
    require(controller['exit_code'] == 0 and controller['model_constructed'] is False and
            command[command.index('--entrypoint') + 1] == 'python3' and
            command[command.index('--entrypoint') + 2] == controller['image_id'], 'CPU_EXECUTION_MISMATCH')
    mounts = [command[i + 1] for i, item in enumerate(command) if item == '--mount']
    expected = [
        f'type=bind,src={root}/source-a33,dst=/source,readonly',
        f'type=bind,src={root}/models/Qwen3-VL-2B-Instruct,dst=/model,readonly',
        f'type=bind,src={root}/qwen3vl_official_metadata.json,dst=/metadata.json,readonly',
        f'type=bind,src={root}/source-a33/experiments/c2c3_closure_20261117/vision17_processor_cpu.py,dst=/probe.py,readonly',
        'type=bind,src=/home/jw/Wan2.2/examples/i2v_input.JPG,dst=/image.JPG,readonly',
        'type=bind,src=/usr/local/Ascend/driver,dst=/driver,readonly',
        f'type=bind,src={root}/vision17-processor-cpu-a3,dst=/out',
    ]
    require(mounts == expected, 'UNREVIEWED_CPU_MOUNT_SET')
    return dict(runtime='runc', network='none', device_mappings=[], readonly_mounts=6,
                exit_code=controller['exit_code'], explicit_npu_compute_invocations=0,
                runtime_api_zero_claim_supported=False)


def inspect_runtime_log(log, processor):
    markers = ('halGetDeviceInfo failed', 'rtGetDeviceCount', 'aclrtGetDeviceCountImpl',
               "Can't get ascend_hal device count")
    observed = [marker for marker in markers if marker in log]
    require(len(observed) == len(markers), 'EXPECTED_RUNTIME_DISCOVERY_EVIDENCE_MISSING')
    terminal = [json.loads(line) for line in log.splitlines() if line.startswith('{')]
    require(len(terminal) == 1 and terminal[0]['outputs'] == processor['outputs'] and
            terminal[0]['status'] == processor['status'], 'LOG_RECEIPT_OUTPUT_MISMATCH')
    return dict(runtime_discovery_observed=True, discovery_markers=observed,
                original_actual_npu_calls=processor['actual_npu_calls'],
                actual_npu_calls_field_is_not_a_runtime_trace=True,
                zero_runtime_api_calls_supported=False,
                explicit_npu_compute_invocations=0,
                explicit_compute_basis='Reviewed frozen processor source and device-isolated controller, not an API trace')


def audit(root, image_reader_path):
    artifacts, loaded = {}, {}
    for name, expected in PINS.items():
        if name.endswith('.json'):
            artifacts[name], loaded[name] = load_pinned(root / name, expected)
        else:
            artifacts[name], _ = stable_file(root / name)
            require(artifacts[name]['sha256'] == expected, 'VERIFIER_SOURCE_CHANGED')
    metadata = loaded['qwen3vl_official_metadata.json']
    receipts = [loaded[name] for name in PINS if name.startswith('vision-')]
    inventories = []
    for receipt in receipts:
        require(receipt['repository'] == MODEL and receipt['revision'] == REVISION and
                receipt['official_metadata_sha256'] == METADATA_SHA and
                receipt['model_qualification'] is False, 'MODEL_RECEIPT_SCOPE_MISMATCH')
        inventories.append(exact_files(receipt['files']))
    model = root / 'models/Qwen3-VL-2B-Instruct'
    actual = {name: stable_file(model / name)[0] for name in NAMES}
    check_model_inventory(metadata, actual, inventories)
    remote = loaded['vision-remote-files-verified-a1.json']
    require(sum(item['bytes'] for item in actual.values()) == remote['total_bytes'] == 4266640306,
            'TOTAL_MODEL_BYTES_MISMATCH')
    require(remote['verification_script_sha256'] == PINS['verify_vision_remote_a1.py'] and
            remote['local_reverification_receipt_sha256'] == PINS['vision-model-files-reverified-a2.json'],
            'VERIFICATION_CHAIN_MISMATCH')
    weight = model / 'model.safetensors'
    upload = model / 'model.safetensors.upload-a1'
    require(snapshot(weight.lstat()) == tuple(actual['model.safetensors']['stable_identity']) and
            snapshot(upload.lstat()) == snapshot(weight.lstat()), 'FINAL_UPLOAD_VERIFIED_INODE_MISMATCH')
    with weight.open('rb') as stream:
        require(snapshot(os.fstat(stream.fileno())) == snapshot(weight.lstat()), 'HEADER_OPEN_IDENTITY_CHANGED')
        size = int.from_bytes(stream.read(8), 'little')
        require(0 < size <= 16 * 1024 * 1024, 'BOUNDED_HEADER_REQUIRED')
        header = json.loads(stream.read(size))
    patch_headers = {key: {field: header[key][field] for field in ('dtype', 'shape')}
                     for key in remote['patch_parameter_headers']}
    require(patch_headers == remote['patch_parameter_headers'] and
            len([key for key in header if key != '__metadata__']) == remote['tensor_header_count'] == 625,
            'SAFETENSORS_HEADER_MISMATCH')
    helper_record, _ = stable_file(image_reader_path)
    require(helper_record['sha256'] == IMAGE_READER_SHA, 'IMAGE_READER_CHANGED')
    spec = importlib.util.spec_from_file_location('read_only_image_reader', image_reader_path)
    reader = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reader)
    source = root / 'source-a33'
    source_manifest_record, source_manifest = load_pinned(source / 'source_manifest.json')
    source_check = reader.check_files(source, source_manifest['files'])
    require(not source_check['mismatches'], 'FROZEN_SOURCE_FILES_CHANGED')
    folder = root / 'vision17-processor-cpu-a3'
    controller_record, controller = load_pinned(folder / 'controller.json')
    processor_record, processor = load_pinned(folder / 'processor.json')
    log_record, log_bytes = stable_file(folder / 'process.log', capture=True)
    command_review = check_cpu_command(controller, root)
    script_record, script_bytes = stable_file(source / 'experiments/c2c3_closure_20261117/vision17_processor_cpu.py', capture=True)
    require(script_record['sha256'] == controller['script_sha256'] == PROCESSOR_SHA,
            'ACTUAL_EXECUTED_PROCESSOR_SOURCE_MISMATCH')
    image = reader.image_metadata(controller['image_id'])
    package = '/usr/local/python3.11.14/lib/python3.11/site-packages/transformers/'
    image_sources = {}
    for filename, field in (('modeling_qwen3_vl.py', 'model_source_sha256'),
                            ('processing_qwen3_vl.py', 'processor_source_sha256')):
        image_sources[filename] = reader.image_file(image, package + 'models/qwen3_vl/' + filename)
        require(image_sources[filename]['sha256'] == processor[field], 'IMAGE_SOURCE_MISMATCH:' + filename)
    image_sources['__init__.py'] = reader.image_file(image, package + '__init__.py')
    require('__version__ = "4.57.6"' in Path(image_sources['__init__.py']['path']).read_text() and
            processor['transformers'] == '4.57.6', 'TRANSFORMERS_VERSION_MISMATCH')
    input_image, _ = stable_file('/home/jw/Wan2.2/examples/i2v_input.JPG')
    require(input_image['sha256'] == processor['image_sha256'], 'PROCESSOR_IMAGE_MISMATCH')
    driver, _ = stable_file('/usr/local/Ascend/driver/lib64/driver/libascend_hal.so')
    require(driver['sha256'] == controller['driver_hal_sha256'], 'DRIVER_LIBRARY_MISMATCH')
    require(processor['config_files'] == {name: actual[name]['sha256'] for name in NAMES if name != 'model.safetensors'},
            'PROCESSOR_CONFIGURATION_MISMATCH')
    require(all(processor[key] is False for key in ('model_constructed', 'weights_loaded', 'raw_tensors_saved', 'c2', 'c3')),
            'CPU_PROCESSOR_SCOPE_MISMATCH')
    require(processor['processor_overrides'] == {}, 'UNREVIEWED_PROCESSOR_OVERRIDES')
    runtime = inspect_runtime_log(log_bytes.decode(), processor)
    failures = {}
    for name in ('vision17-processor-cpu-a1', 'vision17-processor-cpu-a2'):
        crecord, value = load_pinned(root / name / 'controller.json')
        lrecord, data = stable_file(root / name / 'process.log', capture=True)
        require(value['exit_code'] == 1 and 'libascend_hal.so: cannot open shared object file' in data.decode(),
                'HISTORICAL_FAILURE_CHANGED')
        failures[name] = dict(controller=crecord, log=lrecord, exit_code=1,
                              executed_script_sha256=value['script_sha256'],
                              cause='torch_npu import could not load libascend_hal.so', preserved=True)
    # Re-read small immutable originals and final inode after the audit. No
    # launch-time identity is inferred for the original download helper.
    for name, expected in PINS.items():
        require(stable_file(root / name)[0]['sha256'] == expected, 'ORIGINAL_CHANGED_DURING_AUDIT')
    require(snapshot(weight.lstat()) == tuple(actual['model.safetensors']['stable_identity']),
            'WEIGHT_CHANGED_AFTER_VERIFICATION')
    return dict(schema='demoncat.agent7.vision_preparation_review.v1',
        status='INDEPENDENT_FILE_AND_CPU_PREPARATION_REVIEW_COMPLETE', checks_passed=True,
        reviewer_source=stable_file(__file__)[0], image_reader=helper_record,
        artifacts=artifacts, model=dict(repository=MODEL, revision=REVISION, file_count=len(actual),
            total_bytes=sum(item['bytes'] for item in actual.values()), files=actual,
            final_and_upload_same_verified_inode=True, header_tensor_count=625,
            patch_parameter_headers=patch_headers, model_inference_qualified=False),
        frozen_source=dict(manifest=source_manifest_record, commit=source_manifest['base_commit'], **source_check),
        cpu_processor=dict(controller=controller_record, receipt=processor_record, log=log_record,
            executed_script=script_record, command_review=command_review, image_id=image['Id'],
            image_sources=image_sources, driver_library=driver, input_image=input_image,
            outputs=processor['outputs'], geometry=processor['geometry'], runtime_discovery=runtime,
            model_constructed=False, weights_loaded=False, processor_rerun_by_auditor=False),
        historical_cpu_failures=failures,
        identity_limits=dict(original_downloader_launch_script_identity_proven=False,
            original_downloader_hash_semantics='Recorded completion-time helper hash; not a launch-time identity',
            executed_processor_is_source_a33=True, current_future_script_not_substituted=True),
        qualification=dict(model_inference=False, real_npu_fault_execution=False,
            standard_worker=False, b_f_off=False, statistics_mix_e5=False, c2=False, c3=False),
        audit_runtime=dict(npu_calls=0, containers_created=0, model_imports=0,
                           subprocess_allowlist=['docker image inspect']))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--image-reader', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    require(not args.output.exists(), 'FRESH_AUDIT_OUTPUT_REQUIRED')
    result = audit(args.root, args.image_reader)
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps(dict(status=result['status'], checks_passed=result['checks_passed'],
                          file_count=result['model']['file_count'], output=str(args.output))))


if __name__ == '__main__':
    main()
