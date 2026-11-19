"""Write one reviewed case for root's existing queue; never launch anything."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import re
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.worker_core_v11_20261118 import probe_quant_ordinary as probe
from experiments.c2c3_closure_20261117.queue_contract import validate_cases

IMAGE = 'sha256:35f9e569dbe3dc8db90cb361886d7889489efd8fa065860c8de319e13ab025d0'
MODEL = '/home/17-fault-c2c3-20261117/product-qwen06-w8a8-cpu-a4/model'


def adapt(base, *, source_root, source_commit, source_sha, model_sha, build_sha):
    case = deepcopy(base)
    if (case['image'] != IMAGE or not re.fullmatch('[0-9a-f]{40}', source_commit)
            or any(not re.fullmatch('[0-9a-f]{64}', value) for value in (source_sha, model_sha, build_sha))
            or not str(source_root).startswith('/') or any(value in str(source_root) for value in ('\n', '\r', '\0', ','))):
        raise ValueError('EXACT_REVIEWED_IMAGE_SOURCE_MODEL_AND_BUILD_REQUIRED')
    name = 'quant-qwen06-ordinary-six-batches'
    case['id'], case['timeout'] = name, 900
    case['mounts'].append([str(source_root), '/source', 'ro'])
    # Preserve the already used startup vendor/stock loading order. No trace
    # journal, new DSO, late runtime preload or additional Native build.
    bootstrap = case['command'].split('exec python3 /probe.py', 1)[0]
    if (case['command'].count('exec python3 /probe.py') != 1
            or 'ASCEND_CUSTOM_OPP_PATH=/quant/runtime/vendor:' not in bootstrap
            or bootstrap.count('export PYTHONPATH=/installed:/msprobe_packages:$PYTHONPATH') != 1):
        raise ValueError('REVIEWED_QUANT_STARTUP_ORDER_REQUIRED')
    bootstrap = bootstrap.replace('export PYTHONPATH=/installed:/msprobe_packages:$PYTHONPATH',
                                  'export PYTHONPATH=/installed:/source:$PYTHONPATH')
    case['command'] = (bootstrap + 'exec python3 /probe.py --build-sha256 ' + build_sha
        + ' --source-commit ' + source_commit + ' --source-manifest-sha256 ' + source_sha + ' --model-sha256 ' + model_sha)
    case['review'].update(experiment_id=name,
        provider='quant.batch_shared_scale / public Host+Runtime / ordinary automatic single projection; blue-zone acceptance only',
        worker='standard TP1 Qwen dynamic W8A8, B8 first decode; six matched batches plus two healthy batches; no main-project claim',
        expected_hits=1, build_fingerprint=build_sha, source_commit=source_commit, source_manifest_sha256=source_sha,
        model_artifact_sha256=model_sha, actual_runtime_image_id=IMAGE, main_project_integration_verified=False,
        recovery_plan='Use the unchanged reviewed Native ABI: normal, reference, lossless, Native calibration OFF, one F, real OFF. '
            'Every batch is normally enqueued by public Host/Runtime; no RPC model forward. Selected actual input/Q/scale/native '
            'formals and real QMM consumer must match. Require BOTH Q and scale effects, current-input native correct-domain '
            'helper before effect publication, stop/synchronize/drain/recover/remove, same-Worker OFF and healthy outputs. '
            'First failure stays failure. On failed F, common recover may retain hooks for one pass-only actual OFF; it never '
            'repeats F or resets attempted. Unknown owners/pending/failed drains retain ownership. No 135 census, route/GMM '
            'extension, msprobe or tensor dump. Restart/container removal is not recovery. Root alone applies live queue gates.')
    # Preserve the actual bundle's metadata-static allocation from the reviewed
    # generator, even though this short driver does not run a complete census.
    budget = case['review']['resource_budget']
    budget.update(timeout_seconds=900, memory_gib=32, cpus=8,
        protocol_batches=6, health_batches=2, first_decode_requests_per_batch=8, fault_model_calls=1,
        fault_native_helpers=1, admitted_model_calls=4, max_owned_device_bytes=64*1024*1024,
        max_host_reference_bytes=128*1024*1024, complete_readback_bytes_per_tensor=16*1024*1024,
        native_events=32768, row_operator_calls_at_selected_prefix=256,
        work_seconds=660, recovery_and_final_evidence_deadline_seconds=840)
    validate_cases([case], case['review']['physical_cards'])
    return case


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('source-root', 'source-commit', 'installed', 'quant', 'expected-build-sha256'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--model', default=MODEL)
    parser.add_argument('--card', type=int, choices=(2, 5), default=5)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args(argv)
    source = Path(args.source_root).resolve(strict=True)
    manifest = source / 'source_manifest.json'
    if json.loads(manifest.read_bytes()).get('base_commit') != args.source_commit:
        raise ValueError('ACTUAL_FROZEN_SOURCE_COMMIT_REQUIRED')
    probe_path = source / 'experiments/worker_core_v11_20261118/probe_quant_ordinary.py'
    if probe.sha_file(probe_path) != probe.sha_file(probe.__file__):
        raise ValueError('GENERATOR_AND_FROZEN_PROBE_SOURCE_DIFFER')
    probe.installed_source_pins(source, Path(args.installed))
    model = probe.model_identity(args.model)
    build_path = Path(args.quant) / 'build.json'
    build_sha = probe.sha_file(build_path)
    if build_sha != args.expected_build_sha256:
        raise ValueError('EXPLICIT_REVIEWED_NATIVE_BUILD_SHA_REQUIRED')
    build = json.loads(build_path.read_bytes())
    # These are the exact production files checked again by the installed
    # low-level binding. Native compile identity remains separate from Worker source.
    from fault_injection.quant_group_native import PRODUCTION_SOURCES
    for name in PRODUCTION_SOURCES:
        relative = 'fault_injection/native_adapters/' + name
        if build['sources'].get(relative) != probe.sha_file(source / relative):
            raise ValueError('CURRENT_NATIVE_PRODUCTION_SOURCE_DIFFERS:' + name)
    with tempfile.TemporaryDirectory(prefix='ordinary-quant-case-') as temporary:
        output = Path(temporary) / 'base.json'
        subprocess.run([sys.executable, str(source / 'experiments/c2c3_closure_20261117/providers_quant_prepare_inventory_case.py'),
            '--installed', args.installed, '--quant', args.quant, '--probe', str(probe_path), '--source-commit', args.source_commit,
            '--image-id', IMAGE, '--model', args.model, '--card', str(args.card), '--group-abi', '--out', str(output)], check=True)
        base = json.loads(output.read_bytes())
    if len(base) != 1:
        raise ValueError('ONE_REVIEWED_BASE_CASE_REQUIRED')
    case = adapt(base[0], source_root=source, source_commit=args.source_commit, source_sha=probe.sha_file(manifest),
                 model_sha=model['sha256'], build_sha=build_sha)
    case['review']['model_identity'] = model
    case['review']['native_compile_source_commit'] = build['source_commit']
    with args.out.open('x') as stream:
        json.dump([case], stream, indent=2, allow_nan=False)
        stream.write('\n')


if __name__ == '__main__':
    main()
