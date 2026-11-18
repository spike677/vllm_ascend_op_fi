"""Prepare one zero-F B8 decode case. This command never launches a container."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from fault_injection.quant_container_native import validate_bundle


def adapt(case, trace, trace_sha, quant_sha):
    if not str(trace).startswith('/') or any(c in str(trace) for c in ('\n', '\r', '\0', ',')):
        raise ValueError('ABSOLUTE_READONLY_TRACE_BUNDLE_REQUIRED')
    name = 'quant-four-container-one-decode'
    case['id'] = name
    case['mounts'].append([str(trace), '/container-trace', 'ro'])
    case['command'] = ('source /usr/local/Ascend/cann-8.5.1/set_env.sh\n'
        'export ASCEND_CUSTOM_OPP_PATH=/quant/runtime/vendor:/container-trace/runtime/vendor:/vllm-workspace/vllm-ascend/vllm_ascend/_cann_ops_custom/vendors/vllm-ascend\n'
        'export LD_PRELOAD=/container-trace/runtime/vendor/op_api/lib/libcust_opapi.so\n'
        'export PYTHONPATH=/installed:/msprobe_packages:$PYTHONPATH\n'
        'exec python3 /probe.py --quant-build-sha256 ' + quant_sha + ' --trace-build-sha256 ' + trace_sha)
    review = case['review']
    review.update(experiment_id=name, provider='quant read-only four-container entry diagnostic; no F or row authority',
        worker='one standard TP1 eager B8 first decode; 135 full Native/public projections; no repeated integer helper capture',
        build_fingerprint=trace_sha, quant_build_fingerprint=quant_sha, expected_hits=0,
        recovery_plan='Original create/destroy functions run once; no device data copy or numerical helper. '
            'Snapshot all raw entry/return/registration evidence before source classification. Missing calls, wrong caller, '
            'scope/thread ambiguity, failed destroy, live registration or overflow refuse the diagnostic. Remove owned '
            'source/profile/dispatcher/API/model hooks; stop Native observation, save immutable records, release only '
            'observer host buffers after all real recorded destroys complete. Failed destroy retains the registration '
            'and owner for retry; never destroy model objects or use a new Worker as recovery proof. Require unchanged '
            'same-Worker healthy tokens and zero observer resources. Complete 135 denominator persists; C1/C2/C3 false.')
    review['resource_budget'].update(container_calls=4096, container_events=8192,
        container_members=128, copied_container_metadata_bytes=8 * 1024 * 1024, timeout_seconds=1800)
    return case


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('installed', 'quant', 'trace', 'source-commit', 'image-id'):
        p.add_argument('--' + name, required=True)
    p.add_argument('--model', default='/home/haoran/models/DeepSeek-V2-Lite-W8A8')
    p.add_argument('--card', type=int, choices=(2, 5), default=2)
    p.add_argument('--probe', default=str(Path(__file__).with_name('probe_quant_containers.py')))
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    trace = Path(a.trace).resolve(strict=True)
    trace_sha = hashlib.sha256((trace / 'build.json').read_bytes()).hexdigest()
    _, build = validate_bundle(trace, trace_sha)
    if build['image_id'] != a.image_id:
        raise RuntimeError('EXACT_SDK_AND_RUNTIME_IMAGE_REQUIRED')
    quant_sha = hashlib.sha256((Path(a.quant) / 'build.json').read_bytes()).hexdigest()
    with tempfile.TemporaryDirectory(prefix='quant-container-case-') as directory:
        base = Path(directory) / 'base.json'
        subprocess.run([sys.executable, str(ROOT / 'experiments/c2c3_closure_20261117/providers_quant_prepare_inventory_case.py'),
            '--installed', a.installed, '--quant', a.quant, '--probe', a.probe, '--source-commit', a.source_commit,
            '--image-id', a.image_id, '--model', a.model, '--card', str(a.card), '--group-abi', '--out', str(base)], check=True)
        rows = json.loads(base.read_bytes())
    if len(rows) != 1:
        raise RuntimeError('ONE_CENSUS_BASE_CASE_REQUIRED')
    result = adapt(rows[0], trace, trace_sha, quant_sha)
    with a.out.open('x') as stream:
        json.dump([result], stream, indent=2)


if __name__ == '__main__':
    main()
