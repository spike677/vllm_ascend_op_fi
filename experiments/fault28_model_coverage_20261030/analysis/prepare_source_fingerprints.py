"""Read-only installed-source hash audit; writes only new task metadata."""
import hashlib
import json
from pathlib import Path

ROOT = Path('/home/fault28-model-coverage-20261030')
SITE = Path('/home/tik-constfold-cann91-overnight-20261022/venv/lib/python3.11/site-packages')
CONTAINER_SITE = Path('/work/venv/lib/python3.11/site-packages')
MODULES = {
    'transformers.models.qwen3.modeling_qwen3': 'transformers/models/qwen3/modeling_qwen3.py',
    'transformers.integrations.sdpa_attention': 'transformers/integrations/sdpa_attention.py',
    'transformers.masking_utils': 'transformers/masking_utils.py',
    'torch.nn.attention': 'torch/nn/attention/__init__.py',
}


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def write_once(path, payload):
    data = (json.dumps(payload, indent=2, sort_keys=True) + '\n').encode()
    if path.exists():
        assert path.read_bytes() == data, 'Existing task metadata differs: ' + str(path)
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('xb') as stream:
            stream.write(data)


def main():
    route_path = ROOT / 'analysis/hf_sdpa_route_reference.json'
    route = json.loads(route_path.read_bytes())
    modules = {}
    for name, relative in MODULES.items():
        path = SITE / relative
        actual = sha(path)
        assert actual == route['installed_sources'][relative]['sha256'], name
        modules[name] = {'path': str(CONTAINER_SITE / relative), 'host_path': str(path),
                         'sha256': actual, 'bytes': path.stat().st_size}
    versions = {}
    for package, glob in [('torch', 'torch-*.dist-info'), ('torch-npu', 'torch_npu-*.dist-info'),
                          ('transformers', 'transformers-*.dist-info')]:
        paths = list(SITE.glob(glob + '/METADATA'))
        assert len(paths) == 1, (package, paths)
        found = [line.split(': ', 1)[1] for line in paths[0].read_text().splitlines()
                 if line.startswith('Version: ')]
        assert len(found) == 1
        versions[package] = found[0]
    assert versions == {'torch': '2.10.0', 'torch-npu': '2.10.0.post4', 'transformers': '5.5.4'}, versions
    source = {'schema': 'nes.model_source_fingerprint.v1', 'versions': versions, 'modules': modules,
              'backend_sources': {'source_route_reference_sha256': sha(route_path),
                  'source_revision': route['source_revision'], 'official_sources': route['official_sources'],
                  'installed_backend_witnesses': {key: row for key, row in route['installed_sources'].items()
                      if key.startswith('torch_npu/')},
                  'native_route': route['native_route'],
                  'qualification': 'Pinned source-route evidence; actual execution coverage must be measured by this task.'},
              'audit': {'framework_imported': False, 'model_loaded': False, 'npu_calls': 0}}
    # Rehash installed binary/headers as witnesses without loading their code.
    for name, row in source['backend_sources']['installed_backend_witnesses'].items():
        assert sha(SITE / name) == row['sha256'], name
    write_once(ROOT / 'source_fingerprints.json', source)
    files = ['model_manifest.json', 'tokenizer_manifest.json', 'source_fingerprints.json',
             'memory_api_source_audit.json']
    runtime = ROOT / 'runtime_manifest.json'
    vendor = ROOT / 'native_vendor_stage_a1/vendor_manifest.json'
    assert sha(runtime) == 'ca0bf188bdc031ea3e4249c03ac13a9a8f751cd5ae8485655d9048c9f6cdba3b'
    assert sha(vendor) == '42173485d5d3c86b57cd8a847dbce2dfd1fc42004c1a451bb282b7aa211d9ef1'
    inventory = json.loads(runtime.read_bytes())['files']
    for name, row in inventory.items():
        path = ROOT / 'runtime' / name
        assert sha(path) == row['sha256'] and path.stat().st_size == row['bytes']
    vm = json.loads(vendor.read_bytes())
    for row in vm['artifacts']:
        path = ROOT / vm['vendor_root'] / row['path']
        assert sha(path) == row['sha256'] and path.stat().st_size == row['bytes']
    receipt = {'schema': 'nes.model_coverage.dependency_preparation.v1', 'status': 'PREPARED_CPU_ONLY',
               'runtime_manifest_sha256': sha(runtime), 'runtime_files': len(inventory),
               'vendor_manifest_sha256': sha(vendor), 'vendor_artifacts': len(vm['artifacts']),
               'fingerprints': {name: sha(ROOT / name) for name in files},
               'source_dependencies_changed': False, 'npu_calls': 0, 'containers_started': 0}
    write_once(ROOT / 'analysis/dependency_preparation.json', receipt)
    print(json.dumps(receipt, indent=2))


if __name__ == '__main__':
    main()
