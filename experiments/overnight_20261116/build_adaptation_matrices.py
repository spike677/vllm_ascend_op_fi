"""Summarize observed host facts and pinned historical image profiles separately."""
import hashlib
import json
from pathlib import Path

from fault_injection.environment import classify_soc

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
PRIOR = ROOT / 'experiments/provider_adaptation_20261116'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def main():
    host_path=OUT/'host_inventory-a1.json';host=json.loads(host_path.read_text())
    previous=json.loads((PRIOR/'validation_summary.json').read_text())
    interfaces=json.loads((PRIOR/'interface_comparison.json').read_text())
    elf_path=OUT/'environment_elf-a1.json';elf=json.loads(elf_path.read_text())
    smi=next(r['stdout'] for r in host['inventory'] if r['argv']==['npu-smi','info'])
    assert '910B4' in smi and '25.5.0.b060' in smi
    image_rows=[]
    for tag in ('851','90','91','92'):
        path=OUT/'private'/f'version_profile-{tag}.json'
        profile=json.loads(path.read_text())
        prior_build=previous['builds'][f'build-{tag}-a5/build_manifest.json']['manifest']
        assert digest(path)==prior_build['files']['version_profile.json']
        assert profile['profile_sha256']==prior_build['profile_sha256']
        version=profile['installation']['ascend_toolkit_install.info']['version']
        observed=next(x for x in previous['fault39_operator_matrix'] if x['version']==version)
        assert observed['profile']==profile['profile_sha256']
        current=next(x for x in elf['rows'] if x['tag']==tag)
        assert current['returncode']==0 and current['data']['actual_npu_calls']==0
        libraries=current['data']['libraries']
        for library in libraries:
            expected=next(x for x in profile['upstream_libraries'].values() if Path(x['path']).name==library['name'])
            assert library['sha256']==expected['file_sha256'],library['path']
        row=dict(scope='HISTORICAL_ISOLATED_IMAGE_BUILD_AND_RUN',version=version,
                 installation=profile['installation'],toolkit=profile['toolkit'],
                 profile_sha256=profile['profile_sha256'],profile_file_sha256=digest(path),
                 original_profile=f'/home/provider-adaptation-20261116/build-{tag}-a5/version_profile.json',
                 native_adapter=profile['adapter'],abi_facts=profile['abi_facts'],
                 upstream_libraries=profile['upstream_libraries'],runtime_libraries=profile['libraries'],
                 required_exports=profile['required_exports'],headers_count=len(profile['headers']),
                 header_manifest_sha256=hashlib.sha256(json.dumps(profile['headers'],sort_keys=True,separators=(',',':')).encode()).hexdigest(),
                 selected_headers={n:h for n,h in profile['headers'].items() if n.endswith(('op_cache.h','acl_meta.h','aclnn_matmul.h','acl.h'))},
                 current_image_id=current['image_id'],current_elf_inventory=libraries,
                 elf_inventory_sha256=digest(elf_path),
                 elf_limit='Successful readelf -n produced no GNU Build ID; null is observed absence, not a synthetic ID. DT_NEEDED/SONAME and exact file hashes were collected.',
                 current_worker_loaded=None,current_worker_status='NO_CURRENT_SERVING_WORKER_OBSERVED',
                 prior_native_operator='NATIVE_VERIFIED' if observed['status_b']==observed['status_f']=='PASS' else 'UNVERIFIED',
                 prior_off_exact=observed['off_exact'],
                 prior_model_scope='Qwen3 TP1/TP2' if tag=='851' else 'Qwen3 TP2' if tag=='91' else None,
                 current_source_native_revalidation='NOT_RERUN',
                 evidence='../provider_adaptation_20261116/validation_summary.json')
        image_rows.append(row)
        save(ROOT/'profiles/environments'/f'cann-{version}-prior-124.json',row)
    env=dict(schema='demoncat.overnight.environment_matrix.v1',actual_npu_calls=0,
        inventory_source=dict(path='host_inventory-a1.json',sha256=digest(host_path)),
        driver=dict(observed_by='npu-smi info',version='25.5.0.b060'),
        host=dict(architecture='aarch64',observed_ascend_paths=host['host_ascend_paths'],
                  installed_version=None,status='DIRECTORY_INVENTORY_ONLY',
                  note='The host cann-9.0.0 path is observed; directory spelling alone is not installation-version proof'),
        hardware=[dict(scope='124 physical cards2/5',product=classify_soc('910B4'),
                       actual_device_observed=True,provider_support='ONLY_PER_RECEIPT_PINNED_COMBINATIONS'),
                  dict(scope='A3',actual_device_observed=False,provider_support='UNVERIFIED',reason='No A3 device observed')],
        image_installations=image_rows,
        missing_installations=[dict(version=v,status='UNVERIFIED',reason='No actual install profile in inspected host/images')
                               for v in ('8.0.x','8.1.x','8.2.x','8.3.x','8.4.x')],
        model_assets=[dict(r,loaded_worker=False,operator_path='UNVERIFIED_CURRENT_RUN') for r in host['models']],
        limitations=['Image installs are not host installs or current loaded-worker proof.',
                    'No version family or hardware-family blanket compatibility is granted.',
                    'Historical native results are reused with exact profile fingerprints, not rerun certificates.'])
    save(OUT/'environment_matrix.json',env)
    models=json.loads((OUT/'model_matrix.json').read_text())
    for row in models['models']:
        profile=dict(row)
        profile['evidence']='../../experiments/overnight_20261116/model_matrix.json'
        save(ROOT/'profiles/models'/(row['model']+'-cpu.json'),profile)
    print(json.dumps(dict(status='SUMMARIZED_WITH_PROVENANCE',image_profiles=len(image_rows),models=len(models['models']),actual_npu_calls=0)))


if __name__=='__main__':main()
