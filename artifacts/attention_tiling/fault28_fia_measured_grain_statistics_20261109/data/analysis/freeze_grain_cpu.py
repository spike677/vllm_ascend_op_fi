"""Preregister the identical numerical matrix with the scalar observer vendor."""
import hashlib
import json
from pathlib import Path

ROOT=Path('/home/fault28-fia-grain-20261109')
OLD=Path('/home/fault28-fia-graph-20261106')


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    matrix=json.loads((OLD/'setup/fp64_matrix_v1.json').read_bytes())
    vendor=sha(ROOT/'vendor_stage_a5/vendor_manifest.json');equivalence=[]
    for name in matrix['case_ids']:
        before=json.loads((OLD/'setup/fp64_v1'/(name+'.json')).read_bytes())
        case=dict(before)
        if case['operator_variant']=='private_vendor':
            case['approved_vendor_manifest_sha256']=vendor
            case['device_scalar_observer']=True
        target=ROOT/'setup/fp64_v1'/(name+'.json')
        target.write_text(json.dumps(case,indent=2)+'\n')
        excluded={'approved_vendor_manifest_sha256','device_scalar_observer'}
        assert {k:v for k,v in case.items() if k not in excluded}=={k:v for k,v in before.items() if k not in excluded}
        matrix['case_sha256'][name]=sha(target)
        equivalence.append({'case_id':name,'prior_case_sha256':sha(OLD/'setup/fp64_v1'/(name+'.json')),
                            'new_case_sha256':sha(target),'numerical_settings_unchanged':True})
    matrix.update(collection='device_scalar_observer_v1',device_queries=[63,255,511,767,1023,1535,2047],
                  prior_matrix_sha256=sha(OLD/'setup/fp64_matrix_v1.json'))
    (ROOT/'setup/fp64_matrix_v1.json').write_text(json.dumps(matrix,indent=2)+'\n')
    runtime=json.loads((OLD/'setup/fp64_runtime_v1.json').read_bytes())
    runtime['files']={name:sha(ROOT/name) for name in runtime['files']}
    (ROOT/'setup/fp64_runtime_v1.json').write_text(json.dumps(runtime,indent=2)+'\n')
    (ROOT/'setup/device_trace_equivalence.json').write_text(json.dumps({'cases':equivalence,
        'original_input_manifest_sha256':'509123dc725e351d6b10840919c2640b327a13eecceba2ca4a888c4a094f8e62',
        'trace_manifest_sha256':sha(ROOT/'build/device_grain_trace.json'),
        'required_neutrality':'Complete NPU target bytes and all-request hashes, not CPU assumption.'},indent=2)+'\n')
    print(json.dumps({'status':'FROZEN_NOT_NPU_ACCEPTED','cases':len(equivalence),'vendor_sha256':vendor,
                      'matrix_sha256':sha(ROOT/'setup/fp64_matrix_v1.json')}))


if __name__=='__main__':main()
