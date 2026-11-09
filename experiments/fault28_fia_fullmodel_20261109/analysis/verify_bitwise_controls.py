"""Bind dense private comparisons to original full/request byte digests."""
import argparse
import hashlib
import json
from pathlib import Path

ROOT=Path('/home/fault28-fia-fullmodel-20261109')
CONTROLS=('fault_repeat','off_restore_b8','b1_guard_r1','b1_guard_r2','off_restore_b1',
          'baseline_alias_neutrality','fault_alias_neutrality')


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def bound_values(report,side):
    folder=Path(report[side]);assert folder.resolve().is_relative_to((ROOT/'outputs').resolve())
    source=folder/'result.json';assert sha(source)==report[side+'_result_sha256']
    record=json.loads(source.read_bytes());assert record['repeated_all_outputs_equal']
    return {(r['layer'],r['kind']):r for r in record['calls'][0]['values']}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--reports',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args();rows=[]
    for label in CONTROLS:
        path=a.reports/(label+'_complete.json');report=json.loads(path.read_bytes())
        assert report['all_observed_elements_equal']
        b,f=bound_values(report,'B'),bound_values(report,'F')
        for row in report['comparisons']:
            key=(row['layer'],row['kind'])
            assert row['B_sha256']==b[key]['full_sha256']==row['F_sha256']==f[key]['full_sha256']
            assert b[key]['request_sha256']==f[key]['request_sha256']
        rows.append({'label':label,'comparison_sha256':sha(path),'full_byte_sha256_equal':True,
            'per_request_byte_sha256_equal':True,'observed_tensor_count':len(report['comparisons'])})
    for label in ('fault_r1','fault_r2'):
        path=a.reports/(label+'_complete.json');report=json.loads(path.read_bytes())
        b,f=bound_values(report,'B'),bound_values(report,'F');layer=report['first_changed_attention_layer']
        for kind in ('query','key','value'):
            key=(layer,kind);assert b[key]['full_sha256']==f[key]['full_sha256']
            assert b[key]['request_sha256']==f[key]['request_sha256']
        rows.append({'label':label,'first_changed_attention_layer':layer,
            'complete_comparison_sha256':sha(path),'first_attention_QKV_byte_sha_equal_all_requests':True})
    result={'schema':'fault28.fia_fullmodel.byte_digest_control_review.v1',
        'status':'PASS_DENSE_COMPARISONS_AND_ALL_REQUEST_BYTE_DIGESTS',
        'checks':rows,'private_tensor_files_opened':0,'analysis_source_sha256':sha(Path(__file__)),
        'scope':'Dense comparisons already checked every element. This additional review binds their SHA fields to original capture full/request digests, including signed-zero distinctions.'}
    with a.output.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps({'status':result['status'],'checks':len(rows)}))


if __name__=='__main__':main()
