"""Compare reviewed local source hashes with the actual deployed source trees."""
import argparse
import hashlib
import json
from pathlib import Path


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--expected',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();expected=json.loads(a.expected.read_text());rows=[]
    for item in expected:
        raw=(a.root/item['deployed']).read_bytes();actual=hashlib.sha256(raw).hexdigest()
        rows.append(dict(item,actual_sha256=actual,match=actual==item['reviewed_sha256'],
                         canonical_lf_sha256=hashlib.sha256(raw.replace(b'\r\n',b'\n')).hexdigest()))
    profiles=[]
    for version in ('851','90','91','92'):
        path=a.root/f'build-{version}-a5/version_profile.json'
        profile=json.loads(path.read_text());checks=[]
        for name,value in profile['sources'].items():
            path=(a.root/'source-a5/fault39'/name if name.startswith('tools/')
                  else a.root/'source-a5/fault39/native'/name)
            checks.append(dict(path=name,sha256=value,match=digest(path)==value))
        profiles.append(dict(version=profile['installation']['ascend_toolkit_install.info']['version'],
                             profile_sha256=profile['profile_sha256'],source_checks=checks))
    out=dict(schema='demoncat.provider.source_review.v2',files=rows,profiles=profiles,
             status='PASS' if all(x['match'] for x in rows) and all(c['match'] for p in profiles for c in p['source_checks']) else 'FAIL',
             note='Reviewed workspace vs deployed source bytes; Git may normalize CRLF to LF (canonical_lf_sha256). Not a substitute for runtime/NPU acceptance; vendor binaries not exported')
    with a.output.open('x') as stream:json.dump(out,stream,indent=2)
    print(json.dumps(dict(status=out['status'],files=len(rows),profiles=len(profiles))))
    return int(out['status']!='PASS')


if __name__=='__main__':raise SystemExit(main())
