"""Validate every call before bounded per-site export; preserve ordered SHA."""
from collections import Counter
from hashlib import sha256
import json


def compact_matmul_calls(events, forwards, *, fault=True):
    sites = {}
    for e in events:
        site = e['site']
        r = sites.setdefault(site, dict(site=site, observed=0, eligible=0, applied=0,
            consumed=0, restored=0, changed=0, failed=0, reference_controls=0, reasons={}))
        r['observed'] += 1
        if e.get('failed'):
            raise ValueError('MATMUL_CALL_PROOF_FAILED:' + site)
        if not fault and any(e.get(k) for k in ('applied','restored','changed')):
            raise ValueError('MATMUL_DISABLED_CALL_MODIFIED:' + site)
        if not fault and e.get('eligible'):
            if e.get('consumed') is not True:
                raise ValueError('MATMUL_NORMAL_CONSUMER_MISSING:' + site)
            r['consumed'] += 1
        if fault and e.get('eligible'):
            if not (e.get('full_range_control_passed') is True and e.get('applied') is True
                    and e.get('consumed') is True and e.get('restored') is True
                    and e.get('before_sha256') is not None
                    and e['before_sha256'] == e.get('restored_sha256') and not e.get('failed')):
                raise ValueError('MATMUL_CALL_PROOF_FAILED:' + site)
            for k in ('eligible', 'applied', 'consumed', 'restored', 'reference_controls'):
                r[k] += 1
            r['changed'] += int(e.get('changed', 0) > 0)
        elif e.get('reason'):
            reason=e['reason'];r['reasons'][reason]=r['reasons'].get(reason,0)+1
    if len(forwards)!=4 or not all(forwards):
        raise ValueError('MATMUL_FORWARD_COVERAGE_MISSING')
    counts=Counter()
    for f in forwards:
        for site,n in f.items():
            if type(n) is not int or n<=0:raise ValueError('MATMUL_FORWARD_COUNT_INVALID')
            counts[site]+=n
    if dict(counts)!={name:r['observed'] for name,r in sites.items()}:
        raise ValueError('MATMUL_FORWARD_EVENT_JOIN_FAILED')
    for name,r in sites.items():r['forward_counts']=[f.get(name,0) for f in forwards]
    return dict(schema='demoncat.matmul.site_proofs.v1', call_count=len(events),
        ordered_calls_sha256=sha256(json.dumps(events,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest(),
        sites=list(sites.values()))
