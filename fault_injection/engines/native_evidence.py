"""Lossless acceptance predicates, bounded per-site counters for Native copies.

Original per-call SHA pairs are hashed as a complete ordered list. They are
not returned as an oversized repeated Runtime payload. No evidence limit is
raised and no failing copy/consumer/restore is discarded.
"""
import hashlib,json


def compact_content_regions(regions):
    sites={}
    for r in regions:
        if not (r['consumed'] and r['restored'] and r['qmm_calls']==1 and r['region_elements']==64
                and r['after_sha256']==r['source_sha256'] and r['before_sha256']==r['restored_sha256']):
            raise ValueError('NATIVE_CONTENT_REGION_PROOF_FAILED')
        row=sites.setdefault(r['site'],dict(site=r['site'],calls=0,copy_exact=0,restore_exact=0,consumed=0,qmm_calls=0))
        for key in ('calls','copy_exact','restore_exact','consumed','qmm_calls'):row[key]+=1
    return dict(schema='demoncat.native_content.site_proofs.v1',region_count=len(regions),
        ordered_regions_sha256=hashlib.sha256(json.dumps(regions,sort_keys=True,separators=(',',':')).encode()).hexdigest(),
        sites=list(sites.values()))
