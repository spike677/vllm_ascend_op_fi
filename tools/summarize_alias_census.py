"""Bounded alias census text; no injection or model qualification."""
import argparse,collections,hashlib,json
from pathlib import Path

def pages(path):
    raw=Path(path).read_bytes();data=json.loads(raw)
    source=dict(file=Path(path).name,bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())
    rows=data['ranks'] if isinstance(data,dict) else data
    for row in rows:
        counts=collections.Counter();formats=collections.Counter();groups=collections.Counter()
        reports=row.get('reports',[])
        for forward in reports:
            report=forward['report'];counts.update(report['counts']);formats.update(report['view_base_formats'])
            for g in report['groups']:groups[(g['view_op'],g['writer'],g['consumer'],g['base_format'])]+=g['count']
        common=dict(schema='dcat.alias.census.extract.v1',rank=row['rank'],source=source,
                    observation_only=True,injection_enabled=False,qualification_granted=False)
        if 'events' in row:
            events=row['events'];counts.update(dict(natural_output_writes=len(events),
                consumed=sum(e.get('device_consumer_completed') is True for e in events),
                base_changed=sum(e.get('base_changed') is True for e in events)))
            formats.update(str(e.get('format')) for e in events)
            for e in events:groups[('actual_empty_then_view',e['writer'],e['consumer'],str(e.get('format')))]+=1
        yield dict(common,kind='summary',forwards=len(reports) if 'reports' in row else None,counts=dict(counts),view_base_formats=dict(formats),
                   hooks_remaining=row['hooks_remaining'],execute_wrapper_removed=row['execute_wrapper_removed'],
                   private_format_chain_proven=False,limitations='Dispatch-visible candidates; not implicit-copy or device-value proof')
        for key,count in sorted(groups.items()):
            yield dict(common,kind='group',view_op=key[0],writer=key[1],consumer=key[2],base_format=key[3],count=count)

def main():
    p=argparse.ArgumentParser();p.add_argument('receipt');a=p.parse_args()
    for row in pages(a.receipt):
        raw=json.dumps(row,sort_keys=True,ensure_ascii=False,separators=(',',':')).encode()
        row['extract_sha256']=hashlib.sha256(raw).hexdigest()
        raw=json.dumps(row,sort_keys=True,ensure_ascii=False,separators=(',',':')).encode()
        if len(raw)+1>4096:raise ValueError('ALIAS_CENSUS_PAGE_TOO_LARGE')
        import sys
        sys.stdout.buffer.write(raw+b'\n')

if __name__=='__main__':main()
