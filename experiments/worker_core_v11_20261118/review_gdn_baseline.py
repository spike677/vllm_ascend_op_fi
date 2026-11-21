"""Read-only cold/repeat B8 review using complete real hidden/logits hashes."""
import argparse,json,pathlib

def review(directory):
 p=pathlib.Path(directory);report=json.loads((p/'diagnostic.json').read_text());fail=[];details=[]
 if report.get('status')!='DIAGNOSTIC_COMPLETED_NOT_QUALIFICATION':fail.append('DIAGNOSTIC_INCOMPLETE')
 rounds=report.get('rounds',[])
 if len(rounds)!=3 or [r.get('batch') for r in rounds]!=[8,8,1]:fail.append('WRONG_ROUNDS')
 else:
  for r in rounds:
   if any(x.get('hooks') or x.get('patches') for x in r.get('remove',[])):fail.append('RESOURCES_REMAIN')
  if [x['tokens'] for x in rounds[0].get('outputs',[])]!=[x['tokens'] for x in rounds[1].get('outputs',[])]:fail.append('COLD_REPEAT_TOKENS_DIFFER')
 for rank in range(2):
  snapshots=[]
  for step in range(2):
   f=p/('rank'+str(rank)+'-round'+str(step)+'.jsonl')
   if not f.is_file():fail.append('MISSING_RANK_TRACE');continue
   rows=[json.loads(l) for l in f.read_text().splitlines()];selected=[]
   for row in rows:
    if (row.get('kind')=='module_output' and row.get('name')=='') or (row.get('kind')=='api_output' and row.get('name')=='compute_logits'):
     for t in row['stats']:
      if not t.get('finite') or 'sha256' not in t:fail.append('NONFINITE_OR_UNOBSERVABLE')
      selected.append({k:row.get(k) for k in ('forward','kind','name')}|dict(shape=t['shape'],sha256=t.get('sha256'),finite=t.get('finite')))
   if len([r for r in selected if r['kind']=='module_output'])!=2:fail.append('HIDDEN_MISSING')
   if rank==0 and len([r for r in selected if r['name']=='compute_logits'])!=2:fail.append('LOGITS_MISSING')
   snapshots.append(selected)
  if len(snapshots)==2 and snapshots[0]!=snapshots[1]:fail.append('COLD_REPEAT_FULL_ARRAY_HASH_DIFF_RANK'+str(rank))
  details.append(dict(rank=rank,snapshots=snapshots))
 return dict(status='PASS_COLD_REPEAT_B8' if not fail else 'FAIL_BASELINE',failures=sorted(set(fail)),details=details,scope='One independent Worker group; restart comparison required separately; no Batch fault admission')

def main():
 p=argparse.ArgumentParser();p.add_argument('directory');a=p.parse_args();print(json.dumps(review(a.directory),indent=2))
if __name__=='__main__':main()
