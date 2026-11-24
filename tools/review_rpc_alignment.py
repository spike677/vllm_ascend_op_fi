"""Offline comparison of diagnostic host replies and each rank's durable log."""
import argparse
import hashlib
import json
from pathlib import Path


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def review(host, rank_logs, world_size=2):
    if set(rank_logs) != set(range(world_size)) or not host:
        raise ValueError('INCOMPLETE_RANK_LOGS')
    seen=set();actions={}
    if any(len(rows)!=len(host) for rows in rank_logs.values()):
        raise ValueError('DURABLE_LOG_LENGTH_MISMATCH')
    for index, call in enumerate(host):
        token=call['token']
        if token in seen:raise ValueError('DUPLICATE_CALL_TOKEN')
        seen.add(token)
        rows=call['ranks']
        if sorted(row['rank'] for row in rows)!=list(range(world_size)):
            raise ValueError('CALL_RANK_SET_MISMATCH')
        for row in rows:
            if row['token']!=token or row['action']!=call['action']:
                raise ValueError('STALE_CALL_REPLY')
            if digest(row['result'])!=row['result_sha256']:
                raise ValueError('RESULT_DIGEST_MISMATCH')
            if row != rank_logs[row['rank']][index]:
                raise ValueError('DURABLE_REPLY_MISMATCH')
        actions[call['action']]=actions.get(call['action'],0)+1
    return dict(passed=True,calls=len(host),rank_replies=len(host)*world_size,
                actions=actions,exact_durable_replies=True,
                scope='Instrumented public calls listed in actions; not arbitrary yellow RPC code')


def main():
    p=argparse.ArgumentParser();p.add_argument('directory');a=p.parse_args();root=Path(a.directory)
    read=lambda name:[json.loads(line) for line in (root/name).read_text().splitlines() if line.strip()]
    print(json.dumps(review(read('rpc-host.jsonl'),{i:read('rpc-rank-'+str(i)+'.jsonl') for i in (0,1)}),indent=2))


if __name__=='__main__':main()
