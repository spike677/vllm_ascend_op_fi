"""Export two bounded Host receipts for manual deployment review; no registration."""
import argparse
import json
from pathlib import Path
from demoncat.runtime.vllm_ascend.worker_extension.deployment_summary import summarize_deployment, MAX_RECEIPT_BYTES

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scope',type=Path,action='append',required=True)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    if len(args.scope)!=2:parser.error('exactly two --scope receipts required')
    receipts=[]
    for path in args.scope:
        if path.stat().st_size>MAX_RECEIPT_BYTES:raise ValueError('HOST_RECEIPT_LIMIT')
        receipts.append(json.loads(path.read_bytes()))
    result=summarize_deployment(receipts)
    with args.out.open('x',encoding='utf-8') as stream:
        json.dump(result,stream,ensure_ascii=False,sort_keys=True,separators=(',',':'),allow_nan=False)
        stream.write('\n')
    print(json.dumps({'output':str(args.out),'review_ready':result['review_ready'],'registration_written':False}))

if __name__=='__main__':main()
