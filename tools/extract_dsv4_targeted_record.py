"""Read-only bounded targeted-r3 extraction, independently verifiable text pages."""
import argparse, hashlib, json
from pathlib import Path
def canonical(x):return json.dumps(x,sort_keys=True,separators=(",",":"),ensure_ascii=False,allow_nan=False).encode()
def pages(path):
    raw=Path(path).read_bytes();reports=json.loads(raw)
    if not isinstance(reports,list):reports=[reports]
    payload=[]
    for rpc in reports:
        report=rpc.get("result",rpc);rank=rpc.get("rank",report.get("rank"))
        tr=report.get("targeted_observation")
        if not tr:raise ValueError("TARGETED_OBSERVATION_MISSING")
        payload.append(dict(rank=rank,summary={k:v for k,v in tr.items() if k not in ("operations","scheduler_maps")},
                            operations=len(tr["operations"]),qualification_granted=False))
        for step,mapping in tr["scheduler_maps"].items():
            payload.append(dict(rank=rank,step=step,input_rows_count=len(mapping.get("rows",[])),
                                complete=mapping.get("complete"),reason=mapping.get("reason"),
                                mapping_sha256=hashlib.sha256(canonical(mapping)).hexdigest()))
        for op in tr["operations"]:
            payload.append(dict(rank=rank,operation={k:v for k,v in op.items() if k not in ("inputs","outputs","context","edges","forward_context")}))
            for field in ("context","edges"):
                for item in op.get(field,[]):payload.append(dict(rank=rank,operation_id=op["id"],field=field,item=item))
            if "forward_context" in op:
                ctx=op["forward_context"]
                payload.append(dict(rank=rank,operation_id=op["id"],forward_context={k:v for k,v in ctx.items() if k!="mc2_mask"},
                                    mc2_mask_sha256=(ctx.get("mc2_mask") or {}).get("sample_sha256")))
            for side in ("inputs","outputs"):
                for t in op[side]:
                    item={k:v for k,v in t.items() if k!="sample_values"}
                    # Full sample stays in original; prefix locates the actual values without pretending completeness.
                    vals=t.get("sample_values")
                    item["sample_preview"]=vals[:16] if vals is not None else None
                    item["sample_preview_count"]=min(len(vals),16) if vals is not None else 0
                    payload.append(dict(rank=rank,operation_id=op["id"],side=side,tensor=item))
    out=[]
    for i,item in enumerate(payload):
        page=dict(schema="dcat.targeted.extract.v3",source_bytes=len(raw),source_sha256=hashlib.sha256(raw).hexdigest(),page=i,pages=len(payload),payload=item)
        page["extract_sha256"]=hashlib.sha256(canonical(page)).hexdigest()
        if len(canonical(page))+1>4096:raise ValueError("TARGETED_PAGE_EXCEEDS_4096_NO_TRUNCATION")
        out.append(page)
    return out
def verify(path):
    rows=[json.loads(x) for x in Path(path).read_text(encoding="utf-8").splitlines() if x.strip()]
    if not rows:raise ValueError("EMPTY")
    for i,row in enumerate(rows):
        digest=row.pop("extract_sha256")
        if row["page"]!=i or row["pages"]!=len(rows) or hashlib.sha256(canonical(row)).hexdigest()!=digest:raise ValueError("EXTRACT_INVALID")
        if i and (row["source_sha256"],row["source_bytes"])!=(rows[0]["source_sha256"],rows[0]["source_bytes"]):raise ValueError("SOURCE_CHANGED")
    return dict(verified=True,pages=len(rows))
def main():
    import sys
    p=argparse.ArgumentParser();p.add_argument("input");p.add_argument("--verify",action="store_true");a=p.parse_args()
    if a.verify:print(json.dumps(verify(a.input)));return
    for row in pages(a.input):sys.stdout.buffer.write(canonical(row)+b"\n")
if __name__=="__main__":main()

