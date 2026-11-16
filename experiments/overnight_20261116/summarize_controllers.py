"""Read-only environment gates, image identities and controller outcome hashes."""
import hashlib
import json
from pathlib import Path

ROOT=Path('/home/overnight-17-20261116')
rows=[]
for path in sorted(ROOT.glob('*/controller.json')):
    raw=path.read_bytes(); value=json.loads(raw)
    rows.append(dict(path=str(path),sha256=hashlib.sha256(raw).hexdigest(),
        status=value['status'],cards=value['cards'],cases=[
            {k:c[k] for k in ('id','gate','exit_code','timeout','image_id','container_state') if k in c}
            for c in value['cases']]))
out=dict(schema='demoncat.overnight.controller_summaries.v1',controllers=rows,
         external_jobs_stopped=False,resource_lock='/var/tmp/fault22_npu_2_5.lock')
(ROOT/'controller_receipts-a1.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(dict(controllers=len(rows))))
