"""Audit committed r7.1 source against immutable r7, with an explicit change allowlist."""
import argparse,hashlib,json,subprocess
from pathlib import Path
R7='c50efb81dc1650dd4fadcd0d62de79eeb33d7138'
RUNTIME={
 'fault_injection/sites/alias15.py':'new Fault15 natural ND writer/consumer adapter',
 'fault_injection/engines/region_lease.py':'Fault15 release_unmodified; no historical-byte writeback',
 'fault_injection/product_contract.py':'Fault15 candidate and element-count evidence',
 'src/demoncat/runtime/vllm_ascend/worker_extension/continuous_binding.py':'Fault15 supported entry and site factory',
 'src/demoncat/runtime/vllm_ascend/worker_extension/product_binding.py':'Fault15-only TP2 observed-device/CANN admission',
 'src/demoncat/runtime/vllm_ascend/worker_extension/product_request.py':'Fault15-only independent event evidence',
 'experiments/worker_core_v11_20261118/product_http_probe.py':'Fault15 probe selection',
}
META={'pyproject.toml','src/pyproject.toml','fault39/pyproject.toml'}
def git(*args):return subprocess.check_output(['git',*args])
def tree(ref):
 out={}
 for row in git('ls-tree','-r','-z',ref).split(b'\0'):
  if row:
   meta,name=row.split(b'\t',1);mode,kind,oid=meta.decode().split();out[name.decode()]=(mode,kind,oid)
 return out
def compare(head):
 head=git('rev-parse',head).decode().strip();a,b=tree(R7),tree(head);rows=[]
 for name in sorted(set(a)|set(b)):
  if a.get(name)==b.get(name):continue
  if name in RUNTIME:category='FAULT15_RUNTIME';reason=RUNTIME[name]
  elif name in META:category='RELEASE_VERSION_METADATA';reason='three wheels receive new immutable versions'
  elif name=='README.md' or name.startswith('docs/'):category='DOCUMENTATION_AND_EVIDENCE';reason='explicitly listed; not claimed byte-identical'
  elif name.startswith(('tests/','tools/')):category='TEST_REVIEW_OR_PACKAGING';reason='explicitly listed; no mechanism rewrite'
  else:raise ValueError('UNREVIEWED_CHANGE:'+name)
  def record(ref,t):
   if name not in t:return None
   raw=git('show',ref+':'+name);return dict(bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest(),git_blob=t[name][2])
  rows.append(dict(path=name,category=category,reason=reason,r7=record(R7,a),r71=record(head,b)))
 stable_sites=[n for n in a if n.startswith('fault_injection/sites/') and n.endswith('.py')]
 if any(a[n]!=b.get(n) for n in stable_sites):raise ValueError('EXISTING_MECHANISM_SITE_CHANGED')
 return dict(schema='dcat.r71.r7.file_comparison.v1',r7=R7,r71_source_commit=head,
  unchanged_tracked_files=sum(a.get(n)==b.get(n) for n in set(a)|set(b)),
  unchanged_existing_site_files=len(stable_sites),changed_files=rows,
  exact_claim='Every tracked file not listed has the identical Git blob and mode. Listed shared Fault15 interfaces, metadata, tools and docs are exceptions, not byte-identical.',
  qualification_granted=False)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--head',default='HEAD');p.add_argument('--output',required=True);a=p.parse_args();d=compare(a.head);Path(a.output).write_bytes((json.dumps(d,indent=2)+'\n').encode());print(json.dumps(dict(unchanged=d['unchanged_tracked_files'],unchanged_sites=d['unchanged_existing_site_files'],changed=len(d['changed_files']))))
