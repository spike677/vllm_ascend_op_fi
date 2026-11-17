"""Check and fingerprint this branch's reviewed source/summary publication set."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess


BASE='cae9cdf4685543df98598bbc2cf0f3b9892b7635'


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True)
    p.add_argument('--check-index',action='store_true');a=p.parse_args()
    root=Path(__file__).resolve().parents[2]
    def git(*cmd):return subprocess.check_output(['git',*cmd],cwd=root,text=True).splitlines()
    names=set(git('diff','--name-only','--diff-filter=ACMR',BASE))|set(git('ls-files','--others','--exclude-standard'))
    out=a.out.resolve();rows=[]
    for name in sorted(names):
        path=root/name
        if path.resolve()==out:continue
        assert path.is_file() and not path.is_symlink(),name
        assert path.suffix.lower() not in {'.so','.o','.a','.dll','.exe','.pt','.pth','.bin','.safetensors','.zip','.gz'},name
        assert path.name not in {'dump.json','stack.json','construct.json'},name
        assert path.stat().st_size<1024*1024,name
        raw=path.read_bytes();raw.decode('utf-8')
        if a.check_index:
            blob=subprocess.check_output(['git','show',':'+name],cwd=root)
            assert blob==raw,'INDEX_BYTES_DIFFER:'+name
        rows.append(dict(path=name,bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest()))
    assert rows
    frozen=('docs/roadmap/17_fault_acceptance_matrix.json','docs/roadmap/17_fault_full_coverage_roadmap.md')
    assert not git('diff','--name-only',BASE,'--',*frozen),'FROZEN_ROADMAP_CHANGED'
    report=dict(schema='demoncat.engineering.publication_review.v1',base_commit=BASE,
        source_stage_commit=git('rev-parse','HEAD')[0],files=rows,
        contains_new_raw_statistics=False,contains_tensor_payloads=False,contains_model_weights=False,
        contains_native_binaries=False,frozen_roadmap_unchanged=True,
        index_bytes_verified=a.check_index,
        note='Files are fingerprinted before publication. Git commit identity is provided by the containing commit, not a self-referential hash.')
    out.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(files=len(rows),bytes=sum(r['bytes'] for r in rows),sha256=hashlib.sha256(out.read_bytes()).hexdigest())))


if __name__=='__main__':main()
