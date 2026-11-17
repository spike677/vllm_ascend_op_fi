"""Read fixed dependency Git objects; does not edit or execute any binary."""
import hashlib
import json
from pathlib import Path
import re
import subprocess

ROOT=Path('/home/fault28-fia-graph-20261106/dependencies_a4')
pins={'opbase':'cd48c600c4b37146f504217eab2bf3956f7fee8b',
      'ops-tensor':'b24535e30a6f7a3909c53c8b09713a9c0d08e54b',
      'cann-cmake':'7e5691ce110367662017d9d94114845a7d6a8d8a'}
rows=[]
for name,pin in pins.items():
    repo=ROOT/name
    tree=subprocess.check_output(['git','-C',str(repo),'ls-tree','-r',pin],text=True)
    for line in tree.splitlines():
        meta,path=line.split('\t',1)
        if not re.search(r'\.(?:o|a|so(?:\.\d+)*)$',path):continue
        raw=subprocess.check_output(['git','-C',str(repo),'show',pin+':'+path])
        rows.append(dict(repository=name,commit=pin,path=path,blob=meta.split()[2],
            bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest(),is_test=path.startswith('tests/')))
print(json.dumps(rows,indent=2))
