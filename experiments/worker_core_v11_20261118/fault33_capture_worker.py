"""Trusted deployment RPC helpers only; install no binding and grant no admission.

Call inside each existing Worker after the host has authenticated the operator.
Do not expose these Python callables as unauthenticated HTTP endpoints.
"""
import ctypes
import hashlib
import os
from pathlib import Path
import re
import subprocess


def _sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()


def _mapped_files():
    result={}
    for line in Path('/proc/self/maps').read_text().splitlines():
        fields=line.split(None,5)
        if len(fields)!=6 or not fields[5].startswith('/'):continue
        path=fields[5];dev=fields[3];inode=int(fields[4])
        result.setdefault(path,set()).add((dev,inode))
    return result


def _file_identity(path,mapped):
    row=dict(path=path,sha256=None,build_id=None,status='UNPROVEN')
    try:
        before=Path(path).stat()
        dev=f'{os.major(before.st_dev):02x}:{os.minor(before.st_dev):02x}'
        if (dev,before.st_ino) not in mapped[path]:raise ValueError('MAPPED_INODE_MISMATCH')
        row['sha256']=_sha(path)
        notes=subprocess.run(['readelf','-n',path],capture_output=True,text=True,check=True).stdout
        ids=re.findall(r'Build ID:\s*([0-9a-fA-F]+)',notes)
        row['build_id']=ids[0] if len(set(ids))==1 else None
        after=Path(path).stat()
        if (before.st_ino,before.st_size,before.st_mtime_ns)!=(after.st_ino,after.st_size,after.st_mtime_ns):
            raise ValueError('FILE_CHANGED_DURING_HASH')
        row['status']='MAPPED_FILE_VERIFIED' if row['build_id'] else 'BUILD_ID_UNAVAILABLE'
    except (OSError,ValueError,subprocess.SubprocessError) as error:
        row['reason']=str(error)
    return row


def observe(worker, *, sequence):
    mapped=_mapped_files()
    selected=[p for p in mapped if re.search(r'opapi|optiling|opmaster|transformer|ascendcl|libruntime|capture-v2',p,re.I)]
    return dict(schema='dcat.fault33.worker.identity.v1',rank=worker.rank,pid=os.getpid(),sequence=sequence,
                libraries=[_file_identity(p,mapped) for p in sorted(selected)],
                mapped_file_inventory=sorted(mapped),cann_runtime=None,
                cann_runtime_source='Use the existing trusted runtime observation; do not infer from toolkit files',
                qualification_granted=False)


def flush(worker, *, library_path, expected_sha256, sequence):
    path=str(Path(library_path).resolve());mapped=_mapped_files()
    row=_file_identity(path,mapped) if path in mapped else dict(status='NOT_MAPPED')
    if row.get('sha256')!=expected_sha256 or row.get('status') not in ('MAPPED_FILE_VERIFIED','BUILD_ID_UNAVAILABLE'):
        return dict(rank=worker.rank,pid=os.getpid(),sequence=sequence,status='CAPTURE_FLUSH_REFUSED',identity=row)
    lib=ctypes.CDLL(path,mode=os.RTLD_NOLOAD|os.RTLD_LOCAL)
    fn=lib.dcat33_capture_flush;fn.argtypes=[];fn.restype=ctypes.c_int
    rc=fn()
    return dict(rank=worker.rank,pid=os.getpid(),sequence=sequence,status='CAPTURE_SEALED' if rc==0 else 'CAPTURE_FLUSH_REFUSED',
                return_code=rc,library_sha256=expected_sha256,qualification_granted=False)
