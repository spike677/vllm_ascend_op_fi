"""Live discovered normal-only targets. Reports cannot recreate this owner."""
import hashlib
import json
import os
import threading
from .serving_context import require

_ISSUER = object()


class NormalTargetScope:
    def __init__(self, worker, candidates, *, owner, generation):
        require(owner is not None and not isinstance(owner,(str,int,float,bool,dict,list,tuple))
            and type(generation) is int and generation > 0, 'ALL_NORMAL_LIVE_OWNER_REQUIRED')
        rows = tuple((path,module,module.impl) for path,module in candidates)
        require(len(rows)==28 and len({p for p,_,_ in rows})==len({id(m) for _,m,_ in rows})==28,
                'ALL_NORMAL_28_DISTINCT_ACTUAL_OBJECTS_REQUIRED')
        self.worker,self.runner,self.model=worker,worker.model_runner,worker.model_runner.model
        self.owner,self.generation,self.rows=owner,generation,rows
        self.thread=(os.getpid(),threading.get_ident(),threading.get_native_id())
        self._issuer=_ISSUER
        self._json=json.dumps([dict(path=p,module_object_id=id(m),impl_object_id=id(i)) for p,m,i in rows],
                             sort_keys=True,separators=(',',':'))
        self.fingerprint=hashlib.sha256(self._json.encode()).hexdigest()
        self.check(worker,rows[0][0],rows[0][1],owner,generation)

    def __reduce__(self):
        raise TypeError('Live normal targets are not serializable authority')

    def check(self,worker,path,module,owner,generation):
        require(self._issuer is _ISSUER and self.thread==(
            os.getpid(),threading.get_ident(),threading.get_native_id())
            and owner is self.owner and type(generation) is int and generation==self.generation
            and worker is self.worker and worker.model_runner is self.runner and self.runner.model is self.model,
            'ALL_NORMAL_TARGET_OWNER_THREAD_OR_WORKER_CHANGED')
        current=json.dumps([dict(path=p,module_object_id=id(m),impl_object_id=id(i)) for p,m,i in self.rows],
                           sort_keys=True,separators=(',',':'))
        require(current==self._json and hashlib.sha256(current.encode()).hexdigest()==self.fingerprint,
                'ALL_NORMAL_TARGET_SET_CHANGED')
        require(all(self.model.get_submodule(p) is m and m.impl is i for p,m,i in self.rows),
                'ALL_NORMAL_DISCOVERED_OBJECT_REPLACED')
        matches=[n for n,(p,m,_) in enumerate(self.rows) if p==path and m is module]
        require(len(matches)==1,'ALL_NORMAL_TARGET_NOT_IN_LIVE_DISCOVERY')
        return matches[0]

    def report(self):
        self.check(self.worker,self.rows[0][0],self.rows[0][1],self.owner,self.generation)
        return dict(schema='demoncat.paged33.all_normal_targets.v1',targets=json.loads(self._json),
            sha256=self.fingerprint,pid=self.thread[0],python_thread_ident=self.thread[1],native_tid=self.thread[2],
            worker_object_id=id(self.worker),model_object_id=id(self.model),owner_object_id=id(self.owner),
            generation=self.generation,serializable_authority=False,normal_only=True,C1=False,C2=False,C3=False)
