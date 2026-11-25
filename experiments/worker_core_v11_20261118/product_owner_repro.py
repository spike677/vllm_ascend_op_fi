"""Read-only control-boundary census around the unchanged HTTP acceptance flow.

Experiment only. Does not replace observer methods, suppress refusals or clean
foreign wrappers. Full diagnostics remain in the private evidence directory.
"""
import inspect
import json
import os
import sys
import traceback
from pathlib import Path

import product_http_probe as probe


class AcceptanceExtension(probe.AcceptanceExtension):
    def demoncat_acceptance_bootstrap(self, action, options):
        if action == 'owner_census':
            runner = self.model_runner
            runtime = getattr(self, '_demoncat_worker_fault_runtime', None)
            bindings = getattr(self, '_demoncat_continuous_bindings', {})
            def method(name):
                value = vars(runner).get(name)
                observer = None
                if inspect.isfunction(value):
                    observer = inspect.getclosurevars(value).nonlocals.get('self')
                binding = getattr(observer, 'binding', None)
                return dict(name=name, instance_override=value is not None,
                            object_id=id(value) if value is not None else None,
                            module=getattr(value, '__module__', None),
                            qualname=getattr(value, '__qualname__', None),
                            owner=getattr(binding, 'group', {}).get('owner'),
                            observer_id=id(observer) if observer is not None else None)
            providers = []
            for mechanism, lease in getattr(runtime, '_leases', {}).items():
                provider = getattr(lease, 'provider', None)
                observer = getattr(provider, 'observer', None)
                providers.append(dict(mechanism=mechanism,
                    observer_id=id(observer) if observer is not None else None,
                    patches=[dict(name=n, expected_id=id(w),
                                  matches=vars(runner).get(n) is w)
                             for n, w in getattr(observer, 'patches', [])],
                    snapshot=observer.snapshot() if observer is not None else None))
            # Request records can be large; the observer's counters suffice here.
            for row in providers:
                if row['snapshot'] is not None: row['snapshot'].pop('requests', None)
            return dict(rank=self.rank, pid=os.getpid(),
                methods=[method(n) for n in ('execute_model','sample_tokens','_preprocess')],
                bindings=[dict(mechanism=k,owner=b.group['owner'],leases=len(b.leases),
                               warmup_observer=b.warmup_request is not None)
                          for k,b in bindings.items()],
                providers=providers,tenants=len(getattr(runtime,'_tenants',{})))
        try:
            return super().demoncat_acceptance_bootstrap(action, options)
        except Exception:
            print('OWNER_REPRO_BOOTSTRAP_FAILURE action='+str(action), file=sys.stderr)
            traceback.print_exc()
            raise


def main():
    original_popen = probe.subprocess.Popen
    original_rpc = probe.HTTPExecutor.collective_rpc
    sequence = 0
    def popen(command, *args, **kwargs):
        if isinstance(command,list):
            command = ['product_owner_repro.AcceptanceExtension'
                       if v == 'product_http_probe.AcceptanceExtension' else v for v in command]
        return original_popen(command,*args,**kwargs)
    def rpc(self,method,*,args=(),kwargs=None,timeout=120):
        nonlocal sequence
        sequence += 1
        row=dict(sequence=sequence,method=method,action=args[0] if args else None)
        def census():
            return original_rpc(self,'demoncat_acceptance_bootstrap',args=('owner_census',{}),timeout=timeout)
        row['before']=census()
        try:
            result=original_rpc(self,method,args=args,kwargs=kwargs,timeout=timeout)
            row['statuses']=[dict(rank=r.get('rank',r.get('worker',{}).get('rank')),
                status=r.get('status'),reason=r.get('reason'),response=r.get('response'))
                for r in result if isinstance(r,dict)] if isinstance(result,list) else None
            return result
        except Exception as error:
            row['error']=repr(error)
            row['traceback']=traceback.format_exc()
            raise
        finally:
            try:row['after']=census()
            except Exception as error:row['after_error']=repr(error)
            with Path('/evidence/owner-census.jsonl').open('a') as f:
                f.write(json.dumps(row)+'\n')
    probe.subprocess.Popen=popen
    probe.HTTPExecutor.collective_rpc=rpc
    try:return probe.main()
    finally:
        probe.subprocess.Popen=original_popen
        probe.HTTPExecutor.collective_rpc=original_rpc


if __name__=='__main__':raise SystemExit(main())
