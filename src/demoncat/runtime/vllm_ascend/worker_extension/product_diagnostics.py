"""Opt-in status diagnostics; unchanged ProductClient owns every control action."""
from copy import deepcopy
from datetime import datetime, timezone
from .product_host import ProductClient


class _CaptureExecutor:
    def __init__(self, original, client):self.original,self.client=original,client
    def __getattr__(self, name):return getattr(self.original,name)
    def collective_rpc(self, method, *args, **kwargs):
        rows=self.original.collective_rpc(method,*args,**kwargs)
        rpc_args=kwargs.get('args',())
        if method=='demoncat_product_binding' and rpc_args and rpc_args[0]=='status':
            self.client._last_status_rows=deepcopy(rows)
        return rows


class DiagnosticProductClient(ProductClient):
    """Use this trusted host class to retain the exact failing status cohort.

    Does not retry, align ranks, suppress an error or issue extra RPCs. This is
    a new host entry, not a monkeypatch to a frozen Worker or request path.
    """
    def __init__(self, executor, *args, **kwargs):
        self._last_status_rows=None;self.status_failures=[]
        super().__init__(_CaptureExecutor(executor,self),*args,**kwargs)

    def _collect_status(self):
        self._last_status_rows=None
        try:return super()._collect_status()
        except Exception as error:
            rows=self._last_status_rows
            report=dict(schema='dcat.product.status_failure.v1',
                time_utc=datetime.now(timezone.utc).isoformat(),error_type=type(error).__name__,
                error=str(error),host_cursor=self.cursor,host_completed=len(self.requests),
                host_enabled=self.enabled,owner=self.identity.owner,
                expected_ranks=[m.get('rank') for m in self.members],ranks=None)
            if isinstance(rows,list):
                report['ranks']=[]
                for row in rows:
                    requests=row.get('requests') if isinstance(row,dict) else None
                    report['ranks'].append(dict(rank=row.get('rank') if isinstance(row,dict) else None,
                        cursor=row.get('cursor') if isinstance(row,dict) else None,
                        returned_completed=len(requests) if isinstance(requests,list) else None,
                        active_request_id=row.get('active_request_id') if isinstance(row,dict) else None,
                        active=None if not isinstance(row,dict) or 'active_request_id' not in row else row['active_request_id'] is not None,
                        more=row.get('more') if isinstance(row,dict) else None,
                        failure=row.get('failure') if isinstance(row,dict) else None,
                        requests=[dict(request_id=r.get('request_id'),coverage=deepcopy(r.get('coverage')))
                                  if isinstance(r,dict) else dict(malformed_type=type(r).__name__)
                                  for r in requests] if isinstance(requests,list) else None))
            self.status_failures.append(report)
            # Transport/CLI adapters may include this bounded control evidence.
            error.status_diagnostic=deepcopy(report)
            raise

    def diagnostics(self):return deepcopy(self.status_failures)
