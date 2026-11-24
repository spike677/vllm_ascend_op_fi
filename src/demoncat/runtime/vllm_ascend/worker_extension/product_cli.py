"""Reference CLI selection from trusted ProductClient capabilities, no static whitelist.

The host owns sockets, authentication, global single-fault exclusion and clients.
This helper never creates a session, installs a permit, or alters admission.
"""
from copy import deepcopy

def choices_from_clients(clients):
    result=[];seen=set()
    for profile,client in sorted(clients.items()):
        rows=client.capabilities()
        if not rows or len({r.get('mechanism') for r in rows})!=1:raise ValueError('PRODUCT_CLI_RANK_CAPABILITY_MISMATCH')
        fields=('mechanism','mode','product_qualified','range','named_variant','registration_sha256','registration_issue','external_provider')
        selected={k:rows[0].get(k) for k in fields}
        if any({k:r.get(k) for k in fields}!=selected for r in rows):raise ValueError('PRODUCT_CLI_RANK_CAPABILITY_MISMATCH')
        name=selected['mechanism']
        if name in seen:raise ValueError('PRODUCT_CLI_AMBIGUOUS_MECHANISM_PROFILE')
        seen.add(name)
        result.append(dict(profile=profile,status=rows[0]['status'],**deepcopy(selected)))
    return result

def select_product_client(clients,mechanism):
    # CLI supplies TEXT. Do not coerce an integer or infer an omitted variant.
    choices=choices_from_clients(clients)
    matched=[r for r in choices if type(mechanism) is str and r['mechanism']==mechanism]
    if len(matched)!=1:raise ValueError('Choose an explicit mechanism variant')
    row=matched[0]
    if row['status']=='PRODUCT_REGISTRATION_REFUSED':raise ValueError(row['registration_issue'])
    return clients[row['profile']]


class TrustedProductSession:
    """Reference handler for a deployment-owned authenticated session socket.

    Peer authentication, socket permissions and server lifetime stay with the
    host. principal must come from that transport, never from request JSON.
    One session serializes enable/disable. Deployment-wide exclusion is still
    enforced by the existing Worker leases and the host's global routing.
    """
    def __init__(self, clients, *, authorize_operator):
        import threading
        if not callable(authorize_operator):raise TypeError('TRUSTED_AUTHORIZATION_CALLBACK_REQUIRED')
        self.clients=dict(clients);self.authorize_operator=authorize_operator
        self.active=None;self.lock=threading.RLock()

    def dispatch(self, request, *, principal):
        from .product_cli_entry import SCHEMA
        def response(code, **body):return dict(schema=SCHEMA,exit_code=code,**body)
        with self.lock:
            try:
                if type(request) is not dict:raise ValueError('PRODUCT_CLI_REQUEST_SCHEMA_INVALID')
                action=request.get('action')
                if (request.get('schema')!=SCHEMA or action not in ('list','enable','status','disable')
                    or set(request)!=({'schema','action','mechanism'} if action=='enable' else {'schema','action'})):
                    raise ValueError('PRODUCT_CLI_REQUEST_SCHEMA_INVALID')
                if self.authorize_operator(principal,'product_cli.'+action) is not True:
                    raise PermissionError('DEPLOYMENT_OPERATOR_NOT_AUTHORIZED')
                if action=='list':return response(0,candidates=choices_from_clients(self.clients))
                if action=='enable':
                    if self.active is not None:raise ValueError('PRODUCT_SESSION_ALREADY_ACTIVE')
                    client=select_product_client(self.clients,request['mechanism'])
                    # select has called capabilities (census attach). Keep failed
                    # sessions for explicit disable/recovery, never hide owners.
                    self.active=client
                    return response(0,result=client.enable())
                if self.active is None:raise ValueError('PRODUCT_SESSION_NOT_ACTIVE')
                if action=='status':return response(0,result=self.active.status())
                recovery=self.active.disable();detached=self.active.detach()
                self.active=None
                return response(0,recovery=recovery,detach=detached)
            except (ValueError,PermissionError) as error:
                cleanup=getattr(self.active,'recovery',None)
                if (self.active is not None and not self.active.bound and type(cleanup) is dict
                        and cleanup.get('enable_rollback') and cleanup.get('resources_zero')):
                    self.active=None
                return response(4,error=str(error))
            except Exception as error:return response(5,error=str(error),recovery_required=self.active is not None)
