"""Common scope checks for provider bridges. Native state stays provider-owned."""


class ProviderError(RuntimeError):
    def __init__(self, code, detail=None):
        self.code = code
        super().__init__(code if detail is None else f'{code}: {detail}')


class Scope:
    def __init__(self, worker, model, identity):
        self.worker = worker
        self.model = model
        self.identity = dict(identity)
        self._ticket = None
        self._registered = False

    def bind(self, owner, generation):
        if self._ticket is None:
            self._ticket = (owner, generation)
        self.verify(owner, generation)

    def verify(self, owner, generation):
        if self._ticket != (owner, generation):
            raise ProviderError('PROVIDER_OWNER_MISMATCH')


def reject_unknown(settings, keys=()):
    if not isinstance(settings, dict) or set(settings) - set(keys):
        raise ProviderError('UNSUPPORTED_PROVIDER_SETTINGS')


def require_idle(state, *, release_owner=False):
    if (state.get('enabled') is not False or state.get('inflight') != 0
            or state.get('leases') != 0
            or release_owner and state.get('native_registered') is not False):
        raise ProviderError('NATIVE_RECOVERY_PENDING')
