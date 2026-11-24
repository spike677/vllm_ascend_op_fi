"""Typed control refusals and CLI serialization; never wrap inference kernels."""
from .serving_context import ContextRefused
import re


class ProductContractRefused(ValueError):
    exit_code = 4

    def __init__(self, code, rows=()):
        self.code = code
        self.refusals = []
        for row in rows:
            response = row.get('response', {})
            reason = response.get('provider_code') or response.get('reason') or row.get('reason') or row.get('status')
            self.refusals.append(dict(rank=row.get('rank', row.get('worker', {}).get('rank')),
                                      code=reason, status=row.get('status')))
        super().__init__(code + ':' + repr(self.refusals))

    def to_dict(self):
        return dict(status='PRODUCT_CONTRACT_REFUSED', exit_code=4, code=self.code,
                    refusal_codes=sorted({r['code'] for r in self.refusals if isinstance(r['code'], str)}),
                    refusals=self.refusals, error_type=type(self).__name__, reason=str(self))


def product_error_reply(error):
    if isinstance(error, ProductContractRefused):
        return error.to_dict()
    expected = isinstance(error, (ContextRefused, ValueError, PermissionError))
    code=getattr(error, 'code', None)
    if code is None:
        reason=str(error)
        prefix=reason.split(':',1)[0]
        if re.fullmatch(r'[A-Z][A-Z0-9_]+',prefix):code=prefix
        elif reason.startswith('Choose an explicit mechanism variant:'):code='EXPLICIT_MECHANISM_VARIANT_REQUIRED'
        else:code='PRODUCT_CONTRACT_VALIDATION_FAILED' if expected else type(error).__name__
    return dict(status='PRODUCT_CONTRACT_REFUSED' if expected else 'PRODUCT_CONTROL_ERROR',
                exit_code=4 if expected else 5, code=code,
                reason=str(error), error_type=type(error).__name__)


def product_control_reply(operation):
    """Trusted host CLI adapter: use body.exit_code; do not parse exception text."""
    try:
        return dict(status='PRODUCT_CONTROL_OK', exit_code=0, result=operation())
    except Exception as error:
        return product_error_reply(error)
