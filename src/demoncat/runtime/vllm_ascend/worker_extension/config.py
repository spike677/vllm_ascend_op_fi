"""Public mechanism selection, without deployment or Worker selectors."""

from dataclasses import asdict, dataclass
import re


@dataclass(frozen=True)
class FaultConfig:
    """Select a catalog entry; constructing this object grants no eligibility.

    ``fault_id='quant'`` addresses the existing unnumbered quantization family.
    This Worker RPC configuration does not change ModelFaultBackend's separate
    configuration or Fault39's provider-specific ABI and qualification rules.
    """

    fault_id: int | str
    variant: str
    implementation_version: str = '1'

    def __post_init__(self):
        if not (type(self.fault_id) is int and self.fault_id > 0
                or type(self.fault_id) is str and self.fault_id == 'quant'):
            raise ValueError('fault_id must be a positive integer or quant')
        if (type(self.variant) is not str or len(self.variant) > 100
                or re.fullmatch(r'[a-z][a-z0-9_]*(?:\.[a-z][a-z0-9_]*)*', self.variant) is None):
            raise ValueError('An explicit catalog variant is required')
        if (type(self.implementation_version) is not str
                or re.fullmatch(r'[A-Za-z0-9_.-]{1,32}', self.implementation_version) is None):
            raise ValueError('An explicit implementation version is required')
        # Reuse the inventory without interpreting its historical acceptance
        # fields as qualification for the current Worker or model instance.
        from fault_injection.catalog import resolve

        resolve(self.mechanism)

    @property
    def mechanism(self):
        return f'{self.fault_id}.{self.variant}'

    def as_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, values):
        if type(values) is not dict:
            raise TypeError('FaultConfig mapping must be a dict')
        # The constructor rejects every unconsumed field, including settings,
        # rank/device/layer/address selectors and native loading instructions.
        return cls(**values)
