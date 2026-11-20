"""Explicit CPU Native/scheduler fixtures; actual Torch writer/consumer hooks."""
import secrets
import threading
from types import SimpleNamespace as NS

import torch

from demoncat.runtime.vllm_ascend.worker_extension import postwrite41
from demoncat.runtime.vllm_ascend.worker_extension.call_binding import ForwardBinding
from fault_injection.deployments.postwrite import PostwriteRefused, _validate_span, _storage
from fault_injection.deployments.postwrite_evidence import process_start
from fault_injection.deployments.postwrite_kv_inventory import KVWorkloadInventory
from tests.test_postwrite_natural_kv import fixture
from tests.test_postwrite_providers import CpuLease


class OwnedCpuLease(CpuLease):
    def __init__(self, tensor, owner, generation, transport, cache):
        super().__init__(tensor)
        self.owner, self.generation, self.transport = owner, generation, transport
        self.ticket = (owner, generation)
        self.cache, self.cache_storage = cache, _storage(cache)

    def _check(self):
        if (self.owner, self.generation) != self.ticket:
            raise PostwriteRefused('EXPLICIT_CPU_NATIVE_OWNER_MISMATCH')
        if _storage(self.cache) != self.cache_storage:
            raise PostwriteRefused('EXPLICIT_CPU_NATIVE_STORAGE_CHANGED')

    def apply(self, mode, indices):
        self._check()
        return super().apply(mode, indices)

    def recover(self):
        self.transport.restore_attempts += 1
        if self.transport.fail_restore:
            raise RuntimeError('EXPLICIT_CPU_NATIVE_RESTORE_FAILED')
        if not self.closed:
            self._check()
            super().recover()

    def live(self):
        if (self.owner, self.generation) != self.ticket:
            raise PostwriteRefused('EXPLICIT_CPU_NATIVE_OWNER_MISMATCH')
        return super().live()


class CpuSpanTransport:
    transport_kind = 'cpu_test'
    sha256 = 'd' * 64

    def __init__(self):
        self.opened, self.restore_attempts, self.reads = [], 0, 0
        self.fail_restore = self.fail_readback = False

    def open_span(self, cache, offset, elements, owner, generation):
        _validate_span(cache, offset, elements)
        lease = OwnedCpuLease(cache.view(-1).narrow(0, offset, elements), owner, generation, self, cache)
        self.opened.append(lease)
        return lease

    def readback(self):
        self.reads += 1
        if self.fail_readback:
            raise RuntimeError('EXPLICIT_CPU_NATIVE_READBACK_UNAVAILABLE')
        return dict(abi=1, sha256=self.sha256, evidence_kind='cpu_contract')


class DeviceOperator:
    @staticmethod
    def reshape_and_cache(*, key_cache, value_cache, slot_mapping, initial_key, initial_value):
        key_cache.copy_(initial_key)
        value_cache.copy_(initial_value)


class CpuAttention(torch.nn.Module):
    def __init__(self, path, api):
        super().__init__()
        self.fixture = fixture()
        self.impl, self.layer_name, self.api = self.fixture['impl'], path, api
        self.kv_cache = [(self.impl.key_cache, self.impl.value_cache)]
        self.initial_key, self.initial_value = self.impl.key_cache.clone(), self.impl.value_cache.clone()

    def forward(self, query):
        DeviceOperator.reshape_and_cache(key_cache=self.impl.key_cache, value_cache=self.impl.value_cache,
            slot_mapping=self.fixture['metadata'].slot_mapping,
            initial_key=self.initial_key, initial_value=self.initial_value)
        return self.api.npu_fused_infer_attention_score(**dict(self.fixture['actual'], query=query))


class CpuWorkerModel(torch.nn.Module):
    def __init__(self, api, runner):
        super().__init__()
        self.attentions = torch.nn.ModuleList([CpuAttention('actual.' + str(i), api) for i in range(2)])
        self.runner = runner
        self.forward_calls = self.decode_calls = self.consumer_calls = 0
        self.fail_decode = False
        self.output_shift = 0
        def consumer(**actual):
            self.consumer_calls += 1
            return actual['query'] + actual['value'][1, 1].sum()
        api.npu_fused_infer_attention_score = consumer

    def forward(self, input_ids, positions):
        self.forward_calls += 1
        if self.runner.with_prefill:
            return torch.zeros(input_ids.numel(), 16, dtype=torch.bfloat16)
        self.decode_calls += 1
        query = torch.ones(1, 4, 4, dtype=torch.bfloat16)
        for attention in self.attentions:
            query = attention(query)
        if self.fail_decode:
            raise RuntimeError('ACTUAL_CPU_DECODE_FAILED')
        return query.reshape(1, 16) + self.output_shift


def provider_factory(worker, transport, *, tail_elements, group_context=None):
    """Only construction/source/device acceptance is mocked, not the protocol."""
    p = postwrite41.Postwrite41WorkerProvider.__new__(postwrite41.Postwrite41WorkerProvider)
    p.group_context=group_context
    p.worker, p.runner, p.model, p.transport = worker, worker.model_runner, worker.model_runner.model, transport
    p.candidates = [(path, module) for path, module in p.model.named_modules() if isinstance(module, CpuAttention)]
    p.structural_candidates = list(p.candidates)
    p.source = {'explicit_cpu_source_fixture': True}
    p.binding, p.api_module, p.device_operator = ForwardBinding(p.model), p.model.attentions[0].api, DeviceOperator
    p.owner, p.generation, p.thread = secrets.randbits(63) + 1, 0, threading.get_ident()
    p.tail_elements, p.install_epoch = tail_elements, 0
    p.handles, p.patches, p.stack, p.collectors, p.cleanup_receipts, p.hook_owners = [], [], [], [], [], {}
    p.current = p.expected = p.lease = p.collector = p.pending_run = None
    p.phase = p.target = p.coverage = None
    p.qualified = p.lossless = p.fault_attempted = p.installed = False
    p.rows, p.errors, p.history, p.outputs, p.independent_calls = [], [], [], [], []
    p.writers, p.forward_serial = {}, 0
    p.workload, p.process_start = KVWorkloadInventory(), process_start()
    p.model_identity = p._identity()
    return p
