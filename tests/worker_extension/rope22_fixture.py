"""Self-contained CPU calls shaped like fused RoPE; no native/Worker evidence.

Keep exported Core contracts independent of experiment scripts and receipts.
The real production observer, injector and group are imported by the tests.
"""
import inspect
from types import SimpleNamespace

import torch

from fault_injection.rope_census import NaturalRoPETrace


def kernel_signature(q_ptr, q_row_stride, k_ptr, k_row_stride, cos_ptr, cos_row_stride,
        sin_ptr, sin_row_stride, cos_sin_ptr, cos_sin_row_stride, pos_ptr, num_tokens,
        n_qh, n_kh, hd, rope_dim, pad_n_qh, pad_n_kh, pad_rope_dim,
        BLOCK_SIZE, IS_NEOX_STYLE, USE_COS_SIN):
    raise AssertionError('CPU signature fixture, never a native kernel')


class Compiled:
    name = hash = 'CPU_SIMULATION_NOT_NATIVE'

    def run(self, g0, g1, g2, stream, *values):
        q, _, k, _, _, _, _, _, cache, _, positions, _ = values
        cos, sin = cache.index_select(0, positions).chunk(2, dim=-1)
        cos, sin = cos.repeat(1, 2), sin.repeat(1, 2)
        for item in (q, k):
            value = item.view(item.shape[0], -1)
            half = value.shape[-1] // 2
            value.copy_(value*cos + torch.cat((-value[:, half:], value[:, :half]), dim=-1)*sin)


class Jit:
    def __init__(self):
        self.fn, self.compiled, self.pre_run_hooks = kernel_signature, Compiled(), []
        self.params = [SimpleNamespace(name=name) for name in inspect.signature(self.fn).parameters]
        self.non_constexpr_indices = tuple(range(12))
        self.skip_launch = self.replace_position = False

    def run(self, *args, grid, warmup, **kwargs):
        bound_args = dict(inspect.signature(self.fn).bind(*args, **kwargs).arguments)
        if self.replace_position:
            bound_args['pos_ptr'] = bound_args['pos_ptr'].clone()
        non_constexpr_vals = tuple(bound_args[self.params[i].name] for i in self.non_constexpr_indices)
        kernel, grid_0, grid_1, grid_2, stream = self.compiled, grid[0], 1, 1, 0
        if not self.skip_launch:
            kernel.run(grid_0, grid_1, grid_2, stream, *non_constexpr_vals)
        return kernel


JIT = Jit()


def triton_reader(positions, query, key, cos_sin_cache, head_size, rotary_dim, is_neox_style, offsets=None):
    q, k = query.view(query.shape[0], -1, head_size), key.view(key.shape[0], -1, head_size)
    JIT.run(q, q.stride(0), k, k.stride(0), None, None, None, None, cos_sin_cache,
        cos_sin_cache.stride(0), positions, len(positions), q.shape[1], k.shape[1], head_size,
        rotary_dim, q.shape[1], k.shape[1], rotary_dim, BLOCK_SIZE=q.shape[1],
        IS_NEOX_STYLE=is_neox_style, USE_COS_SIN=True, grid=(len(positions),), warmup=False)
    return q.view(query.shape), k.view(key.shape)


OUTER = torch.library.Library('worker_core_rope22_cpu_fixture', 'DEF')
OUTER.define('opaque(Tensor positions, Tensor(a!) query, Tensor(b!) key, Tensor cos_sin_cache, '
             'int head_size, int rotary_dim, bool is_neox_style) -> (Tensor(a!), Tensor(b!))')
OUTER.impl('opaque', triton_reader, 'CPU')


class SharedRope(torch.nn.Module):
    def __init__(self):
        super().__init__()
        frequency = torch.arange(16.)[:, None] * torch.tensor([.1, .2])[None, :]
        self.register_buffer('cos_sin_cache', torch.cat((frequency.cos(), frequency.sin()), -1))

    def forward(self, positions, q, k):
        return torch.ops.worker_core_rope22_cpu_fixture.opaque(positions, q, k, self.cos_sin_cache, 4, 4, True)


class Consumer(torch.nn.Module):
    def forward(self, q, k):
        return q+k


class Layer(torch.nn.Module):
    def __init__(self, shared):
        super().__init__()
        self.rope, self.consumer = shared, Consumer()

    def forward(self, x, positions):
        q, k = self.rope(positions, x+1, x*2)
        return self.consumer(q, k)


class Body(torch.nn.Module):
    def __init__(self):
        super().__init__()
        shared = SharedRope()
        self.layers = torch.nn.ModuleList([Layer(shared) for _ in range(3)])
        self.skip_last = False

    def forward(self, x, positions):
        for layer in list(self.layers)[:2 if self.skip_last else 3]:
            x = layer(x, positions)
        return x


class DirectModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embed, self.body = torch.nn.Embedding(8, 4), Body()
        with torch.no_grad():
            self.embed.weight.copy_(torch.arange(32.).reshape(8, 4)/8)

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self, input_ids, positions):
        return self.body(self.embed(input_ids), positions)


class Processor(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer('weight', torch.arange(16.).reshape(4, 4)/16)

    def forward(self, selected):
        return torch.mm(selected, self.weight)


class WorkerModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model, self.processor = DirectModel(), Processor()
        self.lm_head_alias = self.model.embed

    def forward(self, input_ids, positions):
        return self.model(input_ids, positions)

    def logits(self, hidden):
        return self.processor(hidden[-1:])


def template(model):
    return NaturalRoPETrace(model, triton_reader,
        {'model.body.layers.0.rope': model.model.body.layers[0].rope}, {}, simulation=True,
        outer_operator='worker_core_rope22_cpu_fixture::opaque', triton_kernel=JIT)
