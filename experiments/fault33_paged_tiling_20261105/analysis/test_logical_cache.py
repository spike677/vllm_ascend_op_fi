"""Meaningful CPU checks of layout equivalence, target alignment and full output comparison."""
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from stock_probe import compare, make_logical, payload, physical_cache


def main():
    import torch
    checks = []
    logical = torch.arange(2 * 2 * 257 * 16, dtype=torch.float32).reshape(2, 2, 257, 16)
    for size in (16, 64, 128, 256, 512):
        cache, table = physical_cache(logical, size)
        recovered = cache[table.long()].permute(0, 2, 1, 3, 4).contiguous().view(2, 2, -1, 16)
        assert torch.equal(recovered[:, :, :257], logical)
        assert torch.count_nonzero(recovered[:, :, 257:]) == 0
        assert torch.unique(table).numel() == table.numel()
        assert payload(cache) != payload(logical)
        checks.append('bijective_blocks_tail_padding_' + str(size))
    base = {'batch': 1, 'q_heads': 4, 'kv_heads': 2, 'kv_length': 257, 'head_dim': 16, 'seed': 330011}
    b1 = make_logical(base)
    b8 = make_logical({**base, 'batch': 8})
    assert all(payload(one) == payload(eight[:1]) for one, eight in zip(b1, b8))
    assert all(payload(eight[:1]) != payload(eight[1:2]) for eight in b8)
    checks.append('same_target_different_batch_and_distinct_peers')
    a = torch.zeros((2, 4, 1, 16), dtype=torch.float16)
    b = a.clone(); b[1, 2, 0, 7] = 1
    r = compare(a, b)
    assert r['different_elements'] == 1 and r['request_different_elements'] == [0, 1] and not r['bitwise_equal']
    checks.append('complete_output_covers_nonzero_request_and_nonlast_channel')
    b = a.clone(); b[0, 0, 0, 0] = -0.0
    r = compare(a, b)
    assert r['different_elements'] == 0 and r['bitwise_equal'] is False
    checks.append('signed_zero_not_misreported_bitexact')
    p = Path(__file__)
    print(json.dumps({'schema': 'fault33.logical_cache.cpu.v1', 'status': 'PASS_CPU_LAYOUT_CHECKS',
                      'checks': checks, 'npu_calls': 0, 'source_sha256': hashlib.sha256(p.read_bytes()).hexdigest()}))


if __name__ == '__main__':
    main()
