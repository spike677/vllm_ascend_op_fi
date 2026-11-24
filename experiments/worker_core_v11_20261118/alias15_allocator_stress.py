"""Acceptance-only allocator preconditioning; never a public fault variant.

Fill a separate allocation before it is freed. The real Attention allocation is
then obtained normally, without writing it. Require and report actual reuse.
This tests nonfinite resilience, not the prevalence/source of natural stale data.
"""
import json
import threading
from pathlib import Path


def prime(allocate, complete, args, kwargs):
    seed = allocate(*args, **kwargs)
    address, size = seed.data_ptr(), seed.numel() * seed.element_size()
    seed.fill_(float('nan'))
    complete()
    del seed
    result = allocate(*args, **kwargs)
    return result, dict(seed_address=address, actual_address=result.data_ptr(),
                        bytes=size, reused=address == result.data_ptr())


def install(worker):
    import torch
    from fault_injection.sites.alias15 import Alias15Site
    if getattr(worker, '_alias15_allocator_stress', False):
        raise RuntimeError('ALLOCATOR_STRESS_ALREADY_INSTALLED')
    worker._alias15_allocator_stress = True
    original_install, original_close = Alias15Site.install, Alias15Site.close
    path = Path('/evidence') / f'alias15-allocator-stress-rank{worker.rank}.jsonl'
    used = [0]
    seeded_requests = set()

    def install_site(site):
        result = original_install(site)
        allocate = site.original_empty
        site._stress_saved_empty = allocate

        def empty(*args, **kwargs):
            requests = tuple(site.runner.input_batch.req_ids)
            if (not site.fault or site.active is None or site.active.get('writer') is not None
                    or threading.get_ident() != site.tid or used[0] >= 16
                    or requests in seeded_requests
                    or kwargs.get('dtype') != torch.bfloat16
                    or not str(kwargs.get('device', '')).startswith('npu')):
                return allocate(*args, **kwargs)
            value, facts = prime(allocate, site.complete, args, kwargs)
            used[0] += 1
            seeded_requests.add(requests)
            facts.update(rank=worker.rank, sequence=used[0], site=site.active['site'],
                         request_ids=list(site.runner.input_batch.req_ids),
                         actual_nonfinite=int((~torch.isfinite(value)).sum().item()),
                         shape=list(value.shape), actual_base_written_by_stress=False,
                         diagnostic='SEPARATE_FREED_ALLOCATION_NAN_PRECONDITION')
            with path.open('a') as f:
                f.write(json.dumps(facts)+'\n')
            return value

        site.original_empty = empty
        return result

    def close_site(site):
        if hasattr(site, '_stress_saved_empty'):
            site.original_empty = site._stress_saved_empty
        return original_close(site)

    Alias15Site.install, Alias15Site.close = install_site, close_site
    return dict(rank=worker.rank, diagnostic_only=True, maximum_allocations=16,
                actual_base_written=False, changes_public_variant=False)
