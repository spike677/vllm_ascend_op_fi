"""Collector window around one real model call, before validation/recovery."""
from contextlib import contextmanager


@contextmanager
def statistics_model_call(collector, model, synchronize, *, rank_id=0):
    # Set up all native msprobe API/module hooks before model.__call__, so root
    # module inputs are retained as well as every child/API. No operator filter.
    # stop also runs after failed start, model error or synchronization failure.
    try:
        collector.start(model=model, rank_id=rank_id)
        yield
    finally:
        try:
            synchronize()
        finally:
            collector.stop()
