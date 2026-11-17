"""Read-only actual KV writer and attention call inventory, independent of admission."""


class KVWorkloadInventory:
    def __init__(self, *, max_calls=4096):
        self.max_calls = max_calls
        self.frames, self.current = [], None
        self.total_calls = 0

    def begin(self, forward_id, scheduler):
        if self.current is not None:raise RuntimeError('KV_OBSERVER_FORWARD_ALREADY_ACTIVE')
        self.current = dict(forward_id=str(forward_id), request_ids=list(scheduler['request_ids']),
            scheduler=scheduler, module_calls={}, writers=[], calls=[], complete=False)
        self.frames.append(self.current)

    def module_entered(self, module):
        if self.current is None:raise RuntimeError('KV_OBSERVER_FORWARD_REQUIRED')
        key = str(id(module))
        self.current['module_calls'][key] = self.current['module_calls'].get(key, 0) + 1

    def writer_returned(self, module, receipt):
        if self.current is None:raise RuntimeError('KV_OBSERVER_FORWARD_REQUIRED')
        self.current['writers'].append(dict(module_object_id=id(module), receipt=receipt,
                                           original_returned=True))

    def consumer_entered(self, *, module, path, operator, target_key, eligible, rejection=None):
        if self.current is None or self.total_calls >= self.max_calls:
            raise RuntimeError('KV_OBSERVER_FORWARD_OR_CALL_BOUND')
        self.total_calls += 1
        row = dict(call_id=self.current['forward_id'] + ':' + str(len(self.current['calls']) + 1),
            forward_id=self.current['forward_id'], request_ids=list(self.current['request_ids']),
            module_object_id=id(module), module_path=path, operator=operator, target_key=target_key,
            eligible=eligible, rejection=rejection, original_calls=0, original_returned=False)
        self.current['calls'].append(row)
        return row

    @staticmethod
    def call_original(row, original, args, kwargs):
        if row['original_calls'] != 0:raise RuntimeError('KV_OBSERVER_DUPLICATE_ORIGINAL')
        row['original_calls'] = 1
        result = original(*args, **kwargs)
        row['original_returned'] = True
        return result

    def finish(self, output_sha256):
        if self.current is None:raise RuntimeError('KV_OBSERVER_FORWARD_REQUIRED')
        self.current.update(complete=True, output_sha256=output_sha256)
        result = self.current
        self.current = None
        return result

    def abort(self, reason):
        if self.current is not None:
            self.current['failure'] = reason
            self.current = None
