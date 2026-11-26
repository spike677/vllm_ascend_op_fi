"""Bounded read-only evidence for an isolated acceptance service, never a grant.

Install only around one HTTP round, after product controls, then restore the
exact attributes before the next control. No scheduler or tensor mutation.
"""
import hashlib
import json


class RoundObserver:
    def __init__(self, worker, limit=128):
        self.runner = worker.model_runner
        self.model = self.runner.model
        self.limit = limit
        self.steps = []
        self.logits = []
        self.patches = []
        self.scheduler = None
        self.truncated = False
        self.patch(self.runner, 'execute_model', self.execute)
        self.patch(self.runner, '_preprocess', self.preprocess)
        self.patch(self.model, 'compute_logits', self.compute_logits)

    def patch(self, obj, name, handler):
        present = name in vars(obj)
        saved = vars(obj).get(name)
        original = getattr(obj, name)
        def wrapper(*args, **kwargs):
            return handler(original, *args, **kwargs)
        setattr(obj, name, wrapper)
        self.patches.append((obj, name, present, saved, wrapper))

    def execute(self, original, scheduler, *args, **kwargs):
        self.scheduler = scheduler
        return original(scheduler, *args, **kwargs)

    def preprocess(self, original, *args, **kwargs):
        result = original(*args, **kwargs)
        if len(self.steps) >= self.limit:
            self.truncated = True
            return result
        batch = self.runner.input_batch
        rows = []
        for position, rid in enumerate(batch.req_ids):
            if rid is None:
                continue
            request = self.runner.requests[rid]
            tokens = list(request.prompt_token_ids)
            rows.append(dict(request_id=rid, position=position,
                prompt_sha256=hashlib.sha256(json.dumps(tokens,separators=(',',':')).encode()).hexdigest(),
                prompt_length=len(tokens), computed=int(batch.num_computed_tokens_cpu[position]),
                scheduled=int(self.scheduler.num_scheduled_tokens[rid])))
        self.steps.append(dict(step=len(self.steps), rows=rows))
        return result

    def compute_logits(self, original, *args, **kwargs):
        result = original(*args, **kwargs)
        if result is not None and len(self.logits) < self.limit:
            import torch
            flat = result.detach().reshape(-1,result.shape[-1])
            finite = torch.isfinite(flat)
            values, indices = flat.topk(min(5,flat.shape[-1]),dim=-1)
            counts = (~finite).sum(-1).cpu().tolist()
            ids = indices.cpu().tolist()
            numbers = values.float().cpu().tolist()
            import math
            self.logits.append(dict(step=len(self.steps)-1, shape=list(result.shape),
                rows=[dict(row=i,nonfinite=int(n),top_ids=ids[i],
                    top_values=[v if math.isfinite(v) else str(v) for v in numbers[i]])
                    for i,n in enumerate(counts)]))
        elif result is not None:
            self.truncated = True
        return result

    def stop(self):
        for obj,name,present,saved,wrapper in reversed(self.patches):
            if vars(obj).get(name) is not wrapper:
                raise RuntimeError('EFFECT_OBSERVER_OWNER_CHANGED')
            if present:
                setattr(obj,name,saved)
            else:
                delattr(obj,name)
        return dict(steps=self.steps,logits=self.logits,truncated=self.truncated,
                    methods_restored=True,source='ACTUAL_RUNNER_PREPROCESS_AND_LOGITS')
