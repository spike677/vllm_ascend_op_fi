"""Opt-in bounded operator lineage. Observation only; no noise or capture changes."""
import hashlib
import inspect
import math
import weakref
import json
import torch
from torch.utils._python_dispatch import _disable_current_modes
from experiments.worker_core_v11_20261118.dsv4_probe_journal import CallJournal
from experiments.worker_core_v11_20261118.dsv4_probe_support import scheduler_rows

VIEW_OPS = {"alias", "view", "_unsafe_view", "reshape", "transpose", "t", "permute",
            "detach", "slice", "select", "as_strided", "expand", "squeeze", "unsqueeze"}
CONTEXT_FIELDS = ("replace_allreduce", "num_tokens", "padded_num_tokens", "tp_rank",
                  "tp_size", "ep_rank_id", "ep_world_size", "global_bs", "quant_mode",
                  "is_internal_router", "num_experts", "top_k", "num_expert_group",
                  "topk_group", "scoring_func", "renormalize", "use_grouped_topk")


def flatten(value, path=""):
    if isinstance(value, torch.Tensor):
        yield path, value
    elif isinstance(value, (tuple, list)):
        for i, v in enumerate(value):
            yield from flatten(v, path + "/" + str(i))
    elif isinstance(value, dict):
        for k, v in value.items():
            yield from flatten(v, path + "/" + str(k))


def region(t):
    # Exact byte interval only for contiguous storage; strided envelope is NOT proof.
    base = t.untyped_storage().data_ptr()
    try: version = t._version
    except RuntimeError: version = None  # inference tensors have no version counter
    return dict(object_id=id(t), storage_base=base, data_ptr=t.data_ptr(),
                offset=t.storage_offset(), shape=list(t.shape), stride=list(t.stride()),
                dtype=str(t.dtype), device=str(t.device), contiguous=t.is_contiguous(),
                bytes=t.numel()*t.element_size(), version=version)


def overlap(a, b):
    return (a["device"] == b["device"] and a["storage_base"] == b["storage_base"]
            and a["contiguous"] and b["contiguous"]
            and max(a["data_ptr"], b["data_ptr"]) <
            min(a["data_ptr"]+a["bytes"], b["data_ptr"]+b["bytes"]))


def completed_event(t):
    if t.device.type == "cpu":
        return dict(kind="cpu_synchronous", complete=True, stream=None)
    backend = getattr(torch, t.device.type, None)
    if backend is None:
        return dict(kind="unavailable", complete=False, stream=None)
    stream = backend.current_stream(t.device)
    event = backend.Event()
    event.record(stream)
    event.synchronize()
    return dict(kind="device_event_after_operator", complete=bool(event.query()),
                stream=str(getattr(stream, "npu_stream", getattr(stream, "cuda_stream", stream))),
                limitation="Current stream completion; internal-stream contract requires source review")


def options_checked(options, model):
    config = options.get("targeted_observation")
    if config is None:
        return None
    if not isinstance(config, dict) or set(config)-{"max_operations", "float_budget", "integer_budget", "sample_rows", "sample_columns"}:
        raise ValueError("TARGETED_OPTIONS_INVALID")
    if not options.get("target_layers"):
        raise ValueError("TARGETED_EXPLICIT_LAYER_REQUIRED")
    defaults = dict(max_operations=1024, float_budget=262144, integer_budget=262144,
                    sample_rows=2, sample_columns=512)
    limits = dict(max_operations=4096, float_budget=1048576, integer_budget=1048576,
                  sample_rows=4, sample_columns=2048)
    for key, default in defaults.items():
        value = config.get(key, default)
        if type(value) is not int or not 1 <= value <= limits[key]:
            raise ValueError("TARGETED_BUDGET_OUT_OF_RANGE:"+key)
        defaults[key] = value
    protected = set()
    for t in model.parameters() if hasattr(model, "parameters") else []:
        protected.add((str(t.device), t.untyped_storage().data_ptr()))
    return dict(schema="dcat.targeted.observe.v3", config=defaults, operations=[],
                float_remaining=defaults["float_budget"], integer_remaining=defaults["integer_budget"],
                truncated=False, errors=[], protected=protected, runtime_groups=None,
                scheduler_maps={}, attempted_operations=0,
                qualification_granted=False)


def group_snapshot():
    rows = {}
    for module, name in (
        ("vllm.distributed.parallel_state", "get_tp_group"),
        ("vllm.distributed.parallel_state", "get_ep_group"),
        ("vllm_ascend.distributed.parallel_state", "get_mc2_group"),
    ):
        try:
            import importlib
            group = getattr(importlib.import_module(module), name)()
            rows[name] = {k: vars(group).get(k) for k in ("ranks", "rank_in_group", "world_size", "unique_name")}
            rows[name]["source"] = module+"."+name
            rows[name]["type"] = type(group).__module__+"."+type(group).__qualname__
        except Exception as e:
            rows[name] = dict(unavailable=type(e).__name__, reason=str(e)[:240])
    return rows


def export_targeted(state):
    if state is None:
        return None
    return {k:v for k,v in state.items() if k != "protected"}


class TargetedJournal(CallJournal):
    """Trace natural operations, including direct F.linear and fused selectors."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.trace = self.state["targeted"]
        self.live = []  # weak references: do not extend activation lifetime
        self.ordinal = 0
        self.write_epochs = {}
        self.context_taken = False
        if self.trace["runtime_groups"] is None:
            self.trace["runtime_groups"] = group_snapshot()
        mapping = scheduler_rows(self.runner,self.scheduler)
        self.trace["scheduler_maps"][str(self.state["steps"])] = mapping
        self.mapping_sha = hashlib.sha256(json.dumps(mapping,sort_keys=True,separators=(",",":")).encode()).hexdigest()

    def snapshot(self, t):
        row = region(t)
        if (row["device"], row["storage_base"]) in self.trace["protected"]:
            row["sample_reason"] = "PARAMETER_STORAGE_NOT_CAPTURED"
            return row
        if t.numel() == 0:
            return row
        if t.is_floating_point():
            # Explicit coordinate slice, never flatten/copy an entire large activation.
            cfg = self.trace["config"]
            slices = tuple(slice(0, cfg["sample_rows"] if i < t.ndim-1 else cfg["sample_columns"])
                           for i in range(t.ndim))
            sample = t[slices] if slices else t
            budget = "float_remaining"
        elif t.dtype in (torch.bool, torch.int16, torch.int32, torch.int64):
            sample = t
            slices = None
            budget = "integer_remaining"
        else:
            row["sample_reason"] = "DTYPE_METADATA_ONLY"
            return row
        if sample.numel() > self.trace[budget]:
            row["sample_reason"] = "TARGETED_SAMPLE_BUDGET_EXCEEDED"
            self.trace["truncated"] = True
            return row
        self.trace[budget] -= sample.numel()
        cpu = sample.detach().contiguous().cpu()
        raw = cpu.reshape(-1).view(torch.uint8).numpy().tobytes()
        values = cpu.reshape(-1).tolist()
        row.update(sample_shape=list(cpu.shape), sample_origin=[0]*t.ndim,
                   sample_full=sample.numel()==t.numel(), sample_sha256=hashlib.sha256(raw).hexdigest(),
                   sample_values=[v if not isinstance(v,float) or math.isfinite(v) else str(v) for v in values])
        return row

    def context(self, frames):
        out = []
        for f in frames:
            obj = f.frame.f_locals.get("self")
            values = {}
            for key in CONTEXT_FIELDS:
                v = f.frame.f_locals.get(key)
                if v is None and obj is not None:
                    v = vars(obj).get(key) if hasattr(obj, "__dict__") else None
                if v is None or type(v) in (str, bool, int, float):
                    if v is not None:
                        values[key] = v
            if values:
                out.append(dict(function=f.function, file=f.filename, line=f.lineno, values=values))
        return out[:12]

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        tr = self.trace
        record = None
        inputs = list(flatten((args, kwargs), "input"))
        try:
            with _disable_current_modes():
                frames = inspect.stack(context=0)[1:32]
                layers = [self.modules.get(id(f.frame.f_locals.get("self"))) for f in frames]
                selected = bool(set(layers).intersection(self.state["target_layers"]))
                active = []
                for path,t in inputs:
                    actual = region(t)
                    for ref, old, seq, slot in self.live:
                        source = ref()
                        current = region(source) if source is not None else None
                        key = (old["device"],old["storage_base"])
                        if current is not None and current["version"] == old["version"] and self.write_epochs.get(key,0)==old["write_epoch"] and overlap(old, actual):
                            active.append(dict(producer=seq, output=slot, input=path,
                                               overlap_start=max(old["data_ptr"],actual["data_ptr"]),
                                               overlap_end=min(old["data_ptr"]+old["bytes"],actual["data_ptr"]+actual["bytes"]),
                                               version_evidence="tensor_version" if old["version"] is not None else "observed_schema_writes_only"))
                if selected or active:
                    if tr["attempted_operations"] >= tr["config"]["max_operations"]:
                        tr["truncated"] = True
                    else:
                        tr["attempted_operations"] += 1
                        self.ordinal += 1
                        record = dict(id=len(tr["operations"])+1, selected_layer=selected, step=self.state["steps"], rank=self.rank,
                            phase=self.phase, layer_path=[x for x in layers if x is not None],
                            operator=func._schema.name, schema=str(func._schema), context=self.context(frames),
                            input_rows_ref=dict(step=self.state["steps"], sha256=self.mapping_sha),
                            inputs=[dict(path=p, **self.snapshot(t)) for p,t in inputs], edges=active,
                            scalars={str(i):v for i,v in enumerate(args) if v is None or type(v) in (str,int,float,bool)},
                            keyword_scalars={k:v for k,v in kwargs.items() if v is None or type(v) in (str,int,float,bool)})
                        if selected and not self.context_taken:
                            self.context_taken = True
                            try:
                                from vllm_ascend.ascend_forward_context import _EXTRA_CTX
                                record["forward_context"] = {
                                    "source": "vllm_ascend.ascend_forward_context._EXTRA_CTX",
                                    "padded_num_tokens": getattr(_EXTRA_CTX,"padded_num_tokens",None),
                                    "mc2_mask": self.snapshot(_EXTRA_CTX.mc2_mask)
                                        if isinstance(getattr(_EXTRA_CTX,"mc2_mask",None),torch.Tensor) else None}
                            except Exception as e:
                                record["forward_context"] = {"unavailable": type(e).__name__, "reason":str(e)[:240]}
                        del frames
        except Exception as e:
            if len(tr["errors"])<64:tr["errors"].append(dict(stage="targeted_before",error=repr(e)))
            else:tr["truncated"]=True
        # Parent invokes original exactly once and preserves its exceptions/results.
        result = super().__torch_dispatch__(func, types, args, kwargs)
        # Inference tensors have no version counter. Invalidate lineage on every
        # schema-declared write, including operations outside the selected layer.
        try:
            with _disable_current_modes():
                for i,arg in enumerate(func._schema.arguments):
                    if arg.alias_info is not None and arg.alias_info.is_write:
                        value=args[i] if i<len(args) else kwargs.get(arg.name)
                        for _,t in flatten(value):
                            key=(str(t.device),t.untyped_storage().data_ptr())
                            self.write_epochs[key]=self.write_epochs.get(key,0)+1
        except Exception as e:
            self.live.clear()
            if len(tr["errors"])<64:tr["errors"].append(dict(stage="write_tracking",error=repr(e)))
            else:tr["truncated"]=True
        if record is not None:
            try:
                with _disable_current_modes():
                    outputs = list(flatten(result, "output"))
                    record["outputs"] = [dict(path=p, **self.snapshot(t)) for p,t in outputs]
                    kind = func._schema.name.split("::")[-1].split(".")[0]
                    record["consumer_candidate"] = bool(record["edges"]) and kind not in VIEW_OPS
                    record["consumer_semantics"] = "SOURCE_REVIEW_REQUIRED" if record["consumer_candidate"] else "ALIAS_OR_NO_LINEAGE"
                    tensors = [t for _,t in inputs+outputs]
                    devices = {str(t.device):t for t in tensors}
                    record["completion"] = {d:completed_event(t) for d,t in devices.items()}
                    tr["operations"].append(record)
                    self.live = [(r,m,s,p) for r,m,s,p in self.live if r() is not None][-128:]
                    if record["selected_layer"] or kind in VIEW_OPS:
                        for p,t in outputs:
                            meta=region(t);meta["write_epoch"]=self.write_epochs.get((meta["device"],meta["storage_base"]),0)
                            self.live.append((weakref.ref(t),meta,record["id"],p))
            except Exception as e:
                if len(tr["errors"])<64:tr["errors"].append(dict(stage="targeted_after",error=repr(e)))
                else:tr["truncated"]=True
        return result

