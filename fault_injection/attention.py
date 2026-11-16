"""Model-structural routing through Transformers' public AttentionInterface.

Discovery is deliberately conservative. Registering a name is not evidence that
a model used it, and routing is not evidence that a native fault actually hit.
No model class, layer path, head count, or layer count selects a target here.
"""
from __future__ import annotations

import ast
import copy
import inspect
import sys
import textwrap
import threading
import uuid

from .core import AttentionCall, UnsupportedCall

_SESSION_LOCK = threading.Lock()


def _interfaces():
    from transformers import AttentionInterface
    from transformers.masking_utils import AttentionMaskInterface
    return AttentionInterface, AttentionMaskInterface


def _config_selector(node):
    return (isinstance(node, ast.Attribute) and node.attr == "_attn_implementation"
            and isinstance(node.value, ast.Attribute) and node.value.attr == "config"
            and isinstance(node.value.value, ast.Name) and node.value.value.id == "self")


def _function(module):
    forward = getattr(module, "forward", None)
    return inspect.unwrap(getattr(forward, "__func__", forward))


def _resolve(module):
    """Resolve a registry and eager default from actual forward source, not names."""
    function = _function(module)
    try:
        tree = ast.parse(textwrap.dedent(inspect.getsource(function)))
        namespace = function.__globals__
    except (TypeError, OSError, AttributeError, SyntaxError):
        return None, "forward_source_unavailable"
    matches = []
    for node in ast.walk(tree):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr == "get_interface" and isinstance(node.func.value, ast.Name)
                and node.args and _config_selector(node.args[0])):
            default = node.args[1] if len(node.args) > 1 else next(
                (k.value for k in node.keywords if k.arg == "default"), None)
            if isinstance(default, ast.Name):
                registry = namespace.get(node.func.value.id)
                fallback = namespace.get(default.id)
                if hasattr(registry, "get_interface") and callable(fallback):
                    matches.append((registry, fallback))
        # Legacy shared-registry dispatch: default assignment then registry[key].
        if (isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name)
                and _config_selector(node.slice)):
            registry = namespace.get(node.value.id)
            targets = []
            for assignment in ast.walk(tree):
                if isinstance(assignment, ast.Assign) and assignment.value is node:
                    targets.extend(t.id for t in assignment.targets if isinstance(t, ast.Name))
                elif isinstance(assignment, ast.AnnAssign) and assignment.value is node and isinstance(assignment.target, ast.Name):
                    targets.append(assignment.target.id)
            defaults = []
            for assignment in ast.walk(tree):
                value = getattr(assignment, "value", None)
                assigned = getattr(assignment, "targets", [getattr(assignment, "target", None)])
                if (isinstance(assignment, (ast.Assign, ast.AnnAssign)) and isinstance(value, ast.Name)
                        and any(isinstance(t, ast.Name) and t.id in targets for t in assigned)
                        and callable(namespace.get(value.id))):
                    defaults.append(namespace[value.id])
            if hasattr(registry, "get_interface") and len(defaults) == 1:
                matches.append((registry, defaults[0]))
    unique = {(id(r), id(f)): (r, f) for r, f in matches}
    if len(unique) != 1:
        return None, "shared_registry_dispatch_not_proven" if not unique else "ambiguous_registry_dispatch"
    if getattr(module, "config", None) is None:
        return None, "missing_attention_config"
    return next(iter(unique.values())), "source_verified_registry_candidate"


def _suspect(module):
    # This only includes unsupported modules in a diagnostic list. It never
    # authorizes injection; only _resolve plus runtime membership does that.
    return "attention" in type(module).__name__.lower() or all(
        hasattr(module, name) for name in ("q_proj", "k_proj", "v_proj"))


def _discover(model):
    if not callable(getattr(model, "named_modules", None)):
        raise TypeError("model must expose named_modules()")
    candidates, rejected = [], []
    seen = set()
    for path, module in model.named_modules():
        if id(module) in seen:
            continue
        seen.add(id(module))
        resolved, reason = _resolve(module)
        if resolved:
            candidates.append((path or "<root>", module, *resolved))
        elif _suspect(module):
            rejected.append({"path": path or "<root>", "class": type(module).__qualname__, "reason": reason})
    return candidates, rejected


def inspect_model(model):
    """Read-only discovery; this never registers a backend or claims coverage."""
    candidates, rejected = _discover(model)
    return {"schema": "fault_injection.attention_discovery.v1",
            "candidates": [{"path": path, "class": type(module).__qualname__,
                            "original_backend": getattr(module.config, "_attn_implementation", None),
                            "status": "candidate_not_observed"} for path, module, _, _ in candidates],
            "unsupported_modules": rejected, "actual_calls_observed": 0,
            "native_kernel_supported": None, "npu_validated": False}


class AttentionSession:
    """Install a root-forward-scoped route; call the original model normally.

    `backend.check(call)` must validate the actual layout, mask, precision, cache
    and native capability before `backend.run(call)` is allowed. Candidate
    discovery itself cannot certify any of these.
    """
    _cleanup_retryable = True

    def __init__(self, model, backend, unsupported="error"):
        if unsupported not in ("error", "passthrough"):
            raise ValueError("unsupported must be error or passthrough")
        self.model, self.backend, self.unsupported = model, backend, unsupported
        self._candidates, self._rejected = _discover(model)
        self._by_id = {id(module): (path, module, registry, fallback)
                       for path, module, registry, fallback in self._candidates}
        self._events = []
        self._configs = []
        self._registrations = []
        self._handles = []
        self._owner = None
        self._root_entries = []
        self._entry_exceptions = {}
        self._forward_index = None
        self._forwards = []
        self._rejected_roots = []
        self._active = False
        self._locked = False
        self._closed = False

    def __enter__(self):
        if self._active or self._closed:
            raise RuntimeError("session cannot be entered twice")
        if not self._candidates:
            raise UnsupportedCall("No source-verified shared AttentionInterface calls found")
        if self._rejected and self.unsupported == "error":
            raise UnsupportedCall("Unresolved attention modules: " + ", ".join(r["path"] for r in self._rejected))
        if not _SESSION_LOCK.acquire(blocking=False):
            raise RuntimeError("another AttentionSession owns the process registry")
        self._locked = True
        self._owner = threading.get_ident()
        try:
            attention_type, mask_type = _interfaces()
            # Public registration has no public global unregister in 5.5.4.
            # Require the known ownership-checked cleanup contract before edits.
            for interface in (attention_type, mask_type):
                if not isinstance(getattr(interface, "_global_mapping", None), dict):
                    raise UnsupportedCall("unsupported registry cleanup contract")
            groups = {}
            for path, module, registry, fallback in self._candidates:
                if not isinstance(registry, attention_type):
                    raise UnsupportedCall(f"{path}: registry is not the installed AttentionInterface")
                config = module.config
                old = getattr(config, "_attn_implementation", None)
                if old not in ("eager", "sdpa"):
                    raise UnsupportedCall(f"{path}: original backend {old!r} mask/layout equivalence is not validated")
                if old not in mask_type._global_mapping:
                    raise UnsupportedCall(f"{path}: original mask factory unavailable")
                original = registry.get_interface(old, fallback)
                if not callable(original):
                    raise UnsupportedCall(f"{path}: original attention callable unavailable")
                self._by_id[id(module)] = (path, module, original)
                groups.setdefault(id(config), (config, old))
            rejected_paths = {r["path"] for r in self._rejected}
            for path, module in self.model.named_modules():
                if ((path or "<root>") in rejected_paths
                        and id(getattr(module, "config", None)) in groups):
                    raise UnsupportedCall(f"{path}: unresolved module shares a selected attention config")
            if not all(callable(getattr(self.model, name, None)) for name in
                       ("register_forward_pre_hook", "register_forward_hook")):
                raise UnsupportedCall("Root model must expose forward hooks; direct forward() calls are not covered")
            self._handles.append(self.model.register_forward_pre_hook(self._root_enter, prepend=True))
            self._handles.append(self.model.register_forward_hook(self._root_leave, always_call=True))
            # Preflight all groups before performing any mutation.
            for config, old in groups.values():
                route = "demoncat_" + uuid.uuid4().hex
                wrapper = self._dispatch
                attention_type.register(route, wrapper)
                self._registrations.append((attention_type, route, wrapper))
                mask_factory = mask_type._global_mapping[old]
                mask_type.register(route, mask_factory)
                self._registrations.append((mask_type, route, mask_factory))
                self._configs.append((config, old, route))
                config._attn_implementation = route
            self._active = True
            return self
        except BaseException as error:
            try:
                self.close()
            except BaseException as cleanup:
                raise RuntimeError('Attention installation failed: '+repr(error)+'; cleanup failed: '+repr(cleanup)) from error
            raise

    def _root_enter(self, module, args):
        if not self._active:
            return
        if threading.get_ident() != self._owner:
            self._rejected_roots.append({"reason": "concurrent_root_forward"})
            raise RuntimeError("AttentionSession root forward must run on its owning thread")
        if self._forward_index is not None:
            # PyTorch invokes an always_call posthook even if this prehook
            # raises. Its sentinel must not end the still-active outer call.
            self._root_entries.append(None)
            self._rejected_roots.append({"reason": "recursive_root_forward",
                                         "forward": self._forward_index})
            raise RuntimeError("Recursive root forward is not supported")
        index = len(self._forwards)
        self._forwards.append({"index": index, "status": "running"})
        self._root_entries.append(index)
        self._entry_exceptions[index] = sys.exc_info()[1]
        self._forward_index = index

    def _root_leave(self, module, args, output):
        if threading.get_ident() != self._owner or not self._root_entries:
            return
        index = self._root_entries.pop()
        if index is None:
            return
        entering_exception = self._entry_exceptions.pop(index)
        active_exception = sys.exc_info()[1]
        failed = active_exception is not None and active_exception is not entering_exception
        self._forwards[index]["status"] = "returned" if output is not None and not failed else "no_output_or_exception"
        self._forward_index = None

    def _dispatch(self, module, query, key, value, attention_mask=None,
                  dropout=0.0, scaling=None, **kwargs):
        if not self._active:
            raise RuntimeError("attention route used outside its owning session")
        entry = self._by_id.get(id(module))
        if entry is None or entry[1] is not module:
            # A second model can share a config object. Never inject into it.
            raise UnsupportedCall("Unidentified module reached this model's attention route")
        path, _, original = entry
        if threading.get_ident() != self._owner or self._forward_index is None:
            # Sharing a module/config with another model does not authorize
            # injection outside the selected root's actual forward. Preserve
            # that module's original backend and do not count these calls.
            return original(module, query, key, value, attention_mask,
                            dropout=dropout, scaling=scaling, **kwargs)
        call = AttentionCall(path=path, module=module, query=query, key=key, value=value,
                             attention_mask=attention_mask, dropout=dropout, scaling=scaling, kwargs=dict(kwargs))
        shapes = [list(getattr(t, "shape", ())) for t in (query, key, value)]
        slots = list(range(int(shapes[0][0]))) if len(shapes[0]) == 4 else None
        event = {"sequence": len(self._events), "forward": self._forward_index,
                 "path": path, "qkv_shapes": shapes,
                 "request_slots": slots, "request_identity": "batch_slots_only_not_scheduler_ids",
                 "status": "observed", "reason": None}
        self._events.append(event)
        try:
            decision = self.backend.check(call)
            if not decision.supported:
                event.update(status="unsupported", reason=decision.reason)
                if self.unsupported == "error":
                    raise UnsupportedCall(f"{path}: {decision.reason}")
                output = original(module, query, key, value, attention_mask,
                                  dropout=dropout, scaling=scaling, **kwargs)
                event["status"] = "passthrough"
                return output
            result = self.backend.run(call)
            event["status"] = "routed"
            return result
        except BaseException as error:
            if event["status"] == "observed":
                event.update(status="failed", reason=type(error).__name__ + ": " + str(error))
            raise

    def report(self):
        by_path = {}
        for path, _, _, _ in self._candidates:
            events = [e for e in self._events if e["path"] == path]
            by_path[path] = {"observed_calls": len(events),
                             "routed_calls": sum(e["status"] == "routed" for e in events),
                             "request_slot_executions": sum(len(e["request_slots"] or []) for e in events if e["status"] == "routed")}
        forwards = []
        for forward in self._forwards:
            rows = {}
            for path in by_path:
                events = [e for e in self._events if e["path"] == path and e["forward"] == forward["index"]]
                rows[path] = {"observed_calls": len(events),
                              "routed_calls": sum(e["status"] == "routed" for e in events)}
            covered = bool(rows) and forward["status"] == "returned" and all(
                row["observed_calls"] > 0 and row["observed_calls"] == row["routed_calls"]
                for row in rows.values())
            forwards.append(dict(forward, modules=rows, all_discovered_candidates_routed=covered))
        complete = bool(forwards) and not self._rejected and not self._rejected_roots and all(
            row["all_discovered_candidates_routed"] for row in forwards)
        return {"schema": "fault_injection.attention_coverage.v1", "active": self._active,
                "candidate_modules": len(by_path), "modules": by_path,
                "unsupported_modules": copy.deepcopy(self._rejected), "events": copy.deepcopy(self._events),
                "forwards": forwards, "rejected_root_calls": copy.deepcopy(self._rejected_roots),
                "all_discovered_candidates_routed": complete,
                "native_fault_hit_proven": False, "npu_validation_claimed": False,
                "limits": ["Source discovery may reject custom/compiled dispatch; it does not prove absence of hidden attention.",
                           "Request slots describe actual BNSD calls, not external scheduler request identities.",
                           "Every discovered candidate must be routed in every root forward; skipped branches stay unobserved.",
                           "Direct forward() and shared-child calls outside the selected root use the original backend and do not count.",
                           "Backend execution coverage is separate from native kernel launch and fault-hit verification."]}

    def assert_coverage(self):
        report = self.report()
        if not report["all_discovered_candidates_routed"]:
            raise UnsupportedCall("Attention coverage incomplete; inspect report() for unobserved, unsupported or failed calls")
        return report

    def close(self):
        if self._closed:
            return
        if self._locked and threading.get_ident() != self._owner:
            raise RuntimeError("AttentionSession must be closed on its owning thread")
        if self._forward_index is not None or self._root_entries:
            raise RuntimeError('Attention session cannot close inside an active model forward')
        self._active = False
        errors = []
        remaining = []
        for handle in reversed(self._handles):
            try:
                handle.remove()
            except Exception as exc:
                errors.append(str(exc))
                remaining.append(handle)
        self._handles = list(reversed(remaining))
        remaining = []
        for config, old, route in reversed(self._configs):
            try:
                current = getattr(config, "_attn_implementation", None)
                if current == route:
                    config._attn_implementation = old
                elif current != old:
                    raise RuntimeError('attention config was changed by another owner')
            except Exception as exc:
                errors.append(str(exc))
                remaining.append((config, old, route))
        self._configs = list(reversed(remaining))
        remaining = []
        for interface, route, owned in reversed(self._registrations):
            try:
                if interface._global_mapping.get(route) is owned:
                    del interface._global_mapping[route]
                elif route in interface._global_mapping:
                    raise RuntimeError('owned attention registration was replaced externally')
            except BaseException as exc:
                errors.append(str(exc))
                remaining.append((interface, route, owned))
        self._registrations = list(reversed(remaining))
        if errors:
            raise RuntimeError("; ".join(errors))
        self._closed = True
        if self._locked:
            _SESSION_LOCK.release()
            self._locked = False

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
        return False
