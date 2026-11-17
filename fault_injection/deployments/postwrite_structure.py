"""Explicit non-call structural containers; never fabricated forward sightings."""
import hashlib
import inspect
from pathlib import Path

import torch


def structural_module_inventory(model, module_calls):
    """Classify only exact, unmodified ModuleLists whose parent/children ran.

    ModuleList stores modules but inherits Module's unimplemented forward. A
    model normally iterates its members instead of calling the container. Unknown
    modules, unused executable layers, subclasses and patched forwards remain
    unresolved; they are never discarded from the structural universe.
    """
    observed={identity for _,identity in module_calls}
    modules=dict(model.named_modules())
    rows=[]
    for path,module in modules.items():
        if id(module) in observed:continue
        forward=getattr(module.forward,'__func__',module.forward)
        children=tuple(id(child) for child in module.children())
        parent_path=path.rsplit('.',1)[0] if '.' in path else ''
        parent=modules.get(parent_path)
        reviewed=(bool(path) and type(module) is torch.nn.ModuleList
                  and forward is torch.nn.Module.forward
                  and forward is torch.nn.modules.module._forward_unimplemented and 'forward' not in vars(module)
                  and parent is not None and id(parent) in observed
                  and bool(children) and set(children).issubset(observed))
        row=dict(path=path,object_identity=id(module),class_name=type(module).__module__+'.'+type(module).__qualname__,
            forward_identity=id(forward),direct_children=list(children),parent_identity=id(parent) if parent else None,
            actual_forward_calls=0,structural_noncall=reviewed,
            reason=('EXACT_MODULELIST_UNIMPLEMENTED_FORWARD_WITH_OBSERVED_MEMBERS' if reviewed
                    else 'NO_ACTUAL_BOUNDARY_IN_HEALTHY_DISCOVERY'))
        if reviewed:
            source=Path(inspect.getsourcefile(torch.nn.ModuleList)).resolve(strict=True)
            row.update(container_source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                       unimplemented_forward_source_sha256=hashlib.sha256(inspect.getsource(forward).encode()).hexdigest())
        rows.append(row)
    return rows
