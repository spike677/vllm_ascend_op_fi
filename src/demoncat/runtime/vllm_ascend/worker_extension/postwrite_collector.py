"""Remove only owned postwrite API links beneath a pinned msprobe collector.

All direct descriptors, closure cells, ApiTemplate links and retained registry
entries are checked before any change. Collector wrappers and unrelated owners
remain installed. Unknown wrapper code or a different original API refuses.
"""
from dataclasses import dataclass
import hashlib
import inspect
from pathlib import Path
import sys
from types import CodeType, FunctionType

from .serving_context import require

REGISTRY_MODULE = 'msprobe.core.dump.api_dump.api_registry'
TEMPLATE_MODULE = 'msprobe.pytorch.dump.api_dump.api_register'
REGISTRY_SHA256 = '1b6d5cc168034e994d3cb425474a318c8058366b28afe6d8014835ce4f98edeb'
TEMPLATE_SHA256 = '7f1f1b930941b5c49f4674f2174d73d80b7b84c00fb63a1b1ac0529cbe6073ba'


def _nested(code, *names):
    for name in names:
        matches = [item for item in code.co_consts if isinstance(item, CodeType) and item.co_name == name]
        require(len(matches) == 1, 'POSTWRITE_COLLECTOR_CODE_NOT_UNIQUE')
        code = matches[0]
    return code


def _source(module, sha):
    path = Path(module.__file__).resolve(strict=True)
    raw = path.read_bytes()
    require(hashlib.sha256(raw).hexdigest() == sha, 'POSTWRITE_UNREVIEWED_COLLECTOR_SOURCE')
    return path, raw


class _ReviewedCollector:
    def __init__(self):
        self.registry = sys.modules.get(REGISTRY_MODULE)
        self.template_module = sys.modules.get(TEMPLATE_MODULE)
        require(self.registry is not None and self.template_module is not None,
                'POSTWRITE_COLLECTOR_MODULES_REQUIRED')
        path, raw = _source(self.registry, REGISTRY_SHA256)
        template_path, template_raw = _source(self.template_module, TEMPLATE_SHA256)
        self.template = self.template_module.ApiTemplate
        require(self.template.__init__.__globals__ is vars(self.template_module)
                and Path(self.template.__init__.__code__.co_filename).resolve() == template_path,
                'POSTWRITE_COLLECTOR_TEMPLATE_SOURCE_CHANGED')
        require(self.template.__init__.__code__ == _nested(compile(template_raw,
            self.template.__init__.__code__.co_filename, 'exec', dont_inherit=True), 'ApiTemplate', '__init__'),
            'POSTWRITE_COLLECTOR_TEMPLATE_CODE_CHANGED')
        self.codes = {}
        for factory, names, inner in (
                (self.registry.ApiWrapper.wrap_api_func, ('ApiWrapper', 'wrap_api_func'), ('api_function',)),
                (self.registry.ApiRegistry.register_custom_api,
                 ('ApiRegistry', 'register_custom_api'), ('wrap_api_func', 'api_function'))):
            require(inspect.isfunction(factory) and factory.__globals__ is vars(self.registry)
                    and Path(factory.__code__.co_filename).resolve() == path,
                    'POSTWRITE_COLLECTOR_FACTORY_SOURCE_CHANGED')
            require(factory.__code__ == _nested(compile(raw, factory.__code__.co_filename, 'exec',
                    dont_inherit=True), *names), 'POSTWRITE_COLLECTOR_FACTORY_CODE_CHANGED')
            self.codes[_nested(factory.__code__, *inner)] = '.'.join(names)

    def links(self, function, name):
        require(isinstance(function, FunctionType) and function.__globals__ is vars(self.registry)
                and function.__code__ in self.codes, 'TAIL_COLUMN_WRAPPER_OWNERSHIP_CHANGED:' + name)
        cells = dict(zip(function.__code__.co_freevars, function.__closure__ or ()))
        require('api_name' in cells and cells['api_name'].cell_contents.split('.')[-1] == name
                and 'api_func' in cells, 'POSTWRITE_COLLECTOR_API_IDENTITY_CHANGED:' + name)
        child = cells['api_func'].cell_contents
        links = [('cell', cells['api_func'], None, child)]
        factory = self.codes[function.__code__]
        if factory == 'ApiWrapper.wrap_api_func':
            instance = cells['api_instance'].cell_contents
            require(type(instance) is self.template and vars(instance).get('api_func') is child
                    and instance.api_name == cells['api_name'].cell_contents,
                    'POSTWRITE_COLLECTOR_TEMPLATE_TARGET_CHANGED:' + name)
            links.append(('attr', instance, 'api_func', child))
        else:
            require(cells['api_template'].cell_contents is self.template,
                    'POSTWRITE_COLLECTOR_CUSTOM_TEMPLATE_CHANGED:' + name)
        return child, links, factory

    def retained(self, obj, name, collectors):
        register = getattr(self.template_module, 'api_register', None)
        if register is not None:
            require(type(register) is self.registry.ApiRegistry, 'POSTWRITE_COLLECTOR_REGISTRY_OWNER_CHANGED')
            for field in ('ori_api_attr', 'wrapped_api_attr', 'inner_used_ori_attr', 'inner_used_wrapped_attr'):
                for api_map in getattr(register, field).values():
                    for index, entry in enumerate(api_map.get(name, ())):
                        modules, function = entry
                        if any(module is obj for module in modules):
                            yield 'list', api_map[name], index, entry, function, modules
        for collector in collectors:
            service = getattr(collector, 'service', None)
            if service is None:continue
            registry = getattr(service, 'api_register', None)
            require(registry is None or registry is register, 'POSTWRITE_FOREIGN_COLLECTOR_REGISTRY')
            mapping = getattr(service, 'ori_customer_func', {})
            key = str(obj) + '.' + name
            if key in mapping:
                yield 'dict', mapping, key, mapping[key], mapping[key], None


@dataclass
class _Edit:
    kind: str
    container: object
    key: object
    before: object
    after: object

    def current(self):
        if self.kind == 'cell':return self.container.cell_contents
        if self.kind == 'attr':return inspect.getattr_static(self.container, self.key)
        return self.container[self.key]

    def set(self, value):
        if self.kind == 'cell':self.container.cell_contents = value
        elif self.kind == 'attr':setattr(self.container, self.key, value)
        else:self.container[self.key] = value


def restore_owned_api_patches(patches, *, collectors=()):
    edits, seen, observations = [], {}, []
    reviewed = None
    def add(kind, container, key, before, after):
        if before is after:return
        identity = (kind, id(container), key)
        if identity in seen:
            old = seen[identity]
            require(old.before is before and old.after is after, 'POSTWRITE_CONFLICTING_COLLECTOR_RESTORE')
            return
        edit = _Edit(kind, container, key, before, after)
        seen[identity] = edit
        edits.append(edit)
    def walk(function, name, owned, original, visited=()):
        nonlocal reviewed
        if function is owned:return original
        if function is original:return function
        require(len(visited) < 8 and id(function) not in visited, 'POSTWRITE_COLLECTOR_WRAPPER_CYCLE')
        if reviewed is None:reviewed = _ReviewedCollector()
        child, links, factory = reviewed.links(function, name)
        new_child = walk(child, name, owned, original, visited + (id(function),))
        for kind, container, key, before in links:add(kind, container, key, before, new_child)
        observations.append(dict(api=name, collector_factory=factory, collector_callable_preserved=True))
        return function
    for obj, name, original_descriptor, installed_descriptor in reversed(patches):
        current = inspect.getattr_static(obj, name)
        original = original_descriptor.__func__ if isinstance(original_descriptor, staticmethod) else original_descriptor
        installed = installed_descriptor.__func__ if isinstance(installed_descriptor, staticmethod) else installed_descriptor
        if current is installed_descriptor:
            add('attr', obj, name, current, original_descriptor)
        elif current is not original_descriptor:
            replacement = walk(current, name, installed, original)
            require(replacement is current, 'POSTWRITE_UNEXPECTED_DIRECT_COLLECTOR_RESTORE')
        if sys.modules.get(REGISTRY_MODULE) is not None and sys.modules.get(TEMPLATE_MODULE) is not None:
            if reviewed is None:reviewed = _ReviewedCollector()
            for kind, container, key, before, function, modules in reviewed.retained(obj, name, collectors):
                replacement = walk(function, name, installed, original)
                if replacement is not function:
                    after = (modules, replacement) if kind == 'list' else replacement
                    add(kind, container, key, before, after)
    # Whole group preflight: a foreign API cannot cause partial uninstall.
    require(all(edit.current() is edit.before for edit in edits), 'POSTWRITE_RESTORE_PREFLIGHT_CHANGED')
    applied = []
    try:
        for edit in edits:
            edit.set(edit.after)
            applied.append(edit)
    except BaseException:
        for edit in reversed(applied):
            require(edit.current() is edit.after, 'POSTWRITE_RESTORE_ROLLBACK_OWNER_CHANGED')
            edit.set(edit.before)
        raise
    return dict(owned_patches_removed=len(patches), owned_links_removed=len(edits),
                collector_callables_preserved=True, observations=observations,
                registry_sha256=REGISTRY_SHA256 if reviewed is not None else None)
