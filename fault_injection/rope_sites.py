"""Object-bound call positions for shared modules; no layer-name heuristics."""
from .core import UnsupportedCall


class ModuleSites:
    def __init__(self, model):
        self.model = model
        self.nodes = {}
        self.aliases = {}
        self.edges = {}

        def visit(path, module, ancestors):
            if id(module) in ancestors:
                raise UnsupportedCall('CYCLIC_MODEL_MODULE_GRAPH')
            self.nodes[path] = module
            self.aliases.setdefault(id(module), []).append(path)
            children = tuple(getattr(module, '_modules', {}).items())
            self.edges[path] = children
            for name, child in children:
                if child is not None:
                    visit(name if path == '<root>' else path + '.' + name,
                          child, ancestors | {id(module)})

        visit('<root>', model, set())

    def verify(self, path, module):
        current = self.model
        if path != '<root>':
            for component in path.split('.'):
                current = getattr(current, '_modules', {}).get(component)
                if current is None:
                    raise UnsupportedCall('LIVE_MODULE_REGISTRATION_CHANGED')
        if current is not module or self.nodes.get(path) is not module:
            raise UnsupportedCall('LIVE_MODULE_IDENTITY_CHANGED')
        actual = tuple(getattr(module, '_modules', {}).items())
        expected = self.edges[path]
        if len(actual) != len(expected) or any(a != b or x is not y
                for (a, x), (b, y) in zip(actual, expected)):
            raise UnsupportedCall('LIVE_MODULE_CHILD_REGISTRATION_CHANGED')
        return True

    def resolve(self, module, stack):
        if module is self.model:
            self.verify('<root>', module)
            return '<root>'
        if not stack:
            return None
        parent_path, parent = stack[-1][:2]
        self.verify(parent_path, parent)
        prefix = '' if parent_path == '<root>' else parent_path + '.'
        paths = [path for path in self.aliases.get(id(module), ())
                 if path != parent_path and path.startswith(prefix)]
        if len(paths) != 1:
            # Two registrations below the same active parent cannot be told
            # apart from module identity alone. Never choose the first name.
            raise UnsupportedCall('AMBIGUOUS_OR_FOREIGN_ACTUAL_MODULE_SITE')
        self.verify(paths[0], module)
        return paths[0]

    def owner_sites(self, owners):
        ids = {id(module) for module in owners.values()}
        return {path: module for path, module in self.nodes.items() if id(module) in ids}
