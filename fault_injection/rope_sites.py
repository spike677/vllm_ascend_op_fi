"""Object-bound call positions for shared modules; no layer-name heuristics."""
import inspect

from .core import UnsupportedCall


class ModuleSites:
    def __init__(self, model):
        self.model = model
        self.nodes = {}
        self.aliases = {}
        self.edges = {}
        self.methods = {}

        def visit(path, module, ancestors):
            if id(module) in ancestors:
                raise UnsupportedCall('CYCLIC_MODEL_MODULE_GRAPH')
            self.nodes[path] = module
            self.aliases.setdefault(id(module), []).append(path)
            self.methods.setdefault(id(module), self._methods(module))
            children = tuple(getattr(module, '_modules', {}).items())
            self.edges[path] = children
            for name, child in children:
                if child is not None:
                    visit(name if path == '<root>' else path + '.' + name,
                          child, ancestors | {id(module)})

        visit('<root>', model, set())

    @staticmethod
    def _methods(module):
        from .module_faults import _original_standard_forward
        methods={'forward':_original_standard_forward(module)}
        if hasattr(module,'_forward_method'):methods['_forward_method']=module._forward_method
        result={}
        for name,method in methods.items():
            function=getattr(method,'__func__',None)
            if getattr(method,'__self__',None) is module and getattr(function,'__code__',None) is not None:
                result[name]=(function,function.__code__)
        return result

    def verify(self, path, module):
        current = self.model
        if path != '<root>':
            for component in path.split('.'):
                current = getattr(current, '_modules', {}).get(component)
                if current is None:
                    raise UnsupportedCall('LIVE_MODULE_REGISTRATION_CHANGED')
        if current is not module or self.nodes.get(path) is not module:
            raise UnsupportedCall('LIVE_MODULE_IDENTITY_CHANGED')
        methods=self._methods(module);expected_methods=self.methods[id(module)]
        if methods.keys()!=expected_methods.keys() or any(
                methods[name][0] is not expected_methods[name][0] or methods[name][1] is not expected_methods[name][1]
                for name in methods):
            raise UnsupportedCall('LIVE_ANCESTOR_FORWARD_CHANGED')
        actual = tuple(getattr(module, '_modules', {}).items())
        expected = self.edges[path]
        if len(actual) != len(expected) or any(a != b or x is not y
                for (a, x), (b, y) in zip(actual, expected)):
            raise UnsupportedCall('LIVE_MODULE_CHILD_REGISTRATION_CHANGED')
        return True

    def verify_ancestry(self,path):
        self.verify('<root>',self.model)
        parts=[]
        for component in path.split('.') if path!='<root>' else ():
            parts.append(component);current='.'.join(parts)
            self.verify(current,self.nodes[current])

    def _frame_paths(self,paths,module,stack):
        # Some real model wrappers call forward() directly and bypass Module
        # hooks. Only an executing frame of the *actual frozen bound method*
        # can supply that missing ancestor. A local variable named self, a
        # matching class name, or source text alone cannot establish ancestry.
        frame=inspect.currentframe().f_back
        active_path=stack[-1][0]
        prefix='' if active_path=='<root>' else active_path+'.'
        try:
            while frame is not None:
                owner=frame.f_locals.get('self')
                if owner is not module and id(owner) in self.aliases:
                    methods=self.methods[id(owner)]
                    if any(frame.f_code is code for _,code in methods.values()):
                        ancestors=[p for p in self.aliases[id(owner)]
                            if p==active_path or p.startswith(prefix)]
                        if ancestors:
                            for p in ancestors:self.verify_ancestry(p)
                            paths=[p for p in paths if any(a=='<root>' or p.startswith(a+'.') for a in ancestors)]
                frame=frame.f_back
        finally:
            del frame
        return paths

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
        paths=self._frame_paths(paths,module,stack)
        if len(paths) != 1:
            # Two registrations below the same active parent cannot be told
            # apart from module identity alone. Never choose the first name.
            raise UnsupportedCall('AMBIGUOUS_OR_FOREIGN_ACTUAL_MODULE_SITE')
        self.verify_ancestry(paths[0])
        return paths[0]

    def owner_sites(self, owners):
        ids = {id(module) for module in owners.values()}
        return {path: module for path, module in self.nodes.items() if id(module) in ids}
