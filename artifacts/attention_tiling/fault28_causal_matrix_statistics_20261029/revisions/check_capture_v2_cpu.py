"""Static replay diff and exact installed registration-function CPU mocks. No torch import."""
import ast
import copy
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def functions(tree):
    return {n.name: n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef))}


def main():
    old_path = ROOT / 'revisions/capture_v1/replay.py'
    new_path = ROOT / 'replay.py'
    sdk_path = ROOT / 'revisions/capture_v1/installed_module_processor.py'
    assert sha(old_path) == '2058a953c78f95ac2de7cc03543dc8d85a627664b7808d426c1d98c42aae9e3a'
    assert sha(sdk_path) == '4876e89f7f726cd552fc1bf0e7b7745a898461deec2cb2abf33b02deba7329d7'
    old, new = (ast.parse(path.read_text(encoding='utf-8')) for path in (old_path, new_path))
    compile(new, str(new_path), 'exec')
    before, after = functions(old), functions(new)
    unchanged = []
    for name in before:
        if name not in ('main', 'statistics_receipt'):
            assert ast.dump(before[name]) == ast.dump(after[name]), name
            unchanged.append(name)
    old_main, normalized = before['main'], copy.deepcopy(after['main'])
    class NormalizeHolder(ast.NodeTransformer):
        def visit_Assign(self, node):
            if len(node.targets) == 1 and isinstance(node.targets[0], ast.Name) and node.targets[0].id == 'capture_root':
                assert ast.unparse(node.value) == 'torch.nn.Module()'
                return None
            return self.generic_visit(node)
        def visit_Expr(self, node):
            if isinstance(node.value, ast.Call) and ast.unparse(node.value.func) == 'capture_root.add_module':
                assert ast.unparse(node.value) == "capture_root.add_module('replay', module)"
                return None
            return self.generic_visit(node)
        def visit_Call(self, node):
            if ast.unparse(node.func) == 'debugger.start':
                model = next(k for k in node.keywords if k.arg == 'model')
                assert ast.unparse(model.value) == 'capture_root'
                model.value = ast.Name(id='module', ctx=ast.Load())
            return self.generic_visit(node)
    normalized = NormalizeHolder().visit(normalized)
    assert ast.dump(old_main) == ast.dump(normalized), 'Any additional main-body change is forbidden'
    api_classes = [n for n in ast.walk(new) if isinstance(n, ast.ClassDef) and n.name == 'ReplayModule']
    assert len(api_classes) == 1
    old_class = next(n for n in ast.walk(old) if isinstance(n, ast.ClassDef) and n.name == 'ReplayModule')
    assert ast.dump(api_classes[0]) == ast.dump(old_class)

    sdk = ast.parse(sdk_path.read_text(encoding='utf-8'))
    processor = next(n for n in sdk.body if isinstance(n, ast.ClassDef) and n.name == 'ModuleProcessor')
    methods = []
    for name in ('get_modules_and_names', 'register_module_hook'):
        method = copy.deepcopy(next(n for n in processor.body if isinstance(n, ast.FunctionDef) and n.name == name))
        method.decorator_list = []
        methods.append(method)
    registered = []
    class Module:
        def __init__(self): self.children = {}
        def add_module(self, name, module): self.children[name] = module
        def named_modules(self):
            yield '', self
            yield from self.children.items()
        def __call__(self, *args, **kwargs): return self.forward(*args, **kwargs)
    namespace = {'sys': SimpleNamespace(modules={}), 'ModuleProcessor': SimpleNamespace(module_with_backward_hook={}),
        'is_torch_nn_module': lambda m: isinstance(m, Module), 'logger': SimpleNamespace(warning=lambda *a: None),
        'Const': SimpleNamespace(SEP='.'), 'BaseScope': SimpleNamespace(Module_Type_Module='Module'),
        'wrap_forward_with_hook_safety': lambda m: None,
        'register_forward_pre_hook': lambda m, hook: registered.append((m, hook))}
    exec(compile(ast.Module(body=methods, type_ignores=[]), str(sdk_path), 'exec'), namespace)
    helper = SimpleNamespace(check_is_megatron_module=lambda m: False,
        get_modules_and_names=namespace['get_modules_and_names'], has_register_backward_hook=lambda m: False,
        build_module_hook=lambda name, hook: name)
    native_calls, sentinel = [], object()
    def original_api(*args, **kwargs):
        native_calls.append((args, kwargs))
        return sentinel
    replay_globals = {'torch': SimpleNamespace(nn=SimpleNamespace(Module=Module)),
        'torch_npu': SimpleNamespace(npu_fusion_attention_v3=original_api), 'kwargs': {'scale': 0.08838834764831845}}
    exec(compile(ast.Module(body=[copy.deepcopy(api_classes[0])], type_ignores=[]), str(new_path), 'exec'), replay_globals)
    replay = replay_globals['ReplayModule']()
    namespace['register_module_hook'](helper, replay, object())
    assert registered == [], 'Exact old SDK logic should skip the root replay'
    holder = Module()
    holder.add_module('replay', replay)
    namespace['register_module_hook'](helper, holder, object())
    assert len(registered) == 1 and registered[0][0] is replay and registered[0][1] == 'Module.replay.ReplayModule.'
    inputs = (object(), object(), object())
    returned = replay(*inputs)
    assert returned is sentinel and len(native_calls) == 1 and native_calls[0][0] == inputs
    report = {'schema': 'fault28.causal.capture_v2.cpu_review.v1', 'status': 'CPU_STATIC_REGISTRATION_CHECKS_PASSED',
        'v1_replay_sha256': sha(old_path), 'v2_replay_sha256': sha(new_path), 'sdk_module_processor_sha256': sha(sdk_path),
        'test_sha256': sha(Path(__file__)), 'syntax_valid': True,
        'main_only_changes': ['Create torch.nn.Module capture_root', 'Register unchanged ReplayModule as child named replay',
                              'Pass capture_root to debugger.start; numerical call still directly invokes module'],
        'other_allowed_change': 'Receipt additionally requires the original API construct parent to be a captured ReplayModule node.',
        'unchanged_function_class_asts': unchanged, 'replay_forward_ast_unchanged': True,
        'exact_sdk_registration_function_mock': {'old_root_hook_count': 0, 'new_child_hook_count': 1,
            'new_child_prefix': registered[0][1], 'native_api_spy_call_count': len(native_calls),
            'original_input_objects_preserved': True, 'original_return_object_preserved': True},
        'framework_imports': False, 'containers_started': 0, 'npu_calls': 0,
        'limits': 'Mocks execute exact installed registration-method AST with a minimal module hierarchy. Actual SDK hooks and native Module coverage require fresh stock capture; this is not NPU acceptance.'}
    output = ROOT / 'revisions/capture_v2_cpu_review.json'
    with output.open('x', encoding='utf-8') as stream:
        json.dump(report, stream, indent=2)
        stream.write('\n')
    print(json.dumps({'status': report['status'], 'v2_replay_sha256': report['v2_replay_sha256'], 'report_sha256': sha(output)}))


if __name__ == '__main__':
    main()
