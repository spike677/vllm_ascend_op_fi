"""Same-call public-to-typed normal guard. This module cannot authorize F.

Only the freshly compiled inner API invokes the callback. JSON is a bounded
copy of that live callback's arguments, never a saved receipt used as a permit.
The post-copy check reads Python object/geometry metadata, not device values.
"""
from copy import deepcopy
import ctypes as C
import hashlib
import json
from pathlib import Path

from .serving_context import require

CALLBACK = C.CFUNCTYPE(C.c_int, C.c_char_p, C.c_uint32, C.c_void_p, C.c_size_t)
SYMBOLS = ('dcat_tnd28_public_plan_bind_v2', 'dcat_tnd28_public_plan_clear_v2',
           'dcat_tnd28_public_plan_query_v2')
SCALARS = ('num_heads', 'pre_tokens', 'next_tokens', 'num_key_value_heads',
           'sparse_mode', 'inner_precise', 'block_size', 'antiquant_mode',
           'key_antiquant_mode', 'value_antiquant_mode')


def _unique(pairs):
    value = {}
    for key, item in pairs:
        require(key not in value, 'TND_LIVE_TYPED_DUPLICATE_KEY')
        value[key] = item
    return value


def decode(raw):
    require(type(raw) is bytes and 0 < len(raw) <= 32768, 'TND_LIVE_TYPED_BYTE_BOUND')
    return json.loads(raw, object_pairs_hook=_unique,
                      parse_constant=lambda _: (_ for _ in ()).throw(ValueError('NONFINITE_TYPED_VALUE')))


def _tensor(public, typed, name):
    require(type(typed) is dict and type(typed.get('descriptor')) is int and typed['descriptor'] > 0,
            'TND_LIVE_TYPED_DESCRIPTOR:' + name)
    require(typed['dtype'] == (2 if name == 'atten_mask' else 1)
            and typed['format'] == (2 if name == 'atten_mask' else 47)
            and typed['native_storage_offset'] == 0, 'TND_LIVE_TYPED_DTYPE_STORAGE_FORMAT:' + name)
    # 47 is GetStorageFormat() NCL in the pinned 8.5.1 SDK; TND is the API layout.
    for left, right in (('storage_address', 'storage_address'), ('data_address', 'data_address'),
                        ('storage_offset', 'offset'), ('shape', 'shape'), ('stride', 'stride')):
        require(public[left] == typed[right], 'TND_LIVE_TYPED_INPUT_DIFFERS:' + name + ':' + left)
    element = public['element_bytes']
    require(element == (1 if name == 'atten_mask' else 2)
            and public['storage_bytes'] % element == 0
            and typed['storage_shape'] == [public['storage_bytes'] // element]
            and public['data_address'] == public['storage_address'] + public['storage_offset'] * element,
            'TND_LIVE_TYPED_ALLOCATION_VIEW_DIFFERS:' + name)


def validate_prepare(plan, typed):
    """Exact source123 geometry/attributes, applied to this call's real objects."""
    public = plan.public_metadata
    geometry = dict(query=([2048, 16, 128], [2048, 128, 1], 0),
        key=([2048, 8, 128], [1024, 128, 1], 0),
        value=([2048, 8, 128], [4096, 128, 1], 3072),
        atten_mask=([2048, 2048], [2048, 1], 0))
    require(all((public[name]['shape'], public[name]['stride'], public[name]['storage_offset']) == expected
                for name, expected in geometry.items()), 'TND_LIVE_TYPED_EXACT_OBSERVED_GEOMETRY')
    for name in ('key', 'value'):
        container = typed[name]
        require(type(container) is dict and type(container.get('object')) is int and container['object'] > 0
                and type(container.get('members')) is list and len(container['members']) == 1,
                'TND_LIVE_TYPED_SINGLE_CONTAINER:' + name)
    tensors = dict(query=typed['query'], key=typed['key']['members'][0],
                   value=typed['value']['members'][0], atten_mask=typed['mask'])
    for name, value in tensors.items():
        _tensor(public[name], value, name)
    def attr(name, default=None):
        api = plan.before
        if name in api['inputs']:
            value = api['inputs'][name]
            require(value['kind'] == 'scalar', 'TND_LIVE_PUBLIC_SCALAR_REQUIRED:' + name)
            return value['value']
        return api['omitted_argument_defaults'].get(name, default)
    require(typed['layout'] == attr('input_layout') == 'TND'
            and typed['scale'] == attr('scale') == 128 ** -.5
            and typed['softmax_lse'] is False and attr('softmax_lse_flag') is False,
            'TND_LIVE_TYPED_LAYOUT_SCALE_OR_LSE')
    # The exact pinned V3 -> inner wrapper supplies the final three zero fields.
    require(typed['scalars'] == [attr(name, 0) for name in SCALARS] + [0, 0, 0]
            and typed['optionals'] == [0] * 25, 'TND_LIVE_TYPED_SCALARS_OR_OPTIONALS')
    for name, supplied in (('seq_q', 'actual_seq_lengths'), ('seq_kv', 'actual_seq_lengths_kv')):
        require(type(typed[name]) is dict and type(typed[name].get('object')) is int
                and typed[name]['object'] > 0 and typed[name]['values'] == attr(supplied) == [2048],
                'TND_LIVE_TYPED_ACTUAL_LENGTHS:' + name)
    out, lse = typed['output'], typed['lse']
    require(type(out) is dict and out['descriptor'] > 0 and out['storage_address'] > 0
            and out['data_address'] == out['storage_address'] and all(out[key] == value for key, value in
                dict(offset=0, native_storage_offset=0, dtype=1, format=47, shape=[2048, 16, 128],
                     stride=[2048, 128, 1], storage_shape=[4194304]).items()), 'TND_LIVE_TYPED_OUTPUT_GEOMETRY')
    # The real false-LSE placeholder is temporary and distinct from Python's
    # returned empty tensor. No data is read through either recorded address.
    require(type(lse) is dict and lse['descriptor'] > 0 and lse['storage_address'] > 0
            and lse['data_address'] == lse['storage_address'] and all(lse[key] == value for key, value in
                dict(offset=0, native_storage_offset=0, dtype=0, format=2, shape=[0],
                     stride=[0], storage_shape=[0]).items()), 'TND_LIVE_TYPED_FALSE_LSE_PLACEHOLDER')


class LiveTypedPlan:
    """A retained callback owned by the existing live input plan and library."""
    def __init__(self, query, binding, plan, token):
        from .paged33_loading import symbol_owner
        self.query_owner, self.binding, self.plan, self.token = query, binding, plan, token
        self.callback = CALLBACK(self._callback)
        self.bound = self.install_attempted = self.clear_attempted = False
        self.first_error = None
        self.events = []
        self.functions = {}
        library = query.library
        for name in (*SYMBOLS, 'aclnnInnerFusedInferAttentionScoreGetWorkspaceSize'):
            owner = symbol_owner(library, name)
            require(owner['path'] == str(query.path) and owner['sha256'] == query.sha256,
                    'TND_LIVE_TYPED_SYMBOL_OWNER_CHANGED:' + name)
            if name == 'aclnnInnerFusedInferAttentionScoreGetWorkspaceSize':
                self.prepare_owner = owner
                continue
            function = getattr(library, name)
            function.restype = C.c_int
            function.argtypes = ([C.c_char_p, C.c_void_p, C.c_size_t] if name.endswith('query_v2')
                                 else [C.c_char_p, CALLBACK])
            self.functions[name] = function

    def __reduce_ex__(self, protocol):
        raise TypeError('TND_LIVE_TYPED_PLAN_NOT_SERIALIZABLE')

    def install(self):
        require(not self.install_attempted, 'TND_LIVE_TYPED_SINGLE_INSTALL')
        self.binding.check_current(self.plan, read_values=False)
        self.install_attempted = True  # Callback remains held even if the call/readback fails.
        status = self.functions[SYMBOLS[0]](self.token.encode(), self.callback)
        require(status == 0, 'TND_LIVE_TYPED_BIND_RETURN:' + str(status))
        current = self.state()
        self.bound = current['bound'] == 1
        require(status == 0 and self.bound and current['prepare_entered'] == 0,
                'TND_LIVE_TYPED_BIND_FAILED')

    def _callback(self, token, stage, pointer, length):
        try:
            require(token == self.token.encode() and stage in (1, 2) and len(self.events) == stage - 1
                    and pointer and 0 < length <= 32768 and self.first_error is None,
                    'TND_LIVE_TYPED_CALLBACK_ORDER_OR_BOUND')
            self.binding.check_current(self.plan, read_values=False)
            require(self.plan.row['original_calls'] == 1, 'TND_LIVE_TYPED_ORIGINAL_NOT_ACTIVE')
            raw = C.string_at(pointer, length)
            typed = decode(raw)
            function = typed['function']
            require(function['address'] == self.prepare_owner['address']
                    and function['mapped_file_identity_matches'] is True
                    and Path(function['path']).resolve(strict=True) == self.query_owner.path,
                    'TND_LIVE_TYPED_ACTUAL_GENERATED_FUNCTION_CHANGED')
            validate_prepare(self.plan, typed)
            digest = hashlib.sha256(raw).hexdigest()
            require(stage == 1 or digest == self.events[0]['typed_sha256'],
                    'TND_LIVE_TYPED_PREPARE_BYTES_CHANGED')
            self.events.append(dict(stage=stage, typed_sha256=digest,
                                    read_input_values=False, fault_authorized=False))
            return 1
        except BaseException as error:
            self.first_error = self.first_error or repr(error)[:1024]
            return 0  # Never let ctypes turn a callback exception into an indeterminate return value.

    def state(self):
        self.binding._owner()
        output = C.create_string_buffer(4096)
        require(self.functions[SYMBOLS[2]](self.token.encode(), output, len(output)) == 0,
                'TND_LIVE_TYPED_QUERY_FAILED')
        value = decode(output.value)
        require(value.get('schema') == 'demoncat.tnd28.live_typed_plan.v2'
                and value.get('fault_authorized') is False, 'TND_LIVE_TYPED_QUERY_SCHEMA')
        return value

    def clear(self):
        if not self.install_attempted:return
        current = self.state()
        # Readback also closes a previous successful clear whose reply was lost.
        if current['bound'] == 0:
            require(self.clear_attempted, 'TND_LIVE_TYPED_CALLBACK_DISAPPEARED')
            self.bound = False
            return
        require(current['terminal_uncertain'] == 0
                and (current['prepare_entered'] == 0 or current['safe_release'] == 1),
                'TND_LIVE_TYPED_NATIVE_PENDING_RETAINED')
        self.clear_attempted = True
        status = self.functions[SYMBOLS[1]](self.token.encode(), self.callback)
        after = self.state()
        if after['bound'] == 0:self.bound = False
        require(status == 0 and not self.bound, 'TND_LIVE_TYPED_CLEAR_FAILED')

    def snapshot(self):
        return deepcopy(dict(schema='demoncat.tnd28.live_typed_binding.v2', token=self.token,
            events=self.events, first_error=self.first_error, bound=self.bound,
            install_attempted=self.install_attempted, clear_attempted=self.clear_attempted,
            current_value_proof=False, fault_authorized=False))
