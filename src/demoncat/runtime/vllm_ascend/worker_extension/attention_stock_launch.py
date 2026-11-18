"""Bounded stdlib reader for stock CANN launch evidence, never an ABI decoder.

The current-scope/last-effective-bind algorithm and ELF parser are reused from
our source53 independent diagnostic. Keep raw files as the source of truth;
this module neither imports a device framework nor loads or executes a DSO.
"""
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import struct

SCHEMA = 'demoncat.attention.runtime_observer.v1'
INTEREST = re.compile(r'attention|infer|paged|flash|split|softmax|tiling|kernel|qf16|kvf16', re.I)


def need(value, code):
    if not value: raise ValueError(code)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4*1024**2), b''): value.update(block)
    return value.hexdigest()


def unique(pairs):
    result = {}
    for key, value in pairs:
        need(key not in result, 'DUPLICATE_JSON_KEY:'+key); result[key] = value
    return result


def invalid(value): raise ValueError('NONFINITE_JSON:'+value)


def read_runtime(run):
    path = Path(run)/'runtime/runtime.jsonl'; before = path.stat()
    need(not path.is_symlink() and path.resolve(strict=True).parent == (Path(run)/'runtime').resolve(strict=True),
         'RUNTIME_FILE_LINK_REFUSED')
    need(0 < before.st_size <= 128*1024**2, 'RUNTIME_FILE_BOUND')
    raw = path.read_bytes(); after = path.stat()
    need((before.st_size,before.st_mtime_ns) == (after.st_size,after.st_mtime_ns)
        and raw.endswith(b'\n'), 'RUNTIME_NOT_A_STABLE_COMPLETE_FILE')
    rows = [json.loads(line, object_pairs_hook=unique, parse_constant=invalid) for line in raw.splitlines()]
    need(all(row.get('schema') == SCHEMA for row in rows), 'UNKNOWN_RUNTIME_SCHEMA')
    return rows, dict(path=str(path), bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest(), records=len(rows))


def observe_scope(run, token, identity):
    """Read the stable prefix after the actual API's scope has ended."""
    need(set(identity) == {'pid','python_thread_ident','native_tid','source'}
         and identity['source'] == 'CURRENT_THREAD_GET_IDENT_AND_GET_NATIVE_ID'
         and all(type(identity[k]) is int and identity[k] > 0 for k in ('pid','python_thread_ident','native_tid')),
         'EXPLICIT_THREAD_DOMAINS_REQUIRED')
    rows, prefix = read_runtime(run)
    result = scoped_runtime(run, rows, token, expected_pid=identity['pid'])
    scoped = [row for row in rows if row.get('call_id') == token]
    need(all((row['pid'],row['tid']) == (identity['pid'],identity['native_tid']) for row in scoped),
         'ACTUAL_SCOPE_NATIVE_THREAD_DIFFERS')
    need(not any(row['event'] in ('capture_overflow','unbound_function') for row in scoped)
         and not result['preceding_unscoped_capture_gaps'].get('capture_overflow')
         and all(row.get('result') == 0 for row in scoped if row['event'] in ('device_sync','stream_sync')),
         'ACTUAL_SCOPE_CAPTURE_OR_COMPLETION_INCOMPLETE')
    for launch in result['launches']:
        binary = launch['registered_binary']
        symbols = elf_symbols(binary['path'], binary['sha256'], selected_keys=[launch['tiling_key']])
        entries = symbols.get('actual_selected_entry_symbols', [])
        need(entries and all(row['symbol_type'] == 2 and row['binding'] in (1,2)
             and row['section_index'] != 0 and row['bytes'] > 0 for row in entries),
             'ACTUAL_REGISTERED_ENTRY_SYMBOLS_UNOBSERVED')
        launch['registered_entry_symbols'] = {k:symbols[k] for k in (
            'sha256','bytes','symbols_sha256','actual_selected_keys','actual_selected_entry_symbols')}
    result.update(status='OBSERVED_INTERCEPTED_STOCK_LAUNCHES', runtime_prefix=prefix,
                  runtime_file_can_append=True, thread_identity=dict(identity),
                  raw_runtime_and_registered_elf_retained=True, kernel_argument_abi='NOT_DECODED',
                  native_fault_qualified=False, fault_authorized=False)
    return result


def scoped_runtime(run, rows, token, *, expected_pid):
    """Resolve repeated exact binds, never discard conflicting function history."""
    need(isinstance(token, str) and re.fullmatch('[0-9a-f]{32}', token), 'NATIVE_SCOPE_TOKEN_INVALID')
    scoped = [(i,row) for i,row in enumerate(rows) if row.get('call_id') == token]
    begins = [(i,row) for i,row in scoped if row['event'] == 'begin']
    ends = [(i,row) for i,row in scoped if row['event'] == 'end']
    need(len(begins) == len(ends) == 1, 'SCOPE_BEGIN_END_AMBIGUOUS')
    begin_index, begin = begins[0]; end_index, end = ends[0]
    owner = (begin['pid'], begin['tid'])
    need(owner[0] == expected_pid and begin_index < end_index and begin['monotonic_ns'] < end['monotonic_ns'], 'SCOPE_OWNER_OR_ORDER')
    need(all((row['pid'],row['tid']) == owner and begin_index <= i <= end_index
        and begin['monotonic_ns'] <= row['monotonic_ns'] <= end['monotonic_ns'] for i,row in scoped), 'SCOPED_EVENT_OWNER_OR_ORDER')
    need(not any(row['event'] == 'unbound_function' for _,row in scoped), 'SCOPED_FUNCTION_BINDING_INCOMPLETE')
    outside_gaps = Counter(row['event'] for row in rows if row['event'] in ('capture_overflow','unbound_function')
        and row['pid'] == owner[0] and row['monotonic_ns'] <= end['monotonic_ns'])
    launches = []
    for index, row in scoped:
        if row['event'] != 'launch': continue
        need(row['result'] == 0 and row['snapshot_valid'] is True and row['observer_mutates_arguments'] is False
            and begin['monotonic_ns'] <= row['launch_begin_ns'] <= row['launch_end_ns'] <= row['monotonic_ns'] <= end['monotonic_ns'],
            'LAUNCH_FAILED_SNAPSHOT_INVALID_OR_ORDER')
        need(row['api'] in ('aclrtLaunchKernelWithHostArgs','rtKernelLaunchWithHandle','rtKernelLaunchWithHandleV2')
            and type(row['block_dim']) is int and row['block_dim'] > 0 and type(row['schedule_mode']) is int
            and row['schedule_mode'] >= 0 and str(row['tiling_key']).isdigit(), 'LAUNCH_PROFILE_NOT_OBSERVED')
        registrations = [(i,r) for i,r in enumerate(rows) if r['event'] == 'register' and r['pid'] == owner[0]
            and r['handle'] == row['handle'] and r['monotonic_ns'] <= row['launch_begin_ns']]
        need(bool(registrations), 'REGISTERED_BINARY_JOIN_MISSING')
        same = [(i,r) for i,r in registrations if r['generation'] == row['generation']]
        need(len(same) == 1, 'REGISTERED_GENERATION_AMBIGUOUS')
        reg_index, registration = same[0]
        need(max(registrations, key=lambda item:item[1]['monotonic_ns']) == same[0]
            and reg_index < index and registration['result'] == 0, 'STALE_BINARY_HANDLE_GENERATION')
        name = row['binary_file']; binary = Path(run)/'runtime'/name
        need(name == registration['binary_file'] == 'registered-'+str(owner[0])+'-'+str(row['generation'])+'.bin'
            and re.fullmatch(r'registered-\d+-\d+\.bin', name)
            and not binary.is_symlink() and binary.resolve(strict=True).parent == (Path(run)/'runtime').resolve(strict=True)
            and 0 < binary.stat().st_size == registration['binary_bytes'] <= 128*1024**2, 'REGISTERED_BINARY_FILE_CHANGED')
        binding = None
        if row['api'] == 'aclrtLaunchKernelWithHostArgs':
            need(type(row['function_handle']) is int and row['function_handle'] > 0
                and row['argument_layout'] == 'aclrt_host_placeholders', 'ACTUAL_FUNCTION_HANDLE_MISSING')
            history = [(i,r) for i,r in enumerate(rows) if r['event'] == 'function_bind' and r['pid'] == owner[0]
                and r['function_handle'] == row['function_handle'] and r['monotonic_ns'] <= row['launch_begin_ns']]
            need(bool(history), 'ACTUAL_FUNCTION_JOIN_MISSING')
            identity = (row['handle'], row['generation'], str(row['tiling_key']))
            need(all(r['result'] == 0 and (r['handle'],r['generation'],str(r['tiling_key'])) == identity
                     for _,r in history), 'CONFLICTING_FUNCTION_HANDLE_KEY_OR_GENERATION_HISTORY')
            current = [(i,r) for i,r in history if r.get('call_id') == token]
            need(len(current) == 1, 'CURRENT_SCOPE_FUNCTION_BIND_NOT_UNIQUE')
            bind_index, bound = current[0]
            newest = max(r['monotonic_ns'] for _,r in history)
            need(sum(r['monotonic_ns'] == newest for _,r in history) == 1 and bound['monotonic_ns'] == newest
                and (bound['pid'],bound['tid']) == owner and reg_index < bind_index < index
                and begin['monotonic_ns'] <= bound['monotonic_ns'] <= row['launch_begin_ns'], 'LATEST_EFFECTIVE_BIND_NOT_IN_CURRENT_SCOPE')
            binding = dict(rule='CURRENT_SCOPE_LAST_EFFECTIVE_BIND_WITH_ALL_PRIOR_FUNCTION_IDENTITIES_CONSISTENT',
                function_handle=row['function_handle'], raw_line=bind_index+1, repeated_identical_history=len(history)-1,
                current_scope_binds=1, handle=row['handle'], generation=row['generation'], tiling_key=str(row['tiling_key']))
        args = bytes.fromhex(row['args_hex']); placeholders = bytes.fromhex(row.get('placeholders_hex', ''))
        need(len(args) == row['args_size'] <= 1024**2 and len(placeholders) <= 1024**2, 'RAW_ARGUMENT_SNAPSHOT_SIZE')
        syncs = [r for i,r in scoped if i > index and r['event'] in ('device_sync','stream_sync')
            and r.get('result') == 0 and row['monotonic_ns'] <= r['monotonic_ns'] <= end['monotonic_ns']
            and (r['event'] == 'device_sync' or r.get('stream') == row['stream'])]
        need(bool(syncs), 'LAUNCH_COMPLETION_SYNC_UNOBSERVED')
        launches.append(dict(api=row['api'], pid=owner[0], tid=owner[1], raw_line=index+1,
            call_id=token, handle=row['handle'], generation=row['generation'], function_binding=binding,
            tiling_key=str(row['tiling_key']), block_dim=row['block_dim'], schedule_mode=row['schedule_mode'],
            argument_layout=row['argument_layout'], args_bytes=len(args), args_sha256=hashlib.sha256(args).hexdigest(),
            placeholders_bytes=len(placeholders), placeholders_sha256=hashlib.sha256(placeholders).hexdigest(),
            registered_binary=dict(path=str(binary), bytes=binary.stat().st_size, sha256=sha(binary)),
            kernel_argument_abi='NOT_DECODED', selected_aclnn_library_owner='NOT_OBSERVED'))
    need(0 < len(launches) <= 64, 'NO_OR_UNBOUNDED_INTERCEPTED_LAUNCHES')
    return dict(status='NEW_OFFLINE_RESOLUTION_FROM_PRESERVED_RAW_EVENTS', call_id=token,
        event_counts=dict(sorted(Counter(row['event'] for _,row in scoped).items())), launches=launches,
        preceding_unscoped_capture_gaps=dict(outside_gaps), intercepted_launch_joins_complete=True,
        original_runtime_observation_rewritten=False, stock_private_abi_assumption=False,
        selected_aclnn_owner_observed=False, all_runtime_entrypoints_covered=False)


def elf_symbols(path, expected, *, selected_keys=()):
    path = Path(path)
    need(path.is_file() and 0 < path.stat().st_size <= 128*1024**2 and sha(path) == expected,
         'REGISTERED_ELF_BYTES_CHANGED')
    data = path.read_bytes()
    need(hashlib.sha256(data).hexdigest() == expected, 'REGISTERED_ELF_CHANGED_DURING_READ')
    report = dict(path=str(path), bytes=len(data), sha256=expected, symbols_are_not_abi_proof=True)
    if data[:6] != b'\x7fELF\x02\x01':
        return dict(report, status='NOT_ELF64_LITTLE_ENDIAN_NO_SPECULATIVE_EXTRACTION')
    need(len(data) >= 64, 'TRUNCATED_ELF_HEADER')
    header = struct.unpack_from('<16sHHIQQQIHHHHHH', data)
    _, kind, machine, _, _, _, shoff, flags, ehsize, _, _, shsize, count, names_index = header
    need(ehsize == 64 and shsize == 64 and shoff >= 64 and shoff+64 <= len(data), 'ELF_SECTION_HEADER_INVALID')
    first = struct.unpack_from('<IIQQQQIIQQ', data, shoff)
    if count == 0:count = first[5]
    if names_index == 0xffff:names_index = first[6]
    need(1 <= count <= 4096 and names_index < count and shoff+count*64 <= len(data), 'ELF_SECTION_COUNT_OR_BOUNDS')
    sections = [struct.unpack_from('<IIQQQQIIQQ', data, shoff+index*64) for index in range(count)]
    def content(index):
        section = sections[index]; offset, size = section[4:6]
        need(offset <= len(data) and size <= len(data)-offset, 'ELF_SECTION_OUTSIDE_FILE')
        return data[offset:offset+size]
    names = content(names_index)
    def string(raw, offset):
        need(type(offset) is int and 0 <= offset < len(raw), 'ELF_STRING_OFFSET_INVALID')
        end = raw.find(b'\0', offset)
        need(end >= offset and end-offset <= 8192, 'ELF_STRING_UNTERMINATED_OR_OVERSIZED')
        return raw[offset:end].decode('utf-8', errors='backslashreplace')
    section_names = [string(names, row[0]) for row in sections]
    symbols = []
    for index, section in enumerate(sections):
        if section[1] not in (2, 11):continue  # SHT_SYMTAB / SHT_DYNSYM
        need(section[6] < count and section[9] == 24 and section[5] % 24 == 0,
             'ELF_SYMBOL_TABLE_INVALID')
        strings, entries = content(section[6]), content(index)
        need(len(entries)//24 <= 1000000, 'ELF_SYMBOL_COUNT_BOUND')
        for offset in range(0, len(entries), 24):
            name, info, other, shndx, value, size = struct.unpack_from('<IBBHQQ', entries, offset)
            if not name:continue
            label = string(strings, name)
            if INTEREST.search(label):
                symbols.append(dict(name=label, symbol_type=info & 15, binding=info >> 4,
                                    section_index=shndx, bytes=size, table=section_names[index]))
    symbols.sort(key=lambda item: (item['name'], item['table'], item['section_index']))
    need(all(type(key) is str and key.isdecimal() and len(key) <= 20 for key in selected_keys),
         'ACTUAL_SELECTED_KEYS_INVALID')
    selected_entries = [item for item in symbols if any(re.search('_'+re.escape(key)+r'(?:_|$)',item['name']) for key in selected_keys)]
    internal = [item for item in symbols if not re.match(r'FusedInferAttentionScore_[0-9a-f]+_\d+_', item['name'])]
    # Keep stdout compact; the original ELF and a full-list hash preserve the
    # identity of omitted names. There is no demangler or code execution.
    selected = [dict(item, name=item['name'][:1200], name_truncated=len(item['name']) > 1200) for item in symbols[:48]]
    return dict(report, status='ELF_SYMBOLS_READ', elf_kind=kind, elf_machine=machine, elf_flags=flags,
        section_count=count, sections=[dict(name=section_names[i], bytes=row[5], type=row[1])
            for i,row in enumerate(sections) if INTEREST.search(section_names[i]) or row[1] in (2, 11)],
        matching_symbol_count=len(symbols), symbols_sha256=digest(symbols),
        symbols=selected, omitted_symbols=max(0, len(symbols)-len(selected)),
        actual_selected_keys=sorted(selected_keys), actual_selected_entry_symbols=selected_entries,
        internal_route_symbols=[dict(item, name=item['name'][:1600]) for item in internal[:40]],
        internal_route_symbol_count=len(internal))
