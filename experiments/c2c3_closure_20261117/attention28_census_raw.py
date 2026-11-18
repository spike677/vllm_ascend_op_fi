"""Read preserved stock scope/ELF bytes, including cached auxiliary launches.

No Native execution, ABI decode, symbol demangler or expected-key allowlist.
This is an offline resolution; it never rewrites the original capture gap.
"""
from collections import Counter
import hashlib
from pathlib import Path
import re
import struct

import arch22_normal_bridge_evidence as E


def need(value,reason):
    if not value:raise ValueError(reason)


def scope(run,rows,token,*,expected_pid):
    need(type(token) is str and re.fullmatch('[0-9a-f]{32}',token), 'NATIVE_SCOPE_TOKEN_INVALID')
    selected=[(i,r) for i,r in enumerate(rows) if r.get('call_id') == token]
    starts=[(i,r) for i,r in selected if r['event'] == 'begin'];ends=[(i,r) for i,r in selected if r['event'] == 'end']
    need(len(starts) == len(ends) == 1,'SCOPE_BEGIN_END_AMBIGUOUS')
    first,begin=starts[0];last,end=ends[0];owner=(begin['pid'],begin['tid'])
    need(owner[0] == expected_pid and first < last and begin['monotonic_ns'] < end['monotonic_ns']
        and all((r['pid'],r['tid']) == owner and first <= i <= last
            and begin['monotonic_ns'] <= r['monotonic_ns'] <= end['monotonic_ns'] for i,r in selected),
        'SCOPED_EVENT_OWNER_OR_ORDER')
    gaps=Counter(r['event'] for r in rows if r['pid'] == owner[0] and r['monotonic_ns'] <= end['monotonic_ns']
                 and r['event'] in ('unbound_function','capture_overflow'))
    need(not gaps,'REGISTERED_OR_FUNCTION_CAPTURE_HISTORY_INCOMPLETE')
    launches=[]
    for index,row in selected:
        if row['event'] != 'launch':continue
        need(row['result'] == 0 and row['snapshot_valid'] is True and row['observer_mutates_arguments'] is False
            and begin['monotonic_ns'] <= row['launch_begin_ns'] <= row['launch_end_ns'] <= row['monotonic_ns'] <= end['monotonic_ns'],
            'LAUNCH_FAILED_SNAPSHOT_INVALID_OR_ORDER')
        need(row['api'] in ('aclrtLaunchKernelWithHostArgs','rtKernelLaunchWithHandle','rtKernelLaunchWithHandleV2')
            and type(row['block_dim']) is int and row['block_dim'] > 0
            and type(row['schedule_mode']) is int and row['schedule_mode'] >= 0
            and str(row['tiling_key']).isdecimal(),'ACTUAL_LAUNCH_PROFILE_UNOBSERVED')
        history=[(i,r) for i,r in enumerate(rows) if r['event'] == 'register' and r['pid'] == owner[0]
            and r['handle'] == row['handle'] and r['monotonic_ns'] <= row['launch_begin_ns']]
        registered=[(i,r) for i,r in history if r['generation'] == row['generation']]
        need(len(registered) == 1,'REGISTERED_GENERATION_AMBIGUOUS')
        reg_index,reg=registered[0];newest=max(r['monotonic_ns'] for _,r in history)
        need(reg['result'] == 0 and reg_index < index and reg['monotonic_ns'] == newest
            and sum(r['monotonic_ns'] == newest for _,r in history) == 1,'STALE_REGISTERED_HANDLE_GENERATION')
        name=row['binary_file'];path=Path(run)/'runtime'/name
        need(name == reg['binary_file'] == f"registered-{owner[0]}-{row['generation']}.bin"
            and not path.is_symlink() and path.resolve(strict=True).parent == (Path(run)/'runtime').resolve(strict=True)
            and 0 < path.stat().st_size == reg['binary_bytes'] <= 128*1024**2,'REGISTERED_BINARY_CHANGED')
        binding=None
        if row['api'] == 'aclrtLaunchKernelWithHostArgs':
            need(type(row.get('function_handle')) is int and row['function_handle'] > 0
                and row['argument_layout'] == 'aclrt_host_placeholders','ACTUAL_FUNCTION_HANDLE_UNOBSERVED')
            history=[(i,r) for i,r in enumerate(rows) if r['event'] == 'function_bind' and r['pid'] == owner[0]
                and r['function_handle'] == row['function_handle'] and r['monotonic_ns'] <= row['launch_begin_ns']]
            need(history,'FUNCTION_BIND_HISTORY_ABSENT');identity=(row['handle'],row['generation'],str(row['tiling_key']))
            need(all(r['result'] == 0 and (r['handle'],r['generation'],str(r['tiling_key'])) == identity
                for _,r in history),'CONFLICTING_FUNCTION_HANDLE_KEY_OR_GENERATION_HISTORY')
            newest=max(r['monotonic_ns'] for _,r in history);effective=[(i,r) for i,r in history if r['monotonic_ns'] == newest]
            need(len(effective) == 1,'LAST_EFFECTIVE_BIND_TIME_AMBIGUOUS')
            bind_index,bound=effective[0];current=[(i,r) for i,r in history if r.get('call_id') == token]
            need(reg_index < bind_index < index and reg['monotonic_ns'] <= bound['monotonic_ns']
                and (bound['pid'],bound['tid']) == owner,'ACTUAL_BOUND_FUNCTION_OWNER_OR_ORDER_DIFFERS')
            if current:
                need(len(current) == 1 and current[0] == effective[0] and first < bind_index
                    and begin['monotonic_ns'] <= newest,'CURRENT_SCOPE_BIND_NOT_UNIQUE_LATEST')
                kind='FRESH_CURRENT_SCOPE_BIND'
            else:
                need(bind_index < first and newest < begin['monotonic_ns'],'CACHED_BIND_MUST_PRECEDE_CURRENT_SCOPE')
                previous=bound.get('call_id')
                if previous:
                    finished=[(i,r) for i,r in enumerate(rows) if r.get('call_id') == previous and r['event'] == 'end']
                    need(len(finished) == 1 and bind_index < finished[0][0] < first
                        and (finished[0][1]['pid'],finished[0][1]['tid']) == owner
                        and newest < finished[0][1]['monotonic_ns'] < begin['monotonic_ns'],
                        'CACHED_BIND_PRIOR_SCOPE_NOT_COMPLETED')
                kind='CACHED_PRE_SCOPE_LAST_EFFECTIVE_BIND'
            binding=dict(kind=kind,raw_line=bind_index+1,source_call_id=bound.get('call_id'),
                monotonic_ns=newest,current_scope_bind_count=len(current),history_count=len(history),
                handle=row['handle'],generation=row['generation'],function_handle=row['function_handle'],tiling_key=str(row['tiling_key']))
        arguments=bytes.fromhex(row['args_hex']);placeholders=bytes.fromhex(row.get('placeholders_hex',''))
        need(len(arguments) == row['args_size'] <= 1024**2 and len(placeholders) <= 1024**2,'RAW_ARGUMENT_BYTES_DIFFER')
        sync=[r for i,r in selected if i > index and r['event'] in ('device_sync','stream_sync') and r.get('result') == 0
            and row['monotonic_ns'] <= r['monotonic_ns'] <= end['monotonic_ns']
            and (r['event'] == 'device_sync' or r.get('stream') == row['stream'])]
        need(sync and all(r.get('result') == 0 for _,r in selected if r['event'] in ('device_sync','stream_sync')),
             'ACTUAL_LAUNCH_COMPLETION_UNCONFIRMED')
        launches.append(dict(api=row['api'],pid=owner[0],tid=owner[1],raw_line=index+1,call_id=token,
            handle=row['handle'],generation=row['generation'],function_binding=binding,tiling_key=str(row['tiling_key']),
            block_dim=row['block_dim'],schedule_mode=row['schedule_mode'],argument_layout=row['argument_layout'],
            args_bytes=len(arguments),args_sha256=hashlib.sha256(arguments).hexdigest(),placeholders_bytes=len(placeholders),
            placeholders_sha256=hashlib.sha256(placeholders).hexdigest(),
            registered_binary=dict(path=str(path),bytes=path.stat().st_size,sha256=E.sha(path))))
    need(0 < len(launches) <= 64,'NO_OR_UNBOUNDED_ACTUAL_LAUNCHES')
    return dict(launches=launches,event_counts=dict(Counter(r['event'] for _,r in selected)),
        preceding_unscoped_capture_gaps=dict(gaps),original_capture_rewritten=False)


def entry_symbols(path,expected,*,selected_keys):
    """Read any actual key's symbols, including auxiliary non-Attention entries.

STT_OBJECT companions are retained separately from callable STT_FUNC bodies.
Code-section bytes are hashed without loading or disassembling the ELF.
"""
    path=Path(path);need(not path.is_symlink() and 0 < path.stat().st_size <= 128*1024**2,'ELF_FILE_BOUND_OR_LINK')
    raw=path.read_bytes();need(hashlib.sha256(raw).hexdigest() == expected,'REGISTERED_ELF_BYTES_CHANGED')
    need(raw[:6] == b'\x7fELF\x02\x01' and len(raw) >= 64,'ELF64_LITTLE_ENDIAN_REQUIRED')
    _,kind,machine,_,_,_,shoff,flags,ehsize,_,_,shsize,count,names_index=struct.unpack_from('<16sHHIQQQIHHHHHH',raw)
    need(ehsize == shsize == 64 and 64 <= shoff <= len(raw)-64,'ELF_SECTION_HEADER_INVALID')
    first=struct.unpack_from('<IIQQQQIIQQ',raw,shoff)
    if not count:count=first[5]
    if names_index == 0xffff:names_index=first[6]
    need(1 <= count <= 4096 and names_index < count and shoff+count*64 <= len(raw),'ELF_SECTION_BOUND')
    sections=[struct.unpack_from('<IIQQQQIIQQ',raw,shoff+i*64) for i in range(count)]
    def content(i):
        s=sections[i];need(s[4] <= len(raw) and s[5] <= len(raw)-s[4],'ELF_SECTION_OUTSIDE_FILE');return raw[s[4]:s[4]+s[5]]
    def string(data,offset):
        need(0 <= offset < len(data),'ELF_STRING_OFFSET');end=data.find(b'\0',offset)
        need(end >= offset and end-offset <= 65536,'ELF_STRING_BOUND');return data[offset:end].decode('utf-8',errors='strict')
    names=content(names_index);section_names=[string(names,s[0]) for s in sections]
    need(all(type(k) is str and k.isdecimal() and len(k) <= 20 for k in selected_keys),'ACTUAL_KEYS_INVALID')
    selected=[];internals=[]
    for si,s in enumerate(sections):
        if s[1] not in (2,11):continue
        need(s[6] < count and s[9] == 24 and s[5] % 24 == 0 and s[5]//24 <= 1000000,'ELF_SYMBOL_TABLE_INVALID')
        strings=content(s[6]);symbols=content(si)
        for offset in range(0,len(symbols),24):
            name,info,other,shndx,address,size=struct.unpack_from('<IBBHQQ',symbols,offset)
            if not name:continue
            label=string(strings,name)
            is_selected=any(re.search('_'+re.escape(k)+r'(?:_|$)',label) for k in selected_keys)
            item=dict(name=label,symbol_type=info & 15,binding=info >> 4,section_index=shndx,
                address=address,bytes=size,table=section_names[si])
            if is_selected:
                if item['symbol_type'] == 2:
                    need(0 < shndx < count and size > 0,'SELECTED_FUNCTION_UNDEFINED_OR_EMPTY')
                    sec=sections[shndx];start=address-sec[3]
                    need(sec[1] == 1 and sec[2] & 4 and 0 <= start <= sec[5] and size <= sec[5]-start,
                         'SELECTED_FUNCTION_OUTSIDE_EXECUTABLE_SECTION')
                    code=content(shndx)[start:start+size]
                    item.update(code_sha256=hashlib.sha256(code).hexdigest(),section_name=section_names[shndx])
                selected.append(item)
            elif re.search('FlashAttention|FiaKernel|FiaBlock|SplitFuse|Softmax',label):internals.append(item)
    selected.sort(key=lambda r:(r['name'],r['table']));internals.sort(key=lambda r:(r['name'],r['table']))
    functions=[r for r in selected if r['symbol_type'] == 2 and r['binding'] in (1,2)]
    need(functions,'ACTUAL_REGISTERED_ENTRY_SYMBOLS_UNOBSERVED')
    return dict(path=str(path),sha256=expected,bytes=len(raw),elf_kind=kind,elf_machine=machine,elf_flags=flags,
        actual_selected_keys=list(selected_keys),actual_selected_entry_symbols=functions,
        matching_non_function_symbols=[r for r in selected if r not in functions],symbols_sha256=E.digest(selected),
        internal_route_symbol_count=len(internals),internal_route_symbols=internals[:40],
        internal_route_symbols_sha256=E.digest(internals),entry_reachability_from_internal_names_proven=False)
