"""Bounded CPU-only route evidence from source53 registered ELFs and source.

This reuses the independent current-call registration join, reads no tensor
arrays, imports no device framework and runs no subprocess. Symbols/metadata
are reported as observations, never used to guess or decode a stock ABI.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import struct
import sys

import review_attention33_first_boundary_source53 as original

COMPLETION = 'bdc2183ef5395844c9fc69dbb30b618b2200d9c3cbc85a39be8ebd35acaae472'
INTEREST = re.compile(r'attention|infer|paged|flash|split|softmax|tiling|kernel|qf16|kvf16', re.I)
SOURCE_NAMES = (
    'attention/fused_infer_attention_score/op_host/fused_infer_attention_score_tiling.cpp',
    'attention/fused_infer_attention_score/op_host/flash_attention_infer_tiling.h',
    'attention/fused_infer_attention_score/op_kernel/fused_infer_attention_score.cpp',
    'attention/fused_infer_attention_score/op_kernel/fia_kernel_common.hpp',
    'attention/fused_infer_attention_score/op_host/paged33_fai_policy.h',
)
SOURCE_TERMS = (
    '103000000010300003', '5100000000010200103',
    'QF16_KVF16_OUTF16_NOLSEOUT_TND_PAGEDCACHE_CAUSALMASK_SPLITFUSE_TILING_FD',
    'ComputeSeqLenAndDecodeFlagsForFAI', 'TilingProcess4SplitFuse',
    'RouteToFia', 'MAX_KV_STACK_LEN',
)


def need(ok, reason):
    if not ok:raise ValueError(reason)


def elf_symbols(path, expected):
    path = Path(path)
    need(path.is_file() and 0 < path.stat().st_size <= 128*1024**2 and original.sha(path) == expected,
         'REGISTERED_ELF_BYTES_CHANGED')
    data = path.read_bytes()
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
    # Keep stdout compact; the original ELF and a full-list hash preserve the
    # identity of omitted names. There is no demangler or code execution.
    selected = [dict(item, name=item['name'][:1200], name_truncated=len(item['name']) > 1200) for item in symbols[:48]]
    return dict(report, status='ELF_SYMBOLS_READ', elf_kind=kind, elf_machine=machine, elf_flags=flags,
        section_count=count, sections=[dict(name=section_names[i], bytes=row[5], type=row[1])
            for i,row in enumerate(sections) if INTEREST.search(section_names[i]) or row[1] in (2, 11)],
        matching_symbol_count=len(symbols), symbols_sha256=original.digest(symbols),
        symbols=selected, omitted_symbols=max(0, len(symbols)-len(selected)))


def metadata_matches(root, binary_sha, keys):
    if root is None or not Path(root).is_dir():return dict(status='METADATA_ROOT_NOT_PRESENT', matches=[])
    root = Path(root).resolve(strict=True)
    scanned, matches, total = 0, [], 0
    for path in sorted(root.rglob('*.json')):
        need(not path.is_symlink() and path.resolve(strict=True).is_relative_to(root), 'METADATA_LINK_REFUSED')
        scanned += 1;total += path.stat().st_size
        need(scanned <= 20000 and total <= 256*1024**2 and path.stat().st_size <= 8*1024**2,
             'METADATA_BOUNDED_SEARCH_EXCEEDED')
        data = original.read(path)
        if type(data) is not dict or data.get('sha256') != binary_sha:continue
        entries = data.get('kernelList', [])
        need(type(entries) is list and all(type(row) is dict for row in entries), 'MATCHING_KERNEL_METADATA_INVALID')
        matched = [row for row in entries if str(row.get('tilingKey')) in keys]
        matches.append(dict(path=str(path), metadata_sha256=original.sha(path), bytes=path.stat().st_size,
            kernel_binary_sha256=binary_sha, kernel_entries=matched,
            declared_entry_count=len(entries), parameter_count=len(data.get('parameters', [])),
            attributes={key:data[key] for key in ('binFileName','binFileSuffix','kernelName','opType','coreType',
                'taskRation','opParaSize','magic','intercoreSync') if key in data},
            entry_key_present=bool(matched)))
    return dict(status='EXACT_BINARY_SHA_METADATA_SEARCH', root=str(root), json_files_read=scanned,
                bytes_read=total, matches=matches)


def source_excerpt(root):
    if root is None or not Path(root).is_dir():return dict(status='SOURCE_NOT_PRESENT', files=[])
    root = Path(root).resolve(strict=True)
    files = []
    for name in SOURCE_NAMES:
        path = root/name
        if not path.is_file():continue
        need(not path.is_symlink() and path.resolve(strict=True).is_relative_to(root)
            and path.stat().st_size <= 8*1024**2, 'SOURCE_FILE_BOUND_OR_ESCAPE')
        text = path.read_text(encoding='utf-8'); lines = text.splitlines(); hits = []
        for term in SOURCE_TERMS:
            indices = [i for i, line in enumerate(lines) if term in line]
            hits.append(dict(term=term, occurrences=len(indices), excerpts=[
                dict(line=i+1, text='\n'.join(lines[max(0,i-3):min(len(lines),i+9)])[:2800]) for i in indices[:4]]))
        files.append(dict(path=str(path), bytes=path.stat().st_size, sha256=original.sha(path), hits=hits))
    return dict(status='BOUNDED_EXPLICIT_SOURCE_FILES_ONLY', root=str(root), files=files,
                source_receipt_join='REQUIRES_BUILD_RECEIPT_RECONCILIATION_NOT_IMPLIED_BY_PATH')


def arm(run, expected):
    run = Path(run).resolve(strict=True); result = run/'result.json'
    need(original.sha(result) == expected, 'EXACT_SOURCE53_RESULT_REQUIRED')
    data = original.read(result)
    need(data['schema'] == 'demoncat.paged33.first_boundary_worker_run.v1'
        and [row['phase'] for row in data['phases']] == list(original.PHASES), 'SOURCE53_PHASE_SET_DIFFERS')
    rows, runtime = original.read_runtime(run)
    phases, binaries = [], {}
    for phase in data['phases']:
        boundary = phase['finish']['capture']['first_decode_boundary']
        resolved = original.scoped_runtime(run, rows, boundary['native_call_id'], expected_pid=boundary['pid'])
        for launch in resolved['launches']:
            identity = launch['registered_binary']
            if identity['sha256'] not in binaries:
                binaries[identity['sha256']] = elf_symbols(identity['path'], identity['sha256'])
        phases.append(dict(phase=phase['phase'], original_observation_gap=boundary['runtime_observation'],
                           new_offline_runtime_resolution=resolved))
    return dict(result_path=str(result), result_sha256=expected, runtime=runtime, phases=phases,
                binaries=binaries, selected_aclnn_owner='NOT_OBSERVED')


def prohibit(event, args):
    if event in ('ctypes.dlopen','subprocess.Popen','os.system','socket.connect'):
        raise RuntimeError('READ_ONLY_CPU_ACTION_REFUSED:'+event)
    if event == 'import' and args[0].split('.')[0] in ('torch','torch_npu','vllm','vllm_ascend'):
        raise RuntimeError('DEVICE_FRAMEWORK_IMPORT_REFUSED:'+args[0])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('stock-run','baseline-run','private-build-root'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--stock-source', type=Path)
    parser.add_argument('--stock-metadata-root', type=Path)
    args = parser.parse_args()
    sys.dont_write_bytecode = True;sys.addaudithook(prohibit)
    report = dict(schema='demoncat.attention33.native_route_diagnostic.v1', reader_sha256=original.sha(__file__),
        registration_join_reader_sha256=original.sha(original.__file__), qualification_granted=False,
        bridge_pass=False, fault_authorized=False, C1=False, C2=False, C3=False, E5='NOT_ASSESSED')
    try:
        build = args.private_build_root.resolve(strict=True)
        completion = build/'evidence/completion.json'
        need(original.sha(completion) == COMPLETION, 'EXACT_SOURCE53_PRIVATE_FD_BUILD_REQUIRED')
        record = original.read(completion)
        need(record['source_commit'] == 'c77c0829f1f0f0b8f882db1d9b051a758b605472'
            and record['variant'] == 'tnd_page16_fai_fd_s2_256', 'PRIVATE_FD_BUILD_VARIANT_CHANGED')
        report['build'] = dict(path=str(completion), sha256=COMPLETION, source_commit=record['source_commit'],
                               variant=record['variant'], patch_receipt_sha256=record['patch_receipt_sha256'])
        for label, run, expected in (('stock',args.stock_run,original.STOCK), ('private',args.baseline_run,original.PRIVATE)):
            found = arm(run, expected)
            metadata = args.stock_metadata_root if label == 'stock' else build/'stage/vendors/dcat_f33_fai_fd_transformer/op_impl/ai_core/tbe/kernel'
            for identity, binary in found['binaries'].items():
                keys = {row['tiling_key'] for phase in found['phases']
                        for row in phase['new_offline_runtime_resolution']['launches']
                        if row['registered_binary']['sha256'] == identity}
                binary['matching_metadata'] = metadata_matches(metadata, identity, keys)
            report[label] = found
        report['private_source'] = source_excerpt(build/'source')
        report['stock_source'] = source_excerpt(args.stock_source)
        report.update(status='ORIGINAL_ROUTE_EVIDENCE_EXTRACTED_NO_ARITHMETIC_RECLASSIFICATION',
            limits=['No tensor bins are read or returned by this diagnostic.',
                'An ELF symbol or matching metadata is not a stock parameter ABI or proof of the selected ACLNN library owner.',
                'Missing stock sources remain missing; private source does not substitute for them.',
                'This neither explains the 67-element arithmetic difference conclusively nor marks the mechanism inapplicable.'])
    except Exception as error:
        report.update(status='DIAGNOSTIC_REFUSED_WITH_EVIDENCE_GAP', error=type(error).__name__+': '+str(error)[:500])
    print(json.dumps(report, allow_nan=False, separators=(',',':')))
    return 2 if report['status'] == 'DIAGNOSTIC_REFUSED_WITH_EVIDENCE_GAP' else 0


if __name__ == '__main__':raise SystemExit(main())
