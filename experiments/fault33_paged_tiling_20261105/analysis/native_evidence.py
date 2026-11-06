"""Bind one original paged IFA call to actual NNOP host/launch log records.

No key/profile is supplied by the analyst. Public host raw, when present, is
compared with the real launch metadata; it is not labeled kernel-argument raw.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re


def require(ok, message):
    if not ok:
        raise ValueError(message)


SELECT = re.compile(r'OP\((\d+),.*?\[NnopbaseExecutorDoTiling\]\[(\d+)\] OpName:\[(IncreFlashAttention_\d+)\] Tiling outputs of IncreFlashAttention are \[tilingKey: (\d+), numBlocks: (\d+), scheMode: (\d+), needAtomic: (\d+), aicpuNumBlocks: (\d+), dynBufSize: (\d+)\]')
SIZE = re.compile(r'OP\((\d+),.*?\[NnopbaseExecutorPrepareParamsExt\]\[(\d+)\] OpName:\[(IncreFlashAttention_\d+)\] TilingDataSize is (\d+) bytes')
LAUNCH = re.compile(r'OP\((\d+),.*?\[NnopbaseExecutorLaunchKernel\]\[(\d+)\] OpName:\[(IncreFlashAttention_\d+)\] Launch dynamic kernel IncreFlashAttention task, tilingKey is (\d+), scheMode is (\d+), numBlocks is (\d+)')
MM = re.compile(r'MatmulTiling:\s+(\w+)\s+=\s+(-?\d+)\s*$')
DETERMINISTIC = re.compile(r'OP\((\d+),.*?\[NnopbaseExecutorGenDynamicKey\]\[(\d+)\] OpName:\[(IncreFlashAttention_\d+)\] Verbose opType is IncreFlashAttention, deterministic value is (\d+), highPrecision value is (\d+)')


def parse_window(rows, marker):
    selected, sizes, launches, mm, flags, evidence = [], [], [], [], [], []
    for lineno, line in rows:
        m = SELECT.search(line)
        if m:
            pid, tid, op, key, blocks, schedule, atomic, ai, dyn = m.groups()
            selected.append(dict(pid=int(pid), tid=int(tid), op=op, tiling_key=key, block_dim=int(blocks),
                                 schedule_mode=int(schedule), need_atomic=int(atomic), aicpu_blocks=int(ai), dyn_buf_size=int(dyn)))
        s = SIZE.search(line)
        if s:
            sizes.append(dict(zip(('pid', 'tid', 'op', 'raw_size'), (int(s[1]), int(s[2]), s[3], int(s[4])))))
        l = LAUNCH.search(line)
        if l:
            pid, tid, op, key, schedule, blocks = l.groups()
            launches.append(dict(pid=int(pid), tid=int(tid), op=op, tiling_key=key,
                                 schedule_mode=int(schedule), block_dim=int(blocks)))
        # The outer Python window also contains statistics and a later consumer.
        # Only the pair printed before this IFA's completed host selection can
        # belong to its planning callback; later projection MM reports are out.
        t = MM.search(line) if not selected else None
        if t:
            if t[1] == 'M':
                mm.append({})
            require(mm and t[1] not in mm[-1], 'Ambiguous BMM field sequence')
            mm[-1][t[1]] = int(t[2])
        d = DETERMINISTIC.search(line)
        if d:
            flags.append({'pid': int(d[1]), 'tid': int(d[2]), 'op': d[3],
                          'deterministic': int(d[4]), 'verbose_high_precision_flag': int(d[5])})
        if m or s or l or t or d or '[AdjustPABmm' in line or '[FillTilingSplitKV]' in line:
            evidence.append({'line': lineno, 'text': line.rstrip()})
    require(len(selected) == len(sizes) == len(launches) == 1, 'Expected exactly one selected/size/launch IFA per original call')
    a, z, k = selected[0], sizes[0], launches[0]
    require(a['pid'] == z['pid'] == k['pid'] == marker['pid'], 'Native PID does not match original call')
    require(a['tid'] == marker['tid'] and z['tid'] == k['tid'], 'Host/launch thread binding mismatch')
    require(a['op'] == z['op'] == k['op'], 'Native operator identity mismatch')
    require(len(flags) == 1 and flags[0]['pid'] == a['pid'] and flags[0]['tid'] == a['tid']
            and flags[0]['op'] == a['op'], 'Missing actual native deterministic flag binding')
    require(flags[0]['deterministic'] == 1 and marker['deterministic_algorithms'] is True,
            'Deterministic execution must be enabled in caller and actual NNOP plan')
    require(all(a[name] == k[name] for name in ('tiling_key', 'block_dim', 'schedule_mode')), 'Selected/launch plan mismatch')
    require(len(mm) == 2 and all(all(f in block for f in ('M', 'N', 'Ka', 'Kb', 'singleCoreN', 'baseN', 'baseK')) for block in mm),
            'Expected two complete real SDK BMM reports')
    return {'call_id': marker['call_id'], 'enter': marker, 'selected': a, 'launch': k,
            'native_raw_size': z['raw_size'], 'native_flags': flags[0], 'sdk_bmm': mm, 'native_log_evidence': evidence,
            'kernel_argument_raw_observed': False, 'stock_full_raw_observable': False}


def parse_log(path):
    windows, active, marker, lines = [], False, None, []
    dispatches = []
    with Path(path).open(encoding='utf-8', errors='strict') as stream:
        for lineno, line in enumerate(stream, 1):
            # Count the original NNOP dispatch once. Runtime StreamLaunchKernelV2
            # describes the same launch at a different instrumentation boundary.
            if 'NnopbaseExecutorLaunchKernel' in line and 'IncreFlashAttention' in line:
                owner = re.search(r'\bOP\((\d+),', line)
                require(owner is not None, f'Unparsed NNOP IFA launch PID at line {lineno}')
                launch = LAUNCH.search(line)
                dispatches.append((int(owner[1]), marker['call_id'] if active else None, lineno, launch is not None))
            event = None
            if line.startswith('{'):
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    pass
            if isinstance(event, dict) and event.get('fault33') == 'enter':
                require(not active, 'Nested original call window')
                active, marker, lines = True, event, []
            elif isinstance(event, dict) and event.get('fault33') == 'exit':
                require(active and all(event[k] == marker[k] for k in ('call_id', 'pid', 'tid', 'block_size')), 'Unmatched original call exit')
                windows.append(parse_window(lines, marker))
                active, marker, lines = False, None, []
            elif active:
                lines.append((lineno, line))
    require(not active and windows, 'Incomplete/no original API windows')
    require(len({x['call_id'] for x in windows}) == len(windows), 'Repeated call ID')
    pids = {w['enter']['pid'] for w in windows}
    require(len(pids) == 1, 'Native replay/model case must use one process')
    own = [d for d in dispatches if d[0] in pids]
    require(all(d[3] for d in own), 'Unparsed same-PID NNOP IFA dispatch in complete log')
    require([d[1] for d in own] == [w['call_id'] for w in windows],
            'Whole-log NNOP IFA launch census includes an unscoped, missing or duplicated call')
    for w, d in zip(windows, own):
        w['whole_log_nnop_ifa_census'] = {'pid': d[0], 'calls': len(own), 'launch_line': d[2],
                                        'unscoped_or_unparsed': 0}
    return windows


def bind_host(windows, host_events):
    events = [json.loads(line) for line in Path(host_events).read_text().splitlines() if line]
    require(len(events) == len(windows), 'Need one completed public host callback per original API call')
    by_id = {x['call_id']: x for x in events}
    require(len(by_id) == len(events), 'Repeated public host call ID')
    for row in windows:
        h = by_id[row['call_id']]
        require(h['schema'] == 'fault33.pa.typed_host.v1' and h['observation'] == 'final_public_tiling_context', 'Wrong host observer')
        require(h['pid'] == row['enter']['pid'] and h['tid'] == row['enter']['tid']
                and h['token'] == row['enter']['scope_token'], 'Host callback scope mismatch')
        require(h['callback_status'] == 0 and h['typed_matches_final_serialized_raw']
                and h['serialized_captures'] == 1, 'Missing successful final serialized host raw')
        require(all(h[k] == row['launch'][k] for k in ('tiling_key', 'block_dim', 'schedule_mode')),
                'Public host and actual launch metadata mismatch')
        require(h['raw_size'] == row['native_raw_size'] == len(bytes.fromhex(h['raw_hex'])), 'Host/launch raw byte count mismatch')
        for index in range(2):
            for field, value in row['sdk_bmm'][index].items():
                key = 'bmm' + str(index + 1) + 'TilingData.' + field
                if key in h['typed']:
                    require(h['typed'][key] == value, 'Public host/SDK BMM field mismatch: ' + key)
        row['public_host'] = h
        row['public_host_raw_sha256'] = hashlib.sha256(bytes.fromhex(h['raw_hex'])).hexdigest()
    return windows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--log', type=Path, required=True)
    parser.add_argument('--host-events', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    windows = parse_log(args.log)
    if args.host_events:
        windows = bind_host(windows, args.host_events)
    result = {'schema': 'fault33.pa.actual_native_evidence.v1', 'log_sha256': hashlib.sha256(args.log.read_bytes()).hexdigest(),
              'calls': windows, 'actual_original_calls': len(windows), 'kernel_argument_raw_observed': False,
              'scope': 'Actual NNOP selection and launch plus SDK BMM; optional final public host raw is independently labeled.'}
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    print(json.dumps({'output': str(args.output), 'actual_original_calls': len(windows)}))


if __name__ == '__main__':
    main()
