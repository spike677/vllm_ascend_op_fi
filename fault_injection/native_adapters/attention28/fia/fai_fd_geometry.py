"""Independent coverage/reduction contracts for the actual B1 TND FAI FD path.

The platform core count is an observed input, never inferred from a device name.
Every original public field is still bound by the serializer/final-byte check.
This adds task coverage, split reduction and allocation checks for FD=true.
"""
import struct

VARIANT = 'tnd_page16_fai_fd_s2_256'
NATIVE_KEYS = (5000000000000200100, 5000000000000200103,
               5000000000010200100, 5000000000010200103,
               5100000000010200100, 5100000000010200103)
FAULT_KEYS = (6000000000010200100, 6000000000010200103)
KEYS = (*NATIVE_KEYS, *FAULT_KEYS)
CORE = ('startBIdx', 'startN1Idx', 'startS1Idx', 'startS2Idx', 'endBIdx', 'endN1Idx',
        'endS1Idx', 'endS2Idx', 'firstSplitKVTaskLseOffset', 'firstSplitKVTaskOOffset')
SPLIT = ('batchIdx', 'headStartIdx', 'headEndIdx', 'qStartIdx', 'qEndIdx', 'splitNum',
         'lseTaskOffset', 'oTaskOffset')


def require(ok, message):
    if not ok:raise RuntimeError(message)


def validate_fd_geometry(typed, host, *, key, block_dim, schedule_mode, arm, kv_length):
    require(arm in ('baseline', 'fault', 'off') and type(kv_length) is int and 1025 <= kv_length <= 2048,
            'TND33_FD_SCOPE_DIFFERS')
    aic = host.get('aic');tile = 256 if arm == 'fault' else 512
    require(type(aic) is int and 20 <= aic <= 25 and host.get('route') == 'FAInferTiling'
            and host.get('native_route_or_decode_flags_changed') is False
            and host.get('flash_decode') is True and host.get('decoding') is False
            and host.get('tiling_sink') is False, 'TND33_FD_ACTUAL_PLATFORM_OR_FLAGS_DIFFERS')
    require(host.get('query_length') == 1 and host.get('kv_length') == kv_length
            and host.get('context_max_q') == host.get('context_max_kv') == 0,
            'TND33_FD_ACTUAL_SEQUENCE_SOURCE_DIFFERS')
    # These source fields remain zero in the real c77 ConvertContext path.
    # Sequence lengths come from actual q/kv lists, not these unused members.
    require(all(typed.get(name) == value for name, value in dict(numHeads=16, kvHeads=8, batch=1,
        embeddingSize=128, embeddingSizeV=128, blockSize=16, maxQSeqlen=0, maxKvSeqlen=0,
        firstBatchTaskNum=8, totalTaskNum=8, pseQ=0, pseKv=0, mainLoopTaskNum=0,
        tailLoopTaskNum=0, tailStartBatch=0, tailStartN2=0, tailKvNBlockTile=0).items()),
        'TND33_FD_FINAL_BASE_GEOMETRY_DIFFERS')
    require(typed.get('scaleValue') == struct.unpack('<f', struct.pack('<f', 128**-.5))[0],
            'TND33_FD_FINAL_SCALE_DIFFERS')
    mask = typed.get('maskType')
    require(mask in (0, 1) and typed.get('sparseMode') == (3 if mask else 0), 'TND33_FD_MASK_CONTRACT_DIFFERS')
    require(type(typed.get('numBlocks')) is int and typed['numBlocks'] >= (kv_length + 15)//16
            and type(typed.get('maxNumBlocksPerBatch')) is int
            and typed['maxNumBlocksPerBatch'] >= (kv_length + 15)//16, 'TND33_FD_PAGE_CAPACITY_DIFFERS')
    require(key == 5100000000010200100 + (3 if mask else 0)
            + (900000000000000000 if arm == 'fault' else 0) and schedule_mode == 1,
            'TND33_FD_FINAL_ENTRY_OR_SCHEDULE_DIFFERS')
    arrays = {}
    for group, names in (('coreInfo', CORE), ('splitInfo', SPLIT)):
        for name in names:
            value = typed.get(group + '.' + name)
            require(type(value) is list and len(value) == 26 and all(type(x) is int for x in value),
                    'TND33_FD_PUBLIC_ARRAY_DIFFERS:' + group + '.' + name)
            arrays[group + '.' + name] = value
    core = lambda name, index: arrays['coreInfo.' + name][index]
    need = typed.get('needCoreNum')
    require(type(need) is int and 1 <= need <= aic and block_dim == need, 'TND33_FD_ACTUAL_ACTIVE_CORES_DIFFERS')
    chunks = (kv_length + tile - 1)//tile
    per_core = (16*kv_length + aic - 1)//aic
    previous_end, contributions = 0, []
    for index in range(need):
        require(all(core(name, index) == 0 for name in ('startBIdx', 'endBIdx', 'startS1Idx', 'endS1Idx')),
                'TND33_FD_SINGLE_REQUEST_COORDINATES_DIFFERS')
        start_head, end_head = core('startN1Idx', index), core('endN1Idx', index)
        start_kv, end_kv = core('startS2Idx', index), core('endS2Idx', index)
        require(0 <= start_head <= end_head < 8 and 0 <= start_kv < chunks and 0 <= end_kv <= chunks,
                'TND33_FD_CORE_COORDINATES_OUT_OF_BOUNDS')
        start, end = start_head*chunks + start_kv, end_head*chunks + end_kv
        require(start == previous_end and start < end <= 8*chunks, 'TND33_FD_KV_COVERAGE_GAP_OR_OVERLAP')
        previous_end = end
        cost = sum(2*min(tile, kv_length - (chunk % chunks)*tile) for chunk in range(start, end))
        last_cost = 2*min(tile, kv_length - ((end - 1) % chunks)*tile)
        require((cost >= per_core or end == 8*chunks) and cost - last_cost < per_core,
                'TND33_FD_NATIVE_PER_CORE_WORK_DIFFERENT')
        partials = []
        for head in range(start_head, end_head + 1):
            left, right = max(start, head*chunks), min(end, (head + 1)*chunks)
            if 0 < right - left < chunks:partials.append(head)
        contributions.append(partials)
    require(previous_end == 8*chunks, 'TND33_FD_INCOMPLETE_KV_COVERAGE')
    require(all(core(name, i) == 0 for name in CORE[:8] for i in range(need, 26)),
            'TND33_FD_INACTIVE_CORE_COORDINATES_NONZERO')
    expected_split = {name: [0]*26 for name in SPLIT}
    expected_lse, expected_o = [0]*26, [0]*26
    nodes, last_head, lse_offset, o_offset = -1, None, 0, 0
    for index, partials in enumerate(contributions):
        for pos, head in enumerate(partials):
            if head != last_head:
                nodes += 1;last_head = head
                for name, value in dict(headStartIdx=head*2, headEndIdx=head*2 + 2, qEndIdx=1,
                    lseTaskOffset=lse_offset, oTaskOffset=o_offset).items():expected_split[name][nodes] = value
            expected_split['splitNum'][nodes] += 1
            if pos == 0:expected_lse[index], expected_o[index] = lse_offset, o_offset
            lse_offset += 2;o_offset += 256
    require(typed.get('totalSplitNodeNum') == nodes + 1 and nodes >= 0,
            'TND33_FD_SPLIT_NODE_COUNT_DIFFERS')
    require(all(arrays['splitInfo.' + name] == values for name, values in expected_split.items())
            and arrays['coreInfo.firstSplitKVTaskLseOffset'] == expected_lse
            and arrays['coreInfo.firstSplitKVTaskOOffset'] == expected_o, 'TND33_FD_REDUCTION_ARRAYS_DIFFER')
    lse_bytes, o_bytes = lse_offset*4, o_offset*4
    require(typed.get('splitLseTotalSize') == lse_bytes and typed.get('splitOTotalSize') == o_bytes,
            'TND33_FD_REDUCTION_WORKSPACE_DIFFERS')
    unit = aic*128*tile*3
    require(all(typed.get(name) == unit*width for name, width in
                dict(mm1OutSize=4, smOnlineOutSize=2, mm2OutSize=4, UpdateSize=4).items())
            and typed.get('workSpaceSize') == unit*14 + lse_bytes + o_bytes
            and host.get('workspace') == 16*1024**2 + unit*16 + lse_bytes + o_bytes,
            'TND33_FD_FINAL_WORKSPACE_DIFFERS')
    return dict(route='FAInferTiling / native flash decode', selected_s2=tile, actual_aic=aic,
        actual_active_cores=need, complete_kv_chunks=8*chunks, reduction_nodes=nodes + 1,
        reduction_lse_bytes=lse_bytes, reduction_o_bytes=o_bytes,
        selection='EXACT_FRESH_FD_ENTRY_FINAL_COVERAGE_AND_WORKSPACE', host_device_geometry_agrees=True)
