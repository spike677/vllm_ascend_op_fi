"""Bounded CPU resource estimates for the distinct all-target S2 package.

This module loads no tensor framework, DSO, process, or device. A resource
receipt never grants a Native fault or a coverage proof.
"""
from copy import deepcopy
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import threading
import time

VARIANT='tnd_page16_arch22_group_s2_256'
GROUP_LOG_BYTES=512*1024**2
DEFAULT_LOG_BYTES=128*1024**2
PROTOCOL_REQUESTS=60
RECOVERY_REQUESTS=3
GROWTH_FACTOR=2
_ISSUER=object()


def need(ok,reason):
    if not ok:raise RuntimeError(reason)


def digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def positive(value):return type(value) in (int,float) and math.isfinite(value) and value>0


def plan(inputs,*,variant,source_sha256,build_sha256,case_seconds=1800):
    need(variant==VARIANT and all(type(s) is str and re.fullmatch('[0-9a-f]{64}',s) for s in (source_sha256,build_sha256))
        and type(case_seconds) is int and 1<=case_seconds<=1800,'GROUP_RESOURCE_SCOPE_IDENTITY_REQUIRED')
    rows=inputs['phases'];need(type(rows) is list and len(rows)==5,'GROUP_FIVE_COMPLETE_RESOURCE_PHASES_REQUIRED')
    previous=0;deltas=[];constant=[];history=[];prefixes=[]
    for row in rows:
        start,end=row['runtime_start_bytes'],row['runtime_end_bytes']
        need(type(start) is type(end) is int and previous<=start<end<=DEFAULT_LOG_BYTES
            and all(positive(row[k]) for k in ('elapsed_seconds','indexed_history_seconds','artifact_bytes','capture_json_bytes'))
            and row['indexed_history_seconds']<=row['elapsed_seconds']
            and type(row['artifact_bytes']) is type(row['capture_json_bytes']) is int
            and row['capture_json_bytes']<=128*1024**2 and row['artifact_bytes']<=512*1024**2,
            'GROUP_COMPLETE_MEASURED_RESOURCE_PHASE_REQUIRED')
        # Inter-phase CPU captures/cleanup may append additional runtime facts.
        deltas.append(end-(previous if previous else start));previous=end
        constant.append(row['elapsed_seconds']-row['indexed_history_seconds']);history.append(row['indexed_history_seconds']);prefixes.append(end)
    initial=rows[0]['runtime_start_bytes'];tail=inputs['runtime_final_bytes']-previous
    need(type(inputs['runtime_final_bytes']) is int and 0<=tail and inputs['runtime_final_bytes']<=DEFAULT_LOG_BYTES
        and all(type(inputs[k]) is int and inputs[k]>0 for k in ('registered_binary_bytes','normal_result_bytes'))
        and all(positive(inputs[k]) for k in ('initialization_seconds','ordinary_health_seconds')),
        'GROUP_MEASURED_INITIALIZATION_TAIL_OR_BINARY_BYTES_REQUIRED')
    delta=max(deltas);reserve=max(tail,delta*GROWTH_FACTOR*RECOVERY_REQUESTS)
    projected=initial+delta*GROWTH_FACTOR*PROTOCOL_REQUESTS+reserve
    need(projected<=GROUP_LOG_BYTES,'NO_F_CAPACITY_REFUSED:PROJECTED_COMPLETE_RUNTIME_LOG')
    fixed=max(constant);historical=max(history);scale=max(prefixes)
    factors=[max(1,(initial+delta*GROWTH_FACTOR*(i+1))/scale) for i in range(PROTOCOL_REQUESTS)]
    request_time=fixed*PROTOCOL_REQUESTS+historical*sum(factors)
    recovery_time=RECOVERY_REQUESTS*(fixed+historical*max(1,projected/scale))
    margin=max(120,math.ceil(.15*(request_time+recovery_time)))
    total=math.ceil(inputs['initialization_seconds']+inputs['ordinary_health_seconds']+request_time+recovery_time+margin)
    need(total<=case_seconds,'NO_F_TIME_REFUSED:PROJECTED_COMPLETE_CASE')
    # The result refs and 3 retained controls are included; bins remain on disk.
    per_request=max(r['artifact_bytes'] for r in rows);capture=max(r['capture_json_bytes'] for r in rows)
    disk=GROWTH_FACTOR*per_request*(PROTOCOL_REQUESTS+RECOVERY_REQUESTS)+projected+GROWTH_FACTOR*inputs['registered_binary_bytes']+64*1024**2
    memory=8*projected+12*capture+512*1024**2
    return dict(schema='demoncat.paged33.group_resource_plan.v1',variant=variant,source_manifest_sha256=source_sha256,
        completion_sha256=build_sha256,inputs_sha256=digest(inputs),inputs=deepcopy(inputs),case_seconds=case_seconds,
        protocol_requests=PROTOCOL_REQUESTS,recovery_requests=RECOVERY_REQUESTS,growth_factor=GROWTH_FACTOR,
        maximum_runtime_bytes=GROUP_LOG_BYTES,initial_runtime_bytes=initial,maximum_phase_increment_bytes=delta,
        recovery_runtime_reserve_bytes=reserve,projected_runtime_bytes=projected,maximum_normal_prefix_bytes=scale,
        constant_phase_seconds=fixed,indexed_history_seconds=historical,cumulative_prefix_factors=sum(factors),
        recovery_seconds=recovery_time,margin_seconds=margin,estimated_seconds=total,
        maximum_phase_artifact_bytes=per_request,maximum_capture_json_bytes=capture,
        estimated_disk_bytes=disk,estimated_additional_memory_bytes=memory,
        uncertain_fault_retry=False,budget_extension=False,qualification_granted=False)


def host_resources(directory):
    """Read host/cgroup CPU resource counters, never an accelerator query."""
    root=Path(directory).resolve(strict=True);available=None;limits=[]
    meminfo=Path('/proc/meminfo')
    need(meminfo.is_file(),'NO_F_CAPACITY_REFUSED:HOST_MEMORY_COUNTER_UNAVAILABLE')
    for line in meminfo.read_text().splitlines():
        if line.startswith('MemAvailable:'):
            parts=line.split();need(len(parts)==3 and parts[2]=='kB','GROUP_HOST_MEMORY_COUNTER_FORMAT')
            available=int(parts[1])*1024
    need(type(available) is int and available>0,'NO_F_CAPACITY_REFUSED:HOST_MEMORY_COUNTER_UNAVAILABLE')
    for ceiling,current in ((Path('/sys/fs/cgroup/memory.max'),Path('/sys/fs/cgroup/memory.current')),
        (Path('/sys/fs/cgroup/memory/memory.limit_in_bytes'),Path('/sys/fs/cgroup/memory/memory.usage_in_bytes'))):
        if not ceiling.is_file() and not current.is_file():continue
        need(ceiling.is_file() and current.is_file(),'NO_F_CAPACITY_REFUSED:INCOMPLETE_CGROUP_MEMORY_COUNTER')
        raw=ceiling.read_text().strip();used=int(current.read_text().strip())
        maximum=None if raw=='max' else int(raw)
        need(used>=0 and (maximum is None or maximum>0),'GROUP_CGROUP_MEMORY_COUNTER_FORMAT')
        if maximum is not None:available=min(available,max(0,maximum-used))
        limits.append(dict(limit_path=str(ceiling),current_path=str(current),limit_bytes=maximum,current_bytes=used))
    need(limits,'NO_F_CAPACITY_REFUSED:CGROUP_MEMORY_COUNTER_UNAVAILABLE')
    return dict(pid=os.getpid(),python_thread_ident=threading.get_ident(),native_tid=threading.get_native_id(),
        sampled_monotonic=time.monotonic(),disk_root=str(root),free_disk_bytes=shutil.disk_usage(root).free,
        available_memory_bytes=available,cgroup_memory=limits)


def before_fault(plan_value,*,phase_index,started_monotonic,runtime_bytes,resources,now):
    need(plan_value==plan(plan_value['inputs'],variant=plan_value['variant'],source_sha256=plan_value['source_manifest_sha256'],
        build_sha256=plan_value['completion_sha256'],case_seconds=plan_value['case_seconds']),'GROUP_RESOURCE_PLAN_CHANGED')
    need(plan_value['variant']==VARIANT and plan_value['maximum_runtime_bytes']==GROUP_LOG_BYTES
        and type(phase_index) is int and phase_index in range(3,59,2)
        and positive(started_monotonic) and started_monotonic<=now
        and type(runtime_bytes) is int and 0<=runtime_bytes<=GROUP_LOG_BYTES,
        'GROUP_LIVE_RESOURCE_SCOPE_OR_PHASE_REQUIRED')
    need(all(type(resources.get(k)) is int and resources[k]>0 for k in ('pid','python_thread_ident','native_tid'))
        and (resources.get('pid'),resources.get('python_thread_ident'),resources.get('native_tid'))==
        (os.getpid(),threading.get_ident(),threading.get_native_id()) and positive(resources.get('sampled_monotonic'))
        and 0<=now-resources['sampled_monotonic']<=5,'NO_F_CAPACITY_REFUSED:STALE_OR_FOREIGN_HOST_COUNTERS')
    remaining=PROTOCOL_REQUESTS-phase_index;delta=plan_value['maximum_phase_increment_bytes']*GROWTH_FACTOR
    projected=runtime_bytes+remaining*delta+plan_value['recovery_runtime_reserve_bytes']
    need(projected<=GROUP_LOG_BYTES,'NO_F_CAPACITY_REFUSED:LIVE_RUNTIME_GROWTH')
    scale=plan_value['maximum_normal_prefix_bytes'];fixed=plan_value['constant_phase_seconds'];history=plan_value['indexed_history_seconds']
    factors=sum(max(1,(runtime_bytes+delta*(i+1))/scale) for i in range(remaining))
    reserve=RECOVERY_REQUESTS*(fixed+history*max(1,projected/scale))
    needed=remaining*fixed+history*factors+reserve+plan_value['margin_seconds']+plan_value['inputs']['ordinary_health_seconds']
    elapsed=now-started_monotonic
    need(elapsed+needed<=plan_value['case_seconds'],'NO_F_TIME_REFUSED:LIVE_F_OFF_RECOVERY_RESERVE')
    disk=GROWTH_FACTOR*plan_value['maximum_phase_artifact_bytes']*(remaining+RECOVERY_REQUESTS)+max(0,projected-runtime_bytes) \
        +GROWTH_FACTOR*plan_value['inputs']['registered_binary_bytes']+64*1024**2
    memory=plan_value['estimated_additional_memory_bytes']
    need(all(type(resources[k]) is int and resources[k]>=0 for k in ('free_disk_bytes','available_memory_bytes'))
        and resources['free_disk_bytes']>=disk and resources['available_memory_bytes']>=memory,
        'NO_F_CAPACITY_REFUSED:LIVE_DISK_OR_MEMORY')
    return dict(phase_index=phase_index,elapsed_seconds=elapsed,remaining_seconds=needed,
        runtime_bytes=runtime_bytes,projected_runtime_bytes=projected,required_free_disk_bytes=disk,
        required_available_memory_bytes=memory,resources=deepcopy(resources),plan_sha256=digest(plan_value),
        fault_authorized=False,qualification_granted=False)


class GroupRuntimeLogScope:
    """Capacity only: an exact live normal object binds source/build/attempt."""
    def __init__(self,seal,normal,build_sha256,attempt):
        need(seal is _ISSUER,'GROUP_CAPACITY_PRIVATE_ISSUER_REQUIRED')
        self.normal=normal;self.attempt=Path(attempt).resolve(strict=True)
        self.thread=(os.getpid(),threading.get_ident(),threading.get_native_id())
        data=normal.data()
        self._receipt=dict(variant=VARIANT,source_manifest_sha256=data['source_manifest_sha256'],
            completion_sha256=build_sha256,normal_evidence_sha256=normal.fingerprint,
            attempt_root=str(self.attempt),owner=list(self.thread),maximum_runtime_bytes=GROUP_LOG_BYTES,
            qualification_granted=False)
        self._seal=digest(self._receipt);self._issuer=seal

    @classmethod
    def from_verified_normal(cls,normal,*,build_sha256,attempt):
        from demoncat.runtime.vllm_ascend.worker_extension.paged33_group_evidence import GroupNormalEvidence
        need(type(normal) is GroupNormalEvidence and not normal.simulation,'GROUP_EXACT_NATIVE_NORMAL_OBJECT_REQUIRED')
        data=normal.data()
        need(data['variant']==VARIANT and data['completion_sha256']==build_sha256
            and all(re.fullmatch('[0-9a-f]{64}',str(data[k])) for k in ('source_manifest_sha256','completion_sha256')),
            'GROUP_CAPACITY_SOURCE_OR_BUILD_DIFFERS')
        return cls(_ISSUER,normal,build_sha256,attempt)

    def check(self,directory):
        need(type(self) is GroupRuntimeLogScope and self._issuer is _ISSUER and digest(self._receipt)==self._seal
            and self.thread==(os.getpid(),threading.get_ident(),threading.get_native_id())
            and Path(directory).resolve(strict=True)==self.attempt/'runtime'
            and self.normal.fingerprint==self._receipt['normal_evidence_sha256']
            and self.normal.data()['completion_sha256']==self._receipt['completion_sha256'],
            'GROUP_CAPACITY_OWNER_SOURCE_BUILD_OR_ATTEMPT_CHANGED')
        return deepcopy(self._receipt)

    def __reduce__(self):raise TypeError('Runtime capacity scope is not serializable authority')
