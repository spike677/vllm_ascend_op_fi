"""Owned append-log cache. Every reuse rehashes the complete previous prefix.

This indexes observations, never classifies a kernel or authorizes a fault.
Raw files remain untouched. Consumers receive all conflicting registration and
function histories relevant to every launch, including auxiliary launches.
"""
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import threading
import time

MAX_BYTES=128*1024**2
MAX_LINE=4*1024**2
SCHEMA='demoncat.attention.runtime_observer.v1'


def need(value,reason):
    if not value:raise RuntimeError(reason)


def unique(pairs):
    value={}
    for key,item in pairs:
        need(key not in value,'INDEX_DUPLICATE_JSON_KEY');value[key]=item
    return value


def invalid(value):raise RuntimeError('INDEX_NONFINITE_JSON:'+value)


class RuntimeLogIndex:
    def __init__(self,directory,*,group_scope=None):
        self.root=Path(directory).resolve(strict=True);self.path=self.root/'runtime.jsonl'
        self.group_scope=group_scope
        self.maximum_bytes  # Validate the exact private capacity object before use.
        self.owner=(os.getpid(),threading.get_ident(),threading.get_native_id())
        self.file_identity=None;self.size=0;self.prefix_sha256=hashlib.sha256(b'').hexdigest()
        self._rows=[];self._calls={};self._registers={};self._functions={};self._overflow=[]
        self.parsed_events=0;self.refreshes=0;self.validation_seconds=0.

    @property
    def maximum_bytes(self):
        if self.group_scope is None:return MAX_BYTES
        from .group_resource_budget import GroupRuntimeLogScope,GROUP_LOG_BYTES
        need(type(self.group_scope) is GroupRuntimeLogScope,'INDEX_EXACT_GROUP_CAPACITY_SCOPE_REQUIRED')
        self.group_scope.check(self.root)
        return GROUP_LOG_BYTES

    def _owned(self,directory):
        need(self.owner==(os.getpid(),threading.get_ident(),threading.get_native_id())
            and Path(directory).resolve(strict=True)==self.root,'INDEX_OWNER_OR_DIRECTORY_CHANGED')

    def _refresh(self):
        need(not self.path.is_symlink(),'INDEX_RUNTIME_SYMLINK')
        with self.path.open('rb') as stream:
            before=os.fstat(stream.fileno());identity=(before.st_dev,before.st_ino)
            need(self.file_identity is None or identity==self.file_identity,'INDEX_RUNTIME_REPLACED')
            need(self.size<=before.st_size<=self.maximum_bytes,'INDEX_RUNTIME_TRUNCATED_OR_OVERFLOW')
            prefix=hashlib.sha256();remaining=self.size
            while remaining:
                block=stream.read(min(remaining,1024**2));need(block,'INDEX_OLD_PREFIX_SHORT_READ')
                prefix.update(block);remaining-=len(block)
            need(prefix.hexdigest()==self.prefix_sha256,'INDEX_OLD_PREFIX_BYTES_CHANGED')
            appended=stream.read(before.st_size-self.size)
            need(len(appended)==before.st_size-self.size and (not appended or appended.endswith(b'\n')),
                 'INDEX_APPEND_INCOMPLETE_LINE')
            after=os.fstat(stream.fileno());path_after=self.path.stat()
            sig=lambda s:(s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns)
            need(sig(before)==sig(after)==sig(path_after) and not self.path.is_symlink(),'INDEX_RUNTIME_CHANGED_DURING_READ')
        new=[]
        for raw in appended.splitlines():
            need(0<len(raw)<=MAX_LINE,'INDEX_RUNTIME_LINE_BOUND')
            row=json.loads(raw,object_pairs_hook=unique,parse_constant=invalid)
            need(type(row) is dict and row.get('schema')==SCHEMA and type(row.get('event')) is str
                and type(row.get('call_id')) is str
                and all(type(row.get(k)) is int and row[k]>0 for k in ('pid','tid','monotonic_ns')),
                'INDEX_RUNTIME_EVENT_SCHEMA')
            keys=('handle',) if row['event'] in ('register','launch') else ('function_handle',) if row['event']=='function_bind' else ()
            need(all(type(row.get(k)) is int and row[k]>0 for k in keys),'INDEX_EVENT_IDENTITY_FIELDS')
            if row['event']=='launch' and row.get('function_handle'):
                need(type(row['function_handle']) is int and row['function_handle']>0,'INDEX_EVENT_IDENTITY_FIELDS')
            new.append(row)
        # Commit the cache only after the whole appended snapshot validates.
        for row in new:
            number=len(self._rows);self._rows.append(row)
            self._calls.setdefault(row.get('call_id'),[]).append(number)
            if row['event']=='register':self._registers.setdefault((row['pid'],row['handle']),[]).append(number)
            if row['event']=='function_bind':self._functions.setdefault((row['pid'],row['function_handle']),[]).append(number)
            if row['event']=='capture_overflow':self._overflow.append(number)
        prefix.update(appended);self.prefix_sha256=prefix.hexdigest();self.size=before.st_size;self.file_identity=identity
        self.parsed_events+=len(new);self.refreshes+=1

    def scope(self,directory,token):
        self._owned(directory);need(type(token) is str,'INDEX_TOKEN_TYPE');started=time.monotonic()
        try:
            self._refresh();selected=set(self._calls.get(token,[]));selected.update(self._overflow)
            for number in tuple(selected):
                row=self._rows[number]
                if row['event']!='launch':continue
                selected.update(self._registers.get((row['pid'],row['handle']),[]))
                if row.get('function_handle'):
                    selected.update(self._functions.get((row['pid'],row['function_handle']),[]))
            numbers=sorted(selected)
            return dict(rows=deepcopy([self._rows[i] for i in numbers]),line_numbers=[i+1 for i in numbers],
                prefix_bytes=self.size,prefix_sha256=self.prefix_sha256,
                file_identity=list(self.file_identity),total_events=len(self._rows),parsed_events=self.parsed_events)
        finally:self.validation_seconds+=time.monotonic()-started

    def report(self):
        self._owned(self.root)
        return dict(schema='demoncat.attention.owned_runtime_index.v1',owner=list(self.owner),
            file_identity=list(self.file_identity) if self.file_identity else None,prefix_bytes=self.size,
            prefix_sha256=self.prefix_sha256,parsed_events=self.parsed_events,refreshes=self.refreshes,
            validation_seconds=self.validation_seconds,complete_old_prefix_rehashed=True,
            raw_log_modified=False,qualification_granted=False,maximum_runtime_bytes=self.maximum_bytes,
            group_resource_scope=self.group_scope.check(self.root) if self.group_scope else None)
