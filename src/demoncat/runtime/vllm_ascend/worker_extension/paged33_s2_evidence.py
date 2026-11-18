"""Fresh normal evidence and live first-layer input matching; no F execution.

Only trusted deployment code supplies paths. The complete independent reader
is rerun over immutable original artifacts; a saved success flag is not used
as authority. No object from this module can be supplied through public RPC.
"""
from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
import struct
import sys
from types import SimpleNamespace

from .serving_context import ContextRefused,require

_MINT=object()
FIRST='model.layers.0.self_attn.attn'
VARIANT='tnd_page16_arch22_s2_256'


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def normalized_input(signature):
    """Normalize only proved used-page IDs and the corresponding physical slot."""
    value=deepcopy(signature);table=value['block_table_values']
    require(len(table)==1 and len(table[0])==80,'S2_REFERENCE_COMPLETE_PAGE_TABLE_REQUIRED')
    pages=table[0][:65];count=value['cache_geometry']['key']['shape'][0]
    require(len(set(pages))==65 and all(type(n) is int and 0<=n<count for n in table[0])
        and value['cache_geometry']['key']['shape']==value['cache_geometry']['value']['shape']
        and value['cache_geometry']['key']['shape'][1:]==[16,8,128]
        and value['writer_consumer_slot']==[pages[-1]*16], 'S2_REFERENCE_VALID_LOGICAL_PAGE_MAP_REQUIRED')
    raw=struct.pack('<80i',*table[0])
    require(value['tensors']['block_table']['sha256']==hashlib.sha256(raw).hexdigest()
        and value['tensors']['block_table']['bytes']==len(raw),'S2_REFERENCE_RAW_TABLE_HASH_DIFFERS')
    table[0][:65]=list(range(65))
    value['tensors']['block_table']['sha256']='PROVEN_USED_PAGE_RELOCATION_ONLY'
    value['writer_consumer_slot']=['PROVEN_CURRENT_LOGICAL_POSITION_SLOT']
    return value


class VerifiedNormalEvidence:
    def __init__(self,seal,data,*,simulation=False):
        require(seal is _MINT,'S2_NORMAL_RAW_REVIEW_REQUIRED')
        self._json=json.dumps(data,sort_keys=True,allow_nan=False)
        self.fingerprint=digest(data);self.simulation=simulation

    def __reduce__(self):raise TypeError('Normal evidence must be reconstructed by trusted deployment from original artifacts')

    def data(self):
        value=json.loads(self._json)
        require(digest(value)==self.fingerprint,'S2_NORMAL_EVIDENCE_CHANGED')
        return value

    def match(self,binding):
        from .paged33_logical_binding import LogicalPagedInput
        require(type(binding) is LogicalPagedInput and binding.module_path==FIRST
            and not binding.closed and binding.cpu is self.simulation,'S2_ACTUAL_LIVE_INPUT_REQUIRED')
        reference=self.data()['reference_input'];current=binding.before['signature']
        require(normalized_input(reference)==normalized_input(current),'S2_NEW_CALL_DIFFERS_FROM_FRESH_NORMAL_INPUT')
        return dict(normal_evidence_sha256=self.fingerprint,raw_input_exact=reference==current,
            logical_input_equivalent=True,source_bound_reference=True,
            reference_pages=reference['block_table_values'][0][:65],current_pages=current['block_table_values'][0][:65],
            reference_slot=reference['writer_consumer_slot'],current_slot=current['writer_consumer_slot'],
            allowed_difference_fields=['used_page_ids','full_page_table_sha256','derived_writer_consumer_slot'])


def _module(name,path):
    path=Path(path).resolve(strict=True)
    if name in sys.modules:
        value=sys.modules[name]
        require(Path(value.__file__).resolve()==path,'S2_FOREIGN_NORMAL_READER_MODULE:'+name)
        return value
    spec=importlib.util.spec_from_file_location(name,path)
    value=importlib.util.module_from_spec(spec);sys.modules[name]=value;spec.loader.exec_module(value)
    return value


def normal_observer_root(receipt,expected_sha256):
    """Translate the Native receipt file to the offline reader's build root.

    These are distinct interfaces. Do not search parents or accept an
    arbitrary same-named receipt: the exact canonical file and hash must match.
    The frozen reader separately checks its source, selftest and library bytes.
    """
    path=Path(receipt)
    require(path.is_absolute() and path.name=='result.json'
        and path.parent.name=='attention-preflight' and not path.is_symlink()
        and path.resolve(strict=True)==path and path.is_file(),
        'S2_NORMAL_OBSERVER_RECEIPT_LAYOUT_REQUIRED')
    require(hashlib.sha256(path.read_bytes()).hexdigest()==expected_sha256,
            'S2_NORMAL_OBSERVER_RECEIPT_CHANGED')
    return path.parent.parent


def load_verified_normal_evidence(*,source,source_manifest_sha256,build_root,completion_sha256,
        observer,observer_receipt_sha256,stock_run,stock_sha256,baseline_run,baseline_sha256,
        report,report_sha256,current_execution):
    source=Path(source).resolve(strict=True);folder=source/'experiments/c2c3_closure_20261117'
    common=_module('arch22_normal_bridge_evidence',folder/'arch22_normal_bridge_evidence.py')
    frozen=common.frozen_source(source,source_manifest_sha256,build_variant=VARIANT)
    reader=_module('review_arch22_normal_bridge',folder/'review_arch22_normal_bridge.py')
    require(common.sha(Path(reader.__file__))==frozen['pins'][common.EXPERIMENT+'review_arch22_normal_bridge.py'],
            'S2_FROZEN_NORMAL_READER_DIFFERS')
    require(common.sha(report)==report_sha256,'S2_NORMAL_REVIEW_FILE_CHANGED')
    saved=common.read(report)
    observer_root=normal_observer_root(observer,observer_receipt_sha256)
    args=SimpleNamespace(source=source,source_manifest_sha256=source_manifest_sha256,build_root=Path(build_root),
        completion_sha256=completion_sha256,observer=observer_root,observer_receipt_sha256=observer_receipt_sha256,
        stock_run=Path(stock_run),stock_sha256=stock_sha256,baseline_run=Path(baseline_run),baseline_sha256=baseline_sha256,
        arch22_s2_normal=True)
    # Complete raw replay, including 435 payload pairs, runtime/ELF joins,
    # full final private POD, all cleanup and the original error records.
    fresh=reader.review(args)
    valid=(not fresh['issues'] and fresh.get('normal_payloads_exact') is True
        and fresh.get('complete_payload_pairs')==435 and fresh['build']['variant']==VARIANT
        and fresh['build']['completion_sha256']==completion_sha256
        and fresh['source']['manifest_sha256']==source_manifest_sha256
        and fresh['observer']['receipt_sha256']==observer_receipt_sha256
        and fresh['native_call_scope']==dict(module_path=FIRST,layout='TND',query_tokens=1,kv_length=1025,page_size=16,
            private_calls=4,natural_layers_per_phase=28,normal_only=True))
    if not valid:
        error=ContextRefused('S2_FRESH_NORMAL_RAW_REPLAY_FAILED')
        error.response=dict(schema='demoncat.paged33.normal_replay_failure.v1',
            observer_receipt=str(observer),observer_build_root=str(observer_root),
            observer_receipt_sha256=observer_receipt_sha256,
            replay={key:deepcopy(fresh.get(key)) for key in
                ('status','checks','issues','normal_payloads_exact','complete_payload_pairs','source','build','observer','native_call_scope')},
            qualification_granted=False)
        raise error
    # The published report is provenance only; its booleans cannot replace
    # the replay above. Check its pinned inputs and retain its original hash.
    require(saved['source']['manifest_sha256']==source_manifest_sha256
        and saved['build']['completion_sha256']==completion_sha256
        and saved['arms']['stock']['result_sha256']==stock_sha256
        and saved['arms']['baseline']['result_sha256']==baseline_sha256,'S2_NORMAL_REVIEW_INPUTS_CHANGED')
    original=common.read(Path(baseline_run)/'result.json');execution=original['execution_contract']
    require({k:v for k,v in execution.items() if k!='driver_sha256'}==
            {k:v for k,v in current_execution.items() if k!='driver_sha256'},'S2_NORMAL_MODEL_PACKAGE_OR_RUNNER_CHANGED')
    capture=original['phases'][0]['finish']['capture'];live=capture['first_decode_boundary']['logical_input_binding']
    reference=live['evidence']['before']['signature'];normalized_input(reference)
    data=dict(schema='demoncat.paged33.verified_normal_evidence.v1',variant=VARIANT,
        source_manifest_sha256=source_manifest_sha256,completion_sha256=completion_sha256,
        observer_receipt_sha256=observer_receipt_sha256,published_report_sha256=report_sha256,
        replayed_report_sha256=digest(fresh),stock_result_sha256=stock_sha256,private_result_sha256=baseline_sha256,
        reference_input=reference,immediate_outputs=live['evidence']['outputs'],
        model_outputs={key:{name:capture[key][name] for name in ('shape','dtype','sha256','finite')} for key in ('hidden','logits')},
        execution_contract=execution,normal_only=True,qualification_granted=False)
    require(common.sha(report)==report_sha256 and common.sha(Path(baseline_run)/'result.json')==baseline_sha256,
            'S2_NORMAL_ARTIFACT_CHANGED_DURING_REPLAY')
    return VerifiedNormalEvidence(_MINT,data)
