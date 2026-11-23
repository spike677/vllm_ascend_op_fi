"""Native product request evidence; no admission grant or new lifecycle."""
from .serving_context import require

MECHANISMS = ('41.tail_column', '26.fia_destination', '20.buffer_substitute')


def _sha(value):
    return type(value) is str and len(value) == 64 and all(c in '0123456789abcdef' for c in value)


def request_evidence(binding, site, events, request_ids):
    require(binding.mechanism in MECHANISMS, 'PRODUCT_NATIVE_MECHANISM_REQUIRED')
    library = binding.native_library
    require(type(library) in (list, tuple) and len(library) == 2 and _sha(library[1]),
            'PRODUCT_NATIVE_LIBRARY_IDENTITY_REQUIRED')
    require(site.native.count() == 0, 'PRODUCT_NATIVE_LEASES_REMAIN')
    require(type(request_ids) is list and len(request_ids) == 1, 'PRODUCT_NATIVE_SERIAL_REQUEST_REQUIRED')
    regions = []
    for event in events:
        require(event.get('request_ids') == request_ids, 'PRODUCT_NATIVE_EVENT_REQUEST_MISMATCH')
        if not event.get('eligible', True):
            require(bool(event.get('reason')), 'PRODUCT_NATIVE_SKIP_REASON_REQUIRED')
            continue
        require(event.get('applied') is True and event.get('consumed') is True
                and event.get('restored') is True and not event.get('failed'),
                'PRODUCT_NATIVE_CONSUMPTION_OR_RESTORE_UNPROVEN')
        require(type(event.get('region_elements')) is int and event['region_elements'] > 0
                and type(event.get('consumer_operator')) is str and bool(event['consumer_operator'])
                and _sha(event.get('before_sha256'))
                and event['before_sha256'] == event.get('restored_sha256'),
                'PRODUCT_NATIVE_REGION_PROOF_INCOMPLETE')
        if binding.mechanism != '41.tail_column':
            require(_sha(event.get('source_sha256'))
                    and event['source_sha256'] == event.get('after_sha256'),
                    'PRODUCT_NATIVE_CONTENT_SOURCE_UNPROVEN')
        regions.append({k: event.get(k) for k in (
            'site', 'writer', 'consumer_operator', 'consumer_argument', 'region_elements',
            'source_sha256', 'before_sha256', 'after_sha256', 'restored_sha256',
            'slots', 'page_mapping', 'changed')})
    return dict(library_sha256=library[1], regions=regions, leases=0,
                request_ids=list(request_ids), runtime_identity=getattr(binding,'native_runtime_identity',None),
                qualification_granted=False)


# A2 TP2 measured; A3 TP1/TP2 acceptance candidate after yellow build text review.
# Legacy function names retained for existing trusted bootstrap imports.
BLUE_NATIVE_LIBRARY_SHA256 = 'd9aa0d9488c5c1e8f7fca10c297e4968687ceb6a24866d8346bd33a58afce274'


def validate_blue_native_library(path, sha, source, tp):
    from pathlib import Path
    import hashlib
    require(type(tp) is int and ((source == 'BLUE_COMBINATION' and tp == 2)
            or (source == 'YELLOW_DEDICATED_INSTANCE' and tp in (1,2))),
            'PRODUCT_NATIVE_A3_OR_TP1_NOT_ADMITTED')
    require(type(path) is str and Path(path).is_absolute() and sha == BLUE_NATIVE_LIBRARY_SHA256,
            'PRODUCT_NATIVE_REVIEWED_LIBRARY_REQUIRED')
    require(Path(path).is_file() and hashlib.sha256(Path(path).read_bytes()).hexdigest() == sha,
            'PRODUCT_NATIVE_LIBRARY_HASH_MISMATCH')
    return (path, sha)


def validate_blue_native_worker(worker, source):
    from fault_injection.environment import _live_soc, _toolkit
    soc, _ = _live_soc()
    tp=worker.model_runner.vllm_config.parallel_config.tensor_parallel_size
    blue=source=='BLUE_COMBINATION'
    yellow=source=='YELLOW_DEDICATED_INSTANCE'
    require(soc is not None and ((blue and soc['family']=='A2' and soc['observed_name']=='Ascend910B4' and tp==2)
            or (yellow and soc['family']=='A3' and soc['observed_name']=='Ascend910_9382' and tp in (1,2))),
            'PRODUCT_NATIVE_REQUIRES_A2_TP2_OR_REVIEWED_A3_ACCEPTANCE')
    toolkit = _toolkit()
    require(toolkit['status'] == 'FOUND' and len(toolkit['installations']) == 1
            and toolkit['installations'][0]['fields'].get('version') == '9.1.0',
            'PRODUCT_NATIVE_REQUIRES_CANN91_TOOLKIT')
    identity=dict(source=source,tp=tp,soc=soc,toolkit_version='9.1.0',qualification_granted=False)
    if yellow:
        from fault_injection.environment import inspect_environment
        actual=inspect_environment(additional_libraries=('libascendcl.so','libruntime.so'))
        require(actual['maps_status']=='OBSERVED','PRODUCT_NATIVE_RUNTIME_MAPPING_UNOBSERVED')
        linked=actual['loaded_toolkit_links']['libraries']
        verified=[]
        for name in ('libascendcl.so','libruntime.so'):
            rows=[r for r in actual['loaded_libraries'] if r['name']==name]
            links=[r for r in linked if r['name']==name]
            require(len(rows)==1 and rows[0]['status']=='MAPPED_FILE_VERIFIED'
                and len(links)==1 and links[0]['status']=='PATH_LINKED_TO_INSTALLATION'
                and len(links[0]['matches'])==1 and links[0]['matches'][0]['version']=='9.1.0',
                'PRODUCT_NATIVE_LOADED_CANN91_UNPROVEN:'+name)
            verified.append(dict(name=name,sha256=rows[0]['sha256'],elf_build_id=rows[0].get('elf_build_id'),
                                 version='9.1.0',metadata_sha256=links[0]['matches'][0]['metadata_sha256']))
        identity['loaded_runtime']=verified
        identity['cann_observation_source']='VERIFIED_MAPPED_ELF_LINKED_TO_INSTALLED_VERSION_METADATA'
    return identity
