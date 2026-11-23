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
                request_ids=list(request_ids), qualification_granted=False)
