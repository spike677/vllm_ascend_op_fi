"""Product request policy. No prompt registry, device selectors or admission grant."""
from dataclasses import dataclass


COUNTERS = ('observed', 'eligible', 'attempted', 'applied', 'changed', 'consumed', 'restored', 'failed')
# Native entries remain BLUE A2/TP2-only through product_config/product_binding.
PRODUCT_CANDIDATES = ('18.api_half', '39.output_proxy', '22.wrong_row', '28.fia.denominator_replay', '17.spatial_padding', 'quant.batch_shared_scale', '39.matmul_range', '33.paged_decode', '41.tail_column', '26.fia_destination', '20.buffer_substitute')


@dataclass(frozen=True)
class RequestScope:
    mode: str = 'all'
    def __post_init__(self):
        if self.mode != 'all':
            raise ValueError('PRODUCT_ONLY_ALL_REQUESTS_SUPPORTED')


def request_range(facts, *, max_context=1024, visual=False):
    """Facts must come from the live scheduler, not the end-user payload."""
    problems = []
    for key, expected in (('request_count', 1), ('single_chunk', True), ('fresh_cache', True),
                          ('speculative', False), ('lora', False), ('prompt_embeds', False)):
        if key not in facts or type(facts[key]) is not type(expected) or facts[key] != expected:
            problems.append('UNSUPPORTED_' + key.upper())
    length, generated = facts.get('prompt_length'), facts.get('max_tokens')
    if type(length) is not int or length < 2:
        problems.append('PROMPT_LENGTH_UNPROVEN')
    if type(generated) is not int or not 1 <= generated <= 4:
        problems.append('GENERATION_OUTSIDE_VALIDATED_1_TO_4')
    if type(length) is int and type(generated) is int and length + generated > max_context:
        problems.append('CONTEXT_EXCEEDS_VALIDATED_LIMIT')
    if visual:
        if facts.get('image_count') != 1 or facts.get('image_size') != [64, 64] or facts.get('video_count') != 0:
            problems.append('IMAGE_RANGE_NOT_VALIDATED')
    elif facts.get('image_count') != 0 or facts.get('video_count') != 0:
        problems.append('TEXT_ONLY_RANGE')
    return dict(eligible=not problems, reasons=problems)


def summarize_request(request_id, rank, events, *, skipped=(), completed=False):
    counts = dict.fromkeys(COUNTERS, 0)
    sites = []
    for event in events:
        counts['observed'] += 1
        eligible = event.get('eligible', True)
        counts['eligible'] += int(eligible)
        counts['attempted'] += int(eligible)
        for key in ('applied', 'consumed', 'restored', 'failed'):
            counts[key] += int(bool(event.get(key)))
        counts['changed'] += int(event.get('changed', 0) > 0)
        if eligible and not (event.get('applied') and event.get('consumed') and event.get('restored')
                             and event.get('before_sha256') is not None
                             and event.get('before_sha256') == event.get('restored_sha256')):
            counts['failed'] += int(not event.get('failed'))
        site={key: event.get(key) for key in (
            'site', 'eligible', 'reason', 'changed', 'nonfinite_before', 'nonfinite_after',
            'consumer_operator', 'consumed', 'restored', 'before_sha256', 'restored_sha256')}
        for key in ('full_range_control_passed','range_reference','non_target_exact','partition_reference'):
            if key in event:site[key]=event[key]
        sites.append(site)
    return dict(request_id=request_id, rank=rank, status='NOT_INJECTED' if skipped else 'OBSERVED',
                reasons=list(skipped), coverage=counts, sites=sites,
                changed_elements=sum(e.get('changed', 0) for e in events),
                completed=completed, bytes_restored=completed and counts['failed'] == 0,
                model_baseline_comparison='NOT_PERFORMED_FOR_USER_REQUEST')
