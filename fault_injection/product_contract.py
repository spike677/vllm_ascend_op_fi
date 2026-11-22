"""Product request policy. No prompt registry, device selectors or admission grant."""
from dataclasses import dataclass


COUNTERS = ('observed', 'eligible', 'attempted', 'applied', 'changed', 'consumed', 'restored', 'failed')
PRODUCT_CANDIDATES = ('18.api_half', '39.output_proxy', '22.wrong_row', '28.fia.denominator_replay', '17.spatial_padding')


@dataclass(frozen=True)
class RequestScope:
    mode: str = 'all'
    count: int | None = None

    def __post_init__(self):
        if self.mode not in ('all', 'next') or (self.mode == 'all' and self.count is not None):
            raise ValueError('PRODUCT_SCOPE_INVALID')
        if self.mode == 'next' and (type(self.count) is not int or not 1 <= self.count <= 100000):
            raise ValueError('PRODUCT_POSITIVE_REQUEST_COUNT_REQUIRED')

    def includes(self, ordinal):
        return self.mode == 'all' or ordinal < self.count


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
        sites.append({key: event.get(key) for key in (
            'site', 'eligible', 'reason', 'changed', 'nonfinite_before', 'nonfinite_after',
            'consumer_operator', 'consumed', 'restored', 'before_sha256', 'restored_sha256')})
    return dict(request_id=request_id, rank=rank, status='NOT_INJECTED' if skipped else 'OBSERVED',
                reasons=list(skipped), coverage=counts, sites=sites,
                changed_elements=sum(e.get('changed', 0) for e in events),
                completed=completed, bytes_restored=completed and counts['failed'] == 0,
                model_baseline_comparison='NOT_PERFORMED_FOR_USER_REQUEST')
