"""CPU post-combine row join. No inference, mutation, collective or admission.

The adapter supplies an observed output-to-scheduler-row mapping. Shape, expert
counts and baseline routing alone cannot establish this mapping.
"""


def combine_row_owners(*, request_order, scheduled_counts, computed_tokens,
                       output_token_rows, active_mask):
    def need(ok, reason):
        if not ok:
            raise ValueError(reason)
    need(type(request_order) is list and all(type(r) is str and r for r in request_order)
         and len(set(request_order)) == len(request_order), 'COMBINE_REQUEST_ORDER_UNPROVEN')
    need(type(scheduled_counts) is dict and type(computed_tokens) is dict
         and set(request_order) == set(scheduled_counts) == set(computed_tokens),
         'COMBINE_CURRENT_SCHEDULER_JOIN_REQUIRED')
    rows = []
    for rid in request_order:
        n, start = scheduled_counts[rid], computed_tokens[rid]
        need(type(n) is int and n > 0 and type(start) is int and start >= 0,
             'COMBINE_TOKEN_RANGE_UNPROVEN')
        rows.extend((rid, start + i) for i in range(n))
    need(type(output_token_rows) is list and type(active_mask) is list
         and len(output_token_rows) == len(active_mask)
         and all(type(a) is bool for a in active_mask), 'COMBINE_OUTPUT_MAPPING_REQUIRED')
    selected, owners = [], []
    for row, active in zip(output_token_rows, active_mask):
        if not active:
            need(row is None, 'COMBINE_PADDING_MUST_BE_UNMAPPED')
            owners.append(None)
            continue
        need(type(row) is int and 0 <= row < len(rows), 'COMBINE_ROW_OUTSIDE_SCHEDULER')
        selected.append(row)
        owners.append(rows[row])
    need(len(selected) == len(set(selected)) and set(selected) == set(range(len(rows))),
         'COMBINE_POSTCOMBINE_ROWS_NOT_ONE_TO_ONE')
    return owners


def within_request_shift(owners):
    """Plan the output-row proxy; padding and single-row requests stay put.

    No tensor is modified here. Runtime site/consumer/lease review is still needed.
    """
    groups = {}
    for i, owner in enumerate(owners):
        if owner is not None:
            groups.setdefault(owner[0], []).append(i)
    mapping = list(range(len(owners)))
    for indices in groups.values():
        for dest, source in zip(indices, indices[1:] + indices[:1]):
            mapping[dest] = source
    return mapping
