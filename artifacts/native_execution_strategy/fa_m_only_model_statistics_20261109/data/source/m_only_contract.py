"""M-only controls and factual plan checks shared by capture and acceptance."""


def require_selection(config):
    if config.adapter.version!='validated' or config.selection_target!='query_tile':
        raise ValueError('This experiment requires the validated query_tile-only adapter')


def require_plan(parameters,controlled):
    expected_m=64 if controlled else 128
    if parameters['query_tile']!=expected_m or parameters['key_group_ratio']!=8:
        raise ValueError('Actual execution must retain nRatio8 and use M64 only in controlled mode')

