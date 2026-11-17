"""Strict versioned configuration; no implicit reinterpretation of old triggers."""
from copy import deepcopy
import json
from pathlib import Path

from fault_registry import ROOT, resolve


class ConfigError(ValueError):
    code = 'INVALID_CONFIG'

    def __init__(self, message, path=''):
        super().__init__(message)
        self.path = path

    def to_dict(self):
        return dict(code=self.code, message=str(self), path=self.path)


def _load_schema(name):
    schema = json.loads((ROOT/'schemas'/name).read_text(encoding='utf-8'))
    def expand(value):
        if isinstance(value, dict):
            if '$ref' in value:
                return _load_schema(value['$ref'])
            return {k:expand(v) for k,v in value.items()}
        return [expand(v) for v in value] if isinstance(value,list) else value
    return expand(schema)


def validate_config(config, mechanism=None):
    from jsonschema import Draft202012Validator
    try:
        # Reject NaN/Inf, non-JSON objects and aliasing to caller-owned dictionaries.
        document = json.loads(json.dumps(config, allow_nan=False))
    except (TypeError, ValueError) as exc:
        raise ConfigError('JSON_CONFIG_REQUIRED: ' + str(exc)) from exc
    validator = Draft202012Validator(_load_schema('common.schema.json'))
    error = next(validator.iter_errors(document), None)
    if error:
        raise ConfigError(error.message, '/'.join(map(str,error.absolute_path)))
    try:
        definition = resolve(document['mechanism']['id'])
        if mechanism is not None and resolve(mechanism)['legacy_id'] != definition['legacy_id']:
            raise ConfigError('MECHANISM_MISMATCH')
    except ValueError as exc:
        raise ConfigError(str(exc), 'mechanism/id') from exc
    if document['execution']['backend'] not in definition['backends']:
        raise ConfigError('BACKEND_NOT_REGISTERED', 'execution/backend')
    if document['fault']['type'] != definition['primitive']:
        raise ConfigError('PRIMITIVE_MISMATCH', 'fault/type')
    error = next(Draft202012Validator(_load_schema(str(Path(definition['schema']).relative_to('schemas'))))
                 .iter_errors(document['fault']['parameters']), None)
    if error:
        raise ConfigError(error.message, 'fault/parameters/'+'/'.join(map(str,error.absolute_path)))
    trigger = document['trigger']
    extra = {'nth':{'ordinal'},'every_nth':{'ordinal'},'probability':{'probability','seed'},'condition':{'condition'}}.get(trigger['mode'],set())
    if set(trigger) != {'mode','scope','max_hits'} | extra:
        raise ConfigError('TRIGGER_MODE_FIELDS_MISMATCH', 'trigger')
    if trigger['mode']=='condition':
        condition=trigger['condition']
        if (condition['op']=='in') != isinstance(condition['value'],list):
            raise ConfigError('CONDITION_VALUE_TYPE_MISMATCH','trigger/condition')
    target=document['target']
    if (target['request_scope']=='selected') != bool(target['request_ids']):
        raise ConfigError('SELECTED_REQUEST_IDS_REQUIRED', 'target/request_ids')
    document['mechanism']['id']=definition['legacy_id']
    if document['execution']['cann']=='9.1':document['execution']['cann']='9.1.0'
    return document


def example_config(mechanism):
    row=resolve(mechanism)
    return dict(schema='demoncat.fault_config.v1',mechanism=dict(id=row['legacy_id'],version='1'),
        execution=dict(backend=row['backends'][0],operator=row['operator_paths'][0]['family'],model='observed',
                       cann='cpu' if row['adapter_status']=='CPU_EXECUTABLE' else '9.1.0'),
        target=dict(layer=None,operator_call=None,rank=[0],device=[0],request_scope='all',request_ids=[]),
        trigger=dict(mode='first',scope='request',max_hits=1),
        fault=dict(type=row['primitive'],parameters=deepcopy(row['parameters'])),
        lifecycle=dict(recovery='restore_original',require_drain=True,require_healthy_replay=True),
        evidence=dict(independent_observer=True,capture_baseline=True,capture_fault=True,capture_off=True))


def migrate_legacy(config):
    """Only the exact former FaultConfig is translated; ambiguous fields refuse."""
    from fault_injection.lifecycle import FaultConfig
    try:
        old=FaultConfig.from_dict(config)
    except (ValueError,TypeError) as error:
        raise ConfigError('LEGACY_CONFIG_REJECTED: '+str(error)) from error
    if old.arm!='fault' or old.unsupported!='error' or old.evidence_dir is not None:
        raise ConfigError('LEGACY_MODE_REQUIRES_ORIGINAL_API')
    result=example_config(old.mechanism)
    result['trigger']=dict(mode='every_match',scope='process',max_hits=None)
    return validate_config(result)
