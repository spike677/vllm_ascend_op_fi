"""Quantization Resolver: audited type/version contracts, not scale-name guessing."""
from .surface import stored, tensor_metadata, type_name


class QuantizationResolver:
    def __init__(self, contracts=()):
        # Trusted deployment data, never user-supplied RPC code or arbitrary rules.
        self.contracts = tuple(contracts)

    def resolve(self, module, record, runtime):
        method = stored(module, 'quant_method')
        method_type = type_name(method) if method is not None else None
        match = next((c for c in self.contracts if c['quant_method'] == method_type
                      and c['runtime_fingerprint'] == runtime and c.get('source_sha256')), None)
        if match is None:
            from .ascend_quantization import audited_ascend_contract
            match = audited_ascend_contract(method, runtime)
        fields = []
        registered = {p['name']: p for p in record['parameters']}
        names = {n for n in registered if any(t in n.lower() for t in ('scale', 'zero', 'offset', 'quant'))}
        names |= {n for n in vars(module) if not n.startswith('_') and n != 'quant_method'
                  and any(t in n.lower() for t in ('scale', 'zero', 'offset', 'quant'))}
        if match:
            names |= set(match['fields'])
        for name in sorted(names):
            parameter = registered.get(name)
            rule = match['fields'].get(name) if match else None
            if parameter:
                metadata, source = parameter['metadata'], parameter['kind']
            elif rule and rule.get('source') == 'stored_attribute':
                value = stored(module, name)
                metadata = tensor_metadata(value) if value is not None else {'metadata_status': 'UNKNOWN'}
                source = 'verified_stored_attribute'
            else:
                metadata, source = {'metadata_status': 'UNKNOWN; unverified stored attribute, value not read'}, 'stored_attribute_name_only'
            fields.append({'owner': record['module_name'], 'field': name,
                'category': ('MODEL_QUANTIZATION_SCALE' if 'scale' in rule.get('role', '') else 'MODEL_QUANTIZATION_PARAMETER')
                    if rule else 'UNRESOLVED_QUANTIZATION_FIELD',
                'role': rule.get('role', 'unknown') if rule else 'unknown',
                'target': rule.get('target') if rule else None,
                'granularity': rule.get('granularity', 'unknown') if rule else 'unknown',
                'axis': rule.get('axis') if rule else None, 'group': rule.get('group') if rule else None,
                'block_shape': rule.get('block_shape') if rule else None,
                'dynamic': rule.get('dynamic') if rule else None,
                'encoding': rule.get('encoding', 'unknown') if rule else 'unknown',
                'source': source, 'metadata': metadata, 'numeric_value': 'NOT_CAPTURED',
                'provider_compatibility': rule.get('providers', []) if rule else [],
                'evidence': match['source_sha256'] if rule else 'attribute name only; NOT role proof',
                'rebind': 'EVERY_INVOCATION' if rule and rule.get('dynamic') else 'EVERY_SERVICE_EPOCH'})
        return {'quant_method': method_type, 'resolution': 'AUDITED_IMPLEMENTATION' if match else 'UNKNOWN_IMPLEMENTATION',
                'observed_scheme_type': type_name(stored(method, 'quant_method')) if stored(method, 'quant_method') is not None else None,
                'scheme': match.get('scheme') if match else None,
                'source_evidence': match.get('source_evidence', []) if match else [],
                'activation_scale': match.get('activation_scale') if match else None,
                'model_fields': fields,
                'reference_policies': [{'category': 'REFERENCE_SCALE_POLICY', 'provider': 'tensor_absmax',
                    'target': 'observed floating module input', 'model_scale': False,
                    'capability': 'python.fake_requant.activation', 'status': 'RESEARCH_REFERENCE_ONLY'}]}
