"""Read-only deployment-file loading, separate from the public FaultConfig."""
import hashlib
import json
from pathlib import Path
from .config import FaultConfig
from fault_injection.product_contract import PRODUCT_CANDIDATES


def load_product_deployment(path):
    raw = Path(path).read_bytes()
    if len(raw) > 65536:
        raise ValueError('PRODUCT_DEPLOYMENT_FILE_TOO_LARGE')
    config = json.loads(raw)
    fields = {'schema','profile','fault','model','tp','mode','source','source_manifest',
              'source_manifest_sha256','native_library','native_sha256','canary','evidence_directory'}
    if type(config) is not dict or set(config) != fields or config['schema'] != 'demoncat.product.deployment.v1':
        raise ValueError('PRODUCT_DEPLOYMENT_SCHEMA_INVALID')
    fault = FaultConfig.from_dict(config['fault'])
    if fault.mechanism not in PRODUCT_CANDIDATES or fault.implementation_version != '1':
        raise ValueError('PRODUCT_VARIANT_NOT_ADAPTED')
    if config['mode'] != 'PRODUCT_ACCEPTANCE_ONLY' or config['source'] not in ('BLUE_COMBINATION','YELLOW_DEDICATED_INSTANCE'):
        raise ValueError('PRODUCT_COMBINATION_NOT_REGISTERED')
    if type(config['tp']) is not int or config['tp'] not in (1,2):
        raise ValueError('PRODUCT_TP_NOT_REVIEWED')
    for key in ('model','profile','source_manifest','evidence_directory'):
        if type(config[key]) is not str or not config[key].strip():
            raise ValueError('PRODUCT_DEPLOYMENT_FIELD_REQUIRED:'+key)
    sha = config['source_manifest_sha256']
    if type(sha) is not str or len(sha) != 64 or hashlib.sha256(Path(config['source_manifest']).read_bytes()).hexdigest() != sha:
        raise ValueError('PRODUCT_SOURCE_MANIFEST_MISMATCH')
    if config['native_library'] is not None or config['native_sha256'] is not None:
        raise ValueError('PRODUCT_NATIVE_ADAPTER_PENDING')
    canary = config['canary']
    if type(canary) is not dict or set(canary) != {'prompt','image','max_tokens'} or type(canary['prompt']) is not str or not canary['prompt'] or type(canary['max_tokens']) is not int or canary['max_tokens'] != 4:
        raise ValueError('PRODUCT_CANARY_CONFIG_INVALID')
    if (fault.mechanism == '17.spatial_padding' and not (type(canary['image']) is str and canary['image'])) or (fault.mechanism != '17.spatial_padding' and canary['image'] is not None):
        raise ValueError('PRODUCT_CANARY_IMAGE_POLICY_INVALID')
    if canary['image'] is not None:
        from fault_injection.vision_input import load_fixed_image
        load_fixed_image(canary['image'])  # record identity at execution; no expected SHA requirement
    return config


def configure_product_deployments(worker, paths, *, authority, operator, canary_tokens):
    """Trusted bootstrap only; not a user RPC or a mechanism to grant admission.

    canary_tokens maps each configured profile to its processor-produced expanded
    prompt token IDs. Images must use the actual service multimodal processor.
    Permit installation stays with the existing deployment authority API.
    """
    from .deployment_authorization import authority_for
    from .product_binding import configure_product_profiles
    if authority_for(worker) is not authority or authority.authorize_operator(operator, 'configure_product_profiles') is not True:
        raise PermissionError('TRUSTED_PRODUCT_DEPLOYMENT_AUTHORITY_REQUIRED')
    if not isinstance(paths, (list, tuple)) or not paths:
        raise ValueError('PRODUCT_DEPLOYMENT_FILES_REQUIRED')
    profiles, permits = {}, []
    live = worker.model_runner.vllm_config
    for path in paths:
        deployment = load_product_deployment(path)
        if (deployment['model'] != live.model_config.model
                or deployment['tp'] != live.parallel_config.tensor_parallel_size
                or deployment['source'] != authority.source):
            raise ValueError('PRODUCT_LIVE_DEPLOYMENT_IDENTITY_MISMATCH')
        name = deployment['profile']
        tokens = canary_tokens.get(name)
        if name in profiles or type(tokens) is not list or len(tokens) < 2 or any(type(t) is not int or t < 0 for t in tokens):
            raise ValueError('PRODUCT_PROCESSED_CANARY_TOKENS_REQUIRED')
        mechanism = FaultConfig.from_dict(deployment['fault']).mechanism
        profiles[name] = dict(mechanism=mechanism, canary_prompts=[tokens], mode=deployment['mode'])
        permits.append(dict(profile=name, mechanisms=[mechanism], source=deployment['source'],
                            source_manifest_sha256=deployment['source_manifest_sha256']))
    if set(canary_tokens) != set(profiles):
        raise ValueError('PRODUCT_CANARY_PROFILE_SET_MISMATCH')
    configure_product_profiles(worker, profiles=profiles, authority=authority, operator=operator)
    return dict(status='PRODUCT_DEPLOYMENT_FILES_CONFIGURED', required_permits=permits,
                qualification_granted=False)
