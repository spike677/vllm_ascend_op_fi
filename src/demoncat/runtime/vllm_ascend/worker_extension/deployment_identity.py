"""Read small artifact identities; missing observations never become defaults."""
import hashlib
from pathlib import Path


def model_artifact_identity(model_path):
    result = dict(quant_model_description_sha256=None, safetensors_index_sha256={})
    if not isinstance(model_path, (str, Path)):
        return result
    root = Path(model_path)
    try:
        result['quant_model_description_sha256'] = hashlib.sha256(
            (root / 'quant_model_description.json').read_bytes()).hexdigest()
    except OSError:
        pass
    try:
        for path in sorted(root.glob('*.safetensors.index.json')):
            try:
                result['safetensors_index_sha256'][path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
            except OSError:
                result['safetensors_index_sha256'][path.name] = None
    except OSError:
        pass
    return result


def _sha(value):
    return isinstance(value, str) and len(value) == 64 and all(c in '0123456789abcdef' for c in value)


def registration_identity_issues(actual):
    """4.9 prerequisites; image_digest is optional metadata, never a grant."""
    issues = []
    if not _sha(actual.get('quant_model_description_sha256')):
        issues.append('quant_model_description_sha256: MISSING_OR_INVALID')
    indices = actual.get('safetensors_index_sha256')
    if not isinstance(indices, dict) or not indices or not all(
            isinstance(name, str) and Path(name).name == name and name.endswith('.safetensors.index.json')
            and _sha(value) for name, value in indices.items()):
        issues.append('safetensors_index_sha256: MISSING_OR_INVALID')
    for name in ('max_model_len', 'max_num_batched_tokens', 'max_num_seqs'):
        if type(actual.get(name)) is not int or actual[name] <= 0:
            issues.append(name + ': MISSING_OR_INVALID')
    return issues
