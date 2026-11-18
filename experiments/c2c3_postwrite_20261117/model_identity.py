"""Actual local pretrained artifacts; no model-name or claimed SHA substitution."""
import hashlib
import inspect
import json
from pathlib import Path


def file_sha(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        while chunk := stream.read(1024 * 1024):result.update(chunk)
    return result.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def safetensors_identity(model_path):
    root = Path(model_path).resolve(strict=True)
    single, index = root/'model.safetensors', root/'model.safetensors.index.json'
    if single.is_file():
        names, manifest = ['model.safetensors'], None
    elif index.is_file():
        document = json.loads(index.read_text())
        mapping = document.get('weight_map')
        if not isinstance(mapping, dict) or not mapping:raise ValueError('ACTUAL_SAFETENSORS_INDEX_REQUIRED')
        names, manifest = sorted(set(mapping.values())), dict(file=index.name, sha256=file_sha(index))
    else:
        raise ValueError('LOCAL_SAFETENSORS_WEIGHTS_REQUIRED')
    files = {}
    for name in names:
        path = (root/name).resolve(strict=True)
        if not isinstance(name, str) or not path.is_relative_to(root) or path.suffix != '.safetensors':
            raise ValueError('SAFETENSORS_PATH_OUTSIDE_MODEL')
        files[name] = dict(bytes=path.stat().st_size, sha256=file_sha(path))
    config = root/'config.json'
    if not config.is_file():raise ValueError('ACTUAL_MODEL_CONFIG_REQUIRED')
    record = dict(format='safetensors', files=files, index=manifest, config_sha256=file_sha(config),
                  loader_selection='from_pretrained(use_safetensors=True, local_files_only=True)')
    return dict(path=str(root), manifest=record, weights_sha256=digest(record))


def vllm_safetensors_identity(model_path):
    """Hash the exact local safetensors set, including nonstandard index names.

    vLLM's explicit safetensors loader can consume locally quantized shards
    whose index is not named model.safetensors.index.json. Accept one actual
    index only when it names exactly the files visible to that loader.
    """
    root = Path(model_path).resolve(strict=True)
    actual = {path.name: path for path in root.glob('*.safetensors')}
    indexes = sorted(root.glob('*.safetensors.index.json'))
    if len(indexes) > 1 or not actual:
        raise ValueError('UNAMBIGUOUS_LOCAL_SAFETENSORS_SELECTION_REQUIRED')
    index = None
    if indexes:
        item = indexes[0]
        if item.is_symlink() or not item.is_file():
            raise ValueError('REGULAR_LOCAL_SAFETENSORS_INDEX_REQUIRED')
        mapping = json.loads(item.read_text()).get('weight_map')
        if not isinstance(mapping, dict) or not mapping or not all(isinstance(k, str) and k for k in mapping):
            raise ValueError('ACTUAL_SAFETENSORS_INDEX_REQUIRED')
        names = set()
        for name in mapping.values():
            if (not isinstance(name, str) or '/' in name or '\\' in name
                    or Path(name).name != name or not name.endswith('.safetensors')):
                raise ValueError('FLAT_LOCAL_SAFETENSORS_SHARD_REQUIRED')
            names.add(name)
        if names != set(actual):
            raise ValueError('VLLM_SAFETENSORS_FILE_SELECTION_AMBIGUOUS')
        index = dict(file=item.name, sha256=file_sha(item))
    elif set(actual) != {'model.safetensors'}:
        raise ValueError('LOCAL_SHARD_INDEX_REQUIRED')
    files = {}
    for name, path in sorted(actual.items()):
        if path.is_symlink() or not path.is_file() or not path.resolve().is_relative_to(root):
            raise ValueError('REGULAR_LOCAL_SAFETENSORS_SHARD_REQUIRED')
        files[name] = dict(bytes=path.stat().st_size, sha256=file_sha(path))
    config = root/'config.json'
    if config.is_symlink() or not config.is_file():
        raise ValueError('ACTUAL_MODEL_CONFIG_REQUIRED')
    record = dict(format='safetensors', files=files, index=index, config_sha256=file_sha(config),
                  loader_selection="vllm.LLM(load_format='safetensors', trust_remote_code=False)")
    quant = root/'quant_model_description.json'
    if quant.exists():
        if quant.is_symlink() or not quant.is_file():
            raise ValueError('REGULAR_LOCAL_QUANT_DESCRIPTION_REQUIRED')
        record['quantization_description'] = dict(file=quant.name, bytes=quant.stat().st_size, sha256=file_sha(quant))
    return dict(path=str(root), manifest=record, weights_sha256=digest(record))


def model_source_identity(model):
    files, classes = {}, []
    for module in model.modules():
        cls = type(module)
        name = cls.__module__ + '.' + cls.__qualname__
        path = Path(inspect.getsourcefile(cls)).resolve(strict=True)
        files[str(path)] = file_sha(path)
        classes.append(dict(class_name=name, source=str(path)))
    record = dict(files=files, classes=classes)
    return dict(manifest=record, source_sha256=digest(record))
