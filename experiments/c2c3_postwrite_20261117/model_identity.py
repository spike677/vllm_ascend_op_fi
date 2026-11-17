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
