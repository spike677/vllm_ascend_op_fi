"""Small JSON serialization helpers for the shared v0 contracts."""

from __future__ import annotations

import json
from dataclasses import fields, is_dataclass
from enum import Enum
from typing import Any, Mapping, TypeVar


T = TypeVar("T", bound="JsonContract")


def require(mapping: Mapping[str, Any], key: str) -> Any:
    if key not in mapping:
        raise ValueError(f"missing required field: {key}")
    return mapping[key]


def as_mapping(value: Any, field_name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field_name} must be an object")
    return value


def extra_fields(mapping: Mapping[str, Any], known: set[str]) -> dict[str, Any]:
    return {key: value for key, value in mapping.items() if key not in known}


def enum_value(enum_type: type[Enum], value: Any, field_name: str) -> Any:
    try:
        return enum_type(str(value))
    except ValueError as exc:
        allowed = ", ".join(item.value for item in enum_type)
        raise ValueError(f"invalid {field_name}: {value!r}; expected one of {allowed}") from exc


def encode_json(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value):
        result: dict[str, Any] = {}
        extras: Mapping[str, Any] = {}
        for item in fields(value):
            field_value = getattr(value, item.name)
            if item.name == "extra":
                extras = field_value
            else:
                result[item.name] = encode_json(field_value)
        for key, field_value in extras.items():
            if key not in result:
                result[key] = encode_json(field_value)
        return result
    if isinstance(value, Mapping):
        return {str(key): encode_json(field_value) for key, field_value in value.items()}
    if isinstance(value, (tuple, list)):
        return [encode_json(item) for item in value]
    return value


class JsonContract:
    """Mixin for contracts that expose an explicit ``from_dict`` method."""

    def to_dict(self) -> dict[str, Any]:
        return encode_json(self)

    def to_json(self, *, indent: int | None = 2) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent, sort_keys=True)

    @classmethod
    def from_json(cls: type[T], payload: str) -> T:
        value = json.loads(payload)
        if not isinstance(value, dict):
            raise ValueError("contract JSON root must be an object")
        return cls.from_dict(value)  # type: ignore[attr-defined,no-any-return]
