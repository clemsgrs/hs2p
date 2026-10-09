"""Reject configuration values whose type does not match the typed configs' annotations.

The dataclass annotations on the typed configs are the only type schema. A value is
accepted only when it already has the annotated type, or widens to it without loss; it
is never converted from another type.
"""

import dataclasses
import functools
import os
import types
import typing
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, TypeVar

import numpy as np

_Config = TypeVar("_Config")


class _WrongType(Exception):
    """A value does not match its annotation; carries the expected-type phrase."""

    def __init__(self, expected: str) -> None:
        super().__init__(expected)
        self.expected = expected


def construct_config(cls: type[_Config], values: Mapping[str, Any]) -> _Config:
    """Build ``cls`` from values keyed by their config path (``tiling.params.overlap``).

    The field each value fills is the path's last component. Errors name the full config
    path rather than the dataclass field.
    """
    return cls(**check_config_values(cls, values))


def normalize_config_fields(config: Any) -> None:
    """Check every field of a frozen config dataclass and store the normalized values.

    Called from ``__post_init__``; errors name the field as ``ClassName.field``.
    """
    cls = type(config)
    values = {
        f"{cls.__name__}.{field.name}": getattr(config, field.name)
        for field in dataclasses.fields(cls)
    }
    for field_name, value in check_config_values(cls, values).items():
        object.__setattr__(config, field_name, value)


def check_config_values(cls: type, values: Mapping[str, Any]) -> dict[str, Any]:
    """Return ``values`` keyed by field name, each checked against its annotation.

    Raise one ``ValueError`` listing every value of the wrong type.
    """
    hints = _type_hints(cls)
    checked: dict[str, Any] = {}
    errors: list[str] = []
    for path, value in values.items():
        field_name = path.rsplit(".", 1)[-1]
        try:
            checked[field_name] = _check(hints[field_name], value)
        except _WrongType as wrong:
            errors.append(
                f"Config value {path} must be {wrong.expected}, "
                f"got {type(value).__name__} {value!r}"
            )
    if errors:
        raise ValueError("\n".join(errors))
    return checked


@functools.cache
def _type_hints(cls: type) -> dict[str, Any]:
    return typing.get_type_hints(cls)


def _check(annotation: Any, value: Any) -> Any:
    """Return ``value`` as stored for ``annotation``, or raise ``_WrongType``."""
    if annotation is bool:
        if isinstance(value, (bool, np.bool_)):
            return bool(value)
        raise _WrongType("a bool")
    if annotation is int:
        if _is_integer(value):
            return int(value)
        raise _WrongType("an int")
    if annotation is float:
        if _is_integer(value) or isinstance(value, (float, np.floating)):
            return float(value)
        raise _WrongType("a float")
    if annotation is str:
        if isinstance(value, str):
            return value
        raise _WrongType("a str")
    if annotation is Path:
        if isinstance(value, (str, os.PathLike)):
            return Path(value)
        raise _WrongType("a path")
    origin, args = typing.get_origin(annotation), typing.get_args(annotation)
    if (
        origin in (types.UnionType, typing.Union)
        and len(args) == 2
        and type(None) in args
    ):
        (inner,) = (arg for arg in args if arg is not type(None))
        if value is None:
            return None
        try:
            return _check(inner, value)
        except _WrongType as wrong:
            raise _WrongType(f"{wrong.expected} or None") from None
    if origin is tuple and args and Ellipsis not in args:
        # A fixed-length tuple such as ``tuple[int, int, int]``. Any non-string sequence
        # of that length is accepted (YAML gives a list) and stored as a tuple.
        expected = f"a tuple of {_describe_items(args)}"
        if (
            not isinstance(value, Sequence)
            or isinstance(value, (str, bytes))
            or len(value) != len(args)
        ):
            raise _WrongType(expected)
        try:
            return tuple(_check(arg, item) for arg, item in zip(args, value))
        except _WrongType:
            raise _WrongType(expected) from None
    if origin is Mapping:
        # Only the container is checked: entries are the user's open label maps, which
        # keep their own checks (``min_coverage`` values may be None to drop a label).
        if isinstance(value, Mapping):
            return value
        raise _WrongType("a mapping")
    raise TypeError(f"No configuration value check for annotation {annotation!r}")


def _describe_items(args: tuple[Any, ...]) -> str:
    if len(set(args)) == 1:
        return f"{len(args)} {args[0].__name__}s"
    return ", ".join(arg.__name__ for arg in args)


def _is_integer(value: Any) -> bool:
    """An integer that is not a bool (Python's ``bool`` subclasses ``int``)."""
    return isinstance(value, (int, np.integer)) and not isinstance(value, bool)
