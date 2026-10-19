from __future__ import annotations

from typing import Callable

MethodFn = Callable[..., dict]
METHODS: dict[str, MethodFn] = {}


def register(name: str):
    def deco(fn: MethodFn) -> MethodFn:
        METHODS[name] = fn
        return fn

    return deco


def get_method(name: str) -> MethodFn:
    if name not in METHODS:
        available = ", ".join(sorted(METHODS)) or "(none)"
        raise ValueError(f"Unknown method '{name}'. Available: {available}")
    return METHODS[name]


def list_methods() -> list[str]:
    return sorted(METHODS)
