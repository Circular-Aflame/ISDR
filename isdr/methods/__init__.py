from .registry import get_method, list_methods, register

from . import core  # noqa: F401
from . import relation  # noqa: F401
from . import target  # noqa: F401
from . import unsupervised  # noqa: F401

__all__ = ["get_method", "list_methods", "register"]
