"""Model registry for string-based model lookup.

Provides a simple registry mapping model names (strings) to model classes.
All registered models must be subclasses of BaseModel.
"""

from emo_mocap.models.base import BaseModel

_REGISTRY: dict[str, type[BaseModel]] = {}


def register_model(name: str, cls: type[BaseModel]) -> None:
    """Register a model class under the given name.

    Args:
        name: string identifier for the model (e.g., "stgcn")
        cls: model class, must be a subclass of BaseModel

    Raises:
        TypeError: if cls is not a subclass of BaseModel
        ValueError: if name is already registered
    """
    if not (isinstance(cls, type) and issubclass(cls, BaseModel)):
        raise TypeError(
            f"Cannot register {cls}: must be a subclass of BaseModel"
        )
    if name in _REGISTRY:
        raise ValueError(
            f"Model '{name}' is already registered (to {_REGISTRY[name]})"
        )
    _REGISTRY[name] = cls


def get_model(name: str) -> type[BaseModel]:
    """Look up a registered model class by name.

    Args:
        name: registered model name

    Returns:
        The model class

    Raises:
        KeyError: if name is not registered
    """
    if name not in _REGISTRY:
        available = ", ".join(sorted(_REGISTRY)) or "(none)"
        raise KeyError(
            f"Unknown model '{name}'. Available: {available}"
        )
    return _REGISTRY[name]


def list_models() -> list[str]:
    """Return sorted list of all registered model names."""
    return sorted(_REGISTRY.keys())
