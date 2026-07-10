from typing import Any, Callable, Dict, List, Optional, Type

from src.framework.core.exceptions import RegistryError


class Registry:
    def __init__(self, name: str) -> None:
        self._name = name
        self._items: Dict[str, Any] = {}

    def register(self, name: Optional[str] = None) -> Callable:
        def decorator(cls_or_fn: Any) -> Any:
            key = name or cls_or_fn.__name__
            if key in self._items:
                raise RegistryError(
                    f"'{key}' is already registered in registry '{self._name}'"
                )
            self._items[key] = cls_or_fn
            return cls_or_fn
        return decorator

    def get(self, key: str) -> Any:
        if key not in self._items:
            raise RegistryError(
                f"'{key}' not found in registry '{self._name}'. "
                f"Available: {list(self._items.keys())}"
            )
        return self._items[key]

    def list(self) -> List[str]:
        return list(self._items.keys())

    def __contains__(self, key: str) -> bool:
        return key in self._items


MODELS = Registry("models")
LOSSES = Registry("losses")
METRICS = Registry("metrics")
DATASETS = Registry("datasets")
TRANSFORMS = Registry("transforms")
OPTIMIZERS = Registry("optimizers")
SCHEDULERS = Registry("schedulers")
CALLBACKS = Registry("callbacks")
