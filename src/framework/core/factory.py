from typing import Any, Dict

import src.framework.models  # noqa: F401 — trigger registry decoration
import src.framework.evaluation  # noqa: F401
import src.framework.losses  # noqa: F401
import src.research.faupnet  # noqa: F401
import src.research.uwacl  # noqa: F401

from src.framework.core.registry import (
    CALLBACKS,
    DATASETS,
    LOSSES,
    METRICS,
    MODELS,
    OPTIMIZERS,
    SCHEDULERS,
    TRANSFORMS,
    Registry,
)


def _build_from_registry(registry: Registry, config: Dict[str, Any], **kwargs: Any) -> Any:
    cfg = dict(config)
    name = cfg.pop("name")
    cls = registry.get(name)
    return cls(**cfg, **kwargs)


def build_model(config: Dict[str, Any], **kwargs: Any) -> Any:
    return _build_from_registry(MODELS, config, **kwargs)


def build_loss(config: Dict[str, Any], **kwargs: Any) -> Any:
    return _build_from_registry(LOSSES, config, **kwargs)


def build_metric(config: Dict[str, Any], **kwargs: Any) -> Any:
    return _build_from_registry(METRICS, config, **kwargs)


def build_dataset(config: Dict[str, Any], **kwargs: Any) -> Any:
    return _build_from_registry(DATASETS, config, **kwargs)


def build_transform(config: Dict[str, Any], **kwargs: Any) -> Any:
    return _build_from_registry(TRANSFORMS, config, **kwargs)


def build_optimizer(config: Dict[str, Any], params: Any, **kwargs: Any) -> Any:
    return _build_from_registry(OPTIMIZERS, {**config, "params": params}, **kwargs)


def build_scheduler(config: Dict[str, Any], optimizer: Any, **kwargs: Any) -> Any:
    return _build_from_registry(SCHEDULERS, {**config, "optimizer": optimizer}, **kwargs)


def build_callback(config: Dict[str, Any], **kwargs: Any) -> Any:
    return _build_from_registry(CALLBACKS, config, **kwargs)
