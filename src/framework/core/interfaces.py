from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

import torch
from torch import Tensor


class Configurable(ABC):
    @abstractmethod
    def configure(self, config: Dict[str, Any]) -> None:
        ...


class Trainable(ABC):
    @abstractmethod
    def train_step(self, batch: Any) -> Dict[str, Tensor]:
        ...

    @abstractmethod
    def validation_step(self, batch: Any) -> Dict[str, Tensor]:
        ...


class Evaluable(ABC):
    @abstractmethod
    def evaluate(self, dataloader: torch.utils.data.DataLoader) -> Dict[str, float]:
        ...


class Predictable(ABC):
    @abstractmethod
    def predict(self, x: Tensor) -> Tensor:
        ...

    @abstractmethod
    def predict_with_uncertainty(self, x: Tensor) -> Dict[str, Tensor]:
        ...


class Plottable(ABC):
    @abstractmethod
    def plot(self, save_path: Optional[str] = None) -> None:
        ...
