from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional
import torch


def auto_device(preferred: Optional[str] = None) -> torch.device:
    if preferred:
        return torch.device(preferred)
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


@dataclass
class TrainConfig:
    data_path: str
    model: Literal["bigram", "gpt"] = "bigram"

    # entrenamiento
    batch_size: int = 32
    block_size: int = 128
    max_steps: int = 2000
    eval_interval: int = 200
    learning_rate: float = 3e-4
    device_str: Optional[str] = None  # "cpu" | "mps" | "cuda"

    # gpt
    n_embd: int = 128
    n_head: int = 4
    n_layer: int = 2
    dropout: float = 0.1

    def device(self) -> torch.device:
        return auto_device(self.device_str)

