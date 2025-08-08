from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class BigramConfig:
    vocab_size: int


class BigramLanguageModel(nn.Module):
    """Modelo Bigram simple.

    Aprende P(siguiente_token | token_actual) mediante una tabla de logits.
    """

    def __init__(self, config: BigramConfig):
        super().__init__()
        self.config = config
        # Tabla de logits: para cada token actual, predice distribución del siguiente token
        self.token_embedding_table = nn.Embedding(config.vocab_size, config.vocab_size)

    def forward(self, idx: torch.Tensor, targets: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        # idx: (B, T) con tokens
        # logits: (B, T, vocab_size)
        logits = self.token_embedding_table(idx)
        loss = None
        if targets is not None:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)
        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        block_size: Optional[int] = None,
        forbidden_token_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # idx: (B, T)
        for _ in range(max_new_tokens):
            # Bigram no usa contexto largo, pero mantenemos la misma API
            logits, _ = self(idx[:, -1:])  # usa solo el último token como contexto
            logits = logits[:, -1, :] / max(temperature, 1e-6)
            if forbidden_token_ids is not None and forbidden_token_ids.numel() > 0:
                logits[:, forbidden_token_ids] = float('-inf')
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                thresh = v[:, -1].unsqueeze(-1)
                logits = torch.where(logits < thresh, torch.full_like(logits, float('-inf')), logits)
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)  # (B, 1)
            idx = torch.cat((idx, next_token), dim=1)
        return idx

