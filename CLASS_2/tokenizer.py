from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List
import json


UNKNOWN_TOKEN = "�"  # usado como fallback si un carácter no está en el vocabulario


@dataclass
class CharTokenizer:
    """Tokenizador a nivel de caracteres con persistencia sencilla.

    - Construye vocabulario a partir de texto crudo
    - encode: str -> List[int]
    - decode: List[int] -> str
    - save/load: guardar y cargar el vocabulario (orden de itos)
    """

    stoi: Dict[str, int]
    itos: List[str]

    @classmethod
    def from_text(cls, text: str) -> "CharTokenizer":
        unique_chars = sorted(list(set(text)))
        if UNKNOWN_TOKEN not in unique_chars:
            unique_chars.append(UNKNOWN_TOKEN)
        itos = unique_chars
        stoi = {ch: i for i, ch in enumerate(itos)}
        return cls(stoi=stoi, itos=itos)

    @property
    def vocab_size(self) -> int:
        return len(self.itos)

    def encode(self, s: str) -> List[int]:
        unk_id = self.stoi.get(UNKNOWN_TOKEN, None)
        if unk_id is None:
            # Fallback robusto: si por alguna razón no está, añadimos al vuelo
            self.itos.append(UNKNOWN_TOKEN)
            self.stoi[UNKNOWN_TOKEN] = len(self.itos) - 1
            unk_id = self.stoi[UNKNOWN_TOKEN]
        return [self.stoi.get(ch, unk_id) for ch in s]

    def decode(self, ids: List[int]) -> str:
        result_chars: List[str] = []
        for idx in ids:
            if 0 <= idx < len(self.itos):
                result_chars.append(self.itos[idx])
            else:
                result_chars.append(UNKNOWN_TOKEN)
        return "".join(result_chars)

    def to_dict(self) -> Dict[str, List[str]]:
        return {"itos": self.itos}

    @classmethod
    def from_dict(cls, data: Dict[str, List[str]]) -> "CharTokenizer":
        itos = data["itos"]
        stoi = {ch: i for i, ch in enumerate(itos)}
        return cls(stoi=stoi, itos=itos)

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: str) -> "CharTokenizer":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)

