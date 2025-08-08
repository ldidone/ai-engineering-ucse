from __future__ import annotations

import argparse
import os
from typing import Optional

import torch

from tokenizer import CharTokenizer
from bigram_model import BigramLanguageModel, BigramConfig
from transformer_model import MiniGPT, GPTConfig
from config import auto_device


def load_checkpoint(ckpt_path: str):
    payload = torch.load(ckpt_path, map_location="cpu")
    return payload


def build_model(model_type: str, model_cfg: dict):
    if model_type == "bigram":
        return BigramLanguageModel(BigramConfig(vocab_size=model_cfg["vocab_size"]))
    elif model_type == "gpt":
        return MiniGPT(GPTConfig(**model_cfg))
    else:
        raise ValueError(f"Modelo no soportado: {model_type}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--prompt", type=str, default="")
    parser.add_argument("--max-new-tokens", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--ban-unknown", action="store_true", help="Evita muestrear el token desconocido si existe en el vocabulario")
    args = parser.parse_args()

    device = auto_device(args.device)
    payload = load_checkpoint(args.ckpt)

    tokenizer = CharTokenizer.from_dict(payload["tokenizer"]) if isinstance(payload["tokenizer"], dict) else payload["tokenizer"]
    model_type = payload["model_type"]
    model_cfg = payload["model_config"]
    model = build_model(model_type, model_cfg)
    model.load_state_dict(payload["model_state_dict"])
    model.to(device)
    model.eval()

    if args.prompt:
        start_ids = tokenizer.encode(args.prompt)
        # Elimina tokens fuera de vocabulario (que fueron mapeados a UNKNOWN)
        if "�" in tokenizer.stoi:
            unk_id = tokenizer.stoi["�"]
            start_ids = [tid for tid in start_ids if tid != unk_id]
        # Si todo era OOV, usar salto de línea si existe o primer token
        if not start_ids:
            ch = "\n" if "\n" in tokenizer.stoi else tokenizer.itos[0]
            start_ids = tokenizer.encode(ch)
    else:
        # Si no hay prompt, comenzamos con salto de línea si existe, sino con id 0
        ch = "\n" if "\n" in tokenizer.stoi else tokenizer.itos[0]
        start_ids = tokenizer.encode(ch)

    idx = torch.tensor([start_ids], dtype=torch.long, device=device)
    # Construir máscara de tokens prohibidos si se solicita
    forbidden = None
    if args.ban_unknown and "�" in tokenizer.stoi:
        forbidden = torch.tensor([tokenizer.stoi["�"]], dtype=torch.long, device=device)
    out = model.generate(
        idx,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        block_size=model_cfg.get("block_size", None),
        forbidden_token_ids=forbidden,
    )
    text = tokenizer.decode(out[0].tolist())
    print(text)


if __name__ == "__main__":
    main()

