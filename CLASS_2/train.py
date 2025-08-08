from __future__ import annotations

import argparse
import math
import os
import time
from dataclasses import asdict
from typing import Dict, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from tokenizer import CharTokenizer
from bigram_model import BigramLanguageModel, BigramConfig
from transformer_model import MiniGPT, GPTConfig
from config import TrainConfig, auto_device


def set_seed(seed: int = 1337) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class LMSequenceDataset(Dataset):
    def __init__(self, data: torch.Tensor, block_size: int):
        super().__init__()
        self.data = data
        self.block_size = block_size

    def __len__(self) -> int:
        # Evita longitudes negativas cuando el split es más pequeño que block_size
        return max(0, int(self.data.size(0) - self.block_size))

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.data[idx : idx + self.block_size]
        y = self.data[idx + 1 : idx + 1 + self.block_size]
        return x, y


def build_dataloaders(all_ids: torch.Tensor, block_size: int, batch_size: int) -> Tuple[DataLoader, DataLoader]:
    n = int(0.9 * all_ids.numel())
    train_ids = all_ids[:n]
    val_ids = all_ids[n:]
    train_ds = LMSequenceDataset(train_ids, block_size)
    val_ds = LMSequenceDataset(val_ids, block_size)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, drop_last=True)
    return train_loader, val_loader


@torch.no_grad()
def estimate_loss(model, loader, device: torch.device) -> float:
    model.eval()
    losses = []
    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)
        _, loss = model(xb, yb)
        losses.append(loss.item())
    model.train()
    return float(sum(losses) / max(len(losses), 1))


def save_checkpoint(
    ckpt_path: str,
    model_type: str,
    model_state: Dict,
    tokenizer: CharTokenizer,
    train_cfg: TrainConfig,
    model_cfg: Dict,
    step: int,
    metrics: Dict,
) -> None:
    payload = {
        "model_type": model_type,
        "model_state_dict": model_state,
        "tokenizer": tokenizer.to_dict(),
        "train_config": asdict(train_cfg),
        "model_config": model_cfg,
        "step": step,
        "metrics": metrics,
        "vocab_size": tokenizer.vocab_size,
    }
    os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
    torch.save(payload, ckpt_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="bigram", choices=["bigram", "gpt"])
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--block-size", type=int, default=128)
    parser.add_argument("--max-steps", type=int, default=2000)
    parser.add_argument("--eval-interval", type=int, default=200)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--device", type=str, default=None)
    # gpt
    parser.add_argument("--n-embd", type=int, default=128)
    parser.add_argument("--n-head", type=int, default=4)
    parser.add_argument("--n-layer", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    args = parser.parse_args()

    device = auto_device(args.device)
    set_seed(1337)

    with open(args.data, "r", encoding="utf-8") as f:
        text = f.read()

    tokenizer = CharTokenizer.from_text(text)
    all_ids = torch.tensor(tokenizer.encode(text), dtype=torch.long)

    train_loader, val_loader = build_dataloaders(all_ids, block_size=args.block_size, batch_size=args.batch_size)

    if args.model == "bigram":
        model_type = "bigram"
        model_cfg = {"vocab_size": tokenizer.vocab_size}
        model = BigramLanguageModel(BigramConfig(vocab_size=tokenizer.vocab_size))
    else:
        model_type = "gpt"
        model_cfg = {
            "vocab_size": tokenizer.vocab_size,
            "block_size": args.block_size,
            "n_embd": args.n_embd,
            "n_head": args.n_head,
            "n_layer": args.n_layer,
            "dropout": args.dropout,
        }
        model = MiniGPT(GPTConfig(**model_cfg))

    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    print(f"Training {model_type} on {device} | vocab_size={tokenizer.vocab_size} | data_len={all_ids.numel()}")
    best_val = math.inf
    start_time = time.time()

    step = 0
    while step < args.max_steps:
        for xb, yb in train_loader:
            step += 1
            xb = xb.to(device)
            yb = yb.to(device)
            _, loss = model(xb, yb)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            if step % args.eval_interval == 0 or step == 1:
                train_loss = estimate_loss(model, train_loader, device)
                val_loss = estimate_loss(model, val_loader, device)
                elapsed = time.time() - start_time
                print(f"step {step:5d} | train {train_loss:.4f} | val {val_loss:.4f} | {elapsed:.1f}s")
                best_val = min(best_val, val_loss)

            if step >= args.max_steps:
                break

    ts = time.strftime("%Y%m%d_%H%M%S")
    ckpt_name = f"{model_type}_bs{args.block_size}_steps{args.max_steps}_{ts}.pt"
    ckpt_path = os.path.join(os.path.dirname(__file__), "checkpoints", ckpt_name)
    save_checkpoint(
        ckpt_path=ckpt_path,
        model_type=model_type,
        model_state=model.state_dict(),
        tokenizer=tokenizer,
        train_cfg=TrainConfig(
            data_path=args.data,
            model=model_type,
            batch_size=args.batch_size,
            block_size=args.block_size,
            max_steps=args.max_steps,
            eval_interval=args.eval_interval,
            learning_rate=args.lr,
            device_str=str(device),
            n_embd=args.n_embd,
            n_head=args.n_head,
            n_layer=args.n_layer,
            dropout=args.dropout,
        ),
        model_cfg=model_cfg,
        step=step,
        metrics={"best_val": best_val},
    )
    print(f"Checkpoint guardado en: {ckpt_path}")


if __name__ == "__main__":
    main()

