"""
Minimal causal-LM trainer for DAPH ExFusion v3.1.

Correctness-first: CPU/CUDA/MPS, seed control, checkpointing, grad clip.
Not a distributed MoE trainer.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.optim import AdamW

from .config import DAPHConfigV3
from .model import DAPHHybridModelV3


@dataclass
class TrainConfig:
    steps: int = 50
    batch_size: int = 4
    seq_len: int = 32
    lr: float = 3e-4
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    warmup_steps: int = 5
    log_every: int = 10
    seed: int = 42
    device: str = "cpu"
    # fixed effort for all steps, or "sample" to draw from effort_probs
    effort_mode: str = "disabled"
    # used when effort_mode == "sample": probabilities for E0..E3
    effort_probs: tuple = (0.2, 0.2, 0.3, 0.3)
    output_dir: str = "runs/train_smoke"
    # freeze parameters whose name contains any of these substrings
    freeze_name_contains: tuple = ()


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def synthetic_batches(
    vocab_size: int,
    batch_size: int,
    seq_len: int,
    n_batches: int,
    device: torch.device,
) -> Iterator[Tuple[Tensor, Tensor]]:
    """Random token streams for smoke training (no external dataset required)."""
    for _ in range(n_batches):
        x = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
        # causal LM: predict next token
        yield x[:, :-1], x[:, 1:]


def sample_effort_mode(cfg: TrainConfig, generator: torch.Generator) -> str:
    """Pick fixed_0..fixed_3 according to effort_probs, or return cfg.effort_mode."""
    if cfg.effort_mode != "sample":
        return cfg.effort_mode
    probs = torch.tensor(list(cfg.effort_probs), dtype=torch.float)
    probs = probs / probs.sum()
    idx = int(torch.multinomial(probs, 1, generator=generator).item())
    return f"fixed_{idx}"


def apply_freeze(model: torch.nn.Module, name_substrings: tuple) -> int:
    n = 0
    if not name_substrings:
        return 0
    for name, p in model.named_parameters():
        if any(s in name for s in name_substrings):
            p.requires_grad = False
            n += 1
    return n


def lr_at(step: int, cfg: TrainConfig) -> float:
    if step < cfg.warmup_steps:
        return cfg.lr * float(step + 1) / float(max(cfg.warmup_steps, 1))
    return cfg.lr


def train_smoke(
    model_cfg: Optional[DAPHConfigV3] = None,
    train_cfg: Optional[TrainConfig] = None,
) -> Dict[str, Any]:
    """
    Causal-LM training smoke / multi-effort curriculum on synthetic data.
    When effort_mode="sample", each step draws E0–E3 from effort_probs.
    """
    train_cfg = train_cfg or TrainConfig()
    model_cfg = model_cfg or DAPHConfigV3(
        hidden_size=64,
        latent_size=32,
        num_layers=2,
        num_attention_heads=4,
        state_size=8,
        num_recurrent_per_block=1,
        num_routed_experts=4,
        top_k_experts=2,
        vocab_size=128,
        enable_channel_gates=True,
        use_attn_res=False,
        use_load_balancing=False,
        use_quantile_balancing=False,
        dropout=0.0,
    )
    set_seed(train_cfg.seed)
    device = torch.device(train_cfg.device)

    model = DAPHHybridModelV3(model_cfg).to(device)
    frozen = apply_freeze(model, train_cfg.freeze_name_contains)
    model.train()
    opt = AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=train_cfg.lr,
        weight_decay=train_cfg.weight_decay,
    )
    gen = torch.Generator(device="cpu")
    gen.manual_seed(train_cfg.seed)
    effort_hist: Dict[str, int] = {}

    out_dir = Path(train_cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = out_dir / "metrics.jsonl"
    history: List[Dict[str, Any]] = []
    t0 = time.time()

    batches = synthetic_batches(
        model_cfg.vocab_size,
        train_cfg.batch_size,
        train_cfg.seq_len,
        train_cfg.steps,
        device,
    )

    for step, (x, y) in enumerate(batches):
        for g in opt.param_groups:
            g["lr"] = lr_at(step, train_cfg)
        opt.zero_grad(set_to_none=True)
        emode = sample_effort_mode(train_cfg, gen)
        effort_hist[emode] = effort_hist.get(emode, 0) + 1
        out = model(x, effort_mode=emode)
        logits = out["logits"] if isinstance(out, dict) else out
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            y.reshape(-1),
        )
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad],
            train_cfg.grad_clip,
        )
        opt.step()
        row = {
            "step": step,
            "loss": float(loss.item()),
            "lr": lr_at(step, train_cfg),
            "effort_mode": emode,
            "grad_norm": float(grad_norm) if not isinstance(grad_norm, float) else grad_norm,
            "ppl": float(math.exp(min(loss.item(), 20))),
        }
        history.append(row)
        with metrics_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")
        if step % train_cfg.log_every == 0:
            print(f"step {step:4d}  loss={row['loss']:.4f}  effort={emode}  ppl={row['ppl']:.2f}")

    ckpt = {
        "model": model.state_dict(),
        "model_config": {k: getattr(model_cfg, k) for k in model_cfg.__dataclass_fields__},
        "train_config": asdict(train_cfg),
        "effort_hist": dict(effort_hist),
        "frozen_params": frozen,
        "final_loss": history[-1]["loss"] if history else None,
        "steps": len(history),
        "wall_time_s": time.time() - t0,
        "param_count": sum(p.numel() for p in model.parameters()),
    }
    torch.save(ckpt, out_dir / "checkpoint.pt")
    (out_dir / "summary.json").write_text(
        json.dumps(
            {
                "effort_hist": ckpt["effort_hist"],
                "frozen_params": frozen,
                "final_loss": ckpt["final_loss"],
                "steps": ckpt["steps"],
                "wall_time_s": ckpt["wall_time_s"],
                "param_count": ckpt["param_count"],
                "device": train_cfg.device,
                "seed": train_cfg.seed,
            },
            indent=2,
        )
    )
    print(f"checkpoint → {out_dir / 'checkpoint.pt'}")
    return ckpt



if __name__ == "__main__":
    train_smoke()
