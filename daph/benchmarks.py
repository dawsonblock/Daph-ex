"""
Lightweight benchmarks for DAPH / ExFusion v3.

Measures forward-pass latency and throughput for HybridBlock and full model
under SSM vs KDA recurrent backends. CPU-only by default; pass device="cuda"
when available.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, List, Optional

import torch

from .config import DAPHConfigV3
from .hybrid_block import HybridBlock
from .model import DAPHHybridModelV3


@dataclass
class BenchResult:
    name: str
    batch_size: int
    seq_len: int
    num_runs: int
    mean_ms: float
    std_ms: float
    tokens_per_s: float
    device: str

    def as_dict(self) -> Dict[str, object]:
        return {
            "name": self.name,
            "batch_size": self.batch_size,
            "seq_len": self.seq_len,
            "num_runs": self.num_runs,
            "mean_ms": round(self.mean_ms, 3),
            "std_ms": round(self.std_ms, 3),
            "tokens_per_s": round(self.tokens_per_s, 1),
            "device": self.device,
        }


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


def _time_fn(fn, num_runs: int, warmup: int, device: torch.device) -> BenchResult:
    for _ in range(warmup):
        fn()
    _sync(device)

    times: List[float] = []
    for _ in range(num_runs):
        _sync(device)
        t0 = time.perf_counter()
        fn()
        _sync(device)
        times.append((time.perf_counter() - t0) * 1000.0)

    mean = sum(times) / len(times)
    var = sum((t - mean) ** 2 for t in times) / max(len(times) - 1, 1)
    std = var ** 0.5
    return mean, std


def benchmark_hybrid_block(
    recurrent_type: str = "ssm",
    hidden_size: int = 128,
    batch_size: int = 4,
    seq_len: int = 64,
    num_runs: int = 20,
    warmup: int = 5,
    device: str = "cpu",
) -> BenchResult:
    device_t = torch.device(device)
    cfg = DAPHConfigV3(
        hidden_size=hidden_size,
        latent_size=max(32, hidden_size // 2),
        num_attention_heads=max(1, hidden_size // 32),
        state_size=16,
        num_recurrent_per_block=2,
        num_routed_experts=4,
        top_k_experts=2,
        enable_channel_gates=True,
        recurrent_type=recurrent_type,
        use_quantile_balancing=False,
        dropout=0.0,
    )
    block = HybridBlock(cfg).to(device_t).eval()
    x = torch.randn(batch_size, seq_len, hidden_size, device=device_t)

    def run():
        with torch.no_grad():
            block(x)

    mean, std = _time_fn(run, num_runs, warmup, device_t)
    tokens = batch_size * seq_len
    tps = tokens / (mean / 1000.0) if mean > 0 else 0.0
    return BenchResult(
        name=f"HybridBlock/{recurrent_type}",
        batch_size=batch_size,
        seq_len=seq_len,
        num_runs=num_runs,
        mean_ms=mean,
        std_ms=std,
        tokens_per_s=tps,
        device=device,
    )


def benchmark_model(
    recurrent_type: str = "ssm",
    hidden_size: int = 128,
    num_layers: int = 4,
    batch_size: int = 2,
    seq_len: int = 32,
    num_runs: int = 15,
    warmup: int = 3,
    device: str = "cpu",
) -> BenchResult:
    device_t = torch.device(device)
    cfg = DAPHConfigV3(
        hidden_size=hidden_size,
        latent_size=max(32, hidden_size // 2),
        num_attention_heads=max(1, hidden_size // 32),
        state_size=16,
        num_recurrent_per_block=2,
        num_routed_experts=4,
        top_k_experts=2,
        num_layers=num_layers,
        vocab_size=1000,
        enable_channel_gates=True,
        recurrent_type=recurrent_type,
        use_attn_res=True,
        use_quantile_balancing=False,
        dropout=0.0,
    )
    model = DAPHHybridModelV3(cfg).to(device_t).eval()
    ids = torch.randint(0, 1000, (batch_size, seq_len), device=device_t)

    def run():
        with torch.no_grad():
            model(ids)

    mean, std = _time_fn(run, num_runs, warmup, device_t)
    tokens = batch_size * seq_len
    tps = tokens / (mean / 1000.0) if mean > 0 else 0.0
    return BenchResult(
        name=f"Model/{recurrent_type}/L{num_layers}",
        batch_size=batch_size,
        seq_len=seq_len,
        num_runs=num_runs,
        mean_ms=mean,
        std_ms=std,
        tokens_per_s=tps,
        device=device,
    )


def run_quick_suite(device: str = "cpu") -> List[BenchResult]:
    """Small default suite comparing SSM vs KDA."""
    results: List[BenchResult] = []
    for rtype in ("ssm", "kda"):
        results.append(
            benchmark_hybrid_block(recurrent_type=rtype, device=device, num_runs=10, warmup=2)
        )
        results.append(
            benchmark_model(recurrent_type=rtype, device=device, num_runs=8, warmup=2)
        )
    return results


def print_suite(results: List[BenchResult]) -> None:
    print(f"{'name':<28} {'ms':>10} {'±':>8} {'tok/s':>12} {'device':>8}")
    print("-" * 70)
    for r in results:
        print(
            f"{r.name:<28} {r.mean_ms:>10.2f} {r.std_ms:>8.2f} "
            f"{r.tokens_per_s:>12.1f} {r.device:>8}"
        )


if __name__ == "__main__":
    print_suite(run_quick_suite("cpu"))
