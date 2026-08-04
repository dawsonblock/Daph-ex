"""
Counterfactual E0–E3 experience collection for effort-policy training.

For each task x under a frozen model checkpoint:
  run fixed_0..fixed_3 → (Q_e, C_e, U_e)
  store probe hidden_anchor + utilities for policy supervision.

Does NOT train the controller — only produces experience.

IMPORTANT: use projection_dim=None when collecting data to train the
runtime EffortController (expects full hidden_size). Projection is for
diagnostic offline datasets only until a matching runtime projection exists.

v3.2 hardening:
  - full state_dict digest (not first-param sample)
  - causal LM CE shift + pad masking in default quality proxy
  - projection metadata recorded
  - context manager restores model train/grad state
  - oracle analysis with bootstrap LCB + entropy collapse diagnostics
"""

from __future__ import annotations

import hashlib
import json
import math
import random
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional, Sequence, Tuple

import torch
from torch import Tensor

from .config import DAPHConfigV3
from .model import DAPHHybridModelV3
from .qwen_exfusion import QwenExFusionModel


QualityFn = Callable[[Dict[str, Any], Dict[str, Any]], Tuple[float, str]]
# quality_fn(output_dict, task) -> (quality in [0,1], status_str)


def _tensor_raw_bytes(t: Tensor) -> bytes:
    """Bit-exact bytes for any dtype including bfloat16."""
    # ``Tensor.view(dtype)`` cannot reinterpret a zero-dimensional tensor when
    # the element sizes differ.  ExFusion intentionally has scalar residual
    # scales, so normalize every tensor to a one-dimensional byte-addressable
    # layout before hashing it.
    t = t.detach().cpu().contiguous().reshape(-1)
    # view as uint8 avoids numpy bfloat16 limitation
    return bytes(t.view(torch.uint8).numpy())


def full_state_dict_digest(model: torch.nn.Module) -> str:
    """SHA256 over sorted state_dict: name || dtype || shape || raw bytes."""
    h = hashlib.sha256()
    sd = model.state_dict()
    for name in sorted(sd.keys()):
        ten = sd[name]
        h.update(name.encode())
        h.update(str(ten.dtype).encode())
        h.update(str(tuple(ten.shape)).encode())
        h.update(_tensor_raw_bytes(ten))
    return h.hexdigest()[:32]


def _digest_tensor(t: Tensor) -> str:
    x = t.detach().cpu().contiguous().numpy().tobytes()
    return hashlib.sha256(x).hexdigest()[:16]


def _digest_obj(obj: Any) -> str:
    s = json.dumps(obj, sort_keys=True, default=str)
    return hashlib.sha256(s.encode()).hexdigest()[:16]


@dataclass(frozen=True)
class EffortCounterfactual:
    task_id: str
    input_digest: str
    task_digest: str  # input + mask + labels + verifier_spec
    probe_hidden: List[float]
    quality: Tuple[float, float, float, float]
    compute: Tuple[float, float, float, float]  # normalized per-task
    raw_compute: Tuple[float, float, float, float]  # absolute FLOP proxy
    utility: Tuple[float, float, float, float]
    best_effort: int  # cost-aware tiebreak
    argmax_effort: int  # pure max utility
    verifier_status: Tuple[str, str, str, str]
    model_digest: str
    config_digest: str
    lambda_cost: float
    tie_epsilon: float
    projection_dim: Optional[int] = None
    projection_seed: Optional[int] = None
    projection_digest: Optional[str] = None
    generated_texts: Optional[Tuple[str, str, str, str]] = None
    compute_receipts: Optional[Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any], Dict[str, Any]]] = None
    task_family: Optional[str] = None
    template_id: Optional[str] = None
    difficulty_bucket: Optional[str] = None
    generator_version: Optional[str] = None
    probe_source: str = "unspecified"
    profile_digest: Optional[str] = None
    e2_e3_outcome: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def compute_utility(
    qualities: Sequence[float],
    costs: Sequence[float],
    lambda_cost: float = 0.15,
    tie_epsilon: float = 0.01,
) -> Tuple[Tuple[float, float, float, float], int, int]:
    """
    U_e = Q_e - λ C_e
    Returns (utilities, cost_aware_best, pure_argmax).
    """
    utils = [float(q) - lambda_cost * float(c) for q, c in zip(qualities, costs)]
    argmax = int(max(range(len(utils)), key=lambda e: utils[e]))
    best = 0
    best_u = utils[0]
    for e in range(1, len(utils)):
        if utils[e] > best_u + tie_epsilon:
            best_u = utils[e]
            best = e
        elif abs(utils[e] - best_u) <= tie_epsilon and costs[e] < costs[best]:
            best = e
            best_u = utils[e]
    return (utils[0], utils[1], utils[2], utils[3]), best, argmax


def soft_targets(
    utilities: Sequence[float],
    temperature: float = 0.1,
) -> List[float]:
    scaled = [u / max(temperature, 1e-8) for u in utilities]
    m = max(scaled)
    exps = [math.exp(s - m) for s in scaled]
    z = sum(exps) or 1.0
    return [e / z for e in exps]


def causal_ce_quality(
    logits: Tensor,
    labels: Tensor,
    pad_id: Optional[int] = None,
) -> Tuple[float, str]:
    """
    Proper causal LM quality proxy: logits[:, :-1] vs labels[:, 1:].
    Masks pad positions. Returns (exp(-nll) in [0,1], status).
    """
    if logits.dim() != 3:
        return 0.0, "UNVERIFIABLE"
    if labels.dim() == 1:
        labels = labels.unsqueeze(0)
    # shift
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    if shift_labels.size(1) == 0 or shift_logits.size(1) == 0:
        return 0.0, "UNVERIFIABLE"
    # align lengths
    T = min(shift_logits.size(1), shift_labels.size(1))
    shift_logits = shift_logits[:, :T]
    shift_labels = shift_labels[:, :T]
    vocab = shift_logits.size(-1)
    flat_logits = shift_logits.reshape(-1, vocab)
    flat_labels = shift_labels.reshape(-1)
    if pad_id is not None:
        mask = flat_labels != pad_id
        if mask.sum() == 0:
            return 0.0, "UNVERIFIABLE"
        flat_logits = flat_logits[mask]
        flat_labels = flat_labels[mask]
    nll = torch.nn.functional.cross_entropy(flat_logits, flat_labels, reduction="mean")
    q = float(torch.exp(-nll).clamp(0, 1).item())
    return q, "CE_PROXY_CAUSAL"


class CounterfactualCollector:
    """
    Execute E0–E3 for each task under a frozen model.

    Use as context manager to restore train/grad state:

        with CounterfactualCollector(model) as coll:
            recs = coll.collect_many(tasks)
    """

    def __init__(
        self,
        model: torch.nn.Module,
        lambda_cost: float = 0.15,
        tie_epsilon: float = 0.01,
        cost_mode: str = "flops",
        quality_fn: Optional[QualityFn] = None,
        projection_dim: Optional[int] = None,
        projection_seed: int = 0,
        pad_token_id: Optional[int] = None,
        freeze: bool = True,
        tokenizer: Optional[Any] = None,
    ) -> None:
        self.model = model
        self.lambda_cost = lambda_cost
        self.tie_epsilon = tie_epsilon
        self.cost_mode = cost_mode
        self.quality_fn = quality_fn
        self.projection_dim = projection_dim
        self.projection_seed = projection_seed
        self.pad_token_id = pad_token_id
        self.tokenizer = tokenizer
        self._freeze = freeze
        self._proj: Optional[Tensor] = None
        self._proj_digest: Optional[str] = None

        # save caller state
        self._was_training = model.training
        self._grad_flags = [p.requires_grad for p in model.parameters()]

        if freeze:
            model.eval()
            for p in model.parameters():
                p.requires_grad_(False)

        cfg = getattr(model, "config", None)
        if cfg is not None and hasattr(cfg, "__dataclass_fields__"):
            config_obj = {k: getattr(cfg, k) for k in cfg.__dataclass_fields__}
        elif isinstance(model, QwenExFusionModel):
            config_obj = {
                "architecture": "QwenExFusionModel", "hidden_size": model.hidden_size,
                "num_layers": len(model.layers), "depth_fractions": model.depth_fractions,
                "default_e3_steps": model.default_e3_steps,
                "effort_probe_layer_count": model.effort_probe_layer_count,
                "e3_config": asdict(model.e3_config),
                "e3_region": asdict(model.e3_region),
            }
        else:
            config_obj = {"architecture": type(model).__name__}
        self.config_digest = _digest_obj(config_obj)
        self.model_digest = full_state_dict_digest(model)

    def restore(self) -> None:
        """Restore original training mode and requires_grad flags."""
        for p, flag in zip(self.model.parameters(), self._grad_flags):
            p.requires_grad_(flag)
        if self._was_training:
            self.model.train()
        else:
            self.model.eval()

    def close(self) -> None:
        self.restore()

    def __enter__(self) -> "CounterfactualCollector":
        return self

    def __exit__(self, *args) -> None:
        self.restore()

    def _project(self, h: Tensor) -> List[float]:
        h = h.detach().float().cpu().view(-1)
        if self.projection_dim is None or self.projection_dim >= h.numel():
            self._proj_digest = None
            return h.tolist()
        if self._proj is None:
            g = torch.Generator().manual_seed(self.projection_seed)
            self._proj = torch.randn(h.numel(), self.projection_dim, generator=g)
            self._proj = self._proj / self._proj.norm(dim=0, keepdim=True).clamp_min(1e-6)
            self._proj_digest = _digest_tensor(self._proj)
        return (h @ self._proj).tolist()

    def _default_quality(self, out: Dict[str, Any], task: Dict[str, Any]) -> Tuple[float, str]:
        if "labels" in task and task["labels"] is not None:
            logits = out["logits"]
            labels = task["labels"]
            if not isinstance(labels, Tensor):
                labels = torch.tensor(labels, device=logits.device)
            pad = task.get("pad_token_id", self.pad_token_id)
            return causal_ce_quality(logits, labels, pad_id=pad)
        return 0.0, "UNVERIFIABLE"

    @torch.no_grad()
    def collect_one(self, task: Dict[str, Any]) -> EffortCounterfactual:
        task_id = str(task["task_id"])
        ids = task["input_ids"]
        if not isinstance(ids, Tensor):
            ids = torch.tensor(ids, dtype=torch.long)
        if ids.dim() == 1:
            ids = ids.unsqueeze(0)
        if ids.size(0) != 1:
            raise ValueError(
                f"collect_one expects batch size 1; got B={ids.size(0)}. "
                "Call collect_many with one sequence per task, or use a batched helper."
            )
        mask = task.get("attention_mask")
        if mask is not None and not isinstance(mask, Tensor):
            mask = torch.tensor(mask)
        if mask is not None and mask.dim() == 1:
            mask = mask.unsqueeze(0)

        # task digest includes correctness-defining fields
        td_parts = [_digest_tensor(ids)]
        if mask is not None:
            td_parts.append(_digest_tensor(mask))
        if "labels" in task and task["labels"] is not None:
            lab = task["labels"]
            if not isinstance(lab, Tensor):
                lab = torch.tensor(lab)
            td_parts.append(_digest_tensor(lab))
        if "expected" in task:
            td_parts.append(_digest_obj(task["expected"]))
        if "verifier_spec" in task:
            td_parts.append(_digest_obj(task["verifier_spec"]))
        task_digest = hashlib.sha256("".join(td_parts).encode()).hexdigest()[:16]

        if hasattr(self.model, "compute_effort_probe"):
            probe_input = ids if isinstance(self.model, QwenExFusionModel) else self.model.embed(ids)
            probe_result = self.model.compute_effort_probe(probe_input, mask)
            probe_h, _, decision = probe_result
            anchor = decision.hidden_anchor
            if anchor is None:
                from .effort import EffortController
                anchor = EffortController.pool_last_valid(probe_h, mask)
            probe_source = "internal_qwen_probe"
        else:
            from .effort import EffortController
            hidden = self.model.embed(ids)
            anchor = EffortController.pool_last_valid(hidden, mask)
            probe_source = "prompt_embedding_fallback"
        probe_vec = self._project(anchor[0])

        qualities: List[float] = []
        costs: List[float] = []
        statuses: List[str] = []
        raw_costs: List[float] = []
        receipts: List[Dict[str, Any]] = []
        outs = []
        for e in range(4):
            kwargs = (
                {"return_compute_receipt": True, "precomputed_probe": probe_result}
                if isinstance(self.model, QwenExFusionModel) else {}
            )
            out = self.model(ids, attention_mask=mask, effort_mode=f"fixed_{e}", **kwargs)
            outs.append(out)
            stats = out["compute_stats"]
            receipts.append(dict(stats))
            flops = float(
                stats.get("raw_compute_units")
                or stats.get("estimated_compute_units")
                # Backward-compatible read for legacy collectors only.
                or stats.get("estimated_flops")
                or 0.0
            )
            nominal = float((stats.get("per_sample_compute") or [stats.get("normalized_compute_cost", 0.3)])[0] if isinstance(stats.get("per_sample_compute") or [0.3], list) else stats.get("normalized_compute_cost", 0.3))
            raw_costs.append(flops if self.cost_mode == "flops" and flops > 0 else nominal)

        if isinstance(self.model, QwenExFusionModel) and not all(raw_costs[i] < raw_costs[i + 1] for i in range(3)):
            raise RuntimeError(f"Physical compute ordering violation: {raw_costs}")
        normalization = raw_costs[2] if isinstance(self.model, QwenExFusionModel) else max(raw_costs)
        e2_c = normalization if normalization > 0 else 1.0
        for e, out in enumerate(outs):
            if self.quality_fn is not None:
                q, status = self.quality_fn(out, task)
                q = float(q)
            else:
                q, status = self._default_quality(out, task)
            qualities.append(q)
            costs.append(raw_costs[e] / e2_c)
            statuses.append(status)

        utilities, best, argmax = compute_utility(
            qualities, costs, self.lambda_cost, self.tie_epsilon
        )
        e2_correct = statuses[2] == "CORRECT"
        e3_correct = statuses[3] == "CORRECT"
        outcome = (
            "RESCUE" if not e2_correct and e3_correct else
            "REGRESSION" if e2_correct and not e3_correct else
            "BOTH_CORRECT" if e2_correct and e3_correct else
            "BOTH_WRONG"
        ) if statuses[2] in ("CORRECT", "INCORRECT") and statuses[3] in ("CORRECT", "INCORRECT") else None

        return EffortCounterfactual(
            task_id=task_id,
            input_digest=_digest_tensor(ids),
            task_digest=task_digest,
            probe_hidden=probe_vec,
            quality=(qualities[0], qualities[1], qualities[2], qualities[3]),
            compute=(costs[0], costs[1], costs[2], costs[3]),
            raw_compute=(raw_costs[0], raw_costs[1], raw_costs[2], raw_costs[3]),
            utility=utilities,
            best_effort=best,
            argmax_effort=argmax,
            verifier_status=(statuses[0], statuses[1], statuses[2], statuses[3]),
            model_digest=self.model_digest,
            config_digest=self.config_digest,
            lambda_cost=self.lambda_cost,
            tie_epsilon=self.tie_epsilon,
            projection_dim=self.projection_dim,
            projection_seed=self.projection_seed if self.projection_dim else None,
            projection_digest=self._proj_digest,
            task_family=task.get("task_family"),
            template_id=task.get("template_id"),
            difficulty_bucket=task.get("difficulty_bucket"),
            generator_version=task.get("generator_version"),
            compute_receipts=(receipts[0], receipts[1], receipts[2], receipts[3]),
            probe_source=probe_source,
            profile_digest=getattr(getattr(self.model, "e3_region", None), "source_profile_digest", None),
            e2_e3_outcome=outcome,
        )


    @torch.no_grad()
    def collect_one_generate(
        self,
        task: Dict[str, Any],
        max_new_tokens: int = 16,
        eos_token_id: Optional[int] = None,
    ) -> EffortCounterfactual:
        """
        Counterfactuals via greedy generation + quality_fn/verifier on outputs.
        Requires task batch size 1. quality_fn should score generated answers.
        """
        task_id = str(task["task_id"])
        ids = task["input_ids"]
        if not isinstance(ids, Tensor):
            ids = torch.tensor(ids, dtype=torch.long)
        if ids.dim() == 1:
            ids = ids.unsqueeze(0)
        if ids.size(0) != 1:
            raise ValueError("collect_one_generate expects batch size 1")
        mask = task.get("attention_mask")
        if mask is not None and not isinstance(mask, Tensor):
            mask = torch.tensor(mask)
        if mask is not None and mask.dim() == 1:
            mask = mask.unsqueeze(0)

        td_parts = [_digest_tensor(ids)]
        if mask is not None:
            td_parts.append(_digest_tensor(mask))
        if "expected" in task:
            td_parts.append(_digest_obj(task["expected"]))
        if "verifier_spec" in task:
            td_parts.append(_digest_obj(task["verifier_spec"]))
        task_digest = hashlib.sha256("".join(td_parts).encode()).hexdigest()[:16]

        if hasattr(self.model, "compute_effort_probe"):
            probe_input = ids if isinstance(self.model, QwenExFusionModel) else self.model.embed(ids)
            probe_result = self.model.compute_effort_probe(probe_input, mask)
            probe_h, _, decision = probe_result
            anchor = decision.hidden_anchor
            if anchor is None:
                from .effort import EffortController
                anchor = EffortController.pool_last_valid(probe_h, mask)
            probe_source = "internal_qwen_probe"
        else:
            from .effort import EffortController
            hidden = self.model.embed(ids)
            anchor = EffortController.pool_last_valid(hidden, mask)
            probe_source = "prompt_embedding_fallback"
        probe_vec = self._project(anchor[0])

        qualities, costs, statuses, raw_costs, texts = [], [], [], [], []
        for e in range(4):
            gen = self.model.generate(
                ids,
                attention_mask=mask,
                max_new_tokens=max_new_tokens,
                effort_mode=f"fixed_{e}",
                eos_token_id=eos_token_id,
                tokenizer=self.tokenizer,
            )
            flops = float(
                gen["compute_stats"].get("raw_compute_units")
                or gen["compute_stats"].get("estimated_compute_units")
                or gen["compute_stats"].get("estimated_flops")
                or 0.0
            )
            raw_costs.append(flops if flops > 0 else float(e + 1))
            if self.quality_fn is not None:
                q, status = self.quality_fn(gen, task)
            else:
                q, status = 0.0, "UNVERIFIABLE"
            qualities.append(float(q))
            statuses.append(status)
            gt = gen.get("generated_text")
            if isinstance(gt, list):
                gt = gt[0] if gt else ""
            texts.append(str(gt) if gt is not None else "")

        if isinstance(self.model, QwenExFusionModel) and not all(raw_costs[i] < raw_costs[i + 1] for i in range(3)):
            raise RuntimeError(f"Physical compute ordering violation: {raw_costs}")
        norm_c = raw_costs[2] if isinstance(self.model, QwenExFusionModel) else max(raw_costs)
        norm_c = norm_c if norm_c > 0 else 1.0
        costs = [c / norm_c for c in raw_costs]
        utilities, best, argmax = compute_utility(
            qualities, costs, self.lambda_cost, self.tie_epsilon
        )
        e2_correct = statuses[2] == "CORRECT"
        e3_correct = statuses[3] == "CORRECT"
        outcome = (
            "RESCUE" if not e2_correct and e3_correct else
            "REGRESSION" if e2_correct and not e3_correct else
            "BOTH_CORRECT" if e2_correct and e3_correct else
            "BOTH_WRONG"
        ) if statuses[2] in ("CORRECT", "INCORRECT") and statuses[3] in ("CORRECT", "INCORRECT") else None
        return EffortCounterfactual(
            task_id=task_id,
            input_digest=_digest_tensor(ids),
            task_digest=task_digest,
            probe_hidden=probe_vec,
            quality=(qualities[0], qualities[1], qualities[2], qualities[3]),
            compute=(costs[0], costs[1], costs[2], costs[3]),
            raw_compute=(raw_costs[0], raw_costs[1], raw_costs[2], raw_costs[3]),
            utility=utilities,
            best_effort=best,
            argmax_effort=argmax,
            verifier_status=(statuses[0], statuses[1], statuses[2], statuses[3]),
            model_digest=self.model_digest,
            config_digest=self.config_digest,
            lambda_cost=self.lambda_cost,
            tie_epsilon=self.tie_epsilon,
            projection_dim=self.projection_dim,
            projection_seed=self.projection_seed if self.projection_dim else None,
            projection_digest=self._proj_digest,
            task_family=task.get("task_family"),
            template_id=task.get("template_id"),
            difficulty_bucket=task.get("difficulty_bucket"),
            generator_version=task.get("generator_version"),
            generated_texts=(texts[0], texts[1], texts[2], texts[3]),
            probe_source=probe_source,
            profile_digest=getattr(getattr(self.model, "e3_region", None), "source_profile_digest", None),
            e2_e3_outcome=outcome,
        )


    def collect_many(
        self,
        tasks: Sequence[Dict[str, Any]],
        out_path: Optional[str] = None,
    ) -> List[EffortCounterfactual]:
        records: List[EffortCounterfactual] = []
        for task in tasks:
            records.append(self.collect_one(task))
        if out_path:
            path = Path(out_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("w", encoding="utf-8") as f:
                for r in records:
                    f.write(json.dumps(r.to_dict()) + "\n")
        return records


def oracle_analysis(
    records: Sequence[EffortCounterfactual],
    n_bootstrap: int = 500,
    seed: int = 0,
    min_effect: float = 0.01,
) -> Dict[str, Any]:
    """
    Oracle opportunity analysis with bootstrap LCB and entropy collapse check.
    """
    if not records:
        return {"n": 0}

    n = len(records)
    best_counts = [0, 0, 0, 0]
    argmax_counts = [0, 0, 0, 0]
    sum_q = [0.0, 0.0, 0.0, 0.0]
    sum_c = [0.0, 0.0, 0.0, 0.0]
    sum_u = [0.0, 0.0, 0.0, 0.0]
    oracle_us = []
    best_fixed_us = []

    for r in records:
        best_counts[r.best_effort] += 1
        argmax_counts[r.argmax_effort] += 1
        for e in range(4):
            sum_q[e] += r.quality[e]
            sum_c[e] += r.compute[e]
            sum_u[e] += r.utility[e]
        oracle_us.append(r.utility[r.best_effort])

    mean_u = [s / n for s in sum_u]
    best_fixed = int(max(range(4), key=lambda e: mean_u[e]))
    u_best_fixed = mean_u[best_fixed]
    u_oracle = sum(oracle_us) / n
    gap = u_oracle - u_best_fixed

    # bootstrap gap LCB
    rng = random.Random(seed)
    gaps = []
    for _ in range(n_bootstrap):
        sample = [records[rng.randrange(n)] for _ in range(n)]
        s_u = [0.0, 0.0, 0.0, 0.0]
        o_sum = 0.0
        for r in sample:
            for e in range(4):
                s_u[e] += r.utility[e]
            o_sum += r.utility[r.best_effort]
        mu = [x / n for x in s_u]
        bf = max(range(4), key=lambda e: mu[e])
        gaps.append(o_sum / n - mu[bf])
    gaps.sort()
    lcb_idx = max(0, int(0.05 * len(gaps)) - 1)
    gap_lcb95 = gaps[lcb_idx]

    # entropy of best_effort distribution
    probs = [c / n for c in best_counts]
    entropy = -sum(p * math.log(p + 1e-12) for p in probs)
    max_frac = max(probs)

    has_opp = gap_lcb95 > 0 and gap >= min_effect and max_frac < 0.95

    return {
        "n": n,
        "best_effort_hist": {f"E{e}": best_counts[e] / n for e in range(4)},
        "argmax_effort_hist": {f"E{e}": argmax_counts[e] / n for e in range(4)},
        "mean_quality": {f"E{e}": sum_q[e] / n for e in range(4)},
        "mean_compute": {f"E{e}": sum_c[e] / n for e in range(4)},
        "mean_utility": {f"E{e}": mean_u[e] for e in range(4)},
        "best_fixed_level": best_fixed,
        "U_best_fixed": u_best_fixed,
        "U_oracle": u_oracle,
        "oracle_gap": gap,
        "oracle_gap_lcb95": gap_lcb95,
        "oracle_effort_entropy": entropy,
        "max_effort_fraction": max_frac,
        "has_routing_opportunity": has_opp,
        "min_effect": min_effect,
    }


def qualify_effort_hierarchy(
    records: Sequence[EffortCounterfactual],
    *,
    min_e0_quality_ratio: float = 0.25,
    min_e3_quality_delta: float = 0.0,
    bootstrap_samples: int = 2000,
    confidence: float = 0.95,
    lambda_compute: float = 1.0,
    seed: int = 42,
) -> Dict[str, Any]:
    """Qualify physical modes before any oracle/policy training is allowed."""
    if not records:
        return {"qualified": False, "reason": "NO_RECORDS"}
    physical = all(all(r.raw_compute[e] < r.raw_compute[e + 1] for e in range(3)) for r in records)
    n = len(records)
    quality = [sum(r.quality[e] for r in records) / n for e in range(4)]
    compute = [sum(r.compute[e] for r in records) / n for e in range(4)]
    dominated: List[int] = []
    for e in range(4):
        for other in range(4):
            if other == e:
                continue
            if compute[other] <= compute[e] and quality[other] >= quality[e] and (
                compute[other] < compute[e] or quality[other] > quality[e]
            ):
                dominated.append(e)
                break
    e0_useful = quality[0] >= min_e0_quality_ratio * max(quality[2], 1e-12)
    from .e3_metrics import E3QualificationConfig, qualify_e3_pairs
    verified_pairs = []
    for record in records:
        if record.verifier_status[2] in ("CORRECT", "INCORRECT") and record.verifier_status[3] in ("CORRECT", "INCORRECT"):
            verified_pairs.append({
                "task_id": record.task_id,
                "e2_correct": record.verifier_status[2] == "CORRECT",
                "e3_correct": record.verifier_status[3] == "CORRECT",
                "quality_e2": record.quality[2],
                "quality_e3": record.quality[3],
                "compute_e2": record.compute[2],
                "compute_e3": record.compute[3],
                "task_family": record.task_family or "unspecified",
                "template_id": record.template_id or record.task_family or record.task_id,
                "difficulty": record.difficulty_bucket or "unspecified",
                "difficulty_bucket": record.difficulty_bucket,
            })
    e3_report = qualify_e3_pairs(verified_pairs, E3QualificationConfig(
        lambda_compute=lambda_compute, bootstrap_samples=bootstrap_samples,
        confidence=confidence, min_quality_delta=min_e3_quality_delta, seed=seed,
    )) if verified_pairs else {
        "qualified": False, "policy_training_allowed": False,
        "reason": "NO_PAIRED_VERIFIED_E2_E3_OUTCOMES",
    }
    e3_improves = bool(e3_report["quality_gate"]["passed"])
    e3_cost_effective = bool(e3_report["utility_gate"]["passed"])
    qualified = physical and e0_useful and e3_improves and e3_cost_effective and not dominated
    return {
        "qualified": qualified,
        "physical_compute_ordering": physical,
        "mean_quality": {f"E{e}": quality[e] for e in range(4)},
        "mean_compute": {f"E{e}": compute[e] for e in range(4)},
        "regret_vs_e2": {"E0": quality[2] - quality[0], "E1": quality[2] - quality[1]},
        "e3_quality_delta_vs_e2": quality[3] - quality[2],
        "e3_compute_delta_vs_e2": compute[3] - compute[2],
        "dominated_efforts": [f"E{e}" for e in sorted(set(dominated))],
        "e0_useful": e0_useful,
        "e3_improves": e3_improves,
        "e3_cost_effective": e3_cost_effective,
        "e3_verified_qualification": e3_report,
        "policy_training_allowed": False,
        "requires_oracle_opportunity_gate": True,
        "recommended_action": (
            "DROP_OR_IMPROVE_DOMINATED_ARMS" if dominated else
            "IMPROVE_E3_BEFORE_POLICY" if not e3_improves else
            "IMPROVE_E0" if not e0_useful else
            "PROCEED_TO_ORACLE_GATE"
        ),
    }
