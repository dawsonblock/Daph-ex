"""
Real multi-effort adaptation trainer for ExFusion.

Supports:
  - JSONL text dataset ({"text": "..."} or {"input_ids": [...]})
  - optional HF tokenizer
  - effort_mode="sample" curriculum over E0–E3
  - freeze schedules (name substrings)
  - differential LR groups (pretrained vs new)
  - gradient accumulation, grad clip, checkpoint resume
  - per-effort loss tracking

This is the Stage 1–3 adaptation path after weight transplant.
"""

from __future__ import annotations

import json
import math
import time
import copy
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterator, List, Mapping, Optional, Sequence, Tuple, Union

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.optim import AdamW

from .config import DAPHConfigV3
from .model import DAPHHybridModelV3
from .qwen_exfusion import QwenExFusionModel
from .train import apply_freeze, lr_at, sample_effort_mode, set_seed


@dataclass(frozen=True)
class TrainingStageConfig:
    name: str
    steps: int
    train_parameter_groups: Tuple[str, ...] = ("new", "scales")
    freeze_parameter_groups: Tuple[str, ...] = ("imported",)
    lr_pretrained: float = 1e-5
    lr_new: float = 1e-4
    lr_scales: float = 1e-3
    effort_sampling: Tuple[float, float, float, float] = (0.3, 0.3, 0.2, 0.2)
    distill_e0: bool = True
    distill_e1: bool = True
    train_e3: bool = True
    teacher_mode: str = "fixed_2"
    e3_regression_guard_weight: float = 0.0


@dataclass
class RealTrainConfig:
    steps: int = 1000
    batch_size: int = 4
    seq_len: int = 256
    lr: float = 1e-4
    lr_pretrained: float = 1e-5  # lower LR for imported weights
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    warmup_steps: int = 50
    grad_accum: int = 1
    log_every: int = 20
    eval_every: int = 200
    seed: int = 42
    device: str = "cpu"
    effort_mode: str = "sample"
    effort_probs: Tuple[float, float, float, float] = (0.2, 0.2, 0.3, 0.3)
    # Optional deterministic micro-step schedule. When set, it takes precedence
    # over effort_mode/effort_probs and cycles for the duration of training.
    effort_schedule: Tuple[str, ...] = ()
    freeze_name_contains: Tuple[str, ...] = ()
    # parameter-name substrings treated as "pretrained" for lower LR
    pretrained_name_contains: Tuple[str, ...] = (
        "embed", "lm_head", "attn.q_proj", "attn.k_proj", "attn.v_proj",
        "attn.out_proj", "attn_norm", "final_norm", "moe.shared",
    )
    data_path: str = ""
    val_path: str = ""
    tokenizer_name: str = ""  # optional HF tokenizer id/path
    tokenizer_revision: str = ""  # optional immutable HF revision
    output_dir: str = "runs/adapt"
    resume: str = ""
    distillation_temperature: float = 2.0
    beta_e0: float = 1.0
    beta_e1: float = 1.0
    hidden_distillation_weight: float = 0.0
    stages: Tuple[TrainingStageConfig, ...] = ()
    retention_kl_threshold: Optional[float] = None
    save_periodic_checkpoints: bool = True
    save_final_checkpoint: bool = True
    save_model_artifact: bool = True
    fail_on_nonfinite: bool = True
    answer_only_loss: bool = False


def load_jsonl_texts(path: str) -> List[str]:
    texts: List[str] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if "text" in obj:
                texts.append(str(obj["text"]))
            elif "input" in obj:
                texts.append(str(obj["input"]))
            elif "prompt" in obj:
                texts.append(str(obj["prompt"]))
    if not texts:
        raise ValueError(f"No texts in {path}")
    return texts


def load_jsonl_training_records(
    path: str, *, answer_only_loss: bool = False,
) -> List[Union[str, Dict[str, str]]]:
    """Load ordinary text rows or explicit prompt/answer supervision rows."""
    if not answer_only_loss:
        return load_jsonl_texts(path)
    records: List[Union[str, Dict[str, str]]] = []
    with open(path, encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            prompt = obj.get("prompt")
            answer = obj.get("answer", obj.get("expected"))
            if prompt is None or not str(prompt) or answer is None:
                raise ValueError(
                    "Answer-only training requires prompt plus answer/expected; "
                    f"missing at {path}:{line_number}"
                )
            records.append({"prompt": str(prompt), "answer": str(answer)})
    if not records:
        raise ValueError(f"No answer-only training records in {path}")
    return records


class TextBatcher:
    def __init__(
        self,
        texts: Sequence[Union[str, Mapping[str, str]]],
        *,
        tokenizer: Any,
        seq_len: int,
        batch_size: int,
        device: torch.device,
        seed: int = 0,
        answer_only_loss: bool = False,
    ) -> None:
        self.texts = list(texts)
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.device = device
        self.rng = torch.Generator().manual_seed(seed)
        self.answer_only_loss = bool(answer_only_loss)
        self._i = 0

    def __iter__(self) -> "TextBatcher":
        return self

    def __next__(self) -> Tuple[Tensor, Tensor, Tensor]:
        batch_records: List[Union[str, Mapping[str, str]]] = []
        for _ in range(self.batch_size):
            if self._i >= len(self.texts):
                self._i = 0
                perm = torch.randperm(len(self.texts), generator=self.rng).tolist()
                self.texts = [self.texts[j] for j in perm]
            batch_records.append(self.texts[self._i])
            self._i += 1
        if self.answer_only_loss:
            return self._answer_only_batch(batch_records)
        batch_txt = [str(record) for record in batch_records]
        if self.tokenizer is None:
            ids_list = []
            for text in batch_txt:
                row = [ord(c) % 100 for c in text[: self.seq_len]]
                pad_len = self.seq_len - len(row)
                ids_list.append(row + [0] * pad_len)
            ids = torch.tensor(ids_list, dtype=torch.long, device=self.device)
            mask = (ids != 0).long()  # rough
            labels = ids.clone()
            labels[mask == 0] = -100
        else:
            if getattr(self.tokenizer, "pad_token", None) is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            enc = self.tokenizer(
                batch_txt,
                padding="max_length",
                truncation=True,
                max_length=self.seq_len,
                return_tensors="pt",
            )
            ids = enc["input_ids"].to(self.device)
            mask = enc["attention_mask"].to(self.device)
            labels = ids.clone()
            labels[mask == 0] = -100
        # training uses causal shift inside the loop
        return ids, labels, mask

    def _answer_only_batch(
        self, batch_records: Sequence[Union[str, Mapping[str, str]]],
    ) -> Tuple[Tensor, Tensor, Tensor]:
        rows: List[List[int]] = []
        label_rows: List[List[int]] = []
        masks: List[List[int]] = []
        pad_id = 0
        if self.tokenizer is not None:
            if getattr(self.tokenizer, "pad_token", None) is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            pad_id = int(getattr(self.tokenizer, "pad_token_id", 0) or 0)

        def encode(text: str) -> List[int]:
            if self.tokenizer is None:
                return [ord(char) % 100 for char in text]
            encoded = self.tokenizer(text, add_special_tokens=False)["input_ids"]
            if encoded and isinstance(encoded[0], list):
                encoded = encoded[0]
            return [int(token) for token in encoded]

        for record in batch_records:
            if not isinstance(record, Mapping) or "prompt" not in record or "answer" not in record:
                raise ValueError("Answer-only batches require prompt/answer mappings")
            prompt_ids = encode(str(record["prompt"]))
            answer_ids = encode(str(record["answer"]))
            if not answer_ids:
                raise ValueError("Answer-only training requires at least one answer token")
            answer_ids = answer_ids[: self.seq_len]
            prompt_budget = max(0, self.seq_len - len(answer_ids))
            prompt_ids = prompt_ids[-prompt_budget:] if prompt_budget else []
            combined = prompt_ids + answer_ids
            valid = len(combined)
            padding = self.seq_len - valid
            rows.append(combined + [pad_id] * padding)
            label_rows.append([-100] * len(prompt_ids) + answer_ids + [-100] * padding)
            masks.append([1] * valid + [0] * padding)
        return (
            torch.tensor(rows, dtype=torch.long, device=self.device),
            torch.tensor(label_rows, dtype=torch.long, device=self.device),
            torch.tensor(masks, dtype=torch.long, device=self.device),
        )


def _param_groups(model: torch.nn.Module, cfg: RealTrainConfig) -> List[Dict[str, Any]]:
    pre, neu = [], []
    provenance = getattr(model, "parameter_provenance", None)
    imported = set(provenance.imported_parameter_names) if provenance is not None else None
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if (imported is not None and n in imported) or (
            imported is None and any(s in n for s in cfg.pretrained_name_contains)
        ):
            pre.append(p)
        else:
            neu.append(p)
    groups = []
    if neu:
        groups.append({"params": neu, "lr": cfg.lr})
    if pre:
        groups.append({"params": pre, "lr": cfg.lr_pretrained})
    if not groups:
        groups.append({"params": [p for p in model.parameters() if p.requires_grad], "lr": cfg.lr})
    return groups


def apply_training_stage(model: torch.nn.Module, stage: TrainingStageConfig) -> Dict[str, List[str]]:
    """Apply explicit provenance groups; stage behavior never depends on name fragments."""
    provenance = getattr(model, "parameter_provenance", None)
    if provenance is None:
        raise ValueError("Explicit training stages require a model parameter_provenance manifest")
    groups = {
        "imported": set(provenance.imported_parameter_names),
        "new": set(provenance.new_parameter_names),
        "augmentation": set(provenance.augmentation_parameter_names),
        "scales": set(provenance.scale_parameter_names),
        "continuation": set(provenance.continuation_parameter_names),
        "e3_refinement": set(provenance.e3_refinement_parameter_names),
        "e3_scale": set(provenance.e3_scale_parameter_names),
        "e3_middle_layers": set(provenance.e3_middle_layer_parameter_names),
    }
    unknown = (set(stage.train_parameter_groups) | set(stage.freeze_parameter_groups)) - set(groups)
    if unknown:
        raise ValueError(f"Unknown parameter groups in stage {stage.name}: {sorted(unknown)}")
    train_names = set().union(*(groups[g] for g in stage.train_parameter_groups))
    freeze_names = set().union(*(groups[g] for g in stage.freeze_parameter_groups))
    train_names -= freeze_names
    trained, frozen = [], []
    for name, param in model.named_parameters():
        enabled = name in train_names
        param.requires_grad_(enabled)
        (trained if enabled else frozen).append(name)
    return {"trained": sorted(trained), "frozen": sorted(frozen)}


def _stage_at(stages: Sequence[TrainingStageConfig], step: int) -> Tuple[Optional[TrainingStageConfig], int]:
    cursor = 0
    for stage in stages:
        if step < cursor + stage.steps:
            return stage, step - cursor
        cursor += stage.steps
    return (stages[-1], stages[-1].steps) if stages else (None, step)


def _stage_param_groups(model: torch.nn.Module, stage: TrainingStageConfig) -> List[Dict[str, Any]]:
    provenance = model.parameter_provenance
    imported = set(provenance.imported_parameter_names)
    scales = set(provenance.scale_parameter_names)
    buckets: Dict[str, List[Tensor]] = {"pretrained": [], "new": [], "scales": []}
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name in scales:
            buckets["scales"].append(param)
        elif name in imported:
            buckets["pretrained"].append(param)
        else:
            buckets["new"].append(param)
    lrs = {"pretrained": stage.lr_pretrained, "new": stage.lr_new, "scales": stage.lr_scales}
    return [{"params": values, "lr": lrs[key], "group_name": key} for key, values in buckets.items() if values]


def finite_and_clip_gradients(parameters: Sequence[Tensor], max_norm: float) -> float:
    """Check every gradient and clip with a numerically stable global norm.

    Some MPS reductions can overflow while every individual float32 gradient is
    finite. Accumulating squared norms on CPU in float64 tests the gradients
    themselves rather than that reduction artifact.
    """
    total_sq = 0.0
    grads: List[Tensor] = []
    for parameter in parameters:
        grad = parameter.grad
        if grad is None:
            continue
        if not bool(torch.isfinite(grad).all().item()):
            return float("nan")
        grads.append(grad)
        # Move first: MPS does not implement a float64 cast on-device.
        cpu_grad = grad.detach().to(device="cpu").to(dtype=torch.float64)
        total_sq += float(torch.sum(cpu_grad.square()).item())
    grad_norm = math.sqrt(total_sq)
    if math.isfinite(grad_norm) and max_norm > 0.0 and grad_norm > max_norm:
        scale = float(max_norm) / (grad_norm + 1e-12)
        for grad in grads:
            grad.mul_(scale)
    return grad_norm


def distillation_loss(
    student_logits: Tensor,
    teacher_logits: Tensor,
    labels: Tensor,
    *,
    beta: float,
    temperature: float,
    student_hidden: Optional[Tensor] = None,
    teacher_hidden: Optional[Tensor] = None,
    hidden_weight: float = 0.0,
) -> Tuple[Tensor, Dict[str, float]]:
    """Padding-aware causal CE plus E2 logit and optional hidden distillation."""
    s = student_logits[:, :-1].contiguous()
    t = teacher_logits[:, :-1].detach().contiguous()
    y = labels[:, 1:].contiguous()
    ce = F.cross_entropy(s.reshape(-1, s.size(-1)), y.reshape(-1), ignore_index=-100)
    valid = y.reshape(-1) != -100
    sf = s.reshape(-1, s.size(-1))[valid]
    tf = t.reshape(-1, t.size(-1))[valid]
    temp = max(float(temperature), 1e-6)
    kl = F.kl_div(
        F.log_softmax(sf / temp, dim=-1),
        F.softmax(tf / temp, dim=-1),
        reduction="batchmean",
    ) * temp**2
    total = ce + float(beta) * kl
    details = {"ce": float(ce.detach()), "distill_kl": float(kl.detach())}
    if hidden_weight > 0.0:
        if student_hidden is None or teacher_hidden is None:
            raise ValueError("hidden states are required when hidden_weight > 0")
        sh = student_hidden[:, :-1].reshape(-1, student_hidden.size(-1))[valid]
        th = teacher_hidden[:, :-1].detach().reshape(-1, teacher_hidden.size(-1))[valid]
        hidden_mse = F.mse_loss(sh.float(), th.float())
        total = total + float(hidden_weight) * hidden_mse
        details["hidden_mse"] = float(hidden_mse.detach())
    return total, details


def try_load_tokenizer(name: str, revision: str = "") -> Any:
    if not name:
        return None
    try:
        from transformers import AutoTokenizer  # type: ignore
    except ImportError as e:
        raise ImportError("pip install transformers to use tokenizer_name=") from e
    kwargs: Dict[str, Any] = {"trust_remote_code": True}
    if revision:
        kwargs["revision"] = revision
    return AutoTokenizer.from_pretrained(name, **kwargs)


@torch.no_grad()
def eval_per_effort(
    model: Union[DAPHHybridModelV3, QwenExFusionModel],
    batcher: TextBatcher,
    *,
    n_batches: int = 5,
    detailed: bool = False,
) -> Dict[str, Any]:
    model.eval()
    losses = {f"fixed_{i}": [] for i in range(4)}
    receipts: Dict[str, List[Dict[str, Any]]] = {f"fixed_{i}": [] for i in range(4)}
    for _ in range(n_batches):
        try:
            batch = next(batcher)
        except StopIteration:
            break
        if len(batch) == 3:
            ids, labels, mask = batch
        else:
            ids, labels = batch[0], batch[1]
            mask = None
        for e in range(4):
            kwargs = {"return_compute_receipt": True} if isinstance(model, QwenExFusionModel) else {}
            out = model(ids, attention_mask=mask, effort_mode=f"fixed_{e}", **kwargs)
            logits = out["logits"] if isinstance(out, dict) else out
            if isinstance(out, dict) and "compute_stats" in out:
                receipts[f"fixed_{e}"].append(out["compute_stats"])
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.reshape(-1, shift_logits.size(-1)),
                shift_labels.reshape(-1),
                ignore_index=-100,
            )
            losses[f"fixed_{e}"].append(float(loss.item()))
    model.train()
    mean_losses = {k: (sum(v) / len(v) if v else float("nan")) for k, v in losses.items()}
    if not detailed:
        return mean_losses
    scales = {
        n: float(p.detach()) for n, p in model.named_parameters() if n.endswith("_scale")
    }
    report: Dict[str, Any] = {}
    for mode, ce in mean_losses.items():
        rs = receipts[mode]
        report[mode] = {
            "ce": ce,
            "perplexity": math.exp(ce) if ce < 80 else float("inf"),
            "quality": None,
            "average_executed_layers": (
                sum(float(r.get("executed_layer_count", 0)) for r in rs) / len(rs) if rs else None
            ),
            "estimated_normalized_compute": (
                sum(float(r.get("normalized_compute_cost", 0)) for r in rs) / len(rs) if rs else None
            ),
            "latency_ms": None,
            "augmentation_scales": scales,
        }
    return report


def train_adapt(
    model: Union[DAPHHybridModelV3, QwenExFusionModel],
    cfg: RealTrainConfig,
) -> Dict[str, Any]:
    set_seed(cfg.seed)
    device = torch.device(cfg.device)
    model = model.to(device)
    retention_reference = None
    if cfg.retention_kl_threshold is not None and isinstance(model, QwenExFusionModel):
        retention_reference = copy.deepcopy(model).eval()
        for p in retention_reference.parameters():
            p.requires_grad_(False)
    frozen = apply_freeze(model, cfg.freeze_name_contains)

    tok = (
        try_load_tokenizer(cfg.tokenizer_name, cfg.tokenizer_revision)
        if cfg.tokenizer_name
        else None
    )
    if not cfg.data_path:
        raise ValueError("RealTrainConfig.data_path is required")
    texts = load_jsonl_training_records(
        cfg.data_path, answer_only_loss=cfg.answer_only_loss,
    )
    batcher = TextBatcher(
        texts, tokenizer=tok, seq_len=cfg.seq_len,
        batch_size=cfg.batch_size, device=device, seed=cfg.seed,
        answer_only_loss=cfg.answer_only_loss,
    )
    val_batcher = None
    if cfg.val_path:
        val_texts = load_jsonl_training_records(
            cfg.val_path, answer_only_loss=cfg.answer_only_loss,
        )
        val_batcher = TextBatcher(
            val_texts, tokenizer=tok, seq_len=cfg.seq_len,
            batch_size=cfg.batch_size, device=device, seed=cfg.seed + 1,
            answer_only_loss=cfg.answer_only_loss,
        )

    start_step = 0
    optimizer_steps_completed = 0
    examples_seen = 0
    tokens_seen = 0
    resume_checkpoint = None
    if cfg.resume:
        ck = torch.load(cfg.resume, map_location=device, weights_only=False)
        resume_checkpoint = ck
        model.load_state_dict(ck["model"])
        start_step = int(ck.get("next_micro_step", ck.get("step", 0)))
        optimizer_steps_completed = int(ck.get("optimizer_steps_completed", 0))
        examples_seen = int(ck.get("examples_seen", 0))
        tokens_seen = int(ck.get("tokens_seen", 0))

    active_stage, active_stage_step = _stage_at(cfg.stages, start_step)
    if active_stage is not None:
        stage_membership = apply_training_stage(model, active_stage)
        opt = AdamW(_stage_param_groups(model, active_stage), weight_decay=cfg.weight_decay)
    else:
        stage_membership = {"trained": [], "frozen": frozen}
        opt = AdamW(_param_groups(model, cfg), weight_decay=cfg.weight_decay)
    if resume_checkpoint is not None and "optimizer" in resume_checkpoint:
        opt.load_state_dict(resume_checkpoint["optimizer"])

    gen = torch.Generator(device="cpu").manual_seed(cfg.seed)
    effort_hist: Dict[str, int] = {}
    examples_by_effort: Dict[str, int] = {}
    tokens_by_effort: Dict[str, int] = {}
    optimizer_steps_touching_effort: Dict[str, int] = {}
    pending_efforts: set[str] = set()
    pending_microsteps = 0
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    history: List[Dict[str, Any]] = []
    t0 = time.time()
    model.train()
    opt.zero_grad(set_to_none=True)

    for step in range(start_step, cfg.steps):
        selected_stage, stage_step = _stage_at(cfg.stages, step)
        if selected_stage is not None and selected_stage is not active_stage:
            if pending_microsteps:
                raise RuntimeError("Training stage boundary must align with a completed accumulation window")
            active_stage = selected_stage
            stage_membership = apply_training_stage(model, active_stage)
            opt = AdamW(_stage_param_groups(model, active_stage), weight_decay=cfg.weight_decay)
        for g in opt.param_groups:
            # simple warmup on each group relative to its base lr
            base = g.get("lr", cfg.lr)
            # restore base from initial — store once
            if "base_lr" not in g:
                g["base_lr"] = g["lr"]
            warm = min(1.0, float(step + 1) / float(max(cfg.warmup_steps, 1)))
            g["lr"] = g["base_lr"] * warm

        ids, labels, mask = next(batcher)
        if cfg.effort_schedule:
            emode = cfg.effort_schedule[step % len(cfg.effort_schedule)]
            if emode not in {"fixed_0", "fixed_1", "fixed_2", "fixed_3"}:
                raise ValueError(f"Invalid effort_schedule entry: {emode!r}")
        else:
            effort_probs = active_stage.effort_sampling if active_stage is not None else cfg.effort_probs
            emode = sample_effort_mode(
                type("T", (), {"effort_mode": cfg.effort_mode, "effort_probs": effort_probs})(),
                gen,
            )
        effort_hist[emode] = effort_hist.get(emode, 0) + 1
        examples = int(ids.shape[0])
        valid_tokens = int((labels != -100).sum().item())
        examples_seen += examples
        tokens_seen += valid_tokens
        examples_by_effort[emode] = examples_by_effort.get(emode, 0) + examples
        tokens_by_effort[emode] = tokens_by_effort.get(emode, 0) + valid_tokens
        pending_efforts.add(emode)
        effort_idx = int(emode[-1]) if emode.startswith("fixed_") else 2
        distill_enabled = effort_idx in (0, 1) and (
            active_stage is None
            or (effort_idx == 0 and active_stage.distill_e0)
            or (effort_idx == 1 and active_stage.distill_e1)
        )
        kwargs = (
            {
                "return_compute_receipt": True,
                "return_hidden_state": (distill_enabled and cfg.hidden_distillation_weight > 0.0) or effort_idx == 3,
            }
            if isinstance(model, QwenExFusionModel)
            else {}
        )
        out = model(ids, attention_mask=mask, effort_mode=emode, **kwargs)
        logits = out["logits"] if isinstance(out, dict) else out
        # next-token: logits[:-1] vs labels[1:], ignore pad (-100)
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()
        details: Dict[str, float] = {}
        if isinstance(model, QwenExFusionModel) and distill_enabled:
            with torch.no_grad():
                teacher_mode = active_stage.teacher_mode if active_stage is not None else "fixed_2"
                teacher = model(
                    ids, attention_mask=mask, effort_mode=teacher_mode,
                    return_hidden_state=cfg.hidden_distillation_weight > 0.0,
                )
            teacher_logits = teacher["logits"] if isinstance(teacher, dict) else teacher
            beta = cfg.beta_e0 if effort_idx == 0 else cfg.beta_e1
            loss, details = distillation_loss(
                logits, teacher_logits, labels, beta=beta,
                temperature=cfg.distillation_temperature,
                student_hidden=out.get("hidden_state") if isinstance(out, dict) else None,
                teacher_hidden=teacher.get("hidden_state") if isinstance(teacher, dict) else None,
                hidden_weight=cfg.hidden_distillation_weight,
            )
        else:
            loss = F.cross_entropy(
                shift_logits.reshape(-1, shift_logits.size(-1)),
                shift_labels.reshape(-1), ignore_index=-100,
            )
            if isinstance(model, QwenExFusionModel) and effort_idx == 3:
                details["e3_task_ce"] = float(loss.detach())
                with torch.no_grad():
                    anchor_output = model(
                        ids, attention_mask=mask, effort_mode="fixed_2", return_hidden_state=True,
                    )
                details["e3_hidden_delta_l2"] = float(
                    (out["hidden_state"] - anchor_output["hidden_state"]).norm(dim=-1).mean().detach()
                )
                guard_weight = float(active_stage.e3_regression_guard_weight) if active_stage is not None else 0.0
                if guard_weight > 0.0:
                    valid = shift_labels.reshape(-1) != -100
                    e3_flat = shift_logits.reshape(-1, shift_logits.size(-1))[valid]
                    e2_flat = anchor_output["logits"][:, :-1].reshape(-1, anchor_output["logits"].size(-1))[valid]
                    regression_kl = F.kl_div(
                        F.log_softmax(e3_flat.float(), dim=-1),
                        F.softmax(e2_flat.float(), dim=-1),
                        reduction="batchmean",
                    )
                    weighted_guard = guard_weight * regression_kl
                    loss = loss + weighted_guard
                    details["e3_regression_kl"] = float(regression_kl.detach())
                    details["e3_weighted_regression_guard"] = float(weighted_guard.detach())
                details["e3_regression_guard_weight"] = guard_weight
                active_scales = [
                    parameter.detach().float() for name, parameter in model.named_parameters()
                    if name.endswith("latent_scale") and parameter.requires_grad
                ]
                if active_scales:
                    details["e3_raw_refinement_scale_mean"] = float(torch.stack(active_scales).mean())
                    details["e3_effective_refinement_scale_mean"] = float(
                        torch.stack([
                            model.latent_scale_limit * torch.tanh(value / model.latent_scale_limit)
                            for value in active_scales
                        ]).mean()
                    )
        if cfg.fail_on_nonfinite and not bool(torch.isfinite(loss).item()):
            raise FloatingPointError(
                f"Non-finite loss at micro-step {step} for effort {emode}"
            )
        (loss / cfg.grad_accum).backward()
        pending_microsteps += 1

        if (step + 1) % cfg.grad_accum == 0:
            grad_norm = finite_and_clip_gradients(
                [p for p in model.parameters() if p.requires_grad], cfg.grad_clip
            )
            if cfg.fail_on_nonfinite and not math.isfinite(grad_norm):
                bad = [
                    name for name, parameter in model.named_parameters()
                    if parameter.requires_grad and parameter.grad is not None
                    and not bool(torch.isfinite(parameter.grad).all().item())
                ]
                raise FloatingPointError(
                    f"Non-finite gradient norm at micro-step {step} for effort {emode}; "
                    f"non-finite gradients: {bad}"
                )
            opt.step()
            opt.zero_grad(set_to_none=True)
            optimizer_steps_completed += 1
            for touched in pending_efforts:
                optimizer_steps_touching_effort[touched] = optimizer_steps_touching_effort.get(touched, 0) + 1
            pending_efforts.clear()
            pending_microsteps = 0
        else:
            grad_norm = 0.0

        row = {
            "step": step,
            "loss": float(loss.item()),
            "effort_mode": emode,
            "grad_norm": float(grad_norm) if not isinstance(grad_norm, float) else grad_norm,
            **details,
        }
        history.append(row)
        if step % cfg.log_every == 0:
            print(f"step {step:5d}  loss={row['loss']:.4f}  effort={emode}")

        if val_batcher is not None and step > 0 and step % cfg.eval_every == 0:
            pe = eval_per_effort(model, val_batcher, n_batches=3, detailed=True)
            print(f"  per-effort val loss: {pe}")
            row["val_per_effort"] = pe

        if retention_reference is not None and step > 0 and step % cfg.eval_every == 0:
            with torch.no_grad():
                current = model(ids, attention_mask=mask, effort_mode="fixed_2")
                original = retention_reference(ids, attention_mask=mask, effort_mode="fixed_2")
                retention_kl = F.kl_div(
                    F.log_softmax(current.float(), dim=-1),
                    F.softmax(original.float(), dim=-1), reduction="batchmean",
                ).item()
            row["e2_retention_kl"] = retention_kl
            if retention_kl > float(cfg.retention_kl_threshold):
                raise RuntimeError(
                    f"E2 retention gate failed: KL {retention_kl:.6g} > {cfg.retention_kl_threshold}"
                )

        if (
            cfg.save_periodic_checkpoints
            and step > 0
            and step % max(cfg.eval_every, 1) == 0
            and pending_microsteps == 0
        ):
            ckpt = {
                "model": model.state_dict(),
                "optimizer": opt.state_dict(),
                "step": step + 1,
                "next_micro_step": step + 1,
                "micro_steps_completed": step + 1,
                "optimizer_steps_completed": optimizer_steps_completed,
                "examples_seen": examples_seen,
                "tokens_seen": tokens_seen,
                "effort_hist": dict(effort_hist),
                "train_config": asdict(cfg),
                "frozen_params": frozen,
            }
            torch.save(ckpt, out_dir / f"checkpoint_{step}.pt")

    # Flush a partial accumulation window instead of silently dropping it.
    if pending_microsteps > 0:
        grad_norm = finite_and_clip_gradients(
            [p for p in model.parameters() if p.requires_grad], cfg.grad_clip
        )
        if cfg.fail_on_nonfinite and not math.isfinite(grad_norm):
            raise FloatingPointError("Non-finite gradient norm in final accumulation window")
        opt.step()
        opt.zero_grad(set_to_none=True)
        optimizer_steps_completed += 1
        for touched in pending_efforts:
            optimizer_steps_touching_effort[touched] = optimizer_steps_touching_effort.get(touched, 0) + 1
        pending_efforts.clear()

    provenance = getattr(model, "parameter_provenance", None)
    final = {
        "model": model.state_dict(),
        "optimizer": opt.state_dict(),
        "step": cfg.steps,
        "next_micro_step": cfg.steps,
        "micro_steps_completed": cfg.steps,
        "optimizer_steps_completed": optimizer_steps_completed,
        "examples_seen": examples_seen,
        "tokens_seen": tokens_seen,
        "current_stage": active_stage.name if active_stage is not None else "default",
        "stage_step": _stage_at(cfg.stages, max(cfg.steps - 1, 0))[1] + 1 if cfg.stages else cfg.steps,
        "stage_parameter_membership": stage_membership,
        "effort_hist": dict(effort_hist),
        "examples_by_effort": examples_by_effort,
        "tokens_by_effort": tokens_by_effort,
        "optimizer_steps_touching_effort": optimizer_steps_touching_effort,
        "parameter_provenance": provenance.to_dict() if provenance is not None else None,
        "train_config": asdict(cfg),
        "frozen_params": frozen,
        "history_tail": history[-20:],
        "wall_time_s": time.time() - t0,
    }
    if cfg.save_final_checkpoint:
        torch.save(final, out_dir / "checkpoint_final.pt")
    if isinstance(model, QwenExFusionModel) and cfg.save_model_artifact:
        from .pretrained import save_adapted_checkpoint
        save_adapted_checkpoint(
            model, str(out_dir / "model_final.pt"),
            extra={"training_receipt": {
                "micro_steps_completed": cfg.steps,
                "optimizer_steps_completed": optimizer_steps_completed,
                "examples_seen": examples_seen,
                "tokens_seen": tokens_seen,
                "next_micro_step": cfg.steps,
                "current_stage": final["current_stage"],
                "stage_step": final["stage_step"],
            }},
        )
    (out_dir / "summary.json").write_text(
        json.dumps(
            {
                "effort_hist": effort_hist,
                "steps": cfg.steps,
                "wall_time_s": final["wall_time_s"],
                "frozen_params": frozen,
            },
            indent=2,
        )
    )
    return final
