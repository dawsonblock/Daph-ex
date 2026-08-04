"""
EffortPolicyTrainer — train sequence-level effort controller from counterfactuals.

Frozen base model. Soft utility targets:
  p*(e) = softmax(U_e / tau)
  L = KL(p* || p_phi)

Controls:
  - ShamEffortController (prompt-only, mask-aware)
  - effort_frequency_matched_random
  - compute_matched_random (permutation within raw-compute tolerance)
"""

from __future__ import annotations

import hashlib
import json
import math
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .counterfactual import EffortCounterfactual, soft_targets


@dataclass
class PolicyMetrics:
    loss: float
    top1_acc_cost_aware: float
    top1_acc_argmax: float
    top2_acc: float
    mean_utility_regret: float
    mean_target_l1: float
    brier_cost_aware: float
    brier_soft: float
    ece: float
    mean_realized_utility: float
    mean_confidence: float
    mean_entropy: float


@dataclass(frozen=True)
class EffortPolicyArtifact:
    """Immutable policy package bound to a base-model checkpoint."""

    policy_version: str
    base_model_digest: str
    train_dataset_digest: str
    validation_dataset_digest: str
    split_manifest_digest: str
    feature_dim: int
    feature_spec: str  # "hidden" | "sham_prompt"
    temperature: float
    training_seed: int
    training_config_digest: str
    initial_state_dict_digest: str
    metrics: Dict[str, float]
    state_dict_digest: str
    source_digest: str = ""
    training_status: str = "VERIFIED_FIT"  # or MANUAL_UNVERIFIED
    training_receipt: Optional[Dict[str, Any]] = None
    environment: Dict[str, str] = field(default_factory=dict)
    # legacy alias
    dataset_digest: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def save(self, path: str, state_dict: Dict[str, Tensor]) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"artifact": self.to_dict(), "state_dict": state_dict}, p)

    @staticmethod
    def load(path: str) -> Tuple["EffortPolicyArtifact", Dict[str, Tensor]]:
        obj = torch.load(path, map_location="cpu", weights_only=False)
        art = EffortPolicyArtifact(**obj["artifact"])
        sd = obj["state_dict"]
        actual = _state_dict_digest(sd)
        if actual != art.state_dict_digest:
            raise ValueError(
                f"Policy state_dict digest mismatch: "
                f"artifact={art.state_dict_digest} actual={actual}"
            )
        return art, sd

    def assert_compatible(self, base_model_digest: str) -> None:
        if self.base_model_digest != base_model_digest:
            raise ValueError(
                f"Policy bound to base_model_digest={self.base_model_digest[:12]}… "
                f"but model has {base_model_digest[:12]}…"
            )


def _state_dict_digest(sd: Dict[str, Tensor]) -> str:
    h = hashlib.sha256()
    for k in sorted(sd.keys()):
        t = sd[k].detach().cpu().contiguous().reshape(-1)
        h.update(k.encode())
        h.update(str(t.dtype).encode())
        h.update(str(tuple(t.shape)).encode())
        h.update(bytes(t.view(torch.uint8).numpy()))
    return h.hexdigest()[:32]



def source_tree_digest(root: Optional[str] = None) -> str:
    """Hash daph package sources for implementation identity (not just version string)."""
    if root is None:
        root = str(Path(__file__).resolve().parent)
    root_p = Path(root)
    h = hashlib.sha256()
    paths = sorted(root_p.glob("*.py"))
    # also hash sibling pyproject if present
    pyproj = root_p.parent / "pyproject.toml"
    if pyproj.is_file():
        paths.append(pyproj)
    for path in paths:
        h.update(str(path.name).encode())
        h.update(path.read_bytes())
    return h.hexdigest()[:32]

def dataset_digest(records: Sequence[EffortCounterfactual]) -> str:
    """Hash full counterfactual supervision content (order by task_id)."""
    h = hashlib.sha256()
    ordered = sorted(records, key=lambda r: r.task_digest)
    for r in ordered:
        # Full canonical record — not just task/model digests
        payload = json.dumps(r.to_dict(), sort_keys=True, default=str)
        h.update(payload.encode())
    return h.hexdigest()[:32]


class ShamEffortController(nn.Module):
    """
    Prompt-only baseline. Features from valid tokens / optional decoded text.
    No model hidden states.
    """

    FEAT_DIM = 12

    def __init__(self, num_levels: int = 4, hidden: int = 64) -> None:
        super().__init__()
        self.num_levels = num_levels
        self.net = nn.Sequential(
            nn.Linear(self.FEAT_DIM, hidden),
            nn.GELU(),
            nn.Linear(hidden, num_levels),
        )

    @staticmethod
    def features_from_ids(
        input_ids: Tensor,
        attention_mask: Optional[Tensor] = None,
        prompt_text: Optional[str] = None,
    ) -> Tensor:
        """
        (B, L) ids + optional mask → (B, FEAT_DIM).
        Prefer decoded prompt_text for digit/operator counts when available.
        """
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)
        B, L = input_ids.shape
        device = input_ids.device
        if attention_mask is None:
            attention_mask = torch.ones(B, L, device=device, dtype=torch.long)
        else:
            attention_mask = attention_mask.to(device=device).long()
            if attention_mask.dim() == 1:
                attention_mask = attention_mask.unsqueeze(0)

        feats_list = []
        for b in range(B):
            m = attention_mask[b].bool()
            valid = input_ids[b][m]
            n_valid = max(int(m.sum().item()), 1)
            x = valid.float()
            length = float(n_valid)
            mean_id = float(x.mean()) if n_valid else 0.0
            std_id = float(x.std(unbiased=False)) if n_valid > 1 else 0.0

            # text-derived when available
            text = prompt_text if (prompt_text is not None and B == 1) else ""
            if text:
                n_digit = sum(ch.isdigit() for ch in text)
                n_op = sum(ch in "+-*/=<>" for ch in text)
                n_punct = sum(ch in ".,;:!?()[]{}" for ch in text)
                n_ws = sum(ch.isspace() for ch in text)
                n_nl = text.count("\n")
                n_char = float(len(text))
            else:
                n_digit = n_op = n_punct = n_ws = n_nl = 0
                n_char = length

            row = [
                length,
                math.log1p(length),
                mean_id / 1000.0,
                std_id / 1000.0,
                float(n_digit),
                float(n_op),
                float(n_punct),
                float(n_ws),
                float(n_nl),
                n_char / 100.0,
                float(n_digit) / max(n_char, 1.0),
                float(n_op) / max(n_char, 1.0),
            ]
            feats_list.append(row)
        return torch.tensor(feats_list, dtype=torch.float32, device=device)

    def forward(self, feats: Tensor) -> Tensor:
        return F.softmax(self.net(feats), dim=-1)




@dataclass(frozen=True)
class PolicyTrainingConfig:
    optimizer: str = "AdamW"
    lr: float = 1e-3
    weight_decay: float = 0.0
    batch_size: int = 32
    epochs: int = 20
    temperature: float = 0.1
    seed: int = 0
    feature_spec: str = "hidden"

    def digest(self) -> str:
        return hashlib.sha256(json.dumps(asdict(self), sort_keys=True).encode()).hexdigest()[:32]


@dataclass
class TrainingReceipt:
    """What actually happened during fit() — not just declared config."""
    epochs_requested: int
    epochs_completed: int
    batch_size: int
    optimizer_steps: int
    examples_seen: int
    final_train_metrics: Dict[str, float]
    final_val_metrics: Optional[Dict[str, float]] = None
    config_digest: str = ""
    source_digest: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class EffortPolicyTrainer:
    def __init__(
        self,
        controller: nn.Module,
        config: Optional[PolicyTrainingConfig] = None,
        *,
        lr: Optional[float] = None,
        temperature: Optional[float] = None,
        device: str = "cpu",
        feature_spec: Optional[str] = None,
        seed: Optional[int] = None,
        weight_decay: Optional[float] = None,
        batch_size: Optional[int] = None,
        arm_qualification_report: Optional[Mapping[str, Any]] = None,
        oracle_opportunity_report: Optional[Mapping[str, Any]] = None,
    ) -> None:
        # Authoritative config — kwargs override defaults for convenience
        cfg = config or PolicyTrainingConfig()
        if lr is not None:
            cfg = PolicyTrainingConfig(**{**asdict(cfg), "lr": lr})
        if temperature is not None:
            cfg = PolicyTrainingConfig(**{**asdict(cfg), "temperature": temperature})
        if feature_spec is not None:
            cfg = PolicyTrainingConfig(**{**asdict(cfg), "feature_spec": feature_spec})
        if seed is not None:
            cfg = PolicyTrainingConfig(**{**asdict(cfg), "seed": seed})
        if weight_decay is not None:
            cfg = PolicyTrainingConfig(**{**asdict(cfg), "weight_decay": weight_decay})
        if batch_size is not None:
            cfg = PolicyTrainingConfig(**{**asdict(cfg), "batch_size": batch_size})
        self.config = cfg
        self.controller = controller.to(device)
        if cfg.optimizer.lower() != "adamw":
            raise ValueError(f"Unsupported optimizer {cfg.optimizer}")
        self.opt = torch.optim.AdamW(
            self.controller.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
        )
        self.temperature = cfg.temperature
        self.device = device
        self.feature_spec = cfg.feature_spec
        self.seed = int(cfg.seed)
        self.batch_size = int(cfg.batch_size)
        self.generator = torch.Generator(device="cpu")
        self.generator.manual_seed(self.seed)
        self.initial_state_dict_digest = _state_dict_digest(
            {k: v.detach().cpu().clone() for k, v in self.controller.state_dict().items()}
        )
        self._policy_training_authorized = False
        if arm_qualification_report is not None or oracle_opportunity_report is not None:
            self.authorize_policy_training(arm_qualification_report, oracle_opportunity_report)

    def authorize_policy_training(
        self,
        arm_qualification_report: Optional[Mapping[str, Any]],
        oracle_opportunity_report: Optional[Mapping[str, Any]],
    ) -> None:
        """Open the policy gate only after both scientific prerequisites pass."""
        if not arm_qualification_report or not bool(arm_qualification_report.get("qualified")):
            raise RuntimeError("Policy training blocked: effort arms are not qualified")
        if not oracle_opportunity_report or not bool(oracle_opportunity_report.get("has_routing_opportunity")):
            raise RuntimeError("Policy training blocked: oracle opportunity gate did not pass")
        self._policy_training_authorized = True

    def _batch_from_records(
        self,
        records: Sequence[EffortCounterfactual],
        mode: str = "hidden",
        input_ids_list: Optional[Sequence[Tensor]] = None,
        attention_masks: Optional[Sequence[Optional[Tensor]]] = None,
        prompt_texts: Optional[Sequence[Optional[str]]] = None,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        targets, bests, argmaxes, utils, feats = [], [], [], [], []
        for i, r in enumerate(records):
            p = soft_targets(r.utility, temperature=self.temperature)
            targets.append(p)
            bests.append(r.best_effort)
            argmaxes.append(r.argmax_effort)
            utils.append(list(r.utility))
            if mode == "hidden":
                feats.append(r.probe_hidden)
            else:
                if input_ids_list is None:
                    raise ValueError("sham mode requires input_ids_list")
                mask = attention_masks[i] if attention_masks else None
                text = prompt_texts[i] if prompt_texts else None
                f = ShamEffortController.features_from_ids(
                    input_ids_list[i], attention_mask=mask, prompt_text=text
                )
                feats.append(f.squeeze(0).tolist())
        return (
            torch.tensor(feats, dtype=torch.float32, device=self.device),
            torch.tensor(targets, dtype=torch.float32, device=self.device),
            torch.tensor(bests, dtype=torch.long, device=self.device),
            torch.tensor(argmaxes, dtype=torch.long, device=self.device),
            torch.tensor(utils, dtype=torch.float32, device=self.device),
        )

    def _forward_probs(self, feats: Tensor, mode: str) -> Tensor:
        if mode == "hidden":
            if hasattr(self.controller, "token_level"):
                out = self.controller(feats)
                return out["effort_probs"] if isinstance(out, dict) else out
            logits = self.controller(feats)
            return F.softmax(logits, dim=-1) if logits.dim() == 2 else logits
        return self.controller(feats)

    def train_epoch(
        self,
        records: Sequence[EffortCounterfactual],
        mode: str = "hidden",
        input_ids_list: Optional[Sequence[Tensor]] = None,
        attention_masks: Optional[Sequence[Optional[Tensor]]] = None,
        prompt_texts: Optional[Sequence[Optional[str]]] = None,
        batch_size: Optional[int] = None,
    ) -> PolicyMetrics:
        self.controller.train()
        if batch_size is None:
            batch_size = self.batch_size
        n = len(records)
        order = torch.randperm(n, generator=self.generator).tolist()
        total_loss = 0.0
        n_batches = 0
        all_pred, all_tgt, all_best, all_argmax, all_utils = [], [], [], [], []

        for start in range(0, n, batch_size):
            idx = order[start : start + batch_size]
            batch_recs = [records[i] for i in idx]
            batch_ids = [input_ids_list[i] for i in idx] if input_ids_list else None
            batch_masks = [attention_masks[i] for i in idx] if attention_masks else None
            batch_texts = [prompt_texts[i] for i in idx] if prompt_texts else None
            feats, tgt, best, argmax, utils = self._batch_from_records(
                batch_recs, mode, batch_ids, batch_masks, batch_texts
            )
            probs = self._forward_probs(feats, mode)
            log_p = torch.log(probs.clamp_min(1e-8))
            loss = F.kl_div(log_p, tgt, reduction="batchmean")
            self.opt.zero_grad()
            loss.backward()
            self.opt.step()
            total_loss += float(loss.item())
            n_batches += 1
            all_pred.append(probs.detach())
            all_tgt.append(tgt.detach())
            all_best.append(best.detach())
            all_argmax.append(argmax.detach())
            all_utils.append(utils.detach())

        return self._metrics(
            torch.cat(all_pred),
            torch.cat(all_tgt),
            torch.cat(all_best),
            torch.cat(all_argmax),
            torch.cat(all_utils),
            total_loss / max(n_batches, 1),
        )

    @torch.no_grad()
    def evaluate(
        self,
        records: Sequence[EffortCounterfactual],
        mode: str = "hidden",
        input_ids_list: Optional[Sequence[Tensor]] = None,
        attention_masks: Optional[Sequence[Optional[Tensor]]] = None,
        prompt_texts: Optional[Sequence[Optional[str]]] = None,
    ) -> PolicyMetrics:
        self.controller.eval()
        feats, tgt, best, argmax, utils = self._batch_from_records(
            records, mode, input_ids_list, attention_masks, prompt_texts
        )
        probs = self._forward_probs(feats, mode)
        log_p = torch.log(probs.clamp_min(1e-8))
        loss = float(F.kl_div(log_p, tgt, reduction="batchmean").item())
        return self._metrics(probs, tgt, best, argmax, utils, loss)

    def predict_efforts(
        self,
        records: Sequence[EffortCounterfactual],
        mode: str = "hidden",
        input_ids_list: Optional[Sequence[Tensor]] = None,
        attention_masks: Optional[Sequence[Optional[Tensor]]] = None,
        prompt_texts: Optional[Sequence[Optional[str]]] = None,
    ) -> List[int]:
        self.controller.eval()
        with torch.no_grad():
            feats, _, _, _, _ = self._batch_from_records(
                records, mode, input_ids_list, attention_masks, prompt_texts
            )
            probs = self._forward_probs(feats, mode)
            return probs.argmax(dim=-1).cpu().tolist()


    def fit(
        self,
        train_records: Sequence[EffortCounterfactual],
        validation_records: Optional[Sequence[EffortCounterfactual]] = None,
        *,
        mode: str = "hidden",
        input_ids_list: Optional[Sequence[Tensor]] = None,
        attention_masks: Optional[Sequence[Optional[Tensor]]] = None,
        prompt_texts: Optional[Sequence[Optional[str]]] = None,
        val_input_ids_list: Optional[Sequence[Tensor]] = None,
        val_attention_masks: Optional[Sequence[Optional[Tensor]]] = None,
        val_prompt_texts: Optional[Sequence[Optional[str]]] = None,
    ) -> Tuple[PolicyMetrics, TrainingReceipt]:
        """
        Official training API: runs exactly config.epochs epochs at config.batch_size.
        Returns final train metrics and an execution receipt.
        """
        if not self._policy_training_authorized:
            raise RuntimeError(
                "Policy training blocked: call authorize_policy_training() with passing "
                "effort-arm and oracle-opportunity reports"
            )
        epochs = int(self.config.epochs)
        bs = int(self.config.batch_size)
        steps = 0
        examples = 0
        last_metrics: Optional[PolicyMetrics] = None
        for _ in range(epochs):
            last_metrics = self.train_epoch(
                train_records,
                mode=mode,
                input_ids_list=input_ids_list,
                attention_masks=attention_masks,
                prompt_texts=prompt_texts,
                batch_size=bs,
            )
            # count steps/examples from this epoch
            n = len(train_records)
            steps += (n + bs - 1) // bs
            examples += n
        assert last_metrics is not None
        val_metrics = None
        if validation_records is not None and len(validation_records) > 0:
            vm = self.evaluate(
                validation_records,
                mode=mode,
                input_ids_list=val_input_ids_list,
                attention_masks=val_attention_masks,
                prompt_texts=val_prompt_texts,
            )
            val_metrics = {
                "loss": vm.loss,
                "mean_utility_regret": vm.mean_utility_regret,
                "mean_realized_utility": vm.mean_realized_utility,
                "top1_acc_argmax": vm.top1_acc_argmax,
            }
        receipt = TrainingReceipt(
            epochs_requested=epochs,
            epochs_completed=epochs,
            batch_size=bs,
            optimizer_steps=steps,
            examples_seen=examples,
            final_train_metrics={
                "loss": last_metrics.loss,
                "mean_utility_regret": last_metrics.mean_utility_regret,
                "mean_realized_utility": last_metrics.mean_realized_utility,
                "top1_acc_argmax": last_metrics.top1_acc_argmax,
            },
            final_val_metrics=val_metrics,
            config_digest=self.config.digest(),
            source_digest=source_tree_digest(),
        )
        self._last_receipt = receipt
        return last_metrics, receipt

    def build_artifact(
        self,
        *,
        base_model_digest: str,
        train_records: Sequence[EffortCounterfactual],
        metrics: PolicyMetrics,
        validation_records: Optional[Sequence[EffortCounterfactual]] = None,
        split_manifest_digest: str = "",
        policy_version: str = "1.0",
        feature_dim: Optional[int] = None,
        require_training_receipt: bool = True,
    ) -> Tuple[EffortPolicyArtifact, Dict[str, Tensor]]:
        receipt = getattr(self, "_last_receipt", None)
        if require_training_receipt and receipt is None:
            raise ValueError(
                "No TrainingReceipt — call trainer.fit() before build_artifact(), "
                "or pass require_training_receipt=False for MANUAL_UNVERIFIED artifacts"
            )
        if receipt is not None and receipt.config_digest != self.config.digest():
            raise ValueError(
                "TrainingReceipt config_digest does not match current PolicyTrainingConfig"
            )
        sd = {k: v.detach().cpu().clone() for k, v in self.controller.state_dict().items()}
        if feature_dim is None:
            if hasattr(self.controller, "norm"):
                feature_dim = int(self.controller.norm.normalized_shape[0])
            elif self.feature_spec == "sham_prompt":
                feature_dim = ShamEffortController.FEAT_DIM
            else:
                feature_dim = 0
        train_d = dataset_digest(train_records)
        val_d = dataset_digest(validation_records or [])
        import sys
        src = source_tree_digest()
        env = {
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "daph_version": "3.4.1",
            "device": self.device,
            "source_digest": src,
        }
        art = EffortPolicyArtifact(
            policy_version=policy_version,
            base_model_digest=base_model_digest,
            train_dataset_digest=train_d,
            validation_dataset_digest=val_d,
            split_manifest_digest=split_manifest_digest,
            feature_dim=int(feature_dim),
            feature_spec=self.feature_spec,
            temperature=self.temperature,
            training_seed=self.seed,
            training_config_digest=self.config.digest(),
            initial_state_dict_digest=self.initial_state_dict_digest,
            metrics={
                "loss": metrics.loss,
                "mean_utility_regret": metrics.mean_utility_regret,
                "mean_realized_utility": metrics.mean_realized_utility,
                "top1_acc_argmax": metrics.top1_acc_argmax,
                "ece": metrics.ece,
            },
            state_dict_digest=_state_dict_digest(sd),
            source_digest=src,
            training_status=("VERIFIED_FIT" if receipt is not None else "MANUAL_UNVERIFIED"),
            training_receipt=receipt.to_dict() if receipt is not None else None,
            environment=env,
            dataset_digest=train_d,
        )
        return art, sd

    def _metrics(
        self,
        pred: Tensor,
        tgt: Tensor,
        best: Tensor,
        argmax: Tensor,
        utils: Tensor,
        loss: float,
    ) -> PolicyMetrics:
        chosen = pred.argmax(dim=-1)
        acc_cost = float((chosen == best).float().mean())
        acc_arg = float((chosen == argmax).float().mean())
        top2 = pred.topk(k=min(2, pred.size(-1)), dim=-1).indices
        acc2 = float((top2 == best.unsqueeze(-1)).any(dim=-1).float().mean())
        u_oracle = utils.max(dim=-1).values
        u_chosen = utils.gather(1, chosen.unsqueeze(1)).squeeze(1)
        regret = float((u_oracle - u_chosen).mean())
        l1 = float(((tgt - pred).abs()).mean())
        one_hot = F.one_hot(best, pred.size(-1)).float()
        brier_hard = float(((pred - one_hot) ** 2).mean())
        brier_soft = float(((pred - tgt) ** 2).mean())
        conf = pred.max(dim=-1).values
        # ECE with 10 bins
        ece = _ece(conf, (chosen == best).float(), n_bins=10)
        ent = -(pred.clamp_min(1e-8).log() * pred).sum(dim=-1).mean()
        return PolicyMetrics(
            loss=loss,
            top1_acc_cost_aware=acc_cost,
            top1_acc_argmax=acc_arg,
            top2_acc=acc2,
            mean_utility_regret=regret,
            mean_target_l1=l1,
            brier_cost_aware=brier_hard,
            brier_soft=brier_soft,
            ece=ece,
            mean_realized_utility=float(u_chosen.mean()),
            mean_confidence=float(conf.mean()),
            mean_entropy=float(ent),
        )


def _ece(confidence: Tensor, correct: Tensor, n_bins: int = 10) -> float:
    bins = torch.linspace(0, 1, n_bins + 1, device=confidence.device)
    ece = 0.0
    n = confidence.numel()
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (confidence > lo) & (confidence <= hi) if i > 0 else (confidence >= lo) & (confidence <= hi)
        if mask.sum() == 0:
            continue
        acc = correct[mask].mean()
        conf = confidence[mask].mean()
        ece += float((mask.sum().float() / n) * abs(acc - conf))
    return ece


def effort_frequency_matched_random(
    chosen_efforts: Sequence[int],
    seed: int = 0,
) -> List[int]:
    """Shuffle the multiset of policy-chosen efforts (effort-frequency matched)."""
    rng = random.Random(seed)
    levels = list(chosen_efforts)
    rng.shuffle(levels)
    return levels


# Back-compat alias
matched_random_policy = effort_frequency_matched_random


@dataclass
class MatchedRandomResult:
    efforts: List[int]
    policy_compute: float
    random_compute: float
    relative_difference: float
    matched: bool
    attempts: int


def compute_matched_random(
    chosen_efforts: Sequence[int],
    raw_costs: Sequence[Sequence[float]],
    *,
    seed: int = 0,
    n_candidates: int = 200,
    tol: float = 0.05,
    require_match: bool = True,
) -> MatchedRandomResult:
    """
    Find a permutation of chosen_efforts whose total raw compute is within
    tol relative of the policy's total compute.

    If require_match=True and no candidate satisfies tol, raises ValueError.
    Never silently returns an unmatched permutation as "matched".
    """
    rng = random.Random(seed)
    n = len(chosen_efforts)
    if len(raw_costs) != n:
        raise ValueError("raw_costs length must match chosen_efforts")
    policy_cost = sum(float(raw_costs[i][chosen_efforts[i]]) for i in range(n))
    base = list(chosen_efforts)
    best_perm = list(base)
    best_diff = float("inf")
    best_cost = policy_cost
    for attempt in range(1, n_candidates + 1):
        perm = list(base)
        rng.shuffle(perm)
        cost = sum(float(raw_costs[i][perm[i]]) for i in range(n))
        diff = abs(cost - policy_cost) / max(policy_cost, 1e-8)
        if diff < best_diff:
            best_diff = diff
            best_perm = perm
            best_cost = cost
        if diff <= tol:
            return MatchedRandomResult(
                efforts=perm,
                policy_compute=policy_cost,
                random_compute=cost,
                relative_difference=diff,
                matched=True,
                attempts=attempt,
            )
    result = MatchedRandomResult(
        efforts=best_perm,
        policy_compute=policy_cost,
        random_compute=best_cost,
        relative_difference=best_diff,
        matched=False,
        attempts=n_candidates,
    )
    if require_match:
        raise ValueError(
            f"No compute-matched permutation within tol={tol}; "
            f"best relative_difference={best_diff:.4f} after {n_candidates} attempts"
        )
    return result


@dataclass
class MatchedRandomEnsemble:
    results: List[MatchedRandomResult]
    requested_n: int
    found_n: int
    attempts: int
    acceptance_rate: float


def compute_matched_random_ensemble(
    chosen_efforts: Sequence[int],
    raw_costs: Sequence[Sequence[float]],
    *,
    seed: int = 0,
    n_valid: int = 50,
    n_candidates: int = 2000,
    tol: float = 0.05,
) -> MatchedRandomEnsemble:
    """Collect up to n_valid compute-matched permutations; report found/requested."""
    rng = random.Random(seed)
    n = len(chosen_efforts)
    policy_cost = sum(float(raw_costs[i][chosen_efforts[i]]) for i in range(n))
    base = list(chosen_efforts)
    valid: List[MatchedRandomResult] = []
    seen = set()
    attempt = 0
    for attempt in range(1, n_candidates + 1):
        if len(valid) >= n_valid:
            break
        perm = list(base)
        rng.shuffle(perm)
        key = tuple(perm)
        if key in seen:
            continue
        seen.add(key)
        cost = sum(float(raw_costs[i][perm[i]]) for i in range(n))
        diff = abs(cost - policy_cost) / max(policy_cost, 1e-8)
        if diff <= tol:
            valid.append(
                MatchedRandomResult(
                    efforts=perm,
                    policy_compute=policy_cost,
                    random_compute=cost,
                    relative_difference=diff,
                    matched=True,
                    attempts=attempt,
                )
            )
    return MatchedRandomEnsemble(
        results=valid,
        requested_n=n_valid,
        found_n=len(valid),
        attempts=attempt,
        acceptance_rate=len(valid) / max(attempt, 1),
    )


def evaluate_policy_utility(
    records: Sequence[EffortCounterfactual],
    chosen_efforts: Sequence[int],
) -> float:
    assert len(records) == len(chosen_efforts)
    return sum(r.utility[e] for r, e in zip(records, chosen_efforts)) / max(len(records), 1)


def evaluate_policy_raw_compute(
    records: Sequence[EffortCounterfactual],
    chosen_efforts: Sequence[int],
) -> float:
    assert len(records) == len(chosen_efforts)
    return sum(r.raw_compute[e] for r, e in zip(records, chosen_efforts)) / max(len(records), 1)


def gap_capture(
    u_policy: float,
    u_best_fixed: float,
    u_oracle: float,
) -> float:
    denom = u_oracle - u_best_fixed
    if abs(denom) < 1e-12:
        return 0.0
    return (u_policy - u_best_fixed) / denom


# ---------------------------------------------------------------------------
# Splits, validation, install
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SplitManifest:
    split_version: str
    seed: int
    train_task_digests: Tuple[str, ...]
    validation_task_digests: Tuple[str, ...]
    test_task_digests: Tuple[str, ...]

    def digest(self) -> str:
        payload = json.dumps(asdict(self), sort_keys=True)
        return hashlib.sha256(payload.encode()).hexdigest()[:32]

    def assert_no_leakage(self) -> None:
        tr, va, te = set(self.train_task_digests), set(self.validation_task_digests), set(self.test_task_digests)
        if tr & va:
            raise ValueError(f"train/validation leakage: {tr & va}")
        if tr & te:
            raise ValueError(f"train/test leakage: {tr & te}")
        if va & te:
            raise ValueError(f"validation/test leakage: {va & te}")




@dataclass(frozen=True)
class ExperimentManifest:
    """Five disjoint populations for a clean adaptive-compute experiment.

    ood_task_digests is true OOD only when ood_is_true_ood=True
    (e.g. leave-family-out). Random-partition remainder is an IID holdout.
    """
    qualification_task_digests: Tuple[str, ...]
    train_task_digests: Tuple[str, ...]
    validation_task_digests: Tuple[str, ...]
    test_task_digests: Tuple[str, ...]
    ood_task_digests: Tuple[str, ...]
    seed: int
    protocol_version: str = "1"
    ood_is_true_ood: bool = False

    def digest(self) -> str:
        return hashlib.sha256(json.dumps(asdict(self), sort_keys=True).encode()).hexdigest()[:32]

    def assert_disjoint(self) -> None:
        sets = {
            "Q": set(self.qualification_task_digests),
            "train": set(self.train_task_digests),
            "val": set(self.validation_task_digests),
            "test": set(self.test_task_digests),
            "ood": set(self.ood_task_digests),
        }
        names = list(sets)
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                inter = sets[names[i]] & sets[names[j]]
                if inter:
                    raise ValueError(f"{names[i]}/{names[j]} leakage: {list(inter)[:5]}")

    def all_digests(self) -> set:
        return (
            set(self.qualification_task_digests)
            | set(self.train_task_digests)
            | set(self.validation_task_digests)
            | set(self.test_task_digests)
            | set(self.ood_task_digests)
        )


def make_experiment_manifest(
    records: Sequence[EffortCounterfactual],
    *,
    seed: int = 0,
    qual_frac: float = 0.2,
    train_frac: float = 0.4,
    val_frac: float = 0.15,
    test_frac: float = 0.15,
    # remainder → OOD
    protocol_version: str = "1",
    family_key: Optional[Callable[[EffortCounterfactual], str]] = None,
) -> ExperimentManifest:
    """
    Deterministic partition into Q / train / val / test / OOD.
    If family_key is set, stratify within each family across the five roles
    (IID-balanced). Use make_leave_family_out_manifest for true OOD by family.
    """
    digests = [r.task_digest for r in records]
    if len(digests) != len(set(digests)):
        raise ValueError("duplicate task_digest in experiment manifest input")
    rng = random.Random(seed)
    if family_key is None:
        ordered = sorted(digests)
        rng.shuffle(ordered)
        n = len(ordered)
        n_q = int(n * qual_frac)
        n_tr = int(n * train_frac)
        n_va = int(n * val_frac)
        n_te = int(n * test_frac)
        q = tuple(ordered[:n_q])
        tr = tuple(ordered[n_q : n_q + n_tr])
        va = tuple(ordered[n_q + n_tr : n_q + n_tr + n_va])
        te = tuple(ordered[n_q + n_tr + n_va : n_q + n_tr + n_va + n_te])
        ood = tuple(ordered[n_q + n_tr + n_va + n_te :])
    else:
        by_fam: Dict[str, List[str]] = {}
        for r in records:
            by_fam.setdefault(family_key(r), []).append(r.task_digest)
        q_l, tr_l, va_l, te_l, ood_l = [], [], [], [], []
        for fam, ids in sorted(by_fam.items()):
            ids = sorted(ids)
            rng.shuffle(ids)
            n = len(ids)
            n_q = int(n * qual_frac)
            n_tr = int(n * train_frac)
            n_va = int(n * val_frac)
            n_te = int(n * test_frac)
            q_l.extend(ids[:n_q])
            tr_l.extend(ids[n_q : n_q + n_tr])
            va_l.extend(ids[n_q + n_tr : n_q + n_tr + n_va])
            te_l.extend(ids[n_q + n_tr + n_va : n_q + n_tr + n_va + n_te])
            ood_l.extend(ids[n_q + n_tr + n_va + n_te :])
        q, tr, va, te, ood = tuple(q_l), tuple(tr_l), tuple(va_l), tuple(te_l), tuple(ood_l)
    m = ExperimentManifest(q, tr, va, te, ood, seed, protocol_version, ood_is_true_ood=False)
    m.assert_disjoint()
    return m


def make_leave_family_out_manifest(
    records: Sequence[EffortCounterfactual],
    family_key: Callable[[EffortCounterfactual], str],
    held_out_family: str,
    *,
    seed: int = 0,
    train_frac: float = 0.7,
    val_frac: float = 0.15,
    protocol_version: str = "1-lfo",
) -> ExperimentManifest:
    """
    True leave-family-out: held_out_family → OOD (and optionally test),
    remaining families → qualification / train / val / test stratified.
    """
    digests = [r.task_digest for r in records]
    if len(digests) != len(set(digests)):
        raise ValueError("duplicate task_digest")
    in_fam = [r for r in records if family_key(r) == held_out_family]
    out_fam = [r for r in records if family_key(r) != held_out_family]
    if not in_fam:
        raise ValueError(f"held_out_family={held_out_family!r} has no records")
    if not out_fam:
        raise ValueError("no remaining families for train")
    # All held-out family → true OOD
    ood = tuple(sorted(r.task_digest for r in in_fam))
    # Remaining families: stratify each across Q/train/val/test
    rng = random.Random(seed)
    by_fam: Dict[str, List[str]] = {}
    for r in out_fam:
        by_fam.setdefault(family_key(r), []).append(r.task_digest)
    q_l, tr_l, va_l, te_l = [], [], [], []
    for fam, ids in sorted(by_fam.items()):
        ids = sorted(ids)
        rng.shuffle(ids)
        n = len(ids)
        n_q = max(0, int(n * 0.2))
        n_tr = int((n - n_q) * train_frac)
        n_va = int((n - n_q) * val_frac)
        q_l.extend(ids[:n_q])
        tr_l.extend(ids[n_q : n_q + n_tr])
        va_l.extend(ids[n_q + n_tr : n_q + n_tr + n_va])
        te_l.extend(ids[n_q + n_tr + n_va :])
    m = ExperimentManifest(
        tuple(q_l), tuple(tr_l), tuple(va_l), tuple(te_l), ood,
        seed, protocol_version, ood_is_true_ood=True,
    )
    m.assert_disjoint()
    return m


def make_split_manifest(
    records: Sequence[EffortCounterfactual],
    *,
    seed: int = 0,
    train_frac: float = 0.6,
    val_frac: float = 0.2,
    split_version: str = "1",
    family_key: Optional[Callable[[EffortCounterfactual], str]] = None,
) -> SplitManifest:
    """
    Deterministic random split by task_digest.
    Rejects duplicate task digests. Optional family_key enables stratified split.
    """
    digests = [r.task_digest for r in records]
    if len(digests) != len(set(digests)):
        raise ValueError("duplicate task_digest in split input — run validate_counterfactual_dataset first")
    rng = random.Random(seed)
    if family_key is None:
        ordered = sorted(digests)
        rng.shuffle(ordered)
        n = len(ordered)
        n_train = int(n * train_frac)
        n_val = int(n * val_frac)
        train = tuple(ordered[:n_train])
        val = tuple(ordered[n_train : n_train + n_val])
        test = tuple(ordered[n_train + n_val :])
    else:
        # stratified by family
        by_fam: Dict[str, List[str]] = {}
        for r in records:
            by_fam.setdefault(family_key(r), []).append(r.task_digest)
        train_l, val_l, test_l = [], [], []
        for fam, ids in sorted(by_fam.items()):
            ids = sorted(ids)
            rng.shuffle(ids)
            n = len(ids)
            n_train = int(n * train_frac)
            n_val = int(n * val_frac)
            train_l.extend(ids[:n_train])
            val_l.extend(ids[n_train : n_train + n_val])
            test_l.extend(ids[n_train + n_val :])
        train, val, test = tuple(train_l), tuple(val_l), tuple(test_l)
    m = SplitManifest(split_version, seed, train, val, test)
    m.assert_no_leakage()
    return m


def apply_split(
    records: Sequence[EffortCounterfactual],
    manifest: SplitManifest,
    *,
    strict: bool = True,
) -> Tuple[List[EffortCounterfactual], List[EffortCounterfactual], List[EffortCounterfactual]]:
    digests = [r.task_digest for r in records]
    if len(digests) != len(set(digests)):
        raise ValueError("duplicate task_digest in apply_split input")
    by = {r.task_digest: r for r in records}
    all_manifest = set(manifest.train_task_digests) | set(manifest.validation_task_digests) | set(manifest.test_task_digests)
    if strict:
        missing = all_manifest - set(by.keys())
        if missing:
            raise ValueError(f"manifest tasks missing from records: {list(missing)[:5]}...")
        extra = set(by.keys()) - all_manifest
        if extra:
            raise ValueError(f"records not in any split: {list(extra)[:5]}...")
    def _get(ids: Tuple[str, ...]) -> List[EffortCounterfactual]:
        out = []
        for d in ids:
            if d not in by:
                if strict:
                    raise ValueError(f"missing task_digest {d}")
                continue
            out.append(by[d])
        return out
    return _get(manifest.train_task_digests), _get(manifest.validation_task_digests), _get(manifest.test_task_digests)


@dataclass
class DatasetQualificationReport:
    total: int
    accepted: int
    dropped_unverifiable: int
    failures: Dict[str, int]
    records: List[EffortCounterfactual]


def validate_counterfactual_dataset(
    records: Sequence[EffortCounterfactual],
    *,
    require_all_verified: bool = True,
    expected_model_digest: Optional[str] = None,
    expected_config_digest: Optional[str] = None,
    expected_lambda_cost: Optional[float] = None,
    expected_tie_epsilon: Optional[float] = None,
    probe_dim: Optional[int] = None,
    check_utility_integrity: bool = True,
    return_report: bool = False,
):
    """
    Qualify records for policy training.
    Enforces uniform model/config/lambda/tie semantics.
    Optionally recomputes U/best/argmax and checks integrity.
    """
    from .counterfactual import compute_utility

    seen = set()
    out: List[EffortCounterfactual] = []
    failures: Dict[str, int] = {}
    dropped_unv = 0

    def _fail(key: str, msg: str = "") -> None:
        failures[key] = failures.get(key, 0) + 1
        raise ValueError(f"{key}: {msg}" if msg else key)

    # Uniformity across dataset
    if records:
        models = {r.model_digest for r in records}
        configs = {r.config_digest for r in records}
        lambdas = {r.lambda_cost for r in records}
        ties = {r.tie_epsilon for r in records}
        if expected_model_digest is None and len(models) != 1:
            _fail("mixed_model_digest", str(models))
        if expected_config_digest is None and len(configs) != 1:
            _fail("mixed_config_digest", str(configs))
        if expected_lambda_cost is None and len(lambdas) != 1:
            _fail("mixed_lambda_cost", str(lambdas))
        if expected_tie_epsilon is None and len(ties) != 1:
            _fail("mixed_tie_epsilon", str(ties))
        # Projection semantics: all None (runtime policy) or one shared projection
        proj_dims = {r.projection_dim for r in records}
        proj_seeds = {r.projection_seed for r in records}
        proj_digests = {r.projection_digest for r in records}
        if len(proj_dims) != 1 or len(proj_seeds) != 1 or len(proj_digests) != 1:
            _fail(
                "mixed_projection",
                f"dims={proj_dims} seeds={proj_seeds} digests={proj_digests}",
            )

    for r in records:
        if r.task_digest in seen:
            _fail("duplicate_task_digest", r.task_digest)
        seen.add(r.task_digest)
        if expected_model_digest is not None and r.model_digest != expected_model_digest:
            _fail("model_digest_mismatch", r.task_id)
        if expected_config_digest is not None and r.config_digest != expected_config_digest:
            _fail("config_digest_mismatch", r.task_id)
        if expected_lambda_cost is not None and abs(r.lambda_cost - expected_lambda_cost) > 1e-12:
            _fail("lambda_mismatch", r.task_id)
        if expected_tie_epsilon is not None and abs(r.tie_epsilon - expected_tie_epsilon) > 1e-12:
            _fail("tie_epsilon_mismatch", r.task_id)
        if any(not math.isfinite(x) for x in r.quality + r.compute + r.raw_compute + r.utility):
            _fail("nonfinite", r.task_id)
        if probe_dim is not None and len(r.probe_hidden) != probe_dim:
            _fail("probe_dim", f"{len(r.probe_hidden)} != {probe_dim}")
        if check_utility_integrity:
            u2, b2, a2 = compute_utility(r.quality, r.compute, r.lambda_cost, r.tie_epsilon)
            if any(abs(u2[e] - r.utility[e]) > 1e-5 for e in range(4)):
                _fail("utility_mismatch", r.task_id)
            if int(a2) != int(r.argmax_effort):
                _fail("argmax_mismatch", r.task_id)
            if int(b2) != int(r.best_effort):
                _fail("best_effort_mismatch", r.task_id)
        if require_all_verified:
            if any(s not in ("CORRECT", "INCORRECT") for s in r.verifier_status):
                dropped_unv += 1
                continue
        out.append(r)

    report = DatasetQualificationReport(
        total=len(records),
        accepted=len(out),
        dropped_unverifiable=dropped_unv,
        failures=failures,
        records=out,
    )
    if return_report:
        return report
    return out


def install_effort_policy(
    controller: nn.Module,
    artifact: EffortPolicyArtifact,
    state_dict: Dict[str, Tensor],
    *,
    base_model_digest: str,
    strict_source_match: bool = False,
    require_verified_fit: bool = True,
) -> None:
    """Official install path with full compatibility checks."""
    artifact.assert_compatible(base_model_digest)
    if require_verified_fit and getattr(artifact, "training_status", "VERIFIED_FIT") != "VERIFIED_FIT":
        raise ValueError(
            f"Refusing to install policy with training_status={getattr(artifact, 'training_status', None)}"
        )
    if strict_source_match:
        current = source_tree_digest()
        if artifact.source_digest and artifact.source_digest != current:
            raise ValueError(
                f"source_digest mismatch: artifact={artifact.source_digest} runtime={current}"
            )
    actual = _state_dict_digest(state_dict)
    if actual != artifact.state_dict_digest:
        raise ValueError(
            f"Policy state_dict digest mismatch: artifact={artifact.state_dict_digest} actual={actual}"
        )
    if artifact.feature_spec == "hidden" and hasattr(controller, "norm"):
        dim = int(controller.norm.normalized_shape[0])
        if artifact.feature_dim and dim != artifact.feature_dim:
            raise ValueError(f"feature_dim mismatch: artifact={artifact.feature_dim} controller={dim}")
    controller.load_state_dict(state_dict)
