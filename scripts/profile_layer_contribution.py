#!/usr/bin/env python3
"""Run a reproducible CE layer-contribution profile on an ExFusion checkpoint.

This command is the supervised-CE implementation of the profiling interface.
Verified-reward/RLVR experiments must supply an external adaptation callback;
the report labels this command's objective accurately and never calls it RLVR.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Tuple

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from daph.layer_contribution import (
    LayerAdaptationObjective,
    LayerContributionConfig,
    LayerContributionProfiler,
)
from daph.e3_protocol import ProfileTier, validate_profile_tier
from daph.qwen_exfusion import load_qwen_exfusion_checkpoint


def load_batches(path: str, device: torch.device) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    batches = []
    with Path(path).open() as handle:
        for line in handle:
            row = json.loads(line)
            ids = torch.tensor(row["input_ids"], dtype=torch.long, device=device)
            if ids.dim() == 1:
                ids = ids.unsqueeze(0)
            mask = torch.tensor(row.get("attention_mask", torch.ones_like(ids).tolist()), device=device)
            if mask.dim() == 1:
                mask = mask.unsqueeze(0)
            batches.append((ids, mask))
    if not batches:
        raise ValueError(f"No tokenized examples in {path}")
    return batches


def ce_score(model, batches) -> float:
    model.eval()
    total = 0.0
    with torch.no_grad():
        for ids, mask in batches:
            logits = model(ids, attention_mask=mask, effort_mode="fixed_2")
            loss = F.cross_entropy(logits[:, :-1].reshape(-1, logits.size(-1)), ids[:, 1:].reshape(-1))
            total += float(loss)
    return -(total / len(batches))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--train", required=True, help="JSONL with tokenized input_ids")
    parser.add_argument("--validation", required=True, help="JSONL with tokenized input_ids")
    parser.add_argument("--output", default="artifacts/layer_profile")
    parser.add_argument("--profile-mode", choices=("sparse", "middle_only", "full"), default="sparse")
    parser.add_argument("--layers", help="Comma-separated explicit zero-based indices")
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--profile-tier", choices=tuple(tier.value for tier in ProfileTier), default="PROFILE_SMOKE")
    parser.add_argument("--score-full", type=float, help="Pre-measured full-training score; score is negative CE")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    model = load_qwen_exfusion_checkpoint(args.checkpoint, map_location=args.device).to(device)
    train = load_batches(args.train, device)
    validation = load_batches(args.validation, device)
    explicit = [int(item) for item in args.layers.split(",")] if args.layers else None
    config = LayerContributionConfig(
        profile_mode=args.profile_mode, explicit_layers=explicit,
        training_steps=args.steps, seed=args.seed, validation_metric="negative_causal_ce",
        objective=LayerAdaptationObjective(kind="supervised_ce", name="causal_lm_ce"),
    )
    profiler = LayerContributionProfiler(model, config)

    def adapt(target, _layer, _objective, steps, _seed):
        parameters = [parameter for parameter in target.parameters() if parameter.requires_grad]
        optimizer = torch.optim.AdamW(parameters, lr=args.lr)
        target.train()
        for step in range(steps):
            ids, mask = train[step % len(train)]
            logits = target(ids, attention_mask=mask, effort_mode="fixed_2")
            loss = F.cross_entropy(logits[:, :-1].reshape(-1, logits.size(-1)), ids[:, 1:].reshape(-1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    report = profiler.run(lambda candidate: ce_score(candidate, validation), adapt,
                          full_reference_adapter=adapt if args.score_full is None else None,
                          score_full=args.score_full)
    profiler.save(report, args.output)
    # This command runs one profile seed. A multi-seed profile must aggregate
    # separate runs before it can truthfully claim PILOT/FULL tier status.
    tier_report = validate_profile_tier(
        ProfileTier(args.profile_tier), training_examples=len(train),
        validation_examples=len(validation), seeds=1, updates=args.steps,
    )
    (Path(args.output) / "profile_tier_validation.json").write_text(
        json.dumps(tier_report, indent=2) + "\n"
    )
    print(json.dumps({
        "profile_status": report.profile_status,
        "profile_digest": report.digest(),
        "ranking": report.ranking,
        "best_contiguous_region": report.best_contiguous_region,
        "middle_concentration_observed": report.middle_concentration_observed,
        "profile_tier": tier_report,
    }, indent=2))


if __name__ == "__main__":
    main()
