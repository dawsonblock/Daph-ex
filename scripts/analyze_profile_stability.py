#!/usr/bin/env python3
"""Aggregate independent layer-profile seeds into promotable stability evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from daph.e3_protocol import ProfileTier, profile_stability, validate_profile_tier


def _read_profile(path: Path) -> tuple[int, dict[int, float], dict]:
    manifest = json.loads((path / "manifest.json").read_text())
    rows = [json.loads(line) for line in (path / "per_layer_results.jsonl").read_text().splitlines() if line.strip()]
    if not rows:
        raise ValueError(f"No per-layer results in {path}")
    seed = int(rows[0]["seed"])
    if any(int(row["seed"]) != seed for row in rows):
        raise ValueError(f"Profile directory mixes seeds: {path}")
    return seed, {int(row["layer_index"]): float(row["layer_contribution"]) for row in rows}, manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile-dirs", required=True, help="Comma-separated one-seed profile directories")
    parser.add_argument("--output", required=True)
    parser.add_argument("--profile-tier", choices=tuple(tier.value for tier in ProfileTier), required=True)
    parser.add_argument("--training-examples", type=int, required=True)
    parser.add_argument("--validation-examples", type=int, required=True)
    parser.add_argument("--updates", type=int, required=True)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--contiguous-width", type=int, default=3)
    args = parser.parse_args()
    directories = [Path(value) for value in args.profile_dirs.split(",") if value.strip()]
    if not directories:
        raise ValueError("--profile-dirs must name at least one directory")
    profiles, manifests = {}, []
    for directory in directories:
        seed, values, manifest = _read_profile(directory)
        if seed in profiles:
            raise ValueError(f"Duplicate profile seed: {seed}")
        profiles[seed] = values
        manifests.append(manifest)
    if len(profiles) < 2:
        raise ValueError("Profile stability requires at least two independent profile seeds")
    stability = profile_stability(profiles, top_k=args.top_k, contiguous_width=args.contiguous_width)
    tier = validate_profile_tier(
        ProfileTier(args.profile_tier), training_examples=args.training_examples,
        validation_examples=args.validation_examples, seeds=len(profiles), updates=args.updates,
    )
    promotion_passed = bool(tier["passed"] and stability["stable_for_promotion"])
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    mean_contributions = {
        int(layer): float(values["mean"])
        for layer, values in stability["contribution_by_layer"].items()
    }
    ranking = sorted(mean_contributions, key=lambda layer: (-mean_contributions[layer], layer))
    regions = []
    for start in sorted(mean_contributions):
        region = tuple(range(start, start + args.contiguous_width))
        if all(layer in mean_contributions for layer in region):
            regions.append((
                sum(mean_contributions[layer] for layer in region) / len(region), region,
            ))
    best_region = list(max(
        regions, key=lambda item: (item[0], tuple(-layer for layer in item[1])),
    )[1]) if regions else [ranking[0]]
    digest_payload = {
        "source_digests": [manifest.get("profile_digest") for manifest in manifests],
        "mean_contributions": mean_contributions,
        "ranking": ranking,
        "best_contiguous_region": best_region,
        "tier": tier,
        "stability": stability,
    }
    aggregate_digest = hashlib.sha256(
        json.dumps(digest_payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    report = {
        "profile_tier": tier,
        "profile_stability": stability,
        "profile_directories": [str(directory) for directory in directories],
        "profile_digests": [manifest.get("profile_digest") for manifest in manifests],
        "promotion_passed": promotion_passed,
    }
    (output / "profile_tier_validation.json").write_text(json.dumps({
        **tier,
        "profile_stability": stability,
        "promotion_passed": promotion_passed,
    }, indent=2) + "\n")
    (output / "profile_stability.json").write_text(json.dumps(report, indent=2) + "\n")
    (output / "rankings.json").write_text(json.dumps({
        "ranking": ranking,
        "best_contiguous_region": best_region,
        "mean_contribution_by_layer": {str(key): value for key, value in mean_contributions.items()},
    }, indent=2) + "\n")
    (output / "manifest.json").write_text(json.dumps({
        "profile_status": "AGGREGATED_PROFILE",
        "profile_digest": aggregate_digest,
        "profile_tier": tier["tier"],
        "profile_tier_passed": tier["passed"],
        "profile_stability_passed": stability["stable_for_promotion"],
        "promotion_passed": promotion_passed,
        "source_profile_directories": [str(directory) for directory in directories],
        "source_profile_digests": [manifest.get("profile_digest") for manifest in manifests],
    }, indent=2) + "\n")
    print(json.dumps({"promotion_passed": promotion_passed, **tier}, indent=2))


if __name__ == "__main__":
    main()
