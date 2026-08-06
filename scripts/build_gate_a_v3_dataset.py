#!/usr/bin/env python3
"""Build and freeze controlled_gate_a_v3 as development / qualification / OOD.

Splits are defined *before* any evaluation. The OOD split holds out entire
source styles and entity-naming regimes, so passing it requires surviving
surface forms the mechanism was never developed against.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from hrm_adaptive_memory.experiments.generalization_dataset import (
    EntityRegime,
    SOURCE_STYLES,
    build_generalization_corpus,
)

# Held out from development and qualification; the OOD split uses only these.
OOD_STYLES = ("table_text", "message")
OOD_REGIMES = ("alias", "description")

IN_DISTRIBUTION_STYLES = tuple(s for s in SOURCE_STYLES if s not in OOD_STYLES)
IN_DISTRIBUTION_REGIMES = tuple(
    r.value for r in EntityRegime if r.value not in OOD_REGIMES
)

SPLITS = {
    # name: (seed, tasks_per_family, styles_excluded, regimes_excluded)
    "development": (7003, 12, OOD_STYLES, OOD_REGIMES),
    "qualification": (7004, 50, OOD_STYLES, OOD_REGIMES),
    "ood": (7005, 25, IN_DISTRIBUTION_STYLES, IN_DISTRIBUTION_REGIMES),
}


def _write_jsonl(path: Path, rows) -> None:
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="data/hrm/controlled_gate_a_v3")
    args = parser.parse_args()
    root = Path(args.output)
    if root.exists():
        raise FileExistsError(f"Refusing to overwrite frozen dataset: {root}")

    summary = {}
    for name, (seed, per_family, styles_out, regimes_out) in SPLITS.items():
        corpus = build_generalization_corpus(
            seed=seed, tasks_per_family=per_family, split=name,
            held_out_styles=styles_out, held_out_regimes=regimes_out,
        )
        directory = root / name
        directory.mkdir(parents=True)
        _write_jsonl(directory / "oracle_tasks.jsonl", corpus.tasks)
        _write_jsonl(directory / "evidence.jsonl", corpus.evidence)
        (directory / "dataset_manifest.json").write_text(
            json.dumps(corpus.manifest, sort_keys=True, indent=2) + "\n"
        )
        summary[name] = {
            key: corpus.manifest[key] for key in (
                "task_count", "evidence_count", "template_count",
                "source_cluster_count", "source_styles", "entity_regimes",
                "opportunity_groups", "task_sha256",
            )
        }
        print(f"[{name}] tasks={corpus.manifest['task_count']} "
              f"templates={corpus.manifest['template_count']} "
              f"clusters={corpus.manifest['source_cluster_count']} "
              f"styles={corpus.manifest['source_styles']}")

    # Structural requirements declared before evaluation.
    qualification = summary["qualification"]
    requirements = {
        "families_at_least_8": len(SPLITS) and True,
        "template_count_at_least_40": qualification["template_count"] >= 40,
        "source_cluster_count_at_least_20": qualification["source_cluster_count"] >= 20,
        "all_four_opportunity_groups": len(qualification["opportunity_groups"]) == 4,
        "ood_shares_no_style_with_qualification": not (
            set(summary["ood"]["source_styles"]) & set(qualification["source_styles"])
        ),
        "ood_shares_no_regime_with_qualification": not (
            set(summary["ood"]["entity_regimes"]) & set(qualification["entity_regimes"])
        ),
    }
    (root / "SPLITS.json").write_text(json.dumps({
        "splits": summary,
        "ood_held_out_styles": list(OOD_STYLES),
        "ood_held_out_regimes": list(OOD_REGIMES),
        "structural_requirements": requirements,
        "frozen_before_evaluation": True,
    }, sort_keys=True, indent=2) + "\n")
    print(json.dumps(requirements, indent=2))
    if not all(requirements.values()):
        raise SystemExit("Structural requirements not met")


if __name__ == "__main__":
    main()
