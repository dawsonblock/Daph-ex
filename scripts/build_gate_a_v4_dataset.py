#!/usr/bin/env python3
"""Build, audit, and freeze controlled_gate_a_v4 — fail-closed at every stage.

    generate → structural audit → inferability audit → leakage audit
    → oracle-isolation audit → retrievability audit → full pytest
    → hashes → freeze

No later stage runs if an earlier one fails, and nothing is written to the
dataset directory until every audit has passed. VALID_V4 is the conjunction:

    split isolation ∧ inferability ∧ oracle independence
                    ∧ no leakage ∧ structural diversity
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from hrm_adaptive_memory.experiments.generalization_dataset_v4 import (
    ANSWER_KINDS,
    FAMILIES,
    ITERATIVE_FAMILIES,
    OpportunityGroup,
    _ANSWER_BEARING_KINDS,
    _leaks,
    build_v4_corpus,
    verify_inferable,
)

# Splits are declared here, before any evaluation.
OOD_STYLES = ("table_text", "message")
OOD_REGIMES = ("alias", "description")
ID_STYLES = ("formal_registry", "technical_note", "key_value_log", "change_log")
ID_REGIMES = ("canonical", "abbreviation")

SPLITS = {
    "development": dict(seed=9101, tasks_per_family=12, styles=ID_STYLES, regimes=ID_REGIMES),
    "qualification": dict(seed=9102, tasks_per_family=50, styles=ID_STYLES, regimes=ID_REGIMES),
    "ood": dict(seed=9103, tasks_per_family=25, styles=OOD_STYLES, regimes=OOD_REGIMES),
}

RUNTIME_FORBIDDEN = re.compile(r"#(subject|bridge|value|decoy|near)\b|entity_\d+|value_\d+")


def _norm(text: str) -> str:
    return " ".join(re.findall(r"\w+", text.lower()))


def audit(corpora: dict) -> dict:
    """Every audit that must pass before a single byte is written."""

    problems: list[str] = []
    results: dict = {}

    # --- structural diversity -------------------------------------------------
    q_tasks, _ = corpora["qualification"]
    structural = {
        "families": len({t["family"] for t in q_tasks}),
        "iterative_families": len({t["family"] for t in q_tasks
                                   if t["metadata"]["iterative_family"]}),
        "templates": len({t["template_id"] for t in q_tasks}),
        "source_clusters": len({t["source_cluster_id"] for t in q_tasks}),
        "opportunity_groups": len({t["metadata"]["opportunity_group"] for t in q_tasks}),
        "answer_kinds": len({t["metadata"]["answer_kind"] for t in q_tasks}),
    }
    results["structural_diversity"] = structural
    for name, minimum in (("families", 8), ("iterative_families", 5), ("templates", 40),
                          ("source_clusters", 20), ("opportunity_groups", 4),
                          ("answer_kinds", 4)):
        if structural[name] < minimum:
            problems.append(f"structural: {name}={structural[name]} < {minimum}")

    # --- split isolation, checked on EVIDENCE RECORDS -------------------------
    # v3's test compared task labels and passed while the claim was false.
    styles = {}
    regimes = {}
    for split, (tasks, evidence) in corpora.items():
        styles[split] = {row["metadata"]["source_style"] for row in evidence}
        regimes[split] = {t["metadata"]["entity_regime"] for t in tasks}
    style_overlap = sorted(styles["qualification"] & styles["ood"])
    regime_overlap = sorted(regimes["qualification"] & regimes["ood"])
    results["split_isolation"] = {
        "qualification_evidence_styles": sorted(styles["qualification"]),
        "ood_evidence_styles": sorted(styles["ood"]),
        "style_overlap": style_overlap, "regime_overlap": regime_overlap,
    }
    if style_overlap:
        problems.append(f"split isolation: evidence styles overlap {style_overlap}")
    if regime_overlap:
        problems.append(f"split isolation: entity regimes overlap {regime_overlap}")

    # --- inferability ---------------------------------------------------------
    inferability = {}
    for split, (tasks, evidence) in corpora.items():
        by_id = {row["evidence_id"]: row for row in evidence}
        failures = {t["task_id"]: verify_inferable(t, by_id) for t in tasks}
        failures = {k: v for k, v in failures.items() if v}
        inferability[split] = {"tasks": len(tasks), "not_inferable": len(failures)}
        if failures:
            problems.append(
                f"inferability: {split} has {len(failures)} non-inferable tasks "
                f"(sample {list(failures)[:3]})")
    results["inferability"] = inferability

    # --- retrievability sanity (before any GPU time) -------------------------
    # Proves there is something to find: required records exist, and every
    # identity transition has an explicit textual realisation.
    retrievability = {}
    for split, (tasks, evidence) in corpora.items():
        by_id = {row["evidence_id"]: row for row in evidence}
        missing_records = 0
        unrealised_transitions = 0
        subject_unfindable = 0
        for task in tasks:
            for value in task["required_evidence_ids"]:
                if value not in by_id:
                    missing_records += 1
            corpus_text = " || ".join(
                _norm(by_id[v]["content"]) for v in task["required_evidence_ids"] if v in by_id)
            if _norm(task["_oracle_metadata"]["surfaces"]["subject"]) not in corpus_text:
                subject_unfindable += 1
            for edge in task["_oracle_metadata"]["proof_edges"]:
                if not edge["source"].startswith("surface:"):
                    continue
                record = by_id.get(edge["record_id"])
                phrase = _norm(edge["source"].split("surface:", 1)[1])
                if record is None or phrase not in _norm(record["content"]):
                    unrealised_transitions += 1
        retrievability[split] = {
            "missing_required_records": missing_records,
            "unrealised_identity_transitions": unrealised_transitions,
            "subject_not_findable_in_required": subject_unfindable,
        }
        for key, count in retrievability[split].items():
            if count:
                problems.append(f"retrievability: {split} {key}={count}")
    results["retrievability"] = retrievability

    # --- leakage --------------------------------------------------------------
    leakage = {}
    for split, (tasks, evidence) in corpora.items():
        by_task_prefix: dict[str, list] = {}
        for row in evidence:
            by_task_prefix.setdefault(row["evidence_id"].split("/", 1)[0], []).append(row)
        question_leaks = sum(1 for t in tasks if _leaks(t["question"], t["answer"]))
        distractor_leaks = sum(
            1 for t in tasks for row in by_task_prefix.get(t["task_id"], [])
            if row["metadata"]["record_kind"] not in _ANSWER_BEARING_KINDS
            and _leaks(row["content"], t["answer"]))
        leakage[split] = {"question_leaks": question_leaks,
                          "distractor_leaks": distractor_leaks}
        if question_leaks or distractor_leaks:
            problems.append(f"leakage: {split} {leakage[split]}")
    results["leakage"] = leakage

    # --- oracle isolation ----------------------------------------------------
    # Latent identifiers must never appear in anything the runtime can read.
    isolation = {}
    for split, (tasks, evidence) in corpora.items():
        runtime_hits = [row["evidence_id"] for row in evidence
                        if RUNTIME_FORBIDDEN.search(row["content"])]
        question_hits = [t["task_id"] for t in tasks
                         if RUNTIME_FORBIDDEN.search(t["question"])]
        answer_hits = [t["task_id"] for t in tasks
                       if RUNTIME_FORBIDDEN.search(t["answer"])]
        has_metadata = all("_oracle_metadata" in t for t in tasks)
        isolation[split] = {
            "latent_ids_in_evidence": len(runtime_hits),
            "latent_ids_in_questions": len(question_hits),
            "latent_ids_in_answers": len(answer_hits),
            "every_task_has_proof_graph": has_metadata,
        }
        if runtime_hits or question_hits or answer_hits:
            problems.append(f"oracle isolation: {split} leaks latent ids")
        if not has_metadata:
            problems.append(f"oracle isolation: {split} missing proof graphs")
    results["oracle_isolation"] = isolation

    results["problems"] = problems
    results["VALID_V4"] = not problems
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="data/hrm/controlled_gate_a_v4")
    parser.add_argument("--skip-pytest", action="store_true")
    args = parser.parse_args()
    root = Path(args.output)
    if root.exists():
        raise FileExistsError(f"Refusing to overwrite a frozen dataset: {root}")

    # Stage 1: generate in memory only.
    print("[1/6] generating")
    corpora = {}
    manifests = {}
    for name, config in SPLITS.items():
        corpus = build_v4_corpus(split=name, **config)
        corpora[name] = (list(corpus.tasks), list(corpus.evidence))
        manifests[name] = dict(corpus.manifest)
        print(f"      {name}: {corpus.manifest['task_count']} tasks, "
              f"{corpus.manifest['evidence_count']} records, "
              f"styles={corpus.manifest['evidence_record_styles']}")

    # Stages 2-5: audits. Nothing is written unless all pass.
    print("[2/6] auditing (structural, isolation, inferability, retrievability, leakage, oracle)")
    report = audit(corpora)
    for line in report["problems"]:
        print(f"      FAIL {line}")
    if not report["VALID_V4"]:
        raise SystemExit("VALID_V4 is false; refusing to freeze. No files written.")
    print("      VALID_V4 = true")

    # Stage 6: full test suite must be green before freezing.
    if not args.skip_pytest:
        print("[3/6] running full test suite")
        result = subprocess.run([sys.executable, "-m", "pytest", "-q"], cwd=ROOT,
                                capture_output=True, text=True)
        if result.returncode != 0:
            print(result.stdout[-2000:])
            raise SystemExit("pytest failed; refusing to freeze. No files written.")
        print(f"      {result.stdout.strip().splitlines()[-1]}")

    print("[4/6] writing")
    root.mkdir(parents=True)
    for name, (tasks, evidence) in corpora.items():
        directory = root / name
        directory.mkdir()
        (directory / "oracle_tasks.jsonl").write_text(
            "".join(json.dumps(r, sort_keys=True) + "\n" for r in tasks))
        (directory / "evidence.jsonl").write_text(
            "".join(json.dumps(r, sort_keys=True) + "\n" for r in evidence))
        (directory / "dataset_manifest.json").write_text(
            json.dumps(manifests[name], sort_keys=True, indent=2) + "\n")

    print("[5/6] hashing")
    digests = {}
    for path in sorted(root.rglob("*")):
        if path.is_file():
            digests[str(path.relative_to(root))] = hashlib.sha256(
                path.read_bytes()).hexdigest()
    (root / "AUDIT.json").write_text(json.dumps({
        "generator": "controlled-gate-a-v4",
        "splits": {k: {key: manifests[k][key] for key in
                       ("task_count", "evidence_count", "template_count",
                        "source_cluster_count", "allowed_source_styles",
                        "evidence_record_styles", "entity_regimes")}
                   for k in manifests},
        "audit": report,
        "frozen_before_evaluation": True,
    }, sort_keys=True, indent=2) + "\n")
    (root / "RECEIPTS.sha256").write_text(
        "".join(f"{value}  {key}\n" for key, value in sorted(digests.items())))

    print("[6/6] frozen")
    print(json.dumps({k: v for k, v in report.items() if k != "problems"}, indent=2)[:1400])


if __name__ == "__main__":
    main()
