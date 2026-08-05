#!/usr/bin/env python3
"""Section-9.3 smoke assertions over a context-study evidence directory.

Validates, for every task:
  * receipts exist for every expected arm (failures must be explicit, not dropped);
  * B0 prompt digest != B3 prompt digest whenever oracle evidence exists;
  * B1 and B1b evidence tokens exactly match B3;
  * the gold answer token sequence never appears in B1/B1b prompts outside the
    question, and never appears at all when the question is answer-free;
  * no arm-identity string leaks into any model-visible prompt.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from hrm_adaptive_memory.experiments.context_study import StudyCondition, _normalize

ARM_LEAK_STRINGS = (
    "B0_NO_CONTEXT", "B1_RANDOM_CONTEXT", "B1B_HARD_DISTRACTOR",
    "B2_NAIVE_RETRIEVAL", "B3_ORACLE_EVIDENCE",
    "oracle", "random context", "hard distractor", "automatic retrieval",
    "experimental condition", "#b1:", "#b1b:", "evidence_id=", "source_id=",
    "[CONTEXT CONDITION]", "CAPABILITY_USE", "EVIDENCE_GROUNDED",
)


def _terms(text: str) -> tuple[str, ...]:
    return tuple(re.findall(r"\w+", _normalize(text)))


def _contains_sequence(haystack: tuple[str, ...], needle: tuple[str, ...]) -> bool:
    width = len(needle)
    return bool(width) and any(
        haystack[index:index + width] == needle
        for index in range(len(haystack) - width + 1)
    )


def check(evidence_dir: Path, tasks_path: Path) -> dict:
    receipts = [json.loads(line) for line in (evidence_dir / "per_task_results.jsonl").read_text().splitlines() if line.strip()]
    tasks = {row["task_id"]: row for row in (
        json.loads(line) for line in tasks_path.read_text().splitlines() if line.strip()
    )}
    manifest = json.loads((evidence_dir / "manifest.json").read_text())
    expected_arms = {value.value for value in StudyCondition} if manifest["hard_distractor_control"] else {
        value.value for value in StudyCondition if value != StudyCondition.B1_HARD_DISTRACTOR
    }

    by_task: dict[str, dict[str, dict]] = defaultdict(dict)
    for row in receipts:
        by_task[row["task_id"]][row["condition"]] = row

    problems: list[str] = []
    if set(by_task) != set(tasks):
        problems.append(f"task coverage mismatch: {sorted(set(tasks) ^ set(by_task))}")
    for task_id, arms in sorted(by_task.items()):
        task = tasks.get(task_id)
        if task is None:
            problems.append(f"{task_id}: receipt for unknown task")
            continue
        if set(arms) != expected_arms:
            problems.append(f"{task_id}: arms {sorted(arms)} != expected {sorted(expected_arms)}")
            continue
        b0, b3 = arms["B0_NO_CONTEXT"], arms["B3_ORACLE_EVIDENCE"]
        b1 = arms["B1_RANDOM_CONTEXT"]
        if task["oracle_evidence_ids"] and b0["final_prompt_sha256"] == b3["final_prompt_sha256"]:
            problems.append(f"{task_id}: B0 and B3 prompt digests identical")
        if b1["evidence_tokens"] != b3["evidence_tokens"]:
            problems.append(f"{task_id}: B1 evidence tokens {b1['evidence_tokens']} != B3 {b3['evidence_tokens']}")
        controls = [("B1", b1)]
        if "B1B_HARD_DISTRACTOR" in arms:
            b1b = arms["B1B_HARD_DISTRACTOR"]
            controls.append(("B1b", b1b))
            if b1b["evidence_tokens"] != b3["evidence_tokens"]:
                problems.append(f"{task_id}: B1b evidence tokens {b1b['evidence_tokens']} != B3 {b3['evidence_tokens']}")
        answer_terms = _terms(task["answer"])
        for label, row in controls:
            if _contains_sequence(_terms(row["final_prompt"]), answer_terms):
                problems.append(f"{task_id}: gold answer appears in {label} prompt")
        for condition, row in arms.items():
            prompt = row["final_prompt"]
            for leak in ARM_LEAK_STRINGS:
                if leak in prompt:
                    problems.append(f"{task_id}/{condition}: arm-identity string {leak!r} in prompt")

    report = {
        "report_type": "context_study_receipt_check",
        "evidence_dir": str(evidence_dir),
        "task_count": len(tasks),
        "receipt_count": len(receipts),
        "expected_arms": sorted(expected_arms),
        "problem_count": len(problems),
        "problems": problems,
        "passed": not problems,
    }
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evidence-dir", required=True)
    parser.add_argument("--tasks", required=True)
    parser.add_argument("--output")
    args = parser.parse_args()
    report = check(Path(args.evidence_dir), Path(args.tasks))
    if args.output:
        Path(args.output).write_text(json.dumps(report, sort_keys=True, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if not report["passed"]:
        raise SystemExit("RECEIPT CHECK FAILED")


if __name__ == "__main__":
    main()
