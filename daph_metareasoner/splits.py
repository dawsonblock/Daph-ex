"""Leakage-resistant task split manifests."""

from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping

from .schema import Task, canonical_digest


ALLOWED_SPLITS = {"experience", "validation", "test", "ood"}


def load_tasks(path: str | Path, *, default_split: str | None = None) -> List[Task]:
    tasks = []
    for line_number, line in enumerate(Path(path).read_text().splitlines(), start=1):
        if not line.strip():
            continue
        row = json.loads(line)
        split = row.get("split", default_split)
        if split not in ALLOWED_SPLITS:
            raise ValueError(f"Invalid/missing split at {path}:{line_number}: {split!r}")
        tasks.append(Task(
            task_id=str(row["task_id"]),
            prompt=str(row["prompt"]),
            expected=str(row["expected"]),
            family_id=str(row.get("family_id", row.get("task_family", "unspecified"))),
            split=split,
            template_id=str(row.get("template_id", "unspecified")),
            generator_seed=str(row.get("generator_seed", "unspecified")),
            metadata=dict(row.get("metadata", {})),
        ))
    if not tasks:
        raise ValueError(f"No tasks in {path}")
    return tasks


def _prompt_fingerprint(prompt: str) -> str:
    normalized = re.sub(r"\d+", "<N>", prompt.lower())
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return canonical_digest(normalized)


def build_split_manifest(tasks: Iterable[Task]) -> Dict[str, Any]:
    rows = list(tasks)
    by_id: Dict[str, Task] = {}
    prompt_digests: Dict[str, str] = {}
    template_seed: Dict[tuple[str, str, str], str] = {}
    template_split: Dict[tuple[str, str], str] = {}
    seed_split: Dict[str, str] = {}
    fingerprint_split: Dict[str, str] = {}
    by_split: Dict[str, List[Task]] = defaultdict(list)
    for task in rows:
        if task.task_id in by_id:
            raise ValueError(f"Duplicate task_id across splits: {task.task_id}")
        by_id[task.task_id] = task
        prompt_digest = canonical_digest(task.prompt)
        if prompt_digest in prompt_digests:
            raise ValueError(f"Exact prompt leakage: {task.task_id} and {prompt_digests[prompt_digest]}")
        prompt_digests[prompt_digest] = task.task_id
        key = (task.family_id, task.template_id, task.generator_seed)
        if task.template_id != "unspecified" and task.generator_seed != "unspecified":
            if key in template_seed:
                raise ValueError(f"Template/seed leakage: {task.task_id} and {template_seed[key]}")
            template_seed[key] = task.task_id
        template_key = (task.family_id, task.template_id)
        previous_template_split = template_split.get(template_key)
        if previous_template_split is not None and previous_template_split != task.split:
            raise ValueError(f"Template leakage across splits: {template_key}")
        template_split[template_key] = task.split
        previous_seed_split = seed_split.get(task.generator_seed)
        if previous_seed_split is not None and previous_seed_split != task.split:
            raise ValueError(f"Generator-seed leakage across splits: {task.generator_seed}")
        seed_split[task.generator_seed] = task.split
        fingerprint = _prompt_fingerprint(task.prompt)
        previous_fingerprint_split = fingerprint_split.get(fingerprint)
        if previous_fingerprint_split is not None and previous_fingerprint_split != task.split:
            raise ValueError("Prompt-template fingerprint leakage across splits")
        fingerprint_split[fingerprint] = task.split
        by_split[task.split].append(task)
    train_families = {
        task.family_id for split in ("experience", "validation") for task in by_split.get(split, [])
    }
    ood_families = {task.family_id for task in by_split.get("ood", [])}
    overlap = train_families & ood_families
    if overlap:
        raise ValueError(f"OOD family leakage: {sorted(overlap)}")
    manifest = {
        "splits": {
            split: {
                "tasks": len(items),
                "task_ids": sorted(task.task_id for task in items),
                "families": sorted({task.family_id for task in items}),
                "prompt_template_fingerprints": sorted({_prompt_fingerprint(task.prompt) for task in items}),
            }
            for split, items in sorted(by_split.items())
        },
        "tasks_digest": canonical_digest([task.digest() for task in sorted(rows, key=lambda row: row.task_id)]),
        "immutable_splits": ["test", "ood"],
    }
    manifest["manifest_digest"] = canonical_digest(manifest)
    return manifest
