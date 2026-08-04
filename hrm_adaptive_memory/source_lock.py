"""Validated provenance registry for replaceable external systems."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


@dataclass(frozen=True)
class LockedSource:
    name: str
    upstream: str
    archive_sha256: str
    observed_version: str
    license: str
    role: str
    runtime_enabled: bool
    runtime_digest: str | None = None
    source_revision: str | None = None


class SourceLock:
    def __init__(self, payload: Mapping[str, Any]):
        if payload.get("schema_version") != "1.0":
            raise ValueError("Unsupported source-lock schema")
        self.bundle = dict(payload["source_bundle"])
        self.sources = {
            str(row["name"]): LockedSource(**row)
            for row in payload["sources"]
        }
        if len(self.sources) != len(payload["sources"]):
            raise ValueError("Duplicate names in source lock")
        for source in self.sources.values():
            if len(source.archive_sha256) != 64:
                raise ValueError(f"Invalid archive digest for {source.name}")
            if source.source_revision is not None and len(source.source_revision) < 7:
                raise ValueError(f"Invalid source revision for {source.name}")
            if source.runtime_enabled and not source.runtime_digest:
                raise ValueError(f"Enabled runtime {source.name} needs an immutable runtime digest")

    @classmethod
    def load(cls, path: str | Path) -> "SourceLock":
        return cls(json.loads(Path(path).read_text()))

    def require_runtime(self, name: str) -> LockedSource:
        try:
            source = self.sources[name]
        except KeyError as exc:
            raise KeyError(f"Source {name!r} is not locked") from exc
        if not source.runtime_enabled:
            raise RuntimeError(f"{name} runtime remains gate-blocked in the source lock")
        return source
