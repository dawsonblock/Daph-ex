"""Immutable provider-neutral derivation cache."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Protocol, Sequence

from .contracts import DerivationReceipt, sha256_text


@dataclass(frozen=True)
class Derivation:
    output: str
    receipt: DerivationReceipt


class DerivationProvider(Protocol):
    provider_id: str
    model_revision: str

    async def derive(self, prompt: str, sources: Sequence[str]) -> str: ...


class CachedDerivationStore:
    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def key(
        *, provider: str, model_revision: str, prompt: str, sources: Sequence[str],
    ) -> str:
        payload = "\0".join((provider, model_revision, sha256_text(prompt), *(sha256_text(v) for v in sources)))
        return sha256_text(payload)

    def get(self, cache_key: str) -> Derivation | None:
        path = self.root / f"{cache_key}.json"
        if not path.exists():
            return None
        row = json.loads(path.read_text())
        receipt = DerivationReceipt(**{
            **row["receipt"],
            "source_sha256": tuple(row["receipt"]["source_sha256"]),
        })
        output = str(row["output"])
        if sha256_text(output) != receipt.output_sha256 or receipt.cache_key != cache_key:
            raise RuntimeError("Cached derivation integrity failure")
        return Derivation(output, receipt)

    def put(self, derivation: Derivation) -> None:
        key = derivation.receipt.cache_key
        if sha256_text(derivation.output) != derivation.receipt.output_sha256:
            raise ValueError("Derivation output digest mismatch")
        path = self.root / f"{key}.json"
        payload = json.dumps({"output": derivation.output, "receipt": asdict(derivation.receipt)}, sort_keys=True, indent=2) + "\n"
        if path.exists() and path.read_text() != payload:
            raise FileExistsError("Immutable derivation cache entry already exists")
        path.write_text(payload)


async def derive_cached(
    provider: DerivationProvider, cache: CachedDerivationStore, *, prompt: str,
    sources: Sequence[str], verifier: str, verified: bool,
) -> Derivation:
    key = cache.key(
        provider=provider.provider_id,
        model_revision=provider.model_revision,
        prompt=prompt,
        sources=sources,
    )
    existing = cache.get(key)
    if existing is not None:
        return existing
    output = await provider.derive(prompt, sources)
    receipt = DerivationReceipt(
        provider=provider.provider_id,
        model_revision=provider.model_revision,
        prompt_sha256=sha256_text(prompt),
        source_sha256=tuple(sha256_text(value) for value in sources),
        output_sha256=sha256_text(output),
        verifier=verifier,
        verified=verified,
    )
    derivation = Derivation(output, receipt)
    cache.put(derivation)
    return derivation
