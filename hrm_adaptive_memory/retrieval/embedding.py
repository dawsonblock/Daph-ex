"""Pinned embedding backend for the Gate B dense arm.

Everything that can change a vector is pinned and hashed: model id, revision,
pooling, normalization, dimension, dtype, max sequence length, and the query
prefix. A run whose embedding config digest differs from the frozen protocol
is a different experiment and must be treated as such.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Any, Protocol, Sequence


class EmbeddingBackend(Protocol):
    embedding_id: str

    def embed_query(self, text: str) -> Sequence[float]: ...
    def embed_documents(self, texts: Sequence[str]) -> Sequence[Sequence[float]]: ...


BGE_SMALL = dict(
    model_id="BAAI/bge-small-en-v1.5", revision="5c38ec7c405ec4b44b94cc5a9bb96e735b38267a", pooling="cls",
    normalize=True, dimension=384, max_sequence_length=512,
    # BGE's documented retrieval recipe prefixes the QUERY only, never the
    # documents. Prefixing both would change the measured comparison.
    query_prefix="Represent this sentence for searching relevant passages: ",
)


@dataclass(frozen=True)
class EmbeddingSpec:
    model_id: str = "sentence-transformers/all-MiniLM-L6-v2"
    revision: str = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
    pooling: str = "mean"
    normalize: bool = True
    dimension: int = 384
    dtype: str = "float32"
    max_sequence_length: int = 256
    query_prefix: str = ""
    document_prefix: str = ""

    def digest(self) -> str:
        payload = json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode()).hexdigest()

    def validate_model(self, config: Any) -> None:
        observed = int(getattr(config, "hidden_size", -1))
        if observed != self.dimension:
            raise ValueError(
                f"Pinned embedding dimension mismatch: expected {self.dimension}, got {observed}"
            )


class PinnedTransformerEmbedder:
    """Revision-pinned encoder with explicit pooling and normalization.

    Pooling is implemented here rather than delegated to a framework default
    so the operation is part of the hashed configuration instead of an
    invisible library behavior that can change between versions.
    """

    def __init__(self, spec: EmbeddingSpec | None = None, *, batch_size: int = 64,
                 model: Any = None, tokenizer: Any = None):
        self.spec = spec or EmbeddingSpec()
        self.batch_size = batch_size
        self.embedding_id = f"dense:{self.spec.model_id}@{self.spec.revision[:8]}"
        if model is None or tokenizer is None:
            try:
                from transformers import AutoModel, AutoTokenizer
            except ImportError as exc:  # pragma: no cover - optional install
                raise RuntimeError("Install 'transformers' to use the dense arm") from exc
            tokenizer = tokenizer or AutoTokenizer.from_pretrained(
                self.spec.model_id, revision=self.spec.revision,
            )
            model = model or AutoModel.from_pretrained(
                self.spec.model_id, revision=self.spec.revision,
            ).eval()
        self.tokenizer = tokenizer
        self.model = model
        self.spec.validate_model(model.config)

    def _encode(self, texts: Sequence[str]) -> list[list[float]]:
        import torch

        vectors: list[list[float]] = []
        with torch.no_grad():
            for start in range(0, len(texts), self.batch_size):
                batch = list(texts[start:start + self.batch_size])
                encoded = self.tokenizer(
                    batch, padding=True, truncation=True,
                    max_length=self.spec.max_sequence_length, return_tensors="pt",
                )
                hidden = self.model(**encoded).last_hidden_state
                if self.spec.pooling == "mean":
                    mask = encoded["attention_mask"].unsqueeze(-1).to(hidden.dtype)
                    pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
                elif self.spec.pooling == "cls":
                    pooled = hidden[:, 0]
                else:
                    raise ValueError(f"Unsupported pooling {self.spec.pooling!r}")
                if self.spec.normalize:
                    pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
                vectors.extend(pooled.to(torch.float32).tolist())
        return vectors

    def embed_query(self, text: str) -> list[float]:
        return self._encode([f"{self.spec.query_prefix}{text}"])[0]

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return self._encode([f"{self.spec.document_prefix}{text}" for text in texts])
