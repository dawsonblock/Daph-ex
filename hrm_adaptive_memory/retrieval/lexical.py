from __future__ import annotations

import math
import re
from collections import Counter
from typing import Sequence

from hrm_adaptive_memory.memory.chunking import Chunk


# Internal '.', '/', and '-' are kept ("Plan-000-965", "v1.2", "src/main"), but
# a trailing separator is not part of the token. Without the inner-boundary
# requirement, a sentence-final entity tokenizes as "plan-000-965." and never
# matches the same entity written mid-sentence, which silently hides evidence
# from every lexical query.
_TOKEN = re.compile(r"[A-Za-z0-9_]+(?:[./-][A-Za-z0-9_]+)*")


def tokenize(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


class BM25Retriever:
    def __init__(self, chunks: Sequence[Chunk], *, k1: float = 1.5, b: float = 0.75):
        self.chunks = list(chunks); self.k1 = k1; self.b = b
        self.documents = [Counter(tokenize(chunk.content)) for chunk in self.chunks]
        self.lengths = [sum(doc.values()) for doc in self.documents]
        self.average_length = sum(self.lengths) / max(1, len(self.lengths))
        self.df = Counter(token for doc in self.documents for token in doc)

    def search(self, query: str, top_k: int = 30) -> list[tuple[Chunk, float]]:
        terms = tokenize(query); total = len(self.documents)
        scored = []
        for chunk, doc, length in zip(self.chunks, self.documents, self.lengths):
            score = 0.0
            for term in terms:
                frequency = doc.get(term, 0)
                if not frequency: continue
                idf = math.log(1.0 + (total - self.df[term] + 0.5) / (self.df[term] + 0.5))
                norm = frequency + self.k1 * (1 - self.b + self.b * length / max(self.average_length, 1))
                score += idf * frequency * (self.k1 + 1) / norm
            if score > 0: scored.append((chunk, score))
        return sorted(scored, key=lambda item: (-item[1], item[0].chunk_id))[:top_k]
