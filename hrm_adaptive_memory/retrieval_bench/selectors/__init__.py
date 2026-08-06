"""Selector arms. Every arm consumes the identical frozen candidate pool.

S0 raw ranking · S1 pointwise relevance · S2 relation/connectivity
S3 lightweight cross-encoder · S4 stronger cross-encoder · S5 oracle ceiling

Nothing upstream may vary: retrieval, chain completion, candidate count, query
generation, canonicalization, the reader, and the pool contents are all fixed.
Only selection differs, which is what makes the comparison attributable.
"""
from __future__ import annotations
import re
from typing import Any, Mapping, Sequence

from ...retrieval.canonicalization import extract_identity_links
from ...retrieval.lexical import tokenize


def _norm(t: str) -> str:
    return " ".join(re.findall(r"\w+", t.lower()))


def s0_raw(candidates: Sequence[Mapping[str, Any]], *, budget: int, **_) -> list[str]:
    """Frozen pool order, truncated. The baseline packet."""
    return [c["document_id"] for c in candidates[:budget]]


def s1_relevance(candidates, *, budget: int, question: str, texts, **_) -> list[str]:
    """Pointwise lexical relevance. Can plain relevance recover discarded gold?"""
    q = set(tokenize(question))
    scored = []
    for c in candidates:
        terms = set(tokenize(texts.get(c["document_id"], "")))
        overlap = len(q & terms) / max(1, len(q))
        scored.append((overlap, -c["rank"], c["document_id"]))
    scored.sort(reverse=True)
    return [d for _, _, d in scored[:budget]]


def s2_connectivity(candidates, *, budget: int, question: str, texts,
                    target_relation: str | None = None, **_) -> list[str]:
    """Score the CHAIN, not the document.

    Mirrors the structure already qualified upstream:
        query -> identity candidate -> canonical entity -> relation -> answer
    A candidate earns weight for naming a question entity, for resolving an
    identity referenced by the question, for stating the target relation, and
    for connecting to entities already selected.
    """
    ids = [c["document_id"] for c in candidates]
    text = {d: texts.get(d, "") for d in ids}
    qn = _norm(question)
    rel = _norm(target_relation or "")

    class _R:
        def __init__(self, d, t): self.evidence_id, self.content = d, t
    links = {l.record_id: l for l in extract_identity_links([_R(d, text[d]) for d in ids])}
    # Identity records whose surface the question actually mentions are the
    # entry point of the chain; the canonical name they yield anchors the rest.
    anchors: set[str] = set()
    seeds: list[str] = []
    for d in ids:
        link = links.get(d)
        if link and _norm(link.surface) in qn:
            seeds.append(d); anchors.add(_norm(link.canonical))
    for d in ids:
        if any(w in _norm(text[d]) for w in qn.split() if len(w) > 4):
            anchors.update(_norm(text[d]).split())

    chosen: list[str] = []
    selected_terms: set[str] = set()
    for d in seeds[:budget]:
        chosen.append(d); selected_terms |= set(_norm(text[d]).split())

    remaining = [d for d in ids if d not in chosen]
    while len(chosen) < budget and remaining:
        best, best_score = None, -1e9
        for d in remaining:
            t = _norm(text[d]); terms = set(t.split())
            score = 0.0
            if rel and rel in t: score += 3.0          # states the target relation
            score += 2.0 * len(terms & anchors) / max(1, len(terms))   # chain anchor
            score += 1.5 * len(terms & selected_terms) / max(1, len(terms))  # connectivity
            score -= 1.0 * (len(terms & selected_terms) / max(1, len(terms)) > 0.85)  # redundancy
            score += 0.5 / (1 + ids.index(d))          # mild rank prior
            if score > best_score: best, best_score = d, score
        chosen.append(best); selected_terms |= set(_norm(text[best]).split())
        remaining.remove(best)
    return chosen[:budget]


# Honest name for what this arm actually is. It scores target-relation presence,
# not connectivity, so `s2_connectivity` was a misnomer. It is retained as a
# valid diagnostic control rather than discarded: on descv4_surface, where
# entity anchoring fails and the structural arms go inert, it is the only arm
# with a significant gain (+0.170), because relation scoring needs no entity
# anchor. It remains disqualified as a chain selector -- BridgeRetention 0.000
# on descv4_id, which is forced, since 0/56 bridge records state the target
# relation there.
#
# The historical arm id `S2_connectivity` is left untouched in frozen receipts;
# renaming a measured arm id would break reproducibility of existing evidence.
s_rel_only = s2_connectivity


class DegenerateRerankerError(RuntimeError):
    """A reranker returned scores that cannot express an ordering.

    This exists because of a real measurement failure. An earlier S3 arm used
    cross-encoder/ms-marco-MiniLM-L6-v2, whose forward pass returns NaN for
    every pair under torch 2.10 / transformers 5.14.1 (finite fp32 weights,
    absmax 4.7, NaN from the first attention layer onward). `sorted` compares
    NaN keys as all-False, so the pool came back in its original order and the
    arm silently reported figures identical to S0 to four decimals. A broken
    scorer must fail, never impersonate the baseline.
    """


def make_cross_encoder_selector(model_id: str, revision: str | None = None,
                                max_length: int = 512):
    """S3/S4: rerank the frozen pool with a pinned cross-encoder."""
    import math

    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    kw = {"revision": revision} if revision else {}
    tok = AutoTokenizer.from_pretrained(model_id, **kw)
    mdl = AutoModelForSequenceClassification.from_pretrained(model_id, **kw).eval()

    def _audit(scores: list[float], count: int) -> None:
        bad = sum(1 for s in scores if not math.isfinite(s))
        if bad:
            raise DegenerateRerankerError(
                f"{model_id} returned {bad}/{len(scores)} non-finite scores; "
                "refusing to emit an order that would duplicate the raw pool"
            )
        if count > 1 and len(set(scores)) == 1:
            raise DegenerateRerankerError(
                f"{model_id} scored all {count} candidates identically "
                f"({scores[0]}); this cannot rerank and would duplicate S0"
            )

    def selector(candidates, *, budget: int, question: str, texts, **_) -> list[str]:
        ids = [c["document_id"] for c in candidates]
        pairs = [(question, texts.get(d, "")) for d in ids]
        scores: list[float] = []
        with torch.no_grad():
            for start in range(0, len(pairs), 32):
                chunk = pairs[start:start + 32]
                enc = tok([p[0] for p in chunk], [p[1] for p in chunk], padding=True,
                          truncation=True, max_length=max_length, return_tensors="pt")
                logits = mdl(**enc).logits
                scores.extend((logits[:, -1] if logits.shape[-1] > 1 else logits[:, 0]).tolist())
        _audit(scores, len(ids))
        order = sorted(range(len(ids)), key=lambda i: (-scores[i], i))
        return [ids[i] for i in order[:budget]]

    selector.model_id = model_id
    selector.revision = revision
    return selector


def s5_oracle(candidates, *, budget: int, required: Sequence[str], **_) -> list[str]:
    """CEILING. Oracle selection restricted to what retrieval actually found."""
    ids = [c["document_id"] for c in candidates]
    return [d for d in ids if d in set(required)][:budget]
