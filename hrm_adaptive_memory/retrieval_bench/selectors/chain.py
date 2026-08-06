"""Structural selectors that score evidence CHAINS, not isolated documents.

The motivating measurement: S1 (lexical relevance), S3 and S4 (cross-encoders)
all score each candidate independently, and all three degrade the packet. S1
collapses identity retention from 1.00 to 0.14. The reason is structural rather
than a tuning failure:

    A bridge record can be individually weakly relevant to the question and
    still be essential, because it connects two highly relevant records.

An independent scorer cannot represent that. These selectors build a task-local
graph over the already-retrieved pool and score paths through it.

Everything here is RUNTIME-VISIBLE ONLY: the question text, the candidate
record text, and deterministic parsing. `_oracle_metadata` is never read. The
proof graph stays evaluator-only, for scoring after the fact.

The entity vocabulary is *derived from the pool*, not hardcoded, so the selector
does not encode corpus priors: a trailing noun phrase counts as an entity type
only when it appears with at least two distinct capitalized heads. That rejects
prose openers like "The survey lists" (whose tail follows one head) while
accepting "cable drum" (Shasta / Denali / Pobeda cable drum).
"""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from ...retrieval.canonicalization import extract_identity_links

# A capitalized head followed by one or two lowercase tokens.
_CANDIDATE_MENTION = re.compile(r"\b([A-Z][a-zA-Z]+)((?:\s+[a-z]+){1,2})\b")
# Coded identifiers: SCD-24, KAC-91, PSI2-CEDAR, ALPHA3-DUNE.
_IDENTIFIER = re.compile(r"\b([A-Z][A-Z0-9]{1,}(?:-[A-Z0-9]+)+)\b")
# "Which custody band is held by X?" / "What operating district does X carry?"
_RELATION_IN_QUESTION = re.compile(
    r"^(?:which|what)\s+([a-z]+(?:\s+[a-z]+){0,2}?)\s+(?:is|are|was|does|do)\b")

_MIN_HEADS_PER_TYPE = 2
_MIN_RECORDS_PER_PHRASE = 2

# Closed-class English function words cannot head an entity name. This is a
# grammatical fact, not a corpus prior, and it rejects prose openers such as
# "The survey lists", "Under survey", and "From the field" whose surface shape
# is otherwise identical to "Shasta cable drum".
_FUNCTION_HEADS = frozenset("""
    the a an this that these those his her its their our your my
    from under with for after before during per via at by in on of to into onto
    and but or nor so yet if when while because although though since unless
    is are was were be been being has have had do does did will would can could
    it he she they we you i there here what which who whom whose how why where
""".split())


def _norm(text: str) -> str:
    return " ".join(re.findall(r"\w+", text.lower()))


def parse_target_relation(question: str) -> str | None:
    """Read the asked-for relation out of the QUESTION.

    The relation is legitimate runtime information -- it is stated in the
    question -- so using it is not a leak. The defect in the earlier
    relation-keyword arm was letting relation overlap dominate selection, not
    using the relation at all.
    """
    match = _RELATION_IN_QUESTION.match(question.strip().lower())
    return match.group(1).strip() if match else None


def derive_entity_types(texts: Sequence[str]) -> set[str]:
    """Infer which trailing noun phrases denote entity types, from the pool.

    Two independent kinds of evidence qualify a tail, because either alone is
    too weak on a small pool:

      cross-head  the tail appears with >= 2 distinct capitalized heads
                  ("Shasta / Denali / Pobeda cable drum")
      recurrence  the same full phrase appears in >= 2 distinct records, which
                  is what makes an entity usable as a join key at all

    Function-word heads are excluded first, so a recurring prose opener cannot
    be promoted into an entity type by the recurrence rule.
    """
    heads_per_tail: dict[str, set[str]] = defaultdict(set)
    records_per_phrase: dict[str, set[int]] = defaultdict(set)
    for index, text in enumerate(texts):
        for head, tail in _CANDIDATE_MENTION.findall(text):
            if head.lower() in _FUNCTION_HEADS:
                continue
            tail_norm = " ".join(tail.split())
            heads_per_tail[tail_norm].add(head)
            records_per_phrase[f"{head} {tail_norm}"].add(index)
    qualified = {tail for tail, heads in heads_per_tail.items()
                 if len(heads) >= _MIN_HEADS_PER_TYPE}
    for phrase, records in records_per_phrase.items():
        if len(records) >= _MIN_RECORDS_PER_PHRASE:
            qualified.add(phrase.split(" ", 1)[1])
    return qualified


def extract_mentions(text: str, entity_types: set[str]) -> set[str]:
    """Entity mentions plus coded identifiers, normalized."""
    found: set[str] = set()
    for head, tail in _CANDIDATE_MENTION.findall(text):
        if head.lower() in _FUNCTION_HEADS:
            continue
        tail_norm = " ".join(tail.split())
        if tail_norm in entity_types:
            found.add(_norm(f"{head} {tail_norm}"))
    for identifier in _IDENTIFIER.findall(text):
        found.add(_norm(identifier))
    return found


@dataclass
class TaskGraph:
    """Task-local graph over the retrieved pool. Runtime-visible data only."""

    record_ids: list[str]
    mentions: dict[str, set[str]] = field(default_factory=dict)
    identity: dict[str, tuple[str, str]] = field(default_factory=dict)
    records_by_entity: dict[str, set[str]] = field(default_factory=lambda: defaultdict(set))
    relation_records: set[str] = field(default_factory=set)
    question_entities: set[str] = field(default_factory=set)
    target_relation: str | None = None

    def linked_entities(self, record_id: str) -> set[str]:
        return self.mentions.get(record_id, set())

    def resolves(self, entity: str) -> str | None:
        """Follow a refers_to edge from a surface to its canonical name."""
        for surface, canonical in self.identity.values():
            if _norm(surface) == entity:
                return _norm(canonical)
        return None


class _Rec:
    """Shim so pool rows can feed the qualified canonicalization parser."""

    def __init__(self, record_id: str, content: str):
        self.evidence_id, self.content = record_id, content


def build_task_graph(candidates: Sequence[Mapping[str, Any]], question: str,
                     texts: Mapping[str, str]) -> TaskGraph:
    ids = [c["document_id"] for c in candidates]
    corpus = [texts.get(i, "") for i in ids]
    entity_types = derive_entity_types(corpus + [question])
    graph = TaskGraph(record_ids=ids, target_relation=parse_target_relation(question))
    graph.records_by_entity = defaultdict(set)
    graph.question_entities = extract_mentions(question, entity_types)

    for record_id, content in zip(ids, corpus):
        found = extract_mentions(content, entity_types)
        graph.mentions[record_id] = found
        for entity in found:
            graph.records_by_entity[entity].add(record_id)
        if graph.target_relation and graph.target_relation in _norm(content):
            graph.relation_records.add(record_id)

    # refers_to edges come from the already-qualified runtime canonicalization.
    for link in extract_identity_links([_Rec(i, t) for i, t in zip(ids, corpus)]):
        graph.identity[link.record_id] = (link.surface, link.canonical)
    return graph


def _reachable_entities(graph: TaskGraph) -> set[str]:
    """Question entities plus anything an identity record resolves them to."""
    frontier = set(graph.question_entities)
    for surface, canonical in graph.identity.values():
        if _norm(surface) in frontier:
            frontier.add(_norm(canonical))
    return frontier


# --------------------------------------------------------------------------
# S2a: connectivity to the question subject and to what is already selected.
# Still per-document, but the score is a GRAPH property rather than lexical
# similarity, which isolates "connectivity helps" from "chains help".
# --------------------------------------------------------------------------

def s2a_entity_connectivity(candidates, *, budget: int, question: str, texts, **_) -> list[str]:
    graph = build_task_graph(candidates, question, texts)
    anchors = _reachable_entities(graph)
    chosen: list[str] = []
    live = set(anchors)
    remaining = list(graph.record_ids)
    while len(chosen) < budget and remaining:
        best, best_score = None, -1e18
        for record_id in remaining:
            entities = graph.linked_entities(record_id)
            score = 3.0 * len(entities & live)
            if record_id in graph.identity:
                surface, _canonical = graph.identity[record_id]
                # An identity record is valuable exactly when it resolves
                # something we are already tracking, however little lexical
                # overlap it has with the question.
                score += 4.0 if _norm(surface) in live else 0.5
            if record_id in graph.relation_records:
                score += 1.0
            score -= 0.5 * len(entities - live)
            score += 0.25 / (1 + graph.record_ids.index(record_id))
            if score > best_score:
                best, best_score = record_id, score
        chosen.append(best)
        remaining.remove(best)
        live |= graph.linked_entities(best)
        if best in graph.identity:
            live.add(_norm(graph.identity[best][1]))
    return chosen[:budget]


# --------------------------------------------------------------------------
# S2b: enumerate bounded chains and score whole paths, then pack greedily.
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class Chain:
    records: tuple[str, ...]
    entities: frozenset[str]
    closes_relation: bool
    hops: int


def enumerate_chains(graph: TaskGraph, *, max_hops: int = 3) -> list[Chain]:
    """Bounded paths from a question entity toward a relation-bearing record."""
    start = _reachable_entities(graph)
    chains: list[Chain] = []
    # Each state is (records so far, entities reached).
    frontier: list[tuple[tuple[str, ...], frozenset[str]]] = [((), frozenset(start))]
    for _hop in range(max_hops):
        nxt: list[tuple[tuple[str, ...], frozenset[str]]] = []
        for records, reached in frontier:
            for entity in reached:
                for record_id in graph.records_by_entity.get(entity, ()):
                    if record_id in records:
                        continue
                    grown = records + (record_id,)
                    gained = reached | graph.linked_entities(record_id)
                    if record_id in graph.identity:
                        gained = gained | {_norm(graph.identity[record_id][1])}
                    chains.append(Chain(
                        records=grown, entities=frozenset(gained),
                        closes_relation=record_id in graph.relation_records,
                        hops=len(grown)))
                    nxt.append((grown, gained))
        frontier = nxt
        if not frontier:
            break
    return chains


def _score_chain(chain: Chain, graph: TaskGraph, *, use_relation: bool) -> float:
    score = 2.0 * len(chain.entities & _reachable_entities(graph))
    score += 1.5 * chain.hops                      # completed structure
    if chain.closes_relation:
        # A chain that terminates in the asked-for relation is a candidate
        # answer path. Weighted as ONE component, never dominant.
        score += 3.0 if use_relation else 0.0
    identity_records = sum(1 for r in chain.records if r in graph.identity)
    score += 1.0 * identity_records
    score -= 0.25 * len(chain.records)             # prefer compact chains
    return score


def _pack_chains(graph: TaskGraph, chains: list[Chain], budget: int,
                 *, use_relation: bool) -> list[str]:
    """Greedy chain packing: add whole structures, not top-scoring documents."""
    scored = sorted(
        chains, key=lambda c: (-_score_chain(c, graph, use_relation=use_relation),
                               len(c.records)))
    chosen: list[str] = []
    for chain in scored:
        if len(chosen) >= budget:
            break
        addition = [r for r in chain.records if r not in chosen]
        if not addition:
            continue
        if len(chosen) + len(addition) <= budget:
            chosen.extend(addition)
    # Backfill from pool order only if the graph produced too little structure.
    for record_id in graph.record_ids:
        if len(chosen) >= budget:
            break
        if record_id not in chosen:
            chosen.append(record_id)
    return chosen[:budget]


def s2b_chain_completion(candidates, *, budget: int, question: str, texts, **_) -> list[str]:
    graph = build_task_graph(candidates, question, texts)
    chains = enumerate_chains(graph)
    return _pack_chains(graph, chains, budget, use_relation=False)


def s2c_chain_plus_relation(candidates, *, budget: int, question: str, texts, **_) -> list[str]:
    graph = build_task_graph(candidates, question, texts)
    chains = enumerate_chains(graph)
    return _pack_chains(graph, chains, budget, use_relation=True)
