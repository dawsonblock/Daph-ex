"""controlled_gate_a_v4 — structural generalization with provable inferability.

A new generator rather than a patch to v3, because the semantics change: every
task now carries an evaluator-only proof graph, and construction *fails* if the
answer is not reachable from the question's own surface form through visible
evidence alone.

The four v3 defects this exists to eliminate
(`data/hrm/controlled_gate_a_v3/V3_KNOWN_LIMITATIONS.md`):

1. **Style leakage.** v3 drew a chain's second-hop style from the global tuple
   and hard-coded `change_log` for temporal records, so held-out styles appeared
   in both splits. Here every record's style is drawn from the split's own
   allowed set, and the audit checks *records*, not task labels.
2. **Fake aliases.** v3's alias was a prefix truncation (`Nimbus assembly` →
   `Nimbus As`). Here an alias is a genuinely different name (`Bluebird unit`),
   and the corpus contains an explicit record stating the alias relationship.
3. **Non-inferable descriptions.** v3 asked about "the auxiliary unit listed in
   the intake record" with nothing linking that phrase to any entity — 0 of 120
   were answerable. Here a description is introduced by an explicit record, so
   the identity is recoverable from text.
4. **Non-independent oracle.** The oracle ladder reads latent identity from the
   proof graph, never by re-running the extractor under test.

Latent identifiers (`entity_0042`, `value_0007`) exist only inside
`_oracle_metadata` and are never rendered into any model-visible string.
"""

from __future__ import annotations

import hashlib
import json
import random
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping, Sequence

GENERATOR_VERSION = "controlled-gate-a-v4"


class OpportunityGroup(str, Enum):
    A_ONE_PASS_SUFFICIENT = "A_ONE_PASS_SUFFICIENT"
    B_SECOND_PASS_REQUIRED = "B_SECOND_PASS_REQUIRED"
    C_SECOND_PASS_CONFUSING = "C_SECOND_PASS_CONFUSING"
    D_SECOND_PASS_SUPPRESS = "D_SECOND_PASS_SUPPRESS"


class EntityRegime(str, Enum):
    """How the *question* refers to its subject."""

    CANONICAL = "canonical"          # the same name the evidence uses
    ABBREVIATION = "abbreviation"    # NCM-4, introduced by an explicit record
    ALIAS = "alias"                  # a different name, linked by evidence
    DESCRIPTION = "description"      # a descriptive phrase, linked by evidence


SOURCE_STYLES = (
    "formal_registry", "technical_note", "key_value_log",
    "table_text", "change_log", "message",
)

FAMILIES = (
    "entity_attribute", "ownership_chain", "configuration_chain",
    "dependency_chain", "location_chain", "temporal_chain",
    "numeric_chain", "procedural_chain", "temporal_update", "distractor_heavy",
)

ITERATIVE_FAMILIES = frozenset({
    "ownership_chain", "configuration_chain", "dependency_chain",
    "location_chain", "temporal_chain", "numeric_chain", "procedural_chain",
})

CLUSTERS_PER_CORPUS = 24

_HEADS = (
    "Nimbus", "Bluebird", "Kestrel", "Marlin", "Osprey", "Pelican", "Quail",
    "Raven", "Sparrow", "Teal", "Vireo", "Wren", "Auk", "Bittern", "Curlew",
    "Dunlin", "Egret", "Finch", "Gannet", "Heron", "Ibis", "Jacana",
)
_ROLES = ("control module", "sensor array", "pressure assembly", "relay unit",
          "intake manifold", "drive cluster")
_DESCRIPTORS = ("secondary unit installed during commissioning",
                "standby unit registered at handover",
                "auxiliary unit listed in the intake record",
                "reserve unit noted in the acceptance log")

_SYMBOLIC = ("ALPHA-RED", "BETA-GREEN", "GAMMA-BLUE", "DELTA-AMBER",
             "EPSILON-SLATE", "ZETA-CORAL", "ETA-INDIGO", "THETA-OLIVE")
_ENUM = ("provisional", "accredited", "suspended", "retired", "quarantined")
_BOOLEAN = ("true", "false")
_JSON_KEYS = ("category", "tier", "state", "grade")

ANSWER_KINDS = ("numeric", "symbolic", "enum", "boolean", "json_field")
ANSWER_KIND_CAPACITY = {
    "numeric": 8000, "symbolic": len(_SYMBOLIC), "enum": len(_ENUM),
    "boolean": len(_BOOLEAN), "json_field": len(_ENUM),
}
FAMILY_VALUE_DEMAND = {"distractor_heavy": 5, "temporal_update": 2, "temporal_chain": 2}

_ANSWER_BEARING_KINDS = frozenset({"required", "required_current", "direct_answer"})


# ---------------------------------------------------------------------------
# Latent entities and their surface realisations
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class LatentEntity:
    """One entity, several genuinely different names. Latent id never rendered."""

    latent_id: str
    canonical: str
    alias: str
    abbreviation: str
    description: str

    def surface(self, regime: EntityRegime) -> str:
        return {
            EntityRegime.CANONICAL: self.canonical,
            EntityRegime.ALIAS: self.alias,
            EntityRegime.ABBREVIATION: self.abbreviation,
            EntityRegime.DESCRIPTION: f"the {self.description}",
        }[regime]


def _make_entity(rng: random.Random, latent_id: str, used_heads: set[str]) -> LatentEntity:
    for _ in range(200):
        head, alias_head = rng.sample(_HEADS, 2)
        if head not in used_heads and alias_head not in used_heads:
            used_heads.update({head, alias_head})
            role = rng.choice(_ROLES)
            initials = "".join(word[0].upper() for word in role.split())
            return LatentEntity(
                latent_id=latent_id,
                canonical=f"{head} {role}",
                # A genuinely different name, not a truncation of the canonical.
                alias=f"{alias_head} {rng.choice(_ROLES).split()[0]}",
                abbreviation=f"{head[0].upper()}{initials}-{rng.randrange(1, 9)}",
                description=rng.choice(_DESCRIPTORS),
            )
    raise RuntimeError("Exhausted entity head vocabulary")


# ---------------------------------------------------------------------------
# Style renderers — the style is always supplied by the caller from the split
# ---------------------------------------------------------------------------

def _render(style: str, variant: int, subject: str, relation: str, obj: str) -> str:
    table = {
        "formal_registry": (
            f"The {relation} registry records that {subject} is assigned {obj}.",
            f"Registry entry: {subject} — {relation} — {obj}.",
            f"Per the {relation} register, {obj} is allocated to {subject}.",
            f"It is recorded in the {relation} registry that {subject} holds {obj}.",
        ),
        "technical_note": (
            f"During setup, {subject} was paired with {obj} for {relation}.",
            f"Note: the {relation} for {subject} resolves to {obj}.",
            f"Engineering notes indicate {subject} uses {obj} as its {relation}.",
            f"For {subject}, {relation} requires {obj}.",
        ),
        "key_value_log": (
            f"subject={subject}; {relation}={obj}",
            f"[{relation}] {subject} -> {obj}",
            f'{{"subject": "{subject}", "{relation}": "{obj}"}}',
            f"{relation}:\n  subject: {subject}\n  value: {obj}",
        ),
        "table_text": (
            f"| subject | {relation} |\n| {subject} | {obj} |",
            f"Row: {subject} | {relation} | {obj}",
            f"Table {relation}: {subject} maps to {obj}.",
            f"{subject}\t{relation}\t{obj}",
        ),
        "change_log": (
            f"Changelog: {relation} for {subject} set to {obj}.",
            f"- updated {subject}: {relation} now {obj}",
            f"Revision applied — {subject} {relation} changed to {obj}.",
            f"[change] {subject} :: {relation} := {obj}",
        ),
        "message": (
            f"Quick note — {subject}'s {relation} is {obj}, in case it comes up.",
            f"Hi, confirming that {subject} has {obj} as its {relation}.",
            f"FYI: for {subject} the {relation} we settled on was {obj}.",
            f"Following up: {subject} → {relation} → {obj}.",
        ),
    }
    if style not in table:
        raise ValueError(f"Unknown source style {style!r}")
    return table[style][variant % 4]


_IDENTITY_TEMPLATES = {
    EntityRegime.ALIAS: (
        "{other} is the operational alias for {canonical}.",
        "Alias register: {other} refers to {canonical}.",
        "Note that {other} and {canonical} denote the same unit.",
        "{other} is recorded as an alternate designation of {canonical}.",
    ),
    EntityRegime.ABBREVIATION: (
        "{other} is the short code for {canonical}.",
        "Abbreviation table: {other} = {canonical}.",
        "In shorthand, {canonical} is written {other}.",
        "{other} denotes {canonical} in condensed records.",
    ),
    EntityRegime.DESCRIPTION: (
        "The intake record identifies {canonical} as {other}.",
        "{other} is {canonical}, per the acceptance log.",
        "Commissioning notes describe {canonical} as {other}.",
        "It was {canonical} that served as {other}.",
    ),
}


def _identity_record(regime: EntityRegime, variant: int, entity: LatentEntity) -> str:
    """The record that makes a non-canonical reference recoverable from text."""

    return _IDENTITY_TEMPLATES[regime][variant % 4].format(
        other=entity.surface(regime), canonical=entity.canonical,
    )


def _question(variant: int, subject_surface: str, relation: str) -> str:
    return (
        f"Which {relation} applies to {subject_surface}?",
        f"What is the {relation} for {subject_surface}?",
        f"For {subject_surface}, which {relation} is recorded?",
        f"Identify the {relation} associated with {subject_surface}.",
    )[variant % 4]


# ---------------------------------------------------------------------------
# Records and proof graph
# ---------------------------------------------------------------------------

@dataclass
class ProofEdge:
    record_id: str
    source: str      # latent id or literal surface for the question form
    relation: str
    target: str      # latent id or latent value id

    def to_dict(self) -> dict[str, str]:
        return {"record_id": self.record_id, "source": self.source,
                "relation": self.relation, "target": self.target}


@dataclass
class TaskDraft:
    question: str
    answer: str
    answer_kind: str
    required_ids: list[str]
    rows: list[dict]
    proof_edges: list[ProofEdge]
    latent_subject: str
    latent_bridge: str | None
    target_relation: str
    answer_node: str
    opportunity: OpportunityGroup
    surfaces: dict[str, str] = field(default_factory=dict)


def _row(*, evidence_id, source_id, content, family, cluster, kind, style,
         variant, superseded=False) -> dict:
    return {
        "evidence_id": evidence_id, "source_id": source_id,
        "source_type": f"v4_{style}", "content": content,
        "metadata": {
            "generator_version": GENERATOR_VERSION, "family": family,
            "source_cluster_id": cluster, "record_kind": kind,
            "source_style": style, "template_id": f"{style}-{variant % 4}",
            "superseded": superseded,
        },
    }


def _render_value(rng: random.Random, kind: str, used: set[str]) -> tuple[str, str]:
    """Return (gold_answer, evidence_surface)."""

    for _ in range(80):
        if kind == "symbolic":
            value = rng.choice(_SYMBOLIC); surface = value
        elif kind == "enum":
            value = rng.choice(_ENUM); surface = value
        elif kind == "boolean":
            value = rng.choice(_BOOLEAN); surface = value
        elif kind == "json_field":
            value = rng.choice(_ENUM)
            surface = f'{{"{rng.choice(_JSON_KEYS)}": "{value}"}}'
        else:
            value = str(rng.randrange(1000, 9999)); surface = value
        if value not in used:
            used.add(value)
            return value, surface
    raise RuntimeError("Exhausted answer vocabulary")


def _choose_answer_kind(family: str, ordinal: int) -> str:
    demand = FAMILY_VALUE_DEMAND.get(family, 3)
    eligible = [k for k in ANSWER_KINDS if ANSWER_KIND_CAPACITY[k] >= demand]
    return eligible[ordinal % len(eligible)]


def _leaks(text: str, answer: str) -> bool:
    haystack = tuple(re.findall(r"\w+", text.lower()))
    needle = tuple(re.findall(r"\w+", answer.lower()))
    width = len(needle)
    return bool(width) and any(
        haystack[i:i + width] == needle for i in range(len(haystack) - width + 1)
    )


# ---------------------------------------------------------------------------
# Builders. Every one takes `styles` (the split's allowed set) and draws EVERY
# record's style from it — the v3 defect was a builder reaching past the split.
# ---------------------------------------------------------------------------

def _build_single_hop(
    rng, *, task_id, family, cluster, styles, regime, relation, answer_kind,
    variant, used_heads,
) -> TaskDraft:
    subject = _make_entity(rng, f"{task_id}#subject", used_heads)
    used: set[str] = set()
    answer, surface = _render_value(rng, answer_kind, used)
    style = rng.choice(styles)
    fact_id = f"{task_id}/fact"
    rows = [_row(evidence_id=fact_id, source_id=f"{cluster}/primary",
                 content=_render(style, variant, subject.canonical, relation, surface),
                 family=family, cluster=cluster, kind="required", style=style,
                 variant=variant)]
    edges = [ProofEdge(fact_id, subject.latent_id, relation, f"{task_id}#value")]
    required = [fact_id]

    # A non-canonical reference needs an explicit identity record, or the task
    # is not answerable from text. This is the v3 description defect.
    if regime != EntityRegime.CANONICAL:
        identity_style = rng.choice(styles)
        identity_id = f"{task_id}/identity"
        rows.append(_row(
            evidence_id=identity_id, source_id=f"{cluster}/identity",
            content=_identity_record(regime, variant, subject),
            family=family, cluster=cluster, kind="required_identity",
            style=identity_style, variant=variant))
        edges.insert(0, ProofEdge(
            identity_id, f"surface:{subject.surface(regime)}", "refers_to",
            subject.latent_id))
        required.insert(0, identity_id)

    return TaskDraft(
        question=_question(variant, subject.surface(regime), relation),
        answer=answer, answer_kind=answer_kind, required_ids=required, rows=rows,
        proof_edges=edges, latent_subject=subject.latent_id, latent_bridge=None,
        target_relation=relation, answer_node=f"{task_id}#value",
        opportunity=OpportunityGroup.A_ONE_PASS_SUFFICIENT,
        surfaces={"subject": subject.surface(regime), "canonical": subject.canonical},
    )


def _build_chain(
    rng, *, task_id, family, cluster, styles, regime, first_relation,
    second_relation, answer_kind, variant, opportunity, used_heads,
) -> TaskDraft:
    subject = _make_entity(rng, f"{task_id}#subject", used_heads)
    bridge = _make_entity(rng, f"{task_id}#bridge", used_heads)
    used: set[str] = set()
    answer, surface = _render_value(rng, answer_kind, used)
    link_style, value_style = rng.choice(styles), rng.choice(styles)
    link_id, value_id = f"{task_id}/link", f"{task_id}/value"

    rows = [
        _row(evidence_id=link_id, source_id=f"{cluster}/link",
             content=_render(link_style, variant, subject.canonical,
                             first_relation, bridge.canonical),
             family=family, cluster=cluster, kind="required", style=link_style,
             variant=variant),
        _row(evidence_id=value_id, source_id=f"{cluster}/value",
             content=_render(value_style, variant + 1, bridge.canonical,
                             second_relation, surface),
             family=family, cluster=cluster, kind="required", style=value_style,
             variant=variant + 1),
    ]
    edges = [
        ProofEdge(link_id, subject.latent_id, first_relation, bridge.latent_id),
        ProofEdge(value_id, bridge.latent_id, second_relation, f"{task_id}#value"),
    ]
    required = [link_id, value_id]

    if regime != EntityRegime.CANONICAL:
        identity_id = f"{task_id}/identity"
        rows.append(_row(
            evidence_id=identity_id, source_id=f"{cluster}/identity",
            content=_identity_record(regime, variant, subject),
            family=family, cluster=cluster, kind="required_identity",
            style=rng.choice(styles), variant=variant))
        edges.insert(0, ProofEdge(
            identity_id, f"surface:{subject.surface(regime)}", "refers_to",
            subject.latent_id))
        required.insert(0, identity_id)

    # Dead-end links: other entities connected to the subject that do not lead
    # to the answer, so a bridge detector must discriminate rather than chase.
    for index in range(2):
        decoy = _make_entity(rng, f"{task_id}#decoy{index}", used_heads)
        rows.append(_row(
            evidence_id=f"{task_id}/dead-end-{index}",
            source_id=f"{cluster}/dead-end-{index}",
            content=_render(rng.choice(styles), variant + 2, subject.canonical,
                            rng.choice(("mounted monitor", "spare enclosure")),
                            decoy.canonical),
            family=family, cluster=cluster, kind="dead_end_link",
            style=rng.choice(styles), variant=variant + 2))

    if opportunity == OpportunityGroup.D_SECOND_PASS_SUPPRESS:
        direct_id = f"{task_id}/direct"
        direct_style = rng.choice(styles)
        rows.append(_row(
            evidence_id=direct_id, source_id=f"{cluster}/summary",
            content=_render(direct_style, variant + 3, subject.canonical,
                            second_relation, surface),
            family=family, cluster=cluster, kind="direct_answer",
            style=direct_style, variant=variant + 3))
        edges.append(ProofEdge(direct_id, subject.latent_id, second_relation,
                               f"{task_id}#value"))
        required = ([required[0]] if regime != EntityRegime.CANONICAL else []) + [direct_id]

    if opportunity == OpportunityGroup.C_SECOND_PASS_CONFUSING:
        for index in range(2):
            _, wrong_surface = _render_value(rng, answer_kind, used)
            style = rng.choice(styles)
            rows.append(_row(
                evidence_id=f"{task_id}/rejected-{index}",
                source_id=f"{cluster}/rejected-{index}",
                content=_render(style, variant + 2, bridge.canonical,
                                f"proposed {second_relation}", wrong_surface)
                        + " This proposal was rejected.",
                family=family, cluster=cluster, kind="rejected_candidate",
                style=style, variant=variant + 2))

    return TaskDraft(
        question=_question(variant, subject.surface(regime), second_relation),
        answer=answer, answer_kind=answer_kind, required_ids=required, rows=rows,
        proof_edges=edges, latent_subject=subject.latent_id,
        latent_bridge=bridge.latent_id, target_relation=second_relation,
        answer_node=f"{task_id}#value", opportunity=opportunity,
        surfaces={"subject": subject.surface(regime), "canonical": subject.canonical,
                  "bridge": bridge.canonical},
    )


def _build_temporal(
    rng, *, task_id, family, cluster, styles, regime, relation, answer_kind,
    variant, chained, used_heads,
) -> TaskDraft:
    subject = _make_entity(rng, f"{task_id}#subject", used_heads)
    used: set[str] = set()
    answer, surface = _render_value(rng, answer_kind, used)
    _, stale_surface = _render_value(rng, answer_kind, used)
    rows: list[dict] = []
    edges: list[ProofEdge] = []
    holder_entity = subject
    required: list[str] = []
    opportunity = OpportunityGroup.A_ONE_PASS_SUFFICIENT

    if chained:
        bridge = _make_entity(rng, f"{task_id}#bridge", used_heads)
        link_id = f"{task_id}/link"
        link_style = rng.choice(styles)
        rows.append(_row(evidence_id=link_id, source_id=f"{cluster}/link",
                         content=_render(link_style, variant, subject.canonical,
                                         "assigned unit", bridge.canonical),
                         family=family, cluster=cluster, kind="required",
                         style=link_style, variant=variant))
        edges.append(ProofEdge(link_id, subject.latent_id, "assigned unit",
                               bridge.latent_id))
        holder_entity = bridge
        required.append(link_id)
        opportunity = OpportunityGroup.B_SECOND_PASS_REQUIRED

    current_id, stale_id = f"{task_id}/current", f"{task_id}/superseded"
    # Supersession is expressed inside the split's own styles, not a fixed one.
    current_style, stale_style = rng.choice(styles), rng.choice(styles)
    rows.append(_row(
        evidence_id=current_id, source_id=f"{cluster}/revision-2",
        content="Revision 2 (effective 2031-06-01) supersedes revision 1: "
                + _render(current_style, variant, holder_entity.canonical, relation, surface),
        family=family, cluster=cluster, kind="required_current",
        style=current_style, variant=variant))
    rows.append(_row(
        evidence_id=stale_id, source_id=f"{cluster}/revision-1",
        content="Revision 1 (effective 2030-01-01, since superseded) recorded: "
                + _render(stale_style, variant, holder_entity.canonical, relation, stale_surface),
        family=family, cluster=cluster, kind="superseded", style=stale_style,
        variant=variant, superseded=True))
    edges.append(ProofEdge(current_id, holder_entity.latent_id, relation,
                           f"{task_id}#value"))
    required.append(current_id)

    if regime != EntityRegime.CANONICAL:
        identity_id = f"{task_id}/identity"
        rows.append(_row(
            evidence_id=identity_id, source_id=f"{cluster}/identity",
            content=_identity_record(regime, variant, subject),
            family=family, cluster=cluster, kind="required_identity",
            style=rng.choice(styles), variant=variant))
        edges.insert(0, ProofEdge(identity_id, f"surface:{subject.surface(regime)}",
                                  "refers_to", subject.latent_id))
        required.insert(0, identity_id)

    return TaskDraft(
        question=f"As of the current revision, which {relation} applies to "
                 f"{subject.surface(regime)}?",
        answer=answer, answer_kind=answer_kind, required_ids=required, rows=rows,
        proof_edges=edges, latent_subject=subject.latent_id,
        latent_bridge=holder_entity.latent_id if chained else None,
        target_relation=relation, answer_node=f"{task_id}#value",
        opportunity=opportunity,
        surfaces={"subject": subject.surface(regime), "canonical": subject.canonical},
    )


def _build_distractor_heavy(
    rng, *, task_id, family, cluster, styles, regime, relation, answer_kind,
    variant, used_heads,
) -> TaskDraft:
    subject = _make_entity(rng, f"{task_id}#subject", used_heads)
    near = _make_entity(rng, f"{task_id}#near", used_heads)
    used: set[str] = set()
    answer, surface = _render_value(rng, answer_kind, used)
    style = rng.choice(styles)
    accepted_id = f"{task_id}/accepted"
    rows = [_row(evidence_id=accepted_id, source_id=f"{cluster}/accepted",
                 content=_render(style, variant, subject.canonical, relation, surface),
                 family=family, cluster=cluster, kind="required", style=style,
                 variant=variant)]
    edges = [ProofEdge(accepted_id, subject.latent_id, relation, f"{task_id}#value")]
    required = [accepted_id]

    for label, content_style, content in (
        ("near_entity", rng.choice(styles),
         lambda s: _render(s, variant, near.canonical, relation,
                           _render_value(rng, answer_kind, used)[1])),
        ("near_relation", rng.choice(styles),
         lambda s: _render(s, variant, subject.canonical, f"calibration {relation}",
                           _render_value(rng, answer_kind, used)[1])),
        ("near_status", rng.choice(styles),
         lambda s: _render(s, variant, subject.canonical, f"proposed {relation}",
                           _render_value(rng, answer_kind, used)[1])
                   + " Status: rejected."),
        ("near_temporal", rng.choice(styles),
         lambda s: _render(s, variant, subject.canonical, f"former {relation}",
                           _render_value(rng, answer_kind, used)[1])
                   + " This record is historical."),
    ):
        rows.append(_row(
            evidence_id=f"{task_id}/near-{label}", source_id=f"{cluster}/near-{label}",
            content=content(content_style), family=family, cluster=cluster,
            kind=f"near_duplicate_{label}", style=content_style, variant=variant))

    if regime != EntityRegime.CANONICAL:
        identity_id = f"{task_id}/identity"
        rows.append(_row(
            evidence_id=identity_id, source_id=f"{cluster}/identity",
            content=_identity_record(regime, variant, subject),
            family=family, cluster=cluster, kind="required_identity",
            style=rng.choice(styles), variant=variant))
        edges.insert(0, ProofEdge(identity_id, f"surface:{subject.surface(regime)}",
                                  "refers_to", subject.latent_id))
        required.insert(0, identity_id)

    return TaskDraft(
        question=_question(variant, subject.surface(regime), relation),
        answer=answer, answer_kind=answer_kind, required_ids=required, rows=rows,
        proof_edges=edges, latent_subject=subject.latent_id, latent_bridge=None,
        target_relation=relation, answer_node=f"{task_id}#value",
        opportunity=OpportunityGroup.A_ONE_PASS_SUFFICIENT,
        surfaces={"subject": subject.surface(regime), "canonical": subject.canonical},
    )


_CHAIN_RELATIONS = {
    "ownership_chain": ("registered asset", "ownership tier"),
    "configuration_chain": ("active configuration", "parameter grade"),
    "dependency_chain": ("required dependency", "release channel"),
    "location_chain": ("hosting venue", "service region"),
    "numeric_chain": ("rate record", "billing multiplier"),
    "procedural_chain": ("applicable procedure", "required action"),
}


@dataclass(frozen=True)
class ControlledCorpusV4:
    tasks: tuple[Mapping[str, Any], ...]
    evidence: tuple[Mapping[str, Any], ...]
    manifest: Mapping[str, Any]


def _digest(rows: Sequence[Mapping[str, Any]]) -> str:
    payload = "\n".join(json.dumps(r, sort_keys=True, separators=(",", ":")) for r in rows)
    return hashlib.sha256(payload.encode()).hexdigest()


def verify_inferable(task: Mapping[str, Any], by_id: Mapping[str, Mapping[str, Any]]) -> list[str]:
    """Prove the answer is reachable from the question's own surface via text.

    Returns a list of problems; empty means the task is inferable. This is the
    check whose absence let v3 ship 120 tasks whose subject appeared nowhere in
    their evidence.
    """

    problems: list[str] = []
    meta = task["_oracle_metadata"]
    edges = meta["proof_edges"]

    def norm(text: str) -> str:
        return " ".join(re.findall(r"\w+", text.lower()))

    for edge in edges:
        record = by_id.get(edge["record_id"])
        if record is None:
            problems.append(f"proof edge cites missing record {edge['record_id']}")
            continue
        if edge["record_id"] not in task["required_evidence_ids"]:
            continue  # supporting edge, not part of the minimal required path
        content = norm(record["content"])
        # A surface-anchored edge must realise the question's phrase in text.
        if edge["source"].startswith("surface:"):
            phrase = norm(edge["source"].split("surface:", 1)[1])
            if phrase not in content:
                problems.append(
                    f"identity record {edge['record_id']} does not contain the "
                    f"question's own reference {phrase!r}")

    # The question's subject phrase must be findable somewhere in the required set.
    subject_surface = norm(meta["surfaces"]["subject"])
    required_text = " || ".join(
        norm(by_id[value]["content"]) for value in task["required_evidence_ids"]
        if value in by_id)
    if subject_surface and subject_surface not in required_text:
        problems.append(
            f"question subject {subject_surface!r} appears in no required record")

    # The answer must appear in at least one required record.
    if not any(_leaks(by_id[v]["content"], task["answer"])
               for v in task["required_evidence_ids"] if v in by_id):
        problems.append("answer appears in no required record")
    return problems


def build_v4_corpus(
    *, seed: int, tasks_per_family: int, split: str, styles: Sequence[str],
    regimes: Sequence[str],
) -> ControlledCorpusV4:
    """Build one v4 split. `styles` and `regimes` are the split's ONLY sources."""

    allowed_styles = list(styles)
    allowed_regimes = [EntityRegime(value) for value in regimes]
    unknown = set(allowed_styles) - set(SOURCE_STYLES)
    if unknown:
        raise ValueError(f"Unknown source styles: {sorted(unknown)}")
    if not allowed_styles or not allowed_regimes:
        raise ValueError("A split needs at least one style and one regime")

    rng = random.Random(seed)
    tasks: list[Mapping[str, Any]] = []
    evidence: list[Mapping[str, Any]] = []
    opportunity_cycle = (
        OpportunityGroup.B_SECOND_PASS_REQUIRED,
        OpportunityGroup.B_SECOND_PASS_REQUIRED,
        OpportunityGroup.C_SECOND_PASS_CONFUSING,
        OpportunityGroup.D_SECOND_PASS_SUPPRESS,
    )

    for family in FAMILIES:
        for ordinal in range(tasks_per_family):
            task_id = f"{family}-{ordinal:04d}"
            cluster_index = (ordinal * 7) % CLUSTERS_PER_CORPUS
            primary_style = allowed_styles[cluster_index % len(allowed_styles)]
            cluster = f"{primary_style}/cluster-{cluster_index:02d}"
            variant = (ordinal // 4) % 4
            regime = allowed_regimes[(ordinal * 11) % len(allowed_regimes)]
            answer_kind = _choose_answer_kind(family, ordinal)

            draft: TaskDraft | None = None
            for _attempt in range(40):
                heads: set[str] = set()
                shared = dict(rng=rng, task_id=task_id, family=family, cluster=cluster,
                              styles=allowed_styles, regime=regime,
                              answer_kind=answer_kind, variant=variant,
                              used_heads=heads)
                if family == "entity_attribute":
                    candidate = _build_single_hop(relation="assigned category", **shared)
                elif family == "distractor_heavy":
                    candidate = _build_distractor_heavy(relation="routing class", **shared)
                elif family == "temporal_update":
                    candidate = _build_temporal(relation="operating tier", chained=False, **shared)
                elif family == "temporal_chain":
                    candidate = _build_temporal(relation="operating tier", chained=True, **shared)
                else:
                    first, second = _CHAIN_RELATIONS[family]
                    candidate = _build_chain(
                        first_relation=first, second_relation=second,
                        opportunity=opportunity_cycle[ordinal % len(opportunity_cycle)],
                        **shared)
                required = set(candidate.required_ids)
                if _leaks(candidate.question, candidate.answer):
                    continue
                if any(_leaks(row["content"], candidate.answer) for row in candidate.rows
                       if row["metadata"]["record_kind"] not in _ANSWER_BEARING_KINDS):
                    continue
                draft = candidate
                break
            if draft is None:
                raise RuntimeError(f"Could not build a leak-free task for {task_id}")

            task = {
                "task_id": task_id,
                "question": draft.question,
                "answer": draft.answer,
                "required_evidence_ids": draft.required_ids,
                "oracle_evidence_ids": draft.required_ids,
                "family": family,
                "template_id": f"{family}/{primary_style}-{variant}",
                "source_cluster_id": cluster,
                "split": split,
                "verifier": "numeric" if draft.answer_kind == "numeric" else "canonical",
                "metadata": {
                    "generator_version": GENERATOR_VERSION,
                    "generation_seed": seed,
                    "answer_kind": draft.answer_kind,
                    "entity_regime": regime.value,
                    "source_style": primary_style,
                    "opportunity_group": draft.opportunity.value,
                    "iterative_family": family in ITERATIVE_FAMILIES,
                },
                # Evaluator-only. Never rendered, never indexed, never prompted.
                "_oracle_metadata": {
                    "latent_subject": draft.latent_subject,
                    "latent_bridge": draft.latent_bridge,
                    "target_relation": draft.target_relation,
                    "answer_node": draft.answer_node,
                    "proof_edges": [edge.to_dict() for edge in draft.proof_edges],
                    "surfaces": draft.surfaces,
                },
            }
            tasks.append(task)
            evidence.extend(draft.rows)

    frozen_tasks, frozen_evidence = tuple(tasks), tuple(evidence)
    by_id = {row["evidence_id"]: row for row in frozen_evidence}
    failures = {
        row["task_id"]: verify_inferable(row, by_id)
        for row in frozen_tasks
        if verify_inferable(row, by_id)
    }
    if failures:
        sample = dict(list(failures.items())[:3])
        raise RuntimeError(
            f"{len(failures)} tasks are not inferable from visible evidence; "
            f"refusing to emit corpus. Sample: {sample}")

    record_styles = sorted({row["metadata"]["source_style"] for row in frozen_evidence})
    escaped = set(record_styles) - set(allowed_styles)
    if escaped:
        raise RuntimeError(
            f"Records escaped the split's allowed styles: {sorted(escaped)}")

    manifest = {
        "dataset_id": f"{GENERATOR_VERSION}-{split}",
        "generation_seed": seed, "split": split,
        "tasks_per_family": tasks_per_family,
        "task_count": len(frozen_tasks), "evidence_count": len(frozen_evidence),
        "families": list(FAMILIES), "iterative_families": sorted(ITERATIVE_FAMILIES),
        "template_count": len({r["template_id"] for r in frozen_tasks}),
        "source_cluster_count": len({r["source_cluster_id"] for r in frozen_tasks}),
        "allowed_source_styles": allowed_styles,
        "evidence_record_styles": record_styles,
        "entity_regimes": [r.value for r in allowed_regimes],
        "answer_kinds": list(ANSWER_KINDS),
        "opportunity_groups": sorted({
            r["metadata"]["opportunity_group"] for r in frozen_tasks}),
        "task_sha256": _digest(frozen_tasks),
        "evidence_sha256": _digest(frozen_evidence),
        "inferability_verified": True,
        "claim_strength": "CONTROLLED_SYNTHETIC_BENCHMARK_ONLY",
        "natural_memory_claim_allowed": False,
    }
    return ControlledCorpusV4(frozen_tasks, frozen_evidence, manifest)
