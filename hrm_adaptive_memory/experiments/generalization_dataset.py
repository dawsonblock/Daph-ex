"""controlled_gate_a_v3 — a deliberately adversarial generalization corpus.

v2 let the mechanism reach the oracle ceiling, so it can no longer discriminate.
This generator is built to *break* the current system, and to make the
family-clustered bootstrap a meaningful test rather than a structural
impossibility.

What changes from v2, and why:

- **Iterative dependency spread across families.** v2 had bridge structure in
  exactly one family of five, so no family-level bootstrap could certify a
  universally positive effect. Here multiple families carry it.
- **Genuinely distinct templates.** v2 used `template_id = ordinal % 3` over
  identical wording. Here a template is a real surface form, and the retriever
  must survive changes in phrasing.
- **Real source styles.** Each cluster has a coherent schema (registry prose,
  key/value logs, tables-as-text, change logs, messages, notes), so holding a
  cluster out actually holds out a structure.
- **Multiple entity-naming regimes.** v2's `Adapter-36452` identifiers made
  lexical joins trivially easy. Natural names, aliases, abbreviations, and
  descriptions break the exact-token shortcut.
- **Heterogeneous retrieval opportunity.** Tasks are labelled by whether a
  second pass is necessary, useless, or actively harmful, so ΔU_followup can be
  positive, zero, *and* negative — the precondition for Gate D meaning anything.
- **Ambiguous bridges.** Several linked entities, only one of which is useful,
  plus dead ends, conflicts, and superseded links.
- **Non-numeric answers.** Symbolic labels, enums, booleans, and values embedded
  in JSON-structured records, so results cannot depend on numeric extraction.
  A `json_field` answer is the *value*, not the JSON wrapper: the development
  split showed that demanding unrequested JSON syntax measures formatting
  compliance rather than evidence use (0/28 despite correct extraction).

Answers are generated as latent symbols and rendered last, so an answer cannot
leak into a question or a distractor by construction.
"""

from __future__ import annotations

import hashlib
import json
import random
import re
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Mapping, Sequence

GENERATOR_VERSION = "controlled-gate-a-v3"


class OpportunityGroup(str, Enum):
    """Whether a second retrieval pass is needed, useless, or harmful."""

    A_ONE_PASS_SUFFICIENT = "A_ONE_PASS_SUFFICIENT"
    B_SECOND_PASS_REQUIRED = "B_SECOND_PASS_REQUIRED"
    C_SECOND_PASS_CONFUSING = "C_SECOND_PASS_CONFUSING"
    D_SECOND_PASS_SUPPRESS = "D_SECOND_PASS_SUPPRESS"


class EntityRegime(str, Enum):
    EXACT_ID = "exact_id"
    NATURAL_NAME = "natural_name"
    ALIAS = "alias"
    ABBREVIATION = "abbreviation"
    DESCRIPTION = "description"


SOURCE_STYLES = (
    "formal_registry", "technical_note", "key_value_log",
    "table_text", "change_log", "message",
)

FAMILIES = (
    "entity_attribute",       # single hop
    "ownership_chain",        # 2-hop
    "configuration_chain",    # 2-hop
    "dependency_chain",       # 2-hop
    "location_chain",         # 2-hop
    "temporal_chain",         # 2-hop + supersession
    "numeric_chain",          # 2-hop + arithmetic
    "procedural_chain",       # 2-hop
    "temporal_update",        # single hop + supersession
    "distractor_heavy",       # single hop + adversarial near-duplicates
)

# Source clusters are the unit a cluster-level bootstrap resamples, so there
# must be enough of them for that test to have structural support. Each cluster
# carries exactly one style, making a held-out cluster a held-out structure.
CLUSTERS_PER_CORPUS = 24

ITERATIVE_FAMILIES = frozenset({
    "ownership_chain", "configuration_chain", "dependency_chain",
    "location_chain", "temporal_chain", "numeric_chain", "procedural_chain",
})

# ---------------------------------------------------------------------------
# Surface vocabulary
# ---------------------------------------------------------------------------

_NOUNS = (
    "Atlas", "Beacon", "Cinder", "Delta", "Ember", "Fathom", "Granite", "Harbor",
    "Iris", "Juniper", "Keystone", "Lantern", "Meridian", "Nimbus", "Onyx",
    "Pioneer", "Quarry", "Ridge", "Summit", "Tundra", "Umber", "Vantage",
)
_ROLES = {
    "asset": ("controller", "array", "module", "unit", "assembly"),
    "config": ("profile", "preset", "configuration", "schedule", "policy"),
    "package": ("library", "toolkit", "runtime", "driver", "bundle"),
    "venue": ("hall", "pavilion", "annex", "atrium", "gallery"),
    "procedure": ("procedure", "runbook", "protocol", "checklist", "routine"),
}

_SYMBOLIC_ANSWERS = (
    "ALPHA-RED", "BETA-GREEN", "GAMMA-BLUE", "DELTA-AMBER", "EPSILON-SLATE",
    "ZETA-CORAL", "ETA-INDIGO", "THETA-OLIVE", "IOTA-MAROON", "KAPPA-TEAL",
)
_ENUM_ANSWERS = ("provisional", "accredited", "suspended", "retired", "quarantined")
_BOOLEAN_ANSWERS = ("true", "false")
_JSON_KEYS = ("category", "tier", "state", "grade")


@dataclass(frozen=True)
class ControlledCorpusV3:
    tasks: tuple[Mapping[str, Any], ...]
    evidence: tuple[Mapping[str, Any], ...]
    manifest: Mapping[str, Any]


def _digest(rows: Sequence[Mapping[str, Any]]) -> str:
    payload = "\n".join(json.dumps(row, sort_keys=True, separators=(",", ":")) for row in rows)
    return hashlib.sha256(payload.encode()).hexdigest()


# ---------------------------------------------------------------------------
# Entity rendering: one latent entity, several possible surface forms
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Entity:
    latent: str
    regime: EntityRegime
    surface: str
    query_surface: str  # how the question refers to it (may differ from evidence)


def _make_entity(rng: random.Random, latent: str, regime: EntityRegime, kind: str) -> Entity:
    noun = rng.choice(_NOUNS)
    role = rng.choice(_ROLES.get(kind, ("unit",)))
    number = rng.randrange(100, 999)
    if regime == EntityRegime.EXACT_ID:
        surface = f"{noun}-{number}"
        return Entity(latent, regime, surface, surface)
    if regime == EntityRegime.NATURAL_NAME:
        surface = f"{noun} {role}"
        return Entity(latent, regime, surface, surface)
    if regime == EntityRegime.ALIAS:
        # Evidence and question use *different* names for the same thing.
        surface = f"{noun} {role}"
        return Entity(latent, regime, surface, f"{noun} {role[0].upper()}{role[1]}")
    if regime == EntityRegime.ABBREVIATION:
        letters = f"{noun[0]}{role[0].upper()}{rng.choice('XYZ')}"
        surface = f"{letters}-{rng.randrange(1, 9)}"
        return Entity(latent, regime, surface, surface)
    descriptor = rng.choice((
        "the secondary unit assigned during commissioning",
        "the standby unit registered at handover",
        "the auxiliary unit listed in the intake record",
    ))
    return Entity(latent, regime, f"{noun} {role}", descriptor)


# ---------------------------------------------------------------------------
# Source-style renderers: one relation, six genuinely different surface schemas
# ---------------------------------------------------------------------------

def _render(style: str, variant: int, subject: str, relation: str, obj: str) -> str:
    """Render subject-relation-object in a style, with several templates per style."""

    if style == "formal_registry":
        return (
            f"The {relation} registry records that {subject} is assigned {obj}.",
            f"Registry entry: {subject} — {relation} — {obj}.",
            f"Per the {relation} register, {obj} is allocated to {subject}.",
            f"It is recorded in the {relation} registry that {subject} holds {obj}.",
        )[variant % 4]
    if style == "technical_note":
        return (
            f"During setup, {subject} was paired with {obj} for {relation}.",
            f"Note: the {relation} for {subject} resolves to {obj}.",
            f"Engineering notes indicate {subject} uses {obj} as its {relation}.",
            f"For {subject}, {relation} requires {obj}.",
        )[variant % 4]
    if style == "key_value_log":
        return (
            f"subject={subject}; {relation}={obj}",
            f"[{relation}] {subject} -> {obj}",
            f"{{\"subject\": \"{subject}\", \"{relation}\": \"{obj}\"}}",
            f"{relation}:\n  subject: {subject}\n  value: {obj}",
        )[variant % 4]
    if style == "table_text":
        return (
            f"| subject | {relation} |\n| {subject} | {obj} |",
            f"Row: {subject} | {relation} | {obj}",
            f"Table {relation}: {subject} maps to {obj}.",
            f"{subject}\t{relation}\t{obj}",
        )[variant % 4]
    if style == "change_log":
        return (
            f"Changelog: {relation} for {subject} set to {obj}.",
            f"- updated {subject}: {relation} now {obj}",
            f"Revision applied — {subject} {relation} changed to {obj}.",
            f"[change] {subject} :: {relation} := {obj}",
        )[variant % 4]
    return (
        f"Quick note — {subject}'s {relation} is {obj}, in case it comes up.",
        f"Hi, confirming that {subject} has {obj} as its {relation}.",
        f"FYI: for {subject} the {relation} we settled on was {obj}.",
        f"Following up: {subject} → {relation} → {obj}.",
    )[variant % 4]


def _question(family: str, variant: int, subject: str, relation: str) -> str:
    return (
        f"Which {relation} applies to {subject}?",
        f"What is the {relation} for {subject}?",
        f"For {subject}, which {relation} is recorded?",
        f"Identify the {relation} associated with {subject}.",
    )[variant % 4]


def _row(*, evidence_id: str, source_id: str, content: str, family: str,
         source_cluster_id: str, kind: str, style: str, template_id: str,
         valid_from: str | None = None, superseded: bool = False) -> dict[str, Any]:
    return {
        "evidence_id": evidence_id,
        "source_id": source_id,
        "source_type": f"v3_{style}",
        "content": content,
        "metadata": {
            "generator_version": GENERATOR_VERSION,
            "family": family,
            "source_cluster_id": source_cluster_id,
            "record_kind": kind,
            "source_style": style,
            "template_id": template_id,
            "valid_from": valid_from,
            "superseded": superseded,
        },
    }


# ---------------------------------------------------------------------------
# Answer rendering: latent symbol first, surface last
# ---------------------------------------------------------------------------

def _render_answer(rng: random.Random, kind: str, used: set[str]) -> tuple[str, str, str]:
    """Return (gold_answer, evidence_surface, answer_kind).

    ``evidence_surface`` is how the value appears inside a record, which is not
    always the answer itself: a `json_field` value is *embedded* in a JSON
    object in the evidence while the answer remains the bare value. Requiring
    the model to reproduce JSON syntax that the question never asked for would
    measure formatting compliance rather than evidence use — the development
    split showed exactly that, scoring 0/28 on json answers whose values the
    model had in fact extracted correctly.
    """

    for _ in range(50):
        if kind == "symbolic":
            value = rng.choice(_SYMBOLIC_ANSWERS)
            surface = value
        elif kind == "enum":
            value = rng.choice(_ENUM_ANSWERS)
            surface = value
        elif kind == "boolean":
            value = rng.choice(_BOOLEAN_ANSWERS)
            surface = value
        elif kind == "json_field":
            value = rng.choice(_ENUM_ANSWERS)
            surface = f'{{"{rng.choice(_JSON_KEYS)}": "{value}"}}'
        else:
            value = str(rng.randrange(1000, 9999))
            surface = value
        if value not in used:
            used.add(value)
            return value, surface, kind
    raise RuntimeError("Exhausted answer vocabulary")


ANSWER_KINDS = ("numeric", "symbolic", "enum", "boolean", "json_field")

# How many mutually distinct values each kind can supply within one task. A
# boolean cannot furnish a gold answer plus four distinct near-duplicates, so
# families that need many contrasting values must not be given one.
ANSWER_KIND_CAPACITY = {
    "numeric": 8000, "symbolic": len(_SYMBOLIC_ANSWERS),
    "enum": len(_ENUM_ANSWERS), "boolean": len(_BOOLEAN_ANSWERS),
    "json_field": len(_ENUM_ANSWERS),
}

# Distinct values a family must be able to render (gold plus contrasts).
FAMILY_VALUE_DEMAND = {
    "distractor_heavy": 5,
    "temporal_update": 2,
    "temporal_chain": 2,
}


def _choose_answer_kind(family: str, ordinal: int) -> str:
    """Rotate through answer kinds, skipping any too small for the family."""

    demand = FAMILY_VALUE_DEMAND.get(family, 3)
    eligible = [kind for kind in ANSWER_KINDS if ANSWER_KIND_CAPACITY[kind] >= demand]
    if not eligible:  # pragma: no cover - vocabulary is sized to prevent this
        raise RuntimeError(f"No answer kind can supply {demand} distinct values")
    return eligible[ordinal % len(eligible)]


def _verifier_for(kind: str) -> str:
    return "numeric" if kind == "numeric" else "canonical"


# ---------------------------------------------------------------------------
# Task builders
# ---------------------------------------------------------------------------

def _build_single_hop(
    rng: random.Random, *, task_id: str, family: str, cluster: str, style: str,
    regime: EntityRegime, relation: str, answer_kind: str, used: set[str],
    variant: int,
) -> tuple[dict, list[dict], OpportunityGroup]:
    subject = _make_entity(rng, f"{task_id}#subject", regime, "asset")
    answer, surface, kind = _render_answer(rng, answer_kind, used)
    required = f"{task_id}/fact"
    template_id = f"{style}-{variant % 4}"
    rows = [_row(
        evidence_id=required, source_id=f"{cluster}/primary", family=family,
        source_cluster_id=cluster, kind="required", style=style, template_id=template_id,
        content=_render(style, variant, subject.surface, relation, surface),
    )]
    question = _question(family, variant, subject.query_surface, relation)
    task = {
        "question": question, "answer": answer, "answer_kind": kind,
        "required_evidence_ids": [required], "oracle_evidence_ids": [required],
    }
    return task, rows, OpportunityGroup.A_ONE_PASS_SUFFICIENT


def _build_chain(
    rng: random.Random, *, task_id: str, family: str, cluster: str, style: str,
    regime: EntityRegime, first_relation: str, second_relation: str,
    answer_kind: str, used: set[str], variant: int, opportunity: OpportunityGroup,
) -> tuple[dict, list[dict], OpportunityGroup]:
    """Two-hop: subject → bridge → value, with controlled ambiguity."""

    subject = _make_entity(rng, f"{task_id}#subject", regime, "asset")
    bridge = _make_entity(rng, f"{task_id}#bridge", regime, "config")
    answer, surface, kind = _render_answer(rng, answer_kind, used)
    first, second = f"{task_id}/link", f"{task_id}/value"
    template_id = f"{style}-{variant % 4}"
    second_style = SOURCE_STYLES[(SOURCE_STYLES.index(style) + 1) % len(SOURCE_STYLES)]

    rows = [
        _row(evidence_id=first, source_id=f"{cluster}/link", family=family,
             source_cluster_id=cluster, kind="required", style=style,
             template_id=template_id,
             content=_render(style, variant, subject.surface, first_relation, bridge.surface)),
        _row(evidence_id=second, source_id=f"{cluster}/value", family=family,
             source_cluster_id=cluster, kind="required", style=second_style,
             template_id=f"{second_style}-{(variant + 1) % 4}",
             content=_render(second_style, variant + 1, bridge.surface, second_relation, surface)),
    ]

    # Ambiguity: additional linked entities that are *not* the useful bridge.
    for index in range(2):
        decoy = _make_entity(rng, f"{task_id}#decoy{index}", regime, "config")
        rows.append(_row(
            evidence_id=f"{task_id}/dead-end-{index}", source_id=f"{cluster}/dead-end-{index}",
            family=family, source_cluster_id=cluster, kind="dead_end_link",
            style=style, template_id=f"{style}-{(variant + 2) % 4}",
            content=_render(style, variant + 2, subject.surface,
                            rng.choice(("monitor", "enclosure", "spare part")), decoy.surface),
        ))

    if opportunity == OpportunityGroup.D_SECOND_PASS_SUPPRESS:
        # A record that already answers the question directly, so a follow-up
        # would spend a retrieval call for nothing.
        rows.append(_row(
            evidence_id=f"{task_id}/direct", source_id=f"{cluster}/summary",
            family=family, source_cluster_id=cluster, kind="direct_answer",
            style=style, template_id=f"{style}-{(variant + 3) % 4}",
            content=_render(style, variant + 3, subject.surface, second_relation, surface),
        ))
    if opportunity == OpportunityGroup.C_SECOND_PASS_CONFUSING:
        # The bridge resolves to several plausible values; only the accepted
        # one is correct, so a naive second pass imports confusion.
        for index in range(2):
            _, wrong, _kind = _render_answer(rng, answer_kind, used)
            rows.append(_row(
                evidence_id=f"{task_id}/rejected-{index}", source_id=f"{cluster}/rejected-{index}",
                family=family, source_cluster_id=cluster, kind="rejected_candidate",
                style=second_style, template_id=f"{second_style}-{(variant + 2) % 4}",
                content=_render(second_style, variant + 2, bridge.surface,
                                f"proposed {second_relation}", wrong) + " This proposal was rejected.",
            ))

    question = _question(family, variant, subject.query_surface, second_relation)
    required = [first, second] if opportunity != OpportunityGroup.D_SECOND_PASS_SUPPRESS else [f"{task_id}/direct"]
    task = {
        "question": question, "answer": answer, "answer_kind": kind,
        "required_evidence_ids": required,
        "oracle_evidence_ids": required,
    }
    return task, rows, opportunity


def _build_temporal(
    rng: random.Random, *, task_id: str, family: str, cluster: str, style: str,
    regime: EntityRegime, relation: str, answer_kind: str, used: set[str],
    variant: int, chained: bool,
) -> tuple[dict, list[dict], OpportunityGroup]:
    """Supersession: an older record is present and must be rejected."""

    subject = _make_entity(rng, f"{task_id}#subject", regime, "asset")
    answer, surface, kind = _render_answer(rng, answer_kind, used)
    _, stale, _stale_kind = _render_answer(rng, answer_kind, used)
    template_id = f"{style}-{variant % 4}"
    rows: list[dict] = []
    opportunity = OpportunityGroup.A_ONE_PASS_SUFFICIENT

    if chained:
        bridge = _make_entity(rng, f"{task_id}#bridge", regime, "config")
        link_id, current_id, stale_id = (
            f"{task_id}/link", f"{task_id}/current", f"{task_id}/superseded")
        rows.append(_row(
            evidence_id=link_id, source_id=f"{cluster}/link", family=family,
            source_cluster_id=cluster, kind="required", style=style, template_id=template_id,
            content=_render(style, variant, subject.surface, "assigned unit", bridge.surface)))
        holder = bridge.surface
        required = [link_id, current_id]
        opportunity = OpportunityGroup.B_SECOND_PASS_REQUIRED
    else:
        current_id, stale_id = f"{task_id}/current", f"{task_id}/superseded"
        holder = subject.surface
        required = [current_id]

    rows.append(_row(
        evidence_id=current_id, source_id=f"{cluster}/revision-2", family=family,
        source_cluster_id=cluster, kind="required_current", style="change_log",
        template_id="change_log-0", valid_from="2031-06-01",
        content=f"Revision 2 (effective 2031-06-01) supersedes revision 1: "
                f"{holder} {relation} is now {surface}."))
    rows.append(_row(
        evidence_id=stale_id, source_id=f"{cluster}/revision-1", family=family,
        source_cluster_id=cluster, kind="superseded", style="change_log",
        template_id="change_log-1", valid_from="2030-01-01", superseded=True,
        content=f"Revision 1 (effective 2030-01-01) recorded {holder} {relation} as {stale}. "
                f"This revision has been superseded."))

    question = (
        f"As of the current revision, what {relation} applies to {subject.query_surface}?"
    )
    task = {
        "question": question, "answer": answer, "answer_kind": kind,
        "required_evidence_ids": required, "oracle_evidence_ids": required,
    }
    return task, rows, opportunity


def _build_distractor_heavy(
    rng: random.Random, *, task_id: str, family: str, cluster: str, style: str,
    regime: EntityRegime, relation: str, answer_kind: str, used: set[str],
    variant: int,
) -> tuple[dict, list[dict], OpportunityGroup]:
    """Adversarial near-duplicates varying entity, relation, status, and time."""

    subject = _make_entity(rng, f"{task_id}#subject", regime, "asset")
    answer, surface, kind = _render_answer(rng, answer_kind, used)
    required = f"{task_id}/accepted"
    template_id = f"{style}-{variant % 4}"
    rows = [_row(
        evidence_id=required, source_id=f"{cluster}/accepted", family=family,
        source_cluster_id=cluster, kind="required", style=style, template_id=template_id,
        content=_render(style, variant, subject.surface, relation, surface))]

    near_entity = _make_entity(rng, f"{task_id}#near", regime, "asset")
    variants = [
        # different entity, same relation
        (_render(style, variant, near_entity.surface, relation,
                 _render_answer(rng, answer_kind, used)[1]), "near_entity"),
        # same entity, different relation
        (_render(style, variant, subject.surface, f"calibration {relation}",
                 _render_answer(rng, answer_kind, used)[1]), "near_relation"),
        # same entity and relation, rejected status
        (_render(style, variant, subject.surface, f"proposed {relation}",
                 _render_answer(rng, answer_kind, used)[1]) + " Status: rejected.", "near_status"),
        # same entity and relation, superseded in time
        (f"Formerly, {subject.surface} {relation} was "
         f"{_render_answer(rng, answer_kind, used)[1]}; this record is historical.", "near_temporal"),
    ]
    for index, (content, kind_label) in enumerate(variants):
        rows.append(_row(
            evidence_id=f"{task_id}/near-{kind_label}", source_id=f"{cluster}/near-{index}",
            family=family, source_cluster_id=cluster, kind=f"near_duplicate_{kind_label}",
            style=style, template_id=f"{style}-{(variant + index) % 4}", content=content))

    question = _question(family, variant, subject.query_surface, relation)
    task = {
        "question": question, "answer": answer, "answer_kind": kind,
        "required_evidence_ids": [required], "oracle_evidence_ids": [required],
    }
    return task, rows, OpportunityGroup.A_ONE_PASS_SUFFICIENT


_CHAIN_RELATIONS = {
    "ownership_chain": ("registered asset", "ownership tier"),
    "configuration_chain": ("active configuration", "parameter grade"),
    "dependency_chain": ("required dependency", "release channel"),
    "location_chain": ("hosting venue", "service region"),
    "numeric_chain": ("rate record", "billing multiplier"),
    "procedural_chain": ("applicable procedure", "required action"),
}


# Records that may legitimately state the gold answer. Everything else is a
# distractor and must never contain it.
_ANSWER_BEARING_KINDS = frozenset({"required", "required_current", "direct_answer"})


def _leaks(text: str, answer: str) -> bool:
    haystack = tuple(re.findall(r"\w+", text.lower()))
    needle = tuple(re.findall(r"\w+", answer.lower()))
    width = len(needle)
    return bool(width) and any(
        haystack[index:index + width] == needle
        for index in range(len(haystack) - width + 1)
    )


def build_generalization_corpus(
    *, seed: int = 7003, tasks_per_family: int = 60, split: str = "qualification",
    held_out_styles: Sequence[str] = (), held_out_regimes: Sequence[str] = (),
) -> ControlledCorpusV3:
    """Build a v3 corpus.

    ``held_out_styles``/``held_out_regimes`` are *excluded* here, so an OOD split
    is produced by inverting the sets used for development.
    """

    if tasks_per_family < 1:
        raise ValueError("tasks_per_family must be positive")
    styles = [value for value in SOURCE_STYLES if value not in set(held_out_styles)]
    regimes = [value for value in EntityRegime if value.value not in set(held_out_regimes)]
    if not styles or not regimes:
        raise ValueError("Every source style or entity regime was held out")

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
            # Style, template variant, cluster, and entity regime are stepped by
            # strides coprime to their moduli so they vary independently. Keying
            # them all off one modulus (as v2 did) makes the grouping labels
            # aliases of each other rather than separate structure.
            cluster_index = (ordinal * 7) % CLUSTERS_PER_CORPUS
            style = styles[cluster_index % len(styles)]
            cluster = f"{style}/cluster-{cluster_index:02d}"
            # Derived from a different digit of the ordinal than the style, so
            # the two are not aliases of one another.
            variant = (ordinal // 4) % 4
            regime = regimes[(ordinal * 11) % len(regimes)]
            answer_kind = _choose_answer_kind(family, ordinal)
            used: set[str] = set()

            for _attempt in range(40):
                if family == "entity_attribute":
                    task, rows, group = _build_single_hop(
                        rng, task_id=task_id, family=family, cluster=cluster, style=style,
                        regime=regime, relation="assigned category", answer_kind=answer_kind,
                        used=set(used), variant=variant)
                elif family == "distractor_heavy":
                    task, rows, group = _build_distractor_heavy(
                        rng, task_id=task_id, family=family, cluster=cluster, style=style,
                        regime=regime, relation="routing class", answer_kind=answer_kind,
                        used=set(used), variant=variant)
                elif family == "temporal_update":
                    task, rows, group = _build_temporal(
                        rng, task_id=task_id, family=family, cluster=cluster, style=style,
                        regime=regime, relation="operating tier", answer_kind=answer_kind,
                        used=set(used), variant=variant, chained=False)
                elif family == "temporal_chain":
                    task, rows, group = _build_temporal(
                        rng, task_id=task_id, family=family, cluster=cluster, style=style,
                        regime=regime, relation="operating tier", answer_kind=answer_kind,
                        used=set(used), variant=variant, chained=True)
                else:
                    first_relation, second_relation = _CHAIN_RELATIONS[family]
                    task, rows, group = _build_chain(
                        rng, task_id=task_id, family=family, cluster=cluster, style=style,
                        regime=regime, first_relation=first_relation,
                        second_relation=second_relation, answer_kind=answer_kind,
                        used=set(used), variant=variant,
                        opportunity=opportunity_cycle[ordinal % len(opportunity_cycle)])
                # Structural leakage guard: the answer may never appear in the
                # question, nor in any *distractor*. Several records may
                # legitimately carry the answer (a group-D task is answerable
                # both directly and through its chain); what must never carry
                # it is a record designed to mislead.
                if _leaks(task["question"], task["answer"]):
                    continue
                if any(
                    _leaks(row["content"], task["answer"]) for row in rows
                    if row["metadata"]["record_kind"] not in _ANSWER_BEARING_KINDS
                ):
                    continue
                break
            else:
                raise RuntimeError(f"Could not build a leak-free task for {task_id}")

            tasks.append({
                "task_id": task_id,
                "question": task["question"],
                "answer": task["answer"],
                "required_evidence_ids": task["required_evidence_ids"],
                "oracle_evidence_ids": task["oracle_evidence_ids"],
                "family": family,
                # The template is the generative surface form actually used,
                # not whichever record happens to sort first.
                "template_id": f"{family}/{style}-{variant}",
                "source_cluster_id": cluster,
                "split": split,
                "verifier": _verifier_for(task["answer_kind"]),
                "metadata": {
                    "generator_version": GENERATOR_VERSION,
                    "generation_seed": seed,
                    "answer_kind": task["answer_kind"],
                    "entity_regime": regime.value,
                    "source_style": style,
                    "opportunity_group": group.value,
                    "iterative_family": family in ITERATIVE_FAMILIES,
                },
            })
            evidence.extend(rows)

    frozen_tasks, frozen_evidence = tuple(tasks), tuple(evidence)
    templates = {row["template_id"] for row in frozen_tasks}
    clusters = {row["source_cluster_id"] for row in frozen_tasks}
    manifest = {
        "dataset_id": f"{GENERATOR_VERSION}-{split}",
        "generation_seed": seed,
        "split": split,
        "tasks_per_family": tasks_per_family,
        "task_count": len(frozen_tasks),
        "evidence_count": len(frozen_evidence),
        "families": list(FAMILIES),
        "iterative_families": sorted(ITERATIVE_FAMILIES),
        "template_count": len(templates),
        "source_cluster_count": len(clusters),
        "source_styles": styles,
        "entity_regimes": [value.value for value in regimes],
        "held_out_styles": list(held_out_styles),
        "held_out_regimes": list(held_out_regimes),
        "answer_kinds": list(ANSWER_KINDS),
        "opportunity_groups": sorted({
            row["metadata"]["opportunity_group"] for row in frozen_tasks
        }),
        "task_sha256": _digest(frozen_tasks),
        "evidence_sha256": _digest(frozen_evidence),
        "claim_strength": "CONTROLLED_SYNTHETIC_BENCHMARK_ONLY",
        "natural_memory_claim_allowed": False,
        "purpose": "structural generalization; built to break the v2-saturating mechanism",
    }
    return ControlledCorpusV3(frozen_tasks, frozen_evidence, manifest)
