"""controlled_gate_c2_calibration_v1 — component-selection corpus for Gate C2.

Exists because V4's development split contains only the canonical and
abbreviation regimes, so it structurally cannot observe the capability that
distinguishes BGE from BM25 (description). Selecting a retriever on V4
development would discard BGE for the wrong reason; selecting on V4 OOD would
burn the confirmation set.

This corpus does NOT replace V4. Its sole purpose is component selection, and
V4 stays immutable.

Hard separation from V4 is enforby construction and by test:
  * disjoint entity head vocabulary
  * disjoint role, descriptor, and answer vocabularies
  * distinct surface renderings
  * distinct task-id prefix (`c2cal-`)
  * distinct evidence-id namespace
  * new RNG seeds

Partitions make the structural issue explicit rather than letting aggregates
conceal it:
  C2-CAL-ID       canonical + abbreviation   (BM25's strong regimes)
  C2-CAL-SURFACE  alias + description        (where surface form varies)

A third partition, C2-CAL-HOLDOUT, is reserved and must not be used for
selection: selecting on calibration burns calibration exactly as using V4 OOD
burned it, so a pristine set is set aside up front.
"""

from __future__ import annotations

import hashlib
import json
import random
import re
from dataclasses import dataclass
from enum import Enum
from typing import Any, Mapping, Sequence

GENERATOR_VERSION = "controlled-gate-c2-calibration-v1"

# --- vocabulary, deliberately disjoint from V4 ------------------------------
# V4 uses bird names (Nimbus, Bluebird, Kestrel, ...). These are minerals.
HEADS = (
    "Basalt", "Calcite", "Dolomite", "Feldspar", "Gypsum", "Halite", "Jasper",
    "Kyanite", "Limonite", "Magnetite", "Nephrite", "Olivine", "Pyrite",
    "Quartzite", "Rhodonite", "Serpentine", "Topaz", "Ulexite", "Variscite",
    "Wollastonite", "Xenotime", "Zircon",
)
ROLES = ("intake valve", "coolant loop", "telemetry bus", "ballast tank",
         "actuator ring", "diagnostic bay")
DESCRIPTORS = ("tertiary unit catalogued during retrofit",
               "backup unit entered in the transfer ledger",
               "spare unit noted on the dispatch sheet",
               "replacement unit filed with the survey report")
SYMBOLIC = ("LAMBDA-OCHRE", "MU-VERMILION", "NU-CERULEAN", "XI-SIENNA",
            "OMICRON-MAUVE", "PI-VIRIDIAN", "RHO-UMBER", "SIGMA-AZURE")
ENUM = ("commissioned", "derated", "impounded", "mothballed", "recertified")
BOOLEAN = ("yes", "no")
JSON_KEYS = ("band", "class", "phase", "rating")

SOURCE_STYLES = ("survey_report", "dispatch_sheet", "transfer_ledger",
                 "retrofit_note", "audit_trail", "field_memo")

FAMILIES = ("entity_attribute", "ownership_chain", "configuration_chain",
            "dependency_chain", "location_chain", "temporal_chain",
            "numeric_chain", "procedural_chain", "temporal_update",
            "distractor_heavy")
ITERATIVE = frozenset({"ownership_chain", "configuration_chain", "dependency_chain",
                       "location_chain", "temporal_chain", "numeric_chain",
                       "procedural_chain"})
CHAIN_RELATIONS = {
    "ownership_chain": ("catalogued asset", "custody band"),
    "configuration_chain": ("active preset", "calibration class"),
    "dependency_chain": ("required component", "supply phase"),
    "location_chain": ("hosting bay", "operating district"),
    "numeric_chain": ("tariff record", "settlement factor"),
    "procedural_chain": ("governing routine", "mandated step"),
}
ANSWER_KINDS = ("numeric", "symbolic", "enum", "boolean", "json_field")
CAPACITY = {"numeric": 8000, "symbolic": len(SYMBOLIC), "enum": len(ENUM),
            "boolean": len(BOOLEAN), "json_field": len(ENUM)}
DEMAND = {"distractor_heavy": 5, "temporal_update": 2, "temporal_chain": 2}
ANSWER_BEARING = frozenset({"required", "required_current", "direct_answer"})


class Regime(str, Enum):
    CANONICAL = "canonical"
    ABBREVIATION = "abbreviation"
    ALIAS = "alias"
    DESCRIPTION = "description"


PARTITIONS = {
    "c2_cal_id": (Regime.CANONICAL, Regime.ABBREVIATION),
    "c2_cal_surface": (Regime.ALIAS, Regime.DESCRIPTION),
    "c2_cal_holdout": (Regime.CANONICAL, Regime.ABBREVIATION, Regime.ALIAS, Regime.DESCRIPTION),
}


@dataclass(frozen=True)
class Entity:
    latent_id: str
    canonical: str
    alias: str
    abbreviation: str
    description: str

    def surface(self, regime: Regime) -> str:
        return {Regime.CANONICAL: self.canonical, Regime.ALIAS: self.alias,
                Regime.ABBREVIATION: self.abbreviation,
                Regime.DESCRIPTION: f"the {self.description}"}[regime]


def make_entity(rng, latent_id, used):
    for _ in range(200):
        head, alias_head = rng.sample(HEADS, 2)
        if head not in used and alias_head not in used:
            used.update({head, alias_head})
            role = rng.choice(ROLES)
            initials = "".join(w[0].upper() for w in role.split())
            serial = rng.randrange(1000, 9999)
            return Entity(latent_id, f"{head} {role}",
                          f"{alias_head} {rng.choice(ROLES).split()[0]}",
                          f"{head[0].upper()}{initials}-{rng.randrange(10, 99)}",
                          f"{rng.choice(DESCRIPTORS)} under docket {serial}")
    raise RuntimeError("exhausted heads")


def render(style, variant, subject, relation, obj):
    """Surface renderings distinct from V4's, so templates do not overlap."""
    table = {
        "survey_report": (
            f"Survey finding: {subject} carries {relation} {obj}.",
            f"The survey lists {relation} for {subject} as {obj}.",
            f"Surveyed — {subject} / {relation} / {obj}.",
            f"Under survey, {obj} was recorded as the {relation} of {subject}.",
        ),
        "dispatch_sheet": (
            f"Dispatch: {subject} :: {relation} :: {obj}",
            f"Sheet line — {subject} shows {relation} {obj}.",
            f"{relation} for dispatch of {subject}: {obj}",
            f"Dispatched {subject} with {relation} set at {obj}.",
        ),
        "transfer_ledger": (
            f"Ledger: {relation}({subject}) = {obj}",
            f"Transfer record | {subject} | {relation} | {obj}",
            f'{{"unit": "{subject}", "{relation}": "{obj}"}}',
            f"{relation}\n  unit: {subject}\n  entry: {obj}",
        ),
        "retrofit_note": (
            f"Retrofit note: after work, {subject} reports {relation} {obj}.",
            f"Post-retrofit, the {relation} of {subject} stands at {obj}.",
            f"Noted during retrofit — {subject} to {obj} for {relation}.",
            f"Retrofit outcome for {subject}: {relation} is {obj}.",
        ),
        "audit_trail": (
            f"[audit] {subject} {relation} <- {obj}",
            f"Audit trail entry: {relation} of {subject} amended to {obj}.",
            f"- audited {subject}: {relation} = {obj}",
            f"Audit confirms {obj} as the {relation} held by {subject}.",
        ),
        "field_memo": (
            f"Memo — just to note, {subject} has {relation} {obj}.",
            f"From the field: {subject}'s {relation} came back as {obj}.",
            f"Passing along that {relation} for {subject} is {obj}.",
            f"Field memo: {subject} -> {relation} -> {obj}.",
        ),
    }
    if style not in table:
        raise ValueError(style)
    return table[style][variant % 4]


IDENTITY = {
    Regime.ALIAS: (
        "{other} is the working designation of {canonical}.",
        "Cross-reference: {other} denotes {canonical}.",
        "{other} and {canonical} are the same unit under two names.",
        "Recorded alternate label {other} for {canonical}.",
    ),
    Regime.ABBREVIATION: (
        "{other} abbreviates {canonical}.",
        "Short-code table: {other} stands for {canonical}.",
        "Written briefly, {canonical} appears as {other}.",
        "{other} is the compact form of {canonical}.",
    ),
    Regime.DESCRIPTION: (
        "The transfer ledger names {canonical} as {other}.",
        "{other} is {canonical}, per the survey report.",
        "Retrofit records describe {canonical} as {other}.",
        "It was {canonical} that served as {other}.",
    ),
}


def question(variant, subject, relation):
    return (f"Which {relation} is held by {subject}?",
            f"What {relation} does {subject} carry?",
            f"State the {relation} recorded for {subject}.",
            f"Report the {relation} attached to {subject}.")[variant % 4]


def value(rng, kind, used):
    for _ in range(80):
        if kind == "symbolic":
            v = rng.choice(SYMBOLIC); s = v
        elif kind == "enum":
            v = rng.choice(ENUM); s = v
        elif kind == "boolean":
            v = rng.choice(BOOLEAN); s = v
        elif kind == "json_field":
            v = rng.choice(ENUM); s = f'{{"{rng.choice(JSON_KEYS)}": "{v}"}}'
        else:
            v = str(rng.randrange(1000, 9999)); s = v
        if v not in used:
            used.add(v); return v, s
    raise RuntimeError("exhausted answers")


def leaks(text, answer):
    h = tuple(re.findall(r"\w+", text.lower())); n = tuple(re.findall(r"\w+", answer.lower()))
    w = len(n)
    return bool(w) and any(h[i:i + w] == n for i in range(len(h) - w + 1))


# ---------------------------------------------------------------------------
# Task construction. Same proof-graph shape as V4 so the audit is shared.
# ---------------------------------------------------------------------------

def _row(eid, cluster, content, family, kind, style, variant, superseded=False):
    return {"evidence_id": eid, "source_id": f"{cluster}/{kind}",
            "source_type": f"c2cal_{style}", "content": content,
            "metadata": {"generator_version": GENERATOR_VERSION, "family": family,
                         "source_cluster_id": cluster, "record_kind": kind,
                         "source_style": style, "template_id": f"{style}-{variant % 4}",
                         "superseded": superseded}}


def _identity_row(tid, cluster, family, regime, variant, ent, style):
    return _row(f"{tid}/identity", cluster,
                IDENTITY[regime][variant % 4].format(other=ent.surface(regime),
                                                     canonical=ent.canonical),
                family, "required_identity", style, variant)


def build_task(rng, *, tid, family, regime, styles, cluster, variant, kind, heads):
    """Return (task, rows). Mirrors V4's shape: proof graph + required set."""
    subject = make_entity(rng, f"{tid}#s", heads)
    used: set[str] = set()
    answer, surface = value(rng, kind, used)
    rows: list[dict] = []
    edges: list[dict] = []
    required: list[str] = []
    bridge = None

    def E(rid, src, rel, tgt):
        edges.append({"record_id": rid, "source": src, "relation": rel, "target": tgt})

    if family in CHAIN_RELATIONS:
        first, second = CHAIN_RELATIONS[family]
        bridge = make_entity(rng, f"{tid}#b", heads)
        s1, s2 = rng.choice(styles), rng.choice(styles)
        rows.append(_row(f"{tid}/link", cluster,
                         render(s1, variant, subject.canonical, first, bridge.canonical),
                         family, "required", s1, variant))
        rows.append(_row(f"{tid}/value", cluster,
                         render(s2, variant + 1, bridge.canonical, second, surface),
                         family, "required", s2, variant + 1))
        E(f"{tid}/link", subject.latent_id, first, bridge.latent_id)
        E(f"{tid}/value", bridge.latent_id, second, f"{tid}#v")
        required += [f"{tid}/link", f"{tid}/value"]
        relation = second
        for i in range(2):
            decoy = make_entity(rng, f"{tid}#d{i}", heads)
            rows.append(_row(f"{tid}/dead-end-{i}", cluster,
                             render(rng.choice(styles), variant + 2, subject.canonical,
                                    rng.choice(("mounted gauge", "spare bracket")),
                                    decoy.canonical),
                             family, "dead_end_link", rng.choice(styles), variant + 2))
    elif family == "temporal_update":
        relation = "operating band"
        _, stale = value(rng, kind, used)
        cs, ss = rng.choice(styles), rng.choice(styles)
        rows.append(_row(f"{tid}/current", cluster,
                         "Revision 4 (effective 2032-03-01) supersedes revision 3: "
                         + render(cs, variant, subject.canonical, relation, surface),
                         family, "required_current", cs, variant))
        rows.append(_row(f"{tid}/superseded", cluster,
                         "Revision 3 (effective 2031-01-01, since superseded) recorded: "
                         + render(ss, variant, subject.canonical, relation, stale),
                         family, "superseded", ss, variant, superseded=True))
        E(f"{tid}/current", subject.latent_id, relation, f"{tid}#v")
        required.append(f"{tid}/current")
    elif family == "distractor_heavy":
        relation = "routing band"
        near = make_entity(rng, f"{tid}#n", heads)
        st = rng.choice(styles)
        rows.append(_row(f"{tid}/accepted", cluster,
                         render(st, variant, subject.canonical, relation, surface),
                         family, "required", st, variant))
        E(f"{tid}/accepted", subject.latent_id, relation, f"{tid}#v")
        required.append(f"{tid}/accepted")
        for label, extra in (("near_entity", lambda s: render(s, variant, near.canonical, relation, value(rng, kind, used)[1])),
                             ("near_relation", lambda s: render(s, variant, subject.canonical, f"calibration {relation}", value(rng, kind, used)[1])),
                             ("near_status", lambda s: render(s, variant, subject.canonical, f"proposed {relation}", value(rng, kind, used)[1]) + " Status: declined."),
                             ("near_temporal", lambda s: render(s, variant, subject.canonical, f"former {relation}", value(rng, kind, used)[1]) + " Historical record.")):
            sty = rng.choice(styles)
            rows.append(_row(f"{tid}/near-{label}", cluster, extra(sty), family,
                             f"near_duplicate_{label}", sty, variant))
    else:  # entity_attribute
        relation = "assigned band"
        st = rng.choice(styles)
        rows.append(_row(f"{tid}/fact", cluster,
                         render(st, variant, subject.canonical, relation, surface),
                         family, "required", st, variant))
        E(f"{tid}/fact", subject.latent_id, relation, f"{tid}#v")
        required.append(f"{tid}/fact")

    if regime != Regime.CANONICAL:
        sty = rng.choice(styles)
        rows.append(_identity_row(tid, cluster, family, regime, variant, subject, sty))
        edges.insert(0, {"record_id": f"{tid}/identity",
                         "source": f"surface:{subject.surface(regime)}",
                         "relation": "refers_to", "target": subject.latent_id})
        required.insert(0, f"{tid}/identity")

    q = question(variant, subject.surface(regime), relation)
    task = {"task_id": tid, "question": q, "answer": answer,
            "required_evidence_ids": required, "oracle_evidence_ids": required,
            "family": family, "template_id": f"{family}/{rng.choice(styles)}-{variant}",
            "source_cluster_id": cluster, "split": "",
            "verifier": "numeric" if kind == "numeric" else "canonical",
            "metadata": {"generator_version": GENERATOR_VERSION, "answer_kind": kind,
                         "entity_regime": regime.value, "source_style": rows[0]["metadata"]["source_style"],
                         "opportunity_group": ("B_SECOND_PASS_REQUIRED" if family in CHAIN_RELATIONS
                                               else "A_ONE_PASS_SUFFICIENT"),
                         "iterative_family": family in ITERATIVE},
            "_oracle_metadata": {"latent_subject": subject.latent_id,
                                 "latent_bridge": bridge.latent_id if bridge else None,
                                 "target_relation": relation, "answer_node": f"{tid}#v",
                                 "proof_edges": edges,
                                 "surfaces": {"subject": subject.surface(regime),
                                              "canonical": subject.canonical,
                                              **({"bridge": bridge.canonical} if bridge else {})}}}
    return task, rows


def build_calibration(*, seed: int, partition: str, per_regime: int) -> dict:
    """Build one calibration partition. Regimes are balanced by construction."""
    if partition not in PARTITIONS:
        raise ValueError(f"unknown partition {partition!r}")
    regimes = PARTITIONS[partition]
    rng = random.Random(seed)
    tasks, evidence = [], []
    ordinal = 0
    for regime in regimes:
        for i in range(per_regime):
            family = FAMILIES[i % len(FAMILIES)]
            kind = next(k for k in (ANSWER_KINDS[i % len(ANSWER_KINDS)], "numeric")
                        if CAPACITY[k] >= DEMAND.get(family, 3))
            cidx = (ordinal * 7) % 24
            style = SOURCE_STYLES[cidx % len(SOURCE_STYLES)]
            cluster = f"{style}/c2cal-cluster-{cidx:02d}"
            variant = (ordinal // 4) % 4
            tid = f"c2cal-{regime.value}-{family}-{i:04d}"
            for _ in range(40):
                heads: set[str] = set()
                task, rows = build_task(rng, tid=tid, family=family, regime=regime,
                                        styles=list(SOURCE_STYLES), cluster=cluster,
                                        variant=variant, kind=kind, heads=heads)
                if leaks(task["question"], task["answer"]):
                    continue
                if any(leaks(r["content"], task["answer"]) for r in rows
                       if r["metadata"]["record_kind"] not in ANSWER_BEARING):
                    continue
                break
            else:
                raise RuntimeError(f"could not build leak-free {tid}")
            task["split"] = partition
            tasks.append(task); evidence.extend(rows); ordinal += 1

    def digest(rows):
        return hashlib.sha256("\n".join(
            json.dumps(r, sort_keys=True, separators=(",", ":")) for r in rows).encode()).hexdigest()

    return {"tasks": tasks, "evidence": evidence, "manifest": {
        "dataset_id": f"{GENERATOR_VERSION}-{partition}", "partition": partition,
        "generation_seed": seed, "per_regime": per_regime,
        "task_count": len(tasks), "evidence_count": len(evidence),
        "regimes": [r.value for r in regimes],
        "families": sorted({t["family"] for t in tasks}),
        "source_styles": sorted({r["metadata"]["source_style"] for r in evidence}),
        "answer_kinds": sorted({t["metadata"]["answer_kind"] for t in tasks}),
        "task_sha256": digest(tasks), "evidence_sha256": digest(evidence),
        "purpose": ("component selection only; does not replace V4"
                    if partition != "c2_cal_holdout"
                    else "RESERVED: never use for selection"),
    }}


# ---------------------------------------------------------------------------
# Vocabulary override, for building an independent replication corpus.
# ---------------------------------------------------------------------------

VOCAB_V2 = {
    # V4 used birds, calibration v1 minerals; this is a third unrelated domain.
    "HEADS": ("Andromeda", "Bootes", "Cassiopeia", "Draco", "Eridanus", "Fornax",
              "Grus", "Hydrus", "Indus", "Lacerta", "Monoceros", "Norma",
              "Octans", "Perseus", "Reticulum", "Sculptor", "Tucana", "Vela",
              "Volans", "Aquila", "Carina", "Lyra"),
    "ROLES": ("beacon array", "drift collar", "phase gate", "shroud panel",
              "vector spar", "yaw damper"),
    "DESCRIPTORS": ("quaternary unit logged during shakedown",
                    "standby unit indexed in the berth manifest",
                    "relief unit cited in the commissioning brief",
                    "substitute unit filed under the trials docket"),
    "SYMBOLIC": ("TAU-GARNET", "UPSILON-FLAX", "PHI-COBALT", "CHI-SORREL",
                 "PSI-JADE", "OMEGA-RUST", "ALPHA2-BONE", "BETA2-PLUM"),
    "ENUM": ("provisioned", "quiesced", "restaged", "sequestered", "validated"),
    "BOOLEAN": ("affirmative", "negative"),
    "JSON_KEYS": ("grade2", "tier2", "state2", "band2"),
}


def apply_vocabulary(vocab: Mapping[str, Any]) -> dict[str, Any]:
    """Rebind surface vocabulary; returns the previous values for restoration.

    Structure and audit logic are shared with calibration v1 deliberately; only
    the surface forms differ, so a replication cannot succeed by reusing learned
    vocabulary.
    """

    global HEADS, ROLES, DESCRIPTORS, SYMBOLIC, ENUM, BOOLEAN, JSON_KEYS, CAPACITY
    previous = {"HEADS": HEADS, "ROLES": ROLES, "DESCRIPTORS": DESCRIPTORS,
                "SYMBOLIC": SYMBOLIC, "ENUM": ENUM, "BOOLEAN": BOOLEAN,
                "JSON_KEYS": JSON_KEYS, "CAPACITY": CAPACITY}
    HEADS = tuple(vocab["HEADS"]); ROLES = tuple(vocab["ROLES"])
    DESCRIPTORS = tuple(vocab["DESCRIPTORS"]); SYMBOLIC = tuple(vocab["SYMBOLIC"])
    ENUM = tuple(vocab["ENUM"]); BOOLEAN = tuple(vocab["BOOLEAN"])
    JSON_KEYS = tuple(vocab["JSON_KEYS"])
    CAPACITY = {"numeric": 8000, "symbolic": len(SYMBOLIC), "enum": len(ENUM),
                "boolean": len(BOOLEAN), "json_field": len(ENUM)}
    return previous


def restore_vocabulary(previous: Mapping[str, Any]) -> None:
    global HEADS, ROLES, DESCRIPTORS, SYMBOLIC, ENUM, BOOLEAN, JSON_KEYS, CAPACITY
    HEADS = previous["HEADS"]; ROLES = previous["ROLES"]
    DESCRIPTORS = previous["DESCRIPTORS"]; SYMBOLIC = previous["SYMBOLIC"]
    ENUM = previous["ENUM"]; BOOLEAN = previous["BOOLEAN"]
    JSON_KEYS = previous["JSON_KEYS"]; CAPACITY = previous["CAPACITY"]


VOCAB_V3 = {
    # Fourth unrelated domain: V4 birds, cal_v1 minerals, chain_v2 constellations.
    "HEADS": ("Amazon", "Brahmaputra", "Congo", "Danube", "Ebro", "Fraser", "Ganges",
              "Hudson", "Irrawaddy", "Jordan", "Kolyma", "Loire", "Mekong", "Niger",
              "Orinoco", "Parana", "Rhone", "Severn", "Tagus", "Ural", "Volga", "Yukon"),
    "ROLES": ("sluice head", "weir deck", "levee mount", "culvert ring",
              "spillway arm", "penstock yoke"),
    "DESCRIPTORS": ("ancillary unit charted during survey season",
                    "held unit entered on the lockkeeper roll",
                    "reserve unit named in the dredging order",
                    "alternate unit filed with the catchment review"),
    "SYMBOLIC": ("IOTA-BRONZE", "KAPPA-LINEN", "LAMBDA2-SLATE", "MU2-CLAY",
                 "NU2-MOSS", "XI2-ASH", "PI2-CHALK", "RHO2-PEAT"),
    "ENUM": ("chartered", "dredged", "impeded", "reserved", "surveyed"),
    "BOOLEAN": ("confirmed", "declined"),
    "JSON_KEYS": ("grade3", "tier3", "state3", "band3"),
}


VOCAB_V4D = {
    # Fifth domain: birds, minerals, constellations, rivers, now summits.
    "HEADS": ("Aconcagua", "Belukha", "Chimborazo", "Denali", "Elbrus", "Fuji",
              "Gasherbrum", "Huascaran", "Illimani", "Jaya", "Kazbek", "Logan",
              "Makalu", "Nanda", "Ojos", "Pobeda", "Rainier", "Shasta",
              "Tambora", "Ushba", "Vinson", "Whitney"),
    "ROLES": ("anchor cleat", "brake shoe", "cable drum", "guide rail",
              "hoist frame", "tension block"),
    "DESCRIPTORS": ("unit inspected during the ascent audit",
                    "unit registered on the base camp roll",
                    "unit cited in the traverse report",
                    "unit filed with the ridge survey"),
    "SYMBOLIC": ("SIGMA2-FLINT", "TAU2-OCHRE", "UPSILON2-BIRCH", "PHI2-SABLE",
                 "CHI2-IVORY", "PSI2-CEDAR", "OMEGA2-BASALT", "ALPHA3-DUNE"),
    "ENUM": ("audited", "cleared", "flagged", "staged", "withdrawn"),
    "BOOLEAN": ("true2", "false2"),
    "JSON_KEYS": ("grade4", "tier4", "state4", "band4"),
}
