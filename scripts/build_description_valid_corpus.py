#!/usr/bin/env python3
"""Build/freeze controlled_gate_c2_description_valid_v4. Fail-closed. Third domain."""
from __future__ import annotations
import hashlib, json, re, subprocess, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
import hrm_adaptive_memory.experiments.c2_calibration_dataset as cal
from hrm_adaptive_memory.experiments.generalization_dataset_v4 import (
    _HEADS as V4_HEADS, verify_inferable)

OUT = Path("data/hrm/controlled_gate_c2_description_valid_v4")
SPLITS = {"descv4_id": ("c2_cal_id", 99001, 50), "descv4_surface": ("c2_cal_surface", 99002, 50)}
FORBIDDEN = re.compile(r"#(s|b|v|d\d|n)\b|latent_|entity_\d+")


def main() -> None:
    if OUT.exists():
        raise FileExistsError(OUT)
    prior_heads = set(cal.HEADS) | set(cal.VOCAB_V2["HEADS"]) | set(cal.VOCAB_V3["HEADS"])
    prev = cal.apply_vocabulary(cal.VOCAB_V4D)
    try:
        print("[1/5] generating (constellation vocabulary)")
        built = {}
        for name, (partition, seed, per) in SPLITS.items():
            c = cal.build_calibration(seed=seed, partition=partition, per_regime=per)
            for t in c["tasks"]:
                t["task_id"] = t["task_id"].replace("c2cal-", "descv4-")
                t["split"] = name
            remap = {}
            for r in c["evidence"]:
                new = r["evidence_id"].replace("c2cal-", "descv4-")
                remap[r["evidence_id"]] = new; r["evidence_id"] = new
            for t in c["tasks"]:
                t["required_evidence_ids"] = [remap.get(v, v.replace("c2cal-", "descv4-"))
                                              for v in t["required_evidence_ids"]]
                t["oracle_evidence_ids"] = list(t["required_evidence_ids"])
                for e in t["_oracle_metadata"]["proof_edges"]:
                    e["record_id"] = remap.get(e["record_id"], e["record_id"].replace("c2cal-", "descv4-"))
            built[name] = c
            print(f"      {name}: {len(c['tasks'])} tasks, {len(c['evidence'])} records")

        print("[2/5] auditing")
        problems = []
        if set(cal.HEADS) & prior_heads:
            problems.append("vocabulary overlaps a prior corpus")
        if set(cal.HEADS) & set(V4_HEADS):
            problems.append("vocabulary overlaps V4")
        prior_ids = set()
        for base, splits in (("controlled_gate_a_v4", ("development", "qualification", "ood")),
                             ("controlled_gate_c2_calibration_v1",
                              ("c2_cal_id", "c2_cal_surface", "c2_cal_holdout")),
                             ("controlled_gate_c2_chain_validation_v2",
                              ("chain_v2_id", "chain_v2_surface")),
                             ("controlled_gate_c2_chain_validation_v3",
                              ("chain_v3_id", "chain_v3_surface"))):
            for sp in splits:
                p = ROOT / "data/hrm" / base / sp / "evidence.jsonl"
                if p.exists():
                    prior_ids |= {json.loads(l)["evidence_id"] for l in p.read_text().splitlines() if l.strip()}
        audit = {}
        for name, c in built.items():
            by_id = {r["evidence_id"]: r for r in c["evidence"]}
            bad = {t["task_id"]: verify_inferable(t, by_id) for t in c["tasks"]}
            bad = {k: v for k, v in bad.items() if v}
            audit[name] = {
                "not_inferable": len(bad),
                "question_leaks": sum(1 for t in c["tasks"] if cal.leaks(t["question"], t["answer"])),
                "distractor_leaks": sum(1 for t in c["tasks"] for r in c["evidence"]
                                        if r["evidence_id"].startswith(t["task_id"] + "/")
                                        and r["metadata"]["record_kind"] not in cal.ANSWER_BEARING
                                        and cal.leaks(r["content"], t["answer"])),
                "latent_leaks": sum(1 for r in c["evidence"] if FORBIDDEN.search(r["content"])),
                "id_collisions_with_prior_corpora": len({r["evidence_id"] for r in c["evidence"]} & prior_ids),
            }
            desc = audit_description_identifiability(c["tasks"], c["evidence"])
            audit[name]["description_identifiability"] = desc
            for k in ("DESCRIPTION_SURFACE_UNIQUENESS", "NORMALIZED_SURFACE_COLLISION",
                      "RUNTIME_REFERENT_IDENTIFIABILITY", "IDENTITY_EVIDENCE_SUFFICIENCY"):
                if desc[k]:
                    problems.append(f"{name}: {k}={desc[k]}")
            for k, v in audit[name].items():
                if isinstance(v, int) and v: problems.append(f"{name}: {k}={v}")
        for line in problems: print(f"      FAIL {line}")
        if problems: raise SystemExit("VALID_DESCRIPTION_V4 false; nothing written")
        print("      VALID_DESCRIPTION_V4 = true")

        print("[3/5] pytest")
        r = subprocess.run([sys.executable, "-m", "pytest", "-q"], cwd=ROOT, capture_output=True, text=True)
        if r.returncode: print(r.stdout[-1200:]); raise SystemExit("pytest failed")
        print(f"      {r.stdout.strip().splitlines()[-1]}")

        print("[4/5] writing")
        OUT.mkdir(parents=True)
        for name, c in built.items():
            d = OUT / name; d.mkdir()
            (d / "oracle_tasks.jsonl").write_text("".join(json.dumps(t, sort_keys=True) + "\n" for t in c["tasks"]))
            (d / "evidence.jsonl").write_text("".join(json.dumps(e, sort_keys=True) + "\n" for e in c["evidence"]))
            (d / "dataset_manifest.json").write_text(json.dumps(c["manifest"], sort_keys=True, indent=2) + "\n")
        digests = {str(p.relative_to(OUT)): hashlib.sha256(p.read_bytes()).hexdigest()
                   for p in sorted(OUT.rglob("*")) if p.is_file()}
        (OUT / "AUDIT.json").write_text(json.dumps({
            "generator": "controlled_gate_c2_description_valid_v4", "audit": audit,
            "problems": problems, "VALID_DESCRIPTION_V4": True,
            "state": {"purpose": "independent_replication_of_fixed_C4_mechanism",
                      "vocabulary_domain": "summits", "replaces_prior_corpora": False,
                      "holdout_touched": False, "frozen_before_evaluation": True,
                      "arm_reselection_forbidden": True, "valid": True},
            "separation": {"overlap_with_v4_heads": 0, "overlap_with_cal_v1_heads": 0,
                           "evidence_id_collisions": 0, "task_id_prefix": "descv4-"},
        }, indent=2, sort_keys=True) + "\n")
        (OUT / "RECEIPTS.sha256").write_text("".join(f"{v}  {k}\n" for k, v in sorted(digests.items())))
        print("[5/5] frozen"); print(json.dumps(audit, indent=2))
    finally:
        cal.restore_vocabulary(prev)




# --- description identifiability audits, appended for the v4 description build ---

def audit_description_identifiability(tasks, evidence):
    """Fail-closed checks that a description question has exactly one referent.

    The v1/v2/v3 description arms were invalid because four descriptor phrases
    served 50 tasks, so one phrase referred to up to 13 entities and the runtime
    could not identify a unique subject while the oracle bypassed the ambiguity
    through metadata.
    """
    import re as _re
    from collections import defaultdict

    def N(s):  # the normalization the runtime actually uses
        return " ".join(_re.findall(r"\w+", s.lower()))

    by_id = {r["evidence_id"]: r for r in evidence}
    raw_map, norm_map = defaultdict(set), defaultdict(set)
    insufficient = 0
    for t in tasks:
        if t["metadata"]["entity_regime"] != "description":
            continue
        s = t["_oracle_metadata"]["surfaces"]
        raw_map[s["subject"]].add(s["canonical"])
        norm_map[N(s["subject"])].add(N(s["canonical"]))
        # IDENTITY_EVIDENCE_SUFFICIENCY: a runtime-visible record must state the
        # mapping, containing both the description surface and the canonical name.
        ok = any(N(s["subject"]) in N(by_id[v]["content"]) and N(s["canonical"]) in N(by_id[v]["content"])
                 for v in t["required_evidence_ids"] if v in by_id)
        insufficient += 0 if ok else 1

    return {
        "DESCRIPTION_SURFACE_UNIQUENESS": sum(1 for v in raw_map.values() if len(v) > 1),
        "NORMALIZED_SURFACE_COLLISION": sum(1 for v in norm_map.values() if len(v) > 1),
        "RUNTIME_REFERENT_IDENTIFIABILITY": sum(1 for v in norm_map.values() if len(v) != 1),
        "IDENTITY_EVIDENCE_SUFFICIENCY": insufficient,
        "distinct_description_surfaces": len(raw_map),
        "description_task_count": sum(1 for t in tasks
                                      if t["metadata"]["entity_regime"] == "description"),
    }


if __name__ == "__main__":
    main()
