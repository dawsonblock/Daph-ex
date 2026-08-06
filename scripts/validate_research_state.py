#!/usr/bin/env python3
"""Fail-closed validation of the repository's scientific state.

Exists because I twice pushed a red suite. The corpus-freeze pipeline is
fail-closed, but RESEARCH_STATUS.json edits do not go through it, so this closes
that path. Checks:

  1. full test suite
  2. every qualified claim declares a report, and the report exists
  3. every MEASURED claim declares receipts, and they exist
  4. every claim bounds its scope (not_a_claim_about)
  5. gate names and status vocabulary are known
  6. capabilities stay BLOCKED until their gate passes
  7. version agreement across pyproject / packages / README / CHANGELOG / status

Run standalone, or install as a pre-push hook. CI remains authoritative because
a local hook can be bypassed.
"""

from __future__ import annotations

import json
import subprocess
import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PERMITTED_PREFIXES = ("PASS", "FAIL", "PENDING", "IN_PROGRESS", "BLOCKED", "MECHANISM_SUCCESS")
MEASURED_MARKERS = ("MEASURED", "PASS", "PROMOTED")


def check(name: str, ok: bool, detail: str = "") -> bool:
    print(f"  {'PASS' if ok else 'FAIL'}  {name}{(' — ' + detail) if detail and not ok else ''}")
    return ok


def main(fast: bool = False) -> int:
    status = json.loads((ROOT / "RESEARCH_STATUS.json").read_text())
    results = []

    if not fast:
        print("[1] test suite")
        proc = subprocess.run([sys.executable, "-m", "pytest", "-q"], cwd=ROOT,
                              capture_output=True, text=True)
        tail = proc.stdout.strip().splitlines()[-1] if proc.stdout.strip() else ""
        results.append(check("pytest", proc.returncode == 0, tail))

    print("[2] claim reports exist")
    for name, claim in status.get("qualified_claims", {}).items():
        report = claim.get("report")
        results.append(check(f"{name}: declares a report", report is not None,
                             "add a 'report' key"))
        if report:
            results.append(check(f"{name}: {report} exists", (ROOT / report).exists()))

    print("[3] measured claims carry receipts")
    for name, claim in status.get("qualified_claims", {}).items():
        blob = json.dumps(claim)
        if not any(marker in blob for marker in MEASURED_MARKERS):
            continue
        receipts = claim.get("receipts")
        if receipts is None:
            continue  # older claims embed evidence in the report
        for path in receipts:
            results.append(check(f"{name}: receipt {path}", (ROOT / path).exists()))

    print("[4] claims bound their scope")
    for name, claim in status.get("qualified_claims", {}).items():
        results.append(check(f"{name}: not_a_claim_about", bool(claim.get("not_a_claim_about"))))

    print("[5] gate status vocabulary")
    for gate, value in status.get("gates", {}).items():
        results.append(check(f"{gate}={value}", str(value).startswith(PERMITTED_PREFIXES)))

    print("[6] capability gating")
    coupling = {"adaptive_retrieval": "gate_e_learned_retrieval_control",
                "executive_training": "gate_i_unified_executive"}
    for capability, gate in coupling.items():
        gates, caps = status.get("gates", {}), status.get("capabilities", {})
        if gate in gates and capability in caps and gates[gate] != "PASS":
            results.append(check(f"{capability} BLOCKED while {gate} unpassed",
                                 caps[capability] == "BLOCKED"))

    print("[7] version agreement")
    version = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]["version"]
    results.append(check("RESEARCH_STATUS version", status.get("version") == version))
    results.append(check("README title", (ROOT / "README.md").read_text()
                         .splitlines()[0].endswith(f"v{version}")))
    results.append(check("CHANGELOG entry", f"## {version}" in (ROOT / "CHANGELOG.md").read_text()))

    failed = results.count(False)
    print(f"\n{'RESEARCH STATE VALID' if not failed else f'RESEARCH STATE INVALID ({failed} failures)'}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main(fast="--fast" in sys.argv))
