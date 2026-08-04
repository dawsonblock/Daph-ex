"""
Deterministic task verifiers for counterfactual quality scoring.

Statuses:
  CORRECT, INCORRECT, UNVERIFIABLE, EXECUTION_ERROR, TIMEOUT

IMPORTANT: verifiers operate on decoded text. Raw token IDs without
generated_text are UNVERIFIABLE (token id 42 ≠ string "42").
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

from torch import Tensor


def _normalize_text(s: str) -> str:
    s = s.strip().lower()
    s = re.sub(r"\s+", " ", s)
    return s


@dataclass
class VerifierResult:
    quality: float
    status: str

    def as_tuple(self) -> Tuple[float, str]:
        return self.quality, self.status


def _require_text(result: Dict[str, Any]) -> Optional[str]:
    """Return decoded text or None if only raw ids present."""
    if "generated_text" in result and result["generated_text"] is not None:
        t = result["generated_text"]
        if isinstance(t, list):
            return t[0] if t else ""
        return str(t)
    # Do NOT treat token IDs as text
    return None


class ExactMatchVerifier:
    """Strict equality after light normalization. Not substring / endswith."""

    def __init__(self, case_sensitive: bool = False) -> None:
        self.case_sensitive = case_sensitive

    def __call__(self, result: Dict[str, Any], task: Dict[str, Any]) -> Tuple[float, str]:
        expected = task.get("expected")
        if expected is None:
            return 0.0, "UNVERIFIABLE"
        pred = _require_text(result)
        if pred is None:
            return 0.0, "UNVERIFIABLE"
        exp = str(expected).strip()
        if not self.case_sensitive:
            pred = _normalize_text(pred)
            exp = _normalize_text(exp)
        if pred == exp:
            return 1.0, "CORRECT"
        return 0.0, "INCORRECT"


class FinalAnswerVerifier:
    """Extract last non-empty line or 'answer:' field and exact-match."""

    def __init__(self, case_sensitive: bool = False) -> None:
        self.case_sensitive = case_sensitive

    def __call__(self, result: Dict[str, Any], task: Dict[str, Any]) -> Tuple[float, str]:
        expected = task.get("expected")
        if expected is None:
            return 0.0, "UNVERIFIABLE"
        pred = _require_text(result)
        if pred is None:
            return 0.0, "UNVERIFIABLE"
        m = re.search(r"(?i)answer\s*[:=]\s*(.+)$", pred.strip())
        if m:
            ans = m.group(1).strip()
        else:
            lines = [ln.strip() for ln in pred.strip().splitlines() if ln.strip()]
            ans = lines[-1] if lines else pred.strip()
        exp = str(expected).strip()
        if not self.case_sensitive:
            ans = _normalize_text(ans)
            exp = _normalize_text(exp)
        if ans == exp:
            return 1.0, "CORRECT"
        return 0.0, "INCORRECT"


class NumericVerifier:
    """Extract last number from *decoded text* and compare to expected."""

    def __init__(self, rtol: float = 1e-5, atol: float = 1e-8) -> None:
        self.rtol = rtol
        self.atol = atol

    def __call__(self, result: Dict[str, Any], task: Dict[str, Any]) -> Tuple[float, str]:
        expected = task.get("expected")
        if expected is None:
            return 0.0, "UNVERIFIABLE"
        try:
            exp_val = float(expected)
        except (TypeError, ValueError):
            return 0.0, "UNVERIFIABLE"
        pred = _require_text(result)
        if pred is None:
            return 0.0, "UNVERIFIABLE"
        nums = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", str(pred))
        if not nums:
            return 0.0, "INCORRECT"
        try:
            pred_val = float(nums[-1])
        except ValueError:
            return 0.0, "INCORRECT"
        if abs(pred_val - exp_val) <= self.atol + self.rtol * abs(exp_val):
            return 1.0, "CORRECT"
        return 0.0, "INCORRECT"


def make_quality_fn(verifier: Any):
    def fn(out: Dict[str, Any], task: Dict[str, Any]) -> Tuple[float, str]:
        return verifier(out, task)
    return fn
