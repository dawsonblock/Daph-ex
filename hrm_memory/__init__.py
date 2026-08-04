"""Deprecated compatibility namespace for :mod:`hrm_adaptive_memory`.

Release 3.6 retains these aliases for one release. New code must import the
canonical package. The compatibility namespace is scheduled for removal in
3.7.
"""

from __future__ import annotations

import importlib
import sys
import warnings

warnings.warn(
    "hrm_memory is deprecated; import hrm_adaptive_memory instead",
    DeprecationWarning,
    stacklevel=2,
)

_MODULES = (
    "baseline", "baseline.evaluator", "baseline.metrics",
    "context", "context.packer",
    "controller", "controller.actions", "controller.policy",
    "execution", "execution.counterfactual", "execution.oracle",
    "hrm", "hrm.model", "hrm.recurrent_hooks", "hrm.variable_recurrence",
    "memory", "memory.chunking", "memory.contradiction", "memory.schema", "memory.stores",
    "retrieval", "retrieval.dense", "retrieval.evaluator", "retrieval.hybrid",
    "retrieval.lexical", "retrieval.reranker",
)

for _suffix in _MODULES:
    _module = importlib.import_module(f"hrm_adaptive_memory.{_suffix}")
    sys.modules[f"{__name__}.{_suffix}"] = _module
    if "." not in _suffix:
        globals()[_suffix] = _module

from hrm_adaptive_memory import *  # noqa: F401,F403,E402
from hrm_adaptive_memory import __all__
