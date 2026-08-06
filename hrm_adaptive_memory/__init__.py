"""HRM external-memory and adaptive-compute research package.

ACTIVE_HRM_RESEARCH — the only canonical implementation.  Legacy systems:
daph/ (LEGACY_QWEN_EXFUSION) and daph_metareasoner/ (LEGACY_METAREASONING).

Gate status is machine-readable in RESEARCH_STATUS.json at the repository
root; tests fail if it disagrees with the packaged version.

The package deliberately separates retrieval, context packing, model use, and
adaptive decisions.  Learned control stays blocked until oracle-context and
counterfactual opportunity gates pass.
"""

from .baseline.evaluator import BaselineCondition, BaselineResult, OracleContextGate
from .context.packer import ContextBudget, EvidencePacket, EvidencePacker
from .controller.actions import Action, ActionOutcome, action_utilities
from .controller.policy import ControllerDecision, UtilityController
from .contracts import (
    BackendCapabilities,
    BackendHealth,
    DerivationReceipt,
    HRMStateSnapshot,
    RetrievedEvidence,
    RetrievalBackend,
    RetrievalReceipt,
    RetrievalResult,
)
from .execution.counterfactual import CounterfactualCollector, DecisionState
from .hrm.model import HRMAdapter, HRMModelSpec, PromptCondition
from .hrm.recurrent_hooks import HRMRecurrentTracer, RecurrentStateTrace
from .memory.chunking import Chunk, StructuralChunker
from .memory.lifecycle import MemoryLifecycle
from .memory.schema import MemoryRecord, MemoryStatus, MemoryType
from .memory.stores import (
    ConsolidatedMemoryStore,
    EpisodicMemoryStore,
    ProceduralMemoryStore,
    SemanticMemoryStore,
    SourceMemoryStore,
)
from .retrieval.hybrid import HybridRetriever, RetrievalCandidate

__all__ = [
    "Action", "ActionOutcome", "BackendCapabilities", "BackendHealth", "BaselineCondition",
    "BaselineResult", "Chunk", "ConsolidatedMemoryStore", "ContextBudget",
    "ControllerDecision", "CounterfactualCollector", "DecisionState", "DerivationReceipt",
    "EpisodicMemoryStore", "EvidencePacket", "EvidencePacker", "HRMAdapter",
    "HRMModelSpec", "HRMRecurrentTracer", "HRMStateSnapshot", "HybridRetriever",
    "MemoryLifecycle", "MemoryRecord", "MemoryStatus", "MemoryType", "OracleContextGate",
    "ProceduralMemoryStore", "PromptCondition", "RecurrentStateTrace", "RetrievedEvidence",
    "RetrievalBackend", "RetrievalCandidate", "RetrievalReceipt", "RetrievalResult",
    "SemanticMemoryStore", "SourceMemoryStore", "StructuralChunker", "UtilityController",
    "action_utilities",
]

__version__ = "3.7.1"

