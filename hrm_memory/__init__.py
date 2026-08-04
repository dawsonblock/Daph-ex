"""HRM external-memory and adaptive-compute research package.

The package deliberately separates retrieval, context packing, model use, and
adaptive decisions.  Learned control stays blocked until oracle-context and
counterfactual opportunity gates pass.
"""

from .baseline.evaluator import BaselineCondition, BaselineResult, OracleContextGate
from .context.packer import ContextBudget, EvidencePacket, EvidencePacker
from .controller.actions import Action, ActionOutcome, action_utilities
from .controller.policy import ControllerDecision, UtilityController
from .execution.counterfactual import CounterfactualCollector, DecisionState
from .hrm.model import HRMAdapter, HRMModelSpec, PromptCondition
from .hrm.recurrent_hooks import HRMRecurrentTracer, RecurrentStateTrace
from .memory.chunking import Chunk, StructuralChunker
from .memory.schema import MemoryRecord, MemoryStatus, MemoryType
from .memory.stores import EpisodicMemoryStore, SemanticMemoryStore, SourceMemoryStore
from .retrieval.hybrid import HybridRetriever, RetrievalCandidate

__all__ = [
    "Action", "ActionOutcome", "BaselineCondition", "BaselineResult", "Chunk",
    "ContextBudget", "ControllerDecision", "CounterfactualCollector", "DecisionState",
    "EpisodicMemoryStore", "EvidencePacket", "EvidencePacker", "HRMAdapter",
    "HRMModelSpec", "HRMRecurrentTracer", "HybridRetriever", "MemoryRecord",
    "MemoryStatus", "MemoryType", "OracleContextGate", "PromptCondition",
    "RecurrentStateTrace", "RetrievalCandidate", "SemanticMemoryStore",
    "SourceMemoryStore", "StructuralChunker", "UtilityController", "action_utilities",
]
