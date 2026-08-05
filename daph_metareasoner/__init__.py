"""DAPH Stage 1 marginal-utility controller research package.

LEGACY_METAREASONING — frozen legacy research system.  Active HRM research
lives exclusively in hrm_adaptive_memory/; do not add new HRM work here.
"""

from .adapters import HFCausalLMAdapter
from .analysis import (
    OracleGateConfig,
    answer_stability_policy,
    bootstrap_lcb,
    confidence_threshold_policy,
    entropy_threshold_policy,
    evaluate_offline_policy,
    frequency_matched_random_policy,
    fixed_action_policy,
    fit_family_lookup_policy,
    group_by_state,
    oracle_capture,
    oracle_value_study,
    paired_policy_gate,
    predictor_policy,
    probe_signal_gate,
    prompt_length_policy,
)
from .collector import CollectionConfig, CounterfactualExperienceCollector, load_records
from .features import FeatureNormalizer, StateVectorizer
from .models import (
    ActionValueEnsemble,
    ActionValueModel,
    ContinuationProbe,
    ProbeTrainingConfig,
    ValueTrainingConfig,
    train_probe,
    train_value_ensemble,
    training_digest,
)
from .policy import (
    ConservativeVOCPolicy, ControllerDecision, FixedRuntimePolicy,
    PolicyConfig, ThresholdRuntimePolicy,
)
from .runtime import LoopGuard, OnPathExecutor, RuntimeLimits, RuntimeResult
from .schema import (
    ALL_ACTIONS,
    NON_STOP_ACTIONS,
    Action,
    ActionReceipt,
    BranchResult,
    ExperienceRecord,
    ReasoningState,
    StopReason,
    Task,
    canonical_digest,
    records_digest,
)
from .splits import build_split_manifest, load_tasks
from .utility import UtilityConfig

__all__ = [
    "ALL_ACTIONS", "NON_STOP_ACTIONS", "Action", "ActionReceipt",
    "ActionValueEnsemble", "ActionValueModel", "BranchResult", "CollectionConfig",
    "ConservativeVOCPolicy", "ContinuationProbe", "ControllerDecision", "FixedRuntimePolicy",
    "CounterfactualExperienceCollector", "ExperienceRecord", "FeatureNormalizer",
    "HFCausalLMAdapter", "LoopGuard", "OnPathExecutor", "OracleGateConfig",
    "PolicyConfig", "ProbeTrainingConfig", "ReasoningState", "RuntimeLimits",
    "RuntimeResult", "StateVectorizer", "StopReason", "Task", "UtilityConfig",
    "ThresholdRuntimePolicy", "ValueTrainingConfig", "answer_stability_policy", "bootstrap_lcb",
    "build_split_manifest", "canonical_digest", "confidence_threshold_policy",
    "entropy_threshold_policy", "evaluate_offline_policy", "fit_family_lookup_policy",
    "fixed_action_policy",
    "frequency_matched_random_policy", "group_by_state",
    "load_records", "load_tasks", "oracle_capture", "oracle_value_study",
    "paired_policy_gate", "predictor_policy", "probe_signal_gate", "prompt_length_policy",
    "records_digest", "train_probe", "train_value_ensemble",
    "training_digest",
]
