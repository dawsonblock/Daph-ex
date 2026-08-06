"""
LEGACY_QWEN_EXFUSION — frozen legacy research system.

Active HRM research lives exclusively in hrm_adaptive_memory/.  Do not add new
HRM work here; keep existing tests passing but treat this package as read-only.

DAPH / ExFusion v3

Structured hybrid computation + adaptive effort.
Complementary operators instead of exclusive macro-routing.

Key changes from v2.3 (inspired by Kimi K3):
- LatentMoE experts (project → latent → experts → RMSNorm → up)
- Periodic recurrent + global attention hybrid blocks
- Channel-wise output gates on major paths
- Effort controller (budget-aware) + cost-aware aux loss + early-exit
- Optional Attention Residuals / block-state retrieval
- Architecture-aware DARE → TIES → Fisher merge pipeline
- RFSN integration hooks (evidence events, effort/routing/merge traces)
"""

from .config import DAPHConfigV3
from .latent_moe import LatentMoE
from .gates import ChannelGate
from .hybrid_block import HybridBlock
from .effort import (
    EffortController,
    effort_cost_loss,
    early_exit_mask_from_effort,
)
from .ssm import SelectiveSSM, register_scan_backend, dispatch_selective_scan
from .kda import KimiDeltaAttention, KDA, lower_bound_decay
from .attn_res import BlockAttnRes, AttnResBank, RMSNorm
from .model import DAPHHybridModelV3, LayerCache, ModelCache
from .attention import CausalSelfAttention
from .latent_refine import LatentRefineBlock
from .qwen_compat import QwenCompatModel, QwenCompatBlock
from .qwen_exfusion import (
    QwenExFusionModel, QwenExFusionBlock, augment_qwen_compat_model,
    gate0b_exact_parity, prepare_exfusion_for_training,
    load_qwen_exfusion_checkpoint, ExFusionParameterProvenance, TrainingInitReceipt,
    EffortProbeResult,
)
from .e3_architecture import E3RefinementConfig, E3RegionSelection, resolve_e3_region
from .layer_contribution import (
    LayerAdaptationObjective, LayerContributionConfig, LayerContributionResult,
    LayerContributionReport, LayerContributionProfiler,
    profile_selection_payload,
)
from .hard_case import HardCaseMiningConfig, HardCaseRecord, E3HardCaseMiner
from .e3_experiment import (
    E3ExperimentVariant, canonical_variant_matrix, dose_response_variants,
    location_ablation_variants, run_variant_study,
)
from .e3_training import (
    E3StageConfig, configure_e3_training, e3_verified_objective,
    VerifiedSequenceObjective, AnswerOnlyCEObjective,
    ExternalVerifiedRewardObjective, GRPOObjectiveAdapter,
)
from .compute import EffortComputeReceipt, estimate_compute
from .effort_decision import EffortDecision, ComputeStats, decide_from_probs
from .policy_trainer import (
    EffortPolicyTrainer, ShamEffortController, EffortPolicyArtifact,
    matched_random_policy, effort_frequency_matched_random, compute_matched_random, compute_matched_random_ensemble, MatchedRandomResult, MatchedRandomEnsemble,
    SplitManifest, make_split_manifest, apply_split,
    ExperimentManifest, make_experiment_manifest, make_leave_family_out_manifest,
    validate_counterfactual_dataset, install_effort_policy,
    DatasetQualificationReport, PolicyTrainingConfig, TrainingReceipt, source_tree_digest,
    evaluate_policy_utility, evaluate_policy_raw_compute, gap_capture, dataset_digest,
)
from .pretrained import (
    PretrainedImportReport, import_state_dict, load_pretrained_into_exfusion,
    zero_init_new_modules, try_load_hf_causal_lm, research_config,
    build_qwen_key_map, freeze_pretrained_keys, save_adapted_checkpoint,
)
from .train_real import (
    RealTrainConfig, TrainingStageConfig, train_adapt, load_jsonl_texts,
    eval_per_effort, distillation_loss, apply_training_stage,
)
from .verifiers import ExactMatchVerifier, FinalAnswerVerifier, NumericVerifier, make_quality_fn
from .e3_metrics import (
    E3QualificationConfig, e3_pair_metrics, qualify_e3_pairs,
    grouped_bootstrap, lambda_sweep, materialize_utility_record,
)
from .effort_frontier import build_effort_frontier, qualify_oracle_opportunity, write_effort_frontier
from .e3_protocol import (
    ExperimentTier, ProfileTier, ClaimStrength, ExperimentScale,
    EvidenceMetadata, profile_stability, promote_e3_placement,
    write_evidence_metadata, validate_profile_tier,
)
from .verified_tasks import (
    generate_verified_tasks, natural_heldout_split, calibrated_sensitivity_split,
    choose_calibration_families,
)
from .counterfactual import (
    EffortCounterfactual,
    CounterfactualCollector,
    compute_utility,
    soft_targets,
    oracle_analysis,
    qualify_effort_hierarchy,
)
from .train import TrainConfig, train_smoke
from .rfsn_hooks import (
    ExFusionEvent,
    ExFusionEmitter,
    InMemoryRFSNSink,
    ImmutableVaultSink,
    VaultRecord,
    RFSNSink,
    attach_emitter_to_model,
    AppendOnlyJSONLVault,
    PrototypeEvidenceVault,
)
from .benchmarks import (
    BenchResult,
    benchmark_hybrid_block,
    benchmark_model,
    run_quick_suite,
    print_suite,
)
from .merge import (
    extract_task_vectors,
    apply_dare_preprocessing,
    difficulty_weighted_ties_merge,
    difficulty_weighted_fisher_merge,
    build_empirical_fisher_diagonals,
    merge_task_vectors_dare_ties_fisher,
    apply_merged_task_vector,
    merge_expert_modules,
    is_ssm_core_param,
)

__version__ = "3.7.1"
__all__ = [
    "DAPHConfigV3",
    "LatentMoE",
    "ChannelGate",
    "HybridBlock",
    "EffortController",
    "effort_cost_loss",
    "early_exit_mask_from_effort",
    "SelectiveSSM",
    "register_scan_backend",
    "dispatch_selective_scan",
    "KimiDeltaAttention",
    "KDA",
    "lower_bound_decay",
    "BlockAttnRes",
    "AttnResBank",
    "RMSNorm",
    "DAPHHybridModelV3",
    "LayerCache",
    "ModelCache",
    "CausalSelfAttention",
    "LatentRefineBlock",
    "QwenCompatModel",
    "QwenCompatBlock",
    "QwenExFusionModel",
    "QwenExFusionBlock",
    "augment_qwen_compat_model",
    "gate0b_exact_parity",
    "prepare_exfusion_for_training",
    "load_qwen_exfusion_checkpoint",
    "ExFusionParameterProvenance",
    "TrainingInitReceipt",
    "EffortProbeResult",
    "E3RefinementConfig",
    "E3RegionSelection",
    "resolve_e3_region",
    "LayerAdaptationObjective",
    "LayerContributionConfig",
    "LayerContributionResult",
    "LayerContributionReport",
    "LayerContributionProfiler",
    "profile_selection_payload",
    "HardCaseMiningConfig",
    "HardCaseRecord",
    "E3HardCaseMiner",
    "E3ExperimentVariant",
    "canonical_variant_matrix",
    "dose_response_variants",
    "location_ablation_variants",
    "run_variant_study",
    "E3StageConfig",
    "configure_e3_training",
    "e3_verified_objective",
    "VerifiedSequenceObjective",
    "AnswerOnlyCEObjective",
    "ExternalVerifiedRewardObjective",
    "GRPOObjectiveAdapter",
    "EffortComputeReceipt",
    "estimate_compute",
    "EffortDecision",
    "ComputeStats",
    "decide_from_probs",
    "EffortCounterfactual",
    "CounterfactualCollector",
    "compute_utility",
    "soft_targets",
    "oracle_analysis",
    "qualify_effort_hierarchy",
    "ExactMatchVerifier",
    "FinalAnswerVerifier",
    "NumericVerifier",
    "make_quality_fn",
    "e3_pair_metrics",
    "E3QualificationConfig",
    "qualify_e3_pairs",
    "grouped_bootstrap",
    "lambda_sweep",
    "materialize_utility_record",
    "build_effort_frontier",
    "qualify_oracle_opportunity",
    "write_effort_frontier",
    "ExperimentTier",
    "ProfileTier",
    "ClaimStrength",
    "ExperimentScale",
    "EvidenceMetadata",
    "profile_stability",
    "promote_e3_placement",
    "write_evidence_metadata",
    "validate_profile_tier",
    "generate_verified_tasks",
    "natural_heldout_split",
    "calibrated_sensitivity_split",
    "choose_calibration_families",
    "PretrainedImportReport",
    "import_state_dict",
    "load_pretrained_into_exfusion",
    "zero_init_new_modules",
    "train_adapt",
    "RealTrainConfig",
    "TrainingStageConfig",
    "distillation_loss",
    "apply_training_stage",
    "save_adapted_checkpoint",
    "freeze_pretrained_keys",
    "build_qwen_key_map",
    "research_config",

    "EffortPolicyTrainer",
    "ShamEffortController",
    "matched_random_policy",
    "evaluate_policy_utility",
    "gap_capture",
    "make_leave_family_out_manifest",
    "make_experiment_manifest",
    "ExperimentManifest",
    "MatchedRandomEnsemble",
    "install_effort_policy",
    "DatasetQualificationReport",
    "PolicyTrainingConfig",
    "TrainingReceipt",
    "source_tree_digest",
    "validate_counterfactual_dataset",
    "apply_split",
    "make_split_manifest",
    "SplitManifest",
    "dataset_digest",
    "evaluate_policy_raw_compute",
    "compute_matched_random",
    "compute_matched_random_ensemble",
    "MatchedRandomResult",
    "effort_frequency_matched_random",
    "EffortPolicyArtifact",
    "TrainConfig",
    "train_smoke",
    "AppendOnlyJSONLVault",
    "PrototypeEvidenceVault",
    "ExFusionEvent",
    "ExFusionEmitter",
    "InMemoryRFSNSink",
    "ImmutableVaultSink",
    "VaultRecord",
    "RFSNSink",
    "attach_emitter_to_model",
    "BenchResult",
    "benchmark_hybrid_block",
    "benchmark_model",
    "run_quick_suite",
    "print_suite",
    "extract_task_vectors",
    "apply_dare_preprocessing",
    "difficulty_weighted_ties_merge",
    "difficulty_weighted_fisher_merge",
    "build_empirical_fisher_diagonals",
    "merge_task_vectors_dare_ties_fisher",
    "apply_merged_task_vector",
    "merge_expert_modules",
    "is_ssm_core_param",
]

# Optional mamba-ssm scan backend (no-op if not installed)
try:
    from .mamba_backend import register_mamba_ssm_backend, try_enable_mamba_backend  # noqa: F401
except Exception:
    pass
