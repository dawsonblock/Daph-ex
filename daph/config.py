"""Configuration for DAPH / ExFusion v3."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class DAPHConfigV3:
    # Core dimensions
    hidden_size: int = 768
    intermediate_size: Optional[int] = None  # None → 2*hidden; set for Qwen match
    latent_size: int = 384                # LatentMoE width (≈ 0.5× hidden)
    num_attention_heads: int = 12
    num_key_value_heads: Optional[int] = None  # GQA; None → = num_attention_heads
    use_rope: bool = False
    attention_bias: bool = False
    rope_theta: float = 10000.0
    max_position_embeddings: int = 8192
    norm_type: str = "layer"  # "layer" | "rms"
    rms_norm_eps: float = 1e-6
    shared_ffn: str = "swiglu"  # "swiglu" | "gelu_mlp"
    state_size: int = 16                  # SSM / recurrent state

    # Hybrid block structure
    num_recurrent_per_block: int = 3      # K3-style 3:1
    use_global_attention: bool = True
    recurrent_type: str = "ssm"           # "ssm" | "kda"
    kda_num_heads: int = 4
    kda_g_min: float = -5.0               # lower bound for forget gate

    # LatentMoE
    num_routed_experts: int = 16
    num_shared_experts: int = 2
    top_k_experts: int = 2
    moe_dropout: float = 0.0
    moe_activation: str = "swiglu"     # "swiglu" | "situ"
    moe_beta_gate: float = 1.5
    moe_beta_up: float = 1.5
    use_load_balancing: bool = True       # expert-bias load feedback (integral controller)
    # DEPRECATED alias of use_load_balancing — kept for API compatibility only
    use_quantile_balancing: bool = True
    qb_quantile: float = 0.75             # unused by current controller; reserved
    qb_bias_lr: float = 0.01


    # Routing / effort
    effort_levels: int = 4                # 0=cheap … 3=max
    max_latent_steps: int = 4             # E3 shared-weight refinement budget
    default_e3_steps: int = 2
    early_exit_threshold: float = 0.55

    enable_channel_gates: bool = True

    # Attention
    attn_history_window: Optional[int] = None
    attn_sink_tokens: int = 4
    mask_convention: str = "hf"

    # Depth mixing (AttnRes)
    use_attn_res: bool = False
    num_attn_res_blocks: int = 8
    attn_res_num_heads: int = 4
    attn_res_detach_history: bool = True  # stop-grad depth retrieval by default


    # Model
    num_layers: int = 12                  # number of HybridBlocks
    vocab_size: int = 32000
    dropout: float = 0.1
    tie_word_embeddings: bool = True

    def __post_init__(self) -> None:
        # Single source of truth: load balancing flag
        # If either flag is False, disable both (prefer explicit use_load_balancing).
        if not self.use_load_balancing or not self.use_quantile_balancing:
            object.__setattr__(self, 'use_load_balancing', False)
            object.__setattr__(self, 'use_quantile_balancing', False)
        else:
            object.__setattr__(self, 'use_quantile_balancing', self.use_load_balancing)
        if self.hidden_size % self.num_attention_heads != 0:
            raise ValueError("hidden_size must be divisible by num_attention_heads")
        if self.latent_size <= 0 or self.latent_size > self.hidden_size:
            raise ValueError("latent_size must be in (0, hidden_size]")
        if self.num_routed_experts < self.top_k_experts:
            raise ValueError("num_routed_experts must be >= top_k_experts")
        if self.num_recurrent_per_block < 1:
            raise ValueError("num_recurrent_per_block must be >= 1")
