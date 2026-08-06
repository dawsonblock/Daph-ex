"""Pinned native Transformers adapter for sapientinc/HRM-Text-1B."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Mapping, Sequence


class PromptCondition(str, Enum):
    DIRECT = "direct"
    COT = "cot"
    SYNTH_COT = "synth,cot"
    NOISY = "noisy"


_CONDITION_TOKENS = {
    PromptCondition.DIRECT: "<|object_ref_start|>",
    PromptCondition.COT: "<|object_ref_end|>",
    PromptCondition.SYNTH_COT: "<|quad_end|><|object_ref_end|>",
    PromptCondition.NOISY: "<|quad_start|>",
}


@dataclass(frozen=True)
class HRMModelSpec:
    model_id: str = "sapientinc/HRM-Text-1B"
    revision: str = "9f082d68b8cd0ebc56e33f1c88c45609174c272c"
    architecture: str = "HrmTextForCausalLM"
    hidden_size: int = 1536
    layers_per_stack: int = 16
    high_cycles: int = 2
    low_cycles: int = 3
    max_sequence_length: int = 4096
    prefix_lm: bool = True

    def validate_config(self, config: Any) -> None:
        expected = {
            "model_type": "hrm_text",
            "hidden_size": self.hidden_size,
            "num_layers_per_stack": self.layers_per_stack,
            "H_cycles": self.high_cycles,
            "L_cycles": self.low_cycles,
            "max_position_embeddings": self.max_sequence_length,
            "prefix_lm": self.prefix_lm,
        }
        mismatches = {
            key: (value, getattr(config, key, None))
            for key, value in expected.items()
            if getattr(config, key, None) != value
        }
        architectures = tuple(getattr(config, "architectures", ()) or ())
        if self.architecture not in architectures:
            mismatches["architectures"] = (self.architecture, architectures)
        if mismatches:
            raise ValueError(f"Pinned HRM config mismatch: {mismatches}")


class HRMAdapter:
    """Load and invoke the native HRM checkpoint with correct PrefixLM masking.

    Model and tokenizer injection keeps unit tests light and permits controlled
    backends.  The default loader is intentionally lazy, so the base package
    does not require Transformers 5.9 unless HRM execution is requested.
    """

    def __init__(self, model: Any, tokenizer: Any, *, spec: HRMModelSpec | None = None):
        self.model = model
        self.tokenizer = tokenizer
        self.spec = spec or HRMModelSpec()
        self.spec.validate_config(model.config)

    @classmethod
    def from_pretrained(
        cls, *, spec: HRMModelSpec | None = None, dtype: Any = None,
        device_map: Any = None, attn_implementation: str = "sdpa",
    ) -> "HRMAdapter":
        try:
            import transformers
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:  # pragma: no cover - depends on optional install
            raise RuntimeError("Install the 'hrm' extra to load HRM-Text-1B") from exc
        version = tuple(int(part) for part in transformers.__version__.split(".")[:2])
        if version < (5, 9):
            raise RuntimeError("Native HrmTextForCausalLM requires transformers>=5.9.0")
        pinned = spec or HRMModelSpec()
        tokenizer = AutoTokenizer.from_pretrained(pinned.model_id, revision=pinned.revision)
        kwargs: dict[str, Any] = {
            "revision": pinned.revision,
            "attn_implementation": attn_implementation,
        }
        if dtype is not None:
            kwargs["dtype"] = dtype
        if device_map is not None:
            kwargs["device_map"] = device_map
        model = AutoModelForCausalLM.from_pretrained(pinned.model_id, **kwargs).eval()
        return cls(model, tokenizer, spec=pinned)

    @staticmethod
    def render_prompt(task: str, condition: PromptCondition) -> str:
        token = _CONDITION_TOKENS[PromptCondition(condition)]
        return f"<|im_start|>{token}{task}<|im_end|>"

    def encode(self, task: str, condition: PromptCondition) -> dict[str, Any]:
        encoded = dict(self.tokenizer(self.render_prompt(task, condition), return_tensors="pt"))
        input_ids = encoded["input_ids"]
        try:
            import torch
            encoded["token_type_ids"] = torch.ones_like(input_ids)
        except ImportError:  # pragma: no cover
            encoded["token_type_ids"] = [[1 for _ in row] for row in input_ids]
        return encoded

    def generate_batch(
        self, tasks: Sequence[str], *, condition: PromptCondition,
        max_new_tokens: int = 256, do_sample: bool = False,
        generation_kwargs: Mapping[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Generate for several prompts in one forward pass.

        Left-padding is required: with right-padding the newly generated tokens
        would not be adjacent to each prompt's real final token, so decoding
        would differ from the sequential path. Equivalence with `generate` is
        not assumed — `tests/unit/test_batched_generation.py` asserts that the
        batched and sequential paths produce identical greedy completions, and
        batching must not be used for comparable measurements unless it passes.
        """

        import torch

        if not tasks:
            return []
        prompts = [self.render_prompt(task, condition) for task in tasks]
        previous_side = getattr(self.tokenizer, "padding_side", "right")
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        try:
            encoded = dict(self.tokenizer(prompts, return_tensors="pt", padding=True))
        finally:
            self.tokenizer.padding_side = previous_side
        attention = encoded["attention_mask"]
        # PrefixLM: every real prompt token is prefix context; padding is not.
        encoded["token_type_ids"] = attention.clone()

        device = getattr(self.model, "device", None)
        if device is not None:
            encoded = {
                key: value.to(device) if hasattr(value, "to") else value
                for key, value in encoded.items()
            }
        sequences = self.model.generate(
            **encoded, max_new_tokens=max_new_tokens, do_sample=do_sample,
            **dict(generation_kwargs or {}),
        )
        padded_length = int(encoded["input_ids"].shape[-1])
        results = []
        for index, prompt in enumerate(prompts):
            real_tokens = int(attention[index].sum())
            completion_ids = sequences[index][padded_length:]
            # Trim at the first pad so a short completion is not padded out.
            text = self.tokenizer.decode(completion_ids, skip_special_tokens=False)
            results.append({
                "prompt": prompt,
                "condition": PromptCondition(condition).value,
                "text": text,
                "prompt_tokens": real_tokens,
                "completion_tokens": len(completion_ids),
                "sequences": sequences[index:index + 1],
            })
        return results

    def generate(
        self, task: str, *, condition: PromptCondition, max_new_tokens: int = 256,
        do_sample: bool = False, generation_kwargs: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        inputs = self.encode(task, condition)
        device = getattr(self.model, "device", None)
        if device is not None:
            inputs = {key: value.to(device) if hasattr(value, "to") else value for key, value in inputs.items()}
        kwargs = dict(generation_kwargs or {})
        sequences = self.model.generate(
            **inputs, max_new_tokens=max_new_tokens, do_sample=do_sample, **kwargs,
        )
        prompt_length = int(inputs["input_ids"].shape[-1])
        sequence = sequences[0] if hasattr(sequences, "__getitem__") else sequences
        completion_ids = sequence[prompt_length:]
        return {
            "prompt": self.render_prompt(task, condition),
            "condition": PromptCondition(condition).value,
            "text": self.tokenizer.decode(completion_ids, skip_special_tokens=False),
            "prompt_tokens": prompt_length,
            "completion_tokens": len(completion_ids),
            "sequences": sequences,
        }
