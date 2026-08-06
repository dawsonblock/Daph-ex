"""Batched generation must be indistinguishable from the sequential path.

Batching is a ~5-10x speedup, but only usable if greedy completions are
byte-identical to sequential generation. If left-padding perturbs PrefixLM
masking or attention, batched numbers would not be comparable to the frozen
Gate A/B/C0 results, and the speedup would be bought with a silent
measurement change. This test is the gate on that.

The equivalence check against the real checkpoint is opt-in (it needs the 2.4GB
model): set HRM_EQUIVALENCE_TEST=1. The shape and padding tests always run
against a stub.
"""

from __future__ import annotations

import os

import pytest
import torch

from hrm_adaptive_memory.hrm.model import HRMAdapter, HRMModelSpec, PromptCondition


class _StubConfig:
    model_type = "hrm_text"
    hidden_size = 1536
    num_layers_per_stack = 16
    H_cycles = 2
    L_cycles = 3
    max_position_embeddings = 4096
    prefix_lm = True
    architectures = ("HrmTextForCausalLM",)


class _StubModel:
    """Echoes back input plus deterministic continuation tokens."""

    config = _StubConfig()
    device = None

    def __init__(self):
        self.calls = []

    def generate(self, *, input_ids, attention_mask=None, token_type_ids=None,
                 max_new_tokens=8, do_sample=False, **kwargs):
        self.calls.append({
            "input_ids": input_ids, "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
        })
        batch, length = input_ids.shape
        continuation = torch.arange(1, max_new_tokens + 1).repeat(batch, 1)
        return torch.cat([input_ids, continuation], dim=1)


class _StubTokenizer:
    pad_token_id = 0
    eos_token = "<eos>"
    padding_side = "right"

    def __call__(self, text, return_tensors=None, padding=False, **kwargs):
        texts = [text] if isinstance(text, str) else list(text)
        encoded = [[5 + index for index in range(4 + position)] for position, _ in enumerate(texts)]
        width = max(len(row) for row in encoded)
        ids, mask = [], []
        for row in encoded:
            pad = width - len(row)
            if padding and self.padding_side == "left":
                ids.append([self.pad_token_id] * pad + row)
                mask.append([0] * pad + [1] * len(row))
            else:
                ids.append(row + [self.pad_token_id] * pad)
                mask.append([1] * len(row) + [0] * pad)
        return {"input_ids": torch.tensor(ids), "attention_mask": torch.tensor(mask)}

    def decode(self, ids, skip_special_tokens=False):
        return " ".join(str(int(value)) for value in ids)


def _adapter() -> HRMAdapter:
    return HRMAdapter(_StubModel(), _StubTokenizer(), spec=HRMModelSpec())


def test_batch_uses_left_padding():
    """Right-padding would leave generated tokens detached from the prompt."""

    adapter = _adapter()
    adapter.generate_batch(["a", "bb", "ccc"], condition=PromptCondition.DIRECT, max_new_tokens=4)
    call = adapter.model.calls[-1]
    mask = call["attention_mask"]
    # Left-padded rows have their zeros first.
    for row in mask:
        values = row.tolist()
        if 0 in values:
            assert values.index(1) > values.index(0), "padding must be on the left"


def test_token_type_ids_mark_only_real_prompt_tokens():
    adapter = _adapter()
    adapter.generate_batch(["a", "bb"], condition=PromptCondition.DIRECT, max_new_tokens=2)
    call = adapter.model.calls[-1]
    assert torch.equal(call["token_type_ids"], call["attention_mask"]), (
        "padding must not be marked as PrefixLM context"
    )


def test_prompt_token_count_excludes_padding():
    adapter = _adapter()
    rows = adapter.generate_batch(["a", "bb", "ccc"], condition=PromptCondition.DIRECT,
                                  max_new_tokens=2)
    # The stub encodes progressively longer prompts; counts must differ.
    assert [row["prompt_tokens"] for row in rows] == [4, 5, 6]


def test_padding_side_is_restored():
    adapter = _adapter()
    adapter.tokenizer.padding_side = "right"
    adapter.generate_batch(["a", "bb"], condition=PromptCondition.DIRECT, max_new_tokens=2)
    assert adapter.tokenizer.padding_side == "right"


def test_empty_batch_is_a_no_op():
    assert _adapter().generate_batch([], condition=PromptCondition.DIRECT) == []


@pytest.mark.skipif(
    os.environ.get("HRM_EQUIVALENCE_TEST") != "1",
    reason="set HRM_EQUIVALENCE_TEST=1 to check against the real checkpoint",
)
def test_batched_matches_sequential_on_the_real_checkpoint():
    """The gate: identical greedy completions, or batching is unusable."""

    adapter = HRMAdapter.from_pretrained(spec=HRMModelSpec(), dtype=torch.bfloat16,
                                         device_map="auto")
    prompts = [
        "[OBJECTIVE]\nWhich category applies to Atlas controller?\n[EVIDENCE]\n"
        "[E1]\nThe registry records that Atlas controller is assigned BETA-GREEN.\n"
        "[RESPONSE REQUIREMENT]\nReturn only the answer.",
        "[OBJECTIVE]\nWhat is the tier for Ridge module?\n[EVIDENCE]\n"
        "[E1]\nRegistry entry: Ridge module — tier — accredited.\n"
        "[RESPONSE REQUIREMENT]\nReturn only the answer.",
        "[OBJECTIVE]\nIdentify the grade for LAX-5.\n[EVIDENCE]\n"
        "[E1]\nLAX-5\tgrade\t{\"state\": \"retired\"}\n"
        "[RESPONSE REQUIREMENT]\nReturn only the answer.",
    ]
    sequential = [
        adapter.generate(prompt, condition=PromptCondition.DIRECT, max_new_tokens=64)["text"]
        for prompt in prompts
    ]
    batched = [
        row["text"] for row in
        adapter.generate_batch(prompts, condition=PromptCondition.DIRECT, max_new_tokens=64)
    ]
    mismatches = [
        (index, left, right) for index, (left, right) in enumerate(zip(sequential, batched))
        if left.rstrip("\x00") != right.rstrip("\x00")
    ]
    assert not mismatches, (
        "batched generation diverges from sequential; it must not be used for "
        f"comparable measurements: {mismatches}"
    )
