#!/usr/bin/env python3
"""Dependency-light Stage B plumbing smoke; it does not qualify HRM."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from hrm_adaptive_memory.context.packer import EvidencePacker
from hrm_adaptive_memory.memory.chunking import StructuralChunker
from hrm_adaptive_memory.retrieval.hybrid import HybridRetriever
from hrm_adaptive_memory.retrieval.reranker import LexicalOverlapReranker


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="runs/hrm-memory-smoke")
    args = parser.parse_args()
    chunks = StructuralChunker(target_tokens=80, overlap_tokens=10).prose(
        "# HRM\nHRM uses two H cycles and three L cycles.\n\n# Context\nThe native context window is 4096 tokens.",
        source_id="smoke_source", title="smoke",
    )
    candidates = HybridRetriever(chunks).search(
        "How many H and L cycles does HRM use?", reranker=LexicalOverlapReranker(), final_k=2,
    )
    packet = EvidencePacker().pack(
        objective="Answer the HRM recurrence question.", current_state="No answer yet.", candidates=candidates,
        unresolved=("Whether retrieved evidence is sufficient.",),
    )
    output = Path(args.output); output.mkdir(parents=True, exist_ok=True)
    report = {
        "status": "ENGINEERING_SMOKE_ONLY", "chunks": len(chunks),
        "retrieved": [item.chunk.chunk_id for item in candidates],
        "packet_tokens": packet.token_count, "packet": packet.rendered,
        "oracle_context_qualified": False, "controller_training_allowed": False,
    }
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key != "packet"}, indent=2))


if __name__ == "__main__":
    main()
