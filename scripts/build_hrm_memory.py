#!/usr/bin/env python3
"""Chunk source files into immutable HRM source-memory JSONL."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from hrm_adaptive_memory.memory.chunking import StructuralChunker


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs="+")
    parser.add_argument("--output", required=True)
    parser.add_argument("--target-tokens", type=int, default=500)
    parser.add_argument("--overlap-tokens", type=int, default=75)
    args = parser.parse_args()
    chunker = StructuralChunker(target_tokens=args.target_tokens, overlap_tokens=args.overlap_tokens)
    chunks = []
    for value in args.inputs:
        path = Path(value)
        text = path.read_text()
        if path.suffix == ".py":
            chunks.extend(chunker.code(text, source_id=str(path), title=path.name))
        else:
            chunks.extend(chunker.prose(text, source_id=str(path), title=path.name))
    output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("".join(json.dumps({**chunk.__dict__, "tags": list(chunk.tags)}, sort_keys=True) + "\n" for chunk in chunks))
    print(json.dumps({"sources": len(args.inputs), "chunks": len(chunks), "output": str(output)}, indent=2))


if __name__ == "__main__":
    main()
