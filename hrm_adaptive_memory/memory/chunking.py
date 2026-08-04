"""Structure-aware chunking with provenance-preserving metadata."""

from __future__ import annotations

import ast
import hashlib
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Mapping


def approximate_tokens(text: str) -> int:
    return max(1, (len(text) + 3) // 4)


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    source_id: str
    source_type: str
    title: str
    section: str
    content: str
    token_count: int
    parent_id: str | None = None
    timestamp: str | None = None
    tags: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def text(self) -> str:
        return self.content


class StructuralChunker:
    def __init__(
        self, *, target_tokens: int = 500, overlap_tokens: int = 75,
        token_counter: Callable[[str], int] = approximate_tokens,
    ):
        if target_tokens < 1 or not 0 <= overlap_tokens < target_tokens:
            raise ValueError("Require target_tokens>0 and 0<=overlap<target")
        self.target_tokens = target_tokens
        self.overlap_tokens = overlap_tokens
        self.token_counter = token_counter

    def _make(self, source_id: str, source_type: str, title: str, section: str,
              content: str, index: int, **metadata: Any) -> Chunk:
        digest = hashlib.sha256(f"{source_id}\0{section}\0{index}\0{content}".encode()).hexdigest()[:16]
        return Chunk(
            chunk_id=f"chunk_{digest}", source_id=source_id, source_type=source_type,
            title=title, section=section, content=content.strip(),
            token_count=self.token_counter(content), metadata=metadata,
        )

    def prose(self, text: str, *, source_id: str, title: str = "") -> list[Chunk]:
        headings = re.split(r"(?m)^(#{1,6}\s+.+)$", text)
        sections: list[tuple[str, str]] = []
        current = title or source_id
        for part in headings:
            if re.match(r"^#{1,6}\s+", part):
                current = part.lstrip("# ").strip()
            elif part.strip():
                sections.append((current, part.strip()))
        if not sections and text.strip():
            sections = [(current, text.strip())]
        chunks: list[Chunk] = []
        for section, body in sections:
            paragraphs = [value.strip() for value in re.split(r"\n\s*\n", body) if value.strip()]
            chunks.extend(self._group_units(paragraphs, source_id, "prose", title, section, len(chunks)))
        return chunks

    def code(self, text: str, *, source_id: str, title: str = "") -> list[Chunk]:
        try:
            tree = ast.parse(text)
        except SyntaxError:
            return self._group_units(text.splitlines(), source_id, "code", title, "unparsed", 0)
        lines = text.splitlines()
        nodes = [node for node in tree.body if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))]
        chunks: list[Chunk] = []
        covered: set[int] = set()
        for index, node in enumerate(nodes):
            end = getattr(node, "end_lineno", node.lineno)
            covered.update(range(node.lineno, end + 1))
            content = "\n".join(lines[node.lineno - 1:end])
            chunks.append(self._make(source_id, "code", title, node.name, content, index,
                                     start_line=node.lineno, end_line=end))
        module_lines = [line for index, line in enumerate(lines, 1) if index not in covered]
        if any(line.strip() for line in module_lines):
            chunks[:0] = self._group_units(module_lines, source_id, "code", title, "module", len(chunks))
        return chunks

    def experiment(self, sections: Mapping[str, Any], *, source_id: str, title: str = "") -> list[Chunk]:
        chunks = []
        for index, (section, value) in enumerate(sections.items()):
            content = value if isinstance(value, str) else repr(value)
            chunks.append(self._make(source_id, "experiment", title, section, content, index))
        return chunks

    def conversation(self, episodes: Iterable[Mapping[str, Any]], *, source_id: str) -> list[Chunk]:
        chunks = []
        for index, episode in enumerate(episodes):
            topic = str(episode.get("topic", f"episode_{index}"))
            content = str(episode.get("content") or episode.get("decision") or "")
            if content.strip():
                chunks.append(self._make(source_id, "conversation", topic, topic, content, index,
                                         outcome=episode.get("outcome")))
        return chunks

    def _group_units(self, units: Iterable[str], source_id: str, source_type: str,
                     title: str, section: str, start_index: int) -> list[Chunk]:
        chunks: list[Chunk] = []
        current: list[str] = []
        current_tokens = 0
        for unit in units:
            tokens = self.token_counter(unit)
            if current and current_tokens + tokens > self.target_tokens:
                content = "\n\n".join(current)
                chunks.append(self._make(source_id, source_type, title, section, content, start_index + len(chunks)))
                overlap: list[str] = []
                overlap_count = 0
                for previous in reversed(current):
                    previous_tokens = self.token_counter(previous)
                    if overlap and overlap_count + previous_tokens > self.overlap_tokens:
                        break
                    overlap.insert(0, previous); overlap_count += previous_tokens
                current, current_tokens = overlap, overlap_count
            current.append(unit); current_tokens += tokens
        if current:
            chunks.append(self._make(source_id, source_type, title, section,
                                     "\n\n".join(current), start_index + len(chunks)))
        return chunks
