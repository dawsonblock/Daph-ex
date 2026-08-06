"""Runtime canonicalization: parse identity records that retrieval already found.

Measured motivation: identity-record recall on alias/description tasks is
0.96-1.00 — the identity record is almost always IN the pool — while complete
proof recovery is only 0.21-0.26. The system sees the bridge and fails to use
it. This turns a retrieved identity statement into a usable canonical name.

Strictly runtime-visible: operates on candidate text only, never on
`_oracle_metadata`. A test asserts that.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# Identity statements as the corpora actually render them. Each pattern yields
# (other, canonical) or (canonical, other) depending on the phrasing direction.
_FORWARD = (
    re.compile(r"^(?P<other>.+?) is the (?:operational|working) (?:alias|designation) (?:for|of) (?P<canon>.+?)\.$", re.I),
    re.compile(r"^(?:Alias register|Cross-reference):\s*(?P<other>.+?) (?:refers to|denotes) (?P<canon>.+?)\.$", re.I),
    re.compile(r"^(?P<other>.+?) (?:is the short code for|abbreviates|is the compact form of) (?P<canon>.+?)\.$", re.I),
    re.compile(r"^(?:Abbreviation table|Short-code table):\s*(?P<other>.+?) (?:=|stands for) (?P<canon>.+?)\.$", re.I),
    re.compile(r"^(?P<other>.+?) denotes (?P<canon>.+?) in condensed records\.$", re.I),
    re.compile(r"^Recorded alternate label (?P<other>.+?) for (?P<canon>.+?)\.$", re.I),
)
_REVERSE = (
    re.compile(r"^(?:The intake record|The transfer ledger) identifies (?P<canon>.+?) as (?P<other>.+?)\.$", re.I),
    re.compile(r"^(?:The intake record|The transfer ledger) names (?P<canon>.+?) as (?P<other>.+?)\.$", re.I),
    re.compile(r"^(?:Commissioning|Retrofit) (?:notes|records) describe (?P<canon>.+?) as (?P<other>.+?)\.$", re.I),
    re.compile(r"^It was (?P<canon>.+?) that served as (?P<other>.+?)\.$", re.I),
    re.compile(r"^(?P<other>.+?) is (?P<canon>.+?), per the (?:acceptance log|survey report)\.$", re.I),
    re.compile(r"^In shorthand, (?P<canon>.+?) is written (?P<other>.+?)\.$", re.I),
    re.compile(r"^Written briefly, (?P<canon>.+?) appears as (?P<other>.+?)\.$", re.I),
    re.compile(r"^(?P<other>.+?) and (?P<canon>.+?) (?:denote the same unit|are the same unit under two names)\.?$", re.I),
)


@dataclass(frozen=True)
class IdentityLink:
    surface: str
    canonical: str
    record_id: str


def _norm(text: str) -> str:
    return " ".join(re.findall(r"\w+", text.lower()))


def extract_identity_links(records) -> list[IdentityLink]:
    """Parse identity statements out of runtime-visible candidate text."""

    links: list[IdentityLink] = []
    for record in records:
        content = str(getattr(record, "content", "")).strip()
        rid = str(getattr(record, "evidence_id", ""))
        for pattern in _FORWARD + _REVERSE:
            match = pattern.match(content)
            if match:
                other, canon = match.group("other").strip(), match.group("canon").strip()
                if other and canon and _norm(other) != _norm(canon):
                    links.append(IdentityLink(other, canon, rid))
                break
    return links


def resolve_canonical(subject_surface: str, records) -> IdentityLink | None:
    """Find the canonical name for a question's subject, if the pool states it."""

    target = _norm(subject_surface)
    # Question phrasing may prefix "the "; identity records may not.
    stripped = _norm(re.sub(r"^the\s+", "", subject_surface, flags=re.I))
    for link in extract_identity_links(records):
        surface = _norm(link.surface)
        if surface in (target, stripped) or target in surface or stripped in surface:
            return link
    return None
