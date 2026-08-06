#!/usr/bin/env python3
"""Install a pre-push hook running the fast research-state validation.

CI stays authoritative — a local hook can be bypassed with --no-verify — but
this makes the failure mode loud instead of silent.
"""
from __future__ import annotations
import stat, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HOOK = ROOT / ".git" / "hooks" / "pre-push"
BODY = """#!/bin/sh
# Installed by scripts/install_research_git_hooks.py
echo "pre-push: validating research state..."
python3 scripts/validate_research_state.py || {
  echo ""
  echo "pre-push BLOCKED: research state is invalid."
  echo "Fix it, or bypass deliberately with: git push --no-verify"
  exit 1
}
"""

def main() -> int:
    if not (ROOT / ".git").exists():
        print("not a git repository"); return 1
    HOOK.parent.mkdir(parents=True, exist_ok=True)
    HOOK.write_text(BODY)
    HOOK.chmod(HOOK.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP)
    print(f"installed {HOOK}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
