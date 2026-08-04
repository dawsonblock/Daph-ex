import hashlib
import json
from pathlib import Path


def test_voc_stage1_evidence_manifest_hashes_every_bundled_artifact():
    root = Path(__file__).resolve().parents[1] / "evidence" / "voc_stage1_smoke_v1"
    manifest = json.loads((root / "manifest.json").read_text())
    assert manifest["gates"]["controller_training_allowed"] is False
    assert manifest["gates"]["controller_status"] == "BLOCKED_BEFORE_VALUE_CONTROLLER"
    for relative, expected in manifest["files"].items():
        path = root / relative
        assert path.is_file(), relative
        assert hashlib.sha256(path.read_bytes()).hexdigest() == expected, relative
