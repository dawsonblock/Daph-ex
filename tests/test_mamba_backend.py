#!/usr/bin/env python3
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
from daph.ssm import SelectiveSSM, _SCAN_BACKENDS, register_scan_backend, dispatch_selective_scan
from daph.mamba_backend import register_mamba_ssm_backend


def test_register_without_crash():
    ok = register_mamba_ssm_backend()
    # True if installed, False otherwise — both acceptable
    print(f"mamba_ssm available: {ok}")
    assert isinstance(ok, bool)


def test_eager_scan_still_works():
    m = SelectiveSSM(32, 8)
    x = torch.randn(2, 16, 32)
    y, state = m(x)
    assert y.shape == x.shape
    assert state.shape[0] == 2
    print("eager SelectiveSSM OK")


def test_custom_backend_dispatch():
    calls = {"n": 0}

    def spy(xin, b, c, dt, a, d, h):
        calls["n"] += 1
        # minimal identity-ish: return zeros matching layout
        B, L, H = xin.shape
        N = h.shape[-1]
        y = torch.zeros(B, L, H, device=xin.device, dtype=xin.dtype)
        return y, h

    register_scan_backend("spy_test", spy)
    os.environ["DAPH_SCAN_BACKEND"] = "spy_test"
    try:
        m = SelectiveSSM(16, 4)
        # force re-bind scan
        m._scan = dispatch_selective_scan
        m(torch.randn(1, 8, 16))
        assert calls["n"] >= 1
        print("custom backend dispatch OK")
    finally:
        os.environ.pop("DAPH_SCAN_BACKEND", None)


if __name__ == "__main__":
    test_register_without_crash()
    test_eager_scan_still_works()
    test_custom_backend_dispatch()
    print("\nAll mamba backend tests passed.")
