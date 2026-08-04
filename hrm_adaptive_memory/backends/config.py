"""Validated local sidecar endpoints."""

from __future__ import annotations

import ipaddress
from dataclasses import dataclass
from urllib.parse import urlparse


@dataclass(frozen=True)
class SidecarEndpoint:
    backend_id: str
    base_url: str
    timeout_seconds: float = 10.0
    pinned_version: str = ""

    def __post_init__(self) -> None:
        parsed = urlparse(self.base_url)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname:
            raise ValueError("Sidecar URL must be absolute HTTP(S)")
        host = parsed.hostname
        try:
            loopback = ipaddress.ip_address(host).is_loopback
        except ValueError:
            loopback = host == "localhost"
        if not loopback:
            raise ValueError("Memory sidecars must bind to loopback by default")
        if self.timeout_seconds <= 0 or not self.pinned_version:
            raise ValueError("Sidecars require a positive timeout and pinned version")
