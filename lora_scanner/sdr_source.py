from __future__ import annotations

import numpy as np


def _normalize_gain(gain: str | float) -> str | float:
    if isinstance(gain, str):
        value = gain.strip()
        if not value or value.lower() == "auto":
            return "auto"
        return float(value.replace("_", ""))
    return gain


class RtlSdrSource:
    """Small pyrtlsdr wrapper. Import is delayed so offline mode has no RTL dependency."""

    def __init__(self, center_freq: float, sample_rate: float, gain: str | float = "auto", block_size: int = 262_144):
        try:
            from rtlsdr import RtlSdr
        except Exception as exc:  # pragma: no cover - depends on local driver install
            raise RuntimeError("pyrtlsdr is not available or the RTL-SDR driver is not installed") from exc
        self.sdr = RtlSdr()
        self.sdr.sample_rate = float(sample_rate)
        self.sdr.center_freq = float(center_freq)
        self.sdr.gain = _normalize_gain(gain)
        self.block_size = int(block_size)

    def blocks(self):  # pragma: no cover - hardware loop
        while True:
            yield np.asarray(self.sdr.read_samples(self.block_size), dtype=np.complex64)

    def close(self):  # pragma: no cover
        self.sdr.close()
