from __future__ import annotations

import numpy as np


def upchirp(sf: int, bw: float = 125e3, sample_rate: float | None = None) -> np.ndarray:
    """Return one normalized baseband LoRa upchirp with M=2^SF samples by default.

    The phase convention matches the useful logic in the attached LoRa_Rx_SDR ZIP:
    exp(j*pi*k^2/M) with the -BW/2 linear term. At sample_rate=bw, this reduces
    to one sample per chip and one symbol has M samples.
    """
    n = 1 << int(sf)
    if sample_rate is None:
        sample_rate = bw
    os = float(sample_rate) / float(bw)
    m = int(round(n * os))
    k = np.arange(m, dtype=np.float64) / os
    t_chip = 1.0 / float(bw)
    phase = 2.0 * np.pi * (k * k / (2.0 * n) - 0.5 * k)
    return (np.exp(1j * phase) / np.sqrt(n)).astype(np.complex64)


def downchirp(sf: int, bw: float = 125e3, sample_rate: float | None = None) -> np.ndarray:
    return np.conjugate(upchirp(sf, bw, sample_rate)).astype(np.complex64)


def modulate_symbol(symbol: int, sf: int, bw: float = 125e3) -> np.ndarray:
    """Create one LoRa upchirp symbol by cyclically shifting the base upchirp."""
    n = 1 << int(sf)
    return np.roll(upchirp(sf, bw), -(int(symbol) % n))


def bin_to_signed(bin_index: int, n_bins: int) -> int:
    b = int(bin_index) % int(n_bins)
    return b - int(n_bins) if b >= int(n_bins) // 2 else b


def apply_cfo(samples: np.ndarray, cfo_hz: float, sample_rate: float, start_index: int = 0) -> np.ndarray:
    if not cfo_hz:
        return samples
    n = np.arange(len(samples), dtype=np.float64) + float(start_index)
    rot = np.exp(-1j * 2.0 * np.pi * float(cfo_hz) * n / float(sample_rate))
    return (samples * rot).astype(np.complex64)
