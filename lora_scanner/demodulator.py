from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np

from .chirps import downchirp, upchirp


@dataclass(slots=True)
class SymbolMetrics:
    bins: list[int]
    snr_db: list[float]
    isolation: list[float]
    peak_power: list[float]
    top_bins: list[list[int]] = field(default_factory=list)
    top_power: list[list[float]] = field(default_factory=list)


def fft_peak_metrics(spec: np.ndarray, guard_bins: int = 2) -> tuple[int, float, float, float]:
    power = np.abs(spec) ** 2
    n = len(power)
    peak = int(np.argmax(power))
    peak_power = float(power[peak])
    mask = np.ones(n, dtype=bool)
    for d in range(-guard_bins, guard_bins + 1):
        mask[(peak + d) % n] = False
    noise = float(np.median(power[mask])) if np.any(mask) else 1e-12
    noise = max(noise, 1e-12)
    snr_db = 10.0 * np.log10(max(peak_power - noise, 1e-12) / noise)
    sorted_power = np.partition(power, -2)
    second = float(sorted_power[-2]) if n > 1 else noise
    isolation = peak_power / max(second, 1e-12)
    return peak, float(snr_db), float(isolation), peak_power


def _topk_from_spec(spec: np.ndarray, k: int) -> tuple[list[int], list[float]]:
    power = np.abs(spec) ** 2
    n = len(power)
    kk = max(1, min(int(k), n))
    idx = np.argpartition(power, -kk)[-kk:]
    idx = idx[np.argsort(power[idx])[::-1]]
    return [int(i) for i in idx], [float(power[i]) for i in idx]


def demod_symbol(symbol_samples: np.ndarray, sf: int, mode: str = "upchirp") -> tuple[int, float, float, float]:
    """Demodulate one symbol into an FFT bin.

    mode='upchirp' is used for normal LoRa upchirp data/header symbols and
    preamble symbols, so it multiplies by the downchirp. mode='downchirp' is
    used for SFD downchirps, so it multiplies by the upchirp.
    """
    n = 1 << int(sf)
    if len(symbol_samples) != n:
        raise ValueError(f"expected {n} samples, got {len(symbol_samples)}")
    ref = downchirp(sf) if mode == "upchirp" else upchirp(sf)
    spec = np.fft.fft(symbol_samples * ref)
    return fft_peak_metrics(spec)


def demod_symbols(samples: np.ndarray, start: int, count: int, sf: int, mode: str = "upchirp", top_k: int = 1) -> SymbolMetrics:
    n = 1 << int(sf)
    bins: list[int] = []
    snrs: list[float] = []
    isolations: list[float] = []
    peaks: list[float] = []
    top_bins: list[list[int]] = []
    top_power: list[list[float]] = []
    ref = downchirp(sf) if mode == "upchirp" else upchirp(sf)
    kk = max(1, int(top_k))
    for i in range(int(count)):
        a = int(start) + i * n
        b = a + n
        if a < 0 or b > len(samples):
            break
        spec = np.fft.fft(samples[a:b] * ref)
        peak, snr, iso, pwr = fft_peak_metrics(spec)
        tb, tp = _topk_from_spec(spec, kk)
        if tb and tb[0] != peak:
            if peak in tb:
                j = tb.index(peak)
                tb.insert(0, tb.pop(j))
                tp.insert(0, tp.pop(j))
            else:
                tb.insert(0, peak)
                tp.insert(0, pwr)
                tb = tb[:kk]
                tp = tp[:kk]
        bins.append(peak)
        snrs.append(snr)
        isolations.append(iso)
        peaks.append(pwr)
        top_bins.append(tb)
        top_power.append(tp)
    return SymbolMetrics(bins, snrs, isolations, peaks, top_bins, top_power)
