from __future__ import annotations

from dataclasses import dataclass
from collections import deque
import time
import numpy as np

from .demodulator import demod_symbol


@dataclass(slots=True)
class PreambleCandidate:
    sample_index: int
    bins: list[int]
    snr_db: float
    isolation: float
    rssi_db: float
    detected_at: float


def _stable_bins(bins: list[int], n_bins: int, drift: int) -> bool:
    if len(bins) < 2:
        return False
    ref = int(round(np.median(bins))) % n_bins
    for b in bins:
        d = abs(((int(b) - ref + n_bins // 2) % n_bins) - n_bins // 2)
        if d > drift:
            return False
    return True


class PreambleDetector:
    """FFT-bin repetition detector for LoRa preambles.

    The detector intentionally keeps the criterion close to your thesis plan:
    dechirp each symbol with a downchirp, FFT, then require repeated stable bins.
    """

    def __init__(self, sf: int, repeats: int = 4, drift_bins: int = 1, min_snr_db: float = 5.0, min_isolation: float = 1.25):
        self.sf = int(sf)
        self.n = 1 << self.sf
        self.repeats = int(repeats)
        self.drift_bins = int(drift_bins)
        self.min_snr_db = float(min_snr_db)
        self.min_isolation = float(min_isolation)

    def scan(self, samples: np.ndarray, base_sample_index: int = 0, hop: int | None = None) -> list[PreambleCandidate]:
        hop = int(hop or self.n)
        if len(samples) < self.n * self.repeats:
            return []
        history: deque[tuple[int, int, float, float, float]] = deque(maxlen=self.repeats)
        out: list[PreambleCandidate] = []
        last_emit = -10**12
        for start in range(0, len(samples) - self.n + 1, hop):
            sym = samples[start : start + self.n]
            peak, snr, isolation, _ = demod_symbol(sym, self.sf, mode="upchirp")
            rssi = float(10.0 * np.log10(max(np.mean(np.abs(sym) ** 2), 1e-15)))
            history.append((start, peak, snr, isolation, rssi))
            if len(history) < self.repeats:
                continue
            bins = [h[1] for h in history]
            snrs = [h[2] for h in history]
            isos = [h[3] for h in history]
            if not _stable_bins(bins, self.n, self.drift_bins):
                continue
            if float(np.min(snrs)) < self.min_snr_db or float(np.min(isos)) < self.min_isolation:
                continue
            cand_start = history[0][0]
            abs_start = int(base_sample_index + cand_start)
            # Avoid emitting the same preamble several times as the window slides.
            if abs_start - last_emit < self.n * self.repeats:
                continue
            last_emit = abs_start
            out.append(
                PreambleCandidate(
                    sample_index=abs_start,
                    bins=[int(b) for b in bins],
                    snr_db=float(np.median(snrs)),
                    isolation=float(np.median(isos)),
                    rssi_db=float(np.mean([h[4] for h in history])),
                    detected_at=time.time(),
                )
            )
        return out
