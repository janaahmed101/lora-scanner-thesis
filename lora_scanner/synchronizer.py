from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .chirps import apply_cfo, bin_to_signed
from .demodulator import demod_symbols


@dataclass(slots=True)
class SyncResult:
    ok: bool
    preamble_start: int
    header_start: int
    sfd_start: int
    sto: int
    cfo_hz: float
    preamble_bins: list[int]
    sfd_bins: list[int]
    snr_db: float
    rssi_db: float
    reason: str = ""


class Synchronizer:
    def __init__(self, sf: int, bw_hz: float = 125e3, preamble_len: int = 8, timing_sweep: int = 48):
        self.sf = int(sf)
        self.n = 1 << self.sf
        self.bw_hz = float(bw_hz)
        self.sample_rate = float(bw_hz)
        self.preamble_len = int(preamble_len)
        self.timing_sweep = int(timing_sweep)

    def _score_preamble(self, samples: np.ndarray, start: int) -> tuple[float, list[int], list[float]]:
        metrics = demod_symbols(samples, start, self.preamble_len, self.sf, mode="upchirp")
        if len(metrics.bins) < min(4, self.preamble_len):
            return -np.inf, [], []
        signed = np.array([bin_to_signed(b, self.n) for b in metrics.bins], dtype=float)
        stability_penalty = float(np.std(signed))
        score = float(np.sum(metrics.peak_power) / max(1.0 + stability_penalty, 1.0))
        return score, metrics.bins, metrics.snr_db

    def synchronize(self, samples: np.ndarray, candidate_sample_index: int, base_sample_index: int = 0) -> SyncResult:
        approx = int(candidate_sample_index - base_sample_index)
        if approx < 0:
            return SyncResult(False, candidate_sample_index, 0, 0, 0, 0.0, [], [], 0.0, 0.0, "candidate_before_buffer")

        best = (-np.inf, approx, [], [])
        for delta in range(-self.timing_sweep, self.timing_sweep + 1):
            start = approx + delta
            if start < 0 or start + self.preamble_len * self.n > len(samples):
                continue
            score, bins, snrs = self._score_preamble(samples, start)
            # Timing-shifted preamble windows can produce equally stable bins; prefer the detector boundary unless the local sweep is clearly stronger.
            score = score / (1.0 + 0.02 * abs(delta))
            if score > best[0]:
                best = (score, start, bins, snrs)
        if not np.isfinite(best[0]):
            return SyncResult(False, candidate_sample_index, 0, 0, 0, 0.0, [], [], 0.0, 0.0, "timing_refine_failed")

        preamble_start = int(best[1])
        preamble_bins = [int(b) for b in best[2]]
        signed_bins = [bin_to_signed(b, self.n) for b in preamble_bins]
        cfo_hz = float(np.median(signed_bins) * self.bw_hz / self.n)
        corrected = apply_cfo(samples, cfo_hz, self.sample_rate)

        # Re-score after CFO correction. The corrected preamble should move near bin 0.
        score2, bins2, snrs2 = self._score_preamble(corrected, preamble_start)
        if bins2:
            preamble_bins = [int(b) for b in bins2]

        predicted_sfd = preamble_start + int((self.preamble_len + 2) * self.n)
        best_sfd = (-np.inf, predicted_sfd, [], [])
        for delta in range(-self.timing_sweep, self.timing_sweep + 1):
            sfd_start = predicted_sfd + delta
            if sfd_start < 0 or sfd_start + 2 * self.n > len(corrected):
                continue
            metrics = demod_symbols(corrected, sfd_start, 2, self.sf, mode="downchirp")
            if len(metrics.bins) < 2:
                continue
            # Prefer strong downchirp peaks and near-equal bins; not hard-coded to 0,
            # because residual CFO/SFO can leave a few bins of offset.
            signed = np.array([bin_to_signed(b, self.n) for b in metrics.bins], dtype=float)
            score = float(sum(metrics.peak_power) / (1.0 + np.std(signed)))
            score = score / (1.0 + 0.02 * abs(delta))
            if score > best_sfd[0]:
                best_sfd = (score, sfd_start, metrics.bins, metrics.snr_db)
        if not np.isfinite(best_sfd[0]):
            return SyncResult(False, int(base_sample_index + preamble_start), 0, 0, int(preamble_start - approx), cfo_hz, preamble_bins, [], float(np.median(best[3]) if best[3] else 0.0), 0.0, "sfd_not_found")

        sfd_start = int(best_sfd[1])
        sfd_bins = [int(b) for b in best_sfd[2]]
        # Header starts 2.25 symbols after the first SFD downchirp.
        header_start = int(round(sfd_start + 2.25 * self.n))
        rssi_region = corrected[preamble_start : preamble_start + self.preamble_len * self.n]
        rssi_db = float(10.0 * np.log10(max(np.mean(np.abs(rssi_region) ** 2), 1e-15))) if len(rssi_region) else 0.0
        return SyncResult(
            ok=True,
            preamble_start=int(base_sample_index + preamble_start),
            header_start=int(base_sample_index + header_start),
            sfd_start=int(base_sample_index + sfd_start),
            sto=int(preamble_start - approx),
            cfo_hz=cfo_hz,
            preamble_bins=preamble_bins,
            sfd_bins=sfd_bins,
            snr_db=float(np.median(snrs2 if snrs2 else best[3])) if (snrs2 or best[3]) else 0.0,
            rssi_db=rssi_db,
        )
