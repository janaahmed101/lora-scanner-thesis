"""Robust LoRa synchronizer (oversampled, SFD-locked, joint CFO/STO).

This replaces the original :class:`Synchronizer`, which had two structural
faults that prevented real over-the-air payload decoding:

1. It assumed a fixed 8-symbol preamble and *predicted* the SFD position from
   it.  The LilyGO transmitter uses a 16-symbol preamble, so the prediction
   landed inside the preamble and the lock was wrong.
2. It estimated CFO directly from the preamble bin, attributing the *timing*
   offset (sample-timing offset, STO) to carrier frequency offset.  On a real
   capture the preamble dechirp bin is dominated by STO, so the CFO estimate
   was meaningless and every downstream symbol was misaligned.

The corrected design works on an oversampled channel stream (``os`` samples per
chip) and:

* locks to the **SFD downchirps** directly, independent of preamble length,
* separates **CFO and STO** using the standard up/down-chirp relation
  ``up_bin = CFO + STO`` , ``down_bin = CFO - STO``
  so ``CFO = (up+down)/2`` and ``STO = (up-down)/2`` (verified by calibration),
* refines fractional CFO by parabolic interpolation of the preamble peak,
* extracts symbols with a small per-symbol timing nudge so residual sampling
  frequency offset (SFO / clock drift) does not corrupt later payload symbols.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np

from .chirps import upchirp


@dataclass(slots=True)
class Sync2Result:
    ok: bool
    reason: str = ""
    sfd_start_os: int = 0          # SFD start, in oversampled samples
    header_start_os: int = 0
    payload_start_os: int = 0
    cfo_hz: float = 0.0
    cfo_bins: float = 0.0
    sto_bins: float = 0.0
    snr_db: float = 0.0
    rssi_db: float = 0.0
    preamble_bins: list[int] = field(default_factory=list)
    sfd_bins: list[int] = field(default_factory=list)
    # corrected, decimated-to-1-sample/chip stream and the symbol-grid offsets
    chan1: np.ndarray | None = None
    header_start1: int = 0
    payload_start1: int = 0
    phase: int = 0


class Synchronizer2:
    def __init__(self, sf: int, bw_hz: float = 125e3, oversample: int = 4,
                 max_preamble: int = 24, track_radius_os: int | None = None):
        self.sf = int(sf)
        self.n = 1 << self.sf
        self.bw = float(bw_hz)
        self.os = max(1, int(oversample))
        self.max_preamble = int(max_preamble)
        self.symbol_os = self.n * self.os
        # how far each symbol window may be nudged to track SFO (in os-samples)
        self.track = int(track_radius_os) if track_radius_os is not None else max(1, self.os // 2)
        # Oversampled dechirp references (length N*os) so demodulation keeps the
        # full processing gain instead of decimating away (os-1)/os of the energy.
        self._ref_down_os = np.conjugate(upchirp(self.sf, self.bw, self.bw * self.os)).astype(np.complex64)
        self._ref_up_os = upchirp(self.sf, self.bw, self.bw * self.os).astype(np.complex64)

    # ---- low level: full-rate dechirp of one symbol, folded to N bins ----
    def _power(self, x: np.ndarray, a_os: int, mode: str) -> np.ndarray | None:
        """Dechirp the full N*os-sample window and fold to a length-N power
        spectrum. Folding sums the os spectral replicas (bins k, k+N, ...), which
        recovers the full processing gain and is far less sensitive to sub-chip
        timing error than decimating to 1 sample/chip before the FFT."""
        seg = x[a_os:a_os + self.symbol_os]
        if len(seg) < self.symbol_os:
            return None
        ref = self._ref_down_os if mode == "up" else self._ref_up_os
        spec = np.fft.fft(seg * ref)
        p = np.abs(spec) ** 2
        return p.reshape(self.os, self.n).sum(axis=0)

    @staticmethod
    def _peak(power: np.ndarray) -> tuple[int, float, float]:
        k = int(np.argmax(power))
        pk = float(power[k])
        nz = np.partition(power, -2)[-2] if len(power) > 1 else 1e-12
        iso = pk / max(float(nz), 1e-12)
        return k, pk, iso

    def _frac_peak(self, power: np.ndarray) -> float:
        """Parabolic interpolation of the folded peak -> fractional bin (signed)."""
        k = int(np.argmax(power))
        a, b, c = power[(k - 1) % self.n], power[k], power[(k + 1) % self.n]
        denom = (a - 2 * b + c)
        delta = 0.5 * (a - c) / denom if denom != 0 else 0.0
        kf = k + float(np.clip(delta, -0.5, 0.5))
        return kf if kf < self.n / 2 else kf - self.n

    # ---- main entry ----
    def synchronize(self, x: np.ndarray, candidate_os: int) -> Sync2Result:
        n, os = self.n, self.os
        sym = self.symbol_os
        cand = max(0, int(candidate_os))

        # 1) confirm we are inside a preamble: scan a few symbols, need a stable
        #    upchirp bin (high "up" peak isolation).
        up_ok = 0
        scan0 = cand
        for s in range(8):
            spec = self._power(x, scan0 + s * sym, "up")
            if spec is None:
                break
            _, _, iso = self._peak(spec)
            if iso > 1.2:
                up_ok += 1
        if up_ok < 3:
            return Sync2Result(False, reason="no_preamble_at_candidate")

        # 2) coarse SFD search: step symbol by symbol, find the first place where
        #    the DOWNchirp dechirp dominates the UPchirp dechirp for two symbols.
        #    Power dominance (dn_pk >> up_pk) is the reliable discriminator; the
        #    folded full-rate spectrum has a modest absolute peak-isolation, so we
        #    do NOT gate on isolation here.
        sfd_coarse = -1
        max_steps = self.max_preamble + 6
        for s in range(max_steps):
            a = cand + s * sym
            sp_up = self._power(x, a, "up")
            sp_dn = self._power(x, a, "down")
            sp_dn2 = self._power(x, a + sym, "down")
            if sp_up is None or sp_dn is None or sp_dn2 is None:
                break
            _, up_pk, _ = self._peak(sp_up)
            _, up_pk2, _ = self._peak(self._power(x, a + sym, "up"))
            _, dn_pk, _ = self._peak(sp_dn)
            _, dn_pk2, _ = self._peak(sp_dn2)
            if dn_pk > 3.0 * up_pk and dn_pk2 > 3.0 * up_pk2:
                sfd_coarse = a
                break
        if sfd_coarse < 0:
            return Sync2Result(False, reason="sfd_not_found")

        # 3) fine SFD timing: slide +/- half a symbol at os-sample resolution,
        #    maximise the combined downchirp peak power of the two SFD symbols.
        best = (-np.inf, sfd_coarse)
        for d in range(-sym // 2, sym // 2 + 1):
            a = sfd_coarse + d
            if a < 0:
                continue
            sp0 = self._power(x, a, "down")
            sp1 = self._power(x, a + sym, "down")
            if sp0 is None or sp1 is None:
                continue
            _, pk0, iso0 = self._peak(sp0)
            _, pk1, iso1 = self._peak(sp1)
            score = (pk0 + pk1) * min(iso0, iso1)
            if score > best[0]:
                best = (score, a)
        sfd_start = int(best[1])

        # 4) joint CFO / STO from preamble upchirps (just before sync word) and
        #    the SFD downchirps.  sync word occupies 2 symbols before the SFD.
        up_fracs = []
        for k in (3, 4, 5, 6):
            a = sfd_start - (2 + k) * sym
            if a < 0:
                continue
            sp = self._power(x, a, "up")
            if sp is not None:
                up_fracs.append(self._frac_peak(sp))
        dn_fracs = []
        for j in (0, 1):
            sp = self._power(x, sfd_start + j * sym, "down")
            if sp is not None:
                dn_fracs.append(self._frac_peak(sp))
        if not up_fracs or not dn_fracs:
            return Sync2Result(False, reason="cfo_estimation_failed")
        up_bin = float(np.median(up_fracs))
        dn_bin = float(np.median(dn_fracs))
        cfo_bins = 0.5 * (up_bin + dn_bin)
        sto_bins = 0.5 * (up_bin - dn_bin)
        cfo_hz = cfo_bins * self.bw / n

        # 5) apply CFO to the oversampled stream
        idx = np.arange(len(x), dtype=np.float64)
        corrected = (x * np.exp(-1j * 2.0 * np.pi * cfo_hz * idx / (self.bw * os))).astype(np.complex64)

        # 6) build the 1-sample/chip symbol grid.  Header starts 2.25 symbols
        #    after the SFD; payload 8 symbols after the header.  We also shift by
        #    the residual STO so the decimation lands on the true symbol boundary.
        sto_os = int(round(sto_bins * os))
        header_start_os = sfd_start + int(round(2.25 * sym)) - sto_os
        payload_start_os = header_start_os + 8 * sym
        phase = header_start_os % os

        # SNR / RSSI from the preamble region
        pre = corrected[max(0, sfd_start - 6 * sym):sfd_start]
        rssi_db = float(10.0 * np.log10(max(np.mean(np.abs(pre) ** 2), 1e-15))) if len(pre) else 0.0
        sp = self._power(corrected, header_start_os, "up")
        snr_db = 0.0
        if sp is not None:
            p = sp
            k = int(np.argmax(p))
            noise = np.median(np.delete(p, [(k - 1) % n, k, (k + 1) % n]))
            snr_db = float(10.0 * np.log10(max(p[k] - noise, 1e-12) / max(noise, 1e-12)))

        # 7) produce a decimated-to-1/chip stream aligned to the symbol grid.
        chan1 = corrected[phase::os]
        header_start1 = header_start_os // os
        payload_start1 = payload_start_os // os

        pre_bins = []
        for k in (3, 4, 5):
            sp = self._power(corrected, sfd_start - (2 + k) * sym, "up")
            if sp is not None:
                pre_bins.append(self._peak(sp)[0])
        sfd_bins = []
        for j in (0, 1):
            sp = self._power(corrected, sfd_start + j * sym, "down")
            if sp is not None:
                sfd_bins.append(self._peak(sp)[0])

        return Sync2Result(
            ok=True, sfd_start_os=sfd_start, header_start_os=header_start_os,
            payload_start_os=payload_start_os, cfo_hz=cfo_hz, cfo_bins=cfo_bins,
            sto_bins=sto_bins, snr_db=snr_db, rssi_db=rssi_db,
            preamble_bins=[int(b) for b in pre_bins], sfd_bins=[int(b) for b in sfd_bins],
            chan1=chan1, header_start1=header_start1, payload_start1=payload_start1, phase=phase,
        )

    # ---- symbol extraction with per-symbol SFO tracking ----
    def extract_symbols_linear(self, corrected_os: np.ndarray, start_os: int, count: int,
                               offset_os: float = 0.0, drift_os_per_sym: float = 0.0) -> list[int]:
        """Demodulate ``count`` symbols on a *fixed* linear grid (no greedy carry).

        Symbol k starts at ``start_os + offset_os + k*(symbol_os + drift_os_per_sym)``.
        Used by the receiver's bounded (offset, drift) recovery search, which is
        robust against a single mis-step propagating through the payload."""
        bins: list[int] = []
        for k in range(int(count)):
            a = int(round(start_os + offset_os + k * (self.symbol_os + drift_os_per_sym)))
            p = self._power(corrected_os, a, "up")
            if p is None:
                break
            bins.append(int(np.argmax(p)))
        return bins

    def extract_symbols(self, corrected_os: np.ndarray, start_os: int, count: int) -> list[int]:
        """Demodulate ``count`` consecutive symbols starting at ``start_os``.

        Each symbol window is allowed to nudge by +/- ``self.track`` os-samples;
        the most isolated peak is kept.  This tracks slow sampling-clock drift so
        late payload symbols stay aligned without a global grid search.
        """
        bins: list[int] = []
        cursor = int(start_os)
        for _ in range(int(count)):
            best = None
            for d in range(-self.track, self.track + 1):
                spec = self._power(corrected_os, cursor + d, "up")
                if spec is None:
                    continue
                k, pk, iso = self._peak(spec)
                score = iso
                if best is None or score > best[0]:
                    best = (score, k, d)
            if best is None:
                break
            bins.append(int(best[1]))
            cursor += self.symbol_os + best[2]   # carry the nudge forward (drift tracking)
        return bins
