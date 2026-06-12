from __future__ import annotations

import argparse
import time
from collections import defaultdict
from itertools import product
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import numpy as np

from .channelizer import Channelizer, remove_dc
from .config import ReceiverConfig, EU868_CHANNELS
from .detector import PreambleDetector
from .iq_file_source import IQFileSource
from .sdr_source import RtlSdrSource
from .synchronizer import Synchronizer
from .sync2 import Synchronizer2
from .chirps import apply_cfo
from .demodulator import demod_symbols
from .logger import JsonlLogger
from .phy_decode import decode_header, decode_payload, payload_symbol_count
from .exact_zip_decoder import ExactZipHeader, decode_header_exact, decode_payload_exact, payload_symbol_count as exact_payload_symbol_count


def _parse_channels(value: str) -> list[float]:
    if value.strip().lower() in {"eu868", "all", "8"}:
        return list(EU868_CHANNELS)
    return [float(v) for v in value.replace(";", ",").split(",") if v.strip()]


def _iso_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def _expected_profile_enabled(cfg: ReceiverConfig | None) -> bool:
    return bool(
        cfg
        and (
            cfg.expected_payload_len is not None
            or cfg.expected_cr_index is not None
            or cfg.expected_crc_enabled is not None
            or cfg.expected_payload_ascii.strip()
        )
    )


def _expected_ascii(cfg: ReceiverConfig | None) -> str:
    return str(cfg.expected_payload_ascii if cfg else "").strip().upper()


def _phdr_dict_matches_expected(phdr: dict[str, Any], cfg: ReceiverConfig | None) -> bool:
    if not phdr.get("ok"):
        return False
    if not _expected_profile_enabled(cfg):
        return True
    if cfg.expected_payload_len is not None and int(phdr.get("payload_len", -1)) != int(cfg.expected_payload_len):
        return False
    if cfg.expected_cr_index is not None and int(phdr.get("cr_index", -1)) != int(cfg.expected_cr_index):
        return False
    if cfg.expected_crc_enabled is not None and int(phdr.get("crc_enabled", -1)) != int(cfg.expected_crc_enabled):
        return False
    source = str(phdr.get("source", ""))
    return source == "" or "exact_zip" in source


def _legacy_header_matches_expected(header: Any, cfg: ReceiverConfig | None) -> bool:
    return _phdr_dict_matches_expected(header.as_record(), cfg)


def _can_rescue_header_to_expected(header: Any, cfg: ReceiverConfig | None) -> bool:
    if not _expected_profile_enabled(cfg) or not getattr(header, "ok", False):
        return False
    if cfg.expected_payload_len is None or cfg.expected_cr_index is None or cfg.expected_crc_enabled is None:
        return False
    if int(header.cr_index) != int(cfg.expected_cr_index):
        return False
    if int(header.crc_enabled) != int(cfg.expected_crc_enabled):
        return False
    return int(cfg.expected_payload_len) - int(header.payload_len) == 1


def _rescue_header_to_expected(header: Any, cfg: ReceiverConfig) -> ExactZipHeader:
    return ExactZipHeader(
        ok=True,
        payload_len=int(cfg.expected_payload_len),
        cr_index=int(cfg.expected_cr_index),
        cr_denom=int(cfg.expected_cr_index) + 4,
        crc_enabled=int(cfg.expected_crc_enabled),
        extra_header_bits=header.extra_header_bits,
        raw_bins=header.raw_bins,
        failure_reason="",
    )


def _phdr_ok(record: dict[str, Any], cfg: ReceiverConfig | None = None) -> bool:
    for key in ("phdr", "legacy_zip_phdr"):
        phdr = record.get(key)
        if isinstance(phdr, dict) and _phdr_dict_matches_expected(phdr, cfg):
            return True
    return False


def _payload_record_ok(payload: dict[str, Any], record: dict[str, Any], cfg: ReceiverConfig | None) -> bool:
    ascii_text = str(payload.get("payload_ascii", "")).strip().upper()
    expected = _expected_ascii(cfg)
    if _expected_profile_enabled(cfg):
        if not _phdr_ok(record, cfg):
            return False
        if not (payload.get("crc_ok") or payload.get("ok")):
            return False
        if expected and ascii_text != expected:
            return False
        return True
    return bool(payload.get("ok") or payload.get("crc_ok"))


def _payload_decode_ok(record: dict[str, Any], cfg: ReceiverConfig | None = None) -> bool:
    for key in ("payload_decode", "legacy_zip_payload_decode"):
        payload = record.get(key)
        if not isinstance(payload, dict):
            continue
        if _payload_record_ok(payload, record, cfg):
            return True
    return False


def _annotate_candidate_record(record: dict[str, Any], cfg: ReceiverConfig | None = None) -> bool:
    payload_ok = _payload_decode_ok(record, cfg)
    phdr_ok = _phdr_ok(record, cfg)
    if record.get("phdr_profile_rescue") and not payload_ok:
        phdr_ok = False
    accepted = bool(payload_ok if _expected_profile_enabled(cfg) else (payload_ok or phdr_ok))
    if payload_ok:
        reason = "payload_crc_and_expected_ok" if _expected_profile_enabled(cfg) else "payload_crc_ok"
        candidate_class = "full_payload_packet"
    elif phdr_ok:
        reason = "phdr_valid"
        candidate_class = "phdr_only_packet"
    else:
        reason = str(record.get("failure_reason") or "debug_candidate")
        candidate_class = "debug_candidate"
    record["payload_valid"] = payload_ok
    record["phdr_valid"] = phdr_ok
    record["packet_accepted"] = accepted
    record["candidate_class"] = candidate_class
    record["candidate_reason"] = reason
    return accepted


def _should_apply_lockout(record: dict[str, Any], cfg: ReceiverConfig | None = None) -> bool:
    if not record.get("packet_accepted"):
        return False
    return bool(record.get("payload_valid") or not _expected_profile_enabled(cfg))


def _payload_decode_record(record: dict[str, Any]) -> dict[str, Any]:
    for key in ("payload_decode", "clean_payload_decode", "legacy_zip_payload_decode"):
        payload = record.get(key)
        if isinstance(payload, dict):
            return payload
    return {}


def _candidate_record_rank(record: dict[str, Any], cfg: ReceiverConfig | None = None) -> tuple[Any, ...]:
    payload = _payload_decode_record(record)
    expected = _expected_ascii(cfg)
    ascii_text = str(payload.get("payload_ascii", "")).strip().upper()
    crc_ok = bool(payload.get("crc_ok") or payload.get("ok"))
    ascii_match = bool(expected and ascii_text == expected)
    payload_ok = _payload_record_ok(payload, record, cfg) if payload else False
    phdr_ok = _phdr_ok(record, cfg)
    sync = record.get("sync") if isinstance(record.get("sync"), dict) else {}
    det = record.get("detector") if isinstance(record.get("detector"), dict) else {}
    header = record.get("header") if isinstance(record.get("header"), dict) else {}
    payload_meta = record.get("payload") if isinstance(record.get("payload"), dict) else {}

    def finite_float(value: Any, default: float = -1e9) -> float:
        try:
            out = float(value)
            return out if np.isfinite(out) else default
        except Exception:
            return default

    return (
        1 if payload_ok else 0,
        1 if ascii_match else 0,
        1 if crc_ok else 0,
        1 if phdr_ok else 0,
        1 if sync.get("ok") else 0,
        finite_float(sync.get("snr_db", det.get("snr_db"))),
        finite_float(det.get("isolation")),
        -abs(int(header.get("timing_offset_samples") or 0)),
        -abs(int(payload_meta.get("timing_offset_samples") or payload.get("timing_offset_samples") or 0)),
        -abs(float(payload_meta.get("residual_cfo_bins") or payload.get("residual_cfo_bins") or 0.0)),
        -abs(int(payload_meta.get("bin_offset") or payload.get("bin_offset") or 0)),
        -abs(int(record.get("candidate_sample_index") or 0)),
    )


def _candidate_summary(record: dict[str, Any]) -> dict[str, Any]:
    payload = _payload_decode_record(record)
    phdr = record.get("phdr") if isinstance(record.get("phdr"), dict) else {}
    return {
        "candidate_sample_index": record.get("candidate_sample_index"),
        "candidate_class": record.get("candidate_class"),
        "candidate_reason": record.get("candidate_reason"),
        "phdr_ok": bool(phdr.get("ok")),
        "payload_crc_ok": bool(payload.get("crc_ok") or payload.get("ok")),
        "payload_ascii": payload.get("payload_ascii", ""),
        "payload_timing_offset": payload.get("timing_offset_samples"),
        "payload_bin_offset": payload.get("bin_offset"),
        "payload_residual_cfo_bins": payload.get("residual_cfo_bins"),
        "failure_class": payload.get("failure_class", record.get("failure_class", "")),
        "failure_reason": record.get("failure_reason", ""),
    }


def _classify_payload_failure(raw_bins: list[int], good_len: int) -> str:
    if not raw_bins:
        return "payload_no_bins"

    ref = [166, 182, 0, 181, 154, 72, 181, 227, 216, 63]
    if len(raw_bins) == len(ref):
        n = 256
        diffs: list[int] = []
        for a, b in zip(raw_bins, ref):
            d = (int(a) - int(b)) % n
            if d > n // 2:
                d -= n
            diffs.append(abs(d))

        small = sum(1 for d in diffs if d <= 2)
        if small >= 7:
            return "near_jana_symbol_bin_shift"
        if small >= 5:
            return "partial_jana_then_payload_drift"

    if len(raw_bins) < int(good_len):
        return "payload_no_bins"
    return "payload_bins_not_matching_validation_pattern"


class LoRaReceiver:
    def __init__(self, cfg: ReceiverConfig, logger: JsonlLogger):
        self.cfg = cfg
        self.logger = logger
        self.detectors = {
            float(ch): PreambleDetector(
                cfg.sf,
                repeats=cfg.preamble_repeats,
                drift_bins=cfg.preamble_drift_bins,
                min_snr_db=cfg.preamble_snr_db,
                min_isolation=cfg.min_peak_isolation,
            )
            for ch in cfg.channels
        }
        self.sync = Synchronizer(cfg.sf, cfg.bw, cfg.preamble_len, cfg.timing_sweep)
        self.oversample = max(1, int(getattr(cfg, "oversample", 1)))
        self.sync2 = Synchronizer2(cfg.sf, cfg.bw, oversample=self.oversample) if self.oversample > 1 else None
        self.buffers: dict[float, np.ndarray] = defaultdict(lambda: np.empty(0, dtype=np.complex64))
        self.base_index: dict[float, int] = defaultdict(int)
        self.already_seen: dict[float, list[int]] = defaultdict(list)
        self.valid_lockout_until: dict[float, int] = defaultdict(int)

    def _already_processed_candidate(self, ch: float, sample_index: int) -> bool:
        # Suppress re-detections of the SAME physical packet. The detector can
        # re-find a packet in subsequent overlapping buffer blocks at a slightly
        # different absolute index, so match by proximity (within one packet
        # span), not exact index. Genuine retransmissions are seconds apart
        # (>> one packet span), so they are not suppressed.
        span = self._candidate_group_span_samples()
        recent = [x for x in self.already_seen[ch] if sample_index - x < span * 8]
        self.already_seen[ch] = recent
        if any(abs(sample_index - x) < span for x in recent):
            return True
        self.already_seen[ch].append(sample_index)
        return False

    def _candidate_group_span_samples(self) -> int:
        packet_symbols = self.cfg.preamble_len + 4.25 + 8 + self._payload_symbols_to_wait() + 2
        return int(max(self.cfg.preamble_repeats + 2, packet_symbols) * self.cfg.symbol_samples)

    def _group_nearby_candidates(self, candidates: list[Any]) -> list[list[Any]]:
        if not candidates:
            return []
        span = self._candidate_group_span_samples()
        ordered = sorted(candidates, key=lambda c: int(c.sample_index))
        groups: list[list[Any]] = [[ordered[0]]]
        for cand in ordered[1:]:
            if int(cand.sample_index) - int(groups[-1][-1].sample_index) <= span:
                groups[-1].append(cand)
            else:
                groups.append([cand])
        return groups

    def _candidate_has_enough_samples(self, buf: np.ndarray, base: int, cand: Any) -> bool:
        local_start = int(cand.sample_index) - int(base)
        max_positive_offset = max([0] + [x for x in self._payload_timing_offsets() if x > 0])
        max_header_positive_offset = max([0] + [x for x in self._header_timing_offsets() if x > 0])
        # The detector can fire near the start of the preamble, and real preambles
        # (e.g. the 16-symbol LilyGO preamble) are longer than preamble_len; allow a
        # generous preamble + sync + SFD margin so the whole packet is buffered.
        preamble_budget = max(int(self.cfg.preamble_len), 20) + 4.25
        max_need = int(
            (preamble_budget + 8 + self._payload_symbols_to_wait()) * self.cfg.symbol_samples
            + max_header_positive_offset
            + max_positive_offset
        )
        return local_start >= 0 and local_start + max_need <= len(buf)

    def _payload_symbols_to_wait(self) -> int:
        if (
            self.cfg.expected_payload_len is not None
            and self.cfg.expected_cr_index is not None
            and self.cfg.expected_crc_enabled is not None
        ):
            return exact_payload_symbol_count(
                int(self.cfg.expected_payload_len),
                self.cfg.sf,
                int(self.cfg.expected_cr_index),
                int(self.cfg.expected_crc_enabled),
            )
        return exact_payload_symbol_count(
            min(int(self.cfg.max_payload_bytes), 64),
            self.cfg.sf,
            1,
            1,
        )

    def _payload_timing_offsets(self) -> list[int]:
        offsets = [int(x) for x in self.cfg.payload_timing_offsets]
        offsets.extend([-int(self.cfg.symbol_samples), int(self.cfg.symbol_samples)])
        if 0 not in offsets:
            offsets.append(0)
        return sorted(set(offsets), key=lambda x: (abs(x), x))

    def _payload_bin_offsets(self) -> list[int]:
        offsets = [int(x) for x in getattr(self.cfg, "payload_bin_offsets", [0])]
        if 0 not in offsets:
            offsets.append(0)
        return sorted(set(offsets), key=lambda x: (abs(x), x))

    def _payload_residual_cfo_bins(self) -> list[float]:
        vals = [float(x) for x in getattr(self.cfg, "payload_residual_cfo_bins", [0.0])]
        if 0.0 not in vals:
            vals.insert(0, 0.0)
        return sorted(set(vals), key=lambda x: (abs(x), x))

    def _decode_payload_bins_candidate(
        self,
        corrected_bins: list[int],
        offset: int,
        bin_offset: int,
        demod_bins: list[int],
        legacy_header: Any,
        clean_headers: list[Any],
        rec: dict[str, Any],
        expected: str,
        payload_syms: int,
        residual_cfo_bins: float = 0.0,
        residual_cfo_hz: float = 0.0,
        rescue_note: str = "",
    ) -> tuple[Any, dict[str, Any]]:
        payload = decode_payload_exact(corrected_bins, self.cfg.sf, legacy_header)
        attempt = payload.as_record()
        attempt["source"] = "exact_zip_payload_decoder"
        attempt["offset_samples"] = int(offset)
        attempt["bin_offset"] = int(bin_offset)
        attempt["residual_cfo_bins"] = float(residual_cfo_bins)
        attempt["residual_cfo_hz"] = float(residual_cfo_hz)
        attempt["demod_raw_bins"] = [int(x) for x in demod_bins]
        attempt["raw_bins"] = [int(x) for x in corrected_bins[:payload_syms]]
        attempt["expected_symbols"] = int(payload_syms)
        attempt["ascii_matches_expected"] = bool(expected and payload.payload_ascii.strip().upper() == expected)
        if rescue_note:
            attempt["rescue_note"] = rescue_note
        if int(bin_offset) != 0:
            attempt["bin_offset_note"] = "payload FFT bins corrected before decode"

        if not _payload_record_ok(attempt, rec, self.cfg):
            for clean_header in clean_headers:
                clean_payload = decode_payload(corrected_bins, self.cfg.sf, clean_header)
                clean_attempt = clean_payload.as_record()
                clean_attempt["source"] = "clean_payload_decoder_fallback"
                clean_attempt["offset_samples"] = int(offset)
                clean_attempt["bin_offset"] = int(bin_offset)
                clean_attempt["residual_cfo_bins"] = float(residual_cfo_bins)
                clean_attempt["residual_cfo_hz"] = float(residual_cfo_hz)
                clean_attempt["raw_bins"] = [int(x) for x in corrected_bins[:payload_syms]]
                clean_attempt["demod_raw_bins"] = [int(x) for x in demod_bins]
                clean_attempt["expected_symbols"] = int(payload_syms)
                clean_attempt["ascii_matches_expected"] = bool(expected and clean_attempt.get("payload_ascii", "").strip().upper() == expected)
                clean_attempt["phdr_convention"] = getattr(clean_header, "convention", {})
                clean_attempt["legacy_zip_payload_decode"] = attempt
                if rescue_note:
                    clean_attempt["rescue_note"] = rescue_note
                if int(bin_offset) != 0:
                    clean_attempt["bin_offset_note"] = "payload FFT bins corrected before decode"
                if _payload_record_ok(clean_attempt, rec, self.cfg):
                    return clean_payload, clean_attempt
        return payload, attempt

    def _header_timing_offsets(self) -> list[int]:
        offsets = [int(x) for x in self.cfg.payload_timing_offsets]
        offsets.extend([-int(self.cfg.symbol_samples), int(self.cfg.symbol_samples)])
        if 0 not in offsets:
            offsets.append(0)
        return sorted(set(offsets), key=lambda x: (abs(x), x))

    def _decode_header_with_timing_sweep(
        self,
        corrected: np.ndarray,
        nominal_header_start: int,
    ) -> tuple[Any, Any, list[Any], dict[str, Any], Any, int, list[dict[str, Any]]]:
        attempts: list[dict[str, Any]] = []
        best: tuple[Any, Any, list[Any], dict[str, Any], Any, int] | None = None
        best_score: tuple[int, int] | None = None
        for offset in self._header_timing_offsets():
            metrics = demod_symbols(corrected, nominal_header_start + offset, 8, self.cfg.sf, mode="upchirp")
            attempt: dict[str, Any] = {
                "offset_samples": int(offset),
                "raw_bins": metrics.bins,
                "complete": len(metrics.bins) == 8,
            }
            if len(metrics.bins) < 8:
                attempt["failure_reason"] = f"short_header_symbols:{len(metrics.bins)}<8"
                attempts.append(attempt)
                score = (0, -abs(offset))
            else:
                legacy_header = decode_header_exact(metrics.bins, self.cfg.sf)
                selected, candidates, header_debug = decode_header(metrics.bins, self.cfg.sf, exhaustive=True)
                profile_match = _legacy_header_matches_expected(legacy_header, self.cfg)
                attempt["legacy_zip_phdr"] = legacy_header.as_record()
                attempt["phdr_pass_count"] = header_debug["phdr_pass_count"]
                attempt["selection_status"] = header_debug.get("selection_status")
                attempt["phdr_profile_match"] = profile_match
                attempts.append(attempt)
                if profile_match:
                    score = (4, -abs(offset))
                elif legacy_header.ok and not _expected_profile_enabled(self.cfg):
                    score = (3, -abs(offset))
                elif legacy_header.ok:
                    score = (2, -abs(offset))
                elif selected is not None:
                    score = (1, -abs(offset))
                else:
                    score = (0, -abs(offset))
                candidate = (legacy_header, selected, candidates, header_debug, metrics, int(offset))
                if best_score is None or score > best_score:
                    best_score = score
                    best = candidate
            if best is None and len(metrics.bins) == 8:
                # Should only be reached if decoder imports explode; keep a usable fallback.
                legacy_header = decode_header_exact(metrics.bins, self.cfg.sf)
                selected, candidates, header_debug = decode_header(metrics.bins, self.cfg.sf, exhaustive=True)
                best = (legacy_header, selected, candidates, header_debug, metrics, int(offset))
                best_score = score

        if best is not None:
            legacy_header, selected, candidates, header_debug, metrics, offset = best
            return legacy_header, selected, candidates, header_debug, metrics, offset, attempts

        metrics = demod_symbols(corrected, nominal_header_start, 8, self.cfg.sf, mode="upchirp")
        legacy_header = decode_header_exact(metrics.bins, self.cfg.sf)
        selected, candidates, header_debug = decode_header(metrics.bins, self.cfg.sf, exhaustive=True)
        return legacy_header, selected, candidates, header_debug, metrics, 0, attempts

    def _decode_legacy_payload_with_timing_sweep(
        self,
        corrected: np.ndarray,
        nominal_payload_start: int,
        payload_syms: int,
        legacy_header: Any,
        clean_header_candidates: list[Any],
        rec: dict[str, Any],
    ) -> tuple[Any, Any, dict[str, Any]]:
        attempts: list[dict[str, Any]] = []
        best_payload = None
        best_metrics = None
        best_payload_rec: dict[str, Any] | None = None
        best_offset = 0
        best_bin_offset = 0
        best_score: tuple[float, ...] | None = None
        expected = _expected_ascii(self.cfg)
        profile_enabled = _expected_profile_enabled(self.cfg)
        n_bins = 1 << int(self.cfg.sf)
        bin_hz = float(self.cfg.bw) / float(n_bins)
        top_k = max(1, int(getattr(self.cfg, "payload_topk", 1))) if profile_enabled else 1
        max_rescue_symbols = max(0, int(getattr(self.cfg, "payload_topk_max_symbols", 0))) if profile_enabled else 0
        max_rescue_attempts = max(0, int(getattr(self.cfg, "payload_rescue_max_attempts", 0))) if profile_enabled else 0
        deep_enabled = bool(getattr(self.cfg, "payload_deep_rescue_enabled", False)) and profile_enabled
        deep_max_rescue_symbols = max(
            max_rescue_symbols,
            max(0, int(getattr(self.cfg, "payload_deep_topk_max_symbols", max_rescue_symbols))),
        )
        deep_max_rescue_attempts = max(
            max_rescue_attempts,
            max(0, int(getattr(self.cfg, "payload_deep_rescue_max_attempts", max_rescue_attempts))),
        )
        deep_time_budget_s = max(0.0, float(getattr(self.cfg, "payload_deep_rescue_time_budget_s", 0.0)))
        rescue_attempts_used = 0
        deep_rescue_attempts_used = 0
        rescue_failures_suppressed = 0
        rescue_budget_exhausted = False
        deep_rescue_budget_exhausted = False
        clean_headers = [
            h
            for h in clean_header_candidates
            if getattr(h, "ok", False) and _legacy_header_matches_expected(h, self.cfg)
        ]

        corrected_array = corrected if isinstance(corrected, np.ndarray) else np.asarray(corrected, dtype=np.complex64)
        residual_options: list[tuple[float, float, np.ndarray]] = []
        for residual_cfo_bins in self._payload_residual_cfo_bins():
            residual_cfo_hz = float(residual_cfo_bins) * bin_hz
            payload_corrected = (
                apply_cfo(corrected_array, residual_cfo_hz, self.cfg.bw)
                if residual_cfo_hz
                else corrected_array
            )
            residual_options.append((float(residual_cfo_bins), float(residual_cfo_hz), payload_corrected))

        def classify_if_failed(payload_rec: dict[str, Any]) -> None:
            if expected != "JANA" or _payload_record_ok(payload_rec, rec, self.cfg):
                return
            payload_rec["failure_class"] = _classify_payload_failure(
                payload_rec.get("demod_raw_bins") or payload_rec.get("raw_bins") or [],
                payload_syms,
            )

        def consider(
            payload: Any,
            attempt: dict[str, Any],
            metrics: Any,
            offset: int,
            rescue_rank: int = 0,
            store_attempt: bool = True,
        ) -> bool:
            nonlocal best_payload, best_metrics, best_payload_rec, best_offset, best_bin_offset, best_score
            accepted = _payload_record_ok(attempt, rec, self.cfg)
            if not accepted:
                classify_if_failed(attempt)
            if store_attempt or accepted:
                attempts.append(attempt)
            if not store_attempt and not accepted:
                return False
            score = (
                1.0 if accepted else 0.0,
                1.0 if attempt.get("ascii_matches_expected") else 0.0,
                float(rescue_rank),
                -abs(int(offset)),
                -abs(float(attempt.get("residual_cfo_bins") or 0.0)),
                -abs(int(attempt.get("bin_offset") or 0)),
            )
            if best_score is None or score > best_score:
                best_score = score
                best_payload = payload
                best_metrics = metrics
                best_payload_rec = attempt
                best_offset = int(offset)
                best_bin_offset = int(attempt.get("bin_offset") or 0)
            return accepted

        def try_symbol_rescue(
            metrics: Any,
            demod_bins: list[int],
            offset: int,
            residual_cfo_bins: float,
            residual_cfo_hz: float,
            max_symbols: int,
            max_attempts: int,
            rescue_rank: int,
            deadline: float | None,
            deep: bool,
        ) -> bool:
            nonlocal rescue_attempts_used, deep_rescue_attempts_used
            nonlocal rescue_failures_suppressed, rescue_budget_exhausted, deep_rescue_budget_exhausted
            if top_k <= 1 or max_symbols <= 0 or max_attempts <= 0 or not metrics.top_bins:
                return False
            usable = min(payload_syms, len(metrics.top_bins), len(metrics.isolation))
            weak = sorted(range(usable), key=lambda i: float(metrics.isolation[i]))[:max_symbols]
            choices: list[list[int]] = []
            rescue_positions: list[int] = []
            for i in weak:
                alts: list[int] = []
                for b in metrics.top_bins[i][:top_k]:
                    bb = int(b) % n_bins
                    if bb not in alts:
                        alts.append(bb)
                if demod_bins[i] not in alts:
                    alts.insert(0, demod_bins[i])
                if len(alts) > 1:
                    rescue_positions.append(i)
                    choices.append(alts[:top_k])

            for combo in product(*choices) if choices else []:
                if deadline is not None and time.perf_counter() > deadline:
                    deep_rescue_budget_exhausted = True
                    return False
                if deep:
                    if deep_rescue_attempts_used >= max_attempts:
                        deep_rescue_budget_exhausted = True
                        return False
                    deep_rescue_attempts_used += 1
                else:
                    if rescue_attempts_used >= max_attempts:
                        rescue_budget_exhausted = True
                        return False
                    rescue_attempts_used += 1

                trial_bins = list(demod_bins)
                changed: list[dict[str, int]] = []
                for pos, val in zip(rescue_positions, combo):
                    if int(val) != int(trial_bins[pos]):
                        changed.append({"sym": int(pos), "from": int(trial_bins[pos]), "to": int(val)})
                        trial_bins[pos] = int(val)
                if not changed:
                    continue
                for bin_offset in self._payload_bin_offsets():
                    corrected_bins = [(b + int(bin_offset)) % n_bins for b in trial_bins]
                    prefix = "deep_topk_symbol_rescue" if deep else "topk_symbol_rescue"
                    note = prefix + ":" + ",".join(f"s{x['sym']}:{x['from']}->{x['to']}" for x in changed)
                    payload, attempt = self._decode_payload_bins_candidate(
                        corrected_bins,
                        int(offset),
                        int(bin_offset),
                        demod_bins,
                        legacy_header,
                        clean_headers,
                        rec,
                        expected,
                        payload_syms,
                        residual_cfo_bins=float(residual_cfo_bins),
                        residual_cfo_hz=float(residual_cfo_hz),
                        rescue_note=note,
                    )
                    attempt["topk_rescue_changed_symbols"] = changed
                    attempt["topk_rescue_trial_bins"] = trial_bins
                    if consider(payload, attempt, metrics, int(offset), rescue_rank=rescue_rank, store_attempt=False):
                        return True
                    rescue_failures_suppressed += 1
            return False

        accepted_any = False
        for offset in self._payload_timing_offsets():
            if accepted_any:
                break
            for residual_cfo_bins, residual_cfo_hz, payload_corrected in residual_options:
                if accepted_any:
                    break
                try:
                    metrics = demod_symbols(
                        payload_corrected,
                        nominal_payload_start + offset,
                        payload_syms,
                        self.cfg.sf,
                        mode="upchirp",
                        top_k=top_k,
                    )
                except TypeError:
                    metrics = demod_symbols(payload_corrected, nominal_payload_start + offset, payload_syms, self.cfg.sf, mode="upchirp")
                if len(metrics.bins) < payload_syms:
                    attempt = {
                        "offset_samples": int(offset),
                        "bin_offset": 0,
                        "residual_cfo_bins": float(residual_cfo_bins),
                        "residual_cfo_hz": float(residual_cfo_hz),
                        "ok": False,
                        "crc_ok": False,
                        "payload_ascii": "",
                        "payload_hex": "",
                        "failure_reason": f"short_payload_symbols:{len(metrics.bins)}<{payload_syms}",
                        "raw_bins": metrics.bins,
                    }
                    classify_if_failed(attempt)
                    attempts.append(attempt)
                    score = (
                        0.0,
                        0.0,
                        0.0,
                        -abs(int(offset)),
                        -abs(float(residual_cfo_bins)),
                        0.0,
                    )
                    if best_score is None or score > best_score:
                        best_score = score
                        best_payload = None
                        best_metrics = metrics
                        best_payload_rec = attempt
                        best_offset = int(offset)
                        best_bin_offset = 0
                    continue

                demod_bins = [int(b) for b in metrics.bins[:payload_syms]]

                for bin_offset in self._payload_bin_offsets():
                    corrected_bins = [(b + int(bin_offset)) % n_bins for b in demod_bins]
                    payload, attempt = self._decode_payload_bins_candidate(
                        corrected_bins,
                        int(offset),
                        int(bin_offset),
                        demod_bins,
                        legacy_header,
                        clean_headers,
                        rec,
                        expected,
                        payload_syms,
                        residual_cfo_bins=float(residual_cfo_bins),
                        residual_cfo_hz=float(residual_cfo_hz),
                    )
                    if consider(payload, attempt, metrics, int(offset), rescue_rank=0):
                        accepted_any = True
                        break
                if accepted_any:
                    break

                accepted_any = try_symbol_rescue(
                    metrics,
                    demod_bins,
                    int(offset),
                    float(residual_cfo_bins),
                    float(residual_cfo_hz),
                    max_rescue_symbols,
                    max_rescue_attempts,
                    rescue_rank=1,
                    deadline=None,
                    deep=False,
                )

        if not accepted_any and deep_enabled and top_k > 1 and deep_max_rescue_symbols > 0 and deep_max_rescue_attempts > 0:
            deadline = time.perf_counter() + deep_time_budget_s if deep_time_budget_s > 0 else None
            for offset in self._payload_timing_offsets():
                if accepted_any:
                    break
                for residual_cfo_bins, residual_cfo_hz, payload_corrected in residual_options:
                    if deadline is not None and time.perf_counter() > deadline:
                        deep_rescue_budget_exhausted = True
                        break
                    try:
                        metrics = demod_symbols(
                            payload_corrected,
                            nominal_payload_start + offset,
                            payload_syms,
                            self.cfg.sf,
                            mode="upchirp",
                            top_k=top_k,
                        )
                    except TypeError:
                        metrics = demod_symbols(payload_corrected, nominal_payload_start + offset, payload_syms, self.cfg.sf, mode="upchirp")
                    if len(metrics.bins) < payload_syms:
                        continue
                    demod_bins = [int(b) for b in metrics.bins[:payload_syms]]
                    accepted_any = try_symbol_rescue(
                        metrics,
                        demod_bins,
                        int(offset),
                        float(residual_cfo_bins),
                        float(residual_cfo_hz),
                        deep_max_rescue_symbols,
                        deep_max_rescue_attempts,
                        rescue_rank=2,
                        deadline=deadline,
                        deep=True,
                    )
                    if accepted_any:
                        break
                if deep_rescue_budget_exhausted and not accepted_any:
                    break
        rec["payload_timing_sweep"] = attempts
        rec["payload_topk_rescue"] = {
            "enabled": bool(top_k > 1 and max_rescue_symbols > 0 and max_rescue_attempts > 0),
            "top_k": int(top_k),
            "max_symbols": int(max_rescue_symbols),
            "max_attempts": int(max_rescue_attempts),
            "attempts_used": int(rescue_attempts_used),
            "deep_enabled": bool(deep_enabled),
            "deep_max_symbols": int(deep_max_rescue_symbols),
            "deep_max_attempts": int(deep_max_rescue_attempts),
            "deep_time_budget_s": float(deep_time_budget_s),
            "deep_attempts_used": int(deep_rescue_attempts_used),
            "failures_suppressed": int(rescue_failures_suppressed),
            "budget_exhausted": bool(rescue_budget_exhausted),
            "deep_budget_exhausted": bool(deep_rescue_budget_exhausted),
            "residual_cfo_bins": [float(x) for x in self._payload_residual_cfo_bins()],
            "residual_cfo_bin_hz": float(bin_hz),
        }
        if best_payload is None or best_metrics is None:
            # Keep a normal payload-shaped result even when every sweep offset was short.
            try:
                metrics = demod_symbols(corrected_array, nominal_payload_start, payload_syms, self.cfg.sf, mode="upchirp", top_k=top_k)
            except TypeError:
                metrics = demod_symbols(corrected_array, nominal_payload_start, payload_syms, self.cfg.sf, mode="upchirp")
            payload = decode_payload_exact(metrics.bins, self.cfg.sf, legacy_header)
            payload_rec = payload.as_record()
            payload_rec["timing_offset_samples"] = 0
            payload_rec["bin_offset"] = 0
            payload_rec["residual_cfo_bins"] = 0.0
            payload_rec["residual_cfo_hz"] = 0.0
            payload_rec["demod_raw_bins"] = [int(x) for x in metrics.bins[:payload_syms]]
            payload_rec["raw_bins"] = [int(x) for x in metrics.bins[:payload_syms]]
            payload_rec["expected_symbols"] = int(payload_syms)
            payload_rec["ascii_matches_expected"] = bool(expected and payload_rec.get("payload_ascii", "").strip().upper() == expected)
            classify_if_failed(payload_rec)
            return payload, metrics, payload_rec
        payload_rec = dict(best_payload_rec) if best_payload_rec is not None else best_payload.as_record()
        payload_rec["timing_offset_samples"] = int(best_offset)
        payload_rec["bin_offset"] = int(best_bin_offset)
        payload_rec.setdefault("residual_cfo_bins", 0.0)
        payload_rec.setdefault("residual_cfo_hz", 0.0)
        payload_rec["ascii_matches_expected"] = bool(expected and payload_rec.get("payload_ascii", "").strip().upper() == expected)
        classify_if_failed(payload_rec)
        return best_payload, best_metrics, payload_rec

    def process_channel(self, channel_hz: float, samples: np.ndarray, channel_sample_rate: float) -> int:
        # The demodulator works at an integer number of samples per LoRa chip.
        # With oversample>1 the channel carries 2^SF*oversample samples/symbol so
        # the SFD-locked joint CFO/STO synchronizer can resolve sub-chip timing.
        os = self.oversample
        target_rate = self.cfg.bw * os
        if abs(channel_sample_rate - target_rate) > 1.0:
            from scipy import signal
            import math

            g = math.gcd(int(round(channel_sample_rate)), int(round(target_rate)))
            up = int(round(target_rate)) // g
            down = int(round(channel_sample_rate)) // g
            samples = signal.resample_poly(samples, up, down).astype(np.complex64)

        ch = float(channel_hz)
        self.buffers[ch] = np.concatenate([self.buffers[ch], remove_dc(samples)])
        buf = self.buffers[ch]
        base = self.base_index[ch]
        detector = self.detectors[ch]
        packets_logged = 0

        candidates = detector.scan(buf, base_sample_index=base, hop=self.cfg.symbol_samples)
        if os > 1:
            # The preamble detector demodulates 2^SF-sample windows (1 sample/chip),
            # so it must run on a decimated view. Candidate indices come back in the
            # decimated grid; remap them to the oversampled buffer's absolute index.
            buf1 = buf[::os]
            n_chip = 1 << self.cfg.sf
            cands1 = detector.scan(buf1, base_sample_index=0, hop=n_chip)
            for c in cands1:
                local1 = int(c.sample_index)
                c.sample_index = int(base + local1 * os)
            candidates = cands1
        ready_candidates: list[Any] = []
        for cand in candidates:
            if cand.sample_index < self.valid_lockout_until[ch]:
                continue
            # Wait for more live samples if a candidate arrives near the end of the buffer.
            if not self._candidate_has_enough_samples(buf, base, cand):
                continue
            if self._already_processed_candidate(ch, cand.sample_index):
                continue
            ready_candidates.append(cand)

        for group in self._group_nearby_candidates(ready_candidates):
            if not group or int(group[0].sample_index) < self.valid_lockout_until[ch]:
                continue
            decoded: list[tuple[tuple[Any, ...], Any, dict[str, Any]]] = []
            for cand in group:
                if os > 1 and self.sync2 is not None:
                    record = self.decode_candidate_v2(ch, buf, base, cand.sample_index, cand)
                else:
                    record = self.decode_candidate(ch, buf, base, cand.sample_index, cand)
                _annotate_candidate_record(record, self.cfg)
                decoded.append((_candidate_record_rank(record, self.cfg), cand, record))
            if not decoded:
                continue

            decoded.sort(key=lambda item: item[0], reverse=True)
            _rank, cand, record = decoded[0]
            losers = [item[2] for item in decoded[1:]]
            record["candidate_selection"] = {
                "mode": "nearby_group_best_rank",
                "group_size": len(decoded),
                "group_span_samples": self._candidate_group_span_samples(),
                "sample_indices": [int(item[1].sample_index) for item in decoded],
                "selected_sample_index": int(cand.sample_index),
                "selected_rank": list(_rank),
                "ranking": "payload_crc_expected > payload_crc > phdr_crc > sync_metrics > small_offsets",
            }
            if losers:
                max_losers = 5
                record["candidate_group_losers"] = [_candidate_summary(r) for r in losers[:max_losers]]
                record["candidate_group_losers_suppressed"] = max(0, len(losers) - max_losers)

            accepted = bool(record.get("packet_accepted"))
            if _should_apply_lockout(record, self.cfg):
                lockout_samples = max(0, int(round(float(self.cfg.valid_lockout_seconds) * float(self.cfg.bw))))
                self.valid_lockout_until[ch] = max(self.valid_lockout_until[ch], int(cand.sample_index) + lockout_samples)
                record["lockout_seconds"] = float(self.cfg.valid_lockout_seconds)
                record["lockout_until_sample_index"] = int(self.valid_lockout_until[ch])
            if accepted or record.get("candidate_class") == "phdr_only_packet" or self.cfg.log_debug_candidates:
                self.logger.write(record)
            if accepted:
                packets_logged += 1
            if self.cfg.max_packets and packets_logged >= self.cfg.max_packets:
                break

        # Keep enough overlap for a full packet candidate and timing sweep.
        keep = int(self.cfg.max_packet_symbols * self.cfg.symbol_samples)
        if len(self.buffers[ch]) > keep:
            drop = len(self.buffers[ch]) - keep
            self.buffers[ch] = self.buffers[ch][drop:]
            self.base_index[ch] += drop
        return packets_logged

    def decode_candidate_v2(self, ch: float, buf: np.ndarray, base: int, cand_index: int, cand: Any) -> dict[str, Any]:
        """Oversampled decode path: SFD-locked joint CFO/STO sync + drift tracking.

        Produces the same record schema as ``decode_candidate`` so the GUI,
        ranking and lockout logic are unchanged.
        """
        sync = self.sync2
        os = self.oversample
        local_os = int(cand_index) - int(base)
        try:
            channel_index = min(range(len(self.cfg.channels)), key=lambda i: abs(float(self.cfg.channels[i]) - float(ch)))
        except Exception:
            channel_index = -1
        rec: dict[str, Any] = {
            "timestamp": _iso_now(),
            "channel_index": int(channel_index),
            "channel_hz": float(ch),
            "center_hz": float(self.cfg.center_freq),
            "sample_rate": float(self.cfg.sample_rate),
            "sf": int(self.cfg.sf),
            "bw": float(self.cfg.bw),
            "bw_hz": float(self.cfg.bw),
            "oversample": int(os),
            "candidate_sample_index": int(cand_index),
            "detector": {
                "preamble_bins": cand.bins,
                "snr_db": cand.snr_db,
                "isolation": cand.isolation,
                "rssi_db": cand.rssi_db,
            },
        }
        sr = sync.synchronize(buf, local_os)
        rec["sync"] = {
            "ok": sr.ok,
            "reason": sr.reason,
            "sfd_start": int(base + sr.sfd_start_os),
            "header_start": int(base + sr.header_start_os),
            "cfo_hz": sr.cfo_hz,
            "cfo_bins": sr.cfo_bins,
            "sto_bins": sr.sto_bins,
            "preamble_bins": sr.preamble_bins,
            "sfd_bins": sr.sfd_bins,
            "snr_db": sr.snr_db,
            "rssi_db": sr.rssi_db,
            "synchronizer": "Synchronizer2_sfd_locked_joint_cfo_sto",
        }
        if _expected_profile_enabled(self.cfg):
            rec["expected_profile"] = {
                "payload_len": self.cfg.expected_payload_len,
                "cr_index": self.cfg.expected_cr_index,
                "cr_denom": None if self.cfg.expected_cr_index is None else int(self.cfg.expected_cr_index) + 4,
                "crc_enabled": self.cfg.expected_crc_enabled,
                "payload_ascii": self.cfg.expected_payload_ascii,
            }
        if not sr.ok:
            rec["failure_reason"] = sr.reason
            return rec

        # CFO-correct the oversampled buffer once, then extract symbols with
        # per-symbol drift tracking.
        idx = np.arange(len(buf), dtype=np.float64)
        corrected = (buf * np.exp(-1j * 2.0 * np.pi * sr.cfo_hz * idx / (self.cfg.bw * os))).astype(np.complex64)

        header_bins = sync.extract_symbols(corrected, sr.header_start_os, 8)
        rec["header"] = {
            "raw_bins": header_bins,
            "timing_offset_samples": 0,
            "nominal_start": int(local_os and sr.header_start_os),
            "selected_start": int(sr.header_start_os),
        }
        if len(header_bins) < 8:
            rec["failure_reason"] = "short_header_symbols"
            return rec

        legacy_header = decode_header_exact(header_bins, self.cfg.sf)
        selected, candidates, header_debug = decode_header(header_bins, self.cfg.sf, exhaustive=True)
        rec["legacy_zip_phdr"] = legacy_header.as_record()
        rec["header_debug"] = header_debug
        rec["phdr_candidates"] = [c.as_record() for c in candidates if c.ok]
        rec["phdr_pass_count"] = header_debug["phdr_pass_count"]

        # choose the header to drive payload sizing (legacy first, else clean)
        if legacy_header.ok:
            PL, CR, CRC = int(legacy_header.payload_len), int(legacy_header.cr_index), int(legacy_header.crc_enabled)
            rec["phdr"] = {
                "source": "exact_zip_Decode_Symbols.py",
                "payload_len": legacy_header.payload_len,
                "cr_index": legacy_header.cr_index,
                "cr_denom": legacy_header.cr_denom,
                "crc_enabled": legacy_header.crc_enabled,
                "extra_payload_bits": legacy_header.extra_header_bits,
                "ok": True,
            }
            rec["phdr_profile_match"] = _legacy_header_matches_expected(legacy_header, self.cfg)
        elif selected is not None:
            PL, CR, CRC = int(selected.payload_len), int(selected.cr_index), int(selected.crc_enabled)
            rec["phdr"] = selected.as_record()
            rec["phdr"]["source"] = "clean_exhaustive_decoder"
        else:
            rec["failure_reason"] = "phdr_crc_no_unique_candidate"
            return rec

        if _expected_profile_enabled(self.cfg) and not _phdr_dict_matches_expected(rec["phdr"], self.cfg):
            rec["failure_reason"] = f"unexpected_phdr_profile:L={PL} CR=4/{CR + 4} CRC={CRC}"
            return rec

        payload_syms = exact_payload_symbol_count(PL, self.cfg.sf, CR, CRC)
        payload_bins = sync.extract_symbols(corrected, sr.payload_start_os, payload_syms)
        rec["payload"] = {
            "expected_symbols": int(payload_syms),
            "raw_bins": payload_bins,
            "corrected_bins": payload_bins,
            "timing_offset_samples": 0,
            "bin_offset": 0,
            "residual_cfo_bins": 0.0,
            "residual_cfo_hz": 0.0,
        }
        if len(payload_bins) < payload_syms:
            rec["failure_reason"] = f"short_payload_symbols:{len(payload_bins)}<{payload_syms}"
            return rec

        def _decode_bins(bins: list[int], offset_os: float = 0.0, drift: float = 0.0) -> tuple[dict | None, dict | None]:
            """Run legacy then clean payload decoders on a set of bins.
            Returns (accepted_record_or_None, last_record)."""
            last = None
            if legacy_header.ok:
                lp = decode_payload_exact(bins, self.cfg.sf, legacy_header)
                lr = lp.as_record()
                lr.update(source="exact_zip_payload_decoder", raw_bins=bins[:payload_syms],
                          demod_raw_bins=bins[:payload_syms], expected_symbols=int(payload_syms),
                          offset_samples=float(offset_os), drift_os_per_symbol=float(drift))
                last = lr
                if _payload_record_ok(lr, rec, self.cfg):
                    return lr, lr
            for clean_header in ([selected] if selected is not None else []) + [c for c in candidates if c.ok]:
                cp = decode_payload(bins, self.cfg.sf, clean_header)
                cr = cp.as_record()
                cr.update(source="clean_payload_decoder", raw_bins=bins[:payload_syms],
                          demod_raw_bins=bins[:payload_syms], expected_symbols=int(payload_syms),
                          phdr_convention=getattr(clean_header, "convention", {}),
                          offset_samples=float(offset_os), drift_os_per_symbol=float(drift))
                last = cr
                if _payload_record_ok(cr, rec, self.cfg):
                    return cr, cr
            return None, last

        chosen_rec, last_rec = _decode_bins(payload_bins)
        if chosen_rec is not None:
            rec["legacy_zip_payload_decode" if chosen_rec["source"].startswith("exact") else "clean_payload_decode"] = chosen_rec
        else:
            # Bounded recovery: re-extract on a fixed linear grid (no greedy carry)
            # sweeping a small payload-start micro-offset and a per-symbol drift
            # (sampling-clock / SFO). Accept on payload CRC. This fixes the
            # "good prefix then tail corruption" timing slip seen on real captures.
            o_os = float(os)
            offsets = sorted({round(s, 3) for s in np.arange(-2 * o_os, 2 * o_os + 0.25, 0.5)} | {0.0}, key=abs)
            drifts = sorted({round(d, 3) for d in np.arange(-2.5, 2.5 + 0.001, 0.1)} | {0.0}, key=abs)
            search = {"grid": f"offset±{int(2*o_os)}os/0.5 x drift±2.5os/sym/0.1", "tried": 0}
            for drift in drifts:
                for off in offsets:
                    if off == 0.0 and drift == 0.0:
                        continue
                    search["tried"] += 1
                    bins = sync.extract_symbols_linear(corrected, sr.payload_start_os, payload_syms, off, drift)
                    if len(bins) < payload_syms:
                        continue
                    acc, last_rec = _decode_bins(bins, off, drift)
                    if acc is not None:
                        acc["recovery"] = {"offset_os": off, "drift_os_per_symbol": drift}
                        rec["payload_recovery"] = {**search, "offset_os": off, "drift_os_per_symbol": drift}
                        chosen_rec = acc
                        break
                if chosen_rec is not None:
                    break
            if chosen_rec is None:
                rec["payload_recovery"] = {**search, "result": "no_crc_pass"}
            rec["clean_payload_decode" if (chosen_rec or last_rec or {}).get("source", "").startswith("clean") else "legacy_zip_payload_decode"] = chosen_rec or last_rec

        payload_record = chosen_rec or rec.get("legacy_zip_payload_decode") or rec.get("clean_payload_decode") or {}
        rec["payload_decode"] = payload_record
        if _payload_record_ok(payload_record, rec, self.cfg):
            rec["failure_reason"] = ""
        else:
            rec["failure_reason"] = payload_record.get("failure_reason") or "payload_crc_fail"
            if _expected_ascii(self.cfg) == "JANA":
                payload_record["failure_class"] = _classify_payload_failure(
                    payload_record.get("demod_raw_bins") or payload_record.get("raw_bins") or [], payload_syms
                )
        return rec

    def decode_candidate(self, ch: float, buf: np.ndarray, base: int, cand_index: int, cand: Any) -> dict[str, Any]:
        sync = self.sync.synchronize(buf, cand_index, base_sample_index=base)
        try:
            channel_index = min(range(len(self.cfg.channels)), key=lambda i: abs(float(self.cfg.channels[i]) - float(ch)))
        except Exception:
            channel_index = -1
        rec: dict[str, Any] = {
            "timestamp": _iso_now(),
            "channel_index": int(channel_index),
            "channel_hz": float(ch),
            "center_hz": float(self.cfg.center_freq),
            "sample_rate": float(self.cfg.sample_rate),
            "sf": int(self.cfg.sf),
            "bw": float(self.cfg.bw),
            "bw_hz": float(self.cfg.bw),
            "candidate_sample_index": int(cand_index),
            "detector": {
                "preamble_bins": cand.bins,
                "snr_db": cand.snr_db,
                "isolation": cand.isolation,
                "rssi_db": cand.rssi_db,
            },
            "sync": {
                "ok": sync.ok,
                "reason": sync.reason,
                "preamble_start": sync.preamble_start,
                "sfd_start": sync.sfd_start,
                "header_start": sync.header_start,
                "sto_samples": sync.sto,
                "cfo_hz": sync.cfo_hz,
                "preamble_bins": sync.preamble_bins,
                "sfd_bins": sync.sfd_bins,
                "sfd_bins_display": "raw_sync_bins",
                "sfd_bins_note": "Raw SFD/downchirp bins from synchronization; PHDR/payload CRC is the acceptance signal.",
                "snr_db": sync.snr_db,
                "rssi_db": sync.rssi_db,
            },
        }
        if _expected_profile_enabled(self.cfg):
            rec["expected_profile"] = {
                "payload_len": self.cfg.expected_payload_len,
                "cr_index": self.cfg.expected_cr_index,
                "cr_denom": None if self.cfg.expected_cr_index is None else int(self.cfg.expected_cr_index) + 4,
                "crc_enabled": self.cfg.expected_crc_enabled,
                "payload_ascii": self.cfg.expected_payload_ascii,
            }
        if not sync.ok:
            rec["failure_reason"] = sync.reason
            return rec

        local_header = sync.header_start - base
        corrected = apply_cfo(buf, sync.cfo_hz, self.cfg.bw)
        legacy_header, selected, candidates, header_debug, header_metrics, header_offset, header_attempts = self._decode_header_with_timing_sweep(
            corrected,
            local_header,
        )
        header_bins = header_metrics.bins
        rec["header_timing_sweep"] = header_attempts
        rec["header"] = {
            "raw_bins": header_bins,
            "snr_db": header_metrics.snr_db,
            "isolation": header_metrics.isolation,
            "timing_offset_samples": int(header_offset),
            "nominal_start": int(local_header),
            "selected_start": int(local_header + header_offset),
        }
        if len(header_bins) < 8:
            rec["failure_reason"] = "short_header_symbols"
            return rec

        # First run the ORIGINAL Decode_Symbols.py header decoder copied from the ZIP.
        # This is the exact legacy PHDR path the user asked to preserve.
        rec["legacy_zip_phdr"] = legacy_header.as_record()

        # Also keep the clean exhaustive decoder as a diagnostic cross-check.
        rec["header_debug"] = header_debug
        rec["phdr_candidates"] = [c.as_record() for c in candidates if c.ok]
        rec["phdr_pass_count"] = header_debug["phdr_pass_count"]

        if legacy_header.ok:
            decode_header = legacy_header
            rescue_reason = ""
            if _expected_profile_enabled(self.cfg) and not _legacy_header_matches_expected(legacy_header, self.cfg):
                if _can_rescue_header_to_expected(legacy_header, self.cfg):
                    decode_header = _rescue_header_to_expected(legacy_header, self.cfg)
                    rescue_reason = (
                        f"expected_len_rescue:L={legacy_header.payload_len}->"
                        f"{decode_header.payload_len}"
                    )

            payload_syms = exact_payload_symbol_count(
                int(decode_header.payload_len),
                self.cfg.sf,
                int(decode_header.cr_index),
                int(decode_header.crc_enabled),
            )
            rec["phdr"] = {
                "source": "exact_zip_Decode_Symbols.py" if not rescue_reason else "exact_zip_Decode_Symbols.py+expected_profile_rescue",
                "payload_len": decode_header.payload_len,
                "cr_index": decode_header.cr_index,
                "cr_denom": decode_header.cr_denom,
                "crc_enabled": decode_header.crc_enabled,
                "extra_payload_bits": decode_header.extra_header_bits,
                "ok": True,
            }
            rec["phdr_profile_match"] = _legacy_header_matches_expected(decode_header, self.cfg)
            if rescue_reason:
                rec["phdr_profile_rescue"] = {
                    "enabled": True,
                    "reason": rescue_reason,
                    "legacy_zip_phdr": legacy_header.as_record(),
                    "rescued_phdr": decode_header.as_record(),
                }
            if _expected_profile_enabled(self.cfg) and not rec["phdr_profile_match"]:
                rec["failure_reason"] = (
                    f"unexpected_phdr_profile:L={legacy_header.payload_len} "
                    f"CR=4/{legacy_header.cr_denom} CRC={legacy_header.crc_enabled}"
                )
                return rec

            payload_start = local_header + header_offset + 8 * self.cfg.symbol_samples
            legacy_payload, payload_metrics, payload_record = self._decode_legacy_payload_with_timing_sweep(
                corrected,
                payload_start,
                payload_syms,
                decode_header,
                candidates,
                rec,
            )
            rec["payload"] = {
                "expected_symbols": payload_syms,
                "raw_bins": payload_metrics.bins,
                "corrected_bins": payload_record.get("raw_bins", payload_metrics.bins),
                "snr_db": payload_metrics.snr_db,
                "isolation": payload_metrics.isolation,
                "timing_offset_samples": payload_record.get("timing_offset_samples", 0),
                "bin_offset": payload_record.get("bin_offset", 0),
                "residual_cfo_bins": payload_record.get("residual_cfo_bins", 0.0),
                "residual_cfo_hz": payload_record.get("residual_cfo_hz", 0.0),
                "failure_class": payload_record.get("failure_class", ""),
            }
            if len(payload_metrics.bins) < payload_syms:
                rec["failure_reason"] = f"short_payload_symbols:{len(payload_metrics.bins)}<{payload_syms}"
                return rec
            nested_legacy_payload = payload_record.pop("legacy_zip_payload_decode", None)
            if nested_legacy_payload is not None:
                rec["legacy_zip_payload_decode"] = nested_legacy_payload
                rec["clean_payload_decode"] = payload_record
            else:
                rec["legacy_zip_payload_decode"] = payload_record
            rec["payload_decode"] = payload_record
            if _payload_record_ok(payload_record, rec, self.cfg):
                rec["failure_reason"] = ""
            else:
                rec["failure_reason"] = legacy_payload.failure_reason or "payload_crc_fail_after_timing_sweep"
                if payload_record.get("failure_class"):
                    rec["failure_class"] = payload_record.get("failure_class")
            return rec

        # If the exact ZIP header decoder fails, fall back to the clean diagnostic path.
        if selected is None:
            rec["failure_reason"] = "phdr_crc_no_unique_candidate"
            return rec

        payload_syms = payload_symbol_count(selected.payload_len, self.cfg.sf, selected.cr_index, selected.crc_enabled)
        payload_start = local_header + header_offset + 8 * self.cfg.symbol_samples
        rec["phdr"] = selected.as_record()
        rec["phdr"]["source"] = "clean_exhaustive_decoder_fallback"
        if _expected_profile_enabled(self.cfg) and not _phdr_dict_matches_expected(rec["phdr"], self.cfg):
            rec["failure_reason"] = (
                f"unexpected_phdr_profile:L={selected.payload_len} "
                f"CR=4/{int(selected.cr_index) + 4} CRC={selected.crc_enabled}"
            )
            return rec
        payload_metrics = demod_symbols(corrected, payload_start, payload_syms, self.cfg.sf, mode="upchirp")
        rec["payload"] = {
            "expected_symbols": payload_syms,
            "raw_bins": payload_metrics.bins,
            "corrected_bins": payload_metrics.bins,
            "snr_db": payload_metrics.snr_db,
            "isolation": payload_metrics.isolation,
            "timing_offset_samples": 0,
            "bin_offset": 0,
        }
        if len(payload_metrics.bins) < payload_syms:
            rec["failure_reason"] = f"short_payload_symbols:{len(payload_metrics.bins)}<{payload_syms}"
            return rec

        payload_dec = decode_payload(payload_metrics.bins, self.cfg.sf, selected)
        payload_rec = payload_dec.as_record()
        payload_rec["source"] = "clean_payload_decoder_fallback"
        payload_rec["timing_offset_samples"] = 0
        payload_rec["bin_offset"] = 0
        payload_rec["residual_cfo_bins"] = 0.0
        payload_rec["residual_cfo_hz"] = 0.0
        payload_rec["demod_raw_bins"] = payload_metrics.bins[:payload_syms]
        payload_rec["raw_bins"] = payload_metrics.bins[:payload_syms]
        payload_rec["expected_symbols"] = int(payload_syms)
        if _expected_ascii(self.cfg) == "JANA" and not _payload_record_ok(payload_rec, rec, self.cfg):
            payload_rec["failure_class"] = _classify_payload_failure(
                payload_rec.get("demod_raw_bins") or payload_rec.get("raw_bins") or [],
                payload_syms,
            )
        rec["payload_decode"] = payload_rec
        if not payload_dec.ok:
            rec["failure_reason"] = payload_dec.failure_reason or "payload_decode_failed"
            if payload_rec.get("failure_class"):
                rec["failure_class"] = payload_rec.get("failure_class")
        else:
            rec["failure_reason"] = ""
        return rec


def make_source(args):
    if args.source == "file":
        if not args.iq:
            raise SystemExit("--iq is required when --source file")
        return IQFileSource(args.iq, args.block_size)
    return RtlSdrSource(args.center, args.sample_rate, args.gain, args.block_size)


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Clean LoRaPHY SDR scanner backend, thesis starter version")
    p.add_argument("--source", choices=["file", "rtlsdr"], default="file")
    p.add_argument("--iq", help="offline IQ file: .npz/.npy/.dat/.cf32")
    p.add_argument("--center", type=float, default=868.8e6, help="wideband RF center in Hz")
    p.add_argument("--channels", default="eu868", help="comma list or 'eu868'")
    p.add_argument("--sf", type=int, default=8)
    p.add_argument("--bw", type=float, default=125e3)
    p.add_argument("--sample-rate", type=float, default=2.0e6)
    p.add_argument("--oversample", type=int, default=4, help="channel samples per LoRa chip (>=2 enables the SFD-locked joint CFO/STO synchronizer)")
    p.add_argument("--gain", default="auto")
    p.add_argument("--block-size", type=int, default=262_144)
    p.add_argument("--jsonl", default="logs/live_sf8.jsonl")
    p.add_argument("--max-packets", type=int, default=0)
    p.add_argument("--preamble-repeats", type=int, default=6)
    p.add_argument("--preamble-snr-db", type=float, default=6.0)
    p.add_argument("--min-peak-isolation", type=float, default=1.35)
    p.add_argument("--valid-lockout-seconds", type=float, default=1.8)
    p.add_argument("--debug-candidates", action="store_true", help="also write debug candidates to JSONL")
    p.add_argument("--no-debug-candidates", action="store_true", help="compatibility alias; debug candidates are off by default")
    p.add_argument("--lilygo-hello-profile", action="store_true", help="backward-compatible alias for --expected-payload-ascii HELLO")
    p.add_argument("--expected-payload-len", type=int)
    p.add_argument("--expected-cr-index", type=int, help="LoRa CR index, where 1 means 4/5")
    p.add_argument("--expected-crc-enabled", type=int, choices=[0, 1])
    p.add_argument("--expected-payload-ascii", default="")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    channels = _parse_channels(args.channels)
    expected_payload_ascii = "HELLO" if args.lilygo_hello_profile else str(args.expected_payload_ascii or "").strip()
    expected_payload_len = 5 if args.lilygo_hello_profile else args.expected_payload_len
    expected_cr_index = 1 if args.lilygo_hello_profile else args.expected_cr_index
    expected_crc_enabled = 1 if args.lilygo_hello_profile else args.expected_crc_enabled
    if expected_payload_ascii and expected_payload_len is None:
        expected_payload_len = len(expected_payload_ascii.encode("utf-8"))
    if expected_payload_ascii and expected_cr_index is None:
        expected_cr_index = 1
    if expected_payload_ascii and expected_crc_enabled is None:
        expected_crc_enabled = 1
    cfg = ReceiverConfig(
        sf=args.sf,
        bw=args.bw,
        sample_rate=args.sample_rate,
        center_freq=args.center,
        channels=channels,
        oversample=args.oversample,
        gain=args.gain,
        block_size=args.block_size,
        jsonl_path=Path(args.jsonl),
        max_packets=args.max_packets,
        preamble_repeats=args.preamble_repeats,
        preamble_snr_db=args.preamble_snr_db,
        min_peak_isolation=args.min_peak_isolation,
        valid_lockout_seconds=args.valid_lockout_seconds,
        log_debug_candidates=bool(args.debug_candidates and not args.no_debug_candidates),
        expected_payload_len=expected_payload_len,
        expected_cr_index=expected_cr_index,
        expected_crc_enabled=expected_crc_enabled,
        expected_payload_ascii=expected_payload_ascii,
    )
    logger = JsonlLogger(cfg.jsonl_path)
    rx = LoRaReceiver(cfg, logger)
    source = make_source(args)
    channelizer = Channelizer(args.sample_rate, args.center, channels, args.bw, oversample=cfg.oversample)
    try:
        total = 0
        for block in source.blocks():
            # If the file is already a single LoRa channel at BW sample rate, skip DDC.
            if len(channels) == 1 and abs(args.sample_rate - args.bw) < 1.0:
                ch_outputs = [(channels[0], args.bw, remove_dc(block))]
            else:
                ch_outputs = [(o.center_hz, o.sample_rate, o.samples) for o in channelizer.process(block)]
            for ch_hz, fs, ch_samples in ch_outputs:
                total += rx.process_channel(ch_hz, ch_samples, fs)
                if cfg.max_packets and total >= cfg.max_packets:
                    return 0
            if args.source == "file" and len(block) < args.block_size:
                break
    finally:
        logger.close()
        if hasattr(source, "close"):
            source.close()
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
