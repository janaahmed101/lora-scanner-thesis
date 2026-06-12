from __future__ import annotations

"""Cyber-style PyQt GUI for the LoRaPHY scanner backend.

This is a real scanner controller, not a static dashboard:
- starts/stops live RTL-SDR or offline IQ replay
- runs DDC/channelizer + preamble detector + synchronizer + PHY decoder backend
- displays wideband/channel spectra, live channel activity, packet rows, raw bins, PHDR/payload status

Run with:
    python -m lora_scanner.gui
"""

import csv
import json
import math
import sys
import time
import traceback
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np

try:  # Prefer PyQt5 because it is usually easier to install on Windows scientific stacks.
    from PyQt5.QtCore import QObject, QThread, Qt, pyqtSignal, QRectF, QPointF
    from PyQt5.QtGui import QColor, QFont, QPainter, QPainterPath, QPen, QBrush, QLinearGradient
    from PyQt5.QtWidgets import (
        QApplication,
        QCheckBox,
        QComboBox,
        QFileDialog,
        QFrame,
        QGridLayout,
        QGroupBox,
        QHBoxLayout,
        QHeaderView,
        QLabel,
        QLineEdit,
        QMainWindow,
        QMessageBox,
        QPushButton,
        QSpinBox,
        QDoubleSpinBox,
        QPlainTextEdit,
        QSplitter,
        QTableWidget,
        QTableWidgetItem,
        QTabWidget,
        QVBoxLayout,
        QWidget,
    )
    PYQT_VERSION = 5
except Exception:  # pragma: no cover - depends on the user's GUI install
    from PyQt6.QtCore import QObject, QThread, Qt, pyqtSignal, QRectF, QPointF
    from PyQt6.QtGui import QColor, QFont, QPainter, QPainterPath, QPen, QBrush, QLinearGradient  # type: ignore[no-redef]
    from PyQt6.QtWidgets import (  # type: ignore[no-redef]
        QApplication,
        QCheckBox,
        QComboBox,
        QFileDialog,
        QFrame,
        QGridLayout,
        QGroupBox,
        QHBoxLayout,
        QHeaderView,
        QLabel,
        QLineEdit,
        QMainWindow,
        QMessageBox,
        QPushButton,
        QSpinBox,
        QDoubleSpinBox,
        QPlainTextEdit,
        QSplitter,
        QTableWidget,
        QTableWidgetItem,
        QTabWidget,
        QVBoxLayout,
        QWidget,
    )
    PYQT_VERSION = 6

from .channelizer import Channelizer, remove_dc
from .config import EU868_CHANNELS, ReceiverConfig
from .iq_file_source import IQFileSource
from .logger import JsonlLogger
from .receiver import LoRaReceiver
from .sdr_source import RtlSdrSource


if PYQT_VERSION == 5:
    ALIGN_CENTER = Qt.AlignCenter
    ALIGN_RIGHT = Qt.AlignRight
    HORIZONTAL = Qt.Horizontal
    RESIZE_STRETCH = QHeaderView.Stretch
    RESIZE_CONTENTS = QHeaderView.ResizeToContents
    ANTIALIAS = QPainter.Antialiasing
    NO_PEN = Qt.NoPen
else:  # pragma: no cover
    ALIGN_CENTER = Qt.AlignmentFlag.AlignCenter
    ALIGN_RIGHT = Qt.AlignmentFlag.AlignRight
    HORIZONTAL = Qt.Orientation.Horizontal
    RESIZE_STRETCH = QHeaderView.ResizeMode.Stretch
    RESIZE_CONTENTS = QHeaderView.ResizeMode.ResizeToContents
    ANTIALIAS = QPainter.RenderHint.Antialiasing
    NO_PEN = Qt.PenStyle.NoPen


CYAN = "#00e5ff"
GREEN = "#32ff7e"
PURPLE = "#b35cff"
ORANGE = "#ff7a33"
RED = "#ff3b5f"
BG = "#070d16"
PANEL = "#101b2b"
PANEL2 = "#142236"
GRID = "#12314a"
TEXT = "#d7f7ff"
MUTED = "#79a6c7"


def _safe_float(text: str) -> float:
    # Supports values such as 868.1e6 without exposing eval.
    return float(text.strip().replace("_", ""))


def _safe_gain(text: str) -> str | float:
    value = text.strip()
    if not value or value.lower() == "auto":
        return "auto"
    return _safe_float(value)


def _parse_channels(text: str) -> list[float]:
    t = text.strip().lower()
    if t in {"eu868", "all", "8", "8ch", "eight"}:
        return list(EU868_CHANNELS)
    if t in {"868.1", "868.1e6", "single", "one", "ch1"}:
        return [868.1e6]
    return [_safe_float(v) for v in text.replace(";", ",").split(",") if v.strip()]


def _fmt_freq(freq_hz: float) -> str:
    return f"{freq_hz / 1e6:.1f}"


def _fmt_freq_full(freq_hz: float) -> str:
    return f"{freq_hz / 1e6:.3f} MHz"


def _fmt_float(value: Any, digits: int = 1, suffix: str = "") -> str:
    try:
        if value is None or (isinstance(value, float) and not math.isfinite(value)):
            return "—"
        return f"{float(value):.{digits}f}{suffix}"
    except Exception:
        return "—"


def _short_list(values: Iterable[Any] | None, max_items: int = 12, sep: str = " ") -> str:
    if values is None:
        return ""
    vals = list(values)
    shown = vals[:max_items]
    body = sep.join(str(x) for x in shown)
    if len(vals) > max_items:
        body += f" … +{len(vals) - max_items}"
    return body


def _best_phdr_text(record: dict[str, Any]) -> str:
    phdr = record.get("phdr")
    if isinstance(phdr, dict):
        src = str(phdr.get("source", ""))
        tag = "ZIP" if "exact_zip" in src else "OK"
        if record.get("phdr_profile_rescue"):
            tag = "ZIP*"
        return f"{tag} L={phdr.get('payload_len')} CR=4/{phdr.get('cr_denom')} CRC={phdr.get('crc_enabled')}"
    legacy = record.get("legacy_zip_phdr")
    if isinstance(legacy, dict) and legacy.get("ok"):
        return f"ZIP L={legacy.get('payload_len')} CR=4/{legacy.get('cr_denom')} CRC={legacy.get('crc_enabled')}"
    dbg = record.get("header_debug") or {}
    status = dbg.get("selection_status", "")
    passes = record.get("phdr_pass_count", dbg.get("phdr_pass_count", ""))
    if passes == 0 or status == "none":
        return "NO PHDR"
    if status == "multiple":
        return f"AMBIG ({passes})"
    return str(status or "—")


def _payload_decode_record(record: dict[str, Any]) -> dict[str, Any]:
    for key in ("payload_decode", "clean_payload_decode", "legacy_zip_payload_decode"):
        payload_dec = record.get(key)
        if isinstance(payload_dec, dict):
            return payload_dec
    return {}


def _payload_decode_status(record: dict[str, Any]) -> str:
    expected = ""
    profile = record.get("expected_profile")
    if isinstance(profile, dict):
        expected = str(profile.get("payload_ascii", "")).strip().upper()
    payload_dec = _payload_decode_record(record)
    if payload_dec:
        asc = str(payload_dec.get("payload_ascii", "")).strip()
        ascii_matches_expected = bool(expected and asc.upper() == expected)
        crc_ok = bool(payload_dec.get("ok") or payload_dec.get("crc_ok"))
        if profile and crc_ok and (not expected or ascii_matches_expected) and _phdr_ok(record):
            return "OK"
        if not profile and crc_ok:
            return "OK"
        reason = str(payload_dec.get("failure_reason") or record.get("failure_reason") or "").lower()
        if "crc" in reason:
            return "CRC FAIL"
        if "short_payload" in reason:
            return "SHORT PAYLOAD"
        return str(payload_dec.get("failure_reason") or "PAYLOAD FAIL")
    payload = record.get("payload")
    if isinstance(payload, dict):
        return f"SYMS {len(payload.get('raw_bins') or [])}/{payload.get('expected_symbols')}"
    reason = str(record.get("failure_reason") or "")
    if "short_payload" in reason:
        return "SHORT PAYLOAD"
    return reason or "-"


def _decoded_ascii(record: dict[str, Any]) -> str:
    payload_dec = _payload_decode_record(record)
    text = str(payload_dec.get("payload_ascii", "")).strip() if payload_dec else ""
    return text or "-"


def _payload_crc_text(record: dict[str, Any]) -> str:
    payload_dec = _payload_decode_record(record)
    phdr = record.get("phdr")
    if isinstance(phdr, dict) and int(phdr.get("crc_enabled", 1) or 0) == 0:
        return "OFF"
    if not payload_dec:
        return "-"
    if payload_dec.get("crc_ok") or payload_dec.get("ok"):
        return "PASS"
    reason = str(payload_dec.get("failure_reason") or record.get("failure_reason") or "").lower()
    return "FAIL" if "crc" in reason or payload_dec else "-"


def _raw_bins_text(record: dict[str, Any], section: str, max_items: int) -> str:
    obj = record.get(section)
    raw = obj.get("raw_bins") if isinstance(obj, dict) else None
    if not raw and section == "payload":
        payload_dec = _payload_decode_record(record)
        raw = payload_dec.get("raw_bins") if payload_dec else None
    return _short_list(raw, max_items) or "-"


def _payload_corrected_bins_text(record: dict[str, Any], max_items: int = 9999) -> str:
    payload = record.get("payload")
    corrected = payload.get("corrected_bins") if isinstance(payload, dict) else None
    if corrected is None:
        payload_dec = _payload_decode_record(record)
        corrected = payload_dec.get("raw_bins") if payload_dec else None
    return _short_list(corrected, max_items) or "-"


def _payload_original_bins_text(record: dict[str, Any], max_items: int = 9999) -> str:
    payload_dec = _payload_decode_record(record)
    original = payload_dec.get("demod_raw_bins") if payload_dec else None
    if original is None:
        payload = record.get("payload")
        original = payload.get("raw_bins") if isinstance(payload, dict) else None
    return _short_list(original, max_items) or "-"


def _payload_decoder_source(record: dict[str, Any]) -> str:
    payload_dec = _payload_decode_record(record)
    source = str(payload_dec.get("source", "")).strip() if payload_dec else ""
    if source:
        return source
    if isinstance(record.get("clean_payload_decode"), dict):
        return "clean_payload_decoder_fallback"
    if isinstance(record.get("legacy_zip_payload_decode"), dict):
        return "exact_zip_payload_decoder"
    return "-"


def _header_offset_text(record: dict[str, Any]) -> str:
    header = record.get("header")
    if isinstance(header, dict) and header.get("timing_offset_samples") is not None:
        return str(header.get("timing_offset_samples"))
    return "-"


def _payload_offset_text(record: dict[str, Any]) -> str:
    payload = record.get("payload")
    if isinstance(payload, dict) and payload.get("timing_offset_samples") is not None:
        return str(payload.get("timing_offset_samples"))
    payload_dec = _payload_decode_record(record)
    if payload_dec.get("timing_offset_samples") is not None:
        return str(payload_dec.get("timing_offset_samples"))
    return "-"


def _payload_bin_offset_text(record: dict[str, Any]) -> str:
    payload = record.get("payload")
    if isinstance(payload, dict) and payload.get("bin_offset") is not None:
        return str(payload.get("bin_offset"))
    payload_dec = _payload_decode_record(record)
    if payload_dec.get("bin_offset") is not None:
        return str(payload_dec.get("bin_offset"))
    return "-"


def _payload_residual_cfo_text(record: dict[str, Any], key: str, digits: int) -> str:
    payload = record.get("payload")
    if isinstance(payload, dict) and payload.get(key) is not None:
        return _fmt_float(payload.get(key), digits)
    payload_dec = _payload_decode_record(record)
    if payload_dec.get(key) is not None:
        return _fmt_float(payload_dec.get(key), digits)
    return "-"


def _payload_failure_class_text(record: dict[str, Any]) -> str:
    payload_dec = _payload_decode_record(record)
    if payload_dec.get("failure_class"):
        return str(payload_dec.get("failure_class"))
    payload = record.get("payload")
    if isinstance(payload, dict) and payload.get("failure_class"):
        return str(payload.get("failure_class"))
    return str(record.get("failure_class", ""))


def _payload_sweep_summary(record: dict[str, Any], max_items: int = 14) -> str:
    attempts = record.get("payload_timing_sweep")
    if not isinstance(attempts, list) or not attempts:
        return "-"
    parts: list[str] = []
    for attempt in attempts[:max_items]:
        if not isinstance(attempt, dict):
            continue
        off = attempt.get("offset_samples")
        bin_off = int(attempt.get("bin_offset") or 0)
        label = f"{off}/{bin_off:+d}" if bin_off else str(off)
        if attempt.get("ok") or attempt.get("crc_ok") or attempt.get("ascii_matches_expected"):
            status = "OK"
        elif str(attempt.get("failure_reason", "")).startswith("short_payload"):
            status = "SHORT"
        else:
            status = "FAIL"
        parts.append(f"{label}:{status}")
    if len(attempts) > max_items:
        parts.append(f"+{len(attempts) - max_items}")
    return " ".join(parts) or "-"


def _payload_text(record: dict[str, Any]) -> str:
    status = _payload_decode_status(record)
    ascii_text = _decoded_ascii(record)
    if status == "OK":
        return f"OK {ascii_text}" if ascii_text != "-" else "OK"
    return status or record.get("failure_reason", "") or "-"


def _payload_decode_ok(record: dict[str, Any]) -> bool:
    if "payload_valid" in record:
        return bool(record.get("payload_valid"))
    expected = ""
    profile = record.get("expected_profile")
    if isinstance(profile, dict):
        expected = str(profile.get("payload_ascii", "")).strip().upper()
    for key in ("payload_decode", "clean_payload_decode", "legacy_zip_payload_decode"):
        payload_dec = record.get(key)
        if not isinstance(payload_dec, dict):
            continue
        ascii_text = str(payload_dec.get("payload_ascii", "")).strip().upper()
        crc_ok = bool(payload_dec.get("ok") or payload_dec.get("crc_ok"))
        if profile and crc_ok and (not expected or ascii_text == expected) and _phdr_ok(record):
            return True
        if not profile and crc_ok:
            return True
    return False


def _phdr_ok(record: dict[str, Any]) -> bool:
    for key in ("phdr", "legacy_zip_phdr"):
        phdr = record.get(key)
        if isinstance(phdr, dict) and phdr.get("ok"):
            return True
    return False


def _is_real_packet(record: dict[str, Any]) -> bool:
    candidate_class = str(record.get("candidate_class", ""))
    if candidate_class in {"full_payload_packet", "phdr_only_packet", "packet"}:
        return True
    if candidate_class == "debug_candidate":
        return False
    if record.get("phdr_valid"):
        return True
    if "packet_accepted" in record:
        return bool(record.get("packet_accepted"))
    return _payload_decode_ok(record) or _phdr_ok(record)


def _candidate_reason(record: dict[str, Any]) -> str:
    candidate_class = str(record.get("candidate_class", ""))
    if candidate_class == "full_payload_packet":
        return "full_payload_packet"
    if candidate_class == "phdr_only_packet":
        return "phdr_only_packet"
    if record.get("candidate_reason"):
        return str(record.get("candidate_reason"))
    if _payload_decode_ok(record):
        return "payload_crc_ok"
    if _phdr_ok(record):
        return "phdr_valid"
    return str(record.get("failure_reason") or _best_phdr_text(record) or "debug_candidate")


def _sf_badge_color(sf: Any) -> str:
    palette = {7: CYAN, 8: GREEN, 9: PURPLE, 10: "#d569ff", 11: "#ffb833", 12: ORANGE}
    try:
        return palette.get(int(sf), CYAN)
    except Exception:
        return CYAN


def _packet_class_text(record: dict[str, Any]) -> str:
    candidate_class = str(record.get("candidate_class", ""))
    if candidate_class == "full_payload_packet" or _payload_decode_ok(record):
        return "FULL"
    if candidate_class == "phdr_only_packet" or _phdr_ok(record):
        return "PHDR ONLY"
    return "DEBUG"


CSV_HEADERS = [
    "#",
    "Time",
    "Ch",
    "Freq [MHz]",
    "SF",
    "SNR [dB]",
    "Rel Pwr [dB]",
    "Det Pwr [dB]",
    "CFO est [Hz]",
    "Preamble",
    "SFD raw bins",
    "PHDR",
    "Hdr off [samp]",
    "Payload timing offset",
    "Payload bin offset",
    "Payload residual CFO [bins]",
    "Payload residual CFO [Hz]",
    "Header raw bins",
    "Payload raw bins",
    "Payload corrected bins",
    "Payload original bins",
    "Decode status",
    "Decoded ASCII",
    "Payload CRC",
    "Decoder source",
    "Packet class",
    "Failure reason",
    "Failure class",
]


def _channel_label(record: dict[str, Any]) -> str:
    try:
        closest = min(range(len(EU868_CHANNELS)), key=lambda i: abs(EU868_CHANNELS[i] - float(record.get("channel_hz", 0.0))))
        return f"CH{closest + 1}"
    except Exception:
        return "CH?"


def _csv_row_from_record(index: int, record: dict[str, Any]) -> list[str]:
    sync = record.get("sync") if isinstance(record.get("sync"), dict) else {}
    det = record.get("detector") if isinstance(record.get("detector"), dict) else {}
    ch_mhz = float(record.get("channel_hz", 0.0)) / 1e6
    snr = sync.get("snr_db", det.get("snr_db"))
    rssi = sync.get("rssi_db", det.get("rssi_db"))
    return [
        str(index),
        str(record.get("timestamp", ""))[11:23],
        _channel_label(record),
        f"{ch_mhz:.3f}",
        f"SF{record.get('sf', '')}",
        _fmt_float(snr, 2),
        _fmt_float(rssi, 2),
        _fmt_float(det.get("rssi_db"), 2),
        _fmt_float(sync.get("cfo_hz"), 2),
        _short_list(det.get("preamble_bins") or sync.get("preamble_bins"), 9999),
        _short_list(sync.get("sfd_bins"), 9999),
        _best_phdr_text(record),
        _header_offset_text(record),
        _payload_offset_text(record),
        _payload_bin_offset_text(record),
        _payload_residual_cfo_text(record, "residual_cfo_bins", 2),
        _payload_residual_cfo_text(record, "residual_cfo_hz", 2),
        _raw_bins_text(record, "header", 9999),
        _raw_bins_text(record, "payload", 9999),
        _payload_corrected_bins_text(record),
        _payload_original_bins_text(record),
        _payload_decode_status(record),
        _decoded_ascii(record),
        _payload_crc_text(record),
        _payload_decoder_source(record),
        _packet_class_text(record),
        str(record.get("failure_reason", "")),
        _payload_failure_class_text(record),
    ]


@dataclass(slots=True)
class GuiRunConfig:
    source: str
    iq_path: str
    center_hz: float
    channels: list[float]
    sf: int
    bw_hz: float
    sample_rate: float
    gain: str | float
    block_size: int
    jsonl_path: str
    max_packets: int
    preamble_repeats: int
    preamble_snr_db: float
    min_peak_isolation: float
    valid_lockout_seconds: float
    log_debug_candidates: bool
    expected_payload_len: int | None
    expected_cr_index: int | None
    expected_crc_enabled: int | None
    expected_payload_ascii: str
    show_console_json: bool


class Card(QFrame):
    def __init__(self, value: str, label: str):
        super().__init__()
        self.setObjectName("metricCard")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 10, 12, 10)
        self.value = QLabel(value)
        self.value.setAlignment(ALIGN_CENTER)
        self.value.setObjectName("metricValue")
        self.label = QLabel(label)
        self.label.setAlignment(ALIGN_CENTER)
        self.label.setObjectName("metricLabel")
        layout.addWidget(self.value)
        layout.addWidget(self.label)

    def set_value(self, text: str) -> None:
        self.value.setText(text)


class SpectrumWidget(QWidget):
    """Small dependency-free spectrum plot with the dark neon look from the reference GUI."""

    def __init__(self, title: str, subtitle: str = ""):
        super().__init__()
        self.title = title
        self.subtitle = subtitle
        self.freqs: list[float] = []
        self.powers: list[float] = []
        self.channels_mhz: list[float] = []
        self.active_mhz: float | None = None
        self.no_signal_text = "No signal detected"
        self.setMinimumHeight(210)

    def set_data(self, freqs: Iterable[float], powers: Iterable[float], channels_mhz: Iterable[float] | None = None, active_mhz: float | None = None, subtitle: str | None = None) -> None:
        self.freqs = list(freqs)
        self.powers = list(powers)
        if channels_mhz is not None:
            self.channels_mhz = list(channels_mhz)
        self.active_mhz = active_mhz
        if subtitle is not None:
            self.subtitle = subtitle
        self.update()

    def paintEvent(self, event: Any) -> None:  # pragma: no cover - visual code
        del event
        painter = QPainter(self)
        painter.setRenderHint(ANTIALIAS)
        rect = self.rect().adjusted(10, 10, -10, -10)

        # Panel background
        grad = QLinearGradient(rect.topLeft(), rect.bottomRight())
        grad.setColorAt(0.0, QColor("#0d1928"))
        grad.setColorAt(1.0, QColor("#08111d"))
        painter.setBrush(QBrush(grad))
        painter.setPen(QPen(QColor("#17496a"), 1))
        painter.drawRoundedRect(rect, 8, 8)

        # Header
        painter.setFont(QFont("Consolas", 10, QFont.Bold))
        painter.setPen(QPen(QColor(CYAN)))
        painter.drawText(rect.adjusted(14, 14, -14, -14), int(ALIGN_RIGHT), self.subtitle)
        painter.drawText(rect.adjusted(14, 14, -14, -14), int(0), "▪  " + self.title)

        plot = rect.adjusted(18, 42, -18, -24)

        # Grid
        painter.setPen(QPen(QColor(GRID), 1))
        for i in range(1, 8):
            x = plot.left() + plot.width() * i / 8.0
            painter.drawLine(int(x), plot.top(), int(x), plot.bottom())
        for i in range(1, 6):
            y = plot.top() + plot.height() * i / 6.0
            painter.drawLine(plot.left(), int(y), plot.right(), int(y))

        if len(self.freqs) < 2 or len(self.powers) < 2:
            painter.setFont(QFont("Consolas", 11, QFont.Bold))
            painter.setPen(QPen(QColor("#0b6570")))
            painter.drawText(plot, int(ALIGN_CENTER), self.no_signal_text)
            return

        fmin, fmax = min(self.freqs), max(self.freqs)
        pmin, pmax = min(self.powers), max(self.powers)
        if pmax - pmin < 1e-6:
            pmax = pmin + 1.0

        def map_x(f: float) -> float:
            return plot.left() + (f - fmin) / (fmax - fmin) * plot.width()

        def map_y(p: float) -> float:
            return plot.bottom() - (p - pmin) / (pmax - pmin) * plot.height()

        # Channel markers
        if self.channels_mhz:
            for idx, ch in enumerate(self.channels_mhz, start=1):
                if fmin <= ch <= fmax:
                    x = map_x(ch)
                    if self.active_mhz is not None and abs(ch - self.active_mhz) < 0.02:
                        painter.setBrush(QBrush(QColor(255, 122, 51, 40)))
                        painter.setPen(NO_PEN)
                        painter.drawRect(QRectF(x - 24, plot.top(), 48, plot.height()))
                        painter.setPen(QPen(QColor(ORANGE), 1, Qt.DashLine if PYQT_VERSION == 5 else Qt.PenStyle.DashLine))
                    else:
                        painter.setPen(QPen(QColor("#0d394d"), 1, Qt.DashLine if PYQT_VERSION == 5 else Qt.PenStyle.DashLine))
                    painter.drawLine(QPointF(x, plot.top()), QPointF(x, plot.bottom()))
                    painter.setFont(QFont("Consolas", 9, QFont.Bold))
                    painter.setPen(QPen(QColor(ORANGE if self.active_mhz is not None and abs(ch - self.active_mhz) < 0.02 else "#1caed2")))
                    painter.drawText(QRectF(x - 24, plot.bottom() + 4, 48, 18), int(ALIGN_CENTER), f"CH{idx}")

        # Spectrum line
        path = QPainterPath()
        path.moveTo(map_x(self.freqs[0]), map_y(self.powers[0]))
        for f, p in zip(self.freqs[1:], self.powers[1:]):
            path.lineTo(map_x(f), map_y(p))
        painter.setPen(QPen(QColor(CYAN), 1.4))
        painter.drawPath(path)

        # Glow/peak-hold-ish second line
        if len(self.powers) > 5:
            smooth = np.convolve(np.asarray(self.powers, dtype=float), np.ones(9) / 9.0, mode="same")
            path2 = QPainterPath()
            path2.moveTo(map_x(self.freqs[0]), map_y(float(smooth[0])))
            for f, p in zip(self.freqs[1:], smooth[1:]):
                path2.lineTo(map_x(float(f)), map_y(float(p)))
            painter.setPen(QPen(QColor(50, 255, 126, 110), 1.1))
            painter.drawPath(path2)

        # Axis labels
        painter.setFont(QFont("Consolas", 8))
        painter.setPen(QPen(QColor("#426986")))
        painter.drawText(plot.left(), plot.bottom() + 19, f"{fmin:.1f}")
        painter.drawText(plot.right() - 48, plot.bottom() + 19, f"{fmax:.1f} MHz")
        painter.drawText(plot.left(), plot.top() + 12, f"{pmax:.0f}")
        painter.drawText(plot.left(), plot.bottom() - 4, f"{pmin:.0f}")


class EmittingJsonlLogger(JsonlLogger):
    def __init__(self, path: str | Path | None, packet_signal: Any, console_signal: Any, echo_console_json: bool):
        super().__init__(path)
        self.packet_signal = packet_signal
        self.console_signal = console_signal
        self.echo_console_json = bool(echo_console_json)

    def write(self, record: dict[str, Any]) -> None:
        line = json.dumps(record, ensure_ascii=False, sort_keys=True, default=str)
        if self._fh:
            self._fh.write(line + "\n")
            self._fh.flush()
        self.packet_signal.emit(record)
        if self.echo_console_json:
            self.console_signal.emit(line)


class ScannerWorker(QObject):
    packet = pyqtSignal(object)
    status = pyqtSignal(str)
    error = pyqtSignal(str)
    finished = pyqtSignal()
    channelStats = pyqtSignal(object)
    widebandSpectrum = pyqtSignal(object)
    channelSpectrum = pyqtSignal(object)
    console = pyqtSignal(str)

    def __init__(self, cfg: GuiRunConfig):
        super().__init__()
        self.cfg = cfg
        self._stop = False
        self._source: Any | None = None

    def stop(self) -> None:
        self._stop = True

    def _spectrum_payload(self, samples: np.ndarray, sample_rate: float, center_hz: float, nfft: int = 4096) -> dict[str, Any]:
        if len(samples) < 32:
            return {"freqs_mhz": [], "power_db": []}
        n = min(int(nfft), len(samples))
        x = remove_dc(np.asarray(samples[:n], dtype=np.complex64))
        win = np.hanning(n).astype(np.float32)
        spec = np.fft.fftshift(np.fft.fft(x * win))
        power = 20.0 * np.log10(np.abs(spec) + 1e-9)
        freq = center_hz + np.fft.fftshift(np.fft.fftfreq(n, d=1.0 / sample_rate))
        # Decimate display payload for signal transport/UI performance.
        step = max(1, len(power) // 512)
        return {
            "freqs_mhz": (freq[::step] / 1e6).astype(float).tolist(),
            "power_db": power[::step].astype(float).tolist(),
        }

    def _emit_channel_stats(self, ch_outputs: list[tuple[float, float, np.ndarray]]) -> None:
        stats = []
        best_peak = -1e9
        best_spec: dict[str, Any] | None = None
        best_ch: float | None = None
        for ch_hz, fs, samples in ch_outputs:
            if len(samples) == 0:
                power_db = -999.0
                peak_db = -999.0
            else:
                power_db = float(10.0 * np.log10(max(np.mean(np.abs(samples) ** 2), 1e-15)))
                spec_payload = self._spectrum_payload(samples, fs, ch_hz, nfft=2048)
                peak_db = float(max(spec_payload.get("power_db", [-999.0]) or [-999.0]))
                if peak_db > best_peak:
                    best_peak = peak_db
                    best_spec = spec_payload
                    best_ch = ch_hz
            stats.append({"channel_hz": float(ch_hz), "fs": float(fs), "power_db": power_db, "peak_db": peak_db, "samples": int(len(samples))})
        self.channelStats.emit(stats)
        if best_spec is not None and best_ch is not None:
            best_spec["channel_hz"] = best_ch
            best_spec["subtitle"] = f"CH {best_ch/1e6:.1f} MHz | SF{self.cfg.sf} | BW {self.cfg.bw_hz/1e3:.0f} kHz"
            self.channelSpectrum.emit(best_spec)

    def run(self) -> None:
        source = None
        logger: EmittingJsonlLogger | None = None
        try:
            rcfg = ReceiverConfig(
                sf=self.cfg.sf,
                bw=self.cfg.bw_hz,
                sample_rate=self.cfg.sample_rate,
                center_freq=self.cfg.center_hz,
                channels=self.cfg.channels,
                oversample=int(getattr(self.cfg, "oversample", 4)),
                gain=self.cfg.gain,
                block_size=self.cfg.block_size,
                jsonl_path=Path(self.cfg.jsonl_path) if self.cfg.jsonl_path else None,
                max_packets=self.cfg.max_packets,
                preamble_repeats=self.cfg.preamble_repeats,
                preamble_snr_db=self.cfg.preamble_snr_db,
                min_peak_isolation=self.cfg.min_peak_isolation,
                valid_lockout_seconds=self.cfg.valid_lockout_seconds,
                log_debug_candidates=self.cfg.log_debug_candidates,
                expected_payload_len=self.cfg.expected_payload_len,
                expected_cr_index=self.cfg.expected_cr_index,
                expected_crc_enabled=self.cfg.expected_crc_enabled,
                expected_payload_ascii=self.cfg.expected_payload_ascii,
            )
            logger = EmittingJsonlLogger(rcfg.jsonl_path, self.packet, self.console, self.cfg.show_console_json)
            rx = LoRaReceiver(rcfg, logger)

            if self.cfg.source == "file":
                if not self.cfg.iq_path:
                    raise RuntimeError("Offline file mode selected, but no IQ file path was provided.")
                source = IQFileSource(self.cfg.iq_path, self.cfg.block_size)
                self._source = source
                self.status.emit(f"REPLAYING - {Path(self.cfg.iq_path).name}")
            else:
                source = RtlSdrSource(self.cfg.center_hz, self.cfg.sample_rate, self.cfg.gain, self.cfg.block_size)
                self._source = source
                self.status.emit("SCANNING - RTL-SDR live IQ")

            channelizer = Channelizer(self.cfg.sample_rate, self.cfg.center_hz, self.cfg.channels, self.cfg.bw_hz, oversample=rcfg.oversample)
            total_packets = 0
            block_counter = 0
            for block in source.blocks():
                if self._stop:
                    self.status.emit("PAUSED - scanner stopped by user")
                    break

                if block_counter % 2 == 0:
                    wb = self._spectrum_payload(block, self.cfg.sample_rate, self.cfg.center_hz, nfft=8192)
                    wb["channels_mhz"] = [float(c / 1e6) for c in self.cfg.channels]
                    wb["subtitle"] = f"{min(self.cfg.channels)/1e6:.1f} - {max(self.cfg.channels)/1e6:.1f} MHz | RBW approx"
                    self.widebandSpectrum.emit(wb)

                if len(self.cfg.channels) == 1 and abs(self.cfg.sample_rate - self.cfg.bw_hz) < 1.0:
                    ch_outputs = [(self.cfg.channels[0], self.cfg.bw_hz, remove_dc(block))]
                else:
                    ch_outputs = [(o.center_hz, o.sample_rate, o.samples) for o in channelizer.process(block)]

                if block_counter % 2 == 0:
                    self._emit_channel_stats(ch_outputs)

                for ch_hz, fs, ch_samples in ch_outputs:
                    if self._stop:
                        break
                    total_packets += rx.process_channel(ch_hz, ch_samples, fs)
                    if rcfg.max_packets and total_packets >= rcfg.max_packets:
                        self.status.emit(f"REACHED MAX PACKETS - {total_packets}")
                        self._stop = True
                        break

                block_counter += 1
                if self.cfg.source == "file" and len(block) < self.cfg.block_size:
                    self.status.emit("REPLAY COMPLETE")
                    break
        except Exception:
            if self._stop:
                self.status.emit("PAUSED - scanner stopped by user")
            else:
                self.error.emit(traceback.format_exc())
        finally:
            if logger is not None:
                try:
                    logger.close()
                except Exception:
                    pass
            close_source = self._source or source
            if close_source is not None and hasattr(close_source, "close"):
                try:
                    close_source.close()
                except Exception:
                    pass
            self._source = None
            self.finished.emit()


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("LoRaWAN SDR Scanner — EU868 LoRaPHY")
        self.resize(1780, 940)
        self.worker: ScannerWorker | None = None
        self.thread: QThread | None = None
        self.packet_count = 0
        self.full_packet_count = 0
        self.debug_count = 0
        self.channel_rows: dict[float, int] = {}
        self.packet_records: list[dict[str, Any]] = []
        self.debug_records: list[dict[str, Any]] = []
        self.run_started_at: float | None = None
        self.snr_values: list[float] = []
        self.rssi_values: list[float] = []
        self.last_sf = "—"
        self.last_active_channel_mhz: float | None = None
        self._build_ui()
        self._apply_style()
        self._set_running(False)
        self._source_changed(self.source_combo.currentText())
        self._seed_empty_spectra()

    def _apply_style(self) -> None:
        self.setStyleSheet(f"""
            QMainWindow, QWidget {{ background: {BG}; color: {TEXT}; font-family: 'Consolas', 'Cascadia Mono', monospace; }}
            QLabel {{ color: {TEXT}; }}
            #topBar, #packetPanel, #controlPanel, QGroupBox {{
                background: {PANEL}; border: 1px solid #17496a; border-radius: 8px;
            }}
            #brand {{ color: {CYAN}; font-size: 17px; font-weight: 900; letter-spacing: 7px; }}
            #statusPill {{ color: {GREEN}; font-weight: 800; padding: 7px 14px; background: #071b18; border: 1px solid #155b4c; border-radius: 8px; }}
            #chip {{ color: {CYAN}; padding: 7px 14px; border: 1px solid #176d92; border-radius: 6px; background: #102236; }}
            #metricCard {{ background: #19283b; border: 1px solid #193b5a; border-radius: 7px; }}
            #metricValue {{ color: {GREEN}; font-size: 20px; font-weight: 900; }}
            #metricLabel {{ color: {MUTED}; font-size: 12px; letter-spacing: 1px; }}
            QGroupBox {{ margin-top: 12px; padding: 12px; color: {CYAN}; font-weight: 800; }}
            QGroupBox::title {{ subcontrol-origin: margin; left: 12px; padding: 0 8px; }}
            QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox {{
                background: #091423; color: {TEXT}; border: 1px solid #17496a; border-radius: 5px; padding: 6px;
                selection-background-color: #115e78;
            }}
            QPushButton {{ background: #0b1b2d; color: {CYAN}; border: 1px solid #0bbfe6; border-radius: 6px; padding: 8px 14px; font-weight: 800; }}
            QPushButton:hover {{ background: #0d2a42; }}
            QPushButton:disabled {{ color: #48677a; border-color: #244356; }}
            QPushButton#danger {{ color: {RED}; border-color: {RED}; }}
            QPushButton#start {{ color: {GREEN}; border-color: {GREEN}; }}
            QTableWidget {{ background: #0b1421; alternate-background-color: #101b2b; gridline-color: #15364f; color: {TEXT}; border: 1px solid #17496a; border-radius: 4px; }}
            QHeaderView::section {{ background: #101b2b; color: {MUTED}; border: 1px solid #15364f; padding: 7px; font-weight: 800; }}
            QTableWidget::item:selected {{ background: #133c53; color: #ffffff; }}
            QPlainTextEdit {{ background: #08111d; color: {TEXT}; border: 1px solid #17496a; border-radius: 6px; padding: 8px; }}
            QTabWidget::pane {{ border: 1px solid #17496a; border-radius: 7px; }}
            QTabBar::tab {{ background: #0b1b2d; color: {MUTED}; padding: 8px 16px; border: 1px solid #17496a; }}
            QTabBar::tab:selected {{ color: {CYAN}; background: #12243a; }}
            QCheckBox {{ color: {MUTED}; }}
        """)

    def _build_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(8)

        root.addWidget(self._build_top_bar())

        top_split = QSplitter(HORIZONTAL)
        self.wideband_plot = SpectrumWidget("SDR WIDEBAND SPECTRUM", "868.0 - 869.9 MHz | 1.5 MSPS")
        self.channel_plot = SpectrumWidget("CHANNEL SPECTRUM", "CH — | SF8 | BW 125 kHz")
        self.channel_plot.no_signal_text = "No signal detected"
        top_split.addWidget(self.wideband_plot)
        top_split.addWidget(self.channel_plot)
        top_split.setSizes([880, 880])
        root.addWidget(top_split, 2)

        root.addWidget(self._build_packet_panel(), 4)
        root.addWidget(self._build_control_panel(), 1)

    def _build_top_bar(self) -> QWidget:
        bar = QFrame()
        bar.setObjectName("topBar")
        layout = QHBoxLayout(bar)
        layout.setContentsMargins(18, 10, 18, 10)
        brand = QLabel("◇  LORAWAN SDR SCANNER")
        brand.setObjectName("brand")
        self.status_pill = QLabel("● IDLE — EU868")
        self.status_pill.setObjectName("statusPill")
        self.telemetry_chip = QLabel("2.0 MSPS  |  8 CH  |  125 kHz BW")
        self.telemetry_chip.setObjectName("chip")
        self.start_btn = QPushButton("■ START")
        self.start_btn.setObjectName("start")
        self.stop_btn = QPushButton("▣ PAUSE")
        self.clear_btn = QPushButton("× CLEAR")
        self.clear_btn.setObjectName("danger")
        self.csv_btn = QPushButton("↓ CSV")
        self.start_btn.clicked.connect(self.start_scanner)
        self.stop_btn.clicked.connect(self.stop_scanner)
        self.clear_btn.clicked.connect(self.clear_tables)
        self.csv_btn.clicked.connect(self.export_csv)
        layout.addWidget(brand, 2)
        layout.addStretch(1)
        layout.addWidget(self.status_pill)
        layout.addStretch(1)
        layout.addWidget(self.telemetry_chip)
        layout.addStretch(1)
        layout.addWidget(self.start_btn)
        layout.addWidget(self.stop_btn)
        layout.addWidget(self.clear_btn)
        layout.addWidget(self.csv_btn)
        return bar

    def _build_packet_panel(self) -> QWidget:
        panel = QFrame()
        panel.setObjectName("packetPanel")
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(10, 10, 10, 10)
        header = QHBoxLayout()
        title = QLabel("▪  PACKET DECODER / LOGGER")
        title.setStyleSheet(f"color: {CYAN}; font-weight: 900; font-size: 14px; letter-spacing: 3px;")
        self.packet_hint = QLabel("0 PHDR packets  |  0 full decoded  |  0 debug candidates  |  Payload pass 0.0%")
        self.packet_hint.setStyleSheet(f"color: {MUTED};")
        header.addWidget(title)
        header.addStretch(1)
        header.addWidget(self.packet_hint)
        layout.addLayout(header)

        cards = QHBoxLayout()
        self.card_phdr = Card("0", "PHDR PACKETS")
        self.card_full = Card("0", "FULL DECODED")
        self.card_debug = Card("0", "DEBUG CANDIDATES")
        self.card_pass = Card("0.0%", "PAYLOAD PASS %")
        self.card_snr = Card("—", "AVG SNR")
        for card in [self.card_phdr, self.card_full, self.card_debug, self.card_pass, self.card_snr]:
            cards.addWidget(card)
        layout.addLayout(cards)

        self.packet_table = QTableWidget(0, len(CSV_HEADERS))
        self.packet_table.setAlternatingRowColors(True)
        self.packet_table.setHorizontalHeaderLabels(CSV_HEADERS)
        self.packet_table.horizontalHeader().setSectionResizeMode(RESIZE_CONTENTS)
        self.packet_table.horizontalHeader().setStretchLastSection(True)
        self.packet_table.verticalHeader().setVisible(False)
        self.packet_table.itemSelectionChanged.connect(self._selected_packet_changed)
        layout.addWidget(self.packet_table, 1)

        self._hidden_packet_panels = [
            self._build_detail_panel(),
            self._build_debug_table(),
            self._build_channel_table(),
            self._build_console(),
        ]
        for hidden_panel in self._hidden_packet_panels:
            hidden_panel.setVisible(False)
        return panel

    def _build_control_panel(self) -> QWidget:
        box = QGroupBox("Scanner controls")
        box.setObjectName("controlPanel")
        layout = QGridLayout(box)
        layout.setHorizontalSpacing(8)
        layout.setVerticalSpacing(6)

        self.source_combo = QComboBox()
        self.source_combo.addItems(["rtlsdr", "file"])
        self.source_combo.currentTextChanged.connect(self._source_changed)
        self.iq_edit = QLineEdit("")
        self.iq_edit.setPlaceholderText("path/to/capture.npz, .npy, .dat, .cf32")
        browse_btn = QPushButton("Browse IQ")
        browse_btn.clicked.connect(self._browse_iq)
        self.center_edit = QLineEdit("868.8e6")
        self.channels_combo = QComboBox()
        self.channels_combo.setEditable(True)
        self.channels_combo.addItems([
            "eu868",
            "868.1e6",
            "868.3e6",
            "868.5e6",
            "868.7e6",
            "868.9e6",
            "869.1e6",
            "869.3e6",
            "869.5e6",
            "868.1e6,868.3e6,868.5e6,868.7e6,868.9e6,869.1e6,869.3e6,869.5e6",
        ])
        self.sf_spin = QSpinBox(); self.sf_spin.setRange(6, 12); self.sf_spin.setValue(8)
        self.bw_edit = QLineEdit("125e3")
        self.sample_rate_edit = QLineEdit("2.0e6")
        self.gain_edit = QLineEdit("auto")
        self.block_spin = QSpinBox(); self.block_spin.setRange(16384, 4_194_304); self.block_spin.setSingleStep(16384); self.block_spin.setValue(262144)
        self.jsonl_edit = QLineEdit("logs/gui_live_sf8.jsonl")
        self.max_packets_spin = QSpinBox(); self.max_packets_spin.setRange(0, 100000); self.max_packets_spin.setValue(0)
        self.repeats_spin = QSpinBox(); self.repeats_spin.setRange(4, 16); self.repeats_spin.setValue(6)
        self.snr_spin = QDoubleSpinBox(); self.snr_spin.setRange(-20.0, 80.0); self.snr_spin.setDecimals(1); self.snr_spin.setValue(6.0)
        self.iso_spin = QDoubleSpinBox(); self.iso_spin.setRange(1.0, 100.0); self.iso_spin.setDecimals(2); self.iso_spin.setSingleStep(0.05); self.iso_spin.setValue(1.35)
        self.lockout_spin = QDoubleSpinBox(); self.lockout_spin.setRange(0.0, 10.0); self.lockout_spin.setDecimals(1); self.lockout_spin.setSingleStep(0.1); self.lockout_spin.setValue(1.8)
        self.candidate_log_check = QCheckBox("Debug candidates"); self.candidate_log_check.setChecked(False)
        self.lilygo_profile_check = QCheckBox("Expected payload profile"); self.lilygo_profile_check.setChecked(True)
        self.expected_payload_edit = QLineEdit("JANA")
        self.expected_payload_edit.setPlaceholderText("JANA")
        self.console_json_check = QCheckBox("Echo full JSON")

        widgets = [
            ("Source", self.source_combo), ("IQ replay", self.iq_edit), ("", browse_btn), ("Center Hz", self.center_edit),
            ("Channels", self.channels_combo), ("SF", self.sf_spin), ("BW Hz", self.bw_edit), ("Sample rate", self.sample_rate_edit),
            ("Gain", self.gain_edit), ("Block", self.block_spin), ("JSONL", self.jsonl_edit), ("Max pkts", self.max_packets_spin),
            ("Preamble N", self.repeats_spin), ("SNR gate", self.snr_spin), ("Isolation", self.iso_spin), ("Lockout s", self.lockout_spin),
            ("Profile", self.lilygo_profile_check), ("Payload", self.expected_payload_edit), ("Debug log", self.candidate_log_check), ("Debug", self.console_json_check),
        ]
        for i, (label, widget) in enumerate(widgets):
            col = (i % 4) * 2
            row = i // 4
            if label:
                lab = QLabel(label)
                lab.setStyleSheet(f"color: {MUTED};")
                layout.addWidget(lab, row, col)
            layout.addWidget(widget, row, col + 1)
        self.status_label = QLabel("Ready.")
        self.status_label.setStyleSheet(f"color: {MUTED};")
        layout.addWidget(self.status_label, 5, 0, 1, 8)
        return box

    def _build_debug_table(self) -> QWidget:
        w = QWidget()
        layout = QVBoxLayout(w)
        self.debug_table = QTableWidget(0, 9)
        self.debug_table.setAlternatingRowColors(True)
        self.debug_table.setHorizontalHeaderLabels([
            "#", "Time", "Freq [MHz]", "SF", "SNR [dB]", "Rel Pwr [dB]", "Preamble", "SFD raw bins", "Reason",
        ])
        self.debug_table.horizontalHeader().setSectionResizeMode(RESIZE_CONTENTS)
        self.debug_table.horizontalHeader().setStretchLastSection(True)
        self.debug_table.verticalHeader().setVisible(False)
        self.debug_table.itemSelectionChanged.connect(self._selected_debug_changed)
        layout.addWidget(self.debug_table)
        return w

    def _build_channel_table(self) -> QWidget:
        w = QWidget()
        layout = QVBoxLayout(w)
        self.channel_table = QTableWidget(0, 5)
        self.channel_table.setHorizontalHeaderLabels(["Channel", "Fs", "Power dB", "Peak dB", "Samples"])
        self.channel_table.horizontalHeader().setSectionResizeMode(RESIZE_STRETCH)
        self.channel_table.verticalHeader().setVisible(False)
        layout.addWidget(self.channel_table)
        hint = QLabel("8-channel test: center=868.8e6, channels=eu868, sample rate=2.0e6. Single-channel fallback: center=868.2e6, channels=868.1e6, sample rate=1.5e6.")
        hint.setWordWrap(True)
        hint.setStyleSheet(f"color: {MUTED};")
        layout.addWidget(hint)
        return w

    def _build_detail_panel(self) -> QWidget:
        w = QWidget()
        layout = QGridLayout(w)
        self.summary_text = QPlainTextEdit(); self.summary_text.setReadOnly(True)
        self.raw_json_text = QPlainTextEdit(); self.raw_json_text.setReadOnly(True)
        layout.addWidget(QLabel("Human-readable packet summary"), 0, 0)
        layout.addWidget(QLabel("Raw JSONL record"), 0, 1)
        layout.addWidget(self.summary_text, 1, 0)
        layout.addWidget(self.raw_json_text, 1, 1)
        return w

    def _build_console(self) -> QWidget:
        w = QWidget()
        layout = QVBoxLayout(w)
        self.console_text = QPlainTextEdit(); self.console_text.setReadOnly(True)
        layout.addWidget(self.console_text)
        return w

    def _seed_empty_spectra(self) -> None:
        freqs = np.linspace(868.0, 869.9, 256)
        noise = -95.0 + 2.0 * np.sin(np.linspace(0, 8 * np.pi, 256))
        self.wideband_plot.set_data(freqs, noise, [c / 1e6 for c in EU868_CHANNELS], None, "868.0 - 869.9 MHz | standby")
        ch_freqs = np.linspace(-62.5, 62.5, 256) / 1000.0 + 868.1
        self.channel_plot.set_data(ch_freqs, -95 + np.zeros(256), [], None, "CH 868.1 MHz | SF8 | BW 125 kHz")

    def _source_changed(self, source: str) -> None:
        is_file = source == "file"
        self.iq_edit.setEnabled(is_file)

    def _browse_iq(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Open IQ replay file", "", "IQ files (*.npz *.npy *.dat *.cf32);;All files (*)")
        if path:
            self.iq_edit.setText(path)
            self.source_combo.setCurrentText("file")

    def _set_running(self, running: bool) -> None:
        self.start_btn.setEnabled(not running)
        self.stop_btn.setEnabled(running)
        for widget in [
            self.source_combo, self.channels_combo, self.center_edit, self.sf_spin, self.bw_edit,
            self.sample_rate_edit, self.gain_edit, self.block_spin, self.max_packets_spin,
            self.jsonl_edit, self.repeats_spin, self.snr_spin, self.iso_spin, self.lockout_spin,
            self.lilygo_profile_check, self.expected_payload_edit, self.candidate_log_check,
        ]:
            widget.setEnabled(not running)
        self.status_pill.setText("● SCANNING — EU868" if running else "● IDLE — EU868")
        self.status_pill.setStyleSheet(f"color: {GREEN if running else MUTED};")

    def _read_config(self) -> GuiRunConfig:
        channels_text = self.channels_combo.currentText().strip()
        channels = _parse_channels(channels_text)
        lilygo_profile = bool(self.lilygo_profile_check.isChecked())
        expected_payload_ascii = self.expected_payload_edit.text().strip() if lilygo_profile else ""
        expected_payload_len = len(expected_payload_ascii.encode("utf-8")) if expected_payload_ascii else None
        return GuiRunConfig(
            source=self.source_combo.currentText().strip(),
            iq_path=self.iq_edit.text().strip(),
            center_hz=_safe_float(self.center_edit.text()),
            channels=channels,
            sf=int(self.sf_spin.value()),
            bw_hz=_safe_float(self.bw_edit.text()),
            sample_rate=_safe_float(self.sample_rate_edit.text()),
            gain=_safe_gain(self.gain_edit.text()),
            block_size=int(self.block_spin.value()),
            jsonl_path=self.jsonl_edit.text().strip(),
            max_packets=int(self.max_packets_spin.value()),
            preamble_repeats=int(self.repeats_spin.value()),
            preamble_snr_db=float(self.snr_spin.value()),
            min_peak_isolation=float(self.iso_spin.value()),
            valid_lockout_seconds=float(self.lockout_spin.value()),
            log_debug_candidates=bool(self.candidate_log_check.isChecked()),
            expected_payload_len=expected_payload_len,
            expected_cr_index=1 if lilygo_profile else None,
            expected_crc_enabled=1 if lilygo_profile else None,
            expected_payload_ascii=expected_payload_ascii,
            show_console_json=bool(self.console_json_check.isChecked()),
        )

    def start_scanner(self) -> None:
        if self.worker is not None or self.thread is not None:
            return
        try:
            cfg = self._read_config()
        except Exception as exc:
            QMessageBox.critical(self, "Invalid settings", str(exc))
            return
        self.run_started_at = time.time()
        self.console_text.appendPlainText("Starting scanner with settings:\n" + json.dumps(asdict(cfg), indent=2))
        self.status_label.setText("Starting scanner…")
        self.telemetry_chip.setText(f"{cfg.sample_rate/1e6:.1f} MSPS  |  {len(cfg.channels)} CH  |  {cfg.bw_hz/1e3:.0f} kHz BW")
        self._update_cards()
        self._set_running(True)
        self.thread = QThread()
        self.worker = ScannerWorker(cfg)
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.run)
        self.worker.packet.connect(self.add_packet)
        self.worker.channelStats.connect(self.update_channel_stats)
        self.worker.widebandSpectrum.connect(self.update_wideband_spectrum)
        self.worker.channelSpectrum.connect(self.update_channel_spectrum)
        self.worker.status.connect(self.set_status)
        self.worker.console.connect(self.append_console)
        self.worker.error.connect(self.show_worker_error)
        self.worker.finished.connect(self._worker_finished)
        self.worker.finished.connect(self.thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.thread.finished.connect(self._thread_finished)
        self.thread.finished.connect(self.thread.deleteLater)
        self.thread.start()

    def stop_scanner(self) -> None:
        if self.worker is not None:
            self.status_label.setText("Pausing scanner…")
            self.stop_btn.setEnabled(False)
            self.stop_btn.setText("▣ PAUSING")
            self.worker.stop()

    def _worker_finished(self) -> None:
        self.worker = None
        self._set_running(False)
        self.status_label.setText("Scanner idle.")
        self.stop_btn.setText("▣ PAUSE")

    def _thread_finished(self) -> None:
        self.thread = None

    def set_status(self, text: str) -> None:
        self.status_label.setText(text)
        self.status_pill.setText("● " + text[:42])
        self.console_text.appendPlainText(text)

    def append_console(self, text: str) -> None:
        self.console_text.appendPlainText(text)

    def show_worker_error(self, text: str) -> None:
        self.console_text.appendPlainText("ERROR:\n" + text)
        QMessageBox.critical(self, "Scanner error", text[-4000:])

    def clear_tables(self) -> None:
        self.packet_count = 0
        self.full_packet_count = 0
        self.debug_count = 0
        self.packet_records.clear()
        self.debug_records.clear()
        self.snr_values.clear()
        self.rssi_values.clear()
        self.last_sf = "—"
        self.packet_table.setRowCount(0)
        self.debug_table.setRowCount(0)
        self.channel_table.setRowCount(0)
        self.channel_rows.clear()
        self.summary_text.clear()
        self.raw_json_text.clear()
        self.console_text.clear()
        self.card_phdr.set_value("0")
        self.card_full.set_value("0")
        self.card_debug.set_value("0")
        self.card_pass.set_value("0.0%")
        self.card_snr.set_value("—")
        self.packet_hint.setText("0 PHDR packets  |  0 full decoded  |  0 debug candidates  |  Payload pass 0.0%")
        self._seed_empty_spectra()

    def export_csv(self) -> None:
        path, _ = QFileDialog.getSaveFileName(self, "Export packet table CSV", "lora_packets.csv", "CSV files (*.csv)")
        if not path:
            return
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if self.packet_records:
                writer.writerow(CSV_HEADERS)
                for idx, record in enumerate(self.packet_records, start=1):
                    writer.writerow(_csv_row_from_record(idx, record))
            else:
                headers = [self.packet_table.horizontalHeaderItem(i).text() for i in range(self.packet_table.columnCount())]
                writer.writerow(headers)
                for row in range(self.packet_table.rowCount()):
                    writer.writerow([(self.packet_table.item(row, col).text() if self.packet_table.item(row, col) else "") for col in range(self.packet_table.columnCount())])
        self.status_label.setText(f"CSV exported: {path}")

    def update_wideband_spectrum(self, payload: dict[str, Any]) -> None:
        self.wideband_plot.set_data(
            payload.get("freqs_mhz", []),
            payload.get("power_db", []),
            payload.get("channels_mhz", []),
            self.last_active_channel_mhz,
            payload.get("subtitle"),
        )

    def update_channel_spectrum(self, payload: dict[str, Any]) -> None:
        ch_hz = payload.get("channel_hz")
        self.channel_plot.set_data(payload.get("freqs_mhz", []), payload.get("power_db", []), [], None, payload.get("subtitle"))
        if ch_hz:
            self.last_active_channel_mhz = float(ch_hz) / 1e6

    def update_channel_stats(self, stats: list[dict[str, Any]]) -> None:
        for idx, st in enumerate(stats, start=1):
            ch = float(st["channel_hz"])
            if ch not in self.channel_rows:
                row = self.channel_table.rowCount()
                self.channel_table.insertRow(row)
                self.channel_rows[ch] = row
            row = self.channel_rows[ch]
            values = [
                f"CH{idx}  {_fmt_freq_full(ch)}",
                _fmt_float(st.get("fs"), 0, " Hz"),
                _fmt_float(st.get("power_db"), 1, " dB"),
                _fmt_float(st.get("peak_db"), 1, " dB"),
                str(st.get("samples", "")),
            ]
            for col, val in enumerate(values):
                item = QTableWidgetItem(val)
                item.setTextAlignment(ALIGN_CENTER)
                if col in {2, 3}:
                    item.setForeground(QColor(GREEN if float(st.get("peak_db", -999.0)) > -40 else MUTED))
                self.channel_table.setItem(row, col, item)

    def _update_cards(self) -> None:
        self.card_phdr.set_value(str(self.packet_count))
        self.card_full.set_value(str(self.full_packet_count))
        self.card_debug.set_value(str(self.debug_count))
        pass_pct = (100.0 * self.full_packet_count / self.packet_count) if self.packet_count else 0.0
        self.card_pass.set_value(f"{pass_pct:.1f}%")
        self.card_snr.set_value(f"{float(np.mean(self.snr_values)):.1f}" if self.snr_values else "—")
        self.packet_hint.setText(
            f"{self.packet_count} PHDR packets  |  {self.full_packet_count} full decoded  |  "
            f"{self.debug_count} debug candidates  |  Payload pass {pass_pct:.1f}%"
        )

    def add_packet(self, record: dict[str, Any]) -> None:
        if _is_real_packet(record):
            self._add_real_packet(record)
        else:
            self.add_debug_candidate(record)

    def _add_real_packet(self, record: dict[str, Any]) -> None:
        self.packet_count += 1
        if _payload_decode_ok(record):
            self.full_packet_count += 1
        self.packet_records.append(record)
        row = self.packet_table.rowCount()
        self.packet_table.insertRow(row)
        sync = record.get("sync") or {}
        det = record.get("detector") or {}
        payload = record.get("payload") or {}
        snr = sync.get("snr_db", det.get("snr_db"))
        rssi = sync.get("rssi_db", det.get("rssi_db"))
        if isinstance(snr, (int, float)) and math.isfinite(float(snr)):
            self.snr_values.append(float(snr))
        if isinstance(rssi, (int, float)) and math.isfinite(float(rssi)):
            self.rssi_values.append(float(rssi))
        self.last_sf = f"SF{record.get('sf', '—')}"
        ch_mhz = float(record.get("channel_hz", 0.0)) / 1e6
        self.last_active_channel_mhz = ch_mhz

        values = [
            str(self.packet_count),
            str(record.get("timestamp", ""))[11:23],
            f"CH?",
            f"{ch_mhz:.1f}",
            f"SF{record.get('sf', '')}",
            _fmt_float(snr, 1),
            _fmt_float(rssi, 1),
            _fmt_float(det.get("rssi_db"), 1),
            _fmt_float(sync.get("cfo_hz"), 1),
            "✓ " + _short_list(det.get("preamble_bins") or sync.get("preamble_bins"), 8),
            _short_list(sync.get("sfd_bins"), 4),
            _best_phdr_text(record),
            _header_offset_text(record),
            _payload_offset_text(record),
            _raw_bins_text(record, "header", 16),
            _raw_bins_text(record, "payload", 32),
            _payload_decode_status(record),
            _decoded_ascii(record),
            _payload_crc_text(record),
            _packet_class_text(record),
        ]
        # Try to label EU868 channel index.
        try:
            closest = min(range(len(EU868_CHANNELS)), key=lambda i: abs(EU868_CHANNELS[i] - float(record.get("channel_hz", 0.0))))
            values[2] = f"CH{closest + 1}"
        except Exception:
            pass
        values = _csv_row_from_record(self.packet_count, record)

        for col, val in enumerate(values):
            item = QTableWidgetItem(val)
            item.setTextAlignment(ALIGN_CENTER)
            header = CSV_HEADERS[col] if col < len(CSV_HEADERS) else ""
            if header == "SF":
                item.setForeground(QColor(_sf_badge_color(record.get("sf"))))
            elif header == "SNR [dB]":
                try:
                    item.setForeground(QColor(GREEN if float(snr) >= 0 else RED))
                except Exception:
                    item.setForeground(QColor(MUTED))
            elif header == "PHDR":
                if "OK" in val or val.startswith("ZIP"):
                    item.setForeground(QColor(GREEN))
                    item.setToolTip("PHDR CRC passed and one convention was selected.")
                elif "AMBIG" in val:
                    item.setForeground(QColor(ORANGE))
                    item.setToolTip("More than one header convention passed PHDR CRC; backend refuses to guess.")
                else:
                    item.setForeground(QColor(RED))
            elif header in {"Hdr off [samp]", "Payload timing offset", "Payload bin offset", "Payload residual CFO [bins]", "Payload residual CFO [Hz]"}:
                item.setForeground(QColor(MUTED))
            elif header in {"Header raw bins", "Payload raw bins", "Payload corrected bins", "Payload original bins"}:
                item.setForeground(QColor(PURPLE))
            elif header == "Decode status":
                item.setForeground(QColor(GREEN if val == "OK" else RED if "FAIL" in val else ORANGE))
            elif header == "Payload CRC":
                item.setForeground(QColor(GREEN if val == "PASS" else RED if val == "FAIL" else MUTED))
            elif header == "Packet class":
                item.setForeground(QColor(GREEN if val == "FULL" else ORANGE if val == "PHDR ONLY" else MUTED))
            elif header == "Preamble":
                item.setForeground(QColor(GREEN))
            self.packet_table.setItem(row, col, item)
        self.packet_table.item(row, 0).setData(256, record)
        self.packet_table.scrollToBottom()
        self._show_record(record)
        self._update_cards()
        self.wideband_plot.active_mhz = self.last_active_channel_mhz
        self.wideband_plot.update()

    def add_debug_candidate(self, record: dict[str, Any]) -> None:
        self.debug_count += 1
        self.debug_records.append(record)
        row = self.debug_table.rowCount()
        self.debug_table.insertRow(row)
        sync = record.get("sync") or {}
        det = record.get("detector") or {}
        snr = sync.get("snr_db", det.get("snr_db"))
        rssi = sync.get("rssi_db", det.get("rssi_db"))
        ch_mhz = float(record.get("channel_hz", 0.0)) / 1e6
        values = [
            str(self.debug_count),
            str(record.get("timestamp", ""))[11:23],
            f"{ch_mhz:.1f}",
            f"SF{record.get('sf', '')}",
            _fmt_float(snr, 1),
            _fmt_float(rssi, 1),
            _short_list(det.get("preamble_bins") or sync.get("preamble_bins"), 8),
            _short_list(sync.get("sfd_bins"), 4),
            _candidate_reason(record),
        ]
        for col, val in enumerate(values):
            item = QTableWidgetItem(val)
            item.setTextAlignment(ALIGN_CENTER)
            if col == 3:
                item.setForeground(QColor(_sf_badge_color(record.get("sf"))))
            elif col == 8:
                item.setForeground(QColor(ORANGE if "ambig" in val.lower() else RED))
            else:
                item.setForeground(QColor(MUTED))
            self.debug_table.setItem(row, col, item)
        self.debug_table.item(row, 0).setData(256, record)
        self.debug_table.scrollToBottom()
        self._update_cards()

    def _selected_packet_changed(self) -> None:
        items = self.packet_table.selectedItems()
        if not items:
            return
        row = items[0].row()
        item = self.packet_table.item(row, 0)
        record = item.data(256) if item is not None else None
        if isinstance(record, dict):
            self._show_record(record)

    def _selected_debug_changed(self) -> None:
        items = self.debug_table.selectedItems()
        if not items:
            return
        row = items[0].row()
        item = self.debug_table.item(row, 0)
        record = item.data(256) if item is not None else None
        if isinstance(record, dict):
            self._show_record(record)

    def _show_record(self, record: dict[str, Any]) -> None:
        sync = record.get("sync") or {}
        det = record.get("detector") or {}
        header = record.get("header") or {}
        payload = record.get("payload") or {}
        pdec = record.get("payload_decode") or {}
        hdbg = record.get("header_debug") or {}
        legacy_h = record.get("legacy_zip_phdr") or {}
        legacy_p = record.get("legacy_zip_payload_decode") or {}
        lines = [
            f"Channel: {_fmt_freq_full(float(record.get('channel_hz', 0.0)))} | SF{record.get('sf')} | BW={record.get('bw_hz')} Hz",
            f"Candidate sample: {record.get('candidate_sample_index')}",
            f"Preamble bins: {_short_list(det.get('preamble_bins') or sync.get('preamble_bins'), 32)}",
            f"SFD raw bins: {_short_list(sync.get('sfd_bins'), 16)}",
            f"SFD note: {sync.get('sfd_bins_note', 'raw synchronization display; use PHDR/payload CRC as packet evidence')}",
            f"CFO estimate: {_fmt_float(sync.get('cfo_hz'), 2, ' Hz')} | STO: {sync.get('sto_samples')} samples",
            "CFO/SFD display note: diagnostic estimate from synchronization, not calibrated RF evidence.",
            f"SNR: {_fmt_float(sync.get('snr_db', det.get('snr_db')), 2, ' dB')} | Relative power: {_fmt_float(sync.get('rssi_db', det.get('rssi_db')), 2, ' dB')}",
            "",
            f"Header timing offset: {_header_offset_text(record)} samples",
            f"Header raw bins: {_raw_bins_text(record, 'header', 64)}",
            f"Header odd check: all_odd={hdbg.get('raw_all_odd')} odd_count={hdbg.get('raw_odd_count')} parity={hdbg.get('raw_parity_lsb')}",
            f"PHDR: {_best_phdr_text(record)} | clean_pass_count={record.get('phdr_pass_count', hdbg.get('phdr_pass_count'))}",
            f"Exact ZIP PHDR: ok={legacy_h.get('ok')} L={legacy_h.get('payload_len')} CR=4/{legacy_h.get('cr_denom')} CRC={legacy_h.get('crc_enabled')} reason={legacy_h.get('failure_reason')}",
            f"Payload timing offset: {_payload_offset_text(record)} samples",
            f"Payload bin offset: {_payload_bin_offset_text(record)} bins",
            f"Payload residual CFO: {_payload_residual_cfo_text(record, 'residual_cfo_bins', 2)} bins / {_payload_residual_cfo_text(record, 'residual_cfo_hz', 2)} Hz",
            f"Payload timing sweep: {_payload_sweep_summary(record)}",
            f"Payload raw bins: {_raw_bins_text(record, 'payload', 80)}",
            f"Decode status: {_payload_decode_status(record)}",
            f"Decoded ASCII: {_decoded_ascii(record)}",
            f"Payload CRC: {_payload_crc_text(record)}",
            f"Exact ZIP payload: ok={legacy_p.get('ok')} crc_ok={legacy_p.get('crc_ok')} ascii={legacy_p.get('payload_ascii')} reason={legacy_p.get('failure_reason')}",
            f"Failure reason: {record.get('failure_reason', '')}",
            f"Failure class: {_payload_failure_class_text(record)}",
        ]
        if pdec:
            lines.extend([
                "",
                f"Decoded payload hex: {pdec.get('payload_hex')}",
                f"Decoded payload ascii: {pdec.get('payload_ascii')}",
                f"Payload CRC OK: {pdec.get('crc_ok')}",
                f"Hamming errors: {pdec.get('hamming_errors')}",
            ])
        self.summary_text.setPlainText("\n".join(lines))
        self.raw_json_text.setPlainText(json.dumps(record, ensure_ascii=False, indent=2, sort_keys=True, default=str))


def main() -> int:
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    return app.exec_() if PYQT_VERSION == 5 else app.exec()


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
