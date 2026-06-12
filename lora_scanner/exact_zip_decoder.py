from __future__ import annotations

"""Exact LoRa_Rx_SDR ZIP decoder adapter.

This module deliberately calls the original Decode_Symbols.py functions copied
from the provided LoRa_Rx_SDR.zip. The original file is kept under the top-level
LoRa_Rx_SDR/ package in this project so its import path remains unchanged:

    from LoRa_Rx_SDR import Decode_Symbols

The GUI/backend can use this as a side-by-side or preferred PHY decoder without
rewriting the legacy header/payload bit logic.
"""

from dataclasses import dataclass, asdict
from typing import Any
import math
import numpy as np

try:  # top-level package copied exactly from the ZIP
    from LoRa_Rx_SDR import Decode_Symbols as legacy_decode
except Exception as exc:  # pragma: no cover
    legacy_decode = None
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None


@dataclass(slots=True)
class ExactZipHeader:
    ok: bool
    payload_len: int | None
    cr_index: int | None
    cr_denom: int | None
    crc_enabled: int | None
    extra_header_bits: list[list[int]]
    raw_bins: list[int]
    failure_reason: str = ""

    def as_record(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class ExactZipPayload:
    ok: bool
    payload_hex: str
    payload_ascii: str
    crc_ok: bool
    failure_reason: str
    expected_symbols: int
    raw_bins: list[int]

    def as_record(self) -> dict[str, Any]:
        return asdict(self)


def payload_symbol_count(payload_len: int, sf: int, cr_index: int, crc_enabled: int) -> int:
    """Exact symbol count convention used by Packet_Decoding.py.

    The legacy code names this Nof_Pkt_Syms, then subtracts 8 header symbols.
    This helper returns only the payload symbols following the 8 explicit header
    symbols.
    """
    n_total = int(8 + math.ceil((payload_len * 8 + 28 - 4 * sf + 16 * int(crc_enabled)) / (4 * sf)) * (4 + int(cr_index)))
    return max(0, n_total - 8)


def decode_header_exact(raw_bins: list[int] | np.ndarray, sf: int) -> ExactZipHeader:
    raw = [int(x) for x in list(raw_bins)[:8]]
    if _IMPORT_ERROR is not None:
        return ExactZipHeader(False, None, None, None, None, [], raw, f"legacy_import_failed:{_IMPORT_ERROR}")
    if len(raw) < 8:
        return ExactZipHeader(False, None, None, None, None, [], raw, "short_header")
    try:
        info = legacy_decode.Get_Header_Info(int(sf), np.asarray(raw, dtype=float))
        ok = bool(info[4])
        if not ok:
            return ExactZipHeader(False, None, None, None, None, [], raw, "legacy_phdr_crc_fail")
        pl = int(info[0])
        cr = int(info[1])
        crc = int(info[2])
        extra = np.asarray(info[3], dtype=int)
        return ExactZipHeader(True, pl, cr, cr + 4, crc, extra.astype(int).tolist(), raw, "")
    except Exception as exc:
        return ExactZipHeader(False, None, None, None, None, [], raw, f"legacy_header_exception:{exc}")


def decode_payload_exact(raw_bins: list[int] | np.ndarray, sf: int, header: ExactZipHeader) -> ExactZipPayload:
    raw = [int(x) for x in list(raw_bins)]
    if _IMPORT_ERROR is not None:
        return ExactZipPayload(False, "", "", False, f"legacy_import_failed:{_IMPORT_ERROR}", 0, raw)
    if not header.ok or header.payload_len is None or header.cr_index is None or header.crc_enabled is None:
        return ExactZipPayload(False, "", "", False, "missing_valid_header", 0, raw)

    expected = payload_symbol_count(header.payload_len, int(sf), header.cr_index, header.crc_enabled)
    if len(raw) < expected:
        return ExactZipPayload(False, "", "", False, f"short_payload_symbols:{len(raw)}<{expected}", expected, raw)
    try:
        extra = np.asarray(header.extra_header_bits, dtype=int)
        ok = bool(legacy_decode.Get_Data_Info(
            int(sf),
            int(header.cr_index),
            int(header.payload_len),
            int(header.crc_enabled),
            int(expected),
            np.asarray(raw[:expected], dtype=float),
            extra,
        ))
        phy, _mac = legacy_decode.Get_Decoded_Bytes()
        phy = phy or b""
        ascii_text = "".join(chr(b) if 32 <= int(b) <= 126 else "." for b in phy)
        return ExactZipPayload(
            ok=ok,
            payload_hex=phy.hex().upper(),
            payload_ascii=ascii_text,
            crc_ok=ok,
            failure_reason="" if ok else "legacy_payload_crc_fail",
            expected_symbols=expected,
            raw_bins=raw[:expected],
        )
    except Exception as exc:
        return ExactZipPayload(False, "", "", False, f"legacy_payload_exception:{exc}", expected, raw[:expected])
