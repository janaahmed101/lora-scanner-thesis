from __future__ import annotations

from dataclasses import dataclass, asdict
from itertools import product
import math
from typing import Any
import numpy as np


# Hamming/parity logic is a clean port of the useful Decode_Symbols.py receiver logic
# from the provided LoRa_Rx_SDR ZIP. The old module mixed PHY decoding with global
# state and app/MAC/cloud concerns; this file keeps only LoRaPHY operations.
_G = np.array(
    [
        [1, 0, 0, 0, 1, 0, 1, 1],
        [0, 1, 0, 0, 1, 1, 1, 0],
        [0, 0, 1, 0, 1, 1, 0, 1],
        [0, 0, 0, 1, 0, 1, 1, 1],
    ],
    dtype=int,
)
_H_CR4 = np.transpose(_G[:, 4:8])
_H_CR4 = np.hstack((_H_CR4, np.eye(4, dtype=int)))
_H_CR3 = _H_CR4[0:3, 0:7]
_H_CR4 = np.transpose(np.flipud(_H_CR4))
_H_CR3 = np.transpose(np.flipud(_H_CR3))

_CORR4 = np.zeros((16, 8), dtype=int)
for i in range(8):
    idx = int("".join(str(int(x)) for x in _H_CR4[i, :]), 2)
    _CORR4[idx, i] = 1

_CORR3 = np.zeros((8, 7), dtype=int)
for i in range(7):
    idx = int("".join(str(int(x)) for x in _H_CR3[i, :]), 2)
    _CORR3[idx, i] = 1

_G_HCRC = np.array(
    [
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0],
        [0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1],
        [0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1],
        [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1],
        [0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 1],
        [0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 0],
        [0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0],
    ],
    dtype=int,
)


def _whitening_bytes(n_bytes: int = 255) -> np.ndarray:
    n_outputs = max(262, n_bytes + 8)
    reg = np.ones(9, dtype=int)
    reg[0] = 0
    seq = np.zeros(n_outputs, dtype=int)
    for i in range(n_outputs):
        seq[i] = reg[-1]
        feedback = (reg[8] + reg[7] + reg[6] + reg[3] + reg[0]) % 2
        reg = np.append(feedback, reg[:-1])
    out = np.zeros(n_bytes, dtype=np.uint8)
    for i in range(n_bytes):
        val = 0
        for b in seq[i : i + 8]:
            val = (val << 1) | int(b)
        out[i] = val
    return out

_WHITENING = _whitening_bytes(512)


def bin_to_gray(x: int) -> int:
    return int(x) ^ (int(x) >> 1)


def gray_to_binary(g: int) -> int:
    g = int(g)
    b = 0
    while g:
        b ^= g
        g >>= 1
    return b


def bits_lsb_first(value: int, width: int) -> np.ndarray:
    return np.array([(int(value) >> i) & 1 for i in range(width)], dtype=int)


def symbols_to_bits(symbols: list[int] | np.ndarray, sf: int, adjust: int, gray: str, trim_lsb: int = 0) -> np.ndarray:
    n = 1 << int(sf)
    rows = []
    for s in symbols:
        adj = (int(s) + int(adjust)) % n
        if gray == "bin_to_gray":
            val = bin_to_gray(adj)
        elif gray == "gray_to_bin":
            val = gray_to_binary(adj)
        elif gray == "none":
            val = adj
        else:
            raise ValueError(f"unknown gray mode {gray}")
        rows.append(bits_lsb_first(val, sf)[trim_lsb:])
    return np.vstack(rows).astype(int)


def deinterleave_rows(bits: np.ndarray, direction: int = 1, shift: int = 0) -> np.ndarray:
    out = np.array(bits, dtype=int, copy=True)
    for row in range(out.shape[0]):
        out[row, :] = np.roll(out[row, :], int(direction) * row + int(shift))
    return out


def hamming_correct(block_bits: np.ndarray, cr_index: int) -> tuple[np.ndarray, int, list[list[int]]]:
    b = np.array(block_bits, dtype=int, copy=True)
    errors = 0
    syndromes: list[list[int]] = []
    if cr_index >= 4 and b.shape[0] >= 8:
        synd = (np.mod((b.T @ np.fliplr(_H_CR4)), 2)).astype(int)
        corr = np.zeros((synd.shape[0], 8), dtype=int)
        for i in range(synd.shape[0]):
            idx = int(np.sum(synd[i, :] * (2 ** np.arange(4)[::-1])))
            syndromes.append([int(x) for x in synd[i, :]])
            if idx:
                errors += 1
            corr[i, :] = _CORR4[idx, :]
        b[:8, :] = np.logical_xor(b[:8, :], corr.T).astype(int)
    elif cr_index == 3 and b.shape[0] >= 7:
        synd = (np.mod((b.T @ _H_CR3), 2)).astype(int)
        corr = np.zeros((synd.shape[0], 7), dtype=int)
        for i in range(synd.shape[0]):
            idx = int(np.sum(synd[i, :] * (2 ** np.arange(3)[::-1])))
            syndromes.append([int(x) for x in synd[i, :]])
            if idx:
                errors += 1
            corr[i, :] = _CORR3[idx, :]
        b[:7, :] = np.logical_xor(b[:7, :], corr.T).astype(int)
    return b, errors, syndromes


def _parse_header_bits(header_bits_4x: np.ndarray) -> tuple[int, int, int, np.ndarray, bool, dict[str, Any]]:
    hb = np.array(header_bits_4x, dtype=int, copy=True)
    if hb.shape[1] < 5:
        return 0, 0, 0, np.empty((4, 0), dtype=int), False, {"reason": "too_few_header_columns"}

    hcrc_bits = np.zeros((4, 2), dtype=int)
    hcrc_bits[0, 0] = int(np.sum(hb[:, 0]) % 2)
    comp = hb[:, 1:3].T.reshape(1, 8)
    comp = np.mod(comp @ _G_HCRC, 2)
    hcrc_bits[:, 1] = np.mod(hb[:, 0] + comp[0, 8:12], 2)
    crc_check = np.mod(hcrc_bits + hb[:, 3:5], 2)
    ok = bool(np.sum(crc_check) == 0)

    length_bits = np.flipud(hb[:, 0:2]).reshape(1, 8, order="F")
    payload_len = int("".join(str(int(x)) for x in length_bits[0]), 2)
    cr_bits = np.fliplr(hb[1:4, 2].reshape(1, 3)).astype(int)
    cr_index = int("".join(str(int(x)) for x in cr_bits[0]), 2)
    crc_enabled = int(hb[0, 2])
    extra = hb[:, 5:] if hb.shape[1] > 5 else np.empty((4, 0), dtype=int)
    debug = {
        "computed_hcrc": hcrc_bits.astype(int).tolist(),
        "received_hcrc": hb[:, 3:5].astype(int).tolist(),
        "crc_check_matrix": crc_check.astype(int).tolist(),
        "payload_len": payload_len,
        "cr_index": cr_index,
        "cr_denom": cr_index + 4,
        "crc_enabled": crc_enabled,
    }
    return payload_len, cr_index, crc_enabled, extra, ok, debug


@dataclass(slots=True)
class HeaderCandidate:
    ok: bool
    payload_len: int
    cr_index: int
    crc_enabled: int
    convention: dict[str, Any]
    header_bits: list[list[int]]
    extra_payload_bits: list[list[int]]
    hamming_errors: int
    debug: dict[str, Any]

    @property
    def cr_denom(self) -> int:
        return self.cr_index + 4

    def as_record(self) -> dict[str, Any]:
        d = asdict(self)
        d["cr_denom"] = self.cr_denom
        return d


def decode_header_candidate(raw_bins: list[int] | np.ndarray, sf: int, adjust: int, gray: str, deint_direction: int, deint_shift: int) -> HeaderCandidate:
    bits = symbols_to_bits(raw_bins, sf, adjust=adjust, gray=gray, trim_lsb=2)
    bits = deinterleave_rows(bits, direction=deint_direction, shift=deint_shift)
    corrected, errors, syndromes = hamming_correct(bits, cr_index=4)
    info = corrected[0:4, :]
    pl, cr, crc, extra, ok, debug = _parse_header_bits(info)
    debug["syndromes"] = syndromes
    return HeaderCandidate(
        ok=ok and 0 <= pl <= 255 and 1 <= cr <= 4 and crc in (0, 1),
        payload_len=int(pl),
        cr_index=int(cr),
        crc_enabled=int(crc),
        convention={
            "adjust": int(adjust),
            "gray": gray,
            "deint_direction": int(deint_direction),
            "deint_shift": int(deint_shift),
            "trim_lsb": 2,
        },
        header_bits=info.astype(int).tolist(),
        extra_payload_bits=extra.astype(int).tolist(),
        hamming_errors=int(errors),
        debug=debug,
    )


def decode_header(raw_bins: list[int] | np.ndarray, sf: int, exhaustive: bool = True) -> tuple[HeaderCandidate | None, list[HeaderCandidate], dict[str, Any]]:
    raw = [int(x) for x in raw_bins]
    parity = [int(x) & 1 for x in raw]
    odd_debug = {
        "raw_bins": raw,
        "raw_parity_lsb": parity,
        "raw_all_odd": bool(len(parity) > 0 and all(p == 1 for p in parity)),
        "raw_odd_count": int(sum(parity)),
    }
    adjusts = [0, -1, 1]
    grays = ["bin_to_gray", "gray_to_bin"] if exhaustive else ["bin_to_gray"]
    directions = [1, -1]
    shifts = [-1, 0, 1]
    candidates: list[HeaderCandidate] = []
    for adjust, gray, direction, shift in product(adjusts, grays, directions, shifts):
        try:
            cand = decode_header_candidate(raw, sf, adjust, gray, direction, shift)
            candidates.append(cand)
        except Exception as exc:
            candidates.append(
                HeaderCandidate(False, 0, 0, 0, {"adjust": adjust, "gray": gray, "deint_direction": direction, "deint_shift": shift}, [], [], 0, {"exception": str(exc)})
            )
    passes = [c for c in candidates if c.ok]
    if len(passes) == 1:
        selected = passes[0]
    elif len(passes) > 1:
        # Several conventions can satisfy the header CRC (off-by-one adjusts are
        # often self-consistent). Rather than refusing to decode, pick the
        # canonical LoRa convention; fall back to the smallest |adjust| with a
        # standard gray map. This is what lets the clean payload path run.
        def _rank(c: HeaderCandidate) -> tuple[Any, ...]:
            conv = c.convention
            canonical = (
                int(conv.get("adjust", 99)) == -1
                and conv.get("gray") == "bin_to_gray"
                and int(conv.get("deint_direction", 0)) == 1
                and int(conv.get("deint_shift", 99)) == 0
            )
            return (
                0 if canonical else 1,
                abs(int(conv.get("adjust", 99))),
                0 if conv.get("gray") == "bin_to_gray" else 1,
                abs(int(conv.get("deint_shift", 99))),
                int(c.hamming_errors),
            )
        selected = sorted(passes, key=_rank)[0]
    else:
        selected = None
    odd_debug["phdr_pass_count"] = len(passes)
    odd_debug["selection_status"] = "unique" if selected else ("none" if not passes else "multiple")
    return selected, candidates, odd_debug


def payload_symbol_count(payload_len: int, sf: int, cr_index: int, crc_enabled: int) -> int:
    numerator = payload_len * 8 + 28 - 4 * sf + 16 * int(crc_enabled)
    blocks = max(0, math.ceil(numerator / (4 * sf)))
    return int(blocks * (int(cr_index) + 4))


def lora_bits_to_bytes(bits_4xn: np.ndarray, n_bytes: int) -> bytes:
    bits = np.array(bits_4xn, dtype=int)
    out = np.zeros((8, n_bytes), dtype=int)
    counter = 0
    for col in range(0, n_bytes * 2, 2):
        out[0:4, counter] = bits[:, col]
        out[4:8, counter] = bits[:, col + 1]
        counter += 1
    out = np.fliplr(out.T)
    values = []
    for row in out:
        v = 0
        for bit in row:
            v = (v << 1) | int(bit)
        values.append(v)
    return bytes(values)


def _crc16_lora_bits(dec_bits_final: np.ndarray, payload_len: int) -> np.ndarray:
    # Port of the attached ZIP's payload CRC convention. It returns a 4x4 matrix
    # that is compared against the four CRC nibbles following the payload bits.
    crc_poly = np.array([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1], dtype=int)
    msg = dec_bits_final.astype(int)
    msg_bin = np.transpose(np.reshape(np.transpose(msg), (payload_len, 8)))
    msg_bin = np.flipud(msg_bin)
    msg_bin = np.reshape(np.transpose(msg_bin), (1, payload_len * 8))
    msg_bin = np.fliplr(msg_bin).astype(int)
    if not np.any(msg_bin):
        return np.zeros((4, 4), dtype=int)
    current_degree = int(np.max(np.where(msg_bin == 1)))
    poly = msg_bin.copy()
    while current_degree > 15:
        a = current_degree - 16
        b = current_degree + 1
        poly[0, a:b] = np.logical_xor(crc_poly, poly[0, a:b]).astype(int)
        nz = np.where(poly == 1)[1]
        if len(nz) == 0:
            break
        current_degree = int(np.max(nz))
    crc_bits = poly[0, 0:16]
    return np.transpose(np.reshape(crc_bits, (4, 4))).astype(int)


def _whitening_bits_lora(payload_len: int) -> np.ndarray:
    whitening = _WHITENING[:payload_len]
    bits8 = np.zeros((8, payload_len), dtype=int)
    out = np.zeros((4, payload_len * 2), dtype=int)
    for i, wb in enumerate(whitening):
        bits8[:, i] = bits_lsb_first(int(wb), 8)
        out[:, 2 * i] = bits8[0:4, i]
        out[:, 2 * i + 1] = bits8[4:8, i]
    return out


@dataclass(slots=True)
class PayloadDecodeResult:
    ok: bool
    payload_bytes: bytes
    crc_ok: bool
    failure_reason: str
    decoded_bits_columns: int
    crc_received: list[list[int]]
    crc_computed: list[list[int]]
    hamming_errors: int

    def as_record(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "payload_hex": self.payload_bytes.hex().upper(),
            "payload_ascii": ''.join(chr(b) if 32 <= b <= 126 else '.' for b in self.payload_bytes),
            "crc_ok": self.crc_ok,
            "failure_reason": self.failure_reason,
            "decoded_bits_columns": self.decoded_bits_columns,
            "crc_received": self.crc_received,
            "crc_computed": self.crc_computed,
            "hamming_errors": self.hamming_errors,
        }


def decode_payload(raw_bins: list[int] | np.ndarray, sf: int, header: HeaderCandidate) -> PayloadDecodeResult:
    pl = int(header.payload_len)
    cr = int(header.cr_index)
    crc_enabled = int(header.crc_enabled)
    conv = header.convention
    expected_syms = payload_symbol_count(pl, sf, cr, crc_enabled)
    if len(raw_bins) < expected_syms:
        return PayloadDecodeResult(False, b"", False, f"short_payload_symbols:{len(raw_bins)}<{expected_syms}", 0, [], [], 0)
    rows_per_block = cr + 4
    n_blocks = int(math.ceil(expected_syms / rows_per_block)) if rows_per_block else 0
    symbols = list(map(int, raw_bins[: expected_syms]))
    if n_blocks <= 0:
        return PayloadDecodeResult(False, b"", False, "no_payload_blocks", 0, [], [], 0)

    decoded_cols: list[np.ndarray] = []
    hamming_errors = 0
    for block in range(n_blocks):
        block_syms = symbols[block * rows_per_block : (block + 1) * rows_per_block]
        if len(block_syms) < rows_per_block:
            break
        bits = symbols_to_bits(block_syms, sf, adjust=int(conv.get("adjust", -1)), gray=str(conv.get("gray", "bin_to_gray")), trim_lsb=0)
        bits = deinterleave_rows(bits, direction=int(conv.get("deint_direction", 1)), shift=int(conv.get("deint_shift", 0)))
        corrected, errors, _ = hamming_correct(bits, cr_index=cr)
        hamming_errors += errors
        decoded_cols.append(corrected[0:4, :])
    if not decoded_cols:
        return PayloadDecodeResult(False, b"", False, "no_decoded_blocks", 0, [], [], hamming_errors)
    dec_bits = np.column_stack(decoded_cols).astype(int)

    # For SF>7, LoRa places some payload bits in the explicit header. The attached
    # decoder prepends Header_Info[3] before dewhitening, so we preserve that behavior.
    extra = np.array(header.extra_payload_bits, dtype=int)
    if sf > 7 and extra.size:
        dec_bits_new = np.column_stack((extra, dec_bits))
    else:
        dec_bits_new = dec_bits

    needed_payload_cols = pl * 2
    if dec_bits_new.shape[1] < needed_payload_cols:
        return PayloadDecodeResult(False, b"", False, f"short_decoded_bits:{dec_bits_new.shape[1]}<{needed_payload_cols}", int(dec_bits_new.shape[1]), [], [], hamming_errors)

    whiten = _whitening_bits_lora(pl)
    payload_bits = np.logical_xor(dec_bits_new[:, :needed_payload_cols], whiten).astype(int)
    payload = lora_bits_to_bytes(payload_bits, pl)

    crc_received = np.array(dec_bits_new[:, needed_payload_cols : needed_payload_cols + 4], dtype=int)
    crc_computed = _crc16_lora_bits(payload_bits, pl) if crc_enabled else np.zeros((4, 4), dtype=int)
    if crc_enabled:
        if crc_received.shape != (4, 4):
            return PayloadDecodeResult(False, payload, False, "missing_payload_crc_bits", int(dec_bits_new.shape[1]), crc_received.tolist(), crc_computed.tolist(), hamming_errors)
        crc_ok = bool(np.sum(np.mod(crc_received + crc_computed, 2)) == 0)
        return PayloadDecodeResult(crc_ok, payload if crc_ok else b"", crc_ok, "" if crc_ok else "payload_crc_fail", int(dec_bits_new.shape[1]), crc_received.tolist(), crc_computed.tolist(), hamming_errors)
    return PayloadDecodeResult(True, payload, True, "", int(dec_bits_new.shape[1]), [], [], hamming_errors)
