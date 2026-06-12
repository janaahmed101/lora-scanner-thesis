from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

EU868_CHANNELS_HZ = [
    868.1e6, 868.3e6, 868.5e6, 868.7e6,
    868.9e6, 869.1e6, 869.3e6, 869.5e6,
]
EU868_CHANNELS = EU868_CHANNELS_HZ


@dataclass(slots=True)
class ReceiverConfig:
    sf: int = 8
    bw: float = 125e3
    sample_rate: float = 2.0e6
    center_freq: float = 868.8e6
    channels: Sequence[float] = field(default_factory=lambda: list(EU868_CHANNELS_HZ))
    gain: str | float = "auto"
    block_size: int = 262_144
    # Channel oversampling factor (samples per LoRa chip). 1 keeps the original
    # 1-sample/chip behaviour; >=2 enables sub-chip timing so the SFD-locked
    # joint CFO/STO synchronizer (Synchronizer2) can correct real OTA signals.
    oversample: int = 4
    preamble_len: int = 8
    preamble_repeats: int = 6
    preamble_drift_bins: int = 1
    preamble_snr_db: float = 6.0
    min_peak_isolation: float = 1.35
    timing_sweep: int = 48
    max_payload_bytes: int = 255
    max_packets: int = 0
    valid_lockout_seconds: float = 1.8
    log_debug_candidates: bool = False
    expected_payload_len: int | None = None
    expected_cr_index: int | None = None
    expected_crc_enabled: int | None = None
    expected_payload_ascii: str = ""
    payload_timing_offsets: Sequence[int] = field(
        default_factory=lambda: [0, -1, 1, -2, 2, -4, 4, -8, 8, -16, 16]
    )
    payload_bin_offsets: Sequence[int] = field(default_factory=lambda: [0, -1, 1, -2, 2])
    payload_residual_cfo_bins: Sequence[float] = field(
        default_factory=lambda: [0.0, 0.25, -0.25, 0.5, -0.5, 0.75, -0.75, 1.0, -1.0]
    )
    payload_topk: int = 1
    payload_topk_max_symbols: int = 0
    payload_rescue_max_attempts: int = 0
    payload_deep_rescue_enabled: bool = False
    payload_deep_topk_max_symbols: int = 4
    payload_deep_rescue_max_attempts: int = 250
    payload_deep_rescue_time_budget_s: float = 0.12
    jsonl_path: Path | None = None
    debug: bool = True

    @property
    def n_bins(self) -> int:
        return 1 << int(self.sf)

    @property
    def symbol_samples(self) -> int:
        # Samples per LoRa chip-symbol at the channel rate. With oversample>1 the
        # channel carries 2^SF * oversample samples per symbol; all receiver
        # buffer / lockout / keep arithmetic is expressed in these units and so
        # scales automatically.
        return (1 << int(self.sf)) * max(1, int(self.oversample))

    @property
    def channel_sample_rate(self) -> float:
        return float(self.bw) * max(1, int(self.oversample))

    @property
    def max_packet_symbols(self) -> int:
        # Worst-case explicit LoRa packet: preamble + sync + SFD + header + payload.
        max_payload_syms = int(((self.max_payload_bytes * 8 + 28 - 4 * self.sf + 16) + (4 * self.sf - 1)) // (4 * self.sf)) * 8
        return int(self.preamble_len + 4.25 + 8 + max_payload_syms + 4)
