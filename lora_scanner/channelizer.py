from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from scipy import signal


@dataclass(slots=True)
class ChannelOutput:
    center_hz: float
    sample_rate: float
    samples: np.ndarray


def remove_dc(x: np.ndarray) -> np.ndarray:
    if len(x) == 0:
        return x.astype(np.complex64)
    return (x - np.mean(x)).astype(np.complex64)


class Channelizer:
    """Digital downconverter for one or more 125 kHz LoRa channels."""

    def __init__(self, input_sample_rate: float, rf_center_hz: float, channels_hz: list[float], bw_hz: float = 125e3, taps: int = 129, oversample: int = 1):
        self.input_sample_rate = float(input_sample_rate)
        self.rf_center_hz = float(rf_center_hz)
        self.channels_hz = [float(c) for c in channels_hz]
        self.bw_hz = float(bw_hz)
        self.oversample = max(1, int(oversample))
        self.taps = int(taps)
        cutoff = min(0.48 * self.bw_hz, 0.45 * self.input_sample_rate)
        self.lpf = signal.firwin(self.taps, cutoff, fs=self.input_sample_rate, window="hamming").astype(np.float32)
        self._sample_index = 0
        target_rate = self.bw_hz * self.oversample
        ratio = self.input_sample_rate / target_rate
        self.decim = int(round(ratio))
        if self.decim <= 0 or abs(self.input_sample_rate / self.decim - target_rate) > 1.0:
            raise ValueError(
                "sample_rate/bw/oversample must be an integer decimation. "
                f"Got sample_rate={self.input_sample_rate}, bw={self.bw_hz}, "
                f"oversample={self.oversample}, decim={self.decim}"
            )

    def process(self, wideband: np.ndarray) -> list[ChannelOutput]:
        x = remove_dc(np.asarray(wideband, dtype=np.complex64))
        n = np.arange(len(x), dtype=np.float64) + self._sample_index
        out: list[ChannelOutput] = []
        for ch in self.channels_hz:
            freq_offset = ch - self.rf_center_hz
            mixer = np.exp(-1j * 2.0 * np.pi * freq_offset * n / self.input_sample_rate).astype(np.complex64)
            shifted = x * mixer
            filtered = signal.lfilter(self.lpf, [1.0], shifted).astype(np.complex64)
            if self.decim > 1:
                decimated = filtered[:: self.decim]
                fs_out = self.input_sample_rate / self.decim
            else:
                decimated = filtered
                fs_out = self.input_sample_rate
            out.append(ChannelOutput(ch, float(fs_out), remove_dc(decimated)))
        self._sample_index += len(x)
        return out
