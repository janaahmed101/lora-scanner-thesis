from __future__ import annotations

from pathlib import Path
import numpy as np


def load_iq_file(path: str | Path) -> np.ndarray:
    """Load IQ from .npy, .npz, or raw .dat/.cf32 complex64 files.

    Supported arrays:
    - complex64 / complex128 vector
    - real array with final dimension 2: I,Q
    - .npz containing one of: iq, samples, arr_0
    """
    p = Path(path)
    suffix = p.suffix.lower()
    if suffix == ".npy":
        arr = np.load(p, allow_pickle=False)
    elif suffix == ".npz":
        z = np.load(p, allow_pickle=False)
        key = next((k for k in ("iq", "samples", "arr_0", "x") if k in z.files), z.files[0])
        arr = z[key]
    else:
        raw = np.fromfile(p, dtype=np.complex64)
        if raw.size == 0:
            # fallback: interleaved float32 I,Q
            f = np.fromfile(p, dtype=np.float32)
            if len(f) % 2:
                f = f[:-1]
            raw = f.reshape(-1, 2)[:, 0] + 1j * f.reshape(-1, 2)[:, 1]
        arr = raw
    arr = np.asarray(arr)
    if np.iscomplexobj(arr):
        iq = arr.astype(np.complex64).reshape(-1)
    elif arr.ndim >= 2 and arr.shape[-1] == 2:
        iq = (arr[..., 0].reshape(-1) + 1j * arr[..., 1].reshape(-1)).astype(np.complex64)
    else:
        raise ValueError(f"Cannot interpret IQ shape {arr.shape} from {p}")
    if len(iq) == 0:
        raise ValueError(f"No IQ samples found in {p}")
    return iq


class IQFileSource:
    def __init__(self, path: str | Path, block_size: int = 262_144):
        self.samples = load_iq_file(path)
        self.block_size = int(block_size)

    def blocks(self):
        for i in range(0, len(self.samples), self.block_size):
            yield self.samples[i : i + self.block_size]
