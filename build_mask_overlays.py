#!/usr/bin/env python3
"""Build transparent ASO magnitude-allocation overlays for the static demo."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.io import wavfile


ROOT = Path(__file__).resolve().parent
ASSETS = ROOT / "assets"


def stft_magnitude(signal: np.ndarray) -> np.ndarray:
    n_fft, hop = 1024, 128
    window = np.hanning(n_fft + 1)[:-1]
    padded = np.pad(signal.astype(np.float64), (n_fft // 2, n_fft // 2))
    frames = np.lib.stride_tricks.sliding_window_view(padded, n_fft)[::hop]
    return np.abs(np.fft.rfft(frames * window, axis=1)).T


def read_audio(path: Path) -> tuple[int, np.ndarray]:
    rate, signal = wavfile.read(path)
    if np.issubdtype(signal.dtype, np.integer):
        signal = signal.astype(np.float64) / np.iinfo(signal.dtype).max
    return rate, np.asarray(signal, dtype=np.float64).reshape(-1)


def rgba_overlay(mixture: np.ndarray, estimate: np.ndarray) -> np.ndarray:
    mixture_mag = stft_magnitude(mixture)
    estimate_mag = stft_magnitude(estimate)
    reference = max(float(mixture_mag.max()), 1e-12)

    # ASO retains mixture phase and jointly allocates mixture magnitude. This
    # ratio recovers the selected note's displayed allocation mask.
    allocation = np.clip(estimate_mag / np.maximum(mixture_mag, reference * 1e-6), 0.0, 1.0)
    mixture_db = 20.0 * np.log10(mixture_mag / reference + 1e-8)
    audible_energy = np.clip((mixture_db + 72.0) / 54.0, 0.0, 1.0)
    strength = np.power(allocation, 0.42) * np.power(audible_energy, 0.55)

    low = np.array([0.08, 0.72, 1.00])[:, None, None]
    high = np.array([0.06, 1.00, 0.76])[:, None, None]
    rgb = low * (1.0 - allocation[None, :, :]) + high * allocation[None, :, :]
    alpha = np.clip(1.18 * strength, 0.0, 0.96)[None, :, :]
    return np.ascontiguousarray(np.moveaxis(np.concatenate((rgb, alpha), axis=0), 0, -1))


def main() -> None:
    manifest_path = ASSETS / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    count = 0
    for example in manifest["examples"]:
        mixture_path = ROOT / example["channels"][0]["audio"]
        mix_rate, mixture = read_audio(mixture_path)
        for note in example["notes"]:
            aso = note["outputs"]["aso"]
            aso_path = ROOT / aso["audio"]
            rate, estimate = read_audio(aso_path)
            if rate != mix_rate or len(estimate) != len(mixture):
                raise ValueError(f"unaligned audio: {aso_path}")
            output_path = aso_path.with_name("aso_mask.png")
            plt.imsave(output_path, rgba_overlay(mixture, estimate), origin="lower")
            aso["mask"] = output_path.relative_to(ROOT).as_posix()
            count += 1

    manifest["maskVisualization"] = {
        "source": "ASO output magnitude divided by mixture magnitude",
        "display": "allocation weighted by audible mixture energy",
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {count} ASO magnitude-allocation overlays")


if __name__ == "__main__":
    main()
