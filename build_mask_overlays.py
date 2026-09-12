#!/usr/bin/env python3
"""Build transparent ASO magnitude-allocation overlays for the static demo."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image
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


def rgba_overlay(
    mixture: np.ndarray,
    estimate: np.ndarray,
    mixture_image: np.ndarray,
) -> np.ndarray:
    mixture_mag = stft_magnitude(mixture)
    estimate_mag = stft_magnitude(estimate)
    reference = max(float(mixture_mag.max()), 1e-12)

    # ASO retains mixture phase and jointly allocates mixture magnitude. This
    # ratio recovers the selected note's displayed allocation mask.
    allocation = np.clip(estimate_mag / np.maximum(mixture_mag, reference * 1e-6), 0.0, 1.0)
    mixture_db = 20.0 * np.log10(mixture_mag / reference + 1e-8)
    audible_energy = np.clip((mixture_db + 72.0) / 54.0, 0.0, 1.0)
    strength = np.power(allocation, 0.42) * np.power(audible_energy, 0.55)

    # Convert the STFT mask to the exact pixel grid used by the displayed
    # mixture. Frequency bin zero is the bottom row of the spectrogram.
    raw_alpha = np.flipud(np.clip(1.18 * strength, 0.0, 0.96))
    height, width = mixture_image.shape[:2]
    alpha = np.asarray(
        Image.fromarray(np.round(raw_alpha * 255).astype(np.uint8)).resize(
            (width, height), Image.Resampling.BILINEAR
        ),
        dtype=np.float64,
    ) / 255.0

    # Recolor the original mixture pixels instead of drawing a second
    # spectrogram. Highlighted ridges therefore remain pixel-aligned with the
    # visible mixture while the cyan-green tint indicates ASO ownership.
    base = mixture_image[..., :3].astype(np.float64) / 255.0
    rgb = base.copy()
    rgb[..., 0] *= 0.52
    rgb[..., 1] += 0.72 * (1.0 - rgb[..., 1])
    rgb[..., 2] += 0.48 * (1.0 - rgb[..., 2])
    return np.dstack((np.clip(rgb, 0.0, 1.0), alpha))


def main() -> None:
    manifest_path = ASSETS / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    count = 0
    for example in manifest["examples"]:
        mixture_path = ROOT / example["channels"][0]["audio"]
        mixture_image_path = ROOT / example["channels"][0]["spectrogram"]
        mix_rate, mixture = read_audio(mixture_path)
        mixture_image = np.asarray(Image.open(mixture_image_path).convert("RGB"))
        for note in example["notes"]:
            aso = note["outputs"]["aso"]
            aso_path = ROOT / aso["audio"]
            rate, estimate = read_audio(aso_path)
            if rate != mix_rate or len(estimate) != len(mixture):
                raise ValueError(f"unaligned audio: {aso_path}")
            output_path = aso_path.with_name("aso_mask.png")
            overlay = np.round(rgba_overlay(mixture, estimate, mixture_image) * 255).astype(np.uint8)
            Image.fromarray(overlay, mode="RGBA").save(output_path)
            aso["mask"] = output_path.relative_to(ROOT).as_posix()
            count += 1

    manifest["maskVisualization"] = {
        "source": "ASO output magnitude divided by mixture magnitude",
        "display": "original mixture pixels recolored by ASO allocation",
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {count} ASO magnitude-allocation overlays")


if __name__ == "__main__":
    main()
