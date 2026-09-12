#!/usr/bin/env python3
"""Create short ASO clips that can play immediately from score-note clicks."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.io import wavfile


ROOT = Path(__file__).resolve().parent
MANIFEST = ROOT / "assets" / "manifest.json"
LEAD_SECONDS = 0.025
RELEASE_SECONDS = 0.250
AUDITION_CHANNELS = (
    "mixture", "target", "independent_ungated", "independent_selective",
    "symmetric_ungated", "symmetric_selective", "aso",
)


def waveform_envelope(signal: np.ndarray, bins: int = 32) -> list[float]:
    signal = np.asarray(signal, dtype=np.float64).reshape(-1)
    edges = np.linspace(0, len(signal), bins + 1, dtype=int)
    values = []
    for left, right in zip(edges[:-1], edges[1:]):
        frame = signal[left:max(left + 1, right)]
        values.append(float(np.sqrt(np.mean(frame * frame))))
    peak = max(values, default=0.0)
    if peak > 0:
        values = [value / peak for value in values]
    return [round(value, 3) for value in values]


def main() -> None:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    count = 0
    for example in manifest["examples"]:
        duration = float(example["duration"])
        mixture_source = ROOT / next(
            channel["audio"] for channel in example["channels"] if channel["id"] == "mixture"
        )
        mask_layers = []
        mask_events = []
        for note in example["notes"]:
            start = max(0.0, float(note["start"]) - LEAD_SECONDS)
            end = min(duration, float(note["end"]) + RELEASE_SECONDS)
            auditions = {}
            aso_clip = None
            for channel_id in AUDITION_CHANNELS:
                source = mixture_source if channel_id == "mixture" else ROOT / note["outputs"][channel_id]["audio"]
                rate, signal = wavfile.read(source)
                left = max(0, round(start * rate))
                right = min(len(signal), round(end * rate))
                clip = np.asarray(signal[left:right])
                output = (ROOT / note["outputs"]["aso"]["audio"]).with_name(f"{channel_id}_audition.wav")
                wavfile.write(output, rate, clip)
                auditions[channel_id] = {
                    "audio": output.relative_to(ROOT).as_posix(),
                    "start": round(start, 4),
                }
                if channel_id == "aso":
                    aso_clip = clip
                    note["outputs"]["aso"]["audition"] = output.relative_to(ROOT).as_posix()
                    note["outputs"]["aso"]["auditionStart"] = round(start, 4)
            note["auditions"] = auditions
            note["waveform"] = waveform_envelope(aso_clip)

            mask_path = ROOT / note["outputs"]["aso"]["mask"]
            mask_layers.append(np.asarray(Image.open(mask_path).convert("RGBA"), dtype=np.uint8)[..., 3])
            mask_events.append(int(note["event"]))
            count += 1

        stack = np.stack(mask_layers, axis=0)
        order = np.argsort(stack, axis=0)
        top_indices = order[-1]
        runner_indices = order[-2]
        top_strength = np.take_along_axis(stack, top_indices[None, ...], axis=0)[0]
        runner_strength = np.take_along_axis(stack, runner_indices[None, ...], axis=0)[0]
        events = np.asarray(mask_events, dtype=np.uint8)
        hit_map = np.empty((*top_strength.shape, 4), dtype=np.uint8)
        hit_map[..., 0] = events[top_indices]
        hit_map[..., 1] = events[runner_indices]
        ambiguity = np.clip(
            15.0 * runner_strength / np.maximum(top_strength, 1), 0, 15,
        ).astype(np.uint8)
        strength = np.clip(top_strength // 16, 0, 15).astype(np.uint8)
        hit_map[..., 2] = (strength << 4) | ambiguity
        hit_map[..., 3] = 255
        hit_map_path = mixture_source.with_name("aso_hit_map.png")
        Image.fromarray(hit_map, mode="RGBA").save(hit_map_path)
        example["hitMap"] = hit_map_path.relative_to(ROOT).as_posix()
        example["hitMapSize"] = [int(hit_map.shape[1]), int(hit_map.shape[0])]

    MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote Table 1 auditions, waveform envelopes, and ownership maps for {count} notes")


if __name__ == "__main__":
    main()
