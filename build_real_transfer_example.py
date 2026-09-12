#!/usr/bin/env python3
"""Add the approved real MAESTRO passage to the static interactive demo."""

from __future__ import annotations

import argparse
import io
import json
import math
import shutil
import struct
import zipfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
from scipy.io import wavfile
from scipy.signal import resample_poly


ROOT = Path(__file__).resolve().parent
ASSETS = ROOT / "assets"
RATE = 16_000
SOURCE_RATE = 24_000
EXAMPLE_ID = "real-piano"


def pcm16(signal: np.ndarray) -> np.ndarray:
    return (np.clip(signal, -1, 1) * 32767).astype(np.int16)


def stft_magnitude(signal: np.ndarray) -> np.ndarray:
    n_fft, hop = 1024, 128
    window = np.hanning(n_fft + 1)[:-1]
    padded = np.pad(np.asarray(signal, dtype=np.float64), (n_fft // 2, n_fft // 2))
    frames = np.lib.stride_tricks.sliding_window_view(padded, n_fft)[::hop]
    return np.abs(np.fft.rfft(frames * window, axis=1)).T


def write_spectrogram(path: Path, signal: np.ndarray) -> None:
    magnitude = stft_magnitude(signal)
    reference = max(float(magnitude.max()), 1e-12)
    db = 20.0 * np.log10(magnitude / reference + 1e-8)
    fig, axis = plt.subplots(figsize=(12, 3), dpi=120)
    axis.imshow(
        db, origin="lower", aspect="auto", interpolation="nearest",
        extent=(0, len(signal) / RATE, 0, RATE / 2000),
        cmap="magma", vmin=-80, vmax=0,
    )
    axis.set_xlim(0, len(signal) / RATE)
    axis.set_ylim(0, 8)
    axis.set_axis_off()
    fig.subplots_adjust(0, 0, 1, 1)
    fig.savefig(path, bbox_inches=None, pad_inches=0)
    plt.close(fig)


def variable_length(value: int) -> bytes:
    encoded = bytearray([value & 0x7F])
    while value := value >> 7:
        encoded.insert(0, (value & 0x7F) | 0x80)
    return bytes(encoded)


def write_midi(path: Path, notes: list[dict]) -> None:
    ticks_per_second = 960
    events: list[tuple[int, int, bytes]] = [(0, 0, bytes([0xC0, 0]))]
    for note in notes:
        start = max(0, round(note["start"] * ticks_per_second))
        stop = max(start + 1, round(note["end"] * ticks_per_second))
        events.append((start, 1, bytes([0x90, note["pitch"], 88])))
        events.append((stop, 0, bytes([0x80, note["pitch"], 0])))
    events.sort(key=lambda item: (item[0], item[1]))
    track = bytearray(variable_length(0) + b"\xff\x51\x03\x07\xa1\x20")
    previous = 0
    for tick, _, payload in events:
        track.extend(variable_length(tick - previous))
        track.extend(payload)
        previous = tick
    track.extend(variable_length(0) + b"\xff\x2f\x00")
    path.write_bytes(
        b"MThd" + struct.pack(">IHHH", 6, 0, 1, 480)
        + b"MTrk" + struct.pack(">I", len(track)) + track
    )


def decode_member(archive: zipfile.ZipFile, member: str, samples: int) -> np.ndarray:
    signal, rate = sf.read(io.BytesIO(archive.read(member)), dtype="float32")
    if int(rate) != SOURCE_RATE:
        raise ValueError(f"unexpected note rate: {rate}")
    signal = resample_poly(np.asarray(signal, dtype=np.float32), 2, 3).astype(np.float32)
    if len(signal) < samples:
        signal = np.pad(signal, (0, samples - len(signal)))
    return np.ascontiguousarray(signal[:samples])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    args = parser.parse_args()
    source_manifest = json.loads((args.source / "manifest.json").read_text())
    if source_manifest.get("title") != "Debussy, Etude Pour les accords":
        raise ValueError("unexpected approved source")
    if source_manifest.get("window", {}).get("start_seconds") != 193.5:
        raise ValueError("expected approved Passage 3")

    mixture24, rate = sf.read(args.source / "mixture.wav", dtype="float32")
    if int(rate) != SOURCE_RATE or np.asarray(mixture24).ndim != 1:
        raise ValueError("unexpected mixture format")
    mixture = resample_poly(mixture24, 2, 3).astype(np.float32)
    samples = len(mixture)
    duration = samples / RATE
    output = ASSETS / EXAMPLE_ID
    if output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True)
    wavfile.write(output / "mixture.wav", RATE, pcm16(mixture))
    write_spectrogram(output / "mixture.png", mixture)

    source_events = source_manifest["events"]
    target_original = int(source_manifest["preview_event_indices"][0])
    notes = []
    with zipfile.ZipFile(args.source / "raw_notes.zip") as raw_archive, \
         zipfile.ZipFile(args.source / "aso_notes.zip") as aso_archive:
        for local_event, event in enumerate(source_events):
            start = max(0.0, min(duration, float(event["relative_onset_seconds"])))
            acoustic_end = float(event["acoustic_offset_seconds"]) - float(source_manifest["window"]["start_seconds"])
            end = max(start + 0.01, min(duration, acoustic_end))
            if start >= duration or end <= 0:
                continue
            query = output / f"q{local_event:04d}"
            query.mkdir()
            raw = decode_member(raw_archive, event["raw_member"], samples)
            aso = decode_member(aso_archive, event["aso_member"], samples)
            wavfile.write(query / "symmetric_selective.wav", RATE, pcm16(raw))
            wavfile.write(query / "aso.wav", RATE, pcm16(aso))
            notes.append({
                "event": local_event, "sourceEvent": int(event["event_index"]),
                "pitch": int(event["pitch"]), "start": round(start, 4),
                "end": round(end, 4), "target": int(event["event_index"]) == target_original,
                "outputs": {
                    "symmetric_selective": {
                        "audio": f"assets/{EXAMPLE_ID}/q{local_event:04d}/symmetric_selective.wav",
                    },
                    "aso": {"audio": f"assets/{EXAMPLE_ID}/q{local_event:04d}/aso.wav"},
                },
            })
    if not any(note["target"] for note in notes):
        raise ValueError("approved target note missing")
    write_midi(output / "score.mid", notes)
    default_note = next(note for note in notes if note["target"])
    channels = [
        {
            "id": "mixture", "label": "Original recording",
            "audio": f"assets/{EXAMPLE_ID}/mixture.wav",
            "spectrogram": f"assets/{EXAMPLE_ID}/mixture.png",
        },
        {
            "id": "aso", "label": "Symmetric Gated + ASO",
            **default_note["outputs"]["aso"],
        },
        {
            "id": "symmetric_selective", "label": "Symmetric Gated",
            **default_note["outputs"]["symmetric_selective"],
        },
    ]
    example = {
        "id": EXAMPLE_ID, "title": "Étude ‘Pour les accords’, Passage 3",
        "instrument": "Real polyphonic piano", "pieceId": "maestro-2009-debussy-accords",
        "dataset": "MAESTRO v3 official test split", "qualitative": True,
        "requestId": f"maestro-2009-debussy-accords:{target_original}",
        "requestLabel": "Qualitative transfer on a real 2009 performance",
        "duration": round(duration, 6), "targetPitch": default_note["pitch"],
        "targetEvent": default_note["event"], "notes": notes, "channels": channels,
        "midi": f"assets/{EXAMPLE_ID}/score.mid", "auditionGainCapDb": 15.56,
        "provenance": {
            "source": source_manifest["source_label"],
            "windowSeconds": source_manifest["window"],
            "scope": "Qualitative transfer only; no isolated per-note reference",
        },
    }
    manifest_path = ASSETS / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["datasets"] = list(dict.fromkeys([*manifest.get("datasets", []), "MAESTRO v3 test"] ))
    manifest["examples"] = [example, *[row for row in manifest["examples"] if row["id"] != EXAMPLE_ID]]
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"status": "passed", "example": EXAMPLE_ID, "notes": len(notes)}))


if __name__ == "__main__":
    main()
