#!/usr/bin/env python3
"""Build static demo assets from frozen SCNS-Eval-v3 files and output sprites."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import struct
import zipfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
from scipy.io import wavfile
from scipy.signal import resample_poly


OUTPUT = Path(__file__).resolve().parent / "assets"
SOURCE_RATE = 24_000
RATE = 16_000
LEAD_24K = 4_000
RELEASE_24K = 2_560
MINIMUM_24K = 16_384
QUANTUM_24K = 4_096

EXAMPLES = (
    {
        "id": "piano", "title": "Repeated-note passage", "instrument": "Piano",
        "piece_id": "scns_eval_v3_101", "target_event": 36,
        "program": 0,
        "aso_sha256": "8c40378084328d6311f599ead86081bb01090862db18c03ce806384a35c864d7",
        "aso_si_sdr_db": 16.0444,
        "raw_si_sdr_db": 7.5991,
    },
    {
        "id": "guitar", "title": "Repeated guitar phrase", "instrument": "Acoustic guitar",
        "piece_id": "scns_eval_v3_003", "target_event": 14,
        "program": 24,
        "aso_sha256": "bd78e8eb18c208c521a9e870100a4cd7fd5e6f243806f06ee78c97ba40ebc980",
        "aso_si_sdr_db": 12.6392,
        "raw_si_sdr_db": 2.5292,
    },
    {
        "id": "cello", "title": "Sustained cello phrase", "instrument": "Cello",
        "piece_id": "scns_eval_v3_028", "target_event": 12,
        "program": 42,
        "aso_sha256": "ba232e6b2d3c6bd90b0c8f0b9f3861f4f212e2e436b1a380afd77e2e7db973f9",
        "aso_si_sdr_db": 18.9579,
        "raw_si_sdr_db": 3.6683,
    },
)

CHANNELS = (
    ("mixture", "Original mixture"),
    ("target", "Isolated target"),
    ("independent_ungated", "Independent Ungated"),
    ("independent_selective", "Independent Selective"),
    ("symmetric_ungated", "Symmetric Ungated"),
    ("symmetric_selective", "Symmetric Gated"),
    ("aso", "Symmetric Gated + ASO"),
)


def crop_geometry(total_samples: int, event: dict) -> tuple[int, int]:
    onset = int(event["onset_sample"])
    acoustic = int(event["acoustic_samples"])
    left = max(0, onset - LEAD_24K)
    right = min(total_samples, onset + acoustic + RELEASE_24K)
    desired = max(MINIMUM_24K, right - left)
    desired = min(total_samples, int(np.ceil(desired / QUANTUM_24K) * QUANTUM_24K))
    start_min = max(0, right - desired)
    start_max = min(left, total_samples - desired)
    start = max(0, min((start_min + start_max) // 2, total_samples - desired))
    return start, start + desired


def fit(signal: np.ndarray, samples: int) -> np.ndarray:
    signal = np.asarray(signal, dtype=np.float32).reshape(-1)
    return signal[:samples] if len(signal) >= samples else np.pad(signal, (0, samples - len(signal)))


def stft_magnitude(signal: np.ndarray) -> np.ndarray:
    n_fft, hop = 1024, 128
    window = np.hanning(n_fft + 1)[:-1]
    padded = np.pad(signal.astype(np.float64), (n_fft // 2, n_fft // 2))
    frames = np.lib.stride_tricks.sliding_window_view(padded, n_fft)[::hop]
    return np.abs(np.fft.rfft(frames * window, axis=1)).T


def write_spectrogram(path: Path, signal: np.ndarray, rate: int, reference: float) -> None:
    magnitude = stft_magnitude(signal)
    db = 20.0 * np.log10(magnitude / max(reference, 1e-12) + 1e-8)
    fig, axis = plt.subplots(figsize=(12, 3), dpi=120)
    axis.imshow(
        db, origin="lower", aspect="auto", interpolation="nearest",
        extent=(0, len(signal) / rate, 0, rate / 2000),
        cmap="magma", vmin=-80, vmax=0,
    )
    axis.set_xlim(0, len(signal) / rate)
    axis.set_ylim(0, 8)
    axis.set_axis_off()
    fig.subplots_adjust(0, 0, 1, 1)
    fig.savefig(path, bbox_inches=None, pad_inches=0)
    plt.close(fig)


def synthesize_score(notes: list[dict], samples: int, instrument: str) -> np.ndarray:
    output = np.zeros(samples, dtype=np.float64)
    for note in notes:
        start = max(0, round(note["start"] * RATE))
        stop = min(samples, round(note["end"] * RATE))
        if stop <= start:
            continue
        t = np.arange(stop - start, dtype=np.float64) / RATE
        frequency = 440.0 * 2 ** ((note["pitch"] - 69) / 12)
        tone = sum(weight * np.sin(2 * math.pi * frequency * harmonic * t)
                   for harmonic, weight in enumerate((1.0, 0.36, 0.18, 0.09), start=1))
        attack = np.minimum(1.0, t / 0.008)
        decay = np.exp((-2.2 if instrument == "Piano" else -3.6) * t)
        release = np.minimum(1.0, (stop - start - np.arange(stop - start)) / (0.025 * RATE))
        output[start:stop] += 0.12 * tone * attack * decay * release
    peak = np.max(np.abs(output))
    if peak > 0.72:
        output *= 0.72 / peak
    return output.astype(np.float32)


def variable_length(value: int) -> bytes:
    encoded = bytearray([value & 0x7F])
    while value := value >> 7:
        encoded.insert(0, (value & 0x7F) | 0x80)
    return bytes(encoded)


def write_midi(path: Path, notes: list[dict], program: int) -> None:
    ticks_per_second = 960
    events: list[tuple[int, int, bytes]] = [(0, 0, bytes([0xC0, program]))]
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


def pcm16(signal: np.ndarray) -> np.ndarray:
    return (np.clip(signal, -1, 1) * 32767).astype(np.int16)


def place_on_display(signal: np.ndarray, source_start: int, display_start: int, samples: int) -> np.ndarray:
    output = np.zeros(samples, dtype=np.float32)
    destination_left = source_start - display_start
    source_left = max(0, -destination_left)
    destination_left = max(0, destination_left)
    count = min(len(signal) - source_left, samples - destination_left)
    if count > 0:
        output[destination_left:destination_left + count] = signal[source_left:source_left + count]
    return output


def load_example(source_root: Path, config: dict) -> tuple[dict, np.ndarray, list[dict]]:
    root = source_root / config["piece_id"]
    metadata = json.loads((root / "piece.json").read_text(encoding="utf-8"))
    retained_geometry = {}
    retained_result = root / "result.json"
    if retained_result.exists():
        result = json.loads(retained_result.read_text(encoding="utf-8"))
        retained_geometry = {
            int(row["event_index"]): (int(row["crop_start16"]), int(row["crop_samples16"]))
            for row in result.get("rows", [])
            if "crop_start16" in row and "crop_samples16" in row
        }
    target_event = metadata["events"][config["target_event"]]
    if target_event["event_index"] != config["target_event"]:
        raise ValueError("event index mismatch")
    mixture24, mixture_rate = sf.read(root / "mixture.wav", dtype="float32")
    if mixture_rate != SOURCE_RATE:
        raise ValueError("mixture sample rate mismatch")
    crop_start24, crop_stop24 = crop_geometry(len(mixture24), target_event)
    mixture = resample_poly(mixture24[crop_start24:crop_stop24], 2, 3).astype(np.float32)
    samples = len(mixture)
    display_start16 = round(crop_start24 * RATE / SOURCE_RATE)
    crop_start_seconds = crop_start24 / SOURCE_RATE
    duration = samples / RATE
    sprites = {}
    for key, filename in {
        "aso": "selected_aso_sprite.wav",
        "independent_ungated": "independent_ungated_sprite.wav",
        "independent_selective": "independent_selective_sprite.wav",
        "symmetric_ungated": "symmetric_ungated_sprite.wav",
        "symmetric_selective": "symmetric_selective_sprite.wav",
    }.items():
        sprites[key], sprite_rate = sf.read(root / filename, dtype="float32")
        if sprite_rate != RATE:
            raise ValueError(f"{filename}: sample rate mismatch")

    notes = []
    cursor = 0
    with zipfile.ZipFile(root / "occurrences.zip") as archive:
        for event in metadata["events"]:
            if event["event_index"] in retained_geometry:
                query_start16, query_samples = retained_geometry[event["event_index"]]
            else:
                query_start24, query_stop24 = crop_geometry(len(mixture24), event)
                query_samples = len(resample_poly(mixture24[query_start24:query_stop24], 2, 3))
                query_start16 = round(query_start24 * RATE / SOURCE_RATE)
            event_estimates = {
                key: np.asarray(sprite[cursor:cursor + query_samples], dtype=np.float32)
                for key, sprite in sprites.items()
            }
            cursor += query_samples + round(0.02 * RATE)

            start = max(0.0, event["onset_seconds"] - crop_start_seconds)
            offset_seconds = event.get("offset_seconds", event["offset_sample"] / SOURCE_RATE)
            end = min(duration, offset_seconds - crop_start_seconds)
            if end <= start:
                continue
            raw = archive.read(event["occurrence_member"])
            target_note24, target_rate = sf.read(io.BytesIO(raw), dtype="float32")
            if target_rate != SOURCE_RATE:
                raise ValueError("target sample rate mismatch")
            target_note = resample_poly(target_note24, 2, 3).astype(np.float32)
            target_start16 = round(event["onset_seconds"] * RATE)
            query_signals = {
                "target": place_on_display(target_note, target_start16, display_start16, samples),
                **{
                    key: place_on_display(value, query_start16, display_start16, samples)
                    for key, value in event_estimates.items()
                },
            }
            if event["event_index"] == config["target_event"]:
                aso_sha256 = hashlib.sha256(np.asarray(event_estimates["aso"], dtype="<f4").tobytes()).hexdigest()
                if aso_sha256 != config["aso_sha256"]:
                    raise ValueError(f"{config['piece_id']}: retained ASO waveform hash mismatch")
            notes.append({
                "event": event["event_index"], "pitch": event["pitch"],
                "start": round(start, 4), "end": round(end, 4),
                "target": event["event_index"] == config["target_event"],
                "_signals": query_signals,
            })
    return target_event, mixture, notes


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True,
                        help="Directory containing the staged demo source folders")
    args = parser.parse_args()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema": "note-separation-demo-v2",
        "model": "ASO, final temporal checkpoint step 29,058",
        "datasets": ["SCNS-Eval-v3"],
        "checkpointSha256": "ff89599cef65e5eb98ff061372d8362e0752d2060d167aaf553b24c29577ae2a",
        "evaluationResultSha256": "1fab27ff756bfb3a1757798e31a1a959fd7f52981daa16623cbd0f58a75e67d0",
        "selection": "Curated held-out examples selected for separation quality, audible ASO improvement, and onset or reattack preservation.",
        "examples": [],
    }
    for config in EXAMPLES:
        target_event, mixture, notes = load_example(args.source_root, config)
        samples = len(mixture)
        duration = samples / RATE
        example_dir = OUTPUT / config["id"]
        example_dir.mkdir(parents=True, exist_ok=True)
        reference = float(stft_magnitude(mixture).max())
        wavfile.write(example_dir / "mixture.wav", RATE, pcm16(mixture))
        write_spectrogram(example_dir / "mixture.png", mixture, RATE, reference)
        public_notes = []
        for note in notes:
            query_dir = example_dir / f"q{note['event']:04d}"
            query_dir.mkdir(parents=True, exist_ok=True)
            outputs = {}
            for key, signal in note["_signals"].items():
                wavfile.write(query_dir / f"{key}.wav", RATE, pcm16(signal))
                write_spectrogram(query_dir / f"{key}.png", signal, RATE, reference)
                outputs[key] = {
                    "audio": f"assets/{config['id']}/q{note['event']:04d}/{key}.wav",
                    "spectrogram": f"assets/{config['id']}/q{note['event']:04d}/{key}.png",
                }
            public_notes.append({key: value for key, value in note.items() if key != "_signals"} | {"outputs": outputs})

        midi_audio = synthesize_score(public_notes, samples, config["instrument"])
        wavfile.write(example_dir / "midi.wav", RATE, pcm16(midi_audio))
        write_spectrogram(example_dir / "midi.png", midi_audio, RATE,
                          max(float(stft_magnitude(midi_audio).max()), 1e-6))
        write_midi(example_dir / "score.mid", public_notes, config["program"])
        default_note = next(note for note in public_notes if note["target"])
        labels = dict(CHANNELS)
        channels = [{
            "id": "mixture", "label": labels["mixture"],
            "audio": f"assets/{config['id']}/mixture.wav",
            "spectrogram": f"assets/{config['id']}/mixture.png",
        }]
        channels.extend({"id": key, "label": labels[key], **default_note["outputs"][key]}
                        for key in (
                            "target", "independent_ungated", "independent_selective",
                            "symmetric_ungated", "symmetric_selective", "aso",
                        ))
        channels.append({"id": "midi", "label": "MIDI score",
                         "audio": f"assets/{config['id']}/midi.wav",
                         "spectrogram": f"assets/{config['id']}/midi.png"})
        manifest["examples"].append({
            "id": config["id"], "title": config["title"],
            "instrument": config["instrument"], "pieceId": config["piece_id"],
            "dataset": config.get("dataset", "SCNS-Eval-v3"),
            "requestId": f"{config['piece_id']}:{config['target_event']:04d}",
            "duration": round(duration, 6), "targetPitch": target_event["pitch"],
            "targetEvent": config["target_event"], "notes": public_notes,
            "channels": channels, "midi": f"assets/{config['id']}/score.mid",
            "asoFloatWaveformSha256": config["aso_sha256"],
            "featuredMetrics": {
                "asoSiSdrDb": config["aso_si_sdr_db"],
                "preAsoSiSdrDb": config["raw_si_sdr_db"],
            },
            **({"evidenceResultSha256": config["evidence_result_sha256"]}
               if "evidence_result_sha256" in config else {}),
        })
    (OUTPUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {len(manifest['examples'])} final-model examples to {OUTPUT}")


if __name__ == "__main__":
    main()
