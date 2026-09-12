#!/usr/bin/env python3
"""Add the selected Bach10 Herr Gott quartet using the final ASO outputs."""

from __future__ import annotations

import argparse
import io
import json
import shutil
import zipfile
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.io import wavfile
from scipy.signal import resample_poly

from build_assets import (
    OUTPUT,
    RATE,
    SOURCE_RATE,
    crop_geometry,
    pcm16,
    place_on_display,
    stft_magnitude,
    write_midi,
    write_spectrogram,
)


EXAMPLE_ID = "orchestra"
PIECE_ID = "07-HerrGott"
TARGET_EVENT = 185
EXPECTED_SEPARATOR = "a3535f3ad99a297967dd18a4c4b8a9a8f7d47f3c6b05f72c0e2449d8c4be2d68"
EXPECTED_ASO = "ff89599cef65e5eb98ff061372d8362e0752d2060d167aaf553b24c29577ae2a"
SPRITES = {
    "aso": "aso_sprite.wav",
    "symmetric_selective": "symmetric_sprite.wav",
    "score_informed_nmf": "nmf_sprite.wav",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    args = parser.parse_args()

    metadata = json.loads((args.source / "piece.json").read_text())
    result = json.loads((args.source / "result.json").read_text())
    provenance = result["provenance"]
    if metadata.get("piece_id") != PIECE_ID or result.get("piece") != PIECE_ID:
        raise ValueError("unexpected Bach10 source")
    if provenance.get("separator_checkpoint_sha256") != EXPECTED_SEPARATOR:
        raise ValueError("separator checkpoint mismatch")
    if provenance.get("aso_checkpoint_sha256") != EXPECTED_ASO:
        raise ValueError("ASO checkpoint mismatch")

    events = metadata["events"]
    event_by_id = {int(event["event_index"]): event for event in events}
    row_by_id = {int(row["event_index"]): row for row in result["rows"]}
    if set(event_by_id) != set(row_by_id) or TARGET_EVENT not in event_by_id:
        raise ValueError("selected event metadata is incomplete")

    mixture24, mixture_rate = sf.read(args.source / "mixture.wav", dtype="float32")
    if int(mixture_rate) != SOURCE_RATE or np.asarray(mixture24).ndim != 1:
        raise ValueError("unexpected mixture format")
    target_event = event_by_id[TARGET_EVENT]
    crop_start24, crop_stop24 = crop_geometry(len(mixture24), target_event)
    mixture = resample_poly(mixture24[crop_start24:crop_stop24], 2, 3).astype(np.float32)
    samples = len(mixture)
    duration = samples / RATE
    display_start16 = round(crop_start24 * RATE / SOURCE_RATE)
    crop_start_seconds = crop_start24 / SOURCE_RATE

    sprites = {}
    for channel, filename in SPRITES.items():
        signal, rate = sf.read(args.source / filename, dtype="float32")
        if int(rate) != RATE:
            raise ValueError(f"{filename}: unexpected sample rate")
        sprites[channel] = np.asarray(signal, dtype=np.float32)

    estimates: dict[int, dict[str, np.ndarray]] = {}
    cursor = 0
    for event in events:
        event_id = int(event["event_index"])
        query_samples = int(row_by_id[event_id]["crop_samples16"])
        estimates[event_id] = {
            channel: np.asarray(sprite[cursor:cursor + query_samples], dtype=np.float32)
            for channel, sprite in sprites.items()
        }
        cursor += query_samples + round(0.02 * RATE)
    if any(cursor != len(sprite) for sprite in sprites.values()):
        raise ValueError("sprite layout mismatch")

    output = OUTPUT / EXAMPLE_ID
    if output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True)
    reference = float(stft_magnitude(mixture).max())
    wavfile.write(output / "mixture.wav", RATE, pcm16(mixture))
    write_spectrogram(output / "mixture.png", mixture, RATE, reference)

    notes = []
    with zipfile.ZipFile(args.source / "occurrences.zip") as archive:
        for event in events:
            event_id = int(event["event_index"])
            onset_seconds = float(event["onset_seconds"])
            acoustic_seconds = int(event["acoustic_samples"]) / SOURCE_RATE
            start = max(0.0, onset_seconds - crop_start_seconds)
            end = min(duration, onset_seconds + acoustic_seconds - crop_start_seconds)
            if end <= start:
                continue
            query_start16 = int(row_by_id[event_id]["crop_start16"])
            raw = archive.read(event["occurrence_member"])
            target24, rate = sf.read(io.BytesIO(raw), dtype="float32")
            if int(rate) != SOURCE_RATE:
                raise ValueError("unexpected occurrence sample rate")
            target = resample_poly(target24, 2, 3).astype(np.float32)
            signals = {
                "target": place_on_display(
                    target, round(onset_seconds * RATE), display_start16, samples
                ),
                **{
                    channel: place_on_display(values, query_start16, display_start16, samples)
                    for channel, values in estimates[event_id].items()
                },
            }
            query = output / f"q{event_id:04d}"
            query.mkdir()
            outputs = {}
            for channel, signal in signals.items():
                wavfile.write(query / f"{channel}.wav", RATE, pcm16(signal))
                write_spectrogram(query / f"{channel}.png", signal, RATE, reference)
                outputs[channel] = {
                    "audio": f"assets/{EXAMPLE_ID}/q{event_id:04d}/{channel}.wav",
                    "spectrogram": f"assets/{EXAMPLE_ID}/q{event_id:04d}/{channel}.png",
                }
            notes.append({
                "event": event_id,
                "pitch": int(event["pitch"]),
                "start": round(start, 4),
                "end": round(end, 4),
                "target": event_id == TARGET_EVENT,
                "outputs": outputs,
            })

    write_midi(output / "score.mid", notes, 71)
    default_note = next(note for note in notes if note["target"])
    labels = {
        "target": "Aligned isolated part",
        "aso": "Symmetric Gated + ASO",
        "symmetric_selective": "Symmetric Gated",
        "score_informed_nmf": "Score-Informed NMF",
    }
    channels = [{
        "id": "mixture",
        "label": "Original four-part mixture",
        "audio": f"assets/{EXAMPLE_ID}/mixture.wav",
        "spectrogram": f"assets/{EXAMPLE_ID}/mixture.png",
    }]
    channels.extend(
        {"id": channel, "label": labels[channel], **default_note["outputs"][channel]}
        for channel in ("target", "aso", "symmetric_selective", "score_informed_nmf")
    )
    example = {
        "id": EXAMPLE_ID,
        "title": "Herr Gott, Bach chorale quartet",
        "instrument": "Four-part orchestral ensemble",
        "pieceId": PIECE_ID,
        "dataset": "Bach10 v1.1 held-out quartet",
        "requestId": f"{PIECE_ID}:{TARGET_EVENT:04d}",
        "duration": round(duration, 6),
        "targetPitch": int(target_event["pitch"]),
        "targetEvent": TARGET_EVENT,
        "notes": notes,
        "channels": channels,
        "midi": f"assets/{EXAMPLE_ID}/score.mid",
        "auditionGainCapDb": 15.56,
        "provenance": {
            "separatorCheckpointSha256": EXPECTED_SEPARATOR,
            "asoCheckpointSha256": EXPECTED_ASO,
            "reference": metadata.get("reference_note_definition"),
        },
    }
    manifest_path = OUTPUT / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["datasets"] = list(dict.fromkeys([
        *manifest.get("datasets", []), "Bach10 v1.1 held-out quartet",
    ]))
    manifest["examples"] = [
        row for row in manifest["examples"]
        if row["id"] not in {"guitar", "cello", EXAMPLE_ID}
    ]
    manifest["examples"].append(example)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"status": "passed", "example": EXAMPLE_ID, "notes": len(notes)}))


if __name__ == "__main__":
    main()
