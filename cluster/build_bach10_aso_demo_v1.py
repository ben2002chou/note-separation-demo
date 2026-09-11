#!/usr/bin/env python3
"""Build one Bach10 note-level browser source with the finalized ASO model."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import sys
import zipfile
from pathlib import Path

import numpy as np
from scipy.signal import resample_poly
import soundfile as sf
import torch


PARTS = ("violin", "clarinet", "saxphone", "bassoon")
MODEL_RATE = 16_000
SOURCE_RATE = 44_100
STAGE_RATE = 24_000
RELEASE = 1_707
SILENCE = np.zeros(round(0.02 * MODEL_RATE), dtype=np.float32)
BEST_SHA = "3e8a81d98c76306caf99486c556597f4614dc303c8012a36678b4679e33e7496"
GATED_SHA = "010c5335d6a7df1d2b5fd1dd70bef3f2bc0e23fdcfaec6415da045b0438f7015"
ASO_SHA = "110ce072c6e16317af685fcec0cdb20c75b848da8ba22400b2fcbbd8e8f356b5"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def fit(values: np.ndarray, samples: int) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32).reshape(-1)
    if len(values) < samples:
        values = np.pad(values, (0, samples - len(values)))
    return np.ascontiguousarray(values[:samples])


def resample(values: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
    divisor = math.gcd(source_rate, target_rate)
    return np.ascontiguousarray(
        resample_poly(values, target_rate // divisor, source_rate // divisor), dtype=np.float32
    )


def read_mono(path: Path, expected_rate: int | None = None) -> tuple[np.ndarray, int]:
    values, rate = sf.read(path, dtype="float32", always_2d=True)
    if values.shape[1] != 1 or not np.isfinite(values).all():
        raise ValueError(f"invalid mono audio: {path}")
    if expected_rate is not None and int(rate) != expected_rate:
        raise ValueError(f"sample-rate mismatch: {path}: {rate}")
    return np.ascontiguousarray(values[:, 0]), int(rate)


def note_endpoint(events: list[dict], event: dict, samples: int) -> int:
    onset = round(event["onset_seconds"] * MODEL_RATE)
    annotated = round(event["offset_seconds"] * MODEL_RATE)
    later = [
        round(other["onset_seconds"] * MODEL_RATE)
        for other in events
        if other["part"] == event["part"] and other["onset_seconds"] > event["onset_seconds"]
    ]
    return min(samples, annotated + RELEASE, min(later) if later else samples)


def window_part(values: np.ndarray, event: dict, events: list[dict], crop_start: int, crop_samples: int) -> np.ndarray:
    output = np.zeros(crop_samples, dtype=np.float32)
    onset = round(event["onset_seconds"] * MODEL_RATE)
    endpoint = note_endpoint(events, event, len(values))
    left = max(onset, crop_start)
    right = min(endpoint, crop_start + crop_samples)
    if right > left:
        output[left - crop_start:right - crop_start] = values[left:right]
        fade = min(320, right - left)
        if fade:
            phase = np.linspace(0.0, np.pi, fade, endpoint=True)
            output[right - crop_start - fade:right - crop_start] *= (
                0.5 * (1.0 + np.cos(phase))
            ).astype(np.float32)
    return output


def si_sdr(target: np.ndarray, estimate: np.ndarray) -> float:
    target = np.asarray(target, dtype=np.float64)
    estimate = np.asarray(estimate, dtype=np.float64)
    denom = float(np.dot(target, target))
    if denom <= 1e-12:
        return float("-inf")
    projection = target * (float(np.dot(estimate, target)) / denom)
    residual = estimate - projection
    return float(10.0 * np.log10((np.dot(projection, projection) + 1e-12) /
                                 (np.dot(residual, residual) + 1e-12)))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dna-root", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--piece", default="07-HerrGott")
    parser.add_argument("--best-checkpoint", type=Path, required=True)
    parser.add_argument("--gated-checkpoint", type=Path, required=True)
    parser.add_argument("--aso-authority", type=Path, required=True)
    parser.add_argument("--nmf-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--query-microbatch", type=int, default=4)
    args = parser.parse_args()

    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    if sha256(args.best_checkpoint) != BEST_SHA or sha256(args.gated_checkpoint) != GATED_SHA:
        raise ValueError("separator checkpoint hash mismatch")
    authority = json.loads(args.aso_authority.read_text())
    if authority["checkpoint"]["sha256"] != ASO_SHA or sha256(Path(authority["checkpoint"]["path"])) != ASO_SHA:
        raise ValueError("ASO checkpoint hash mismatch")

    sys.path[:0] = [str(args.dna_root), str(args.dna_root / "src")]
    from scripts.evaluate_bach10_dna_instrument_v1 import load_base, load_events
    from scripts.evaluate_occurrence_joint_tfc_tdf_separator import infer_dual_hpss
    from scripts.evaluate_selected_aso_scns_eval_v2_v1 import load_aso
    from scripts.train_occurrence_score_remix_ssl import load_dual_hpss_checkpoint
    from scripts.train_occurrence_atomic_merger_canary import teacher_evidence
    from scripts.evaluate_adaptive_anchor_blend_v1 import stft, istft

    base = load_base()
    device = torch.device("cuda")
    piece_root = args.dataset_root / args.piece
    mixture44, _ = read_mono(piece_root / f"{args.piece}.wav", SOURCE_RATE)
    mixture = resample(mixture44, SOURCE_RATE, MODEL_RATE)
    events = load_events(piece_root)
    stems = {}
    nmf = {}
    for part in PARTS:
        values, _ = read_mono(piece_root / f"{args.piece}-{part}.wav", SOURCE_RATE)
        stems[part] = fit(resample(values, SOURCE_RATE, MODEL_RATE), len(mixture))
        values, rate = read_mono(args.nmf_root / f"{args.piece}-{part}.wav")
        nmf[part] = fit(resample(values, rate, MODEL_RATE), len(mixture))

    inference_rows = []
    prepared = []
    for event in events:
        crop_start, crop_stop = base.crop_geometry(len(mixture), event)
        crop = np.ascontiguousarray(mixture[crop_start:crop_stop])
        onset = round(event["onset_seconds"] * MODEL_RATE) - crop_start
        offset = note_endpoint(events, event, len(mixture)) - crop_start
        inference_rows.append({
            "mixture": crop, "dna": np.zeros_like(crop), "pitch": event["pitch"],
            "valid_samples": len(crop), "onset_sample": max(0, min(len(crop) - 1, onset)),
            "offset_sample": max(max(0, min(len(crop) - 1, onset)) + 1, min(len(crop), offset)),
        })
        prepared.append({"event": event, "crop": crop, "crop_start": crop_start})

    best, best_info = load_dual_hpss_checkpoint(args.best_checkpoint, device)
    gated, gated_info = load_dual_hpss_checkpoint(args.gated_checkpoint, device)
    aso, aso_info = load_aso(args.aso_authority, device)
    if best_info["sha256"] != BEST_SHA or gated_info["sha256"] != GATED_SHA:
        raise ValueError("loaded separator identity mismatch")
    gated_notes = infer_dual_hpss(gated, inference_rows, device=device, batch_size=args.batch_size)

    methods = {name: [] for name in ("aso", "symmetric", "hpss", "nmf")}
    targets = []
    rows = []
    for number, item in enumerate(prepared, 1):
        event = item["event"]
        crop = item["crop"]
        crop_start = item["crop_start"]
        crop_stop = crop_start + len(crop)
        context = []
        for other in events:
            onset = round(other["onset_seconds"] * MODEL_RATE)
            endpoint = note_endpoint(events, other, len(mixture))
            if endpoint > crop_start and onset < crop_stop:
                context.append({
                    "event_index": other["event_index"], "pitch": other["pitch"],
                    "onset": max(0, min(len(crop), onset - crop_start)),
                    "offset": max(1, min(len(crop), endpoint - crop_start)),
                })
        positions = {row["event_index"]: index for index, row in enumerate(context)}
        bank = {
            "mixture": torch.from_numpy(crop).to(device)[None],
            "valid": torch.tensor([len(crop)], dtype=torch.long, device=device),
            "assignment": torch.zeros(len(context), dtype=torch.long, device=device),
            "pitch": torch.tensor([row["pitch"] for row in context], dtype=torch.long, device=device),
            "onset": torch.tensor([row["onset"] for row in context], dtype=torch.long, device=device),
            "offset": torch.tensor([row["offset"] for row in context], dtype=torch.long, device=device),
        }
        with torch.inference_mode():
            raw_wave, _, _ = teacher_evidence(best, bank, query_microbatch=args.query_microbatch)
            mixture_spec = stft(bank["mixture"][0])
            raw_spec = stft(raw_wave)
            total = raw_spec.abs().sum(0, keepdim=True)
            shares = torch.where(total > 1e-20, raw_spec.abs() / total.clamp_min(1e-20), torch.zeros_like(raw_spec.abs()))
            magnitude = shares * mixture_spec[None]
            if len(context) == 1:
                aso_wave = istft(magnitude, len(crop)).float()
                closure = float((magnitude.sum(0) - mixture_spec).abs().amax())
            else:
                aso_spec, diagnostics = aso(
                    mixture_spec[None], magnitude[None], raw_spec[None], bank["pitch"][None],
                    onsets=bank["onset"][None], acoustic_offsets=bank["offset"][None],
                )
                aso_wave = istft(aso_spec[0], len(crop)).float()
                closure = float(diagnostics.get("sum_error_max_abs", diagnostics.get("magnitude_closure_max_abs")))
        position = positions[event["event_index"]]
        target = window_part(stems[event["part"]], event, events, crop_start, len(crop))
        nmf_note = window_part(nmf[event["part"]], event, events, crop_start, len(crop))
        selected = {
            "aso": np.ascontiguousarray(aso_wave[position].cpu().numpy(), dtype=np.float32),
            "symmetric": np.ascontiguousarray(raw_wave[position].cpu().numpy(), dtype=np.float32),
            "hpss": fit(gated_notes[event["event_index"]], len(crop)),
            "nmf": nmf_note,
        }
        targets.append(target)
        for name, values in selected.items():
            methods[name].append(values)
        active_at_onset = sum(
            other["onset_seconds"] <= event["onset_seconds"] < other["offset_seconds"]
            for other in events
        )
        rows.append({
            "event_index": event["event_index"], "part": event["part"], "pitch": event["pitch"],
            "onset_seconds": event["onset_seconds"], "offset_seconds": event["offset_seconds"],
            "crop_start16": crop_start, "crop_samples16": len(crop), "context_notes": len(context),
            "active_at_onset": active_at_onset, "aso_closure_max_abs": closure,
            "si_sdr_db": {name: si_sdr(target, values) for name, values in selected.items()},
        })
        if number % 20 == 0:
            print(json.dumps({"completed": number, "total": len(prepared)}), flush=True)

    args.output_dir.mkdir(parents=True)
    sf.write(args.output_dir / "mixture.wav", resample(mixture, MODEL_RATE, STAGE_RATE), STAGE_RATE, subtype="FLOAT")
    for name, notes in methods.items():
        sf.write(args.output_dir / f"{name}_sprite.wav", np.concatenate([x for pair in zip(notes, [SILENCE] * len(notes)) for x in pair]), MODEL_RATE, subtype="FLOAT")

    metadata_events = []
    occurrence_buffer = io.BytesIO()
    with zipfile.ZipFile(occurrence_buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for event, target in zip(events, targets, strict=True):
            onset16 = round(event["onset_seconds"] * MODEL_RATE)
            endpoint16 = note_endpoint(events, event, len(mixture))
            occurrence = stems[event["part"]][onset16:endpoint16]
            occurrence24 = resample(occurrence, MODEL_RATE, STAGE_RATE)
            member = f"event_{event['event_index']:04d}.wav"
            wav = io.BytesIO()
            sf.write(wav, occurrence24, STAGE_RATE, format="WAV", subtype="FLOAT")
            archive.writestr(member, wav.getvalue())
            metadata_events.append({
                **event, "onset_sample": round(event["onset_seconds"] * STAGE_RATE),
                "acoustic_samples": len(occurrence24), "occurrence_member": member,
            })
    (args.output_dir / "occurrences.zip").write_bytes(occurrence_buffer.getvalue())
    (args.output_dir / "piece.json").write_text(json.dumps({
        "piece_id": args.piece, "dataset": "Bach10 v1.1 held-out quartet",
        "reference_note_definition": "score-windowed isolated monophonic part, capped at next onset",
        "events": metadata_events,
    }, indent=2) + "\n")
    (args.output_dir / "result.json").write_text(json.dumps({
        "schema": "bach10_aso_demo_source_v1", "status": "passed", "piece": args.piece,
        "events": len(events), "rows": rows,
        "provenance": {
            "aso_authority": str(args.aso_authority), "aso_checkpoint_sha256": ASO_SHA,
            "best_checkpoint_sha256": BEST_SHA, "gated_checkpoint_sha256": GATED_SHA,
            "dataset_root": str(args.dataset_root), "nmf_root": str(args.nmf_root),
            "script_sha256": sha256(Path(__file__)), "aso_contract": aso_info,
        },
    }, indent=2) + "\n")
    print(json.dumps({"status": "passed", "output": str(args.output_dir), "events": len(events)}))


if __name__ == "__main__":
    main()
