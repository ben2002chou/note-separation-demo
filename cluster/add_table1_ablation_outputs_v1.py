#!/usr/bin/env python3
"""Add the four Table 1 separator outputs to final-ASO demo pieces."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import shutil
import sys
import zipfile
from pathlib import Path

import numpy as np
import soundfile as sf
import torch


RATE = 16_000
SILENCE = np.zeros(round(0.02 * RATE), dtype=np.float32)
ARMS = (
    "independent_ungated",
    "independent_selective",
    "symmetric_ungated",
    "symmetric_selective",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def sprite(values: list[np.ndarray]) -> np.ndarray:
    return np.concatenate(
        [item for pair in zip(values, [SILENCE] * len(values), strict=True) for item in pair]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--frozen-metrics", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--piece", action="append", required=True)
    parser.add_argument(
        "--arm-checkpoint", action="append", nargs=3, metavar=("ARM", "PATH", "SHA256"), required=True
    )
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()

    if args.output_root.exists():
        raise FileExistsError(args.output_root)
    if sha256(args.manifest) != args.manifest_sha256:
        raise ValueError("manifest hash mismatch")
    checkpoints = {arm: (Path(path), digest) for arm, path, digest in args.arm_checkpoint}
    if set(checkpoints) != set(ARMS):
        raise ValueError(f"expected checkpoints for {ARMS}, received {sorted(checkpoints)}")
    for arm, (path, digest) in checkpoints.items():
        if sha256(path) != digest:
            raise ValueError(f"{arm}: checkpoint hash mismatch")

    sys.path[:0] = [str(args.snapshot), str(args.snapshot / "src"), str(args.snapshot / "scripts")]
    from scripts import evaluate_scns_eval_v3_factorial_arm_v1 as factorial
    from scripts import evaluate_occurrence_joint_tfc_tdf_separator as occurrence

    manifest = json.loads(args.manifest.read_text())
    metrics = factorial.load_module(args.frozen_metrics)
    by_piece: dict[str, list[dict]] = {}
    for row in manifest["rows"]:
        by_piece.setdefault(row["piece_id"], []).append(row)
    piece_lookup = {piece["piece_id"]: piece for piece in manifest["pieces"]}

    prepared: dict[str, dict] = {}
    for piece_id in args.piece:
        rows = by_piece[piece_id]
        source = Path(piece_lookup[piece_id]["root"])
        metadata = json.loads((source / "piece.json").read_text())
        mixture24, rate = metrics.read_audio(source / "mixture.wav")
        if rate != 24_000:
            raise ValueError(f"{piece_id}: source-rate mismatch")
        examples = []
        event_order = []
        with zipfile.ZipFile(source / "occurrences.zip") as archive:
            for row in rows:
                start, stop = int(row["crop_start_24k"]), int(row["crop_stop_24k"])
                mixture16 = metrics.to_16k(mixture24[start:stop], 24_000)
                crop_start16 = round(start * RATE / 24_000)
                onset16 = round(int(row["onset_sample_24k"]) * RATE / 24_000) - crop_start16
                offset16 = round(int(row["conditioning_offset_sample_24k"]) * RATE / 24_000) - crop_start16
                samples = len(mixture16)
                onset16 = max(0, min(samples - 1, onset16))
                offset16 = max(onset16 + 1, min(samples, offset16))
                examples.append({
                    "mixture": mixture16,
                    "dna": np.zeros(samples, np.float32),
                    "pitch": int(row["pitch"]),
                    "valid_samples": samples,
                    "onset_sample": onset16,
                    "offset_sample": offset16,
                })
                event_order.append(int(row["event_index"]))
        metadata_order = [
            int(event["event_index"])
            for event in metadata["events"]
            if int(event["event_index"]) in set(event_order)
        ]
        if metadata_order != event_order:
            raise ValueError(f"{piece_id}: manifest and metadata event orders differ")
        prepared[piece_id] = {"rows": rows, "examples": examples, "event_order": event_order}

    estimates: dict[str, dict[str, list[np.ndarray]]] = {piece: {} for piece in args.piece}
    provenance = {}
    device = torch.device("cuda")
    for arm in ARMS:
        checkpoint, digest = checkpoints[arm]
        model, info = occurrence.load_dual_hpss_checkpoint(checkpoint, device)
        contract = factorial.validate_checkpoint(info, arm)
        provenance[arm] = {
            "checkpoint": str(checkpoint), "checkpoint_sha256": digest, "contract": contract
        }
        for piece_id in args.piece:
            bundle = prepared[piece_id]
            strengths = [
                factorial.selective_strength(row, bundle["rows"]) for row in bundle["rows"]
            ] if "selective" in arm else None
            values = occurrence.infer_dual_hpss(
                model,
                bundle["examples"],
                device=device,
                batch_size=args.batch_size,
                input_gate_strengths=strengths,
            )
            estimates[piece_id][arm] = [np.asarray(value, dtype=np.float32) for value in values]
        del model
        torch.cuda.empty_cache()
        print(json.dumps({"arm": arm, "status": "rendered"}), flush=True)

    args.output_root.mkdir(parents=True)
    for piece_id in args.piece:
        source = args.source_root / piece_id
        output = args.output_root / piece_id
        shutil.copytree(source, output)
        for arm in ARMS:
            sf.write(
                output / f"{arm}_sprite.wav",
                sprite(estimates[piece_id][arm]),
                RATE,
                subtype="FLOAT",
            )
        result_path = output / "result.json"
        result = json.loads(result_path.read_text())
        result["table1_ablation_provenance"] = provenance
        result_path.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps({"piece": piece_id, "status": "passed"}), flush=True)


if __name__ == "__main__":
    main()
