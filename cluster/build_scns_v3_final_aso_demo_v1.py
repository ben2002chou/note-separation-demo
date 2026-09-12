#!/usr/bin/env python3
"""Render selected SCNS-Eval-v3 pieces with the final temporal ASO checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import shutil
import sys
import zipfile
from collections import OrderedDict
from pathlib import Path

import numpy as np
import soundfile as sf
import torch


RATE = 16_000
SILENCE = np.zeros(round(0.02 * RATE), dtype=np.float32)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def sprite(values: list[np.ndarray]) -> np.ndarray:
    return np.concatenate([item for pair in zip(values, [SILENCE] * len(values)) for item in pair])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--frozen-metrics", type=Path, required=True)
    parser.add_argument("--separator-checkpoint", type=Path, required=True)
    parser.add_argument("--separator-sha256", required=True)
    parser.add_argument("--aso-checkpoint", type=Path, required=True)
    parser.add_argument("--aso-sha256", required=True)
    parser.add_argument("--aggregate-result", type=Path, required=True)
    parser.add_argument("--aggregate-sha256", required=True)
    parser.add_argument("--piece", action="append", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--query-microbatch", type=int, default=4)
    args = parser.parse_args()

    if args.output_root.exists():
        raise FileExistsError(args.output_root)
    for path, expected in (
        (args.manifest, args.manifest_sha256),
        (args.separator_checkpoint, args.separator_sha256),
        (args.aso_checkpoint, args.aso_sha256),
        (args.aggregate_result, args.aggregate_sha256),
    ):
        if sha256(path) != expected:
            raise ValueError(f"hash mismatch: {path}")

    sys.path[:0] = [str(args.snapshot), str(args.snapshot / "src"), str(args.snapshot / "scripts")]
    from scripts import evaluate_scns_eval_v3_factorial_arm_v1 as frozen
    from scripts.evaluate_selected_separator_aso_scns_eval_v3_v1 import (
        infer_bank,
        load_aso,
        source_context,
    )
    from scripts.train_occurrence_score_remix_ssl import load_dual_hpss_checkpoint

    manifest = json.loads(args.manifest.read_text())
    aggregate = json.loads(args.aggregate_result.read_text())
    if manifest.get("status") != "frozen" or aggregate.get("status") != "passed":
        raise ValueError("authority is not frozen and passed")
    authority_rows = {row["request_id"]: row for row in aggregate["rows"]}
    metrics = frozen.load_module(args.frozen_metrics)
    device = torch.device("cuda")
    separator, separator_info = load_dual_hpss_checkpoint(args.separator_checkpoint, device)
    aso, aso_info = load_aso(args.aso_checkpoint, args.aso_sha256, device)

    args.output_root.mkdir(parents=True)
    for piece_id in args.piece:
        piece = next(item for item in manifest["pieces"] if item["piece_id"] == piece_id)
        source = Path(piece["root"])
        metadata = json.loads((source / "piece.json").read_text())
        piece_rows = [row for row in manifest["rows"] if row["piece_id"] == piece_id]
        rows_by_event = {int(row["event_index"]): row for row in piece_rows}
        mixture24, source_rate = metrics.read_audio(source / "mixture.wav")
        if source_rate != 24_000:
            raise ValueError("source-rate mismatch")

        groups: OrderedDict[tuple[int, int], list[dict]] = OrderedDict()
        for row in piece_rows:
            groups.setdefault((int(row["crop_start_24k"]), int(row["crop_stop_24k"])), []).append(row)
        estimates: dict[int, dict[str, np.ndarray]] = {}
        diagnostics = []
        for (start, stop), group_rows in groups.items():
            mixture = metrics.to_16k(mixture24[start:stop], 24_000)
            context = source_context(metadata, start, len(mixture))
            positions = {int(item["event_index"]): index for index, item in enumerate(context)}
            banks, bank_diagnostics = infer_bank(
                mixture, context, separator, aso, device, args.query_microbatch
            )
            diagnostics.append({"crop_start_24k": start, "crop_stop_24k": stop, **bank_diagnostics})
            for row in group_rows:
                event = int(row["event_index"])
                position = positions[event]
                estimates[event] = {
                    "selected_aso": np.asarray(banks["selected_aso"][position], dtype=np.float32),
                    "raw_source_bank": np.asarray(banks["raw_source_bank"][position], dtype=np.float32),
                    "magnitude_partition": np.asarray(banks["magnitude_partition"][position], dtype=np.float32),
                }

        event_order = [int(event["event_index"]) for event in metadata["events"] if int(event["event_index"]) in rows_by_event]
        if set(event_order) != set(rows_by_event) or set(estimates) != set(rows_by_event):
            raise ValueError(f"incomplete event coverage for {piece_id}")
        output = args.output_root / piece_id
        output.mkdir()
        shutil.copy2(source / "mixture.wav", output / "mixture.wav")
        shutil.copy2(source / "occurrences.zip", output / "occurrences.zip")
        (output / "piece.json").write_text(json.dumps({
            **metadata,
            "events": [event for event in metadata["events"] if int(event["event_index"]) in rows_by_event],
        }, indent=2) + "\n")
        for method, filename in (
            ("selected_aso", "selected_aso_sprite.wav"),
            ("raw_source_bank", "symmetric_sprite.wav"),
            ("magnitude_partition", "magnitude_sprite.wav"),
        ):
            sf.write(output / filename, sprite([estimates[event][method] for event in event_order]), RATE, subtype="FLOAT")

        result_rows = []
        for event in event_order:
            row = rows_by_event[event]
            authority = authority_rows[row["request_id"]]
            result_rows.append({
                "event_index": event,
                "request_id": row["request_id"],
                "crop_start16": round(int(row["crop_start_24k"]) * RATE / 24_000),
                "crop_samples16": int(row["samples_16k"]),
                "metrics": authority["metrics"],
            })
        (output / "result.json").write_text(json.dumps({
            "schema": "scns_eval_v3_final_aso_demo_source_v1",
            "status": "passed",
            "piece_id": piece_id,
            "rows": result_rows,
            "bank_diagnostics": diagnostics,
            "provenance": {
                "manifest_sha256": args.manifest_sha256,
                "aggregate_result_sha256": args.aggregate_sha256,
                "separator_checkpoint_sha256": args.separator_sha256,
                "aso_checkpoint_sha256": args.aso_sha256,
                "separator": separator_info,
                "aso_contract": aso_info.get("contract"),
                "script_sha256": sha256(Path(__file__)),
            },
        }, indent=2) + "\n")
        print(json.dumps({"piece": piece_id, "events": len(event_order)}), flush=True)


if __name__ == "__main__":
    main()
