#!/usr/bin/env python3
"""Create short ASO clips that can play immediately from score-note clicks."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.io import wavfile


ROOT = Path(__file__).resolve().parent
MANIFEST = ROOT / "assets" / "manifest.json"
LEAD_SECONDS = 0.025
RELEASE_SECONDS = 0.250


def main() -> None:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    count = 0
    for example in manifest["examples"]:
        duration = float(example["duration"])
        for note in example["notes"]:
            aso = note["outputs"]["aso"]
            source = ROOT / aso["audio"]
            rate, signal = wavfile.read(source)
            start = max(0.0, float(note["start"]) - LEAD_SECONDS)
            end = min(duration, float(note["end"]) + RELEASE_SECONDS)
            left = max(0, round(start * rate))
            right = min(len(signal), round(end * rate))
            clip = np.asarray(signal[left:right])
            output = source.with_name("aso_audition.wav")
            wavfile.write(output, rate, clip)
            aso["audition"] = output.relative_to(ROOT).as_posix()
            aso["auditionStart"] = round(start, 4)
            count += 1

    MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {count} score-click audition clips")


if __name__ == "__main__":
    main()
