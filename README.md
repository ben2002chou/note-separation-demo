# Score-informed note-separation demo

This repository is a self-contained static site for GitHub Pages. It uses no
runtime dependencies or server-side code.

Live demo: <https://benschou.com/note-separation-demo/>

To preview it locally:

```sh
python3 -m http.server 8000
```

Then open <http://localhost:8000>.

The committed audio, score metadata, and spectrograms are precomputed.
To rebuild them after staging the frozen SCNS-Eval-v3 examples and their
retained method sprites, run:

```sh
python build_assets.py --source-root /path/to/staged-assets
```

The build script preserves one shared amplitude scale within each example. The
three curated examples cover piano, acoustic guitar, and cello. The headline
The proposed extraction uses the final temporal ASO checkpoint at step 29,058,
and the comparison controls expose the four matched 30k separator ablations
reported in Table 1. Browser audio
and ownership maps are loaded only when an example or note is selected.

Run `python build_mask_overlays.py` after rebuilding the audio assets. It
recovers each note's displayed ASO magnitude allocation from the aligned
mixture and ASO output and writes the transparent overlays used by the score.
Run `python build_note_auditions.py` to create short Table 1 method clips,
compact waveform envelopes, and the ASO ownership maps used for spectrogram
clicks.
