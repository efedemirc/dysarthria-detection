"""Resumable TORGO feature-extraction driver for the CLVAD pipeline.

Runs the acoustic analytic set (including the added speech_mfcc analytic) over
extracted TORGO head-mic recordings and writes ONE JSON record per utterance to
``--records-dir``. It is resumable: utterances whose record already exists are
skipped, so the driver can be invoked repeatedly (useful under short execution
windows) until the corpus is fully processed.

A separate step (assemble_csv.py) flattens the per-record JSON into
``features_torgo.csv`` with label + speaker columns for the modelling harness.

Usage
-----
    python build_features.py --engine-root <repo> --audio-root <torgo_raw> \
        --records-dir <out> [--limit N] [--per-speaker-cap K]

Labels
------
speaker_id = first path component (e.g. M01, MC01).
group      = 'dysarthric' if speaker starts with M/F but NOT MC/FC, else 'control'.
The binary detection label is derived from `group`. Severity labels for
dysarthric speakers are joined later from a documented severity map (kept out of
this script so the heuristic provenance stays explicit and auditable).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List


ACOUSTIC_ANALYTIC_NAMES = [
    "PITCH_RANGE", "PITCH_STABILITY", "PITCH_CONTOUR", "PAUSE_ANALYSIS",
    "SPEECH_PROSODY", "SPEECH_VAD_CONFIDENCE", "SPEECH_SPECTRAL", "SPEECH_MFCC",
    "SPEECH_RHYTHM", "FORMANT_ANALYSIS", "VOICE_QUALITY", "AUDIO_QUALITY",
    "WAVEFORM_STRUCTURE",
    # Grabe-Low nPVI + intonation-reset rhythm metrics (updated engine). Requires
    # speech_delivery_guard.patch so single-segment clips skip instead of crashing;
    # nPVI is populated only on multi-segment (phrase/sentence) utterances.
    "SPEECH_DELIVERY_DYNAMICS",
]

FEATURE_SECTIONS = [
    "vocal_features", "pitch_contour_features", "rhythmic_features",
    "waveform_features", "audio_quality_features",
]


import re

# TORGO speaker codes: F01, M03, MC02, FC01, ... (control prefixes MC/FC).
_SPEAKER_RE = re.compile(r"^(?:MC|FC|M|F)\d+$", re.IGNORECASE)


def speaker_of(audio_path: Path, audio_root: Path) -> str:
    """Return the TORGO speaker code, tolerating archive 'wrapper' folders.

    Handles both a flat layout (``M01/.../wav_headMic/x.wav``) and the 7-Zip
    layout where each archive extracts into a wrapper (``M/M01/.../x.wav``):
    it scans the path for the first component that looks like a speaker code,
    falling back to the first component if none matches.
    """
    parts = audio_path.relative_to(audio_root).parts
    for part in parts:
        if _SPEAKER_RE.match(part):
            return part
    return parts[0]


def group_of(speaker: str) -> str:
    return "control" if speaker.upper().startswith(("MC", "FC")) else "dysarthric"


def flatten(d: dict, prefix: str = "") -> Dict[str, float]:
    out: Dict[str, float] = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(flatten(v, key + "."))
        elif isinstance(v, (int, float)) and not isinstance(v, bool):
            out[key] = float(v)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine-root", required=True)
    ap.add_argument("--audio-root", required=True)
    ap.add_argument("--records-dir", required=True)
    ap.add_argument("--limit", type=int, default=None,
                    help="Max NEW utterances to process this invocation.")
    ap.add_argument("--per-speaker-cap", type=int, default=None,
                    help="Process at most this many utterances per speaker (sampling).")
    ap.add_argument("--time-budget", type=float, default=38.0,
                    help="Stop starting new files after this many seconds (fits 45s windows).")
    args = ap.parse_args()

    engine_root = Path(args.engine_root).resolve()
    audio_root = Path(args.audio_root).resolve()
    records_dir = Path(args.records_dir).resolve()
    records_dir.mkdir(parents=True, exist_ok=True)

    sys.path.insert(0, str(engine_root))
    from src.schemas.clvad_enums import AnalyticID
    from src.pipeline.run_pipeline import run_pipeline

    # Resolve names against the live enum; skip any not present (e.g. an older
    # engine without SPEECH_DELIVERY_DYNAMICS) so extraction still runs.
    analytics = []
    for n in ACOUSTIC_ANALYTIC_NAMES:
        if hasattr(AnalyticID, n):
            analytics.append(getattr(AnalyticID, n))
        else:
            print(f"note: analytic {n} not in this engine's registry — skipping it")

    wavs = sorted(audio_root.rglob("*/wav_headMic/*.wav"))
    # apply per-speaker cap deterministically
    if args.per_speaker_cap:
        by_spk: Dict[str, List[Path]] = {}
        for w in wavs:
            by_spk.setdefault(speaker_of(w, audio_root), []).append(w)
        capped: List[Path] = []
        for spk, lst in by_spk.items():
            capped.extend(lst[: args.per_speaker_cap])
        wavs = sorted(capped)

    started = time.monotonic()
    processed = skipped = failed = 0
    for w in wavs:
        spk = speaker_of(w, audio_root)
        # source_id encodes the relative path so it is unique and speaker-tagged
        sid = "__".join(w.relative_to(audio_root).with_suffix("").parts)
        out_path = records_dir / f"{sid}.json"
        if out_path.exists():
            skipped += 1
            continue
        if args.limit is not None and processed >= args.limit:
            break
        if (time.monotonic() - started) > args.time_budget:
            break
        try:
            rec = run_pipeline(
                {"audio_path": str(w), "source_id": sid, "audio_variant": "speech"},
                active_analytics=analytics, content_mode="speech",
            )
            scalars: Dict[str, float] = {}
            for sec in FEATURE_SECTIONS:
                if isinstance(rec.get(sec), dict):
                    scalars.update(flatten(rec[sec], sec + "."))
            row = {
                "source_id": sid,
                "speaker_id": spk,
                "group": group_of(spk),
                "features": scalars,
            }
            out_path.write_text(json.dumps(row))
            processed += 1
        except Exception as exc:  # keep going; log the failure
            (records_dir / f"{sid}.error").write_text(f"{type(exc).__name__}: {exc}")
            failed += 1

    total_records = len(list(records_dir.glob("*.json")))
    print(f"processed={processed} skipped={skipped} failed={failed} "
          f"total_records={total_records} remaining={len(wavs)-total_records}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
