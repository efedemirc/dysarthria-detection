#!/usr/bin/env python3
"""One-command experiment runner for the dysarthria detection pipeline.

Chains the four stages over an extracted TORGO corpus:
    build_features -> assemble_csv -> train_baselines -> make_figures
writing the feature CSV, metrics JSON, and figures into repo-relative folders.

Prerequisites (see README.md):
  * CLVAD engine repo available locally with engine_extension/speech_mfcc.patch applied
  * TORGO extracted so paths look like <torgo-root>/M01/.../wav_headMic/*.wav
  * pip install -r requirements.txt  &&  python -m nltk.downloader cmudict

Example:
    python run_experiment.py --engine-root ../creative-analysis-engine \
        --torgo-root ../TORGO --shap
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
EI = ROOT / "pipeline"


def run(script: Path, *args: str) -> None:
    cmd = [sys.executable, str(script), *args]
    print("\n› " + " ".join(cmd))
    subprocess.run(cmd, check=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--engine-root", required=True, help="Path to the CLVAD engine repo (patch applied).")
    ap.add_argument("--torgo-root", required=True, help="Path to extracted TORGO speaker folders.")
    ap.add_argument("--records-dir", default=str(ROOT / "torgo_records"))
    ap.add_argument("--csv", default=str(ROOT / "results" / "features_torgo.csv"))
    ap.add_argument("--out-dir", default=str(ROOT / "results"))
    ap.add_argument("--figures-dir", default=str(ROOT / "figures"))
    ap.add_argument("--per-speaker-cap", type=int, default=None,
                    help="Sample at most N utterances per speaker (omit to use all).")
    ap.add_argument("--shap", action="store_true", help="Also compute SHAP attributions.")
    a = ap.parse_args()

    Path(a.out_dir).mkdir(parents=True, exist_ok=True)
    Path(a.figures_dir).mkdir(parents=True, exist_ok=True)

    # 1) Feature extraction (large time budget so it completes in one pass).
    feat_args = ["--engine-root", a.engine_root, "--audio-root", a.torgo_root,
                 "--records-dir", a.records_dir, "--time-budget", "1000000"]
    if a.per_speaker_cap:
        feat_args += ["--per-speaker-cap", str(a.per_speaker_cap)]
    run(EI / "build_features.py", *feat_args)

    # 2) Assemble the feature matrix.
    run(EI / "assemble_csv.py", "--records-dir", a.records_dir, "--out", a.csv)

    # 3) Train baselines (detection + severity) + per-domain ablation.
    tb_args = ["--csv", a.csv, "--out-dir", a.out_dir, "--task", "both", "--ablation"]
    if a.shap:
        tb_args.append("--shap")
    run(EI / "train_baselines.py", *tb_args)

    # 4) Figures for each task.
    for task in ("detection", "severity"):
        run(EI / "make_figures.py", "--csv", a.csv, "--out-dir", a.figures_dir, "--task", task)

    print(f"\nDone.\n  CSV:     {a.csv}\n  metrics: {a.out_dir}\n  figures: {a.figures_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
