"""Honest learning curve: test set is ALWAYS the whole corpus.

For each training size k, run the standard 15-fold leave-one-speaker-out loop —
so every one of the 8,216 recordings is tested exactly once — but within each
fold train on a random k-subset of the 14 available training speakers rather
than all of them. Pool predictions across folds and score over the whole corpus.
Only the training size varies; the evaluation is always full-corpus. k=14
reproduces the full LOSO result (~0.699).
"""
from __future__ import annotations
import argparse, json
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import f1_score
from train_baselines import assign_domains, META


def svm():
    return Pipeline([("i", SimpleImputer(strategy="median")), ("s", StandardScaler()),
                     ("c", SVC(kernel="rbf", C=10, gamma="scale", class_weight="balanced"))])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--figures-dir", default="figures")
    ap.add_argument("--out-dir", default="results")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    feats = [c for c in df.columns if c not in META]
    temporal = assign_domains(feats)[0]["temporal"]
    X = df[temporal].apply(pd.to_numeric, errors="coerce").values
    y = df["label_detection"].values
    spk = df["speaker_id"].values

    folds = list(LeaveOneGroupOut().split(X, y, spk))  # 15 folds, test = each speaker once
    spk_class = {s: y[spk == s][0] for s in set(spk)}  # each speaker is one class
    # k must be >= 2 so a training subset can include both classes.
    ks = [2, 3, 4, 6, 8, 10, 12, 14]
    curve = {}
    print("Honest learning curve (test = all 8,216 recordings each point):")
    for k in ks:
        reps = 1 if k == 14 else 4
        rep_scores = []
        for rep in range(reps):
            rng = np.random.RandomState(1000 * k + rep)
            yt, yp = [], []
            for tr, te in folds:
                train_speakers = sorted(set(spk[tr]))              # the 14 available
                if k >= len(train_speakers):
                    pick = list(train_speakers)
                else:
                    classes = sorted(set(spk_class.values()))
                    # guarantee >=1 speaker per class, then fill the rest at random
                    pick = [rng.choice([s for s in train_speakers if spk_class[s] == c])
                            for c in classes]
                    rest = [s for s in train_speakers if s not in pick]
                    if k > len(pick):
                        pick += list(rng.choice(rest, size=k - len(pick), replace=False))
                mask = np.isin(spk, pick) & np.isin(np.arange(len(spk)), tr)
                m = svm(); m.fit(X[mask], y[mask])
                yt.extend(y[te]); yp.extend(m.predict(X[te]))
            rep_scores.append(f1_score(yt, yp, average="macro"))
        curve[k] = (float(np.mean(rep_scores)), float(np.std(rep_scores)))
        print(f"  k={k:2d} train speakers -> macro-F1 {np.mean(rep_scores):.3f} "
              f"± {np.std(rep_scores):.3f}  (test n=8216)")

    ks_ = list(curve); ms = [curve[k][0] for k in ks_]; ss = [curve[k][1] for k in ks_]
    fig, ax = plt.subplots(figsize=(5, 3.6))
    ax.errorbar(ks_, ms, yerr=ss, marker="o", capsize=3)
    ax.set_xlabel("number of training speakers")
    ax.set_ylabel("detection macro-F1 (full corpus)")
    ax.set_title("Learning curve — temporal features")
    ax.grid(alpha=.3); fig.tight_layout()
    p = Path(args.figures_dir) / "learning_curve_detection.png"
    p.parent.mkdir(parents=True, exist_ok=True); fig.savefig(p, dpi=150); plt.close(fig)
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    (Path(args.out_dir) / "learning_curve.json").write_text(
        json.dumps({str(k): [round(m, 4), round(s, 4)] for k, (m, s) in curve.items()}, indent=2))
    print(f"wrote {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
