"""Speaker-level detection evaluation: feature selection + two screening designs.

Reads a features CSV (v1 or v3) and, for the DETECTION task, reports results
under the identical speaker-independent (leave-one-speaker-out, LOSO) protocol
as the utterance-level baselines, but delivering ONE verdict per speaker — the
way a screening tool is actually used. Two aggregation designs are reported:

  A. Majority vote — train on utterances, predict every held-out utterance, then
     take the majority label per speaker. (Original iteration-2 design.)
  B. Feature aggregation — collapse each speaker's utterances to a single vector
     (mean + std of every feature), then train/predict at the speaker level with
     LOSO over the 15 speakers. This is the clinically correct unit of decision.

Both are combined with feature selection (temporal subset vs all features). All
scaling/imputation/fitting happens INSIDE the training folds — no leakage.

HONESTY NOTE printed at run time: at 15 speakers the speaker-level metric is
high variance (a one-speaker change moves macro-F1 by ~0.07), so results are
reported across THREE models rather than cherry-picked, and the recommended
headline is the model whose ERRORS are clinically coherent (borderline cases),
not merely the highest score.

No audio or engine needed; operates on the existing CSV.

Usage
-----
    python speaker_level_eval.py --csv results/features_torgo.csv \
        --out-dir results --figures-dir figures
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix

from train_baselines import assign_domains, META


def pipe(clf):
    return Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("clf", clf),
    ])


def utterance_models():
    return {
        "svm_rbf": pipe(SVC(kernel="rbf", C=10, gamma="scale", class_weight="balanced")),
        "random_forest": pipe(RandomForestClassifier(n_estimators=400, class_weight="balanced",
                                                     random_state=42, n_jobs=-1)),
    }


def aggregation_models():
    # Moderate regularisation: at n=15 with mean+std features, strong models overfit.
    return {
        "random_forest": pipe(RandomForestClassifier(n_estimators=300, class_weight="balanced",
                                                     random_state=42, n_jobs=-1)),
        "logreg": pipe(LogisticRegression(C=1.0, max_iter=2000, class_weight="balanced")),
        "svm_rbf": pipe(SVC(kernel="rbf", C=1, gamma="scale", class_weight="balanced")),
    }


def _speaker_majority(y_true, y_pred, spk):
    st, sp = [], []
    for s in sorted(set(spk)):
        m = spk == s
        st.append(y_true[m][0])
        vals, cnts = np.unique(y_pred[m], return_counts=True)
        sp.append(vals[cnts.argmax()])
    return np.array(st), np.array(sp)


def _speaker_report(ids, st, sp):
    labels = sorted(set(st) | set(sp))
    return {
        "macro_f1": float(f1_score(st, sp, average="macro", zero_division=0)),
        "accuracy": float(accuracy_score(st, sp)),
        "n_correct": int((st == sp).sum()), "n_speakers": int(len(st)),
        "misclassified": sorted(str(s) for s, t, p in zip(ids, st, sp) if t != p),
        "labels": labels,
        "confusion_matrix": confusion_matrix(st, sp, labels=labels).tolist(),
    }


def eval_majority_vote(df, cols, model):
    X = df[cols].apply(pd.to_numeric, errors="coerce").values
    y = df["label_detection"].values
    spk = df["speaker_id"].values
    yt, yp, sg = [], [], []
    for tr, te in LeaveOneGroupOut().split(X, y, spk):
        model.fit(X[tr], y[tr])
        yt.extend(y[te]); yp.extend(model.predict(X[te])); sg.extend(spk[te])
    yt, yp, sg = np.array(yt), np.array(yp), np.array(sg)
    st, sp = _speaker_majority(yt, yp, sg)
    ids = sorted(set(sg))
    rep = _speaker_report(ids, st, sp)
    rep_utt = {"macro_f1": float(f1_score(yt, yp, average="macro", zero_division=0)),
               "accuracy": float(accuracy_score(yt, yp))}
    return {"utterance": rep_utt, "speaker": rep}


def eval_feature_aggregation(df, cols, model):
    """One vector per speaker (mean+std of each feature); LOSO over speakers."""
    spk = df["speaker_id"].values
    y = df["label_detection"].values
    ids, rows, labs = [], [], []
    for s in sorted(set(spk)):
        m = spk == s
        sub = df.loc[m, cols].apply(pd.to_numeric, errors="coerce")
        rows.append(np.concatenate([sub.mean().values, sub.std().values]))
        labs.append(y[m][0]); ids.append(s)
    X, Y = np.array(rows), np.array(labs)
    preds = np.empty_like(Y)
    for tr, te in LeaveOneGroupOut().split(X, Y, ids):
        model.fit(X[tr], Y[tr]); preds[te] = model.predict(X[te])
    return _speaker_report(ids, Y, preds)


def confusion_fig(rep, title, path):
    labels = rep["labels"]; cm = np.array(rep["confusion_matrix"])
    fig, ax = plt.subplots(figsize=(4.2, 3.8))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(labels))); ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right"); ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.set_title(title)
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.tight_layout(); path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150); plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--figures-dir", default="figures")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    allf = [c for c in df.columns if c not in META]
    domains, _ = assign_domains(allf)
    feature_sets = {"all": allf, "temporal": domains["temporal"]}

    out = {"task": "detection", "protocol": "leave-one-speaker-out",
           "feature_set_sizes": {k: len(v) for k, v in feature_sets.items()},
           "majority_vote": {"results": {}},
           "speaker_aggregation": {"method": "mean+std per speaker, LOSO over speakers",
                                   "results": {}}}

    print("== A. MAJORITY VOTE (utterance model -> per-speaker majority) ==")
    for fs, cols in feature_sets.items():
        out["majority_vote"]["results"][fs] = {}
        for mname, model in utterance_models().items():
            r = eval_majority_vote(df, cols, model)
            out["majority_vote"]["results"][fs][mname] = r
            s = r["speaker"]
            print(f"  {fs:9s} {mname:14s} utt F1={r['utterance']['macro_f1']:.3f} | "
                  f"SPEAKER F1={s['macro_f1']:.3f} ({s['n_correct']}/{s['n_speakers']}) "
                  f"miss={s['misclassified']}")

    print("\n== B. FEATURE AGGREGATION (mean+std per speaker; LOSO over 15) ==")
    print("   [n=15 -> high variance; reported across models, not cherry-picked]")
    for fs, cols in feature_sets.items():
        out["speaker_aggregation"]["results"][fs] = {}
        for mname, model in aggregation_models().items():
            r = eval_feature_aggregation(df, cols, model)
            out["speaker_aggregation"]["results"][fs][mname] = r
            print(f"  {fs:9s} {mname:14s} SPEAKER F1={r['macro_f1']:.3f} "
                  f"({r['n_correct']}/{r['n_speakers']}) miss={r['misclassified']}")

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    (Path(args.out_dir) / "metrics_speaker_level.json").write_text(json.dumps(out, indent=2))

    # Recommended headline: temporal + RF aggregation (coherent, borderline-only errors).
    rec = out["speaker_aggregation"]["results"]["temporal"]["random_forest"]
    confusion_fig(rec, "Speaker-level detection (temporal, RF aggregation)",
                  Path(args.figures_dir) / "cm_detection_speaker_agg.png")
    # Keep the majority-vote figure too, for comparison.
    mv = out["majority_vote"]["results"]["temporal"]["random_forest"]["speaker"]
    confusion_fig(mv, "Speaker-level detection (temporal, majority vote)",
                  Path(args.figures_dir) / "cm_detection_speaker_level.png")
    print(f"\nwrote {Path(args.out_dir)/'metrics_speaker_level.json'} and 2 confusion figures")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
