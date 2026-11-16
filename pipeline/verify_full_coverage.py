"""Prove the per-utterance detection result is tested on the WHOLE dataset.

Under leave-one-speaker-out (LOSO), every recording sits in exactly one held-out
test fold, so across the 15 folds every recording is predicted exactly once with
a model that never saw its speaker. This script makes that auditable:

  * asserts each row is tested exactly once (total test predictions == n rows);
  * prints the per-speaker test-fold sizes and confirms they sum to the corpus;
  * reports per-utterance macro-F1 for SVM / RF / LogReg on all-features and the
    temporal subset, so the headline number is reproduced from the raw CSV.

Usage
-----
    python verify_full_coverage.py --csv results/features_torgo.csv
"""
from __future__ import annotations

import argparse
import numpy as np
import pandas as pd
from collections import Counter
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import f1_score, accuracy_score

from train_baselines import assign_domains, META


def pipe(clf):
    return Pipeline([("impute", SimpleImputer(strategy="median")),
                     ("scale", StandardScaler()), ("clf", clf)])


def models():
    return {
        "svm_rbf": pipe(SVC(kernel="rbf", C=10, gamma="scale", class_weight="balanced")),
        "random_forest": pipe(RandomForestClassifier(n_estimators=400, class_weight="balanced",
                                                     random_state=42, n_jobs=-1)),
        "logreg": pipe(LogisticRegression(max_iter=2000, class_weight="balanced")),
    }


def evaluate(X, y, groups, model):
    # tested_index records how many times each ROW is used as a test example.
    tested = np.zeros(len(y), dtype=int)
    y_true, y_pred = [], []
    fold_sizes = {}
    for tr, te in LeaveOneGroupOut().split(X, y, groups):
        held_out = groups[te[0]]
        fold_sizes[held_out] = len(te)
        tested[te] += 1
        model.fit(X[tr], y[tr])
        y_true.extend(y[te]); y_pred.extend(model.predict(X[te]))
    return np.array(y_true), np.array(y_pred), tested, fold_sizes


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    n = len(df)
    feats = [c for c in df.columns if c not in META]
    domains, _ = assign_domains(feats)
    y = df["label_detection"].values
    groups = df["speaker_id"].values

    print(f"CSV: {args.csv}")
    print(f"recordings (rows) : {n}")
    print(f"speakers          : {len(set(groups))}  -> {sorted(set(groups))}")
    print(f"class balance     : {dict(Counter(y))}")
    print(f"LOSO folds        : {len(set(groups))} (one per speaker)\n")

    # Coverage proof uses the temporal-subset SVM; coverage is identical for any model.
    Xt = df[domains["temporal"]].apply(pd.to_numeric, errors="coerce").values
    yt, yp, tested, fold_sizes = evaluate(Xt, y, groups, models()["svm_rbf"])

    every_once = bool(np.all(tested == 1))
    print("=== WHOLE-DATASET COVERAGE CHECK ===")
    print(f"rows tested exactly once : {int((tested == 1).sum())} / {n}")
    print(f"rows tested 0 times      : {int((tested == 0).sum())}")
    print(f"rows tested >1 time      : {int((tested > 1).sum())}")
    print(f"sum of test-fold sizes   : {sum(fold_sizes.values())} (must equal {n})")
    print(f"EVERY recording tested exactly once: {every_once}")
    print("per-speaker held-out test sizes:")
    for s in sorted(fold_sizes): print(f"    {s:6s} {fold_sizes[s]}")

    print("\n=== PER-UTTERANCE macro-F1 (all 8k+ recordings, LOSO) ===")
    for fs, cols in {"all": feats, "temporal": domains["temporal"]}.items():
        X = df[cols].apply(pd.to_numeric, errors="coerce").values
        for mname, model in models().items():
            a, b, _, _ = evaluate(X, y, groups, model)
            print(f"  {fs:9s} {mname:14s} macro-F1={f1_score(a,b,average='macro'):.3f} "
                  f"acc={accuracy_score(a,b):.3f}  (n_tested={len(a)})")

    assert every_once and sum(fold_sizes.values()) == n, "COVERAGE FAILED"
    print("\nOK: the per-utterance result is computed over the ENTIRE corpus, "
          "each recording held out and tested exactly once.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
