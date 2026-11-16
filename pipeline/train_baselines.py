"""Speaker-independent baseline experiments on the TORGO feature matrix.

Trains SVM (RBF) and Random Forest baselines for:
  * detection  (dysarthric vs control)
  * severity   (3-class, dysarthric speakers only)
under LEAVE-ONE-SPEAKER-GROUP-OUT cross-validation (no speaker in both train and
test). Reports per-class / macro precision-recall-F1, accuracy, and confusion
matrices, and Random Forest feature importances. Optionally computes SHAP.

Outputs JSON metrics + PNG confusion matrices + importance ranking to --out-dir.

Usage
-----
    python train_baselines.py --csv features_torgo.csv --out-dir results [--task detection|severity|both] [--shap]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import (classification_report, confusion_matrix,
                             f1_score, accuracy_score)

META = ["source_id", "speaker_id", "group", "label_detection", "label_severity"]

# Operationalisation of the four clinical feature domains used in the thesis
# ablation (Table 5.3). Predicates are applied to the full (prefixed) column name
# in PRIORITY ORDER; each feature is assigned to the first domain it matches, so
# the subsets are disjoint. Engine-generic fields that fit none of the four
# clinical domains (e.g. audio-quality, waveform-energy, readiness scores) are
# left unassigned and appear only in the "combined (all)" row.
FEATURE_DOMAINS = {
    "spectral": lambda c: ("mfcc_" in c or "zero_crossing_rate" in c
                           or "spectral_centroid" in c or "spectral_bandwidth" in c
                           or "spectral_flatness" in c or "spectral_rolloff" in c),
    "voice_quality": lambda c: c.endswith(("jitter", "shimmer", "hnr")),
    "prosodic": lambda c: ("pitch" in c or c.endswith(("f1_mean", "f2_mean", "f3_mean"))
                           or "intensity" in c or c.endswith("speaking_time_ratio")),
    "temporal": lambda c: ("pause" in c or "burst" in c or "hesitation" in c
                           or "speech_activation" in c or "speech_activity" in c
                           or "npvi" in c  # Grabe-Low durational rhythm variability
                           or c.endswith(("voiced_ratio", "unvoiced_ratio",
                                          "usable_speech_ratio", "silence_ratio"))),
}


def assign_domains(feat_cols):
    """Map each feature column to its first-matching clinical domain (disjoint)."""
    domains = {d: [] for d in FEATURE_DOMAINS}
    unassigned = []
    for c in feat_cols:
        for d, pred in FEATURE_DOMAINS.items():
            if pred(c):
                domains[d].append(c)
                break
        else:
            unassigned.append(c)
    return domains, unassigned


def load_xy(csv: str, task: str):
    df = pd.read_csv(csv)
    feat_cols = [c for c in df.columns if c not in META]
    if task == "detection":
        sub = df
        y = sub["label_detection"].values
    else:  # severity: dysarthric speakers only
        sub = df[df["group"] == "dysarthric"].copy()
        y = sub["label_severity"].values
    X = sub[feat_cols].apply(pd.to_numeric, errors="coerce").values
    groups = sub["speaker_id"].values
    return X, y, groups, feat_cols


def models():
    return {
        "svm_rbf": Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("clf", SVC(kernel="rbf", C=10, gamma="scale", class_weight="balanced")),
        ]),
        "random_forest": Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("clf", RandomForestClassifier(n_estimators=400, class_weight="balanced",
                                           random_state=42, n_jobs=-1)),
        ]),
    }


def run_task(X, y, groups, feat_cols, task, out_dir, do_shap=False):
    n_speakers = len(np.unique(groups))
    print(f"\n=== TASK: {task} | samples={len(y)} | speakers={n_speakers} | "
          f"classes={sorted(set(y))} ===")
    if n_speakers < 2:
        print("  SKIP: need >=2 speaker groups for speaker-independent CV.")
        return None

    logo = LeaveOneGroupOut()
    results = {}
    for name, model in models().items():
        y_true_all, y_pred_all = [], []
        usable_folds = 0
        for tr, te in logo.split(X, y, groups):
            if len(set(y[tr])) < 2:
                # degenerate fold (training set has a single class) — skip
                continue
            model.fit(X[tr], y[tr])
            y_true_all.extend(y[te])
            y_pred_all.extend(model.predict(X[te]))
            usable_folds += 1
        if usable_folds == 0 or not y_true_all:
            print(f"  {name}: no usable speaker-independent folds "
                  f"(too few speakers per class).")
            results[name] = {"usable_folds": 0}
            continue
        labels = sorted(set(y_true_all) | set(y_pred_all))
        rep = classification_report(y_true_all, y_pred_all, labels=labels,
                                    output_dict=True, zero_division=0)
        cm = confusion_matrix(y_true_all, y_pred_all, labels=labels).tolist()
        macro_f1 = f1_score(y_true_all, y_pred_all, labels=labels,
                            average="macro", zero_division=0)
        acc = accuracy_score(y_true_all, y_pred_all)
        results[name] = {
            "usable_folds": usable_folds, "accuracy": acc, "macro_f1": macro_f1,
            "labels": labels, "confusion_matrix": cm, "report": rep,
        }
        print(f"  {name}: folds={usable_folds} acc={acc:.3f} macro_F1={macro_f1:.3f}")

    # RF feature importance on a full-data fit (for interpretation)
    rf = models()["random_forest"]
    rf.fit(X, y)
    imp = rf.named_steps["clf"].feature_importances_
    ranking = sorted(zip(feat_cols, imp), key=lambda t: t[1], reverse=True)
    results["rf_feature_importance_top20"] = [
        {"feature": f, "importance": float(i)} for f, i in ranking[:20]
    ]

    if do_shap:
        try:
            import shap
            Xi = SimpleImputer(strategy="median").fit_transform(X)
            expl = shap.TreeExplainer(rf.named_steps["clf"])
            sv = expl.shap_values(Xi)
            results["shap_available"] = True
        except Exception as exc:
            results["shap_available"] = False
            results["shap_error"] = f"{type(exc).__name__}: {exc}"

    Path(out_dir).mkdir(parents=True, exist_ok=True)
    (Path(out_dir) / f"metrics_{task}.json").write_text(json.dumps(results, indent=2))
    print(f"  top features:", [f for f, _ in ranking[:6]])
    return results


def cv_macro_f1(X, y, groups):
    """Best speaker-independent macro-F1 across the two baselines for a given X."""
    best = {"best_macro_f1": None, "best_model": None, "per_model": {}}
    for name, model in models().items():
        yt, yp = [], []
        for tr, te in LeaveOneGroupOut().split(X, y, groups):
            if len(set(y[tr])) < 2:
                continue
            model.fit(X[tr], y[tr]); yt.extend(y[te]); yp.extend(model.predict(X[te]))
        if not yt:
            best["per_model"][name] = None
            continue
        labels = sorted(set(yt) | set(yp))
        mf1 = f1_score(yt, yp, labels=labels, average="macro", zero_division=0)
        best["per_model"][name] = mf1
        if best["best_macro_f1"] is None or mf1 > best["best_macro_f1"]:
            best["best_macro_f1"], best["best_model"] = mf1, name
    return best


def run_ablation(csv, out_dir):
    """Per-domain feature ablation (Table 5.3): macro-F1 per clinical domain
    subset vs the combined set, under speaker-independent CV, for both tasks."""
    df = pd.read_csv(csv)
    all_feats = [c for c in df.columns if c not in META]
    domains, unassigned = assign_domains(all_feats)
    print(f"\n=== ABLATION === domain sizes: "
          f"{ {d: len(cols) for d, cols in domains.items()} } | "
          f"unassigned (combined-only): {len(unassigned)}")
    subsets = {**domains, "combined": all_feats}
    out = {"domain_membership": domains, "unassigned": unassigned, "tasks": {}}
    for task in ("detection", "severity"):
        _, y, groups, _ = load_xy(csv, task)
        sub = df if task == "detection" else df[df["group"] == "dysarthric"].copy()
        out["tasks"][task] = {}
        for sname, cols in subsets.items():
            if not cols:
                out["tasks"][task][sname] = None
                continue
            X = sub[cols].apply(pd.to_numeric, errors="coerce").values
            res = cv_macro_f1(X, y, groups)
            out["tasks"][task][sname] = res
            mf1 = res["best_macro_f1"]
            print(f"  {task:9s} {sname:14s} macro_F1="
                  f"{'n/a' if mf1 is None else format(mf1, '.3f')} ({res['best_model']})")
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    (Path(out_dir) / "metrics_ablation.json").write_text(json.dumps(out, indent=2))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--task", choices=["detection", "severity", "both"], default="both")
    ap.add_argument("--shap", action="store_true")
    ap.add_argument("--ablation", action="store_true",
                    help="Also compute the per-domain feature ablation (Table 5.3).")
    args = ap.parse_args()

    tasks = ["detection", "severity"] if args.task == "both" else [args.task]
    for task in tasks:
        X, y, groups, feat_cols = load_xy(args.csv, task)
        run_task(X, y, groups, feat_cols, task, args.out_dir, do_shap=args.shap)
    if args.ablation:
        run_ablation(args.csv, args.out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
