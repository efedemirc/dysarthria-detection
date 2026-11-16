"""Generate results figures from the TORGO feature matrix.

Produces, for the detection task (and severity when >=2 classes/speakers exist):
  * confusion matrix  (speaker-independent CV aggregated)
  * Random Forest feature-importance bar chart (top 20)
  * SHAP summary (beeswarm) on a full-data RF fit

Figures are saved as PNG to --out-dir. Captions in the thesis should state the
speaker count and that preliminary figures use the in-sandbox subset.

Usage
-----
    python make_figures.py --csv features_torgo.csv --out-dir figures [--task detection]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import confusion_matrix

META = ["source_id", "speaker_id", "group", "label_detection", "label_severity"]


def load_xy(csv, task):
    df = pd.read_csv(csv)
    feat_cols = [c for c in df.columns if c not in META]
    if task == "detection":
        sub = df
        y = sub["label_detection"].values
    else:
        sub = df[df["group"] == "dysarthric"].copy()
        y = sub["label_severity"].values
    X = sub[feat_cols].apply(pd.to_numeric, errors="coerce").values
    groups = sub["speaker_id"].values
    return X, y, groups, feat_cols


def plot_confusion(X, y, groups, out_png, title):
    logo = LeaveOneGroupOut()
    yt, yp = [], []
    model = Pipeline([("impute", SimpleImputer(strategy="median")),
                      ("clf", RandomForestClassifier(n_estimators=400,
                       class_weight="balanced", random_state=42, n_jobs=-1))])
    for tr, te in logo.split(X, y, groups):
        if len(set(y[tr])) < 2:
            continue
        model.fit(X[tr], y[tr]); yt.extend(y[te]); yp.extend(model.predict(X[te]))
    if not yt:
        print("  confusion: no usable folds"); return False
    labels = sorted(set(yt) | set(yp))
    cm = confusion_matrix(yt, yp, labels=labels)
    fig, ax = plt.subplots(figsize=(4.5, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(labels))); ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right"); ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.set_title(title)
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.colorbar(im, fraction=0.046, pad=0.04); fig.tight_layout()
    fig.savefig(out_png, dpi=150); plt.close(fig)
    print(f"  saved {out_png}"); return True


def plot_importance(X, y, feat_cols, out_png, title):
    model = Pipeline([("impute", SimpleImputer(strategy="median")),
                      ("clf", RandomForestClassifier(n_estimators=400,
                       class_weight="balanced", random_state=42, n_jobs=-1))])
    model.fit(X, y)
    imp = model.named_steps["clf"].feature_importances_
    order = np.argsort(imp)[::-1][:20]
    names = [feat_cols[i].replace("rhythmic_features.", "").replace("vocal_features.", "")
             .replace("audio_quality_features.", "").replace("pitch_contour_features.", "")
             .replace("waveform_features.", "") for i in order]
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.barh(range(len(order))[::-1], imp[order], color="#3b6ea5")
    ax.set_yticks(range(len(order))[::-1]); ax.set_yticklabels(names, fontsize=8)
    ax.set_xlabel("Random Forest importance"); ax.set_title(title)
    fig.tight_layout(); fig.savefig(out_png, dpi=150); plt.close(fig)
    print(f"  saved {out_png}")


def plot_shap(X, y, feat_cols, out_png, title):
    try:
        import shap
        Xi = SimpleImputer(strategy="median").fit_transform(X)
        rf = RandomForestClassifier(n_estimators=300, class_weight="balanced",
                                    random_state=42, n_jobs=-1).fit(Xi, y)
        names = [c.split(".")[-1] for c in feat_cols]
        expl = shap.TreeExplainer(rf)
        sv = expl.shap_values(Xi)
        # binary RF -> list of 2 arrays or (n, f, 2); pick positive class
        if isinstance(sv, list):
            sv_plot = sv[1]
        elif getattr(sv, "ndim", 2) == 3:
            sv_plot = sv[:, :, 1]
        else:
            sv_plot = sv
        plt.figure()
        shap.summary_plot(sv_plot, Xi, feature_names=names, show=False, max_display=15)
        plt.title(title); plt.tight_layout()
        plt.savefig(out_png, dpi=150, bbox_inches="tight"); plt.close()
        print(f"  saved {out_png}"); return True
    except Exception as exc:
        print(f"  shap failed: {type(exc).__name__}: {exc}"); return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--task", choices=["detection", "severity"], default="detection")
    args = ap.parse_args()
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    X, y, groups, feat_cols = load_xy(args.csv, args.task)
    n_spk = len(set(groups))
    tag = f"({args.task}, {n_spk} speakers)"
    print(f"=== {args.task}: {len(y)} samples, {n_spk} speakers ===")
    plot_confusion(X, y, groups, out / f"cm_{args.task}.png", f"Confusion matrix {tag}")
    plot_importance(X, y, feat_cols, out / f"importance_{args.task}.png",
                    f"Top features {tag}")
    plot_shap(X, y, feat_cols, out / f"shap_{args.task}.png", f"SHAP summary {tag}")


if __name__ == "__main__":
    main()
