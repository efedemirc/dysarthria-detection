"""Rigour additions computed over the WHOLE corpus (no subsampled headline).

For the detection task on the temporal feature subset (the second-iteration
model), this computes, all under the speaker-independent protocol:

  1. the full-corpus macro-F1 reference (reproduces 0.699 over all 8,216);
  2. a speaker-cluster bootstrap 95% CI on that macro-F1 — resampling the 15
     speakers with replacement over the full set of held-out predictions, so the
     interval honestly reflects the small speaker cohort;
  3. trivial baselines (majority-class, stratified-random) over all recordings,
     to contextualise the result against chance;
  4. a learning curve: macro-F1 as a function of the number of TRAINING speakers
     (a diagnostic of whether more speakers would help — the headline stays the
     full-corpus value at the right-hand end).

Usage:
    python rigour_analysis.py --csv results/features_torgo.csv --figures-dir figures
"""
from __future__ import annotations
import argparse, json
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.dummy import DummyClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import f1_score
from train_baselines import assign_domains, META

RNG = np.random.RandomState(42)


def svm():
    return Pipeline([("i", SimpleImputer(strategy="median")), ("s", StandardScaler()),
                     ("c", SVC(kernel="rbf", C=10, gamma="scale", class_weight="balanced"))])


def loso_predictions(X, y, spk):
    yt, yp, sg = [], [], []
    for tr, te in LeaveOneGroupOut().split(X, y, spk):
        m = svm(); m.fit(X[tr], y[tr])
        yt.extend(y[te]); yp.extend(m.predict(X[te])); sg.extend(spk[te])
    return np.array(yt), np.array(yp), np.array(sg)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--figures-dir", default="figures")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    feats = [c for c in df.columns if c not in META]
    temporal = assign_domains(feats)[0]["temporal"]
    X = df[temporal].apply(pd.to_numeric, errors="coerce").values
    y = df["label_detection"].values
    spk = df["speaker_id"].values
    speakers = np.array(sorted(set(spk)))
    out = {}

    # 1. full-corpus reference
    yt, yp, sg = loso_predictions(X, y, spk)
    full = f1_score(yt, yp, average="macro")
    out["full_corpus_macro_f1"] = round(float(full), 4)
    out["n_recordings"] = int(len(yt))
    print(f"[1] full-corpus temporal-SVM macro-F1 = {full:.3f}  (n={len(yt)}, all recordings)")

    # 2. speaker-cluster bootstrap CI (resample speakers, pool their predictions)
    by_spk = {s: (yt[sg == s], yp[sg == s]) for s in speakers}
    boot = []
    for _ in range(2000):
        pick = RNG.choice(speakers, size=len(speakers), replace=True)
        tt = np.concatenate([by_spk[s][0] for s in pick])
        pp = np.concatenate([by_spk[s][1] for s in pick])
        boot.append(f1_score(tt, pp, average="macro"))
    lo, hi = np.percentile(boot, [2.5, 97.5])
    out["bootstrap_ci95"] = [round(float(lo), 4), round(float(hi), 4)]
    print(f"[2] speaker-cluster bootstrap 95% CI = [{lo:.3f}, {hi:.3f}]  (2000 resamples over 15 speakers)")

    # 3. trivial baselines over all recordings (LOSO)
    for strat in ["most_frequent", "stratified", "uniform"]:
        tt, pp = [], []
        for tr, te in LeaveOneGroupOut().split(X, y, spk):
            d = DummyClassifier(strategy=strat, random_state=0); d.fit(X[tr], y[tr])
            tt.extend(y[te]); pp.extend(d.predict(X[te]))
        f = f1_score(tt, pp, average="macro")
        out[f"baseline_{strat}"] = round(float(f), 4)
        print(f"[3] baseline {strat:13s} macro-F1 = {f:.3f}")

    # 4. learning curve: macro-F1 vs number of TRAINING speakers
    print("[4] learning curve (macro-F1 vs #training speakers):")
    ks = list(range(2, len(speakers)))            # 2 .. 14 training speakers
    curve = {}
    for k in ks:
        scores = []
        for rep in range(8):                       # several random speaker subsets
            train_spk = RNG.choice(speakers, size=k, replace=False)
            test_spk = np.array([s for s in speakers if s not in set(train_spk)])
            tr = np.isin(spk, train_spk); te = np.isin(spk, test_spk)
            if len(set(y[tr])) < 2:                # need both classes to train
                continue
            m = svm(); m.fit(X[tr], y[tr])
            scores.append(f1_score(y[te], m.predict(X[te]), average="macro"))
        curve[k] = (float(np.mean(scores)), float(np.std(scores)))
        print(f"    k={k:2d} train speakers -> macro-F1 {np.mean(scores):.3f} ± {np.std(scores):.3f}")
    out["learning_curve"] = {str(k): [round(m, 4), round(s, 4)] for k, (m, s) in curve.items()}

    # plot
    ks_ = list(curve); ms = [curve[k][0] for k in ks_]; ss = [curve[k][1] for k in ks_]
    fig, ax = plt.subplots(figsize=(5, 3.6))
    ax.errorbar(ks_, ms, yerr=ss, marker="o", capsize=3)
    ax.axhline(full, ls="--", color="gray", label=f"full-corpus LOSO ({full:.3f})")
    ax.set_xlabel("number of training speakers"); ax.set_ylabel("detection macro-F1")
    ax.set_title("Learning curve (temporal features)"); ax.legend(fontsize=8); ax.grid(alpha=.3)
    fig.tight_layout()
    from pathlib import Path
    p = Path(args.figures_dir) / "learning_curve_detection.png"
    p.parent.mkdir(parents=True, exist_ok=True); fig.savefig(p, dpi=150); plt.close(fig)
    print(f"    wrote {p}")

    print("\nJSON:", json.dumps(out))
    from pathlib import Path as P
    (P(args.figures_dir).parent / "rigour_analysis.json").write_text(json.dumps(out, indent=2)) \
        if (P(args.figures_dir).parent).exists() else None
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
